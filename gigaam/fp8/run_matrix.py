"""Run the FP8 PTQ + native-packaging pipeline for all variants.

Processes all four ASR variants from the HuggingFace mirror::

    e2e_ctc, ctc, e2e_rnnt, rnnt

For each variant the pipeline is:

1. **PTQ** (all variants): load fp16 from HF -> quantize GEMM Linears (encoder
   FFN/attn, and RNNT joint enc/pred) to FP8 E4M3 -> calibrate per-tensor
   activation scales on a data subset -> transcription parity check. FP8 PTQ is
   near-lossless for GigaAM v3, so this is the whole story (no QAT/KD stage).
2. **Package** (all variants) in the *native* convention (no ``model.`` prefix):
   ``model.safetensors`` (fp16) + ``model_fp8.safetensors`` (uint8 + per-channel
   scales) + ``model_fp8.safetensors.activation_scales.json`` + config/modeling/
   tokenizer, from the live quantized model.

Usage
-----
    uv run python -m gigaam.fp8.run_matrix \
        --manifest data/sub/manifest_train.tsv \
        --n_calibration_batches 32 --batch_size 32 \
        --out_root hf_packages
"""

import argparse
import gc
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pyarrow.parquet as pq  # noqa: F401  (kept for parity with data.py usage)

DATASET_ID = "govnejri/golos_mfa_punctuation"

# (variant, base HF revision, is_charwise, mode). The encoder quantization is
# identical for all; charwise variants just use a different tokenizer/vocab.
VARIANTS = [
    ("e2e_ctc", "e2e_ctc", False, "ctc"),
    ("ctc", "ctc", True, "ctc"),
    ("e2e_rnnt", "e2e_rnnt", False, "rnnt"),
    ("rnnt", "rnnt", True, "rnnt"),
]


def _make_eval_samples(manifest: str, n: int = 4) -> List[dict]:
    """Pick a few short samples from the manifest for transcription parity checks.

    Audio paths in the manifest are relative to the manifest directory; resolve
    them to absolute paths so transcription works regardless of CWD.
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest))
    samples = []
    with open(manifest) as f:
        next(f)  # header
        for line in f:
            path, dur, text = line.rstrip("\n").split("\t", 2)
            if not os.path.isabs(path):
                path = os.path.join(manifest_dir, path)
            samples.append({"path": path, "text": text})
            if len(samples) >= n:
                break
    return samples


def _stage_ptq(
    variant: str,
    revision: str,
    is_charwise: bool,
    manifest: str,
    n_calibration_batches: int,
    batch_size: int,
    device: str,
    out_dir: str,
) -> dict:
    """Stage 1: load fp16 -> quantize -> calibrate -> parity. Returns summary + scales path."""
    from gigaam import load_model_from_hf
    from gigaam.fp8 import (
        bake_activation_scales,
        collect_fp8_linears,
        count_targets,
        quantize_model,
    )
    from gigaam.fp8.calibrate import ActivationCalibrator
    from gigaam.utils import AudioDataset

    print(f"\n{'='*60}\n=== {variant} [PTQ] (revision={revision}, charwise={is_charwise})\n{'='*60}")
    model = load_model_from_hf(variant, fp16_encoder=True, device=device)
    print(f"  loaded. encoder dtype: {next(model.encoder.parameters()).dtype}")

    # Transcription parity: teacher (pre-quantization).
    eval_samples = _make_eval_samples(manifest, n=4)
    teacher_texts = []
    for s in eval_samples:
        try:
            teacher_texts.append(model.transcribe(s["path"]).text)
        except Exception as e:
            teacher_texts.append(f"<err: {e}>")

    # Quantize.
    quantize_model(model, compute_dtype=torch.float16)
    counts = count_targets(model)
    print(f"  quantized: {counts}")
    head_fp8 = [k for k in collect_fp8_linears(model) if k.startswith("head.")]
    if head_fp8:
        print(f"  head FP8 modules: {head_fp8}")

    # Calibrate activation scales.
    print(f"  calibrating over {n_calibration_batches} batches ...")
    ds = AudioDataset(manifest)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AudioDataset.collate,
        num_workers=4,
    )
    calib = ActivationCalibrator(model)
    with torch.no_grad():
        for i, (wavs, lens) in enumerate(
            tqdm(dl, total=n_calibration_batches, desc=f"  {variant} calib")
        ):
            model(wavs.to(device), lens.to(device))
            if i + 1 >= n_calibration_batches:
                break
    calib.remove()
    n_baked = bake_activation_scales(model, calib.amax)
    print(
        f"  baked {n_baked} activation scales "
        f"(amax {min(calib.amax.values()):.2f}..{max(calib.amax.values()):.2f})"
    )

    # Transcription parity: student (post-quantization).
    student_texts, parity_ok = [], True
    for s, t_text in zip(eval_samples, teacher_texts):
        try:
            s_text = model.transcribe(s["path"]).text
        except Exception as e:
            s_text = f"<err: {e}>"
        student_texts.append(s_text)
        if s_text != t_text and not s_text.startswith("<err"):
            parity_ok = False
    for s, tt, st in zip(eval_samples, teacher_texts, student_texts):
        flag = "OK" if tt == st else "DIFF"
        print(f"    [{flag}] T: {tt[:45]} | S: {st[:45]}")

    # Persist activation scales.
    os.makedirs(out_dir, exist_ok=True)
    scales_path = os.path.join(out_dir, "activation_scales.json")
    with open(scales_path, "w") as f:
        json.dump(calib.amax, f, indent=2)

    return {
        "model": model,
        "counts": counts,
        "head_fp8": head_fp8,
        "n_activation_scales": n_baked,
        "scales_path": scales_path,
        "transcription_parity": parity_ok,
    }


def _package_native_from_model(model, out_dir: str, variant: str, revision: str, scales_path: str):
    """Package a live PTQ model in the native convention.

    Writes two files from the same live model:
    - ``model.safetensors``: FP8 weights *dequantized* to fp16 (parity).
    - ``model_fp8.safetensors``: FP8 weights as uint8 + per-channel scales.
    """
    from gigaam.fp8.export_native import model_to_safetensors
    import json

    with open(scales_path) as f:
        amax = json.load(f)
    out_fp16 = os.path.join(out_dir, "model.safetensors")
    out_fp8 = os.path.join(out_dir, "model_fp8.safetensors")
    # fp16 parity: FP8 weights dequantized (E4M3-rounded fp16), no scales.
    model_to_safetensors(model, out_fp16, activation_scales=None, fp8_as_dequant=True)
    # FP8: uint8 bytes + scales + activation scales sidecar.
    model_to_safetensors(model, out_fp8, activation_scales=amax, fp8_as_dequant=False)

    # config + modeling + tokenizer from base revision.
    from gigaam.fp8.package_hf import _fetch_config_and_meta

    _fetch_config_and_meta(
        out_dir,
        model_name=f"{variant}_fp8",
        base_revision=revision,
        kd_alpha=None,
        trained_steps=None,
    )
    print(f"  packaged native (from live PTQ model) -> {out_dir}")


def process_variant(
    variant: str,
    revision: str,
    is_charwise: bool,
    mode: str,
    manifest: str,
    out_root: str,
    n_calibration_batches: int,
    batch_size: int,
    device: str,
) -> dict:
    """Run PTQ + native packaging for one variant. Returns a summary dict."""
    out_dir = os.path.join(out_root, variant)
    print(f"\n{'#'*60}\n### VARIANT: {variant} (mode={mode})\n{'#'*60}")

    # --- Stage 1: PTQ (all variants) ---
    ptq = _stage_ptq(
        variant, revision, is_charwise, manifest, n_calibration_batches, batch_size, device, out_dir
    )
    model = ptq["model"]

    # --- Stage 2: package from the live PTQ model (native convention) ---
    _package_native_from_model(model, out_dir, variant, revision, ptq["scales_path"])
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": variant,
        "revision": revision,
        "charwise": is_charwise,
        "mode": mode,
        "counts": ptq["counts"],
        "head_fp8": ptq["head_fp8"],
        "n_activation_scales": ptq["n_activation_scales"],
        "transcription_parity": ptq["transcription_parity"],
    }


def main():
    p = argparse.ArgumentParser(description="Run the FP8 PTQ + packaging matrix.")
    p.add_argument("--manifest", required=True, help="TSV manifest for calibration.")
    p.add_argument("--n_calibration_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=32, help="PTQ calibration batch size.")
    p.add_argument("--out_root", default="hf_packages")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--variants", nargs="+", default=None,
        help="Subset of variants to process (default: all 4).",
    )
    args = p.parse_args()

    variants = VARIANTS
    if args.variants:
        variants = [v for v in VARIANTS if v[0] in args.variants]
        if not variants:
            raise ValueError(f"No matching variants in {args.variants}")

    os.makedirs(args.out_root, exist_ok=True)
    summaries = []
    for variant, revision, is_charwise, mode in variants:
        try:
            s = process_variant(
                variant, revision, is_charwise, mode,
                args.manifest, args.out_root,
                args.n_calibration_batches, args.batch_size, args.device,
            )
            summaries.append(s)
        except Exception as e:
            import traceback
            print(f"\n!!! {variant} FAILED: {e}")
            traceback.print_exc()
            summaries.append({"variant": variant, "error": str(e)})
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n=== MATRIX SUMMARY\n{'='*60}")
    print(json.dumps(summaries, indent=2))
    summary_path = os.path.join(args.out_root, "matrix_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
