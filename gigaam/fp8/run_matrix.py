"""Run the full FP8 pipeline (PTQ + optional KD + native packaging) for all variants.

Processes all four ASR variants from the HuggingFace mirror::

    e2e_ctc, ctc, e2e_rnnt, rnnt

For each variant the pipeline is:

1. **PTQ** (all variants): load fp16 from HF -> quantize GEMM Linears (encoder
   FFN/attn, and RNNT joint enc/pred) to FP8 E4M3 -> calibrate per-tensor
   activation scales on a data subset -> transcription parity check.
2. **KD** (RNNT variants only; CTC FP8 PTQ is lossless so KD adds no value):
   launch ``train_utils/train.py`` as a subprocess with per-variant defaults to
   run grid-level knowledge distillation against a frozen fp16 teacher. Produces
   a Lightning checkpoint with the KD-recovered weights.
3. **Package** (all variants) in the *native* convention (no ``model.`` prefix):
   ``model.safetensors`` (fp16) + ``model_fp8.safetensors`` (uint8 + per-channel
   scales) + ``model_fp8.safetensors.activation_scales.json`` + config/modeling/
   tokenizer. KD variants package from the best checkpoint; PTQ-only variants
   package from the live quantized model.

Usage
-----
    uv run python -m gigaam.fp8.run_matrix \
        --manifest data/sub/manifest_train.tsv \
        --val_manifest data/val/manifest_train.tsv \
        --n_calibration_batches 32 --batch_size 32 \
        --out_root hf_packages

To skip KD (PTQ-only matrix, fast)::

    uv run python -m gigaam.fp8.run_matrix ... --no_kd
"""

import argparse
import gc
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pyarrow.parquet as pq  # noqa: F401  (kept for parity with data.py usage)

DATASET_ID = "govnejri/golos_mfa_punctuation"

# (variant, base HF revision, is_charwise, mode). The encoder quantization is
# identical for all; charwise variants just use a different tokenizer/vocab.
# `mode` selects the KD path: "rnnt" variants get grid-level KD; "ctc" variants
# are PTQ-only (FP8 PTQ is lossless on CTC, so KD adds no value).
VARIANTS = [
    ("e2e_ctc", "e2e_ctc", False, "ctc"),
    ("ctc", "ctc", True, "ctc"),
    ("e2e_rnnt", "e2e_rnnt", False, "rnnt"),
    ("rnnt", "rnnt", True, "rnnt"),
]

# Per-variant KD hyperparameters. Both RNNT variants materialise a [b, T, U+1, V]
# grid per sub-batch plus a matching teacher grid during KD; e2e_rnnt (V=1025) is
# the heaviest. On a 12 GB dev GPU a long-utterance batch can spike VRAM and OOM
# mid-epoch even for charwise rnnt (V=34, ~30x lighter per cell), so both use a
# small batch. rnnt_subbatch_size=2 keeps each grid slice within budget.
KD_DEFAULTS: Dict[str, Dict[str, object]] = {
    "e2e_rnnt": {"batch_size": 4, "eval_batch_size": 4, "rnnt_subbatch_size": 2},
    "rnnt": {"batch_size": 4, "eval_batch_size": 4, "rnnt_subbatch_size": 2},
}


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


def _find_best_checkpoint(ckpt_root: str, variant: str) -> Optional[str]:
    """Find the best KD checkpoint under ``ckpt_root`` for ``variant``.

    ``train.py`` writes ``gigaam-<variant>-epoch=..-step=..-val_wer=X.ckpt`` and
    monitors ``val_wer`` (mode=min). We glob all matching files and pick the one
    with the lowest val_wer in its filename.
    """
    pattern = os.path.join(ckpt_root, "**", f"gigaam-{variant}-*val_wer=*.ckpt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    best, best_wer = None, float("inf")
    for path in candidates:
        m = re.search(r"val_wer=([0-9.]+)\.ckpt$", os.path.basename(path))
        if not m:
            continue
        wer = float(m.group(1))
        if wer < best_wer:
            best, best_wer = path, wer
    return best


def _run_kd_subprocess(
    variant: str,
    revision: str,
    is_charwise: bool,
    manifest: str,
    val_manifest: str,
    ckpt_root: str,
    scales_path: str,
    kd_max_steps: int,
    kd_alpha: float,
    kd_batch_size: Optional[int],
    rnnt_subbatch_size: Optional[int],
    accelerator: str,
) -> str:
    """Launch ``train.py`` for one RNNT variant and return the best checkpoint path.

    Runs as a subprocess so the full Lightning Trainer setup (DDP, callbacks,
    checkpointing) is reused verbatim. Per-variant hyperparameter defaults come
    from :data:`KD_DEFAULTS`, overridable via the CLI.
    """
    hp = dict(KD_DEFAULTS[variant])
    if kd_batch_size is not None:
        hp["batch_size"] = hp["eval_batch_size"] = kd_batch_size
    if rnnt_subbatch_size is not None:
        hp["rnnt_subbatch_size"] = rnnt_subbatch_size
    val_check = max(1, kd_max_steps // 4)

    # train.py lives in train_utils/ and uses `from module import ...`, so run it
    # with cwd = train_utils/ and pass absolute manifest paths.
    # __file__ = <repo>/gigaam/fp8/run_matrix.py -> parents: [fp8, gigaam, <repo>]
    repo_root = Path(__file__).resolve().parents[2]
    train_dir = repo_root / "train_utils"
    cmd = [
        sys.executable, "train.py",
        "--model_name", variant,
        "--from_hf",
        "--train_manifest", str(Path(manifest).resolve()),
        "--val_manifest", str(Path(val_manifest).resolve()),
        "--batch_size", str(hp["batch_size"]),
        "--eval_batch_size", str(hp["eval_batch_size"]),
        "--rnnt_subbatch_size", str(hp["rnnt_subbatch_size"]),
        "--max_steps", str(kd_max_steps),
        "--val_check_steps", str(val_check),
        "--fp8",
        "--kd_alpha", str(kd_alpha),
        "--calibration_scales", str(Path(scales_path).resolve()),
        "--output_dir", str(Path(ckpt_root).resolve()),
        "--skip_initial_validation",
        "--accelerator", accelerator,
        "--devices", "1",
        "--lr", "2e-5",
    ]
    if is_charwise:
        cmd.append("--raw_text")

    print(f"  launching KD subprocess (cwd={train_dir}):\n    {' '.join(cmd)}")
    # expandable_segments reduces CUDA fragmentation on the 12 GB dev GPU so a
    # long-utterance grid slice can find a contiguous block mid-epoch.
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    result = subprocess.run(cmd, cwd=str(train_dir), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"KD subprocess for {variant} exited {result.returncode}")

    best = _find_best_checkpoint(ckpt_root, variant)
    if best is None:
        raise RuntimeError(
            f"KD for {variant} produced no checkpoint matching {variant} in {ckpt_root}"
        )
    print(f"  KD best checkpoint: {best}")
    return best


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
    """Package a live PTQ model in native convention (CTC path, no KD).

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
    val_manifest: str,
    out_root: str,
    n_calibration_batches: int,
    batch_size: int,
    device: str,
    do_kd: bool,
    kd_max_steps: int,
    kd_alpha: float,
    kd_batch_size: Optional[int],
    rnnt_subbatch_size: Optional[int],
    ckpt_root: str,
    accelerator: str,
) -> dict:
    """Run the full pipeline for one variant. Returns a summary dict."""
    out_dir = os.path.join(out_root, variant)
    print(f"\n{'#'*60}\n### VARIANT: {variant} (mode={mode})\n{'#'*60}")

    # --- Stage 1: PTQ (all variants) ---
    ptq = _stage_ptq(
        variant, revision, is_charwise, manifest, n_calibration_batches, batch_size, device, out_dir
    )
    model = ptq["model"]

    kd_ran = False
    if do_kd and mode == "rnnt":
        # Free the PTQ model before launching the KD subprocess (which loads its
        # own copy from HF); the subprocess produces a fresh checkpoint.
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # --- Stage 2: KD (RNNT only) ---
        best_ckpt = _run_kd_subprocess(
            variant, revision, is_charwise, manifest, val_manifest, ckpt_root,
            ptq["scales_path"], kd_max_steps, kd_alpha, kd_batch_size, rnnt_subbatch_size,
            accelerator,
        )
        kd_ran = True

        # --- Stage 3: package from KD checkpoint (native convention) ---
        from gigaam.fp8.package_hf import package_checkpoint_native

        package_checkpoint_native(
            ckpt_path=best_ckpt,
            out_dir=out_dir,
            model_name=f"{variant}_fp8_kd",
            base_revision=revision,
            activation_scales_path=ptq["scales_path"],
            kd_alpha=kd_alpha,
            trained_steps=kd_max_steps,
        )
    else:
        # --- Stage 3: package from live PTQ model (native convention, no KD) ---
        reason = "disabled (--no_kd)" if not do_kd else "CTC (PTQ is lossless)"
        print(f"  skipping KD for {variant}: {reason}")
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
        "kd": kd_ran,
    }


def main():
    p = argparse.ArgumentParser(description="Run the full FP8 matrix (PTQ + KD + package).")
    p.add_argument("--manifest", required=True, help="TSV manifest for calibration + KD train.")
    p.add_argument("--val_manifest", default=None, help="TSV manifest for KD validation.")
    p.add_argument("--n_calibration_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=32, help="PTQ calibration batch size.")
    p.add_argument("--out_root", default="hf_packages")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--accelerator", default="gpu" if torch.cuda.is_available() else "cpu",
        help="Lightning accelerator for the KD subprocess (gpu/cpu).",
    )
    p.add_argument(
        "--variants", nargs="+", default=None,
        help="Subset of variants to process (default: all 4).",
    )
    # KD controls.
    p.add_argument("--no_kd", action="store_true", help="Skip KD entirely (PTQ-only matrix).")
    p.add_argument("--kd_max_steps", type=int, default=200)
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument(
        "--kd_batch_size", type=int, default=None,
        help="Override KD batch size for all RNNT variants (default: per-variant).",
    )
    p.add_argument(
        "--rnnt_subbatch_size", type=int, default=None,
        help="Override RNNT sub-batch size (default: per-variant; e2e_rnnt=2).",
    )
    p.add_argument(
        "--ckpt_root", default="train_utils/checkpoints",
        help="Where KD checkpoints are written / searched.",
    )
    args = p.parse_args()

    if not args.no_kd and args.val_manifest is None:
        # KD needs a val manifest. Fall back to the train manifest if not given.
        args.val_manifest = args.manifest
        print(f"[info] --val_manifest not given; using --manifest for KD validation")

    variants = VARIANTS
    if args.variants:
        variants = [v for v in VARIANTS if v[0] in args.variants]
        if not variants:
            raise ValueError(f"No matching variants in {args.variants}")
    # If running KD, any RNNT variant requires a non-trivial GPU.
    rnnt_present = any(v[3] == "rnnt" for v in variants)
    if not args.no_kd and rnnt_present and args.accelerator == "cpu":
        print("[warn] KD with RNNT on CPU will be extremely slow; consider --accelerator gpu.")

    os.makedirs(args.out_root, exist_ok=True)
    summaries = []
    for variant, revision, is_charwise, mode in variants:
        try:
            s = process_variant(
                variant, revision, is_charwise, mode,
                args.manifest, args.val_manifest, args.out_root,
                args.n_calibration_batches, args.batch_size, args.device,
                do_kd=not args.no_kd,
                kd_max_steps=args.kd_max_steps, kd_alpha=args.kd_alpha,
                kd_batch_size=args.kd_batch_size, rnnt_subbatch_size=args.rnnt_subbatch_size,
                ckpt_root=args.ckpt_root, accelerator=args.accelerator,
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
