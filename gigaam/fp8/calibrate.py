"""Calibrate FP8 activation scales and report post-training-quantization (PTQ) WER.

Workflow
--------
1. Load the fp16 teacher (``v3_e2e_ctc`` by default).
2. ``quantize_encoder`` the student (weights are per-channel FP8 immediately).
3. Run ``n_batches`` of real data through the student, recording the running
   max-abs (amax) of each ``FP8Linear`` input activation, per-tensor.
4. ``bake_activation_scales`` the per-tensor activation scales into the student.
5. Measure greedy-decode WER on a held-out manifest and compare to the fp16
   teacher WER (the gap KD must close).

This is Stage 1 of the pipeline: the cheapest checkpoint and the quantification of
how much accuracy FP8 PTQ alone costs before any knowledge distillation.

Usage
-----
    uv run python -m gigaam.fp8.calibrate \\
        --manifest path/to/manifest_val.tsv \\
        --model_name v3_e2e_ctc \\
        --n_batches 64 --batch_size 16
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import gigaam
from gigaam.fp8 import (
    FP8Linear,
    bake_activation_scales,
    export_fp8_model,
    quantize_model,
)
from gigaam.fp8.linear import E4M3_MAX
from gigaam.utils import AudioDataset


class ActivationCalibrator:
    """Collects running per-tensor amax of every FP8Linear's input activation.

    Hooks both encoder and head FP8Linears; keys are model-root dotted paths
    (``encoder.<...>`` / ``head.<...>``), matching ``bake_activation_scales``.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.amax: Dict[str, float] = {}
        self._handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, FP8Linear):
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(_module, inputs):
            x = inputs[0].detach().float()
            cur = float(x.abs().amax())
            if cur > self.amax.get(name, 0.0):
                self.amax[name] = cur

        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.inference_mode()
def measure_wer(
    model: gigaam.GigaAMASR,
    manifest: str,
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "cuda",
    max_duration: float = 25.0,
    limit: int = 0,
) -> "tuple[float, int, int]":
    """Greedy-decode ``model`` on ``manifest`` and return (wer, errors, words).

    Reuses the model's own preprocessor + head + decoding for parity with eval.py.
    """
    import editdistance

    ds = AudioDataset(manifest)
    samples = ds.samples
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=num_workers,
    )
    errors = words = 0
    n = 0
    for wavs, wav_lens in tqdm(dl, desc="Decoding"):
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)
        encoded, enc_len = model(wavs, wav_lens)
        decoded = model.decoding.decode(model.head, encoded, enc_len)
        for (text, _, _), sample in zip(decoded, samples):
            ref = (sample.text or "").split()
            hyp = text.split()
            errors += editdistance.eval(ref, hyp)
            words += max(len(ref), 1)
            n += 1
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
    wer = errors / max(words, 1) * 100.0
    return wer, errors, words


def main():
    p = argparse.ArgumentParser(
        description="Calibrate FP8 activation scales + PTQ WER."
    )
    p.add_argument(
        "--manifest", required=True, help="TSV manifest for calibration + WER."
    )
    p.add_argument("--model_name", default="v3_e2e_ctc")
    p.add_argument(
        "--n_batches", type=int, default=64, help="Batches for activation calibration."
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--max_duration", type=float, default=25.0)
    p.add_argument(
        "--eval_limit", type=int, default=0, help="Cap WER eval samples (0 = all)."
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--export_dir", default=None, help="If set, export the calibrated student here."
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Patch the encoder into streaming mode (chunked block attention + "
        "causal convs/subsampling) BEFORE quantization. The resulting FP8 student "
        "is a streaming model; WER is measured through its chunked-mask forward. "
        "Order matters: enable_streaming always runs before quantize_model.",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="Streaming chunk in encoder frames (40 ms each). 8 = 320 ms latency.",
    )
    p.add_argument(
        "--left_context_chunks",
        type=int,
        default=-1,
        help="Streaming left-context window in chunks (-1 = unlimited past).",
    )
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"Loading teacher {args.model_name} (fp16) ...")
    teacher = gigaam.load_model(args.model_name, fp16_encoder=True, device=device)

    # --- Optional: convert the encoder to streaming mode BEFORE quantization ---
    # enable_streaming rebuilds the encoder from plain nn.Linear, so it must run
    # before quantize_model (which swaps in FP8Linear); the reverse order would
    # silently wipe the quantization. See gigaam.model.enable_streaming.
    if args.streaming:
        print(
            f"Patching encoder into streaming mode: chunk={args.chunk_size} frames "
            f"(~{args.chunk_size * 40} ms), left_context="
            f"{'unlimited' if args.left_context_chunks < 0 else args.left_context_chunks}"
        )
        gigaam.enable_streaming(
            teacher,
            chunk_size=args.chunk_size,
            left_context_chunks=args.left_context_chunks,
        )

    # --- Teacher WER baseline ---
    print("Measuring fp16 teacher WER ...")
    t_wer, t_err, t_w = measure_wer(
        teacher,
        args.manifest,
        args.eval_batch_size,
        args.num_workers,
        str(device),
        args.max_duration,
        args.eval_limit,
    )
    print(f"  teacher WER = {t_wer:.2f}% ({t_err}/{t_w})")

    # --- Build student: quantize encoder Linears to FP8 ---
    print("Quantizing encoder Linears to FP8 ...")
    student = teacher  # quantize in place on the same object
    quantize_model(student, compute_dtype=torch.float16)

    # --- Collect activation amax on real data ---
    print(f"Calibrating activation scales over {args.n_batches} batches ...")
    ds = AudioDataset(args.manifest)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=AudioDataset.collate,
        num_workers=args.num_workers,
    )
    calib = ActivationCalibrator(student)
    it = iter(dl)
    for b in tqdm(range(args.n_batches), desc="Calibrating"):
        try:
            wavs, wav_lens = next(it)
        except StopIteration:
            break
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)
        student(wavs, wav_lens)
    calib.remove()

    n_baked = bake_activation_scales(student, calib.amax)
    print(
        f"  baked {n_baked} activation scales "
        f"(amax range {min(calib.amax.values()):.3f}..{max(calib.amax.values()):.3f})"
    )

    # --- PTQ WER ---
    print("Measuring FP8-PTQ student WER ...")
    s_wer, s_err, s_w = measure_wer(
        student,
        args.manifest,
        args.eval_batch_size,
        args.num_workers,
        str(device),
        args.max_duration,
        args.eval_limit,
    )
    print(f"  student WER = {s_wer:.2f}% ({s_err}/{s_w})")
    print(f"  delta = {s_wer - t_wer:+.2f}% (KD target: recover toward {t_wer:.2f}%)")

    if args.export_dir:
        scales_path = Path(args.export_dir) / "activation_scales.json"
        scales_path.parent.mkdir(parents=True, exist_ok=True)
        scales_path.write_text(json.dumps(calib.amax, indent=2), encoding="utf-8")
        print(f"  wrote activation scales -> {scales_path}")
        export_fp8_model(
            student, args.export_dir, model_name=f"{args.model_name}_fp8_ptq"
        )
        print(f"  exported calibrated student -> {args.export_dir}")


if __name__ == "__main__":
    main()
