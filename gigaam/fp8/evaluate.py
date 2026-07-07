"""Evaluate fp16 vs FP8 transcription quality on a manifest.

Transcribes every audio in ``--manifest`` with the original fp16 model, then
quantizes in place (PTQ) and transcribes again with the FP8 model. Reports:

- WER and CER of each model against the manifest reference text.
- The direct fp16-vs-FP8 disagreement rate (how often the transcripts differ,
  word- and character-level), which isolates the quantization effect from the
  reference-text quality (the dataset text is auto-punctuated/un-normalized, so
  its WER is inflated identically for both models).

This reuses :func:`load_model_from_hf` + :func:`quantize_model` + the model's own
``transcribe`` path, so the comparison reflects real inference behaviour. The
fp16 model is decoded *before* in-place quantization so the two transcriptions
are independent.

Usage
-----
    uv run python -m gigaam.fp8.evaluate \
        --model_name e2e_ctc --manifest data/val/manifest_train.tsv \
        --n_calibration_batches 32 --batch_size 16
"""

import argparse
import editdistance
import json
import os
import sys
from typing import List

import torch
from tqdm.auto import tqdm

import gigaam
from gigaam.utils import AudioDataset


def _levenshtein_per_sample(refs: List[str], hyps: List[str]):
    """Return (total_char_errors, total_chars, total_word_errors, total_words)."""
    ce = cw = we = ww = 0
    for r, h in zip(refs, hyps):
        rc, hc = r, h
        ce += editdistance.eval(rc, hc)
        cw += max(len(rc), 1)
        rw, hw = r.split(), h.split()
        we += editdistance.eval(rw, hw)
        ww += max(len(rw), 1)
    return ce, cw, we, ww


@torch.inference_mode()
def transcribe_all(model, manifest: str, max_duration: float = 25.0) -> List[str]:
    """Transcribe every audio in ``manifest`` with ``model``, in order."""
    from torch.utils.data import DataLoader

    ds = AudioDataset(manifest, max_duration=max_duration)
    dl = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=ds.collate_fn, num_workers=0
    )
    texts: List[str] = []
    dev = next(model.parameters()).device
    for wavs, wav_lens in tqdm(dl, desc="Transcribing"):
        encoded, enc_len = model(wavs.to(dev), wav_lens.to(dev))
        decoded = model.decoding.decode(model.head, encoded, enc_len)
        texts.append(decoded[0][0])
    return texts


def _read_refs(manifest: str) -> List[str]:
    refs = []
    with open(manifest) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            refs.append(parts[2] if len(parts) > 2 else "")
    return refs


def main():
    p = argparse.ArgumentParser(description="Compare fp16 vs FP8 transcription quality.")
    p.add_argument("--model_name", required=True, help="Variant: e2e_ctc, ctc, e2e_rnnt, rnnt.")
    p.add_argument("--manifest", required=True, help="TSV manifest (path, duration, text).")
    p.add_argument("--n_calibration_batches", type=int, default=32,
                   help="Batches for FP8 activation calibration.")
    p.add_argument("--batch_size", type=int, default=16, help="Calibration batch size.")
    p.add_argument("--max_duration", type=float, default=25.0)
    p.add_argument("--limit", type=int, default=0, help="Cap samples (0 = all).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="Write JSON results here.")
    args = p.parse_args()

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    # --- Load fp16 model ---
    print(f"Loading fp16 {args.model_name} ...")
    model = gigaam.load_model_from_hf(args.model_name, fp16_encoder=True, device=args.device)
    model.eval()
    refs = _read_refs(args.manifest)
    if args.limit:
        refs = refs[: args.limit]

    # --- fp16 transcription ---
    print(f"\n[1/3] fp16 transcription ({len(refs)} samples) ...")
    fp16_texts = transcribe_all(model, args.manifest, args.max_duration)
    if args.limit:
        fp16_texts = fp16_texts[: args.limit]

    # --- Quantize + calibrate in place ---
    print(f"\n[2/3] FP8 PTQ: quantize + calibrate ({args.n_calibration_batches} batches) ...")
    from gigaam.fp8 import quantize_model, bake_activation_scales
    from gigaam.fp8.calibrate import ActivationCalibrator
    from torch.utils.data import DataLoader

    quantize_model(model, compute_dtype=torch.float16)
    ds = AudioDataset(args.manifest, max_duration=args.max_duration)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=ds.collate_fn, num_workers=4)
    calib = ActivationCalibrator(model)
    with torch.no_grad():
        for i, (wavs, lens) in enumerate(tqdm(dl, total=args.n_calibration_batches,
                                              desc="Calibrating")):
            model(wavs.to(args.device), lens.to(args.device))
            if i + 1 >= args.n_calibration_batches:
                break
    calib.remove()
    n_baked = bake_activation_scales(model, calib.amax)
    print(f"  baked {n_baked} activation scales")

    # --- FP8 transcription ---
    print(f"\n[3/3] FP8 transcription ({len(refs)} samples) ...")
    fp8_texts = transcribe_all(model, args.manifest, args.max_duration)
    if args.limit:
        fp8_texts = fp8_texts[: args.limit]

    # --- Metrics ---
    # The primary metric is the fp16<->FP8 disagreement: it isolates the
    # quantization effect from reference-text quality. (This dataset's text is
    # auto-punctuated with un-normalized spellings, so WER-vs-GT is dominated by
    # reference artifacts that are identical for both models.) We still report the
    # GT comparison as secondary context, with the delta that filters out the
    # shared reference noise.
    n = len(refs)
    print("\n" + "=" * 64)
    print(f"=== {args.model_name}  ({n} samples)")
    print("=" * 64)

    def gt_metrics(hyps):
        ce, cw, we, ww = _levenshtein_per_sample(refs, hyps)
        return {"wer": we / ww * 100, "cer": ce / cw * 100,
                "word_errors": we, "words": ww, "char_errors": ce, "chars": cw}

    fp16_m = gt_metrics(fp16_texts)
    fp8_m = gt_metrics(fp8_texts)

    # Primary: fp16 vs FP8 disagreement.
    ce, cw, we, ww = _levenshtein_per_sample(fp16_texts, fp8_texts)
    n_changed = sum(1 for a, b in zip(fp16_texts, fp8_texts) if a != b)
    n_exact = n - n_changed
    disagree = {
        "wer": we / ww * 100, "cer": ce / cw * 100,
        "word_errors": we, "words": ww, "char_errors": ce, "chars": cw,
        "samples_changed": n_changed,
        "samples_exact_match": n_exact,
        "exact_match_rate": n_exact / n * 100,
    }
    print(f"\n  [quantization effect]  fp16 vs FP8 disagreement")
    print(f"    word disagreement : {disagree['wer']:.2f}%  ({we}/{ww} words differ)")
    print(f"    char disagreement: {disagree['cer']:.2f}%  ({ce}/{cw} chars differ)")
    print(f"    transcripts identical for {n_exact}/{n} samples "
          f"({disagree['exact_match_rate']:.1f}%)")

    # Secondary: vs-GT (noisy reference). Only the delta (fp8-gt minus fp16-gt)
    # is meaningful; the absolute values reflect reference-text artifacts.
    print(f"\n  [reference context, shared-noise]  vs dataset text")
    print(f"    fp16 WER={fp16_m['wer']:.2f}%  FP8 WER={fp8_m['wer']:.2f}%  "
          f"(Δ={fp8_m['wer']-fp16_m['wer']:+.2f})")
    print(f"    fp16 CER={fp16_m['cer']:.2f}%  FP8 CER={fp8_m['cer']:.2f}%  "
          f"(Δ={fp8_m['cer']-fp16_m['cer']:+.2f})")

    summary = {
        "model_name": args.model_name,
        "manifest": args.manifest,
        "n_samples": n,
        "fp8_vs_fp16_disagreement": disagree,
        "fp16_vs_gt": fp16_m,
        "fp8_vs_gt": fp8_m,
    }
    print("\n" + json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
