"""
Evaluate a streaming CTC model. Primary metric is the RESTORATION GAP: WER of the
streaming model using the OFFLINE model's output as reference (how well streaming
reproduces offline). Absolute WER vs the dataset transcription is reported as a
secondary sanity check. Also measures real-time factor.

Example:
    python eval_streaming.py \
        --streaming_ckpt /ckpts/gigaam-...streaming.ckpt \
        --offline v3_e2e_ctc \
        --manifest /data/golos/manifest_val.tsv \
        --limit 500
"""

import argparse
import csv
import time
from pathlib import Path

import editdistance
import torch
from torch.nn.utils.rnn import pad_sequence

import gigaam


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r = ref.split()
    return editdistance.eval(r, hyp.split()), max(len(r), 1)


def cer(ref: str, hyp: str) -> tuple[int, int]:
    """Character error rate. editdistance over the raw strings (no split): the
    edit distance between two strings IS the character-level distance. The
    reference length is the character count (excluding the trailing newline)."""
    r = ref.rstrip("\n")
    return editdistance.eval(r, hyp.rstrip("\n")), max(len(r), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--streaming_ckpt",
        required=True,
        help="Fine-tuned streaming .ckpt, or a base model name to eval zero-shot "
        "(will be patched with enable_streaming).",
    )
    p.add_argument(
        "--offline", default="v3_e2e_ctc", help="Offline teacher/reference model"
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Streams decoded together (StreamingBatch) and offline batch size.",
    )
    p.add_argument(
        "--chunk_size", type=int, default=8, help="Only for zero-shot patching"
    )
    p.add_argument(
        "--left_context_chunks",
        type=int,
        default=16,
        help="Only for zero-shot patching",
    )
    p.add_argument(
        "--right_context_chunks",
        type=int,
        default=0,
        choices=[0, 1],
        help="Only for zero-shot patching",
    )
    p.add_argument(
        "--from_hf",
        action="store_true",
        help="Load the offline reference (and zero-shot base) from the HF mirror.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load = gigaam.load_model_from_hf if args.from_hf else gigaam.load_model

    # fp32 throughout so the restoration gap is a clean streaming-vs-offline
    # comparison, not confounded by fp16 rounding differences.
    if Path(args.streaming_ckpt).is_file():
        # A fine-tuned .ckpt is always loaded via load_model (it reads the ckpt and
        # rebuilds the base, falling back to HF automatically if the CDN is down).
        stream_model = gigaam.load_model(
            args.streaming_ckpt, device=args.device, fp16_encoder=False
        )
    else:  # zero-shot: patch a base model into streaming mode
        stream_model = load(args.streaming_ckpt, device=args.device, fp16_encoder=False)
        gigaam.enable_streaming(
            stream_model,
            chunk_size=args.chunk_size,
            left_context_chunks=args.left_context_chunks,
            right_context_chunks=args.right_context_chunks,
        )
    stream_model.eval()
    assert getattr(stream_model.encoder, "streaming", False), "not a streaming model"

    offline = load(args.offline, device=args.device, fp16_encoder=False).eval()

    data_dir = Path(args.manifest).resolve().parent
    rows = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]

    gap_e = gap_w = abs_e = abs_w = 0
    gap_cer_e = gap_cer_n = abs_cer_e = abs_cer_n = 0
    audio_s = infer_s = 0.0
    paths = [
        str(
            (
                Path(row["path"])
                if Path(row["path"]).is_absolute()
                else data_dir / row["path"]
            ).resolve()
        )
        for row in rows
    ]
    # Batched: offline references through one padded forward per batch, the
    # streaming model through StreamingBatch (identical results to one stream at
    # a time; RTF is then the batched streaming compute per second of audio).
    ref_texts: list[str] = []
    hyps: list[str] = []
    for s in range(0, len(paths), args.batch_size):
        chunk = paths[s : s + args.batch_size]
        wavs = [gigaam.load_audio(p).to(offline._device) for p in chunk]
        lens = torch.tensor([w.shape[-1] for w in wavs], device=offline._device)
        with torch.inference_mode():
            enc, enc_len = offline.forward(
                pad_sequence(wavs, batch_first=True).to(offline._dtype), lens
            )
            ref_texts.extend(
                t for t, _, _ in offline.decoding.decode(offline.head, enc, enc_len)
            )
        t0 = time.perf_counter()
        hyps.extend(r.text for r in stream_model.transcribe_streaming_batch(chunk))
        infer_s += time.perf_counter() - t0
        print(f"  {min(s + args.batch_size, len(paths))}/{len(paths)}", flush=True)

    for i, row in enumerate(rows):
        ref_off, hyp = ref_texts[i], hyps[i]
        audio_s += float(row["duration"])

        e, w = wer(ref_off, hyp)
        gap_e += e
        gap_w += w
        ce, cn = cer(ref_off, hyp)
        gap_cer_e += ce
        gap_cer_n += cn
        if "transcription" in row and row["transcription"]:
            e2, w2 = wer(row["transcription"].lower(), hyp)
            abs_e += e2
            abs_w += w2
            ce2, cn2 = cer(row["transcription"].lower(), hyp)
            abs_cer_e += ce2
            abs_cer_n += cn2
        if i < 5:
            print(f"OFF: {ref_off}\nSTR: {hyp}\n")

    print(f"n={len(rows)}")
    print(
        f"RESTORATION GAP (streaming vs offline) WER: {100 * gap_e / max(gap_w, 1):.2f}%   "
        f"CER: {100 * gap_cer_e / max(gap_cer_n, 1):.2f}%"
    )
    if abs_w:
        print(
            f"absolute vs dataset                         WER: {100 * abs_e / max(abs_w, 1):.2f}%   "
            f"CER: {100 * abs_cer_e / max(abs_cer_n, 1):.2f}%"
        )
    print(
        f"RTF (streaming compute / audio):            {infer_s / max(audio_s, 1e-9):.3f}"
    )


if __name__ == "__main__":
    main()
