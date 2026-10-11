"""
Pseudo-label an audio manifest with the OFFLINE teacher model (self-distillation
Tier 1). Writes a new TSV whose ``transcription`` column is the teacher's own
greedy output, so the streaming student is trained to *restore* the offline model
rather than to fit (possibly imperfect) dataset labels.

Only the audio is used from the input manifest; any existing ``transcription`` is
ignored. Output rows keep the original ``path``/``duration`` and order.

Audio is decoded by ``--num_workers`` DataLoader processes (torchaudio; WAV,
Opus, FLAC, ...) in duration-sorted batches, so the GPU is fed continuously;
the teacher runs batched in fp16 on the GPU.

Example:
    python pseudo_label.py \\
        --teacher v3_e2e_ctc \\
        --in_manifest /data/golos/manifest_train.tsv \\
        --out_manifest /data/golos/manifest_train.teacher.tsv \\
        --batch_seconds 2000 --num_workers 16
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

import gigaam
from gigaam.utils import AudioDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="v3_e2e_ctc", help="Offline teacher model name")
    p.add_argument("--in_manifest", required=True)
    p.add_argument("--out_manifest", required=True)
    p.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Maximum clips per batch (the audio budget below usually binds first).",
    )
    p.add_argument(
        "--batch_seconds",
        type=float,
        default=2000.0,
        help="Audio seconds per batch. Batches are duration-sorted, so short clips "
        "are grouped by the hundreds and every forward carries enough work to "
        "fill the GPU; long clips form small batches automatically.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) // 2)),
        help="Audio-decoding worker processes feeding the GPU.",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--from_hf",
        action="store_true",
        help="Load teacher from the HuggingFace mirror",
    )
    p.add_argument(
        "--max_batch_seconds",
        type=float,
        default=40.0,
        help="Clips longer than this are decoded alone (memory guard).",
    )
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    loader = gigaam.load_model_from_hf if args.from_hf else gigaam.load_model
    model = loader(args.teacher, device=args.device)
    model.eval()
    assert isinstance(model, gigaam.GigaAMASR), "teacher must be an ASR model"
    dev = model._device

    with open(args.in_manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # No duration filter: every input row gets a label, in order.
    ds = AudioDataset(args.in_manifest, tokenizer=None, min_duration=0.0)
    assert len(ds) == len(rows), "manifest rows and dataset samples differ"
    durations = [s.duration for s in ds.samples]
    print(
        f"Pseudo-labeling {len(rows)} clips ({sum(durations) / 3600:.1f} h) with "
        f"offline {args.teacher} on {dev} ..."
    )

    # Duration-sorted batches (little padding) closed by an audio budget: the
    # padded size is len(batch) * max duration, which the sort keeps close to
    # the real audio. Long clips are decoded alone.
    batches: List[List[int]] = []
    cur: List[int] = []
    for i in sorted(range(len(ds)), key=durations.__getitem__):
        if durations[i] > args.max_batch_seconds:
            batches.append([i])
            continue
        padded = (len(cur) + 1) * durations[i]  # sorted: this clip is the longest
        if cur and (padded > args.batch_seconds or len(cur) >= args.batch_size):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    print(
        f"  {len(batches)} batches, {len(rows) / max(len(batches), 1):.0f} clips "
        f"per batch on average"
    )

    dl_kw = dict(num_workers=args.num_workers, pin_memory=dev.type == "cuda")
    if args.num_workers > 0:
        dl_kw["prefetch_factor"] = 4
    dl = DataLoader(ds, batch_sampler=batches, collate_fn=ds.collate_fn, **dl_kw)

    out_text: List[str] = [""] * len(rows)
    done = 0
    audio_s = 0.0
    t0 = time.perf_counter()
    for b, (batch_idx, (wav_pad, lens)) in enumerate(zip(batches, dl)):
        wav_pad = wav_pad.to(dev, non_blocking=True).to(model._dtype)
        lens = lens.to(dev)
        encoded, enc_len = model.forward(wav_pad, lens)
        decoded = model.decoding.decode(model.head, encoded, enc_len)
        for i, (text, _ids, _fr) in zip(batch_idx, decoded):
            out_text[i] = text
        done += len(batch_idx)
        audio_s += sum(durations[i] for i in batch_idx)
        if (b + 1) % 50 == 0 or done == len(rows):
            el = time.perf_counter() - t0
            print(
                f"  {done}/{len(rows)} clips, {audio_s / 3600:.1f} h audio, "
                f"{audio_s / max(el, 1e-9):.0f}x realtime",
                flush=True,
            )

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "duration", "transcription"])
        for row, txt in zip(rows, out_text):
            n_empty += not txt.strip()
            w.writerow([row["path"], row["duration"], txt])
    print(f"Wrote {out_path} ({len(rows)} rows, {n_empty} empty transcripts)")


if __name__ == "__main__":
    main()
