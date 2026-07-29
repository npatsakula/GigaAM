"""
Pseudo-label an audio manifest with the OFFLINE teacher model (self-distillation
Tier 1). Writes a new TSV whose ``transcription`` column is the teacher's own
greedy output, so the streaming student is trained to *restore* the offline model
rather than to fit (possibly imperfect) dataset labels.

Only the audio is used from the input manifest; any existing ``transcription`` is
ignored. Output rows keep the original ``path``/``duration`` and order.

Example:
    python pseudo_label.py \
        --teacher v3_e2e_ctc \
        --in_manifest /data/golos/manifest_train.tsv \
        --out_manifest /data/golos/manifest_train.teacher.tsv \
        --batch_size 32
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

import gigaam
from gigaam.preprocess import SAMPLE_RATE, load_audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="v3_e2e_ctc", help="Offline teacher model name")
    p.add_argument("--in_manifest", required=True)
    p.add_argument("--out_manifest", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--from_hf", action="store_true", help="Load teacher from the HuggingFace mirror"
    )
    p.add_argument(
        "--max_batch_seconds",
        type=float,
        default=40.0,
        help="Clips longer than this are decoded alone (memory guard).",
    )
    return p.parse_args()


def _rows(manifest: str):
    data_dir = Path(manifest).resolve().parent
    with open(manifest) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pth = Path(row["path"])
            abspath = str((pth if pth.is_absolute() else data_dir / pth).resolve())
            yield row, abspath


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    loader = gigaam.load_model_from_hf if args.from_hf else gigaam.load_model
    model = loader(args.teacher, device=args.device)
    model.eval()
    assert isinstance(model, gigaam.GigaAMASR), "teacher must be an ASR model"
    dev = model._device

    rows = list(_rows(args.in_manifest))
    print(f"Pseudo-labeling {len(rows)} clips with offline {args.teacher} ...")

    # Sort by duration to minimise padding waste; remember original order.
    order = sorted(range(len(rows)), key=lambda i: float(rows[i][0]["duration"]))
    out_text: list[str] = [""] * len(rows)

    def flush(batch_idx: list[int]) -> None:
        wavs = [load_audio(rows[i][1]).to(dev) for i in batch_idx]
        lens = torch.tensor([w.shape[-1] for w in wavs], device=dev)
        wav_pad = pad_sequence(wavs, batch_first=True).to(model._dtype)
        encoded, enc_len = model.forward(wav_pad, lens)
        decoded = model.decoding.decode(model.head, encoded, enc_len)
        for i, (text, _ids, _fr) in zip(batch_idx, decoded):
            out_text[i] = text

    batch: list[int] = []
    done = 0
    for i in order:
        dur = float(rows[i][0]["duration"])
        if dur > args.max_batch_seconds:
            flush([i])
            done += 1
            continue
        batch.append(i)
        if len(batch) >= args.batch_size:
            flush(batch)
            done += len(batch)
            batch = []
            if done % (args.batch_size * 20) == 0:
                print(f"  {done}/{len(rows)}")
    if batch:
        flush(batch)
        done += len(batch)
    print(f"  {done}/{len(rows)} done")

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "duration", "transcription"])
        for i, (row, _abspath) in enumerate(rows):
            txt = out_text[i]
            n_empty += not txt.strip()
            w.writerow([row["path"], row["duration"], txt])
    print(f"Wrote {out_path} ({len(rows)} rows, {n_empty} empty transcripts)")


if __name__ == "__main__":
    main()
