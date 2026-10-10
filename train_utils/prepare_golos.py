"""
Materialize ``govnejri/golos_mfa_punctuation`` (Golos, 970k clips / ~1044 h,
16 kHz mono WAV in parquet) to WAV files plus TSV manifests for fine-tuning.

Shards are processed in parallel worker processes and rows are written as they
are decoded, so memory stays flat however much of the dataset is requested.
Validation is held out by whole shards (the last ones), so train and val never
share a shard.

Example (everything, ~16 GB download, ~33 GB of WAV):

    python prepare_golos.py --out_dir /data/golos --workers 16

A bounded subset for a quick experiment (first ~50 h of train, ~1 h of val):

    python prepare_golos.py --out_dir /data/golos_small --train_hours 50 --val_hours 1

Output::

    <out_dir>/audio/<shard>/<row>.wav
    <out_dir>/manifest_train.tsv   path  duration  transcription   (punctuated text)
    <out_dir>/manifest_val.tsv

The ``transcription`` column keeps the dataset's punctuated text, which is what
the e2e models produce; ``train.py --raw_text`` normalizes it for the
character-level (non-e2e) models. For streaming distillation the training
transcripts are replaced by the offline teacher's output anyway
(``pseudo_label.py``); the text here serves validation WER.
"""

import argparse
import csv
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

DATASET_ID = "govnejri/golos_mfa_punctuation"


def _shard_files(split: str) -> List[str]:
    from huggingface_hub import HfApi

    files = [
        f
        for f in HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
        if f.startswith(f"data/{split}-") and f.endswith(".parquet")
    ]
    if not files:
        raise RuntimeError(
            f"no parquet shards found for split {split!r} in {DATASET_ID}"
        )
    return sorted(files)


def _materialize_shard(
    shard: str,
    out_dir: str,
    max_duration: float,
    min_duration: float,
    max_hours: Optional[float],
) -> Tuple[str, List[Tuple[str, float, str]], float]:
    """Download one shard (HF cache) and write its clips. Returns (shard name,
    manifest rows, seconds written). ``max_hours`` stops early for subsets."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(DATASET_ID, filename=shard, repo_type="dataset")
    name = Path(shard).stem  # e.g. train-00003-of-00034
    audio_dir = Path(out_dir) / "audio" / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, float, str]] = []
    seconds = 0.0
    budget = None if max_hours is None else max_hours * 3600.0
    pf = pq.ParquetFile(local)
    idx = 0
    for batch in pf.iter_batches(batch_size=256, columns=["audio", "text", "duration"]):
        for r in batch.to_pylist():
            idx += 1
            dur = float(r["duration"])
            if dur < min_duration or dur > max_duration:
                continue
            try:
                arr, sr = sf.read(
                    io.BytesIO(r["audio"]["bytes"]), dtype="int16", always_2d=False
                )
            except Exception:
                continue
            if arr.ndim == 2:
                arr = arr.mean(axis=1).astype("int16")
            if sr != 16000:
                continue  # the dataset is 16 kHz; skip anything odd rather than resample
            rel = f"audio/{name}/{idx:07d}.wav"
            sf.write(str(Path(out_dir) / rel), arr, sr, subtype="PCM_16")
            text = " ".join(r["text"].split())
            rows.append((rel, round(arr.shape[0] / sr, 3), text))
            seconds += arr.shape[0] / sr
            if budget is not None and seconds >= budget:
                return name, rows, seconds
    return name, rows, seconds


def _write_manifest(path: Path, rows: List[Tuple[str, float, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "duration", "transcription"])
        for rel, dur, text in rows:
            w.writerow([rel, f"{dur:.3f}", text])


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--split", default="train", help="dataset split (the dataset has only 'train')"
    )
    p.add_argument(
        "--train_hours",
        type=float,
        default=None,
        help="cap on training audio (default: all)",
    )
    p.add_argument(
        "--val_hours",
        type=float,
        default=2.0,
        help="validation audio, from held-out shards",
    )
    p.add_argument(
        "--val_shards",
        type=int,
        default=1,
        help="number of trailing shards reserved for validation",
    )
    p.add_argument("--max_duration", type=float, default=20.0)
    p.add_argument("--min_duration", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = _shard_files(args.split)
    val_shards = shards[-args.val_shards :] if args.val_shards > 0 else []
    train_shards = shards[: len(shards) - len(val_shards)]
    print(
        f"{len(shards)} shards: {len(train_shards)} train, {len(val_shards)} val",
        flush=True,
    )

    def run(shard_list: List[str], hours: Optional[float], label: str):
        if not shard_list:
            return []
        # A bounded budget reads only as many shards as it needs (each shard
        # holds ~30 h), spread evenly over the split for speaker/domain
        # diversity, and takes an equal slice from each.
        per_shard = None
        if hours is not None:
            k = max(1, min(len(shard_list), -(-int(hours) // 25) or 1))
            stride = max(1, len(shard_list) // k)
            shard_list = shard_list[::stride][:k]
            per_shard = hours / len(shard_list) + 1e-9
        rows: List[Tuple[str, float, str]] = []
        total = 0.0
        with ProcessPoolExecutor(
            max_workers=max(1, min(args.workers, len(shard_list)))
        ) as ex:
            futs = [
                ex.submit(
                    _materialize_shard,
                    s,
                    str(out),
                    args.max_duration,
                    args.min_duration,
                    per_shard,
                )
                for s in shard_list
            ]
            for fut in as_completed(futs):
                name, shard_rows, seconds = fut.result()
                rows.extend(shard_rows)
                total += seconds
                print(
                    f"  [{label}] {name}: {len(shard_rows)} clips, {seconds / 3600:.2f} h "
                    f"(running {total / 3600:.1f} h)",
                    flush=True,
                )
        rows.sort()
        return rows

    val_rows = run(val_shards, args.val_hours, "val")
    train_rows = run(train_shards, args.train_hours, "train")
    _write_manifest(out / "manifest_train.tsv", train_rows)
    _write_manifest(out / "manifest_val.tsv", val_rows)
    h_train = sum(r[1] for r in train_rows) / 3600
    h_val = sum(r[1] for r in val_rows) / 3600
    print(
        f"train: {len(train_rows)} clips, {h_train:.1f} h -> {out / 'manifest_train.tsv'}"
    )
    print(f"val:   {len(val_rows)} clips, {h_val:.2f} h -> {out / 'manifest_val.tsv'}")
    if not train_rows:
        sys.exit("no training rows written")


if __name__ == "__main__":
    main()
