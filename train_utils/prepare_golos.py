"""
Materialize ``govnejri/golos_mfa_punctuation`` (Golos, 970k clips / ~1044 h)
to audio files plus TSV manifests for fine-tuning.

Despite the dataset card, the parquet rows hold **Ogg Opus** (16 kHz mono)
blobs, not WAV. Decoding 970k of them is what makes a naive export slow and
120 GB large. This script is fast and resumable instead:

* Clips are written as the original bytes: ``.opus`` for the Opus blobs
  (~15 GB for the whole set), ``.wav`` for plain PCM16 WAV blobs. Nothing is
  decoded on the hot path; the training loader, ``gigaam.load_audio`` and
  soundfile all decode Opus directly (libopus, identical samples). Pass
  ``--decode_to_wav`` to get the old decoded 16-bit WAV output (~120 GB).
* The work unit is one parquet *row group* (~5.8k clips; 5 per shard), not a
  whole shard, so 16 workers stay busy to the end instead of waiting on the
  slowest shard.
* Every finished unit leaves a manifest fragment under ``<out_dir>/fragments``;
  a rerun skips units whose fragment exists and only merges. Interrupt and
  restart freely.

Validation is held out by whole shards (the last ones), so train and val never
share a shard.

Example (everything: ~16 GB parquet download cached under ~/.cache/huggingface,
~15 GB of .opus written; the cache can be deleted afterwards):

    python prepare_golos.py --out_dir /data/golos --workers 16

A bounded subset for a quick experiment (first ~50 h of train, ~1 h of val):

    python prepare_golos.py --out_dir /data/golos_small --train_hours 50 --val_hours 1

Output::

    <out_dir>/audio/<shard>/rg<k>_<row>.opus   (or .wav)
    <out_dir>/fragments/<split>__<shard>__rg<k>.tsv   (resume markers)
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
SAMPLE_RATE = 16000


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


def _download(shard: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET_ID, filename=shard, repo_type="dataset")


def _num_row_groups(shard: str) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(_download(shard)).num_row_groups


def _probe(b: bytes) -> Optional[Tuple[float, str]]:
    """(seconds, extension) if the blob is a 16 kHz mono stream we can store
    as-is (Ogg Opus -> ``.opus``, PCM16 WAV -> ``.wav``), else None."""
    if len(b) < 44:
        return None
    if b[:4] == b"OggS":
        ext = "opus"
    elif b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        ext = "wav"
    else:
        return None
    import soundfile as sf

    info = sf.info(io.BytesIO(b))  # header parse only, no decode
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        return None
    if ext == "opus" and info.subtype != "OPUS":
        return None
    if ext == "wav" and info.subtype != "PCM_16":
        return None
    return info.frames / SAMPLE_RATE, ext


def _materialize_unit(
    shard: str,
    rg: int,
    out_dir: str,
    max_duration: float,
    min_duration: float,
    max_seconds: Optional[float],
    decode_to_wav: bool,
) -> Tuple[str, int, int, float]:
    """Write one row group's clips (raw bytes, or decoded PCM16 WAV when
    ``decode_to_wav`` or the blob is an unexpected format) and its manifest
    fragment. Returns (shard name, row group, clips written, seconds)."""
    import pyarrow.parquet as pq
    import soundfile as sf

    name = Path(shard).stem  # e.g. train-00003-of-00034
    audio_dir = Path(out_dir) / "audio" / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    table = pq.ParquetFile(_download(shard)).read_row_group(
        rg, columns=["audio", "text", "duration"]
    )
    audio = table.column("audio").combine_chunks().field("bytes").to_pylist()
    texts = table.column("text").to_pylist()
    durations = table.column("duration").to_pylist()

    rows: List[Tuple[str, float, str]] = []
    seconds = 0.0
    for j, (b, text, dur) in enumerate(zip(audio, texts, durations)):
        dur = float(dur)
        if dur < min_duration or dur > max_duration:
            continue
        probe = None if decode_to_wav else _probe(b)
        if probe is not None:
            exact, ext = probe
            rel = f"audio/{name}/rg{rg}_{j:05d}.{ext}"
            with open(Path(out_dir) / rel, "wb") as f:
                f.write(b)
        else:  # decode and re-encode as 16-bit WAV
            rel = f"audio/{name}/rg{rg}_{j:05d}.wav"
            try:
                arr, sr = sf.read(io.BytesIO(b), dtype="int16", always_2d=False)
            except Exception:
                continue
            if arr.ndim == 2:
                arr = arr.mean(axis=1).astype("int16")
            if sr != SAMPLE_RATE:
                continue
            sf.write(str(Path(out_dir) / rel), arr, sr, subtype="PCM_16")
            exact = arr.shape[0] / sr
        rows.append((rel, round(exact, 3), " ".join(text.split())))
        seconds += exact
        if max_seconds is not None and seconds >= max_seconds:
            break

    frag = _fragment_path(out_dir, shard, rg)
    _write_rows(frag.with_suffix(".tmp"), rows, header=False)
    os.replace(frag.with_suffix(".tmp"), frag)  # atomic: a fragment is complete
    return name, rg, len(rows), seconds


def _fragment_path(out_dir: str, shard: str, rg: int) -> Path:
    return Path(out_dir) / "fragments" / f"{Path(shard).stem}__rg{rg}.tsv"


def _write_rows(path: Path, rows: List[Tuple[str, float, str]], header: bool) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if header:
            w.writerow(["path", "duration", "transcription"])
        for rel, dur, text in rows:
            w.writerow([rel, f"{dur:.3f}", text])


def _read_fragment(path: Path) -> List[Tuple[str, float, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r[0], float(r[1]), r[2]) for r in csv.reader(f, delimiter="\t") if r]


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
    p.add_argument(
        "--decode_to_wav",
        action="store_true",
        help="Decode every clip to 16-bit WAV instead of storing the original "
        "Opus bytes (8x the disk, ~an hour of CPU; only if a consumer cannot "
        "read Opus).",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    (out / "fragments").mkdir(parents=True, exist_ok=True)
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
        # diversity, and takes an equal slice from each row group.
        if hours is not None:
            k = max(1, min(len(shard_list), -(-int(hours) // 25) or 1))
            stride = max(1, len(shard_list) // k)
            shard_list = shard_list[::stride][:k]
        # Row groups are the work units (downloads happen here, in parallel).
        with ProcessPoolExecutor(
            max_workers=max(1, min(args.workers, len(shard_list)))
        ) as ex:
            n_rg = dict(zip(shard_list, ex.map(_num_row_groups, shard_list)))
        units = [(s, rg) for s in shard_list for rg in range(n_rg[s])]
        per_unit = None if hours is None else hours * 3600.0 / len(units) + 1e-9
        todo = [u for u in units if not _fragment_path(str(out), *u).exists()]
        print(
            f"  [{label}] {len(units)} row groups in {len(shard_list)} shards, "
            f"{len(units) - len(todo)} already done",
            flush=True,
        )
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [
                ex.submit(
                    _materialize_unit,
                    s,
                    rg,
                    str(out),
                    args.max_duration,
                    args.min_duration,
                    per_unit,
                    args.decode_to_wav,
                )
                for s, rg in todo
            ]
            for fut in as_completed(futs):
                name, rg, n, seconds = fut.result()
                done += 1
                print(
                    f"  [{label}] {name} rg{rg}: {n} clips, {seconds / 3600:.2f} h "
                    f"({done}/{len(todo)})",
                    flush=True,
                )
        rows: List[Tuple[str, float, str]] = []
        for u in units:
            rows.extend(_read_fragment(_fragment_path(str(out), *u)))
        rows.sort()
        return rows

    val_rows = run(val_shards, args.val_hours, "val")
    train_rows = run(train_shards, args.train_hours, "train")
    _write_rows(out / "manifest_train.tsv", train_rows, header=True)
    _write_rows(out / "manifest_val.tsv", val_rows, header=True)
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
