"""Materialize a bounded subset of the Golos-punctuation HF dataset to TSV.

The full dataset (``govnejri/golos_mfa_punctuation``) is ~970k rows / ~1044h — far
more than calibration / KD needs. This helper streams a bounded number of rows from
the parquet shards, writes the audio to ``<out_dir>/audio/`` and a TSV manifest
(``path``, ``duration``, ``transcription``) that ``gigaam.utils.AudioDataset`` reads
directly. It mirrors the ``save_split`` pattern in ``train_utils/utils.py``.

The dataset's ``text`` field is punctuated, which matches the e2e models
(``v3_e2e_ctc`` / ``v3_e2e_rnnt``) — so no text normalization is applied here.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import soundfile as sf
from tqdm.auto import tqdm

DATASET_ID = "govnejri/golos_mfa_punctuation"


def _decode_audio_bytes(audio_struct: dict) -> "tuple":
    """Decode an HF audio struct ``{bytes, path}`` to (array, sample_rate).

    The parquet stores raw bytes; decode via soundfile on the embedded WAV/FLAC.
    """
    import io

    raw = audio_struct["bytes"]
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if arr.ndim == 2:  # mono merge
        arr = arr.mean(axis=1)
    return arr, sr


def materialize_subset(
    out_dir: str,
    n_samples: int = 1024,
    split: str = "train",
    max_duration: float = 20.0,
    min_duration: float = 0.1,
    shard_offset: int = 0,
    workers: int = 8,
    seed: Optional[int] = None,
    normalize_for_charwise: bool = False,
) -> str:
    """Stream ``n_samples`` rows from the dataset and write a TSV manifest.

    Parameters
    ----------
    out_dir : str
        Output directory; audio goes to ``<out_dir>/audio/``, manifest to
        ``<out_dir>/manifest_<split>.tsv``.
    n_samples : int
        Number of samples to keep (after duration filtering).
    split : str
        Dataset split (``train`` / ``test`` / ...).
    max_duration, min_duration : float
        Duration filters in seconds (matches AudioDataset semantics).
    shard_offset : int
        Start reading from this parquet shard (34 shards exist). Useful to avoid
        re-reading shard 0 for a second (validation) subset.
    workers : int
        Parallel audio-writing workers.
    seed : int, optional
        If given, shuffle rows deterministically before sampling; otherwise rows
        are taken in dataset order (cheap, reproducible).

    Returns
    -------
    str
        Path to the written manifest TSV.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    out = Path(out_dir)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = [
        f
        for f in api.list_repo_files(DATASET_ID, repo_type="dataset")
        if f.startswith(f"data/{split}-") and f.endswith(".parquet")
    ]
    files.sort()
    files = files[shard_offset:] or files

    if seed is not None:
        import random

        rng = random.Random(seed)

    rows = []  # (text, duration, audio_struct)
    for fn in files:
        if len(rows) >= n_samples * 4:  # over-fetch to survive filtering
            break
        local = api.hf_hub_download(DATASET_ID, filename=fn, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(
            batch_size=1024,
            columns=["text", "duration", "audio"],
        ):
            d = batch.to_pylist()
            for r in d:
                dur = float(r["duration"])
                if dur < min_duration or dur > max_duration:
                    continue
                rows.append((r["text"].strip(), dur, r["audio"]))
                if len(rows) >= n_samples * 4:
                    break
            if len(rows) >= n_samples * 4:
                break

    if seed is not None:
        rng.shuffle(rows)
    rows = rows[: n_samples * 3]  # a little slack before final trim

    def _normalize_charwise(text: str) -> str:
        """Lowercase, drop ё→е, collapse whitespace, keep only Russian letters/space.

        Matches train_utils/utils.normalize_raw_text so the non-e2e (charwise vocab)
        models get references in their vocabulary.
        """
        text = text.replace("ё", "е").replace("Ё", "Е")
        text = " ".join(text.split())
        return "".join(c for c in text.lower() if ("а" <= c <= "я") or c == " ")

    def _write(i: int) -> "Optional[str]":
        text, dur, audio_struct = rows[i]
        try:
            arr, sr = _decode_audio_bytes(audio_struct)
        except Exception:
            return None
        if arr.shape[0] / sr < min_duration or arr.shape[0] / sr > max_duration:
            return None
        rel = f"audio/{i:07d}.wav"
        sf.write(str(audio_dir / f"{i:07d}.wav"), arr, sr, subtype="PCM_16")
        out_text = _normalize_charwise(text) if normalize_for_charwise else text
        return f"{rel}\t{dur:.3f}\t{out_text}"

    lines: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(rows)))) as ex:
        for ln in tqdm(
            ex.map(_write, range(len(rows))),
            total=len(rows),
            desc="Writing audio",
        ):
            if ln is not None:
                lines.append(ln)
            if len(lines) >= n_samples:
                break

    manifest = out / f"manifest_{split}.tsv"
    manifest.write_text(
        "path\tduration\ttranscription\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(lines)} samples -> {manifest}")
    return str(manifest)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Materialize a Golos-punctuation subset to TSV."
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples", type=int, default=1024)
    p.add_argument("--split", default="train")
    p.add_argument("--max_duration", type=float, default=20.0)
    p.add_argument("--min_duration", type=float, default=0.1)
    p.add_argument("--shard_offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--normalize_for_charwise",
        action="store_true",
        help="Normalize text for non-e2e (charwise vocab) models: lowercase, "
        "drop punctuation, keep only Russian letters/space.",
    )
    args = p.parse_args()
    materialize_subset(
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        split=args.split,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        shard_offset=args.shard_offset,
        workers=args.workers,
        seed=args.seed,
        normalize_for_charwise=args.normalize_for_charwise,
    )
