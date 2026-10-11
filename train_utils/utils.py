import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import soundfile as sf
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from tqdm.auto import tqdm

from gigaam.preprocess import SAMPLE_RATE


def normalize_raw_text(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = " ".join(text.split())
    return "".join(
        c for c in text.lower() if ord("а") <= ord(c) <= ord("я") or c == " "
    )


def compute_wer(preds: List[Dict[str, Any]]) -> Tuple[float, float, int, int, int, int]:
    """
    Word error rates for prediction rows with ``text`` (reference) and ``pred_text`` (hypothesis).

    Returns (wer_e2e_pct, wer_raw_pct, e2e_err, e2e_words, raw_err, raw_words).
    """
    e2e_err = e2e_w = raw_err = raw_w = 0
    for r in preds:
        ref_s, hyp_s = r["text"].strip(), r["pred_text"].strip()
        rw, hw = ref_s.split(), hyp_s.split()
        e2e_err += editdistance.eval(rw, hw)
        e2e_w += len(rw)
        nr, nh = normalize_raw_text(ref_s), normalize_raw_text(hyp_s)
        rr, hh = nr.split(), nh.split()
        raw_err += editdistance.eval(rr, hh)
        raw_w += len(rr)
    return (
        e2e_err / max(e2e_w, 1) * 100,
        raw_err / max(raw_w, 1) * 100,
        e2e_err,
        e2e_w,
        raw_err,
        raw_w,
    )


def save_split(ds, split: str, out_dir: str, max_dur: float, workers: int) -> List[str]:
    audio_dir = Path(out_dir) / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    w = max(1, min(workers, n))

    def process_one(i: int) -> Optional[str]:
        sample = ds[i]
        text = sample["text"].strip()
        arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        if len(arr) / sr > max_dur:
            return None
        rel_p = f"audio/{split}/{i:06d}.wav"
        p = audio_dir / f"{i:06d}.wav"
        if not p.exists():
            sf.write(str(p), arr, sr)
        return f"{rel_p}\t{len(arr) / sr:.3f}\t{text}"

    with ThreadPoolExecutor(max_workers=w) as ex:
        lines = list(
            tqdm(
                ex.map(process_one, range(n)),
                total=n,
                desc=f"{split} ({n})",
            )
        )
    return [ln for ln in lines if ln is not None]


def load_tonebooks(out_dir: str, max_duration: float = 30.0, workers: int = 8) -> Path:
    """
    Load ToneBooks dataset and save manifests to the output directory.
    Creates train / val .tsv files with the following columns: path, duration, transcription.
    """

    from datasets import Audio, load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Vikhrmodels/ToneBooks...")
    ds = load_dataset("Vikhrmodels/ToneBooks")
    train, val = ds["train"], ds.get("validation") or ds.get("test")
    train = train.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    val = val.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    print(f"Splits: {list(ds.keys())}, train={len(train)}, val={len(val)}")

    for fname, rows in (
        (
            "manifest_train.tsv",
            save_split(train, "train", out_dir, max_duration, workers),
        ),
        ("manifest_val.tsv", save_split(val, "val", out_dir, max_duration, workers)),
    ):
        path = out / fname
        path.write_text(
            "path\tduration\ttranscription\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        print(f"  {path} ({len(rows)} samples)")

    print(f"\nDone! Manifests at {out}")
    return out


class StepProgressBar(TQDMProgressBar):

    def __init__(self, steps_per_epoch: Optional[int] = None):
        super().__init__()
        self._steps_per_epoch = steps_per_epoch

    def on_train_epoch_start(self, trainer: Any, pl_module: Any):
        super().on_train_epoch_start(trainer, pl_module)
        acc = trainer.accumulate_grad_batches
        if acc <= 1:
            return
        if self._steps_per_epoch:
            remaining = (
                trainer.max_steps - trainer.global_step
                if trainer.max_steps > 0
                else self._steps_per_epoch
            )
            total = min(self._steps_per_epoch, remaining)
        else:
            total = (self.train_progress_bar.total or 0) // acc
        self.train_progress_bar.reset(total=total)

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ):
        acc = trainer.accumulate_grad_batches
        if acc <= 1:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            return
        if (batch_idx + 1) % acc == 0:
            self.train_progress_bar.n = (batch_idx + 1) // acc
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.train_progress_bar.refresh()


class EpochTimeLogger(Callback):
    def on_train_epoch_start(self, trainer: Any, pl_module: Any):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any):
        duration = time.time() - self.start_time
        if trainer.is_global_zero:
            print(f"[epoch {trainer.current_epoch}] time: {duration:.2f} sec")


def _fmt_float(v: float) -> str:
    return f"{v:g}".replace("+0", "+").replace("-0", "-")


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-") or "exp"


def build_exp_name(args) -> str:
    """
    Create a unique experiment name based on the command line arguments.
    Ignores arguments which do not affect training dynamics.
    """
    if args.exp_name:
        return _sanitize_name(args.exp_name)
    parts = [f"{args.model_name.replace('_', '')}"]
    parts += [f"lr{_fmt_float(args.lr)}", f"wd{_fmt_float(args.weight_decay)}"]
    parts.append(f"b{args.batch_size}")
    if args.accumulate_grad_batches > 1:
        parts.append(f"agb{args.accumulate_grad_batches}")
    if args.devices > 1:
        parts.append(f"{args.devices}gpu")
    if args.max_steps is not None:
        parts.append(f"{args.max_steps}steps")
        parts.append(f"vcs{args.val_check_steps}")
    else:
        parts.append(f"{args.max_epochs}ep")
        if args.val_check_interval != 1.0:
            parts.append(f"vci{_fmt_float(args.val_check_interval)}")
    if args.warmup_ratio != 0.1:
        parts.append(f"wmp{_fmt_float(args.warmup_ratio)}")
    if args.freeze_encoder:
        parts.append("frenc")
    if args.activation_checkpointing:
        parts.append("acckpt")
    if args.val_first_batches is not None:
        parts.append(f"vfb{args.val_first_batches}")
    if args.raw_text:
        parts.append("raw")
    parts.append(f"dur{_fmt_float(args.min_duration)}-{_fmt_float(args.max_duration)}s")
    if args.gradient_clip_val != 1.0:
        parts.append(f"gc{_fmt_float(args.gradient_clip_val)}")
    if args.precision != "16":
        parts.append(f"pr-{str(args.precision).replace('-', '')}")
    if args.seed != 42:
        parts.append(f"seed{args.seed}")
    if args.disable_spec_augment:
        parts.append("nospecaug")
    if not args.disable_spec_augment:
        if args.freq_masks != 2:
            parts.append(f"fm{args.freq_masks}")
        if args.freq_width != 27:
            parts.append(f"fw{args.freq_width}")
        if args.time_masks != 2:
            parts.append(f"tm{args.time_masks}")
        if args.time_width != 20:
            parts.append(f"tw{args.time_width}")
    return _sanitize_name("_".join(parts))


def prepare_experiment_dirs(output_dir: str, exp_name: str) -> Tuple[str, str]:
    model_dir = os.path.join(output_dir, "models", exp_name)
    tb_dir = os.path.join(output_dir, "tb_logs")
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        warnings.warn(
            f"Checkpoint dir is not empty: {model_dir}. Checkpoints may be overwritten.",
            stacklevel=1,
        )
    os.makedirs(model_dir, exist_ok=True)
    return model_dir, tb_dir


class DurationBucketBatchSampler:
    """
    Batches of similar-duration clips, so padded batches carry little silence.

    Each epoch: sort clips by duration, cut the sorted order into buckets of
    ``batch_size * bucket_mult`` clips, shuffle inside every bucket, cut buckets
    into batches and shuffle the batches. Clips in a batch then differ by at
    most a bucket's duration spread while the epoch order stays random. Pass as
    ``DataLoader(batch_sampler=...)``; Lightning calls :meth:`set_epoch`.

    Under DDP (``torch.distributed`` initialized) every rank iterates its own
    strided slice of the batch list, so ``Trainer(use_distributed_sampler=False)``
    must be set; batches are truncated to a common count per rank.
    """

    def __init__(
        self,
        durations: List[float],
        batch_size: int,
        bucket_mult: int = 50,
        seed: int = 0,
        drop_last: bool = True,
        max_seconds: Optional[float] = None,
    ):
        """``max_seconds`` closes a batch when its padded audio
        (``len(batch) * longest clip``) would exceed the budget, so every step
        carries about the same work and short clips are batched by the hundreds;
        ``batch_size`` is then the cap on clips per batch. ``None`` keeps
        fixed-size batches."""
        self.durations = list(durations)
        self.batch_size = batch_size
        self.bucket_mult = max(1, bucket_mult)
        self.seed = seed
        self.drop_last = drop_last
        self.max_seconds = max_seconds
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rank_world(self) -> Tuple[int, int]:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _batches(self) -> List[List[int]]:
        import random

        rng = random.Random(self.seed + self.epoch)
        order = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        batches: List[List[int]] = []
        if self.max_seconds is None:
            bucket = self.batch_size * self.bucket_mult
            buckets = [order[s : s + bucket] for s in range(0, len(order), bucket)]
        else:
            # Buckets of ~bucket_mult batches' worth of audio, so a bucket spans a
            # narrow duration range whatever the clip cap is.
            budget = self.bucket_mult * self.max_seconds
            buckets, cur, acc = [], [], 0.0
            for i in order:
                cur.append(i)
                acc += self.durations[i]
                if acc >= budget:
                    buckets.append(cur)
                    cur, acc = [], 0.0
            if cur:
                buckets.append(cur)
        for idx in buckets:
            rng.shuffle(idx)
            if self.max_seconds is None:
                for b in range(0, len(idx), self.batch_size):
                    batch = idx[b : b + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch)
                continue
            cur: List[int] = []
            cur_max = 0.0
            for i in idx:
                d = self.durations[i]
                padded = (len(cur) + 1) * max(cur_max, d)
                if cur and (padded > self.max_seconds or len(cur) >= self.batch_size):
                    batches.append(cur)
                    cur, cur_max = [], 0.0
                cur.append(i)
                cur_max = max(cur_max, d)
            if cur and (not self.drop_last or len(cur) > 1):
                batches.append(cur)
        rng.shuffle(batches)
        rank, world = self._rank_world()
        if world > 1:
            n = len(batches) // world * world
            batches = batches[:n][rank::world]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
