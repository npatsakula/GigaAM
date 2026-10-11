"""DurationBucketBatchSampler: similar-duration batches, full coverage, epoch reshuffle."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train_utils"))

from utils import DurationBucketBatchSampler  # noqa: E402


@pytest.mark.parametrize("n,batch", [(1000, 16), (37, 8)])
def test_bucket_sampler(n, batch):
    import random

    rng = random.Random(0)
    durations = [rng.uniform(0.3, 20.0) for _ in range(n)]
    sampler = DurationBucketBatchSampler(
        durations, batch_size=batch, bucket_mult=10, seed=1
    )
    batches = list(sampler)
    assert len(batches) == len(sampler) == n // batch
    flat = [i for b in batches for i in b]
    assert len(flat) == len(set(flat)) and set(flat) <= set(range(n))
    # every batch spans at most one bucket's duration range
    sorted_d = sorted(durations)
    bucket = batch * 10
    for b in batches:
        ds = sorted(durations[i] for i in b)
        lo = sorted_d.index(ds[0])
        hi = sorted_d.index(ds[-1])
        assert hi - lo < bucket * 2, "batch spans more than a bucket"
    # a batch of similar durations wastes little padding (needs several buckets)
    if n > batch * 10 * 3:
        waste = sum(
            max(durations[i] for i in b) * batch - sum(durations[i] for i in b)
            for b in batches
        )
        audio = sum(durations[i] for i in flat)
        assert waste / audio < 0.15
    # epochs reshuffle; the drop_last remainder rotates so every clip is seen
    seen = set(flat)
    sampler.set_epoch(1)
    batches2 = list(sampler)
    assert batches2 != batches and len(batches2) == len(batches)
    for epoch in range(1, 6):
        sampler.set_epoch(epoch)
        seen |= {i for b in sampler for i in b}
    assert seen == set(range(n))


def test_bucket_sampler_rank_partition(monkeypatch):
    sampler = DurationBucketBatchSampler(
        [float(i % 7) for i in range(640)], batch_size=8
    )
    full = list(sampler)
    monkeypatch.setattr(sampler, "_rank_world", lambda: (1, 4))
    part = list(sampler)
    assert len(part) == len(full) // 4
    assert part == full[: len(full) // 4 * 4][1::4]


def test_bucket_sampler_audio_budget():
    """With max_seconds, batches close on padded audio (clips x longest clip):
    short clips are batched by the hundreds, long ones in small batches, every
    batch within budget, little padding, full coverage."""
    import random

    rng = random.Random(0)
    # Golos-like: mostly 1-6 s, a tail to 20 s.
    durations = [min(20.0, 0.5 + rng.expovariate(1 / 3.5)) for _ in range(20000)]
    # bucket_mult=5 here: a bucket is ~5 batches (~900 clips), i.e. the same
    # ~1% slice of the duration distribution that the default 50 batches is on
    # the 970k-clip production set.
    sampler = DurationBucketBatchSampler(
        durations, batch_size=512, max_seconds=640, bucket_mult=5
    )
    batches = list(sampler)
    sizes = [len(b) for b in batches]
    padded = [len(b) * max(durations[i] for i in b) for b in batches]
    audio = [sum(durations[i] for i in b) for b in batches]
    assert max(padded) <= 640
    assert max(sizes) > 100 and min(sizes) >= 2
    flat = [i for b in batches for i in b]
    assert len(flat) == len(set(flat)) == len(durations)
    # padding waste stays small because buckets span a narrow duration range
    assert sum(padded) / sum(audio) < 1.15
    # most batches are close to the budget (constant work per step); each
    # bucket ends with one partial batch, i.e. ~1 in bucket_mult here
    full = sum(p > 0.8 * 640 for p in padded)
    assert full / len(batches) > 0.75
