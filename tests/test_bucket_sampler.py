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
