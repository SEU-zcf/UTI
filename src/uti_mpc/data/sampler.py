from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """Sample P classes and Q examples per class with deterministic epoch seeding."""

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int,
        samples_per_class: int,
        seed: int = 42,
        batches_per_epoch: int | None = None,
        buckets: Sequence[int] | None = None,
        stratify_by_bucket: bool = False,
    ) -> None:
        self.p = classes_per_batch
        self.q = samples_per_class
        self.seed = seed
        self.epoch = 0
        if buckets is not None and len(buckets) != len(labels):
            raise ValueError("buckets must have the same length as labels")
        if stratify_by_bucket and buckets is None:
            raise ValueError("stratify_by_bucket requires bucket assignments")
        self.stratify_by_bucket = stratify_by_bucket
        by_class: dict[int, list[int]] = defaultdict(list)
        by_class_bucket: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, label in enumerate(labels):
            by_class[int(label)].append(index)
            if buckets is not None:
                by_class_bucket[int(label)][int(buckets[index])].append(index)
        self.by_class = {key: np.asarray(value, dtype=np.int64) for key, value in by_class.items()}
        self.by_class_bucket = {
            label: {bucket: np.asarray(indices, dtype=np.int64) for bucket, indices in by_bucket.items()}
            for label, by_bucket in by_class_bucket.items()
        }
        self.classes = np.asarray(sorted(self.by_class), dtype=np.int64)
        if len(self.classes) < self.p:
            raise ValueError(f"P={self.p} exceeds number of training classes={len(self.classes)}")
        natural = math.ceil(len(labels) / (self.p * self.q))
        self.batches_per_epoch = batches_per_epoch or max(1, natural)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            selected = rng.choice(self.classes, size=self.p, replace=False)
            batch: list[int] = []
            for label in selected:
                label_id = int(label)
                if not self.stratify_by_bucket:
                    pool = self.by_class[label_id]
                    chosen = rng.choice(pool, size=self.q, replace=len(pool) < self.q)
                    batch.extend(chosen.tolist())
                    continue
                pools = self.by_class_bucket[label_id]
                available_buckets = np.asarray(sorted(pools), dtype=np.int64)
                base, remainder = divmod(self.q, len(available_buckets))
                allocations = np.full(len(available_buckets), base, dtype=np.int64)
                if remainder:
                    allocations[rng.permutation(len(available_buckets))[:remainder]] += 1
                for bucket, amount in zip(available_buckets, allocations, strict=True):
                    if amount == 0:
                        continue
                    pool = pools[int(bucket)]
                    chosen = rng.choice(pool, size=int(amount), replace=len(pool) < amount)
                    batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch
