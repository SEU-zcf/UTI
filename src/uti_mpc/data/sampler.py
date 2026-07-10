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
    ) -> None:
        self.p = classes_per_batch
        self.q = samples_per_class
        self.seed = seed
        self.epoch = 0
        by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            by_class[int(label)].append(index)
        self.by_class = {key: np.asarray(value, dtype=np.int64) for key, value in by_class.items()}
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
                pool = self.by_class[int(label)]
                chosen = rng.choice(pool, size=self.q, replace=len(pool) < self.q)
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch

