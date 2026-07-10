from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OpenSetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    known_classes: tuple[int, ...]
    unknown_classes: tuple[int, ...]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            train=self.train,
            validation=self.validation,
            test=self.test,
            known_classes=np.asarray(self.known_classes),
            unknown_classes=np.asarray(self.unknown_classes),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OpenSetSplit":
        payload = np.load(path)
        return cls(
            payload["train"],
            payload["validation"],
            payload["test"],
            tuple(int(value) for value in payload["known_classes"]),
            tuple(int(value) for value in payload["unknown_classes"]),
        )


def build_open_set_split(
    labels: Sequence[int] | np.ndarray,
    known_classes: Sequence[int],
    unknown_classes: Sequence[int],
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction_of_development: float = 0.1,
) -> OpenSetSplit:
    labels_array = np.asarray(labels, dtype=np.int64)
    known = tuple(sorted(int(value) for value in known_classes))
    unknown = tuple(sorted(int(value) for value in unknown_classes))
    if set(known) & set(unknown):
        raise ValueError("Known and unknown class sets overlap")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in known + unknown:
        indices = np.flatnonzero(labels_array == label)
        if len(indices) == 0:
            raise ValueError(f"Class {label} has no processed samples")
        indices = rng.permutation(indices)
        test_count = max(1, int(round(len(indices) * test_fraction)))
        class_test = indices[:test_count]
        test.extend(class_test.tolist())
        if label in unknown:
            continue
        development = indices[test_count:]
        validation_count = max(1, int(round(len(development) * validation_fraction_of_development)))
        validation.extend(development[:validation_count].tolist())
        train.extend(development[validation_count:].tolist())
    return OpenSetSplit(
        train=np.asarray(sorted(train), dtype=np.int64),
        validation=np.asarray(sorted(validation), dtype=np.int64),
        test=np.asarray(sorted(test), dtype=np.int64),
        known_classes=known,
        unknown_classes=unknown,
    )

