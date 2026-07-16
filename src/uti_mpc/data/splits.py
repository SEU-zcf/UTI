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
    train_groups: tuple[str, ...] = ()
    validation_groups: tuple[str, ...] = ()
    test_groups: tuple[str, ...] = ()
    cache_fingerprint: str = ""

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
            train_groups=np.asarray(self.train_groups, dtype=np.str_),
            validation_groups=np.asarray(self.validation_groups, dtype=np.str_),
            test_groups=np.asarray(self.test_groups, dtype=np.str_),
            cache_fingerprint=np.asarray(self.cache_fingerprint),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OpenSetSplit":
        payload = np.load(path)
        files = set(payload.files)
        return cls(
            payload["train"],
            payload["validation"],
            payload["test"],
            tuple(int(value) for value in payload["known_classes"]),
            tuple(int(value) for value in payload["unknown_classes"]),
            tuple(str(value) for value in payload["train_groups"])
            if "train_groups" in files
            else (),
            tuple(str(value) for value in payload["validation_groups"])
            if "validation_groups" in files
            else (),
            tuple(str(value) for value in payload["test_groups"])
            if "test_groups" in files
            else (),
            str(payload["cache_fingerprint"].item())
            if "cache_fingerprint" in files
            else "",
        )

    def validate_groups(self) -> None:
        train = set(self.train_groups)
        validation = set(self.validation_groups)
        test = set(self.test_groups)
        if train & validation or train & test or validation & test:
            raise ValueError("Capture-grouped split contains overlapping captures")


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


def build_grouped_open_set_split(
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    known_classes: Sequence[int],
    unknown_classes: Sequence[int],
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction_of_development: float = 0.1,
    cache_fingerprint: str = "",
) -> OpenSetSplit:
    """Split complete captures while keeping every unknown capture test-only.

    Known classes with exactly two captures use the documented rare-class rule:
    one capture trains and one tests, with no validation capture for that class.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    groups_array = np.asarray(groups, dtype=np.str_)
    if len(labels_array) != len(groups_array):
        raise ValueError("labels and groups must have the same length")
    known = tuple(sorted(int(value) for value in known_classes))
    unknown = tuple(sorted(int(value) for value in unknown_classes))
    if set(known) & set(unknown):
        raise ValueError("Known and unknown class sets overlap")
    rng = np.random.default_rng(seed)
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    test_groups: set[str] = set()
    for label in known + unknown:
        selected = labels_array == label
        class_groups = np.unique(groups_array[selected])
        if len(class_groups) == 0:
            raise ValueError(f"Class {label} has no processed captures")
        shuffled = class_groups[rng.permutation(len(class_groups))]
        if label in unknown:
            test_groups.update(str(value) for value in shuffled)
            continue
        if len(shuffled) < 2:
            raise ValueError(
                f"Known class {label} needs at least two captures for grouped testing"
            )
        if len(shuffled) == 2:
            test_groups.add(str(shuffled[0]))
            train_groups.add(str(shuffled[1]))
            continue
        test_count = max(1, int(round(len(shuffled) * test_fraction)))
        test_count = min(test_count, len(shuffled) - 2)
        development = shuffled[test_count:]
        validation_count = max(
            1, int(round(len(development) * validation_fraction_of_development))
        )
        validation_count = min(validation_count, len(development) - 1)
        test_groups.update(str(value) for value in shuffled[:test_count])
        validation_groups.update(
            str(value) for value in development[:validation_count]
        )
        train_groups.update(str(value) for value in development[validation_count:])

    train = np.flatnonzero(np.isin(groups_array, list(train_groups)))
    validation = np.flatnonzero(np.isin(groups_array, list(validation_groups)))
    test = np.flatnonzero(np.isin(groups_array, list(test_groups)))
    split = OpenSetSplit(
        train=np.asarray(train, dtype=np.int64),
        validation=np.asarray(validation, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        known_classes=known,
        unknown_classes=unknown,
        train_groups=tuple(sorted(train_groups)),
        validation_groups=tuple(sorted(validation_groups)),
        test_groups=tuple(sorted(test_groups)),
        cache_fingerprint=cache_fingerprint,
    )
    split.validate_groups()
    if set(labels_array[split.train]) - set(known):
        raise RuntimeError("Grouped split leaked unknown classes into training")
    return split
