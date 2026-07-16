import numpy as np

from uti_mpc.data.splits import build_grouped_open_set_split, build_open_set_split


def test_open_set_split_is_deterministic_and_leak_free():
    labels = np.repeat(np.arange(1, 5), 50)
    first = build_open_set_split(labels, [1, 2], [3, 4], seed=42)
    second = build_open_set_split(labels, [1, 2], [3, 4], seed=42)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.test, second.test)
    assert set(labels[first.train]) == {1, 2}
    assert set(labels[first.validation]) == {1, 2}
    assert set(labels[first.test]) == {1, 2, 3, 4}
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)


def test_capture_grouped_split_is_disjoint_and_uses_two_capture_fallback():
    labels = np.asarray([1] * 8 + [2] * 4 + [3] * 4)
    groups = np.asarray(
        ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2
        + ["rare_train"] * 2 + ["rare_test"] * 2
        + ["unknown_a"] * 2 + ["unknown_b"] * 2
    )
    split = build_grouped_open_set_split(
        labels,
        groups,
        known_classes=[1, 2],
        unknown_classes=[3],
        seed=42,
        cache_fingerprint="abc",
    )
    split.validate_groups()
    assert split.cache_fingerprint == "abc"
    assert not set(split.train_groups) & set(split.test_groups)
    assert not set(split.validation_groups) & set(split.test_groups)
    assert not ({"rare_train", "rare_test"} & set(split.validation_groups))
    assert {"unknown_a", "unknown_b"}.issubset(split.test_groups)
    assert set(labels[split.train]) == {1, 2}
