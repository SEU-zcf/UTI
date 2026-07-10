import numpy as np

from uti_mpc.data.splits import build_open_set_split


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

