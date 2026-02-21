"""Core behavior tests for ``MumPyArray``."""

import pytest

import mumpy as mp

from .conftest import assert_array_equal


def test_array_returns_wrapper_and_ndarray_alias() -> None:
    x = mp.array([1, 2, 3])

    assert type(x) is mp.MumPyArray
    assert isinstance(x, mp.ndarray)
    assert mp.ndarray is mp.MumPyArray


def test_array_wrapper_basic_properties_and_indexing() -> None:
    x = mp.arange(6).reshape(2, 3)

    assert x.shape == (2, 3)
    assert x.ndim == 2
    assert x.size == 6
    assert x.itemsize > 0
    assert x.nbytes >= x.size * x.itemsize

    scalar = x[0, 1]
    sub = x[:, 1:]
    assert isinstance(scalar, mp.MumPyScalar)
    assert isinstance(sub, mp.MumPyArray)
    assert scalar.item() == 1
    assert_array_equal(sub, [[1, 2], [4, 5]])


def test_array_wrapper_iter_yields_wrapped_values() -> None:
    x = mp.arange(6).reshape(2, 3)
    rows = list(iter(x))

    assert len(rows) == 2
    assert all(isinstance(row, mp.MumPyArray) for row in rows)
    assert_array_equal(rows[0], [0, 1, 2])
    assert_array_equal(rows[1], [3, 4, 5])


def test_array_wrapper_truthiness_matches_numpy_style() -> None:
    with pytest.raises(ValueError, match="truth value"):
        bool(mp.array([1, 2]))

    assert bool(mp.array([1])) is True
    assert bool(mp.array([0])) is False


def test_array_wrapper_at_indexer_wraps_results() -> None:
    x = mp.array([1, 2, 3], dtype=mp.float32)
    y = x.at[1].add(7.0)

    assert isinstance(y, mp.MumPyArray)
    assert_array_equal(y, [1.0, 9.0, 3.0])


def test_array_wrapper_repr_str_and_tolist() -> None:
    x = mp.array([[1, 2], [3, 4]])

    assert "array(" in repr(x)
    assert "array(" in str(x)
    assert x.tolist() == [[1, 2], [3, 4]]
