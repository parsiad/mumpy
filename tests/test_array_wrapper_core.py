"""Core behavior tests for ``MumPyArray``."""

import operator
from typing import cast

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


def test_array_wrapper_iter_yields_scalars_for_1d_arrays() -> None:
    x = mp.arange(5)
    values = list(iter(x))

    assert len(values) == 5
    assert all(isinstance(v, mp.MumPyScalar) for v in values)
    assert [v.item() for v in values] == [0, 1, 2, 3, 4]


def test_permutation_iteration_yields_scalars() -> None:
    x = mp.random.permutation(mp.arange(10))
    values = list(iter(x))

    assert len(values) == 10
    assert all(isinstance(v, mp.MumPyScalar) for v in values)
    assert sorted(int(v.item()) for v in values) == list(range(10))


def test_float64_wrapper_indexing_and_iteration_route_when_needed() -> None:
    x = cast("mp.MumPyArray", mp.arange(12, dtype=mp.float64).reshape(3, 4))

    scalar = x[0, 1]
    assert isinstance(scalar, mp.MumPyScalar)
    assert scalar.item() == 1.0

    row = x[1]
    assert isinstance(row, mp.MumPyArray)
    assert row.dtype == mp.float64
    assert_array_equal(row, [4.0, 5.0, 6.0, 7.0])

    fancy = x[[0, 2], [1, 3]]
    assert isinstance(fancy, mp.MumPyArray)
    assert fancy.dtype == mp.float64
    assert_array_equal(fancy, [1.0, 11.0])

    rows = list(iter(x))
    assert len(rows) == 3
    assert all(isinstance(v, mp.MumPyArray) for v in rows)
    assert_array_equal(rows[0], [0.0, 1.0, 2.0, 3.0])


def test_array_wrapper_truthiness_matches_numpy_style() -> None:
    with pytest.raises(ValueError, match="truth value"):
        bool(mp.array([1, 2]))

    assert bool(mp.array([1])) is True
    assert bool(mp.array([0])) is False


def test_array_wrapper_inplace_ops_raise() -> None:
    x = mp.array([1.0, 2.0], dtype=mp.float64)
    ints = mp.array([1, 2], dtype=mp.int64)
    mat = mp.array([[1.0, 2.0], [3.0, 4.0]], dtype=mp.float64)
    cases = (
        (operator.iadd, x, 1.0),
        (operator.isub, x, 1.0),
        (operator.imul, x, 2.0),
        (operator.itruediv, x, 2.0),
        (operator.ifloordiv, ints, 2),
        (operator.imod, ints, 2),
        (operator.ipow, x, 2.0),
        (operator.imatmul, mat, mat),
        (operator.iand, ints, 1),
        (operator.ior, ints, 1),
        (operator.ixor, ints, 1),
    )

    for op, target, operand in cases:
        with pytest.raises(TypeError, match="immutable"):
            op(target, operand)


def test_array_wrapper_at_indexer_wraps_results() -> None:
    x = mp.array([1, 2, 3], dtype=mp.float32)
    y = x.at[1].add(7.0)

    assert isinstance(y, mp.MumPyArray)
    assert_array_equal(y, [1.0, 9.0, 3.0])


def test_array_wrapper_setflags_supports_write_false_only() -> None:
    x = mp.array([1.0, 2.0], dtype=mp.float64)

    assert x.setflags(write=False) is None

    with pytest.raises(ValueError, match="writeable flag"):
        x.setflags(write=True)

    with pytest.raises(NotImplementedError, match="write=False"):
        x.setflags(align=True)


def test_array_wrapper_repr_str_and_tolist() -> None:
    x = mp.array([[1, 2], [3, 4]])

    assert "array(" in repr(x)
    assert "array(" in str(x)
    assert x.tolist() == [[1, 2], [3, 4]]
