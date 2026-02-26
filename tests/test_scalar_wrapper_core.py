"""Core behavior tests for ``MumPyScalar``."""

import numpy as np
import pytest

import mumpy as mp
from mumpy._wrapping import MumPyScalar


def test_scalar_wrapper_from_reduction_has_dtype_and_item() -> None:
    s = mp.sum(mp.array([1, 2, 3]))

    assert isinstance(s, mp.MumPyScalar)
    assert str(s.dtype).removeprefix("mlx.core.") == "int64"
    assert s.item() == 6


def test_scalar_wrapper_numeric_conversions_and_hash() -> None:
    s = MumPyScalar(3)
    f = MumPyScalar(2.5)

    assert int(s) == 3
    assert float(f) == 2.5
    assert bool(MumPyScalar(1)) is True
    assert bool(MumPyScalar(0)) is False
    assert hash(s) == hash(3)


def test_scalar_wrapper_supports_index_for_integer_dtypes() -> None:
    s = MumPyScalar(mp.array(2, dtype=mp.int32))

    assert s.__index__() == 2
    assert [10, 20, 30][s] == 30


def test_scalar_wrapper_index_rejects_non_integer_dtypes() -> None:
    s_float = MumPyScalar(mp.array(2.0, dtype=mp.float32))
    with pytest.raises(TypeError, match="cannot be interpreted as an integer index"):
        _ = s_float.__index__()

    bool_value = True
    s_bool = MumPyScalar(mp.array(bool_value, dtype=mp.bool_))
    with pytest.raises(TypeError, match="cannot be interpreted as an integer index"):
        _ = s_bool.__index__()


def test_scalar_wrapper_rejects_non_scalar_array() -> None:
    with pytest.raises(TypeError, match="scalar"):
        MumPyScalar(mp.array([1, 2, 3]))


def test_scalar_wrapper_repr_is_informative() -> None:
    s = MumPyScalar(mp.array(5, dtype=mp.int32))

    assert "MumPyScalar(" in repr(s)
    assert "int32" in repr(s)


def test_scalar_wrapper_view_matches_numpy_semantics() -> None:
    s = mp.sum(mp.array([1.0, 2.0, 3.0], dtype=mp.float64))

    same_view = s.view()
    assert isinstance(same_view, mp.MumPyScalar)
    assert same_view is not s
    assert same_view.mx is s.mx
    assert same_view.dtype == s.dtype
    assert same_view.item() == s.item()

    int64_view = s.view(mp.int64)
    expected = np.asarray(s).view(np.int64)
    assert isinstance(int64_view, mp.MumPyScalar)
    assert int64_view.dtype == mp.int64
    assert int64_view.item() == expected.item()
