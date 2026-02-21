"""Core behavior tests for ``MumPyScalar``."""

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


def test_scalar_wrapper_rejects_non_scalar_array() -> None:
    with pytest.raises(TypeError, match="scalar"):
        MumPyScalar(mp.array([1, 2, 3]))


def test_scalar_wrapper_repr_is_informative() -> None:
    s = MumPyScalar(mp.array(5, dtype=mp.int32))

    assert "MumPyScalar(" in repr(s)
    assert "int32" in repr(s)
