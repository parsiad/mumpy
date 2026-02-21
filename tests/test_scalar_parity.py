"""Parity and interop checks for ``MumPyScalar``."""

import numpy as np

import mumpy as mp

from .conftest import assert_allclose


def test_scalar_arithmetic_and_comparisons_behave_like_numpy_scalars() -> None:
    a = mp.sum(mp.array([1, 2, 3]))
    b = mp.sum(mp.array([4, 5]))

    got_add = a + b
    got_cmp = a < b

    assert isinstance(got_add, mp.MumPyScalar)
    assert isinstance(got_cmp, mp.MumPyScalar)
    assert got_add.item() == 15
    assert bool(got_cmp) is True


def test_scalar_numpy_conversion_and_mean_interop() -> None:
    s = mp.mean(mp.array([1, 2, 3], dtype=mp.int32))

    arr = np.asarray(s)
    assert arr.shape == ()
    assert str(arr.dtype) == "float64"
    assert_allclose(s, np.float64(2.0), rtol=0, atol=0)
