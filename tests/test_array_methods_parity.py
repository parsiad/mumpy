"""Array method parity checks against top-level MumPy functions."""

from typing import cast

import numpy as np

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_array_methods_match_top_level_reductions_and_shape_ops() -> None:
    x = mp.arange(12, dtype=mp.float32).reshape(3, 4)

    assert isinstance(x.mean(), mp.MumPyScalar)
    assert_allclose(x.mean(), mp.mean(x), rtol=0, atol=0)
    assert_allclose(x.std(), mp.std(x), rtol=0, atol=0)
    assert_allclose(x.var(), mp.var(x), rtol=0, atol=0)
    assert_allclose(x.sum(), mp.sum(x), rtol=0, atol=0)

    assert_array_equal(x.reshape(2, 6), mp.reshape(x, (2, 6)))
    assert_array_equal(x.transpose(), mp.transpose(x))
    assert_array_equal(x.flatten(), mp.flatten(x))
    assert_array_equal(x.cumsum(), mp.cumsum(x))
    assert_array_equal(x.cumprod(), mp.cumprod(x))


def test_array_properties_transpose_real_imag_are_wrapped() -> None:
    z = mp.array([[1 + 2j, 3 - 4j], [5 + 0j, -1j]])

    assert isinstance(z.T, mp.MumPyArray)
    assert isinstance(z.real, mp.MumPyArray)
    assert isinstance(z.imag, mp.MumPyArray)
    assert_array_equal(z.T, mp.transpose(z))
    assert_array_equal(z.real, mp.real(z))
    assert_array_equal(z.imag, mp.imag(z))


def test_float64_shape_methods_route_when_needed() -> None:
    x = mp.random.randn(256, 1)

    squeezed = x.squeeze()
    assert isinstance(squeezed, mp.MumPyArray)
    assert squeezed.dtype == mp.float64
    assert squeezed.shape == (256,)
    assert_allclose(squeezed, np.asarray(x).squeeze(), rtol=0.0, atol=0.0)

    reshaped = x.reshape(1, 256)
    assert isinstance(reshaped, mp.MumPyArray)
    assert reshaped.dtype == mp.float64
    assert reshaped.shape == (1, 256)
    assert_allclose(reshaped, np.asarray(x).reshape(1, 256), rtol=0.0, atol=0.0)

    transposed = x.transpose()
    assert isinstance(transposed, mp.MumPyArray)
    assert transposed.dtype == mp.float64
    assert transposed.shape == (1, 256)
    assert_allclose(transposed, np.asarray(x).transpose(), rtol=0.0, atol=0.0)


def test_float64_reduction_methods_route_when_needed() -> None:
    x = cast("mp.MumPyArray", mp.random.randn(100, 10))

    maxed = x.max(axis=0)
    assert isinstance(maxed, mp.MumPyArray)
    assert maxed.dtype == mp.float64
    assert maxed.shape == (10,)
    assert_allclose(maxed, np.asarray(x).max(axis=0), rtol=0.0, atol=0.0)

    mined = x.min(axis=0)
    assert isinstance(mined, mp.MumPyArray)
    assert mined.dtype == mp.float64
    assert mined.shape == (10,)
    assert_allclose(mined, np.asarray(x).min(axis=0), rtol=0.0, atol=0.0)

    argmaxed = x.argmax(axis=0)
    assert isinstance(argmaxed, mp.MumPyArray)
    assert argmaxed.shape == (10,)
    assert_array_equal(argmaxed, np.asarray(x).argmax(axis=0))


def test_view_matches_numpy_semantics() -> None:
    x = cast("mp.MumPyArray", mp.random.randn(32))

    same_view = x.view()
    assert isinstance(same_view, mp.MumPyArray)
    assert same_view is not x
    assert same_view.mx is x.mx
    assert same_view.dtype == x.dtype
    assert same_view.shape == x.shape
    assert_array_equal(same_view, np.asarray(x))

    float32_view = x.view(mp.float32)
    expected = np.asarray(x).view(np.float32)
    assert isinstance(float32_view, mp.MumPyArray)
    assert float32_view.dtype == mp.float32
    assert float32_view.shape == expected.shape
    assert_array_equal(float32_view, expected)


def test_take_method_matches_top_level_take() -> None:
    x = cast("mp.MumPyArray", mp.arange(12, dtype=mp.float64).reshape(3, 4))

    taken_rows = x.take([0, 2], axis=0)
    assert isinstance(taken_rows, mp.MumPyArray)
    assert_array_equal(taken_rows, mp.take(x, indices=[0, 2], axis=0))

    taken_flat = x.take([1, 5, 9])
    assert isinstance(taken_flat, mp.MumPyArray)
    assert_array_equal(taken_flat, mp.take(x, indices=[1, 5, 9]))
