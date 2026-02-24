"""Operator behavior for ``MumPyArray`` and ``MumPyScalar``."""

import mlx.core as mx
import numpy as np

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_array_binary_operators_with_mixed_operands() -> None:
    x = mp.array([1, 2, 3], dtype=mp.int32)
    y = mp.array([4, 5, 6], dtype=mp.int32)
    raw = mx.array([10, 20, 30], dtype=mx.int32)
    npy = np.array([100, 200, 300], dtype=np.int32)

    assert isinstance(x + y, mp.MumPyArray)
    assert_array_equal(x + y, [5, 7, 9])
    assert_array_equal(x + raw, [11, 22, 33])
    assert_array_equal(raw + x, [11, 22, 33])
    assert_array_equal(x + npy, [101, 202, 303])
    assert_array_equal(npy + x, [101, 202, 303])


def test_array_unary_and_comparison_operators() -> None:
    x = mp.array([-1, 2, -3], dtype=mp.int32)

    assert_array_equal(abs(x), [1, 2, 3])
    assert_array_equal(-x, [1, -2, 3])
    eq = x == mp.array([-1, 0, -3], dtype=mp.int32)
    assert isinstance(eq, mp.MumPyArray)
    assert_array_equal(eq, [True, False, True])


def test_matrix_multiply_and_scalar_ops() -> None:
    a = mp.array([[1.0, 2.0], [3.0, 4.0]])
    b = mp.array([[2.0, 0.0], [1.0, 2.0]])
    assert_allclose(a @ b, np.array([[4.0, 4.0], [10.0, 8.0]]), rtol=1e-6, atol=1e-6)

    s = mp.sum(mp.array([1, 2, 3]))
    out = s + 5
    assert isinstance(out, mp.MumPyScalar)
    assert out.item() == 11


def test_float64_wrapper_elementwise_ops_route_when_needed() -> None:
    x = mp.random.randn(128)

    prod = 0.0 * x
    assert isinstance(prod, mp.MumPyArray)
    assert prod.dtype == mp.float64
    assert_allclose(prod, np.zeros(128, dtype=np.float64), rtol=0.0, atol=0.0)

    eq = x == mp.copy(x)
    assert isinstance(eq, mp.MumPyArray)
    assert_array_equal(eq, np.ones(128, dtype=bool))

    sinx = mp.sin(x)
    assert isinstance(sinx, mp.MumPyArray)
    assert sinx.dtype == mp.float64
    assert_allclose(sinx, np.sin(np.asarray(x)), rtol=1e-6, atol=1e-6)

    clipped_low = mp.maximum(x, 0.0)
    assert isinstance(clipped_low, mp.MumPyArray)
    assert clipped_low.dtype == mp.float64
    assert_allclose(clipped_low, np.maximum(np.asarray(x), 0.0), rtol=0.0, atol=0.0)
