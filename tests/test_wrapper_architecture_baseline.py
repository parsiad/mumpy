"""Wrapper architecture behavior checks for method/function parity and raw escape hatches."""

import mlx.core as mx

import mumpy as mp

from .conftest import assert_allclose


def test_wrapper_method_and_function_reductions_match() -> None:
    x = mp.random.randn(4096)

    got_func = mp.mean(x)
    got_method = x.mean()

    assert isinstance(got_func, mp.MumPyScalar)
    assert isinstance(got_method, mp.MumPyScalar)
    assert_allclose(got_func, got_method, rtol=0, atol=0)


def test_wrapper_method_and_function_astype_match() -> None:
    x = mp.random.randn(1024)

    got_func = mp.astype(x, mp.float32)
    got_method = x.astype(mp.float32)

    assert isinstance(got_func, mp.MumPyArray)
    assert isinstance(got_method, mp.MumPyArray)
    assert str(got_func.dtype).removeprefix("mlx.core.") == "float32"
    assert str(got_method.dtype).removeprefix("mlx.core.") == "float32"
    assert_allclose(got_func, got_method, rtol=0, atol=0)


def test_wrapper_raw_escape_hatch_exposes_mlx_array() -> None:
    x = mp.array([1, 2, 3])

    assert isinstance(x.mx, mx.array)
    assert not isinstance(x.mx, mp.MumPyArray)
