"""Representative compatibility checks for raw MLX array inputs."""

import mlx.core as mx

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_top_level_functions_accept_raw_mlx_arrays() -> None:
    raw = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)

    got_mean = mp.mean(raw)
    got_sum = mp.sum(raw)
    got_reshaped = mp.reshape(raw, (3, 1))

    assert isinstance(got_mean, mp.MumPyScalar)
    assert isinstance(got_sum, mp.MumPyScalar)
    assert isinstance(got_reshaped, mp.MumPyArray)
    assert_allclose(got_mean, 2.0, rtol=0, atol=0)
    assert_allclose(got_sum, 6.0, rtol=0, atol=0)
    assert_array_equal(got_reshaped, [[1.0], [2.0], [3.0]])


def test_submodule_functions_accept_raw_mlx_arrays() -> None:
    raw = mx.array([1.0, -2.0, 3.0], dtype=mx.float32)
    mat = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)

    got_norm = mp.linalg.norm(raw)
    got_fft = mp.fft.fft(raw)
    got_added = mp.add(mat, mat)

    assert isinstance(got_norm, mp.MumPyScalar)
    assert isinstance(got_fft, mp.MumPyArray)
    assert isinstance(got_added, mp.MumPyArray)
    assert_allclose(got_norm, mp.linalg.norm(mp.array([1.0, -2.0, 3.0], dtype=mp.float32)), rtol=1e-6, atol=1e-6)
    assert_allclose(got_fft, mp.fft.fft(mp.array([1.0, -2.0, 3.0], dtype=mp.float32)), rtol=1e-6, atol=1e-6)
    assert_array_equal(got_added, [[2.0, 4.0], [6.0, 8.0]])
