"""NumPy protocol interop for MumPy wrappers."""

import numpy as np
import pytest

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_numpy_asarray_converts_wrapper_to_ndarray() -> None:
    x = mp.array([1, 2, 3], dtype=mp.int32)

    arr = np.asarray(x)
    assert isinstance(arr, np.ndarray)
    assert str(arr.dtype) == "int32"
    np.testing.assert_array_equal(arr, np.array([1, 2, 3], dtype=np.int32))


def test_numpy_ufuncs_return_wrapped_results() -> None:
    x = mp.array([1.0, 2.0, 3.0])
    out = np.add(x, 1.5)
    mixed = np.add(x, np.array([1.0, 1.0, 1.0]))

    assert isinstance(out, mp.MumPyArray)
    assert isinstance(mixed, mp.MumPyArray)
    assert_allclose(out, [2.5, 3.5, 4.5], rtol=1e-6, atol=1e-6)
    assert_allclose(mixed, [2.0, 3.0, 4.0], rtol=1e-6, atol=1e-6)


def test_numpy_array_function_paths_return_wrapped_results() -> None:
    x = mp.array([1.0, 2.0, 3.0, 4.0])

    got_sum = np.sum(x)
    got_mean = np.mean(x)
    got_reshaped = np.reshape(x, (2, 2))

    assert isinstance(got_sum, mp.MumPyScalar)
    assert isinstance(got_mean, mp.MumPyScalar)
    assert isinstance(got_reshaped, mp.MumPyArray)
    assert_allclose(got_sum, 10.0, rtol=0, atol=0)
    assert_allclose(got_mean, 2.5, rtol=0, atol=0)
    assert_array_equal(got_reshaped, [[1.0, 2.0], [3.0, 4.0]])


def test_numpy_ufuncs_allow_out_none_but_reject_real_out_buffers() -> None:
    x = mp.array([1.0, 2.0, 3.0])

    out_none = np.add(x, 1.5, out=None)

    assert isinstance(out_none, mp.MumPyArray)
    assert_allclose(out_none, [2.5, 3.5, 4.5], rtol=1e-6, atol=1e-6)

    with pytest.raises(TypeError):
        np.add(x, 1.5, out=np.empty(3, dtype=np.float64))
