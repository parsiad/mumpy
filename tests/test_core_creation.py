"""Tests for array creation and basic shape conversion helpers."""

import numpy as np
import pytest

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_array_and_asarray_basic_behavior():
    arr = mp.array([1, 2, 3], dtype="float32", ndmin=2)
    assert arr.shape == (1, 3)
    assert str(arr.dtype) == "mlx.core.float32"
    assert_array_equal(arr, np.array([[1, 2, 3]], dtype=np.float32))

    same = mp.asarray(arr)
    assert same.shape == arr.shape
    assert_array_equal(same, arr)

    from_numpy = mp.asanyarray(np.array([4, 5, 6], dtype=np.int16))
    assert_array_equal(from_numpy, np.array([4, 5, 6], dtype=np.int16))

    contig = mp.ascontiguousarray([7, 8])
    assert_array_equal(contig, [7, 8])


def test_dtype_parser_aliases_and_array_dtype_lookup():
    assert mp.dtype("float32") is mp.float32
    assert mp.dtype("f8") is mp.float64
    assert mp.dtype(int) is mp.int64
    assert mp.dtype(mp.array([1], dtype=mp.int16)) is mp.int16


def _dtype_name(x):
    dt = getattr(x, "dtype", x)
    return str(dt).removeprefix("mlx.core.")


def test_finfo_and_iinfo_match_numpy():
    fi = mp.finfo(mp.float32)
    np_fi = np.finfo(np.float32)
    assert fi.bits == np_fi.bits
    assert fi.eps == np_fi.eps
    assert fi.max == np_fi.max

    ii = mp.iinfo("int16")
    np_ii = np.iinfo(np.int16)
    assert ii.bits == np_ii.bits
    assert ii.min == np_ii.min
    assert ii.max == np_ii.max


def test_creation_routines_and_like_variants():
    z = mp.zeros((2, 3), dtype=mp.int16)
    o = mp.ones((2, 3))
    f = mp.full((2, 3), 7, dtype=mp.int32)
    assert_array_equal(z, np.zeros((2, 3), dtype=np.int16))
    assert_array_equal(o, np.ones((2, 3), dtype=np.float32))
    assert_array_equal(f, np.full((2, 3), 7, dtype=np.int32))

    base = mp.array([[1, 2], [3, 4]], dtype=mp.float32)
    assert_array_equal(mp.zeros_like(base), np.zeros((2, 2), dtype=np.float32))
    assert_array_equal(mp.ones_like(base, dtype=mp.int32), np.ones((2, 2), dtype=np.int32))
    assert_array_equal(mp.full_like(base, 9, shape=(1, 4)), np.full((1, 4), 9, dtype=np.float32))


def test_constructor_default_dtype_parity_for_selected_creation_paths():
    assert _dtype_name(mp.array([1, 2, 3])) == np.array([1, 2, 3]).dtype.name
    assert _dtype_name(mp.array([1.0, 2.0])) == np.array([1.0, 2.0]).dtype.name
    assert _dtype_name(mp.asarray([1, 2, 3])) == np.asarray([1, 2, 3]).dtype.name

    assert _dtype_name(mp.zeros((2,))) == np.zeros((2,)).dtype.name
    assert _dtype_name(mp.ones((2,))) == np.ones((2,)).dtype.name
    assert _dtype_name(mp.full((2,), 1)) == np.full((2,), 1).dtype.name
    assert _dtype_name(mp.full((2,), 1.0)) == np.full((2,), 1.0).dtype.name

    assert _dtype_name(mp.arange(5)) == np.arange(5).dtype.name
    assert _dtype_name(mp.arange(1.5, 5.5, 0.5)) == np.arange(1.5, 5.5, 0.5).dtype.name
    assert _dtype_name(mp.linspace(0.0, 1.0, num=5)) == np.linspace(0.0, 1.0, num=5).dtype.name
    assert _dtype_name(mp.logspace(0, 2, num=3, base=10.0)) == np.logspace(0, 2, num=3, base=10.0).dtype.name
    assert _dtype_name(mp.eye(3)) == np.eye(3).dtype.name
    assert _dtype_name(mp.identity(4)) == np.identity(4).dtype.name
    assert _dtype_name(mp.tri(3, 4)) == np.tri(3, 4).dtype.name


def test_creation_explicit_dtype_overrides_preserved():
    assert _dtype_name(mp.array([1, 2, 3], dtype=mp.float32)) == "float32"
    assert _dtype_name(mp.asarray([1, 2, 3], dtype=mp.int16)) == "int16"
    assert _dtype_name(mp.zeros((2,), dtype=mp.float32)) == "float32"
    assert _dtype_name(mp.ones((2,), dtype=mp.int16)) == "int16"
    assert _dtype_name(mp.full((2,), 1, dtype=mp.int32)) == "int32"
    assert _dtype_name(mp.arange(5, dtype=mp.int32)) == "int32"
    assert _dtype_name(mp.linspace(0, 1, 5, dtype=mp.float32)) == "float32"


def test_empty_and_empty_like_are_zero_fallbacks():
    # MLX has no uninitialized allocation API, so MumPy intentionally uses zeros.
    e = mp.empty((2, 2), dtype=mp.int32)
    el = mp.empty_like(mp.array([1.0, 2.0]))
    assert_array_equal(e, np.zeros((2, 2), dtype=np.int32))
    assert_array_equal(el, np.zeros((2,), dtype=np.float32))


def test_range_construction_helpers():
    assert_array_equal(mp.arange(5), np.arange(5, dtype=np.int32))
    assert_allclose(mp.arange(1.5, 5.5, 0.5), np.arange(1.5, 5.5, 0.5, dtype=np.float32))

    x = mp.linspace(0.0, 1.0, num=5)
    assert_allclose(x, np.linspace(0.0, 1.0, num=5, dtype=np.float32))

    y = mp.linspace(0.0, 1.0, num=4, endpoint=False)
    assert_allclose(y, np.linspace(0.0, 1.0, num=4, endpoint=False, dtype=np.float32))

    z, step = mp.linspace(-2.0, 2.0, num=5, retstep=True)
    assert_allclose(z, np.linspace(-2.0, 2.0, num=5, dtype=np.float32))
    assert step.shape == ()
    assert step.item() == pytest.approx(1.0)

    ls = mp.logspace(0, 2, num=3, base=10.0)
    assert_allclose(ls, np.logspace(0, 2, num=3, base=10.0, dtype=np.float32))


def test_matrix_creation_helpers():
    assert_array_equal(mp.eye(3), np.eye(3, dtype=np.float32))
    assert_array_equal(mp.eye(3, M=4, k=1, dtype=mp.int32), np.eye(3, 4, k=1, dtype=np.int32))
    assert_array_equal(mp.identity(4), np.identity(4, dtype=np.float32))
    assert_array_equal(mp.tri(3, 4, k=-1, dtype=mp.int32), np.tri(3, 4, k=-1, dtype=np.int32))


def test_diag_and_diagflat():
    mat = mp.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(mp.diag(mat, k=1), np.diag(np.array([[1, 2, 3], [4, 5, 6]]), k=1))
    assert_array_equal(mp.diag(mp.array([7, 8]), k=-1), np.diag(np.array([7, 8]), k=-1))
    assert_array_equal(mp.diagflat([[1, 2], [3, 4]], k=1), np.diagflat(np.array([[1, 2], [3, 4]]), k=1))


def test_shape_introspection_and_scalar_helpers():
    x = mp.array([[1, 2, 3], [4, 5, 6]])
    assert mp.shape(x) == (2, 3)
    assert mp.ndim(x) == 2
    assert mp.size(x) == 6
    assert mp.size(x, axis=1) == 3
    assert mp.item(mp.array(42)) == 42
    assert mp.tolist(x) == [[1, 2, 3], [4, 5, 6]]
    assert mp.isscalar(3.14)
    assert not mp.isscalar(x)
    assert not mp.isscalar(np.array([1, 2]))


def test_fortran_order_paths_and_invalid_order_validation():
    x_np = np.array([[1, 2], [3, 4]], order="F")
    assert_array_equal(mp.array(x_np, order="F"), np.array(x_np, order="F"))
    assert_array_equal(mp.array(x_np, order="K"), np.array(x_np, order="K"))
    assert_array_equal(mp.asarray(x_np, order="A"), np.asarray(x_np, order="A"))
    assert_array_equal(mp.reshape(np.arange(6), (2, 3), order="F"), np.reshape(np.arange(6), (2, 3), order="F"))
    assert_array_equal(mp.reshape(x_np, (4,), order="A"), np.reshape(x_np, (4,), order="A"))
    assert_array_equal(mp.ravel(x_np, order="F"), np.ravel(x_np, order="F"))
    assert_array_equal(mp.ravel(x_np, order="K"), np.ravel(x_np, order="K"))

    with pytest.raises(ValueError, match="Unsupported order"):
        mp.array([1, 2], order="Z")
