"""Tests for math operations and reduction behavior."""

import numpy as np
import pytest

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def _dtype_name(x):
    dt = getattr(x, "dtype", x)
    return str(dt).removeprefix("mlx.core.")


def test_elementwise_arithmetic_and_unary_ops():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_np = np.array([[2.0, 5.0], [7.0, 11.0]], dtype=np.float32)
    x = mp.array(x_np)
    y = mp.array(y_np)

    assert_allclose(mp.add(x, y), x_np + y_np)
    assert_allclose(mp.subtract(x, y), x_np - y_np)
    assert_allclose(mp.multiply(x, y), x_np * y_np)
    assert_allclose(mp.divide(y, x), y_np / x_np)
    assert_allclose(mp.true_divide(y, x), y_np / x_np)
    assert_array_equal(mp.floor_divide([[5, 7]], [[2, 3]]), np.floor_divide([[5, 7]], [[2, 3]]))
    assert_allclose(mp.remainder(y, x), np.remainder(y_np, x_np))
    assert_allclose(mp.mod(y, x), np.mod(y_np, x_np))
    assert_allclose(mp.power(x, 2), np.power(x_np, 2))
    assert_allclose(mp.square(x), np.square(x_np))
    assert_allclose(mp.reciprocal(x), np.reciprocal(x_np))

    assert_allclose(mp.negative(x), -x_np)
    assert_allclose(mp.positive(x), x_np)
    assert_allclose(mp.abs(-x), np.abs(-x_np))
    assert_allclose(mp.absolute(-x), np.abs(-x_np))
    assert_allclose(mp.sign(mp.array([-2, 0, 3])), np.sign(np.array([-2, 0, 3])))


def test_transcendental_angle_and_rounding_ops():
    v_np = np.array([0.25, 0.5, 1.0], dtype=np.float32)
    v = mp.array(v_np)
    assert_allclose(mp.exp(v), np.exp(v_np))
    assert_allclose(mp.expm1(v), np.expm1(v_np))
    assert_allclose(mp.log(v), np.log(v_np))
    assert_allclose(mp.log1p(v), np.log1p(v_np))
    assert_allclose(mp.log2(v), np.log2(v_np))
    assert_allclose(mp.log10(v), np.log10(v_np))
    assert_allclose(mp.sqrt(v), np.sqrt(v_np))
    assert_allclose(mp.rsqrt(v), 1.0 / np.sqrt(v_np))

    angles_np = np.array([0.0, np.pi / 4, np.pi / 2], dtype=np.float32)
    angles = mp.array(angles_np)
    assert_allclose(mp.sin(angles), np.sin(angles_np))
    assert_allclose(mp.cos(angles), np.cos(angles_np))
    assert_allclose(mp.tan(angles_np[:2]), np.tan(angles_np[:2]))
    assert_allclose(
        mp.arcsin(mp.array([0.0, 1.0], dtype=mp.float32)),
        np.arcsin(np.array([0.0, 1.0], dtype=np.float32)),
    )
    assert_allclose(
        mp.arccos(mp.array([0.0, 1.0], dtype=mp.float32)),
        np.arccos(np.array([0.0, 1.0], dtype=np.float32)),
    )
    assert_allclose(
        mp.arctan(mp.array([0.0, 1.0], dtype=mp.float32)),
        np.arctan(np.array([0.0, 1.0], dtype=np.float32)),
    )
    assert_allclose(mp.sinh(v), np.sinh(v_np))
    assert_allclose(mp.cosh(v), np.cosh(v_np))
    assert_allclose(mp.tanh(v), np.tanh(v_np))

    degs = mp.array([0.0, 90.0, 180.0], dtype=mp.float32)
    assert_allclose(mp.deg2rad(degs), np.deg2rad(np.array([0.0, 90.0, 180.0], dtype=np.float32)))
    assert_allclose(mp.rad2deg(mp.deg2rad(degs)), np.array([0.0, 90.0, 180.0], dtype=np.float32))
    assert_allclose(
        mp.degrees(mp.array([0.0, np.pi], dtype=mp.float32)),
        np.degrees(np.array([0.0, np.pi], dtype=np.float32)),
    )
    assert_allclose(
        mp.radians(mp.array([0.0, 180.0], dtype=mp.float32)),
        np.radians(np.array([0.0, 180.0], dtype=np.float32)),
    )

    x = mp.array([-1.7, -0.2, 0.2, 1.7], dtype=mp.float32)
    x_np = np.array([-1.7, -0.2, 0.2, 1.7], dtype=np.float32)
    assert_allclose(mp.floor(x), np.floor(x_np))
    assert_allclose(mp.ceil(x), np.ceil(x_np))
    assert_allclose(mp.round(x, 0), np.round(x_np, 0))
    assert_allclose(mp.trunc(x), np.trunc(x_np))


def test_comparisons_logical_bitwise_and_nan_helpers():
    a = mp.array([1, 2, 3, 4])
    b = mp.array([2, 2, 1, 4])
    a_np = np.array([1, 2, 3, 4])
    b_np = np.array([2, 2, 1, 4])

    assert_array_equal(mp.equal(a, b), np.equal(a_np, b_np))
    assert_array_equal(mp.not_equal(a, b), np.not_equal(a_np, b_np))
    assert_array_equal(mp.greater(a, b), np.greater(a_np, b_np))
    assert_array_equal(mp.greater_equal(a, b), np.greater_equal(a_np, b_np))
    assert_array_equal(mp.less(a, b), np.less(a_np, b_np))
    assert_array_equal(mp.less_equal(a, b), np.less_equal(a_np, b_np))

    ta = mp.array([True, True, False, False])
    tb = mp.array([True, False, True, False])
    ta_np = np.array([True, True, False, False])
    tb_np = np.array([True, False, True, False])
    assert_array_equal(mp.logical_and(ta, tb), np.logical_and(ta_np, tb_np))
    assert_array_equal(mp.logical_or(ta, tb), np.logical_or(ta_np, tb_np))
    assert_array_equal(mp.logical_xor(ta, tb), np.logical_xor(ta_np, tb_np))
    assert_array_equal(mp.logical_not(ta), np.logical_not(ta_np))

    ia = mp.array([1, 6, 3], dtype=mp.int32)
    ib = mp.array([4, 2, 7], dtype=mp.int32)
    ia_np = np.array([1, 6, 3], dtype=np.int32)
    ib_np = np.array([4, 2, 7], dtype=np.int32)
    assert_array_equal(mp.bitwise_and(ia, ib), np.bitwise_and(ia_np, ib_np))
    assert_array_equal(mp.bitwise_or(ia, ib), np.bitwise_or(ia_np, ib_np))
    assert_array_equal(mp.bitwise_xor(ia, ib), np.bitwise_xor(ia_np, ib_np))
    assert_array_equal(mp.bitwise_not(ia), np.bitwise_not(ia_np))

    assert_array_equal(mp.maximum(a, b), np.maximum(a_np, b_np))
    assert_array_equal(mp.minimum(a, b), np.minimum(a_np, b_np))

    weird = mp.array([mp.nan, mp.inf, -mp.inf, 1.0], dtype=mp.float32)
    assert_array_equal(mp.isnan(weird), np.isnan(np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)))
    assert_array_equal(mp.isinf(weird), np.isinf(np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)))
    assert_array_equal(
        mp.isfinite(weird),
        np.isfinite(np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)),
    )
    assert_allclose(
        mp.nan_to_num(weird, nan=7.0, posinf=8.0, neginf=-9.0),
        np.nan_to_num(
            np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32),
            nan=7.0,
            posinf=8.0,
            neginf=-9.0,
        ),
    )

    assert_array_equal(
        mp.isclose([1.0, 2.0], [1.0, 2.00001], atol=1e-3),
        np.isclose([1.0, 2.0], [1.0, 2.00001], atol=1e-3),
    )
    assert mp.allclose([1.0, 2.0], [1.0, 2.00001], atol=1e-3)
    assert mp.array_equal([1, 2, 3], np.array([1, 2, 3]))


def test_reductions_statistics_and_cumulatives():
    x_np = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    x = mp.array(x_np)

    assert_allclose(mp.sum(x), np.sum(x_np))
    assert_allclose(mp.sum(x, axis=1, keepdims=True), np.sum(x_np, axis=1, keepdims=True))
    assert_allclose(mp.prod(x, axis=0), np.prod(x_np, axis=0))
    assert_allclose(mp.mean(x, axis=0), np.mean(x_np, axis=0))
    assert_allclose(mp.std(x, axis=1, ddof=1), np.std(x_np, axis=1, ddof=1))
    assert_allclose(mp.var(x, axis=1, ddof=1), np.var(x_np, axis=1, ddof=1))
    assert_allclose(mp.median(x, axis=0), np.median(x_np, axis=0))
    assert_allclose(mp.min(x, axis=1), np.min(x_np, axis=1))
    assert_allclose(mp.max(x, axis=0), np.max(x_np, axis=0))
    assert_allclose(mp.amin(x), np.amin(x_np))
    assert_allclose(mp.amax(x), np.amax(x_np))
    assert_allclose(mp.ptp(x, axis=1), np.ptp(x_np, axis=1))

    ints = mp.array([[0, 1, 0], [2, 0, 3]], dtype=mp.int32)
    ints_np = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int32)
    assert_array_equal(mp.argmin(x, axis=1), np.argmin(x_np, axis=1))
    assert_array_equal(mp.argmax(x, axis=0), np.argmax(x_np, axis=0))
    assert_array_equal(mp.all([[True, False], [True, True]], axis=1), np.all([[True, False], [True, True]], axis=1))
    assert_array_equal(
        mp.any([[False, False], [False, True]], axis=0),
        np.any([[False, False], [False, True]], axis=0),
    )
    assert_array_equal(mp.count_nonzero(ints), np.count_nonzero(ints_np))
    assert_array_equal(mp.count_nonzero(ints, axis=1), np.count_nonzero(ints_np, axis=1))

    assert_array_equal(mp.cumsum([1, 2, 3, 4]), np.cumsum(np.array([1, 2, 3, 4])))
    assert_array_equal(mp.cumprod([1, 2, 3, 4]), np.cumprod(np.array([1, 2, 3, 4])))
    assert_array_equal(mp.diff([1, 4, 9, 16]), np.diff(np.array([1, 4, 9, 16])))
    assert_array_equal(mp.diff([True, True, False, True]), np.diff(np.array([True, True, False, True])))
    assert_array_equal(mp.diff([1, 2, 3], prepend=0), np.diff(np.array([1, 2, 3]), prepend=0))
    with pytest.raises(ValueError, match="order must be non-negative"):
        mp.diff([1, 2, 3], n=-1)


def test_quantiles_nan_reductions_and_query_helpers():
    x_np = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    x = mp.array(x_np)

    # Quantiles / percentiles (non-NaN path)
    base_np = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 16.0]], dtype=np.float32)
    base = mp.array(base_np)
    assert_allclose(mp.quantile(base, 0.5), np.quantile(base_np, 0.5), rtol=1e-6, atol=1e-6)
    assert_allclose(
        mp.quantile(base, [0.25, 0.75], axis=1, keepdims=True),
        np.quantile(base_np, [0.25, 0.75], axis=1, keepdims=True),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        mp.percentile(base, [25, 75], axis=0),
        np.percentile(base_np, [25, 75], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(NotImplementedError):
        mp.quantile(base, 0.5, overwrite_input=True)
    with pytest.raises(NotImplementedError):
        mp.percentile(base, 50, out=object())

    # NaN-aware reductions
    assert_allclose(mp.nansum(x), np.nansum(x_np))
    assert_allclose(mp.nanprod(x, axis=1), np.nanprod(x_np, axis=1))
    assert_allclose(mp.nanmean(x, axis=0), np.nanmean(x_np, axis=0))
    assert_allclose(mp.nanstd(x, axis=1, ddof=0), np.nanstd(x_np, axis=1, ddof=0), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.nanvar(x, axis=1, ddof=0), np.nanvar(x_np, axis=1, ddof=0), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.nanmedian(x, axis=0), np.nanmedian(x_np, axis=0))
    assert_allclose(mp.nanmin(x, axis=1), np.nanmin(x_np, axis=1))
    assert_allclose(mp.nanmax(x, axis=0), np.nanmax(x_np, axis=0))

    cx_np = np.array(
        [[1.0 + 2.0j, np.nan + 1.0j, 3.0 + 0.0j], [2.0 + np.nan * 1j, 4.0 + 5.0j, np.nan + 0.0j]],
        dtype=np.complex64,
    )
    assert_allclose(mp.nanmin(cx_np, axis=1), np.nanmin(cx_np, axis=1), rtol=1e-6, atol=1e-6, equal_nan=True)
    assert_allclose(mp.nanmax(cx_np, axis=0), np.nanmax(cx_np, axis=0), rtol=1e-6, atol=1e-6, equal_nan=True)
    with pytest.raises(TypeError):
        mp.nanquantile(cx_np, [0.25, 0.75], axis=1)
    with pytest.raises(TypeError):
        np.nanquantile(cx_np, [0.25, 0.75], axis=1)

    # Query helpers
    q_np = np.array([[0, 2, 0], [3, 0, 4]], dtype=np.int32)
    q = mp.array(q_np)
    nz = mp.nonzero(q)
    nz_np = np.nonzero(q_np)
    assert len(nz) == len(nz_np) == 2
    for got, exp in zip(nz, nz_np, strict=False):
        assert_array_equal(got, exp)
    assert_array_equal(mp.flatnonzero(q), np.flatnonzero(q_np))
    assert_array_equal(mp.argwhere(q), np.argwhere(q_np))

    sorted_np = np.array([1, 3, 5, 7], dtype=np.int32)
    assert_array_equal(mp.searchsorted(sorted_np, [0, 4, 8]), np.searchsorted(sorted_np, [0, 4, 8]))
    sorter = np.argsort(np.array([30, 10, 20]))
    assert_array_equal(
        mp.searchsorted(np.array([30, 10, 20]), [15, 25], sorter=sorter),
        np.searchsorted(np.array([30, 10, 20]), [15, 25], sorter=sorter),
    )


def test_sorting_uniques_and_complex_helpers():
    x_np = np.array([3, 1, 2, 3, 1], dtype=np.int32)
    x = mp.array(x_np)
    assert_array_equal(mp.sort(x), np.sort(x_np))
    assert_array_equal(mp.argsort(x), np.argsort(x_np))
    assert_array_equal(mp.unique(x), np.unique(x_np))

    c_np = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    c = mp.array(c_np)
    assert_allclose(mp.real(c), np.real(c_np))
    assert_allclose(mp.imag(c), np.imag(c_np))
    assert_allclose(mp.conj(c), np.conj(c_np))
    assert_allclose(mp.conjugate(c), np.conjugate(c_np))


def test_vector_matrix_and_tensor_products():
    a_np = np.array([1, 2, 3], dtype=np.float32)
    b_np = np.array([4, 5, 6], dtype=np.float32)
    a = mp.array(a_np)
    b = mp.array(b_np)
    assert_allclose(mp.dot(a, b), np.dot(a_np, b_np))

    m_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    n_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    m = mp.array(m_np)
    n = mp.array(n_np)
    assert_allclose(mp.dot(m, n), np.dot(m_np, n_np))
    assert_allclose(mp.matmul(m, n), np.matmul(m_np, n_np))

    t1_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t2_np = np.arange(20, dtype=np.float32).reshape(4, 5)
    assert_allclose(mp.dot(mp.array(t1_np), mp.array(t2_np)), np.dot(t1_np, t2_np))
    assert_allclose(mp.inner(m, m), np.inner(m_np, m_np))
    assert_allclose(mp.outer(a, b), np.outer(a_np, b_np))
    assert_allclose(mp.tensordot(mp.array(t1_np), mp.array(t2_np), axes=1), np.tensordot(t1_np, t2_np, axes=1))
    assert_allclose(mp.einsum("ij,jk->ik", m, n), np.einsum("ij,jk->ik", m_np, n_np))
    assert_allclose(mp.kron([[1, 2], [3, 4]], [[0, 5], [6, 7]]), np.kron([[1, 2], [3, 4]], [[0, 5], [6, 7]]))

    cx_np = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    cy_np = np.array([5 + 6j, 7 + 8j], dtype=np.complex64)
    assert_allclose(mp.vdot(mp.array(cx_np), mp.array(cy_np)), np.vdot(cx_np, cy_np))

    c1_np = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c2_np = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert_allclose(mp.cross(mp.array(c1_np), mp.array(c2_np)), np.cross(c1_np, c2_np))


def test_selected_reduction_default_dtype_parity_and_values():
    ints = mp.array([1, 2, 3], dtype=mp.int32)
    ints_np = np.array([1, 2, 3], dtype=np.int32)

    assert _dtype_name(mp.sum(ints)) == np.sum(ints_np).dtype.name
    assert _dtype_name(mp.prod(ints)) == np.prod(ints_np).dtype.name
    assert _dtype_name(mp.mean(ints)) == np.mean(ints_np).dtype.name
    assert _dtype_name(mp.var(ints)) == np.var(ints_np).dtype.name
    assert _dtype_name(mp.std(ints)) == np.std(ints_np).dtype.name
    assert _dtype_name(mp.nansum(ints)) == np.nansum(ints_np).dtype.name
    assert _dtype_name(mp.nanprod(ints)) == np.nanprod(ints_np).dtype.name
    assert _dtype_name(mp.average(ints)) == np.average(ints_np).dtype.name
    assert _dtype_name(mp.average(ints, weights=[1, 2, 3])) == np.average(ints_np, weights=[1, 2, 3]).dtype.name

    assert_allclose(mp.mean(ints), np.mean(ints_np))
    assert_allclose(mp.var(ints), np.var(ints_np))
    assert_allclose(mp.std(ints), np.std(ints_np))
    assert_allclose(mp.average(ints, weights=[1, 2, 3]), np.average(ints_np, weights=[1, 2, 3]))

    f32 = mp.array([1.0, np.nan, 3.0], dtype=mp.float32)
    f32_np = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    assert _dtype_name(mp.nanmean(f32)) == np.nanmean(f32_np).dtype.name
    assert _dtype_name(mp.nanvar(f32)) == np.nanvar(f32_np).dtype.name
    assert _dtype_name(mp.nanstd(f32)) == np.nanstd(f32_np).dtype.name


def test_selected_reduction_explicit_dtype_overrides_preserved():
    ints = mp.array([1, 2, 3], dtype=mp.int32)
    assert _dtype_name(mp.sum(ints, dtype=mp.int32)) == "int32"
    assert _dtype_name(mp.prod(ints, dtype=mp.int32)) == "int32"
    assert _dtype_name(mp.mean(ints, dtype=mp.float32)) == "float32"
    assert _dtype_name(mp.var(ints, dtype=mp.float32)) == "float32"
    assert _dtype_name(mp.std(ints, dtype=mp.float32)) == "float32"
    assert _dtype_name(mp.nansum(ints, dtype=mp.int32)) == "int32"
    assert _dtype_name(mp.nanprod(ints, dtype=mp.int32)) == "int32"


def test_eval_executes_without_error():
    x = mp.arange(10)
    y = mp.sin(x)
    mp.eval(x, y)
    assert y.shape == (10,)


def test_additional_core_default_dtype_parity_for_cumulatives_histogram_and_indices():
    ints_mp = mp.array([1, 2, 3], dtype=mp.int32)
    ints_np = np.array([1, 2, 3], dtype=np.int32)

    assert _dtype_name(mp.cumsum(ints_mp)) == np.cumsum(ints_np).dtype.name
    assert _dtype_name(mp.cumprod(ints_mp)) == np.cumprod(ints_np).dtype.name
    assert _dtype_name(mp.nancumsum(ints_mp)) == np.nancumsum(ints_np).dtype.name
    assert _dtype_name(mp.nancumprod(ints_mp)) == np.nancumprod(ints_np).dtype.name
    assert (
        _dtype_name(mp.trace(mp.array([[1, 2], [3, 4]], dtype=mp.int32)))
        == np.trace(np.array([[1, 2], [3, 4]], dtype=np.int32)).dtype.name
    )
    assert _dtype_name(mp.float_power(2, 3)) == np.float_power(2, 3).dtype.name

    hist_mp, edges_mp = mp.histogram(mp.array([1, 2, 3], dtype=mp.int32), bins=3)
    hist_np, edges_np = np.histogram(np.array([1, 2, 3], dtype=np.int32), bins=3)
    assert _dtype_name(hist_mp) == hist_np.dtype.name
    assert _dtype_name(edges_mp) == edges_np.dtype.name

    assert (
        _dtype_name(mp.searchsorted(mp.array([1, 3, 5], dtype=mp.int32), [0, 4, 7]))
        == np.searchsorted(np.array([1, 3, 5], dtype=np.int32), [0, 4, 7]).dtype.name
    )
    assert _dtype_name(mp.digitize([0.5, 1.5], [0.0, 1.0, 2.0])) == np.digitize([0.5, 1.5], [0.0, 1.0, 2.0]).dtype.name
    assert (
        _dtype_name(mp.flatnonzero(mp.array([[0, 1], [2, 0]], dtype=mp.int32)))
        == np.flatnonzero(np.array([[0, 1], [2, 0]], dtype=np.int32)).dtype.name
    )
    assert (
        _dtype_name(mp.argwhere(mp.array([[0, 1], [2, 0]], dtype=mp.int32)))
        == np.argwhere(np.array([[0, 1], [2, 0]], dtype=np.int32)).dtype.name
    )

    ur_mp = mp.unravel_index(mp.array([1, 3], dtype=mp.int32), (2, 2))
    ur_np = np.unravel_index(np.array([1, 3], dtype=np.int32), (2, 2))
    for got, exp in zip(ur_mp, ur_np, strict=False):
        assert _dtype_name(got) == exp.dtype.name

    assert (
        _dtype_name(mp.ravel_multi_index(([1, 0], [0, 1]), (2, 2)))
        == np.ravel_multi_index(([1, 0], [0, 1]), (2, 2)).dtype.name
    )


def test_out_none_is_accepted_for_ufunc_style_core_functions() -> None:
    x = mp.array([1.0, 2.0, 4.0], dtype=mp.float64)
    y = mp.array([2.0, 4.0, 8.0], dtype=mp.float64)

    assert_allclose(mp.add(x, y, out=None), np.add(np.asarray(x), np.asarray(y)))
    assert_allclose(mp.divide(y, x, out=None), np.divide(np.asarray(y), np.asarray(x)))
    assert_allclose(mp.sin(x, out=None), np.sin(np.asarray(x)))

    with pytest.raises(TypeError, match="out=None"):
        mp.divide(y, x, out=mp.zeros_like(x))
