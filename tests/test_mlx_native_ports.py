"""Tests for newly ported MLX-native paths and no-fallback expectations."""

import numpy as np
import pytest

import mumpy as mp
import mumpy._bridge as mp_bridge
import mumpy._core as mp_core

from .conftest import assert_allclose, assert_array_equal, to_numpy


def _fail_numpy(*_args: object, **_kwargs: object):
    msg = "NumPy fallback should not be used on this path"
    raise AssertionError(msg)


def test_core_numeric_ports_match_numpy_and_avoid_numpy(monkeypatch):
    monkeypatch.setattr(mp_core, "_numpy", _fail_numpy)
    monkeypatch.setattr(mp_bridge, "numpy_module", _fail_numpy)

    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    y = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    z = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    assert_allclose(mp.exp2(x), np.exp2(x), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.copysign(y, x - 0.75), np.copysign(y, x - 0.75), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.hypot([3.0, 5.0], [4.0, 12.0]), np.hypot([3.0, 5.0], [4.0, 12.0]), rtol=1e-6, atol=1e-6)
    assert_array_equal(mp.signbit([-1.0, 0.0, 2.0]), np.signbit([-1.0, 0.0, 2.0]))
    assert_array_equal(
        mp.signbit(np.array([-0.0, 0.0], dtype=np.float32)),
        np.signbit(np.array([-0.0, 0.0], dtype=np.float32)),
    )
    assert_allclose(mp.logaddexp2(x, z), np.logaddexp2(x, z), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.sinc([-1.0, 0.0, 1.0]), np.sinc([-1.0, 0.0, 1.0]), rtol=1e-6, atol=1e-6)
    assert_allclose(
        mp.fmax([1.0, np.nan, 3.0], [2.0, 4.0, np.nan]),
        np.fmax([1.0, np.nan, 3.0], [2.0, 4.0, np.nan]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        mp.fmin([1.0, np.nan, 3.0], [2.0, 4.0, np.nan]),
        np.fmin([1.0, np.nan, 3.0], [2.0, 4.0, np.nan]),
        rtol=1e-6,
        atol=1e-6,
    )

    assert_allclose(mp.geomspace(1.0, 16.0, num=5), np.geomspace(1.0, 16.0, num=5), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.geomspace(-1.0, -16.0, num=5), np.geomspace(-1.0, -16.0, num=5), rtol=1e-6, atol=1e-6)
    assert_allclose(
        mp.geomspace(
            np.array([1.0, 4.0], dtype=np.float32),
            np.array([16.0, 64.0], dtype=np.float32),
            num=4,
            axis=1,
        ),
        np.geomspace(
            np.array([1.0, 4.0], dtype=np.float32),
            np.array([16.0, 64.0], dtype=np.float32),
            num=4,
            axis=1,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        mp.geomspace(-1.0, 1.0, num=5),
        np.geomspace(-1.0, 1.0, num=5),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="Geometric sequence cannot include zero"):
        mp.geomspace(0.0, 1.0, num=4)

    m = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert_array_equal(mp.rollaxis(m, 1, 0), np.rollaxis(m, 1, 0))
    assert_array_equal(mp.matrix_transpose(m), np.matrix_transpose(m))
    assert_allclose(mp.vecdot(m, m), np.vecdot(m, m), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.matvec(m, [1.0, 2.0, 3.0]), np.matvec(m, [1.0, 2.0, 3.0]), rtol=1e-6, atol=1e-6)
    assert_allclose(
        mp.vecmat([1.0, 2.0], np.arange(6, dtype=np.float32).reshape(2, 3)),
        np.vecmat([1.0, 2.0], np.arange(6, dtype=np.float32).reshape(2, 3)),
        rtol=1e-6,
        atol=1e-6,
    )

    nan_arr = np.array([1.0, np.nan, 2.0, np.nan], dtype=np.float32)
    assert_allclose(mp.nancumsum(nan_arr), np.nancumsum(nan_arr), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.nancumprod(nan_arr), np.nancumprod(nan_arr), rtol=1e-6, atol=1e-6)

    assert_array_equal(mp.vander([1, 2, 3], N=3), np.vander([1, 2, 3], N=3))
    assert_array_equal(mp.trim_zeros([0, 0, 1, 2, 0], trim="fb"), np.trim_zeros(np.array([0, 0, 1, 2, 0]), trim="fb"))
    u_mp = mp.unwrap([0.0, 3.0, 6.5])
    u_np = np.unwrap([0.0, 3.0, 6.5])
    assert_allclose(u_mp, u_np, rtol=1e-6, atol=1e-6)

    c = mp.array([1 + 1e-9j, 2 - 1e-10j], dtype=mp.complex64)
    assert_allclose(
        mp.real_if_close(c),
        np.real_if_close(np.array([1 + 1e-9j, 2 - 1e-10j], dtype=np.complex64)),
        rtol=1e-6,
        atol=1e-6,
    )

    chunks = mp.unstack(np.arange(6).reshape(2, 3), axis=0)
    if hasattr(np, "unstack"):
        np_chunks = np.unstack(np.arange(6).reshape(2, 3), axis=0)
    else:
        np_chunks = tuple(np.squeeze(v, axis=0) for v in np.split(np.arange(6).reshape(2, 3), 2, axis=0))
    for got, exp in zip(chunks, np_chunks, strict=False):
        assert_array_equal(got, exp)

    finite = mp.asarray_chkfinite([1.0, 2.0, 3.0])
    assert_array_equal(finite, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        mp.asarray_chkfinite([1.0, np.inf])

    a = mp.arange(5)
    b = mp.arange(5)
    assert mp.may_share_memory(a, a)
    assert not mp.may_share_memory(a, b)
    assert mp.shares_memory(a, a)
    assert not mp.shares_memory(a, b)


def test_core_index_search_ports_match_numpy_and_avoid_numpy(monkeypatch):
    monkeypatch.setattr(mp_core, "_numpy", _fail_numpy)
    monkeypatch.setattr(mp_bridge, "numpy_module", _fail_numpy)

    sorted_vals = np.array([1, 3, 5, 7], dtype=np.int32)
    probes = np.array([0, 3, 4, 9], dtype=np.int32)
    assert_array_equal(
        mp.searchsorted(sorted_vals, probes, side="left"),
        np.searchsorted(sorted_vals, probes, side="left"),
    )
    assert_array_equal(
        mp.searchsorted(sorted_vals, probes, side="right"),
        np.searchsorted(sorted_vals, probes, side="right"),
    )

    unsorted = np.array([30, 10, 20], dtype=np.int32)
    sorter = np.argsort(unsorted)
    assert_array_equal(
        mp.searchsorted(unsorted, [15, 25], sorter=sorter),
        np.searchsorted(unsorted, [15, 25], sorter=sorter),
    )

    bins_inc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    bins_dec = bins_inc[::-1]
    vals = np.array([-1.0, 0.5, 1.5, 3.0], dtype=np.float32)
    assert_array_equal(mp.digitize(vals, bins_inc), np.digitize(vals, bins_inc))
    assert_array_equal(mp.digitize(vals, bins_dec), np.digitize(vals, bins_dec))

    p_src = np.array([7, 2, 9, 1, 5, 4], dtype=np.int32)
    kth = 2
    p_vals = to_numpy(mp.partition(p_src, kth))
    assert_array_equal(np.sort(p_vals), np.sort(p_src))
    assert p_vals[kth] == np.partition(p_src, kth)[kth]
    assert np.all(p_vals[:kth] <= p_vals[kth])
    assert np.all(p_vals[kth + 1 :] >= p_vals[kth])

    p_idx = to_numpy(mp.argpartition(p_src, kth))
    assert_array_equal(np.sort(p_idx), np.arange(p_src.shape[0]))
    p_take = p_src[p_idx]
    assert p_take[kth] == np.partition(p_src, kth)[kth]
    assert np.all(p_take[:kth] <= p_take[kth])
    assert np.all(p_take[kth + 1 :] >= p_take[kth])

    bc_x = np.array([0, 1, 1, 2, 4, 4, 4], dtype=np.int32)
    bc_w = np.array([1.0, 0.5, 1.5, 1.0, 2.0, 1.0, 0.5], dtype=np.float32)
    assert_array_equal(mp.bincount(bc_x), np.bincount(bc_x))
    assert_allclose(mp.bincount(bc_x, weights=bc_w), np.bincount(bc_x, weights=bc_w), rtol=1e-6, atol=1e-6)

    h_data = np.array([0.1, 0.2, 0.8, 1.5, 2.2, 2.8], dtype=np.float32)
    h_mp, e_mp = mp.histogram(h_data, bins=3, range=(0.0, 3.0))
    h_np, e_np = np.histogram(h_data, bins=3, range=(0.0, 3.0))
    assert_array_equal(h_mp, h_np)
    assert_allclose(e_mp, e_np, rtol=1e-6, atol=1e-6)
    hd_mp, ed_mp = mp.histogram(h_data, bins=[0.0, 1.0, 2.0, 3.0], density=True)
    hd_np, ed_np = np.histogram(h_data, bins=[0.0, 1.0, 2.0, 3.0], density=True)
    assert_allclose(hd_mp, hd_np, rtol=1e-6, atol=1e-6)
    assert_allclose(ed_mp, ed_np, rtol=1e-6, atol=1e-6)

    h2x = np.array([0.1, 0.2, 0.8, 1.5, 1.8, 2.2], dtype=np.float32)
    h2y = np.array([0.0, 0.7, 1.1, 1.4, 2.1, 2.9], dtype=np.float32)
    h2_mp, x2_mp, y2_mp = mp.histogram2d(h2x, h2y, bins=3, range=((0.0, 3.0), (0.0, 3.0)))
    h2_np, x2_np, y2_np = np.histogram2d(h2x, h2y, bins=3, range=((0.0, 3.0), (0.0, 3.0)))
    assert_array_equal(h2_mp, h2_np)
    assert_allclose(x2_mp, x2_np, rtol=1e-6, atol=1e-6)
    assert_allclose(y2_mp, y2_np, rtol=1e-6, atol=1e-6)

    hdd_sample = np.stack([h2x, h2y, np.array([0.5, 0.5, 0.2, 1.2, 1.7, 2.8], dtype=np.float32)], axis=1)
    hdd_mp, hdd_edges_mp = mp.histogramdd(hdd_sample, bins=(2, 2, 2), range=((0.0, 3.0), (0.0, 3.0), (0.0, 3.0)))
    hdd_np, hdd_edges_np = np.histogramdd(hdd_sample, bins=(2, 2, 2), range=((0.0, 3.0), (0.0, 3.0), (0.0, 3.0)))
    assert_array_equal(hdd_mp, hdd_np)
    for got, exp in zip(hdd_edges_mp, hdd_edges_np, strict=False):
        assert_allclose(got, exp, rtol=1e-6, atol=1e-6)

    xp_p = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    fp_p = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    xq_p = np.array([-0.5, 0.25, 1.75, 2.5], dtype=np.float32)
    assert_allclose(
        mp.interp(xq_p, xp_p, fp_p, period=2.0),
        np.interp(xq_p, xp_p, fp_p, period=2.0),
        rtol=1e-6,
        atol=1e-6,
    )

    pw_x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    pw_mp = mp.piecewise(pw_x, [pw_x < 0, pw_x > 0], [lambda z: z**2, lambda z: z + 1, -7.0])
    pw_np = np.piecewise(pw_x, [pw_x < 0, pw_x > 0], [lambda z: z**2, lambda z: z + 1, -7.0])
    assert_allclose(pw_mp, pw_np, rtol=1e-6, atol=1e-6)
    pw_vec_mp = mp.piecewise(pw_x, [pw_x < 0, pw_x >= 0], [lambda z: z * 2.0, lambda z: z + 3.0])
    pw_vec_np = np.piecewise(pw_x, [pw_x < 0, pw_x >= 0], [lambda z: z * 2.0, lambda z: z + 3.0])
    assert_allclose(pw_vec_mp, pw_vec_np, rtol=1e-6, atol=1e-6)

    pad_base = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert_array_equal(
        mp.pad(pad_base, ((2, 1), (3, 2)), mode="reflect"),
        np.pad(pad_base, ((2, 1), (3, 2)), mode="reflect"),
    )
    assert_array_equal(
        mp.pad(pad_base, ((2, 1), (3, 2)), mode="symmetric"),
        np.pad(pad_base, ((2, 1), (3, 2)), mode="symmetric"),
    )

    td_a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    td_b = np.arange(60, dtype=np.float32).reshape(4, 5, 3)
    assert_allclose(
        mp.tensordot(td_a, td_b, axes=([2, 1], [0, 2])),
        np.tensordot(td_a, td_b, axes=([2, 1], [0, 2])),
        rtol=1e-6,
        atol=1e-6,
    )

    q = np.array([[0, 2, 0], [3, 0, 4]], dtype=np.int32)
    for got, exp in zip(mp.nonzero(q), np.nonzero(q), strict=False):
        assert_array_equal(got, exp)
    assert_array_equal(mp.flatnonzero(q), np.flatnonzero(q))
    assert_array_equal(mp.argwhere(q), np.argwhere(q))

    arr = np.arange(12).reshape(3, 4)
    assert_array_equal(mp.compress([True, False, True], arr, axis=0), np.compress([True, False, True], arr, axis=0))
    assert_array_equal(mp.extract(arr % 2 == 0, arr), np.extract(arr % 2 == 0, arr))
    assert_array_equal(mp.insert(np.array([1, 2, 3]), 1, [9, 8]), np.insert(np.array([1, 2, 3]), 1, [9, 8]))
    assert_array_equal(
        mp.insert(arr, 1, np.array([99, 98, 97, 96], dtype=np.int64), axis=0),
        np.insert(arr, 1, np.array([99, 98, 97, 96], dtype=np.int64), axis=0),
    )
    assert_array_equal(mp.delete(arr, [0, 2], axis=0), np.delete(arr, [0, 2], axis=0))
    assert_array_equal(
        mp.delete(np.array([0, 1, 2, 3]), [True, False, True, False]),
        np.delete(np.array([0, 1, 2, 3]), [True, False, True, False]),
    )

    assert_array_equal(
        mp.ediff1d([1, 4, 9], to_begin=[0], to_end=[99]),
        np.ediff1d([1, 4, 9], to_begin=[0], to_end=[99]),
    )
    assert_array_equal(mp.diff(arr, axis=1, prepend=0, append=100), np.diff(arr, axis=1, prepend=0, append=100))
    assert_allclose(
        mp.trapezoid([[0.0, 1.0, 2.0], [1.0, 3.0, 5.0]], x=[0.0, 0.5, 2.0], axis=1),
        np.trapezoid([[0.0, 1.0, 2.0], [1.0, 3.0, 5.0]], x=[0.0, 0.5, 2.0], axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        mp.interp([-1.0, 0.5, 3.0], [0.0, 1.0, 2.0], [10.0, 20.0, 40.0]),
        np.interp([-1.0, 0.5, 3.0], [0.0, 1.0, 2.0], [10.0, 20.0, 40.0]),
        rtol=1e-6,
        atol=1e-6,
    )

    ch_idx = np.array([[0, 1], [1, 0]], dtype=np.int32)
    assert_array_equal(
        mp.choose(ch_idx, [np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]])]),
        np.choose(ch_idx, [np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]])]),
    )
    conds = [arr % 2 == 0, arr % 3 == 0]
    choices = [arr * 10, -arr]
    assert_array_equal(mp.select(conds, choices, default=7), np.select(conds, choices, default=7))
    assert_array_equal(
        mp.unique(np.array([3, 1, 2, 3, np.nan, np.nan], dtype=np.float32)),
        np.unique(np.array([3, 1, 2, 3, np.nan, np.nan], dtype=np.float32)),
    )
    a1 = np.array([1, 2, 3, 2, 4], dtype=np.int32)
    a2 = np.array([2, 4], dtype=np.int32)
    assert_array_equal(mp.in1d(a1, a2), np.isin(a1.ravel(), a2.ravel()))
    assert_array_equal(mp.isin(arr, [1, 5, 9]), np.isin(arr, [1, 5, 9]))
    vals_i, idx1_i, idx2_i = mp.intersect1d(a1, np.array([2, 4, 5], dtype=np.int32), return_indices=True)
    np_vals_i, np_idx1_i, np_idx2_i = np.intersect1d(a1, np.array([2, 4, 5], dtype=np.int32), return_indices=True)
    assert_array_equal(vals_i, np_vals_i)
    assert_array_equal(idx1_i, np_idx1_i)
    assert_array_equal(idx2_i, np_idx2_i)
    assert_array_equal(mp.union1d(a1, a2), np.union1d(a1, a2))
    assert_array_equal(mp.setdiff1d(a1, a2), np.setdiff1d(a1, a2))
    assert_array_equal(mp.setxor1d(a1, a2), np.setxor1d(a1, a2))

    gx, gy = mp.meshgrid(mp.arange(2), mp.arange(3), copy=False)
    nx, ny = np.meshgrid(np.arange(2), np.arange(3), copy=False)
    assert_array_equal(gx, nx)
    assert_array_equal(gy, ny)
    ca = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    cb = (np.arange(12, dtype=np.float32) + 1).reshape(2, 3, 2)
    assert_allclose(
        mp.cross(ca, cb, axisa=1, axisb=1, axisc=0),
        np.cross(ca, cb, axisa=1, axisb=1, axisc=0),
        rtol=1e-6,
        atol=1e-6,
    )

    grad_base = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float32)
    grad_x = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
    assert_allclose(mp.gradient(grad_base), np.gradient(grad_base), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.gradient(grad_base, grad_x), np.gradient(grad_base, grad_x), rtol=1e-6, atol=1e-6)
    g_mp = mp.gradient(arr.astype(np.float32), axis=(0, 1))
    g_np = np.gradient(arr.astype(np.float32), axis=(0, 1))
    for got, exp in zip(g_mp, g_np, strict=False):
        assert_allclose(got, exp, rtol=1e-6, atol=1e-6)

    stats = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    assert_allclose(
        mp.quantile(arr.astype(np.float32), [0.25, 0.75], axis=1),
        np.quantile(arr.astype(np.float32), [0.25, 0.75], axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(mp.nanquantile(stats, 0.5, axis=1), np.nanquantile(stats, 0.5, axis=1), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.nanmedian(stats, axis=0), np.nanmedian(stats, axis=0), rtol=1e-6, atol=1e-6)
    arr3 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    w23 = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    assert_allclose(
        mp.average(arr3, axis=(1, 2), weights=w23),
        np.average(arr3, axis=(1, 2), weights=w23),
        rtol=1e-6,
        atol=1e-6,
    )
    fw = np.array([1, 2, 1], dtype=np.int32)
    aw = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    cmat = np.array([[1.0, 2.0, 4.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    assert_allclose(
        mp.cov(cmat, fweights=fw, aweights=aw),
        np.cov(cmat, fweights=fw, aweights=aw),
        rtol=1e-5,
        atol=1e-5,
    )
    assert_allclose(mp.corrcoef(cmat), np.corrcoef(cmat), rtol=1e-5, atol=1e-5)

    dense = mp.indices((2, 3))
    dense_np = np.indices((2, 3))
    assert_array_equal(dense, dense_np)
    sparse = mp.indices((2, 3), sparse=True)
    sparse_np = np.indices((2, 3), sparse=True)
    for got, exp in zip(sparse, sparse_np, strict=False):
        assert_array_equal(got, exp)

    ix_mp = mp.ix_(np.array([True, False, True]), np.array([1, 3]))
    ix_np = np.ix_(np.array([True, False, True]), np.array([1, 3]))
    for got, exp in zip(ix_mp, ix_np, strict=False):
        assert_array_equal(got, exp)

    for got, exp in zip(mp.unravel_index([0, 5, 7], (2, 2, 2)), np.unravel_index([0, 5, 7], (2, 2, 2)), strict=False):
        assert_array_equal(got, exp)
    for got, exp in zip(
        mp.unravel_index([0, 5, 7], (2, 2, 2), order="F"),
        np.unravel_index([0, 5, 7], (2, 2, 2), order="F"),
        strict=False,
    ):
        assert_array_equal(got, exp)
    assert_array_equal(
        mp.ravel_multi_index(([0, 1, 1], [0, 0, 1]), (2, 2)),
        np.ravel_multi_index(([0, 1, 1], [0, 0, 1]), (2, 2)),
    )
    assert_array_equal(
        mp.ravel_multi_index(([2, -1], [0, 3]), (2, 3), mode=("clip", "wrap")),
        np.ravel_multi_index(([2, -1], [0, 3]), (2, 3), mode=("clip", "wrap")),
    )

    for mp_idx, np_idx in [
        (mp.diag_indices(3), np.diag_indices(3)),
        (mp.diag_indices_from(np.arange(27).reshape(3, 3, 3)), np.diag_indices_from(np.arange(27).reshape(3, 3, 3))),
        (mp.triu_indices(3, k=1, m=4), np.triu_indices(3, k=1, m=4)),
        (mp.tril_indices(3, k=0, m=4), np.tril_indices(3, k=0, m=4)),
    ]:
        for got, exp in zip(mp_idx, np_idx, strict=False):
            assert_array_equal(got, exp)


def test_linalg_fft_ports_match_numpy_and_avoid_numpy(monkeypatch):
    monkeypatch.setattr(mp_core, "_numpy", _fail_numpy)
    monkeypatch.setattr(mp_bridge, "numpy_module", _fail_numpy)

    a = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    assert_allclose(mp.linalg.svdvals(a), np.linalg.svdvals(a), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.vector_norm(a, axis=1), np.linalg.vector_norm(a, axis=1), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.matrix_norm(a), np.linalg.matrix_norm(a), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.vecdot(a, a), np.linalg.vecdot(a, a), rtol=1e-6, atol=1e-6)
    assert_array_equal(
        mp.linalg.matrix_transpose(np.arange(6).reshape(2, 3)),
        np.linalg.matrix_transpose(np.arange(6).reshape(2, 3)),
    )

    t = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert_allclose(mp.linalg.tensorinv(t), np.linalg.tensorinv(t), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.tensorsolve(t, b), np.linalg.tensorsolve(t, b), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.det(a), np.linalg.det(a), rtol=1e-5, atol=1e-5)
    sgn, logabs = mp.linalg.slogdet(a)
    np_sgn, np_logabs = np.linalg.slogdet(a)
    assert_allclose(sgn, np_sgn, rtol=1e-6, atol=1e-6)
    assert_allclose(logabs, np_logabs, rtol=1e-6, atol=1e-6)
    assert_array_equal(mp.linalg.matrix_rank(a), np.array(np.linalg.matrix_rank(a)))
    assert_allclose(mp.linalg.cond(a), np.linalg.cond(a), rtol=1e-5, atol=1e-5)
    assert_allclose(
        mp.linalg.multi_dot([np.eye(2, dtype=np.float32), a, a]),
        np.linalg.multi_dot([np.eye(2, dtype=np.float32), a, a]),
        rtol=1e-5,
        atol=1e-5,
    )
    x_sol, residuals, rank, svals = mp.linalg.lstsq(
        np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float32),
        np.array([1.0, 2.0, 2.5], dtype=np.float32),
    )
    np_x, np_res, np_rank, np_s = np.linalg.lstsq(
        np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float32),
        np.array([1.0, 2.0, 2.5], dtype=np.float32),
        rcond=None,
    )
    assert_allclose(x_sol, np_x, rtol=1e-5, atol=1e-5)
    assert_allclose(residuals, np_res, rtol=1e-5, atol=1e-5)
    assert_array_equal(rank, np.array(np_rank))
    assert_allclose(svals, np_s, rtol=1e-5, atol=1e-5)
    wide = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    q_c, r_c = mp.linalg.qr(wide, mode="complete")
    np_qc, np_rc = np.linalg.qr(wide, mode="complete")
    assert_allclose(q_c, np_qc, rtol=1e-5, atol=1e-5)
    assert_allclose(r_c, np_rc, rtol=1e-5, atol=1e-5)
    tall = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    u_r, s_r, vt_r = mp.linalg.svd(tall, full_matrices=False)
    np_ur, np_sr, np_vtr = np.linalg.svd(tall, full_matrices=False)
    assert_allclose(s_r, np_sr, rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(u_r)), np.abs(np_ur), rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(vt_r)), np.abs(np_vtr), rtol=1e-5, atol=1e-5)
    assert_allclose(to_numpy(u_r) @ np.diag(to_numpy(s_r)) @ to_numpy(vt_r), tall, rtol=1e-5, atol=1e-5)
    hu, hs, hvt = mp.linalg.svd(a, hermitian=True)
    np_hu, np_hs, np_hvt = np.linalg.svd(a, hermitian=True)
    assert_allclose(hs, np_hs, rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(hu)), np.abs(np_hu), rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(hvt)), np.abs(np_hvt), rtol=1e-5, atol=1e-5)

    hfreq = np.array([1.0 + 0.0j, 2.0 + 1.0j, 0.5 + 0.0j], dtype=np.complex64)
    assert_allclose(mp.fft.hfft(hfreq), np.fft.hfft(hfreq), rtol=1e-5, atol=1e-5)
    full_real = np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32)
    assert_allclose(mp.fft.ihfft(full_real), np.fft.ihfft(full_real), rtol=1e-5, atol=1e-5)


def test_random_mlx_native_ports_reproducible_and_avoid_numpy(monkeypatch):
    # Reproducibility first (before monkeypatching seed's NumPy sync path).
    mp.random.seed(314)
    mod1 = {
        "gumbel": mp.random.gumbel(loc=1.0, scale=2.0, size=6),
        "exponential": mp.random.exponential(scale=1.5, size=6),
        "stdexp": mp.random.standard_exponential(size=6),
        "stdgamma": mp.random.standard_gamma(shape=2.0, size=6),
        "gamma": mp.random.gamma(shape=2.0, scale=1.5, size=6),
        "chisq": mp.random.chisquare(df=4.0, size=6),
        "f": mp.random.f(dfnum=5.0, dfden=7.0, size=6),
        "beta": mp.random.beta(a=2.0, b=3.0, size=6),
        "dirichlet": mp.random.dirichlet([1.0, 2.0, 3.0], size=4),
        "t": mp.random.standard_t(df=5.0, size=6),
        "multi": mp.random.multinomial(5, [0.2, 0.3, 0.5], size=4),
        "nbinom": mp.random.negative_binomial(n=3.0, p=0.4, size=6),
        "rayleigh": mp.random.rayleigh(scale=2.0, size=6),
        "logistic": mp.random.logistic(loc=0.0, scale=1.0, size=6),
        "lognormal": mp.random.lognormal(mean=0.2, sigma=0.7, size=6),
        "cauchy": mp.random.standard_cauchy(size=6),
        "tri": mp.random.triangular(left=-1.0, mode=0.5, right=3.0, size=6),
        "weibull": mp.random.weibull(a=1.5, size=6),
        "power": mp.random.power(a=2.5, size=6),
        "pareto": mp.random.pareto(a=3.0, size=6),
        "geometric": mp.random.geometric(p=0.3, size=6),
        "poisson": mp.random.poisson(lam=2.5, size=6),
        "binomial": mp.random.binomial(n=8, p=0.35, size=6),
        "choice": mp.random.choice([10, 20, 30], size=6, replace=True, p=[0.2, 0.3, 0.5]),
        "permuted": mp.random.permuted(np.arange(12).reshape(3, 4), axis=1),
        "bytes": mp.random.bytes(12),
    }
    mp.random.seed(314)
    mod2 = {
        "gumbel": mp.random.gumbel(loc=1.0, scale=2.0, size=6),
        "exponential": mp.random.exponential(scale=1.5, size=6),
        "stdexp": mp.random.standard_exponential(size=6),
        "stdgamma": mp.random.standard_gamma(shape=2.0, size=6),
        "gamma": mp.random.gamma(shape=2.0, scale=1.5, size=6),
        "chisq": mp.random.chisquare(df=4.0, size=6),
        "f": mp.random.f(dfnum=5.0, dfden=7.0, size=6),
        "beta": mp.random.beta(a=2.0, b=3.0, size=6),
        "dirichlet": mp.random.dirichlet([1.0, 2.0, 3.0], size=4),
        "t": mp.random.standard_t(df=5.0, size=6),
        "multi": mp.random.multinomial(5, [0.2, 0.3, 0.5], size=4),
        "nbinom": mp.random.negative_binomial(n=3.0, p=0.4, size=6),
        "rayleigh": mp.random.rayleigh(scale=2.0, size=6),
        "logistic": mp.random.logistic(loc=0.0, scale=1.0, size=6),
        "lognormal": mp.random.lognormal(mean=0.2, sigma=0.7, size=6),
        "cauchy": mp.random.standard_cauchy(size=6),
        "tri": mp.random.triangular(left=-1.0, mode=0.5, right=3.0, size=6),
        "weibull": mp.random.weibull(a=1.5, size=6),
        "power": mp.random.power(a=2.5, size=6),
        "pareto": mp.random.pareto(a=3.0, size=6),
        "geometric": mp.random.geometric(p=0.3, size=6),
        "poisson": mp.random.poisson(lam=2.5, size=6),
        "binomial": mp.random.binomial(n=8, p=0.35, size=6),
        "choice": mp.random.choice([10, 20, 30], size=6, replace=True, p=[0.2, 0.3, 0.5]),
        "permuted": mp.random.permuted(np.arange(12).reshape(3, 4), axis=1),
        "bytes": mp.random.bytes(12),
    }
    for key in (
        "gumbel",
        "exponential",
        "stdexp",
        "stdgamma",
        "gamma",
        "chisq",
        "f",
        "beta",
        "dirichlet",
        "t",
        "multi",
        "nbinom",
        "rayleigh",
        "logistic",
        "lognormal",
        "cauchy",
        "tri",
        "weibull",
        "power",
        "pareto",
        "geometric",
        "poisson",
        "binomial",
        "choice",
        "permuted",
    ):
        assert_array_equal(mod1[key], mod2[key])
    assert mod1["bytes"] == mod2["bytes"]

    g1 = mp.random.default_rng(99)
    g2 = mp.random.default_rng(99)
    for fn, kwargs in [
        ("gumbel", {"loc": 0.0, "scale": 1.0, "size": 5}),
        ("exponential", {"scale": 2.0, "size": 5}),
        ("standard_exponential", {"size": 5}),
        ("standard_gamma", {"shape": 2.0, "size": 5}),
        ("gamma", {"shape": 2.0, "scale": 1.5, "size": 5}),
        ("chisquare", {"df": 4.0, "size": 5}),
        ("f", {"dfnum": 5.0, "dfden": 7.0, "size": 5}),
        ("beta", {"a": 2.0, "b": 3.0, "size": 5}),
        ("dirichlet", {"alpha": [1.0, 2.0, 3.0], "size": 3}),
        ("standard_t", {"df": 5.0, "size": 5}),
        ("multinomial", {"n": 5, "pvals": [0.2, 0.3, 0.5], "size": 3}),
        ("negative_binomial", {"n": 3.0, "p": 0.4, "size": 5}),
        ("rayleigh", {"scale": 1.5, "size": 5}),
        ("logistic", {"loc": 1.0, "scale": 0.5, "size": 5}),
        ("lognormal", {"mean": 0.1, "sigma": 0.8, "size": 5}),
        ("standard_cauchy", {"size": 5}),
        ("triangular", {"left": -2.0, "mode": 0.0, "right": 3.0, "size": 5}),
        ("weibull", {"a": 1.7, "size": 5}),
        ("power", {"a": 2.2, "size": 5}),
        ("pareto", {"a": 2.8, "size": 5}),
        ("geometric", {"p": 0.4, "size": 5}),
        ("poisson", {"lam": 2.5, "size": 5}),
        ("binomial", {"n": 8, "p": 0.35, "size": 5}),
        ("choice", {"a": [10, 20, 30], "size": 5, "p": [0.2, 0.3, 0.5]}),
        ("permuted", {"x": np.arange(12).reshape(3, 4), "axis": 1}),
    ]:
        assert_array_equal(getattr(g1, fn)(**kwargs), getattr(g2, fn)(**kwargs))
    assert g1.bytes(10) == g2.bytes(10)

    # Basic sanity/range checks.
    assert np.all(to_numpy(mod1["exponential"]) >= 0)
    assert np.all(to_numpy(mod1["stdexp"]) >= 0)
    assert np.all(to_numpy(mod1["stdgamma"]) >= 0)
    assert np.all(to_numpy(mod1["gamma"]) >= 0)
    assert np.all(to_numpy(mod1["chisq"]) >= 0)
    assert np.all(to_numpy(mod1["f"]) >= 0)
    assert np.all((to_numpy(mod1["beta"]) >= 0) & (to_numpy(mod1["beta"]) <= 1))
    assert_allclose(to_numpy(mod1["dirichlet"]).sum(axis=1), np.ones(4, dtype=np.float32), rtol=1e-5, atol=1e-5)
    assert np.all(to_numpy(mod1["dirichlet"]) > 0)
    assert_array_equal(to_numpy(mod1["multi"]).sum(axis=1), np.array([5, 5, 5, 5]))
    assert np.all(to_numpy(mod1["nbinom"]) >= 0)
    assert np.all(to_numpy(mod1["rayleigh"]) >= 0)
    assert np.all(to_numpy(mod1["weibull"]) >= 0)
    assert np.all(to_numpy(mod1["power"]) >= 0)
    assert np.all(to_numpy(mod1["pareto"]) >= 0)
    assert np.all((to_numpy(mod1["tri"]) >= -1.0) & (to_numpy(mod1["tri"]) <= 3.0))
    assert np.all(to_numpy(mod1["geometric"]) >= 1)
    assert np.all(to_numpy(mod1["poisson"]) >= 0)
    assert np.all((to_numpy(mod1["binomial"]) >= 0) & (to_numpy(mod1["binomial"]) <= 8))
    assert set(to_numpy(mod1["choice"]).tolist()).issubset({10, 20, 30})
    assert_array_equal(np.sort(to_numpy(mod1["permuted"]), axis=1), np.sort(np.arange(12).reshape(3, 4), axis=1))
    assert isinstance(mod1["bytes"], (bytes, bytearray))
    assert len(mod1["bytes"]) == 12

    # Assert no NumPy fallback is used for these methods.
    mp.random.seed(2718)
    monkeypatch.setattr(mp_bridge, "numpy_module", _fail_numpy)
    monkeypatch.setattr(mp.random, "_global_numpy_rng", _fail_numpy)
    monkeypatch.setattr(mp.random, "_numpy_rng_from_key", _fail_numpy)
    _ = mp.random.gumbel(size=4)
    _ = mp.random.exponential(size=4)
    _ = mp.random.standard_exponential(size=4)
    _ = mp.random.rayleigh(size=4)
    _ = mp.random.logistic(size=4)
    _ = mp.random.lognormal(size=4)
    _ = mp.random.standard_cauchy(size=4)
    _ = mp.random.triangular(size=4)
    _ = mp.random.standard_gamma(shape=2.0, size=4)
    _ = mp.random.gamma(shape=2.0, scale=1.5, size=4)
    _ = mp.random.chisquare(df=4.0, size=4)
    _ = mp.random.f(dfnum=5.0, dfden=7.0, size=4)
    _ = mp.random.beta(a=2.0, b=3.0, size=4)
    _ = mp.random.dirichlet([1.0, 2.0, 3.0], size=3)
    _ = mp.random.standard_t(df=5.0, size=4)
    _ = mp.random.multinomial(5, [0.2, 0.3, 0.5], size=3)
    _ = mp.random.negative_binomial(n=3.0, p=0.4, size=4)
    _ = mp.random.weibull(a=1.2, size=4)
    _ = mp.random.power(a=2.0, size=4)
    _ = mp.random.pareto(a=3.0, size=4)
    _ = mp.random.geometric(p=0.5, size=4)
    _ = mp.random.poisson(lam=2.5, size=4)
    _ = mp.random.binomial(n=8, p=0.35, size=4)
    _ = mp.random.poisson(lam=120.0, size=4)
    _ = mp.random.binomial(n=500, p=0.35, size=4)
    _ = mp.random.choice([1, 2, 3], size=4, p=[0.2, 0.3, 0.5])
    _ = mp.random.permuted(np.arange(12).reshape(3, 4), axis=1)
    _ = mp.random.bytes(4)

    g = mp.random.default_rng(11)
    _ = g.gumbel(size=4)
    _ = g.exponential(size=4)
    _ = g.standard_exponential(size=4)
    _ = g.rayleigh(size=4)
    _ = g.logistic(size=4)
    _ = g.lognormal(size=4)
    _ = g.standard_cauchy(size=4)
    _ = g.triangular(size=4)
    _ = g.standard_gamma(shape=2.0, size=4)
    _ = g.gamma(shape=2.0, scale=1.5, size=4)
    _ = g.chisquare(df=4.0, size=4)
    _ = g.f(dfnum=5.0, dfden=7.0, size=4)
    _ = g.beta(a=2.0, b=3.0, size=4)
    _ = g.dirichlet([1.0, 2.0, 3.0], size=3)
    _ = g.standard_t(df=5.0, size=4)
    _ = g.multinomial(5, [0.2, 0.3, 0.5], size=3)
    _ = g.negative_binomial(n=3.0, p=0.4, size=4)
    _ = g.weibull(a=1.2, size=4)
    _ = g.power(a=2.0, size=4)
    _ = g.pareto(a=3.0, size=4)
    _ = g.geometric(p=0.5, size=4)
    _ = g.poisson(lam=2.5, size=4)
    _ = g.binomial(n=8, p=0.35, size=4)
    _ = g.poisson(lam=120.0, size=4)
    _ = g.binomial(n=500, p=0.35, size=4)
    _ = g.choice([1, 2, 3], size=4, p=[0.2, 0.3, 0.5])
    _ = g.permuted(np.arange(12).reshape(3, 4), axis=1)
    _ = g.bytes(4)
