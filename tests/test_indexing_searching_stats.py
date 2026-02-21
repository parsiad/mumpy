"""Tests for indexing, searching, and histogram/search helpers."""

import numpy as np

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_append_insert_and_delete_match_numpy():
    a_np = np.array([1, 2, 3], dtype=np.int32)
    b_np = np.array([4, 5], dtype=np.int32)
    assert_array_equal(mp.append(a_np, b_np), np.append(a_np, b_np))
    assert_array_equal(mp.insert(a_np, 1, [9, 8]), np.insert(a_np, 1, [9, 8]))
    assert_array_equal(mp.delete(a_np, [0, 2]), np.delete(a_np, [0, 2]))

    m_np = np.arange(6).reshape(2, 3)
    assert_array_equal(mp.append(m_np, [[6, 7, 8]], axis=0), np.append(m_np, [[6, 7, 8]], axis=0))
    assert_array_equal(mp.insert(m_np, 1, [99, 98], axis=1), np.insert(m_np, 1, [99, 98], axis=1))
    assert_array_equal(mp.delete(m_np, 1, axis=0), np.delete(m_np, 1, axis=0))


def test_bincount_digitize_histogram_and_interp():
    x_np = np.array([0, 1, 1, 2, 4, 4, 4], dtype=np.int32)
    w_np = np.array([1.0, 0.5, 1.5, 1.0, 2.0, 1.0, 0.5], dtype=np.float32)
    assert_array_equal(mp.bincount(x_np), np.bincount(x_np))
    assert_allclose(mp.bincount(x_np, weights=w_np), np.bincount(x_np, weights=w_np), rtol=1e-6, atol=1e-6)
    assert_array_equal(mp.bincount(x_np, minlength=8), np.bincount(x_np, minlength=8))

    bins_np = np.array([0.0, 1.0, 2.5, 5.0], dtype=np.float32)
    vals_np = np.array([-1.0, 0.5, 1.0, 2.0, 4.5, 5.0], dtype=np.float32)
    assert_array_equal(mp.digitize(vals_np, bins_np), np.digitize(vals_np, bins_np))
    assert_array_equal(mp.digitize(vals_np, bins_np, right=True), np.digitize(vals_np, bins_np, right=True))

    data_np = np.array([0.1, 0.2, 0.8, 1.5, 2.2, 2.8], dtype=np.float32)
    h, edges = mp.histogram(data_np, bins=3, range=(0.0, 3.0))
    nh, nedges = np.histogram(data_np, bins=3, range=(0.0, 3.0))
    assert_array_equal(h, nh)
    assert_allclose(edges, nedges, rtol=1e-6, atol=1e-6)

    hd, ed = mp.histogram(data_np, bins=[0.0, 1.0, 2.0, 3.0], density=True)
    nhd, ned = np.histogram(data_np, bins=[0.0, 1.0, 2.0, 3.0], density=True)
    assert_allclose(hd, nhd, rtol=1e-6, atol=1e-6)
    assert_allclose(ed, ned, rtol=1e-6, atol=1e-6)

    xp = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    fp = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    xq = np.array([-1.0, 0.25, 1.5, 3.0], dtype=np.float32)
    assert_allclose(mp.interp(xq, xp, fp), np.interp(xq, xp, fp), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.interp(xq, xp, fp, left=-5.0, right=99.0), np.interp(xq, xp, fp, left=-5.0, right=99.0))


def test_indices_ix_and_ravel_unravel_helpers():
    dense = mp.indices((2, 3), dtype=mp.int32)
    dense_np = np.indices((2, 3), dtype=np.int32)
    assert_array_equal(dense, dense_np)

    sparse = mp.indices((2, 3, 4), sparse=True)
    sparse_np = np.indices((2, 3, 4), sparse=True)
    assert len(sparse) == len(sparse_np) == 3
    for got, exp in zip(sparse, sparse_np, strict=False):
        assert_array_equal(got, exp)

    ix = mp.ix_([0, 2], [1, 3, 4])
    ix_np = np.ix_([0, 2], [1, 3, 4])
    for got, exp in zip(ix, ix_np, strict=False):
        assert_array_equal(got, exp)

    ur = mp.unravel_index([1, 5, 7], (2, 2, 2))
    ur_np = np.unravel_index([1, 5, 7], (2, 2, 2))
    for got, exp in zip(ur, ur_np, strict=False):
        assert_array_equal(got, exp)

    flat = mp.ravel_multi_index(([0, 1, 1], [1, 0, 1]), (2, 3))
    flat_np = np.ravel_multi_index(([0, 1, 1], [1, 0, 1]), (2, 3))
    assert_array_equal(flat, flat_np)


def test_numpy_dtype_inputs_are_accepted():
    arr = mp.array([1, 2, 3], dtype=np.float64)
    assert str(arr.dtype) == "mlx.core.float64"
    idx = mp.indices((2, 2), dtype=np.int64)
    assert str(idx.dtype) == "mlx.core.int64"
