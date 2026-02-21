"""Tests for index helpers and extra linalg compatibility functions."""

import numpy as np

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_diagonal_and_triangular_index_helpers_match_numpy():
    di = mp.diag_indices(3)
    ndi = np.diag_indices(3)
    for got, exp in zip(di, ndi, strict=False):
        assert_array_equal(got, exp)

    arr_np = np.arange(27).reshape(3, 3, 3)
    di_from = mp.diag_indices_from(arr_np)
    ndi_from = np.diag_indices_from(arr_np)
    for got, exp in zip(di_from, ndi_from, strict=False):
        assert_array_equal(got, exp)

    tri_np = np.arange(12).reshape(3, 4)
    for mp_idx, np_idx in [
        (mp.triu_indices(3, k=1, m=4), np.triu_indices(3, k=1, m=4)),
        (mp.tril_indices(3, k=0, m=4), np.tril_indices(3, k=0, m=4)),
        (mp.triu_indices_from(tri_np, k=-1), np.triu_indices_from(tri_np, k=-1)),
        (mp.tril_indices_from(tri_np, k=1), np.tril_indices_from(tri_np, k=1)),
    ]:
        for got, exp in zip(mp_idx, np_idx, strict=False):
            assert_array_equal(got, exp)


def test_linalg_cond_and_multi_dot_match_numpy():
    a = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    assert_allclose(mp.linalg.cond(a), np.linalg.cond(a), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.cond(a, p=1), np.linalg.cond(a, p=1), rtol=1e-5, atol=1e-5)

    arrays = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        np.array([[1.0], [2.0]], dtype=np.float32),
    ]
    assert_allclose(mp.linalg.multi_dot(arrays), np.linalg.multi_dot(arrays), rtol=1e-5, atol=1e-5)
