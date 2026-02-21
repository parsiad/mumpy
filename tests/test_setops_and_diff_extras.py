"""Tests for set operations and discrete difference helpers."""

import numpy as np

import mumpy as mp

from .conftest import assert_array_equal


def test_diff_prepend_append_and_ediff1d_match_numpy():
    x_np = np.array([[1, 4, 9], [16, 25, 36]], dtype=np.int32)
    assert_array_equal(mp.diff(x_np, axis=1, prepend=0), np.diff(x_np, axis=1, prepend=0))
    assert_array_equal(mp.diff(x_np, axis=1, append=100), np.diff(x_np, axis=1, append=100))
    assert_array_equal(mp.diff(x_np, axis=1, prepend=0, append=100), np.diff(x_np, axis=1, prepend=0, append=100))

    y_np = np.array([1, 2, 4, 7], dtype=np.int32)
    assert_array_equal(mp.ediff1d(y_np), np.ediff1d(y_np))
    assert_array_equal(
        mp.ediff1d(y_np, to_begin=[-1, -2], to_end=[9]),
        np.ediff1d(y_np, to_begin=[-1, -2], to_end=[9]),
    )


def test_set_operations_and_membership_helpers_match_numpy():
    a_np = np.array([1, 2, 2, 3, 5, 8], dtype=np.int32)
    b_np = np.array([2, 3, 4, 8, 13], dtype=np.int32)

    assert_array_equal(mp.in1d(a_np, b_np), np.isin(a_np.ravel(), b_np.ravel()).ravel())
    assert_array_equal(mp.isin(a_np.reshape(2, 3), b_np), np.isin(a_np.reshape(2, 3), b_np))
    assert_array_equal(mp.isin(a_np, b_np, invert=True), np.isin(a_np, b_np, invert=True))

    inter = mp.intersect1d(a_np, b_np)
    assert_array_equal(inter, np.intersect1d(a_np, b_np))

    vals, i1, i2 = mp.intersect1d(a_np, b_np, return_indices=True)
    np_vals, np_i1, np_i2 = np.intersect1d(a_np, b_np, return_indices=True)
    assert_array_equal(vals, np_vals)
    assert_array_equal(i1, np_i1)
    assert_array_equal(i2, np_i2)

    assert_array_equal(mp.union1d(a_np, b_np), np.union1d(a_np, b_np))
    assert_array_equal(mp.setdiff1d(a_np, b_np), np.setdiff1d(a_np, b_np))
    assert_array_equal(mp.setxor1d(a_np, b_np), np.setxor1d(a_np, b_np))
