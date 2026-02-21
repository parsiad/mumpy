"""Tests for selection, sorting, and piecewise extras."""

import numpy as np

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_partition_argpartition_lexsort_and_sort_complex():
    x_np = np.array([7, 2, 9, 1, 5, 4], dtype=np.int32)
    kth = 2
    part_mp = np.asarray(mp.partition(x_np, kth))
    part_np = np.partition(x_np, kth)
    np.testing.assert_array_equal(np.sort(part_mp), np.sort(part_np))
    assert part_mp[kth] == part_np[kth]
    assert np.all(part_mp[:kth] <= part_mp[kth])
    assert np.all(part_mp[kth + 1 :] >= part_mp[kth])

    argp_mp = np.asarray(mp.argpartition(x_np, kth))
    argp_np = np.argpartition(x_np, kth)
    np.testing.assert_array_equal(np.sort(argp_mp), np.arange(x_np.shape[0]))
    vals_mp = x_np[argp_mp]
    vals_np = x_np[argp_np]
    assert vals_mp[kth] == vals_np[kth]
    assert np.all(vals_mp[:kth] <= vals_mp[kth])
    assert np.all(vals_mp[kth + 1 :] >= vals_mp[kth])

    keys = (np.array([1, 1, 0, 0]), np.array([3, 2, 1, 0]))
    assert_array_equal(mp.lexsort(keys), np.lexsort(keys))

    c_np = np.array([3 + 1j, 1 + 3j, 1 + 2j, 2 + 0j], dtype=np.complex64)
    assert_array_equal(mp.sort_complex(c_np), np.sort_complex(c_np))


def test_nanarg_compress_extract_and_take_along_axis():
    x_np = np.array([[np.nan, 2.0, 1.0], [3.0, np.nan, 0.0]], dtype=np.float32)
    assert_array_equal(mp.nanargmin(x_np, axis=1), np.nanargmin(x_np, axis=1))
    assert_array_equal(mp.nanargmax(x_np, axis=0), np.nanargmax(x_np, axis=0))

    ints_np = np.arange(10)
    cond_np = ints_np % 2 == 0
    assert_array_equal(mp.compress(cond_np, ints_np), np.compress(cond_np, ints_np))
    assert_array_equal(mp.extract(cond_np, ints_np), np.extract(cond_np, ints_np))

    arr_np = np.array([[10, 30, 20], [40, 60, 50]], dtype=np.int32)
    idx_np = np.array([[2, 0], [1, 2]], dtype=np.int32)
    assert_array_equal(mp.take_along_axis(arr_np, idx_np, axis=1), np.take_along_axis(arr_np, idx_np, axis=1))


def test_choose_select_and_piecewise_match_numpy():
    a_np = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    choices = [
        np.array([10, 10, 10, 10, 10]),
        np.array([20, 20, 20, 20, 20]),
        np.array([30, 30, 30, 30, 30]),
    ]
    assert_array_equal(mp.choose(a_np, choices), np.choose(a_np, choices))

    x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    conds = [x_np < 0, x_np > 0]
    choices = [np.array([-1.0] * 5, dtype=np.float32), np.array([1.0] * 5, dtype=np.float32)]
    assert_array_equal(mp.select(conds, choices, default=0.0), np.select(conds, choices, default=0.0))

    pw_mp = mp.piecewise(x_np, [x_np < 0, x_np >= 0], [lambda z: z**2, lambda z: z + 1])
    pw_np = np.piecewise(x_np, [x_np < 0, x_np >= 0], [lambda z: z**2, lambda z: z + 1])
    assert_allclose(pw_mp, pw_np, rtol=1e-6, atol=1e-6)
