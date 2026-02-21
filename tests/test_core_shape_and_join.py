"""Tests for shape manipulation, joins, and padding behavior."""

import numpy as np
import pytest

import mumpy as mp

from .conftest import assert_array_equal


def test_reshape_transpose_and_axis_permutations():
    x = mp.arange(24).reshape(2, 3, 4)
    assert_array_equal(mp.reshape(x, (4, 6)), np.arange(24).reshape(4, 6))
    assert_array_equal(mp.transpose(x, (2, 0, 1)), np.transpose(np.arange(24).reshape(2, 3, 4), (2, 0, 1)))
    assert_array_equal(mp.swapaxes(x, 0, 2), np.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2))
    assert_array_equal(mp.moveaxis(x, 0, -1), np.moveaxis(np.arange(24).reshape(2, 3, 4), 0, -1))
    assert_array_equal(
        mp.moveaxis(x, (0, 2), (2, 0)),
        np.moveaxis(np.arange(24).reshape(2, 3, 4), (0, 2), (2, 0)),
    )


def test_expand_squeeze_and_flatten_variants():
    x = mp.array([1, 2, 3])
    assert_array_equal(mp.expand_dims(x, 0), np.expand_dims(np.array([1, 2, 3]), 0))
    assert_array_equal(mp.expand_dims(x, (0, 2)), np.expand_dims(np.expand_dims(np.array([1, 2, 3]), 0), 2))

    y = mp.array([[[1], [2], [3]]])
    assert_array_equal(mp.squeeze(y), np.squeeze(np.array([[[1], [2], [3]]])))
    assert_array_equal(mp.squeeze(y, axis=(0, 2)), np.squeeze(np.array([[[1], [2], [3]]]), axis=(0, 2)))

    z = mp.arange(6).reshape(2, 3)
    assert_array_equal(mp.flatten(z), np.arange(6).reshape(2, 3).flatten())
    assert_array_equal(mp.ravel(z), np.arange(6).reshape(2, 3).ravel())


def test_atleast_nd_helpers():
    assert_array_equal(mp.atleast_1d(5), np.atleast_1d(5))
    assert_array_equal(mp.atleast_2d([1, 2, 3]), np.atleast_2d([1, 2, 3]))
    assert_array_equal(mp.atleast_3d([1, 2, 3]), np.atleast_3d([1, 2, 3]))

    a, b = mp.atleast_2d([1, 2], [[3, 4]])
    assert_array_equal(a, np.atleast_2d([1, 2]))
    assert_array_equal(b, np.atleast_2d([[3, 4]]))


def test_concatenate_stack_and_column_row_helpers():
    a = mp.array([1, 2])
    b = mp.array([3, 4])
    assert_array_equal(mp.concatenate([a, b]), np.concatenate([np.array([1, 2]), np.array([3, 4])]))
    assert_array_equal(
        mp.concatenate([a.reshape(1, 2), b.reshape(1, 2)], axis=None),
        np.concatenate([np.array([[1, 2]]), np.array([[3, 4]])], axis=None),
    )
    assert_array_equal(mp.stack([a, b]), np.stack([np.array([1, 2]), np.array([3, 4])]))
    assert_array_equal(mp.hstack([a, b]), np.hstack([np.array([1, 2]), np.array([3, 4])]))
    assert_array_equal(mp.vstack([a, b]), np.vstack([np.array([1, 2]), np.array([3, 4])]))
    assert_array_equal(mp.row_stack([a, b]), np.vstack([np.array([1, 2]), np.array([3, 4])]))
    assert_array_equal(mp.dstack([a, b]), np.dstack([np.array([1, 2]), np.array([3, 4])]))
    assert_array_equal(mp.column_stack([a, b]), np.column_stack([np.array([1, 2]), np.array([3, 4])]))


def test_concatenate_and_stack_dtype_keyword():
    a = mp.array([1, 2], dtype=mp.int32)
    b = mp.array([3, 4], dtype=mp.int32)

    concat = mp.concatenate([a, b], dtype=mp.float32)
    expected_concat = np.concatenate(
        [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
        dtype=np.float32,
    )
    assert_array_equal(concat, expected_concat)
    assert np.asarray(concat).dtype == np.float32

    stack = mp.stack([a, b], dtype=mp.float32)
    expected_stack = np.stack(
        [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
        dtype=np.float32,
    )
    assert_array_equal(stack, expected_stack)
    assert np.asarray(stack).dtype == np.float32


def test_split_and_array_split_helpers():
    x = mp.arange(10)
    assert [v.tolist() for v in mp.split(x, 5)] == [v.tolist() for v in np.split(np.arange(10), 5)]
    assert [v.tolist() for v in mp.split(x, [3, 7])] == [v.tolist() for v in np.split(np.arange(10), [3, 7])]
    assert [v.tolist() for v in mp.array_split(x, 3)] == [v.tolist() for v in np.array_split(np.arange(10), 3)]

    y = mp.arange(12).reshape(3, 4)
    assert [v.tolist() for v in mp.hsplit(y, 2)] == [v.tolist() for v in np.hsplit(np.arange(12).reshape(3, 4), 2)]
    assert [v.tolist() for v in mp.vsplit(y, 3)] == [v.tolist() for v in np.vsplit(np.arange(12).reshape(3, 4), 3)]

    z = mp.arange(24).reshape(2, 3, 4)
    assert [v.tolist() for v in mp.dsplit(z, 2)] == [v.tolist() for v in np.dsplit(np.arange(24).reshape(2, 3, 4), 2)]

    with pytest.raises(ValueError, match="vsplit only works"):
        mp.vsplit(mp.arange(3), 1)
    with pytest.raises(ValueError, match="dsplit only works"):
        mp.dsplit(mp.arange(6).reshape(2, 3), 1)


def test_broadcast_repeat_tile_and_take():
    b1, b2 = mp.broadcast_arrays(mp.array([1, 2, 3]), mp.array([[10], [20]]))
    n1, n2 = np.broadcast_arrays(np.array([1, 2, 3]), np.array([[10], [20]]))
    assert_array_equal(b1, n1)
    assert_array_equal(b2, n2)

    assert_array_equal(mp.broadcast_to([1, 2, 3], (2, 3)), np.broadcast_to(np.array([1, 2, 3]), (2, 3)))
    assert_array_equal(mp.repeat([[1, 2], [3, 4]], 2, axis=1), np.repeat(np.array([[1, 2], [3, 4]]), 2, axis=1))
    assert_array_equal(mp.tile([[1, 2]], (2, 3)), np.tile(np.array([[1, 2]]), (2, 3)))
    assert_array_equal(
        mp.take(mp.arange(6).reshape(2, 3), [2, 0], axis=1),
        np.take(np.arange(6).reshape(2, 3), [2, 0], axis=1),
    )


def test_where_clip_pad_roll_flip_rotations():
    cond = mp.array([[True, False], [False, True]])
    x = mp.array([[1, 2], [3, 4]])
    y = mp.array([[10, 20], [30, 40]])
    assert_array_equal(mp.where(cond, x, y), np.where(np.array(cond), np.array(x), np.array(y)))
    assert_array_equal(mp.where(cond, [1, 2], [10, 20]), np.where(np.array(cond), [1, 2], [10, 20]))
    assert_array_equal(mp.clip([-1, 0.5, 3], 0, 1), np.clip(np.array([-1, 0.5, 3]), 0, 1))

    base = np.arange(6).reshape(2, 3)
    arr = mp.array(base)
    assert_array_equal(mp.roll(arr, 1, axis=1), np.roll(base, 1, axis=1))
    assert_array_equal(mp.flip(arr), np.flip(base))
    assert_array_equal(mp.flip(arr, axis=0), np.flip(base, axis=0))
    assert_array_equal(mp.fliplr(arr), np.fliplr(base))
    assert_array_equal(mp.flipud(arr), np.flipud(base))
    assert_array_equal(mp.rot90(arr), np.rot90(base))
    assert_array_equal(mp.rot90(arr, 3), np.rot90(base, 3))

    assert_array_equal(mp.pad([1, 2, 3], (1, 2)), np.pad(np.array([1, 2, 3]), (1, 2)))
    assert_array_equal(
        mp.pad([[1, 2], [3, 4]], ((1, 1), (2, 0)), mode="edge"),
        np.pad(np.array([[1, 2], [3, 4]]), ((1, 1), (2, 0)), mode="edge"),
    )
    assert_array_equal(mp.pad([1, 2], 1, mode="reflect"), np.pad(np.array([1, 2]), 1, mode="reflect"))
    assert_array_equal(
        mp.pad(np.array([1, 2, 3], dtype=np.int32), (4, 5), mode="reflect"),
        np.pad(np.array([1, 2, 3], dtype=np.int32), (4, 5), mode="reflect"),
    )
    assert_array_equal(
        mp.pad(np.array([[1, 2], [3, 4]], dtype=np.int32), ((2, 1), (3, 2)), mode="symmetric"),
        np.pad(np.array([[1, 2], [3, 4]], dtype=np.int32), ((2, 1), (3, 2)), mode="symmetric"),
    )
    assert_array_equal(
        mp.pad(np.array([5], dtype=np.int32), 3, mode="reflect"),
        np.pad(np.array([5], dtype=np.int32), 3, mode="reflect"),
    )
    with pytest.raises(ValueError, match="can't extend empty axis"):
        mp.pad(np.array([], dtype=np.float32), 1, mode="reflect")
    with pytest.raises(ValueError, match="can't extend empty axis"):
        mp.pad(np.array([], dtype=np.float32), 1, mode="symmetric")


def test_diagonal_trace_triangular_and_meshgrid():
    x = np.arange(1, 10).reshape(3, 3)
    m = mp.array(x)
    assert_array_equal(mp.diagonal(m), np.diagonal(x))
    assert_array_equal(mp.trace(m), np.trace(x))
    assert_array_equal(mp.tril(m, k=-1), np.tril(x, k=-1))
    assert_array_equal(mp.triu(m, k=1), np.triu(x, k=1))

    gx, gy = mp.meshgrid(mp.arange(3), mp.arange(2), indexing="xy")
    nx, ny = np.meshgrid(np.arange(3), np.arange(2), indexing="xy")
    assert_array_equal(gx, nx)
    assert_array_equal(gy, ny)

    ix, iy = mp.meshgrid(mp.arange(3), mp.arange(2), indexing="ij", sparse=True)
    nix, niy = np.meshgrid(np.arange(3), np.arange(2), indexing="ij", sparse=True)
    assert_array_equal(ix, nix)
    assert_array_equal(iy, niy)


def test_meshgrid_copy_false_matches_numpy():
    gx, gy = mp.meshgrid(mp.arange(2), mp.arange(3), copy=False)
    nx, ny = np.meshgrid(np.arange(2), np.arange(3), copy=False)
    assert_array_equal(gx, nx)
    assert_array_equal(gy, ny)
