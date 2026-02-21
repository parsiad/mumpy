"""Tests for statistics integration helpers and histogram APIs."""

import numpy as np

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal


def test_average_cov_and_corrcoef_match_numpy():
    x_np = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]], dtype=np.float32)
    w_np = np.array([1.0, 2.0, 1.0], dtype=np.float32)

    assert_allclose(mp.average(x_np, axis=1), np.average(x_np, axis=1), rtol=1e-6, atol=1e-6)
    assert_allclose(
        mp.average(x_np, axis=1, weights=w_np),
        np.average(x_np, axis=1, weights=w_np),
        rtol=1e-6,
        atol=1e-6,
    )

    avg, sw = mp.average(x_np, axis=1, weights=w_np, returned=True, keepdims=True)
    np_avg, np_sw = np.average(x_np, axis=1, weights=w_np, returned=True, keepdims=True)
    assert_allclose(avg, np_avg, rtol=1e-6, atol=1e-6)
    assert_allclose(sw, np_sw, rtol=1e-6, atol=1e-6)

    v1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    assert_allclose(mp.cov(v1, v2), np.cov(v1, v2), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.corrcoef(v1, v2), np.corrcoef(v1, v2), rtol=1e-6, atol=1e-6)


def test_nanquantile_and_nanpercentile_match_numpy():
    x_np = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    assert_allclose(mp.nanquantile(x_np, 0.5), np.nanquantile(x_np, 0.5), rtol=1e-6, atol=1e-6)
    assert_allclose(
        mp.nanquantile(x_np, [0.25, 0.75], axis=1, keepdims=True),
        np.nanquantile(x_np, [0.25, 0.75], axis=1, keepdims=True),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        mp.nanpercentile(x_np, [25, 75], axis=0),
        np.nanpercentile(x_np, [25, 75], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gradient_trapezoid_and_trapz_match_numpy():
    y_np = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float32)
    x_np = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    assert_allclose(mp.trapezoid(y_np, x=x_np), np.trapezoid(y_np, x=x_np), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.trapz(y_np, dx=0.5), np.trapezoid(y_np, dx=0.5), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.gradient(y_np), np.gradient(y_np), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.gradient(y_np, x_np), np.gradient(y_np, x_np), rtol=1e-6, atol=1e-6)

    z_np = np.arange(9, dtype=np.float32).reshape(3, 3)
    g_mp = mp.gradient(z_np, axis=(0, 1))
    g_np = np.gradient(z_np, axis=(0, 1))
    assert isinstance(g_mp, list)
    for gm, gn in zip(g_mp, g_np, strict=False):
        assert_allclose(gm, gn, rtol=1e-6, atol=1e-6)


def test_histogram2d_and_histogramdd_match_numpy():
    x = np.array([0.1, 0.2, 0.8, 1.5, 1.8, 2.2], dtype=np.float32)
    y = np.array([0.0, 0.7, 1.1, 1.4, 2.1, 2.9], dtype=np.float32)
    h, xe, ye = mp.histogram2d(x, y, bins=3, range=((0.0, 3.0), (0.0, 3.0)))
    nh, nxe, nye = np.histogram2d(x, y, bins=3, range=((0.0, 3.0), (0.0, 3.0)))
    assert_array_equal(h, nh)
    assert_allclose(xe, nxe, rtol=1e-6, atol=1e-6)
    assert_allclose(ye, nye, rtol=1e-6, atol=1e-6)

    sample = np.stack([x, y, np.array([0.5, 0.5, 0.2, 1.2, 1.7, 2.8], dtype=np.float32)], axis=1)
    hd, edges = mp.histogramdd(sample, bins=(2, 2, 2), range=((0.0, 3.0), (0.0, 3.0), (0.0, 3.0)))
    nhd, nedges = np.histogramdd(sample, bins=(2, 2, 2), range=((0.0, 3.0), (0.0, 3.0), (0.0, 3.0)))
    assert_array_equal(hd, nhd)
    assert len(edges) == len(nedges) == 3
    for got, exp in zip(edges, nedges, strict=False):
        assert_allclose(got, exp, rtol=1e-6, atol=1e-6)
