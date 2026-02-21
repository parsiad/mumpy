"""Deterministic randomized differential tests against NumPy."""

import numpy as np

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal, to_numpy


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_randomized_elementwise_and_reductions_match_numpy():
    for seed in range(20):
        rng = _rng(seed)
        shape = tuple(int(x) for x in rng.integers(1, 5, size=2))
        a_np = rng.normal(size=shape).astype(np.float32)
        b_np = rng.normal(size=shape).astype(np.float32)
        a = mp.array(a_np)
        b = mp.array(b_np)

        assert_allclose(mp.add(a, b), a_np + b_np, rtol=1e-5, atol=1e-5)
        assert_allclose(mp.subtract(a, b), a_np - b_np, rtol=1e-5, atol=1e-5)
        assert_allclose(mp.multiply(a, b), a_np * b_np, rtol=1e-5, atol=1e-5)

        denom_np = b_np + 3.0
        assert_allclose(mp.divide(a, denom_np), a_np / denom_np, rtol=1e-5, atol=1e-5)
        assert_allclose(mp.maximum(a, b), np.maximum(a_np, b_np), rtol=1e-6, atol=1e-6)
        assert_allclose(mp.minimum(a, b), np.minimum(a_np, b_np), rtol=1e-6, atol=1e-6)
        assert_allclose(mp.clip(a, -0.5, 0.5), np.clip(a_np, -0.5, 0.5), rtol=1e-6, atol=1e-6)

        cond_np = a_np > b_np
        assert_array_equal(mp.where(cond_np, a_np, b_np), np.where(cond_np, a_np, b_np))

        axis = int(rng.integers(0, 2))
        assert_allclose(mp.sum(a, axis=axis), np.sum(a_np, axis=axis), rtol=1e-5, atol=1e-5)
        assert_allclose(mp.mean(a, axis=axis), np.mean(a_np, axis=axis), rtol=1e-5, atol=1e-5)
        assert_allclose(mp.std(a, axis=axis), np.std(a_np, axis=axis), rtol=2e-5, atol=2e-5)
        assert_allclose(mp.var(a, axis=axis), np.var(a_np, axis=axis), rtol=2e-5, atol=2e-5)
        assert_array_equal(mp.argmax(a, axis=axis), np.argmax(a_np, axis=axis))
        assert_array_equal(mp.argmin(a, axis=axis), np.argmin(a_np, axis=axis))

        q = rng.choice(np.array([0.25, 0.5, 0.75], dtype=np.float32), size=2, replace=False)
        assert_allclose(mp.quantile(a, q, axis=axis), np.quantile(a_np, q, axis=axis), rtol=1e-5, atol=1e-5)


def test_randomized_shape_join_and_indexing_helpers_match_numpy():
    for seed in range(20, 35):
        rng = _rng(seed)
        shape = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        x_np = rng.integers(-10, 10, size=shape, dtype=np.int32)
        x = mp.array(x_np)

        # Shape transforms
        assert_array_equal(mp.reshape(x, (shape[0] * shape[1],)), np.reshape(x_np, (shape[0] * shape[1],)))
        assert_array_equal(mp.transpose(x), np.transpose(x_np))
        assert_array_equal(mp.flip(x, axis=1), np.flip(x_np, axis=1))
        shift = int(rng.integers(-3, 4))
        assert_array_equal(mp.roll(x, shift, axis=0), np.roll(x_np, shift, axis=0))

        # Join/split round-trip
        top, bottom = mp.vsplit(x, 2) if shape[0] % 2 == 0 else (x[:1], x[1:])
        if shape[0] % 2 == 0:
            assert_array_equal(mp.vstack([top, bottom]), x_np)

        parts_mp = mp.array_split(x, 3, axis=1)
        parts_np = np.array_split(x_np, 3, axis=1)
        assert len(parts_mp) == len(parts_np)
        for pm, pn in zip(parts_mp, parts_np, strict=False):
            assert_array_equal(pm, pn)
        assert_array_equal(mp.concatenate(parts_mp, axis=1), np.concatenate(parts_np, axis=1))

        # Search/query helpers
        flat_np = np.sort(rng.integers(-5, 20, size=10, dtype=np.int32))
        probes_np = rng.integers(-5, 20, size=6, dtype=np.int32)
        assert_array_equal(mp.searchsorted(flat_np, probes_np), np.searchsorted(flat_np, probes_np))

        nonzero_mp = mp.nonzero(x)
        nonzero_np = np.nonzero(x_np)
        for got, exp in zip(nonzero_mp, nonzero_np, strict=False):
            assert_array_equal(got, exp)
        assert_array_equal(mp.argwhere(x), np.argwhere(x_np))

        # Index conversion helpers
        idx = rng.integers(0, shape[0] * shape[1], size=5, dtype=np.int32)
        ur_mp = mp.unravel_index(idx, shape)
        ur_np = np.unravel_index(idx, shape)
        for got, exp in zip(ur_mp, ur_np, strict=False):
            assert_array_equal(got, exp)
        assert_array_equal(mp.ravel_multi_index(ur_mp, shape), np.ravel_multi_index(ur_np, shape))


def test_randomized_linalg_and_fft_match_numpy():
    for seed in range(35, 45):
        rng = _rng(seed)
        n = int(rng.integers(2, 5))
        a0 = rng.normal(size=(n, n)).astype(np.float32)
        a_np = (a0 @ a0.T + np.eye(n, dtype=np.float32) * 0.5).astype(np.float32)
        b_np = rng.normal(size=(n,)).astype(np.float32)

        assert_allclose(mp.linalg.solve(a_np, b_np), np.linalg.solve(a_np, b_np), rtol=2e-4, atol=2e-4)
        assert_allclose(mp.linalg.inv(a_np), np.linalg.inv(a_np), rtol=2e-4, atol=2e-4)
        chol = mp.linalg.cholesky(a_np)
        assert_allclose(to_numpy(chol) @ to_numpy(chol).T, a_np, rtol=2e-4, atol=2e-4)
        assert_allclose(mp.linalg.det(a_np), np.linalg.det(a_np), rtol=2e-4, atol=2e-4)

        length = int(rng.integers(4, 12))
        x_np = rng.normal(size=(length,)).astype(np.float32)
        assert_allclose(mp.fft.fft(x_np), np.fft.fft(x_np), rtol=2e-4, atol=2e-4)
        assert_allclose(mp.fft.ifft(mp.fft.fft(x_np)), np.fft.ifft(np.fft.fft(x_np)), rtol=2e-4, atol=2e-4)
        assert_allclose(mp.fft.rfft(x_np), np.fft.rfft(x_np), rtol=2e-4, atol=2e-4)
        assert_allclose(
            mp.fft.irfft(mp.fft.rfft(x_np), n=length),
            np.fft.irfft(np.fft.rfft(x_np), n=length),
            rtol=2e-4,
            atol=2e-4,
        )


def test_randomized_histogram_interp_pad_tensordot_and_piecewise_match_numpy():
    for seed in range(45, 55):
        rng = _rng(seed)

        data = rng.normal(size=16).astype(np.float32)
        lo = float(np.min(data) - 0.25)
        hi = float(np.max(data) + 0.25)
        bins_n = int(rng.integers(2, 6))
        assert_allclose(
            mp.histogram(data, bins=bins_n, range=(lo, hi))[0],
            np.histogram(data, bins=bins_n, range=(lo, hi))[0],
            rtol=1e-6,
            atol=1e-6,
        )
        edges = np.sort(rng.uniform(lo, hi, size=5).astype(np.float32))
        if np.unique(edges).size < edges.size:
            edges = np.linspace(lo, hi, num=5, dtype=np.float32)
        h_mp, e_mp = mp.histogram(data, bins=edges, density=True)
        h_np, e_np = np.histogram(data, bins=edges, density=True)
        assert_allclose(h_mp, h_np, rtol=2e-5, atol=2e-5, equal_nan=True)
        assert_allclose(e_mp, e_np, rtol=1e-6, atol=1e-6)

        xp = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        fp = rng.normal(size=3).astype(np.float32)
        xq = rng.uniform(-3.0, 5.0, size=7).astype(np.float32)
        assert_allclose(mp.interp(xq, xp, fp, period=2.0), np.interp(xq, xp, fp, period=2.0), rtol=1e-6, atol=1e-6)

        pad_src = rng.integers(-5, 6, size=(2, 3), dtype=np.int32)
        pw = ((int(rng.integers(0, 4)), int(rng.integers(0, 4))), (int(rng.integers(0, 4)), int(rng.integers(0, 4))))
        assert_array_equal(mp.pad(pad_src, pw, mode="reflect"), np.pad(pad_src, pw, mode="reflect"))
        assert_array_equal(mp.pad(pad_src, pw, mode="symmetric"), np.pad(pad_src, pw, mode="symmetric"))

        a = rng.normal(size=(2, 3, 4)).astype(np.float32)
        b = rng.normal(size=(4, 5, 3)).astype(np.float32)
        assert_allclose(
            mp.tensordot(a, b, axes=([2, 1], [0, 2])),
            np.tensordot(a, b, axes=([2, 1], [0, 2])),
            rtol=1e-5,
            atol=1e-5,
        )

        x = rng.normal(size=6).astype(np.float32)
        conds = [x < -0.25, x > 0.25]
        assert_allclose(
            mp.piecewise(x, conds, [lambda z: z * z, lambda z: z + 1.0, -2.0]),
            np.piecewise(x, conds, [lambda z: z * z, lambda z: z + 1.0, -2.0]),
            rtol=1e-6,
            atol=1e-6,
        )

        x2 = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)

        def subset_fn(z):
            return np.array([z[0] + 10.0, z[-1] + 20.0], dtype=np.float32)

        assert_allclose(
            mp.piecewise(x2, [x2 < 0, x2 >= 0], [subset_fn, subset_fn]),
            np.piecewise(x2, [x2 < 0, x2 >= 0], [subset_fn, subset_fn]),
            rtol=1e-6,
            atol=1e-6,
        )
