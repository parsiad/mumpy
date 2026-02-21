#!/usr/bin/env python
"""Run randomized MumPy vs NumPy differential soak checks."""

import argparse
import sys
import time

import numpy as np

import mumpy as mp


def _assert_allclose(a, b, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, equal_nan=True)


def _assert_array_equal(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _run_case(seed: int):
    rng = np.random.default_rng(seed)

    # Elementwise + reductions
    shape = tuple(int(x) for x in rng.integers(1, 6, size=2))
    a_np = rng.normal(size=shape).astype(np.float32)
    b_np = rng.normal(size=shape).astype(np.float32)
    _assert_allclose(mp.add(a_np, b_np), a_np + b_np)
    _assert_allclose(mp.subtract(a_np, b_np), a_np - b_np)
    _assert_allclose(mp.multiply(a_np, b_np), a_np * b_np)
    _assert_allclose(mp.clip(a_np, -1.0, 1.0), np.clip(a_np, -1.0, 1.0))
    _assert_allclose(mp.sum(a_np, axis=0), np.sum(a_np, axis=0))
    _assert_allclose(mp.mean(a_np, axis=1), np.mean(a_np, axis=1))
    _assert_allclose(mp.std(a_np, axis=0), np.std(a_np, axis=0), rtol=2e-5, atol=2e-5)

    q = [0.25, 0.5, 0.75]
    _assert_allclose(mp.quantile(a_np, q, axis=0), np.quantile(a_np, q, axis=0))

    # Shape/index helpers
    x_np = rng.integers(-10, 10, size=shape, dtype=np.int32)
    _assert_array_equal(mp.flip(x_np, axis=1), np.flip(x_np, axis=1))
    shift = int(rng.integers(-3, 4))
    _assert_array_equal(mp.roll(x_np, shift, axis=0), np.roll(x_np, shift, axis=0))
    _assert_array_equal(mp.argwhere(x_np), np.argwhere(x_np))
    idx = rng.integers(0, shape[0] * shape[1], size=5, dtype=np.int32)
    ur_mp = mp.unravel_index(idx, shape)
    ur_np = np.unravel_index(idx, shape)
    for got, exp in zip(ur_mp, ur_np, strict=False):
        _assert_array_equal(got, exp)
    _assert_array_equal(mp.ravel_multi_index(ur_mp, shape), np.ravel_multi_index(ur_np, shape))

    # Searching + histograms
    sorted_np = np.sort(rng.integers(-8, 12, size=10, dtype=np.int32))
    probes = rng.integers(-8, 12, size=7, dtype=np.int32)
    _assert_array_equal(mp.searchsorted(sorted_np, probes), np.searchsorted(sorted_np, probes))
    _assert_array_equal(mp.isin(x_np, sorted_np), np.isin(x_np, sorted_np))
    _assert_array_equal(
        mp.bincount(np.abs(x_np).ravel().astype(np.int32), minlength=5),
        np.bincount(np.abs(x_np).ravel().astype(np.int32), minlength=5),
    )

    # Random distributions (reproducibility spot checks)
    mp.random.seed(seed)
    r1 = mp.random.normal(size=(2, 3))
    mp.random.seed(seed)
    r2 = mp.random.normal(size=(2, 3))
    _assert_array_equal(r1, r2)

    # Linalg / FFT on small safe cases
    n = int(rng.integers(2, 5))
    m0 = rng.normal(size=(n, n)).astype(np.float32)
    m_np = (m0 @ m0.T + np.eye(n, dtype=np.float32) * 0.25).astype(np.float32)
    rhs_np = rng.normal(size=(n,)).astype(np.float32)
    _assert_allclose(mp.linalg.solve(m_np, rhs_np), np.linalg.solve(m_np, rhs_np), rtol=3e-4, atol=3e-4)
    _assert_allclose(mp.linalg.det(m_np), np.linalg.det(m_np), rtol=3e-4, atol=3e-4)

    v_np = rng.normal(size=(int(rng.integers(4, 16)),)).astype(np.float32)
    _assert_allclose(mp.fft.fft(v_np), np.fft.fft(v_np), rtol=3e-4, atol=3e-4)
    _assert_allclose(mp.fft.rfft(v_np), np.fft.rfft(v_np), rtol=3e-4, atol=3e-4)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run randomized differential checks against NumPy.")
    parser.add_argument("seconds", type=float, metavar="SECONDS", help="Duration limit in seconds.")
    parser.add_argument("--batch-size", type=int, default=50, help="Cases per batch.")
    parser.add_argument("--progress-seconds", type=float, default=15.0, help="Progress print interval.")
    args = parser.parse_args()
    if args.seconds <= 0:
        sys.stderr.write("--seconds must be > 0\n")
        return 2

    end_time = time.time() + float(args.seconds)

    started = time.time()
    last_progress = 0.0
    batch = 0
    case = 0
    while time.time() < end_time:
        for _ in range(args.batch_size):
            _run_case(seed=case)
            case += 1
        batch += 1
        now = time.time()
        if now - last_progress >= args.progress_seconds:
            elapsed = now - started
            cases_done = batch * args.batch_size
            rate = cases_done / max(elapsed, 1e-9)
            eta = max(end_time - now, 0.0)
            msg = (
                f"[soak] batches={batch} cases={cases_done} elapsed={elapsed:.1f}s "
                f"rate={rate:.1f}/s remaining={eta / 60:.1f}m local={time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            sys.stdout.write(f"{msg}\n")
            sys.stdout.flush()
            last_progress = now

    total = time.time() - started
    msg = f"[soak] completed batches={batch} total_cases={batch * args.batch_size} total_elapsed={total:.1f}s"
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
