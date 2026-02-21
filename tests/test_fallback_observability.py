"""Tests for public fallback observability counters and capture helpers."""

import threading

import numpy as np

import mumpy as mp


def test_fallback_counts_increment_for_known_fallback_and_reset():
    mp.fallbacks.reset_counts()
    assert mp.fallbacks.get_counts() == {}

    with mp.fallbacks.capture_counts(reset=True) as cap:
        _ = mp.lexsort((np.array([1, 1, 0, 0]), np.array([3, 2, 1, 0])))

    assert cap.before == {}
    assert cap.after.get("core.lexsort:numpy", 0) >= 1
    assert cap.delta.get("core.lexsort:numpy", 0) >= 1

    counts = mp.fallbacks.get_counts()
    assert counts.get("core.lexsort:numpy", 0) >= 1

    mp.fallbacks.reset_counts()
    assert mp.fallbacks.get_counts() == {}


def test_capture_counts_delta_isolated_and_native_paths_do_not_increment():
    mp.fallbacks.reset_counts()
    _ = mp.sort_complex(np.array([1 + 2j, 0 + 1j], dtype=np.complex64))
    baseline = mp.fallbacks.get_counts()
    assert baseline.get("core.sort_complex:numpy", 0) >= 1

    with mp.fallbacks.capture_counts(reset=False) as cap:
        _ = mp.histogram(np.array([0.1, 0.4, 0.8], dtype=np.float32), bins=2)

    assert cap.before == baseline
    assert cap.delta == {}
    assert cap.after == baseline


def test_piecewise_nonvectorized_callable_records_fallback_site():
    mp.fallbacks.reset_counts()
    x = np.array([-2.0, -0.5, 0.0, 0.5], dtype=np.float32)

    def subset_fn(_z):
        return np.array([10.0, 20.0], dtype=np.float32)

    with mp.fallbacks.capture_counts(reset=True) as cap:
        got = mp.piecewise(x, [x < 0, x >= 0], [subset_fn, subset_fn])

    np.testing.assert_allclose(np.asarray(got), np.piecewise(x, [x < 0, x >= 0], [subset_fn, subset_fn]))
    assert cap.delta.get("core.piecewise:nonvectorized_callable", 0) >= 1


def test_fallback_counter_thread_safety_smoke():
    mp.fallbacks.reset_counts()

    def worker():
        _ = mp.lexsort((np.array([1, 0, 1]), np.array([2, 1, 0])))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    counts = mp.fallbacks.get_counts()
    assert counts.get("core.lexsort:numpy", 0) >= 4
