"""Tests for dynamic NumPy fallback bridge behavior and parity."""

import numpy as np
import pytest

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal, to_numpy


def test_top_level_numpy_fallback_ufuncs_and_helpers():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    ints_a = np.array([12, 18, 7], dtype=np.int32)
    ints_b = np.array([8, 24, 21], dtype=np.int32)

    assert_allclose(mp.atan2(x, y), np.atan2(x, y), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.exp2(x), np.exp2(x), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.fmod([-3.5, 4.5], 2.0), np.fmod([-3.5, 4.5], 2.0), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.copysign([-1.0, 2.0], [3.0, -4.0]), np.copysign([-1.0, 2.0], [3.0, -4.0]), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.hypot([3.0, 5.0], [4.0, 12.0]), np.hypot([3.0, 5.0], [4.0, 12.0]), rtol=1e-6, atol=1e-6)
    assert_array_equal(mp.gcd(ints_a, ints_b), np.gcd(ints_a, ints_b))
    assert_array_equal(mp.lcm(ints_a, ints_b), np.lcm(ints_a, ints_b))
    n1 = np.array([0.0, 1.0], dtype=np.float32)
    n2 = np.array([1.0, 2.0], dtype=np.float32)
    assert_allclose(mp.nextafter(n1, n2), np.nextafter(n1, n2), rtol=0, atol=0)
    assert_array_equal(mp.signbit([-1.0, 0.0, 2.0]), np.signbit([-1.0, 0.0, 2.0]))

    frac_mp, exp_mp = mp.frexp([1.0, 8.0, 10.0])
    frac_np, exp_np = np.frexp([1.0, 8.0, 10.0])
    assert_allclose(frac_mp, frac_np, rtol=1e-6, atol=1e-6)
    assert_array_equal(exp_mp, exp_np)

    frac2_mp, int2_mp = mp.modf([1.25, -2.75, 3.0])
    frac2_np, int2_np = np.modf([1.25, -2.75, 3.0])
    assert_allclose(frac2_mp, frac2_np, rtol=1e-6, atol=1e-6)
    assert_allclose(int2_mp, int2_np, rtol=1e-6, atol=1e-6)
    assert_allclose(mp.ldexp(frac_mp, exp_mp), np.ldexp(frac_np, exp_np), rtol=1e-6, atol=1e-6)
    la = np.array([1.0, 2.0], dtype=np.float32)
    lb = np.array([2.0, 3.0], dtype=np.float32)
    assert_allclose(mp.logaddexp(la, lb), np.logaddexp(la, lb), rtol=1e-6, atol=1e-6)

    assert mp.broadcast_shapes((2, 1, 3), (1, 4, 3)) == np.broadcast_shapes((2, 1, 3), (1, 4, 3))
    assert_allclose(mp.geomspace(1.0, 16.0, num=5), np.geomspace(1.0, 16.0, num=5), rtol=1e-6, atol=1e-6)
    assert mp.result_type(mp.array([1], dtype=mp.int32), mp.array([1.0], dtype=mp.float32)) == np.result_type(
        np.array([1], dtype=np.int32),
        np.array([1.0], dtype=np.float32),
    )
    assert mp.promote_types(np.int16, np.float32) == np.promote_types(np.int16, np.float32)
    assert mp.can_cast(np.int16, np.int32) == np.can_cast(np.int16, np.int32)
    assert mp.finfo(np.float32).eps == np.finfo(np.float32).eps
    assert mp.iinfo(np.int16).max == np.iinfo(np.int16).max

    # Mutation-style wrappers operate on NumPy arrays when passed NumPy arrays.
    fill = np.zeros((3, 3), dtype=np.int32)
    mp.fill_diagonal(fill, 7)
    assert_array_equal(fill, np.array([[7, 0, 0], [0, 7, 0], [0, 0, 7]], dtype=np.int32))

    conv_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    conv_b = np.array([1.0, 1.0], dtype=np.float32)
    corr_b = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    assert_allclose(mp.convolve(conv_a, conv_b), np.convolve(conv_a, conv_b), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.correlate(conv_a, corr_b), np.correlate(conv_a, corr_b), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.vander([1, 2, 3], N=3), np.vander([1, 2, 3], N=3), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.unwrap([0.0, 3.0, 6.5]), np.unwrap([0.0, 3.0, 6.5]), rtol=1e-6, atol=1e-6)

    assert callable(mp.atan2)
    assert "atan2" not in mp.__all__
    assert "atan2" not in dir(mp)


def test_linalg_and_fft_fallbacks_work():
    a_np = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    assert_allclose(mp.linalg.svdvals(a_np), np.linalg.svdvals(a_np), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.vector_norm(a_np, axis=1), np.linalg.vector_norm(a_np, axis=1), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.matrix_norm(a_np), np.linalg.matrix_norm(a_np), rtol=1e-5, atol=1e-5)
    assert_allclose(mp.linalg.trace(a_np), np.linalg.trace(a_np), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.linalg.diagonal(a_np), np.linalg.diagonal(a_np), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.linalg.outer([1, 2], [3, 4]), np.linalg.outer([1, 2], [3, 4]), rtol=1e-6, atol=1e-6)
    assert_allclose(mp.linalg.matmul(a_np, a_np), np.linalg.matmul(a_np, a_np), rtol=1e-5, atol=1e-5)
    assert_allclose(
        mp.linalg.tensordot(np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), axes=1),
        np.linalg.tensordot(np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), axes=1),
        rtol=1e-6,
        atol=1e-6,
    )

    t_np = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
    inv_t_mp = mp.linalg.tensorinv(t_np)
    inv_t_np = np.linalg.tensorinv(t_np)
    assert_allclose(inv_t_mp, inv_t_np, rtol=1e-4, atol=1e-4)
    b_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert_allclose(mp.linalg.tensorsolve(t_np, b_np), np.linalg.tensorsolve(t_np, b_np), rtol=1e-4, atol=1e-4)

    mt = mp.linalg.matrix_transpose(np.arange(6).reshape(2, 3))
    assert_array_equal(mt, np.linalg.matrix_transpose(np.arange(6).reshape(2, 3)))
    assert issubclass(mp.linalg.LinAlgError, Exception)
    with pytest.raises(mp.linalg.LinAlgError):
        np.linalg.inv(np.zeros((2, 2)))

    hfreq = np.array([1.0 + 0.0j, 2.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex64)
    assert_allclose(mp.fft.hfft(hfreq), np.fft.hfft(hfreq), rtol=1e-5, atol=1e-5)
    full_real = np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32)
    assert_allclose(mp.fft.ihfft(full_real), np.fft.ihfft(full_real), rtol=1e-5, atol=1e-5)


def test_random_module_numpy_fallbacks_are_available_and_seeded():
    mp.random.seed(2025)
    got_gamma_1 = mp.random.gamma(shape=2.0, scale=1.5, size=5)
    mp.random.seed(2025)
    got_gamma_2 = mp.random.gamma(shape=2.0, scale=1.5, size=5)
    assert_allclose(got_gamma_1, got_gamma_2, rtol=0, atol=0)
    assert np.all(to_numpy(got_gamma_1) >= 0)

    mp.random.seed(123)
    got_beta_1 = mp.random.beta(a=2.0, b=5.0, size=6)
    mp.random.seed(123)
    got_beta_2 = mp.random.beta(a=2.0, b=5.0, size=6)
    assert_allclose(got_beta_1, got_beta_2, rtol=0, atol=0)
    beta_np = to_numpy(got_beta_1)
    assert np.all((beta_np >= 0) & (beta_np <= 1))

    mp.random.seed(77)
    got_gumbel_1 = mp.random.gumbel(loc=1.0, scale=2.0, size=4)
    mp.random.seed(77)
    got_gumbel_2 = mp.random.gumbel(loc=1.0, scale=2.0, size=4)
    assert_allclose(got_gumbel_1, got_gumbel_2, rtol=0, atol=0)

    mp.random.seed(88)
    got_multi = mp.random.multinomial(5, [0.2, 0.3, 0.5], size=3)
    mp.random.seed(88)
    got_multi_2 = mp.random.multinomial(5, [0.2, 0.3, 0.5], size=3)
    assert_array_equal(got_multi, got_multi_2)
    assert_array_equal(to_numpy(got_multi).sum(axis=1), np.array([5, 5, 5]))

    mp.random.seed(99)
    dir_mp = mp.random.dirichlet([1.0, 2.0, 3.0], size=4)
    mp.random.seed(99)
    dir_mp_2 = mp.random.dirichlet([1.0, 2.0, 3.0], size=4)
    assert_allclose(dir_mp, dir_mp_2, rtol=0, atol=0)
    dir_np = to_numpy(dir_mp)
    assert_allclose(dir_np.sum(axis=1), np.ones(4, dtype=np.float32), rtol=1e-5, atol=1e-5)
    assert np.all(dir_np > 0)

    payload = mp.random.bytes(16)
    assert isinstance(payload, (bytes, bytearray))
    assert len(payload) == 16


def test_generator_numpy_fallback_methods_are_available_and_reproducible():
    g1 = mp.random.default_rng(42)
    g2 = mp.random.default_rng(42)

    assert_allclose(g1.gamma(shape=2.0, scale=1.0, size=5), g2.gamma(shape=2.0, scale=1.0, size=5), rtol=0, atol=0)
    assert_allclose(g1.beta(a=2.0, b=3.0, size=5), g2.beta(a=2.0, b=3.0, size=5), rtol=0, atol=0)
    assert_allclose(g1.gumbel(loc=0.0, scale=1.0, size=5), g2.gumbel(loc=0.0, scale=1.0, size=5), rtol=0, atol=0)
    assert_allclose(g1.standard_t(df=5.0, size=5), g2.standard_t(df=5.0, size=5), rtol=0, atol=0)
    assert_allclose(
        g1.triangular(left=-1.0, mode=0.0, right=2.0, size=5),
        g2.triangular(left=-1.0, mode=0.0, right=2.0, size=5),
        rtol=0,
        atol=0,
    )
    assert_allclose(g1.weibull(a=1.5, size=5), g2.weibull(a=1.5, size=5), rtol=0, atol=0)
    assert_array_equal(g1.multinomial(4, [0.2, 0.3, 0.5], size=3), g2.multinomial(4, [0.2, 0.3, 0.5], size=3))

    mat = np.arange(12).reshape(3, 4)
    p1 = g1.permuted(mat, axis=1)
    p2 = g2.permuted(mat, axis=1)
    assert_array_equal(p1, p2)
    assert p1.shape == (3, 4)

    b1 = g1.bytes(12)
    b2 = g2.bytes(12)
    assert b1 == b2
    assert len(b1) == 12

    children1 = g1.spawn(2)
    children2 = g2.spawn(2)
    assert len(children1) == len(children2) == 2
    for c1, c2 in zip(children1, children2, strict=False):
        assert isinstance(c1, mp.random.Generator)
        assert isinstance(c2, mp.random.Generator)
        assert_array_equal(c1.integers(0, 100, size=5), c2.integers(0, 100, size=5))

    assert hasattr(g1, "bit_generator")
    assert hasattr(g1.bit_generator, "random_raw")


def test_vecdot_matvec_vecmat_fallbacks_if_available():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    if hasattr(np, "vecdot"):
        assert_allclose(mp.vecdot(a, b), np.vecdot(a, b), rtol=1e-6, atol=1e-6)
    if hasattr(np, "matvec"):
        m = np.arange(6, dtype=np.float32).reshape(2, 3)
        assert_allclose(mp.matvec(m, a), np.matvec(m, a), rtol=1e-6, atol=1e-6)
    if hasattr(np, "vecmat"):
        m = np.arange(6, dtype=np.float32).reshape(3, 2)
        assert_allclose(mp.vecmat(a, m), np.vecmat(a, m), rtol=1e-6, atol=1e-6)


def test_fallback_tracer_records_dynamic_module_and_generator_paths():
    mp.fallbacks.reset_counts()

    _ = mp.random.hypergeometric(10, 5, 4, size=3)
    g = mp.random.default_rng(123)
    _ = g.hypergeometric(10, 5, 4, size=3)

    counts = mp.fallbacks.get_counts()
    assert counts.get("random.dynamic:hypergeometric", 0) >= 1
    assert counts.get("random.generator.dynamic:hypergeometric", 0) >= 1


def test_dynamic_bridge_selected_dtype_parity_and_unrepresentable_python_returns():
    got_angle = mp.angle([1 + 1j, 1 - 1j])
    assert isinstance(got_angle, mp.MumPyArray)
    exp_angle = np.angle([1 + 1j, 1 - 1j])
    assert str(got_angle.dtype).removeprefix("mlx.core.") == exp_angle.dtype.name

    g = mp.random.default_rng(999)
    g_np = np.random.default_rng(999)
    got_hyper = g.hypergeometric(10, 5, 4, size=6)
    assert isinstance(got_hyper, mp.MumPyArray)
    exp_hyper = g_np.hypergeometric(10, 5, 4, size=6)
    assert str(got_hyper.dtype).removeprefix("mlx.core.") == exp_hyper.dtype.name

    text_repr = mp.array2string(mp.array([1, 2, 3], dtype=mp.int32))
    assert isinstance(text_repr, str)
