"""Tests for core linear algebra and FFT operations."""

import numpy as np
import pytest

import mumpy as mp

from .conftest import assert_allclose, assert_array_equal, to_numpy


def _dtype_name(x):
    dt = getattr(x, "dtype", x)
    return str(dt).removeprefix("mlx.core.")


def test_linalg_core_solvers_and_factorizations():
    a_np = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    b_np = np.array([1.0, 2.0], dtype=np.float32)
    a = mp.array(a_np)
    b = mp.array(b_np)

    assert_allclose(mp.linalg.norm(a), np.linalg.norm(a_np), rtol=1e-4, atol=1e-5)
    assert_allclose(mp.linalg.inv(a), np.linalg.inv(a_np), rtol=1e-4, atol=1e-5)
    assert_allclose(mp.linalg.solve(a, b), np.linalg.solve(a_np, b_np), rtol=1e-4, atol=1e-5)
    assert_allclose(mp.linalg.pinv(a), np.linalg.pinv(a_np), rtol=1e-4, atol=1e-4)

    chol = mp.linalg.cholesky(a)
    assert_allclose(chol @ chol.T, a_np, rtol=1e-4, atol=1e-4)

    q, r = mp.linalg.qr(a)
    assert_allclose(q @ r, a_np, rtol=1e-4, atol=1e-4)
    r_only = mp.linalg.qr(a, mode="r")
    assert_allclose(r_only, r, rtol=1e-4, atol=1e-4)

    u, s, vt = mp.linalg.svd(a)
    s_diag = np.diag(to_numpy(s))
    assert_allclose(to_numpy(u) @ s_diag @ to_numpy(vt), a_np, rtol=1e-4, atol=1e-4)
    assert_allclose(mp.linalg.svd(a, compute_uv=False), np.linalg.svd(a_np, compute_uv=False), rtol=1e-4, atol=1e-4)


def test_linalg_eigen_and_matrix_power_helpers():
    a_np = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    a = mp.array(a_np)

    eigvals, eigvecs = mp.linalg.eig(a)
    np_eigvals, _np_eigvecs = np.linalg.eig(a_np)
    assert_allclose(np.sort_complex(to_numpy(eigvals)), np.sort_complex(np_eigvals), rtol=1e-4, atol=1e-4)
    # Eigenvectors can differ by sign/phase, so validate the decomposition instead.
    recon = to_numpy(eigvecs) @ np.diag(to_numpy(eigvals)) @ np.linalg.inv(to_numpy(eigvecs))
    assert_allclose(recon, a_np, rtol=1e-4, atol=1e-4)

    w, v = mp.linalg.eigh(a)
    np_w, _np_v = np.linalg.eigh(a_np)
    assert_allclose(w, np_w, rtol=1e-4, atol=1e-4)
    assert_allclose(to_numpy(v) @ np.diag(to_numpy(w)) @ to_numpy(v).T, a_np, rtol=1e-4, atol=1e-4)
    assert_allclose(mp.linalg.eigvals(a), np_eigvals, rtol=1e-4, atol=1e-4)
    assert_allclose(mp.linalg.eigvalsh(a), np_w, rtol=1e-4, atol=1e-4)

    assert_allclose(mp.linalg.matrix_power(a, 0), np.linalg.matrix_power(a_np, 0))
    assert_allclose(mp.linalg.matrix_power(a, 3), np.linalg.matrix_power(a_np, 3), rtol=1e-4, atol=1e-4)
    assert_allclose(mp.linalg.matrix_power(a, -1), np.linalg.matrix_power(a_np, -1), rtol=1e-4, atol=1e-4)


def test_linalg_misc_helpers_and_error_paths():
    x = mp.array([[1.0, 0.0, 0.0]], dtype=mp.float32)
    y = mp.array([[0.0, 1.0, 0.0]], dtype=mp.float32)
    assert_allclose(mp.linalg.cross(x, y), np.cross(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])))

    a_np = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    b_np = np.array([1.0, 2.0, 2.5], dtype=np.float32)
    x_sol, residuals, rank, s = mp.linalg.lstsq(a_np, b_np)
    np_x, *_ = np.linalg.lstsq(a_np, b_np, rcond=None)
    assert_allclose(x_sol, np_x, rtol=1e-4, atol=1e-4)
    assert_allclose(np.asarray(a_np) @ to_numpy(x_sol), a_np @ np_x, rtol=1e-4, atol=1e-4)
    np_full = np.linalg.lstsq(a_np, b_np, rcond=None)
    assert_allclose(residuals, np_full[1], rtol=1e-5, atol=1e-5)
    assert_array_equal(rank, np.array(np_full[2]))
    assert s.shape[-1] == 2

    q_c, r_c = mp.linalg.qr(a_np, mode="complete")
    np_qc, np_rc = np.linalg.qr(a_np, mode="complete")
    assert_allclose(q_c, np_qc, rtol=1e-5, atol=1e-5)
    assert_allclose(r_c, np_rc, rtol=1e-5, atol=1e-5)

    u_r, s_r, vt_r = mp.linalg.svd(a_np, full_matrices=False)
    np_ur, np_sr, np_vtr = np.linalg.svd(a_np, full_matrices=False)
    assert_allclose(s_r, np_sr, rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(u_r)), np.abs(np_ur), rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(vt_r)), np.abs(np_vtr), rtol=1e-5, atol=1e-5)
    assert_allclose(to_numpy(u_r) @ np.diag(to_numpy(s_r)) @ to_numpy(vt_r), a_np, rtol=1e-5, atol=1e-5)

    herm = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    hu, hs, hvt = mp.linalg.svd(herm, hermitian=True)
    np_hu, np_hs, np_hvt = np.linalg.svd(herm, hermitian=True)
    assert_allclose(hs, np_hs, rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(hu)), np.abs(np_hu), rtol=1e-5, atol=1e-5)
    assert_allclose(np.abs(to_numpy(hvt)), np.abs(np_hvt), rtol=1e-5, atol=1e-5)
    assert_allclose(to_numpy(hu) @ np.diag(to_numpy(hs)) @ to_numpy(hvt), herm, rtol=1e-5, atol=1e-5)

    x_r, *_ = mp.linalg.lstsq(a_np, b_np, rcond=1e-6)
    np_x_r, *_ = np.linalg.lstsq(a_np, b_np, rcond=1e-6)
    assert_allclose(x_r, np_x_r, rtol=1e-4, atol=1e-4)


def test_linalg_det_slogdet_and_matrix_rank():
    a_np = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    a = mp.array(a_np)
    assert_allclose(mp.linalg.det(a), np.linalg.det(a_np), rtol=1e-5, atol=1e-5)

    sign, logdet = mp.linalg.slogdet(a)
    np_sign, np_logdet = np.linalg.slogdet(a_np)
    assert_allclose(sign, np_sign, rtol=1e-6, atol=1e-6)
    assert_allclose(logdet, np_logdet, rtol=1e-6, atol=1e-6)

    rank_full = mp.linalg.matrix_rank(a)
    rank_def = mp.linalg.matrix_rank(np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32))
    assert_array_equal(rank_full, np.array(np.linalg.matrix_rank(a_np)))
    assert_array_equal(
        rank_def,
        np.array(np.linalg.matrix_rank(np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32))),
    )


def test_fft_roundtrips_and_frequency_helpers():
    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = mp.array(x_np)
    fft_x = mp.fft.fft(x)
    assert_allclose(fft_x, np.fft.fft(x_np), rtol=1e-4, atol=1e-4)
    assert_allclose(mp.fft.ifft(fft_x), np.fft.ifft(np.fft.fft(x_np)), rtol=1e-4, atol=1e-4)

    rfft_x = mp.fft.rfft(x)
    assert_allclose(rfft_x, np.fft.rfft(x_np), rtol=1e-4, atol=1e-4)
    assert_allclose(mp.fft.irfft(rfft_x, n=4), np.fft.irfft(np.fft.rfft(x_np), n=4), rtol=1e-4, atol=1e-4)

    x2_np = np.arange(9, dtype=np.float32).reshape(3, 3)
    x2 = mp.array(x2_np)
    assert_allclose(mp.fft.fft2(x2), np.fft.fft2(x2_np), rtol=1e-4, atol=1e-4)
    assert_allclose(mp.fft.ifft2(mp.fft.fft2(x2)), np.fft.ifft2(np.fft.fft2(x2_np)), rtol=1e-4, atol=1e-4)

    x3_np = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    x3 = mp.array(x3_np)
    assert_allclose(mp.fft.fftn(x3), np.fft.fftn(x3_np), rtol=1e-4, atol=1e-4)
    assert_allclose(mp.fft.ifftn(mp.fft.fftn(x3)), np.fft.ifftn(np.fft.fftn(x3_np)), rtol=1e-4, atol=1e-4)
    assert_allclose(mp.fft.rfftn(x3), np.fft.rfftn(x3_np), rtol=1e-4, atol=1e-4)
    assert_allclose(
        mp.fft.irfftn(mp.fft.rfftn(x3), s=x3_np.shape),
        np.fft.irfftn(np.fft.rfftn(x3_np), s=x3_np.shape, axes=(0, 1, 2)),
        rtol=1e-4,
        atol=1e-4,
    )

    vec_np = np.arange(6, dtype=np.float32)
    vec = mp.array(vec_np)
    assert_array_equal(mp.fft.fftshift(vec), np.fft.fftshift(vec_np))
    assert_array_equal(mp.fft.ifftshift(mp.fft.fftshift(vec)), np.fft.ifftshift(np.fft.fftshift(vec_np)))
    assert_allclose(mp.fft.fftfreq(8, d=0.5), np.fft.fftfreq(8, d=0.5))
    assert_allclose(mp.fft.rfftfreq(9, d=0.25), np.fft.rfftfreq(9, d=0.25))

    with pytest.raises(ValueError, match="n must be positive"):
        mp.fft.fftfreq(0)
    with pytest.raises(ValueError, match="n must be positive"):
        mp.fft.rfftfreq(0)


def test_fft_and_linalg_selected_default_dtype_parity():
    assert _dtype_name(mp.fft.fftfreq(8, d=0.5)) == np.fft.fftfreq(8, d=0.5).dtype.name
    assert _dtype_name(mp.fft.rfftfreq(9, d=0.25)) == np.fft.rfftfreq(9, d=0.25).dtype.name

    a_i = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert _dtype_name(mp.linalg.trace(a_i)) == np.linalg.trace(a_i).dtype.name

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    _, residuals_mp, *_ = mp.linalg.lstsq(a, b)
    _, residuals_np, *_ = np.linalg.lstsq(a, b, rcond=None)
    assert residuals_mp.shape == residuals_np.shape == (0,)
    assert _dtype_name(residuals_mp) == residuals_np.dtype.name
