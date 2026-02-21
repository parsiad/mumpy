from typing import Any, Literal, cast, overload

import mlx.core as mx

from .. import _bridge as bridge
from .. import _core as core
from .. import _dynamic_exports as dynamic_exports

_LINALG_STREAM = getattr(mx, "cpu", None)
_FALLBACK_ATTR_CACHE: dict[str, Any] = {}
_NUMPY_EXPORT_EXCEPTIONS = (NotImplementedError, bridge.FallbackDisabledError)
_scalar_bool = core._scalar_bool
_prod_int = core._prod_int


def __getattr__(name: str) -> Any:
    return dynamic_exports.resolve_module_fallback_attr(
        name,
        cache=_FALLBACK_ATTR_CACHE,
        native_namespace=mx.linalg,
        numpy_namespace_getter=lambda: bridge.numpy_module().linalg,
        bridge_module=bridge,
        bridge_namespace="linalg",
        public_module_name="mumpy.linalg",
    )


def _asarray(a: Any) -> mx.array:
    return core.asarray(a)


def _lin_alg_error_type() -> type[Exception]:
    try:
        return bridge.numpy_module().linalg.LinAlgError
    except (AttributeError, NotImplementedError, bridge.FallbackDisabledError):
        return ValueError


def _float_eps(dtype: Any) -> float:
    return core._float_eps_for_dtype(dtype)


_QRMode = Literal["reduced", "complete", "r", "raw"]


def _normalize_linalg_norm_axis(axis: int | tuple[int, int] | None) -> int | list[int] | None:
    if isinstance(axis, tuple):
        return list(axis)
    return axis


def _narrow_qr_mode(mode: str) -> _QRMode | None:
    if mode in {"reduced", "complete", "r", "raw"}:
        return cast("_QRMode", mode)
    return None


def _canonicalize_svd_signs(u: mx.array, vh: mx.array) -> tuple[mx.array, mx.array]:
    # Make SVD signs/phases deterministic (similar to sklearn.utils.extmath.svd_flip).
    max_idx = core.argmax(core.abs(u), axis=-2)
    idx = core.expand_dims(max_idx, -2)
    idx = core.broadcast_to(idx, (*tuple(int(d) for d in idx.shape[:-2]), 1, int(u.shape[-1])))
    u_max = core.squeeze(core.take_along_axis(u, idx, axis=-2), axis=-2)
    mag = core.abs(u_max)
    phase = core.where(core.equal(mag, 0), 1, u_max / mag)
    u = u * core.expand_dims(core.conjugate(phase), -2)
    vh = core.expand_dims(phase, -1) * vh
    return u, vh


def norm(
    a: Any,
    ord: Any = None,
    axis: int | tuple[int, int] | None = None,
    keepdims: bool = False,
) -> mx.array:
    axis_arg = _normalize_linalg_norm_axis(axis)
    return mx.linalg.norm(_asarray(a), ord=ord, axis=axis_arg, keepdims=keepdims, stream=_LINALG_STREAM)


def inv(a: Any) -> mx.array:
    return mx.linalg.inv(_asarray(a), stream=_LINALG_STREAM)


def pinv(a: Any) -> mx.array:
    return mx.linalg.pinv(_asarray(a), stream=_LINALG_STREAM)


def solve(a: Any, b: Any) -> mx.array:
    return mx.linalg.solve(_asarray(a), _asarray(b), stream=_LINALG_STREAM)


def cholesky(a: Any, upper: bool = False) -> mx.array:
    return mx.linalg.cholesky(_asarray(a), upper=upper, stream=_LINALG_STREAM)


@overload
def qr(a: Any, mode: Literal["r"]) -> mx.array: ...


@overload
def qr(a: Any, mode: Literal["reduced", "complete", "raw"] = "reduced") -> Any: ...


def qr(a: Any, mode: str = "reduced") -> Any:
    if mode in {"reduced", "r"}:
        q, r = mx.linalg.qr(_asarray(a), stream=_LINALG_STREAM)
        if mode == "r":
            return r
        return q, r
    if mode == "complete":
        arr = _asarray(a)
        q, r = mx.linalg.qr(arr, stream=_LINALG_STREAM)
        if arr.ndim >= 2 and int(arr.shape[-2]) <= int(arr.shape[-1]):
            return q, r
    bridge.record_fallback("linalg.qr:mode_numpy")
    onp = bridge.numpy_module()
    narrowed_mode = _narrow_qr_mode(mode)
    result = onp.linalg.qr(onp.asarray(a), mode=narrowed_mode if narrowed_mode is not None else cast("Any", mode))
    return bridge.wrap_from_numpy(result)


def svd(
    a: Any,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
) -> Any:
    arr = _asarray(a)
    if hermitian:
        if arr.ndim < 2 or int(arr.shape[-1]) != int(arr.shape[-2]):
            msg = "Last 2 dimensions of the array must be square"
            raise ValueError(msg)
        vals, vecs = mx.linalg.eigh(arr, stream=_LINALG_STREAM)
        s = core.abs(vals)
        order = core.argsort(s, axis=-1)[..., ::-1]
        s_sorted = core.take_along_axis(s, order, axis=-1)
        if not compute_uv:
            return s_sorted
        vals_sorted = core.take_along_axis(vals, order, axis=-1)
        idx = core.expand_dims(order, -2)
        idx = core.broadcast_to(idx, tuple(int(d) for d in vecs.shape))
        v_sorted = core.take_along_axis(vecs, idx, axis=-1)
        phase = core.where(core.equal(s_sorted, 0), 1, vals_sorted / s_sorted)
        u = v_sorted * core.expand_dims(phase, -2)
        vh = core.swapaxes(core.conjugate(v_sorted), -1, -2)
        u, vh = _canonicalize_svd_signs(u, vh)
        return u, s_sorted, vh

    result = mx.linalg.svd(arr, compute_uv=compute_uv, stream=_LINALG_STREAM)
    if not compute_uv:
        return result
    u, s, vh = result
    k = int(s.shape[-1])
    if not full_matrices:
        u = u[..., :, :k]
        vh = vh[..., :k, :]
    u, vh = _canonicalize_svd_signs(u, vh)
    return u, s, vh


def eig(a: Any) -> tuple[mx.array, mx.array]:
    return mx.linalg.eig(_asarray(a), stream=_LINALG_STREAM)


def eigvals(a: Any) -> mx.array:
    return mx.linalg.eigvals(_asarray(a), stream=_LINALG_STREAM)


def eigh(a: Any, UPLO: str = "L") -> tuple[mx.array, mx.array]:
    return mx.linalg.eigh(_asarray(a), UPLO=UPLO, stream=_LINALG_STREAM)


def eigvalsh(a: Any, UPLO: str = "L") -> mx.array:
    return mx.linalg.eigvalsh(_asarray(a), UPLO=UPLO, stream=_LINALG_STREAM)


def cross(a: Any, b: Any, axis: int = -1) -> mx.array:
    return mx.linalg.cross(_asarray(a), _asarray(b), axis=axis, stream=_LINALG_STREAM)


def matrix_power(a: Any, n: int) -> mx.array:
    if not isinstance(n, int):
        msg = "exponent must be an integer"
        raise TypeError(msg)
    arr = _asarray(a)
    if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
        msg = "input must be (..., M, M)"
        raise ValueError(msg)
    if n == 0:
        eye = mx.eye(arr.shape[-1], dtype=arr.dtype)
        if arr.ndim == 2:
            return eye
        batch_shape = arr.shape[:-2]
        return mx.broadcast_to(eye, batch_shape + eye.shape)
    if n < 0:
        arr = mx.linalg.inv(arr, stream=_LINALG_STREAM)
        n = -n
    result = None
    base = arr
    while n > 0:
        if n & 1:
            result = base if result is None else mx.matmul(result, base)
        base = mx.matmul(base, base)
        n >>= 1
    if result is None:
        msg = "Internal error: matrix_power result was not computed"
        raise RuntimeError(msg)
    return result


def det(a: Any) -> mx.array:
    vals = mx.linalg.eigvals(_asarray(a), stream=_LINALG_STREAM)
    return core.real_if_close(core.prod(vals, axis=-1))


def slogdet(a: Any) -> tuple[mx.array, mx.array]:
    vals = mx.linalg.eigvals(_asarray(a), stream=_LINALG_STREAM)
    abs_vals = core.abs(vals)
    zero_vals = core.equal(abs_vals, 0)
    phases = core.where(zero_vals, 1, vals / abs_vals)
    sign = core.prod(phases, axis=-1)
    logabs = core.sum(core.where(zero_vals, 0.0, core.log(abs_vals)), axis=-1)
    has_zero = core.any(zero_vals, axis=-1)
    sign = core.where(has_zero, 0, sign)
    logabs = core.where(has_zero, -core.inf, logabs)
    return core.real_if_close(sign), core.real_if_close(logabs)


def matrix_rank(a: Any, tol: Any | None = None, hermitian: bool = False) -> mx.array:
    arr = _asarray(a)
    if arr.ndim == 0:
        arr = core.reshape(arr, (1, 1))
    elif arr.ndim == 1:
        arr = core.reshape(arr, (1, int(arr.shape[0])))
    if hermitian:
        if int(arr.shape[-1]) != int(arr.shape[-2]):
            msg = "Last 2 dimensions of the array must be square"
            raise ValueError(msg)
        s = core.abs(mx.linalg.eigvalsh(arr, stream=_LINALG_STREAM))
    else:
        s = svd(arr, full_matrices=False, compute_uv=False)
    if tol is None:
        eps = _float_eps(s.dtype)
        dim_scale = float(max(int(arr.shape[-2]), int(arr.shape[-1])))
        tol_arr = core.max(s, axis=-1, keepdims=True) * dim_scale * eps
    else:
        tol_arr = _asarray(tol)
        if tol_arr.ndim > 0:
            tol_arr = core.expand_dims(tol_arr, -1)
    return core.sum(core.greater(s, tol_arr), axis=-1).astype(mx.int64)


def cond(x: Any, p: Any | None = None) -> mx.array:
    arr = _asarray(x)
    if p is None or p in (2, -2):
        s = svd(arr, full_matrices=False, compute_uv=False)
        smax = core.max(s, axis=-1)
        smin = core.min(s, axis=-1)
        if p == -2:
            return smin / smax
        return smax / smin
    return norm(arr, ord=p) * norm(inv(arr), ord=p)


def multi_dot(arrays: list[Any] | tuple[Any, ...]) -> mx.array:
    if len(arrays) < 2:
        msg = "Expecting at least two arrays."
        raise ValueError(msg)
    mats = [_asarray(a) for a in arrays]
    left_vec = mats[0].ndim == 1
    right_vec = mats[-1].ndim == 1
    if left_vec:
        mats[0] = core.reshape(mats[0], (1, int(mats[0].shape[0])))
    if right_vec:
        mats[-1] = core.reshape(mats[-1], (int(mats[-1].shape[0]), 1))
    for mid in mats[1:-1]:
        if mid.ndim != 2:
            msg = "All middle arrays must be two-dimensional."
            raise ValueError(msg)
    result = mats[0]
    for mat in mats[1:]:
        result = core.matmul(result, mat)
    if left_vec and right_vec:
        return result[0, 0]
    if left_vec:
        return core.squeeze(result, axis=0)
    if right_vec:
        return core.squeeze(result, axis=-1)
    return result


def lstsq(a: Any, b: Any, rcond: Any | None = None) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    a_arr = _asarray(a)
    b_arr = _asarray(b)
    if a_arr.ndim != 2:
        msg = "a must be two-dimensional"
        raise ValueError(msg)
    with core._cpu_default_device_for_dtypes(a_arr.dtype, b_arr.dtype):
        m, n = int(a_arr.shape[0]), int(a_arr.shape[1])
        b_was_1d = b_arr.ndim == 1
        if b_was_1d:
            if int(b_arr.shape[0]) != m:
                msg = "Incompatible dimensions"
                raise ValueError(msg)
            b2 = core.reshape(b_arr, (m, 1))
        elif b_arr.ndim == 2:
            if int(b_arr.shape[0]) != m:
                msg = "Incompatible dimensions"
                raise ValueError(msg)
            b2 = b_arr
        else:
            msg = "b must have 1 or 2 dimensions"
            raise ValueError(msg)

        u, s, vh = svd(a_arr, full_matrices=False, compute_uv=True)
        eps = _float_eps(s.dtype)
        if rcond is None:
            rcond_val = eps * float(max(m, n))
        else:
            rcond_val = float(_asarray(rcond).item()) if isinstance(rcond, mx.array) else float(rcond)
            if rcond_val < 0:
                rcond_val = eps
        cutoff = core.max(s) * rcond_val
        keep = core.greater(s, cutoff)
        rank = core.sum(keep.astype(mx.int64))
        s_inv = core.where(keep, 1.0 / s, 0.0)

        u_h = core.swapaxes(core.conjugate(u), -1, -2)
        tmp = core.matmul(u_h, b2)
        tmp = tmp * core.expand_dims(s_inv, -1)
        v = core.swapaxes(core.conjugate(vh), -1, -2)
        x_sol = core.matmul(v, tmp)

        if m > n and _scalar_bool(core.equal(rank, n)):
            resid = b2 - core.matmul(a_arr, x_sol)
            resid_sq = core.abs(resid) * core.abs(resid)
            residuals = core.sum(resid_sq, axis=0)
        else:
            resid_dtype = a_arr.dtype
            if core._is_bool_dtype(resid_dtype) or core._is_integer_dtype(resid_dtype):
                resid_dtype = core._default_float_output_dtype()
            elif core._is_complex_dtype(resid_dtype):
                resid_dtype = mx.float32 if resid_dtype is mx.complex64 else mx.float64
            residuals = core.zeros((0,), dtype=resid_dtype)

        if b_was_1d:
            x_sol = core.squeeze(x_sol, axis=1)
        return x_sol, residuals, rank, s


def diagonal(x: Any, /, *, offset: int = 0) -> mx.array:
    return core.diagonal(x, offset=offset, axis1=-2, axis2=-1)


def trace(x: Any, /, *, offset: int = 0, dtype: Any | None = None) -> mx.array:
    return core.trace(x, offset=offset, axis1=-2, axis2=-1, dtype=dtype)


def outer(x1: Any, x2: Any, /) -> mx.array:
    return core.outer(x1, x2)


def matmul(x1: Any, x2: Any, /) -> mx.array:
    return core.matmul(x1, x2)


def tensordot(x1: Any, x2: Any, /, *, axes: int | tuple[list[int], list[int]] = 2) -> mx.array:
    return core.tensordot(x1, x2, axes=axes)


def vecdot(x1: Any, x2: Any, /, *, axis: int = -1) -> mx.array:
    return core.vecdot(x1, x2, axis=axis)


def matrix_transpose(x: Any, /) -> mx.array:
    return core.matrix_transpose(x)


def svdvals(x: Any, /) -> mx.array:
    return svd(x, compute_uv=False)


def vector_norm(x: Any, /, *, axis: Any = None, keepdims: bool = False, ord: Any = 2) -> mx.array:
    return norm(x, ord=ord, axis=axis, keepdims=keepdims)


def matrix_norm(x: Any, /, *, keepdims: bool = False, ord: Any = "fro") -> mx.array:
    return norm(x, ord=ord, axis=(-2, -1), keepdims=keepdims)


def tensorinv(a: Any, ind: int = 2) -> mx.array:
    arr = _asarray(a)
    ind_i = int(ind)
    if ind_i <= 0 or ind_i >= arr.ndim:
        msg = "ind must satisfy 0 < ind < a.ndim"
        raise ValueError(msg)
    left = tuple(int(s) for s in arr.shape[:ind_i])
    right = tuple(int(s) for s in arr.shape[ind_i:])
    left_n = _prod_int(left)
    right_n = _prod_int(right)
    if left_n != right_n:
        msg = "Last dimensions of the array must be square"
        raise _lin_alg_error_type()(msg)
    invm = inv(core.reshape(arr, (left_n, right_n)))
    return core.reshape(invm, right + left)


def tensorsolve(a: Any, b: Any, axes: tuple[int, ...] | None = None) -> mx.array:
    arr = _asarray(a)
    rhs = _asarray(b)
    if axes is not None:
        axes_norm = tuple(ax + arr.ndim if ax < 0 else ax for ax in axes)
        if len(set(axes_norm)) != len(axes_norm):
            msg = "repeated axis in axes"
            raise ValueError(msg)
        remain = [i for i in range(arr.ndim) if i not in axes_norm]
        arr = core.transpose(arr, remain + list(axes_norm))
    b_dims = tuple(int(s) for s in rhs.shape)
    lead = tuple(int(s) for s in arr.shape[: rhs.ndim])
    tail = tuple(int(s) for s in arr.shape[rhs.ndim :])
    if lead != b_dims:
        msg = "Input arrays have incompatible shapes"
        raise _lin_alg_error_type()(msg)
    n = _prod_int(lead)
    m = _prod_int(tail)
    if n != m:
        msg = "Input arrays have incompatible shapes"
        raise _lin_alg_error_type()(msg)
    mat = core.reshape(arr, (n, m))
    vec = core.reshape(rhs, (n,))
    sol = solve(mat, vec)
    return core.reshape(sol, tail)


__all__ = [
    "cholesky",
    "cond",
    "cross",
    "det",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "lstsq",
    "matrix_power",
    "matrix_rank",
    "multi_dot",
    "norm",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
]


def __dir__() -> list[str]:
    return dynamic_exports.dynamic_dir(
        __all__,
        mx.linalg,
        numpy_namespace_getter=lambda: bridge.numpy_module().linalg,
        numpy_exceptions=_NUMPY_EXPORT_EXCEPTIONS,
    )


__all__ = dynamic_exports.extend_all_with_fallback_names(  # pyright: ignore[reportUnsupportedDunderAll]
    __all__,
    mx.linalg,
    numpy_namespace_getter=lambda: bridge.numpy_module().linalg,
    numpy_exceptions=_NUMPY_EXPORT_EXCEPTIONS,
)
