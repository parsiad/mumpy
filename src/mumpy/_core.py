import builtins
import math
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from typing import Any, Literal, cast, overload

import mlx.core as mx

from . import _bridge as bridge

ndarray = mx.array
newaxis = None

# Common dtype aliases.
bool_ = mx.bool_
int8 = mx.int8
int16 = mx.int16
int32 = mx.int32
int64 = mx.int64
uint8 = mx.uint8
uint16 = mx.uint16
uint32 = mx.uint32
uint64 = mx.uint64
float16 = mx.float16
float32 = mx.float32
float64 = mx.float64
bfloat16 = mx.bfloat16
complex64 = mx.complex64

# Scalar constants.
pi = mx.pi
e = mx.e
inf = mx.inf
nan = mx.nan


_DTYPE_BY_NAME = {
    "bool": bool_,
    "bool_": bool_,
    "int8": int8,
    "i1": int8,
    "int16": int16,
    "i2": int16,
    "short": int16,
    "int32": int32,
    "i4": int32,
    "int": int64,
    "intc": int32,
    "int64": int64,
    "i8": int64,
    "long": int64,
    "uint8": uint8,
    "u1": uint8,
    "uint16": uint16,
    "u2": uint16,
    "uint32": uint32,
    "u4": uint32,
    "uint64": uint64,
    "u8": uint64,
    "float16": float16,
    "f2": float16,
    "half": float16,
    "float32": float32,
    "f4": float32,
    "single": float32,
    "float": float64,
    "float64": float64,
    "f8": float64,
    "double": float64,
    "complex64": complex64,
    "c8": complex64,
    "complex": complex64,
    "bfloat16": bfloat16,
}

_PYTYPE_TO_DTYPE = {
    bool: bool_,
    int: int64,
    float: float64,
    complex: complex64,
}

_NUMPY_DEFAULT_INT_DTYPE = int64
_NUMPY_DEFAULT_UINT_DTYPE = uint64
_NUMPY_DEFAULT_FLOAT_DTYPE = float64
_NUMPY_DEFAULT_COMPLEX_DTYPE = complex64

_QuantileMethod = Literal["linear", "lower", "higher", "nearest", "midpoint"]

_NO_VALUE = object()
_FALLBACK_ATTR_CACHE: dict[str, Any] = {}
FallbackDisabledError = bridge.FallbackDisabledError
strict_fallbacks_enabled = bridge.strict_fallbacks_enabled
set_strict_fallbacks = bridge.set_strict_fallbacks
_numpy = bridge.numpy_module
_numpy_coerce_value = bridge.coerce_to_numpy
_numpy_wrap_value = bridge.wrap_from_numpy
_numpy_callable = bridge.numpy_callable
_hybrid_mx_numpy_callable = bridge.hybrid_callable


def _numpy_dtype(dtype: Any | None) -> Any | None:
    if dtype is None:
        return None
    np_mod = _numpy()
    resolved = _resolve_dtype(dtype)
    try:
        return np_mod.dtype(str(resolved).removeprefix("mlx.core."))
    except (TypeError, ValueError):
        return resolved


def _resolve_dtype(dtype: Any | None) -> Any | None:
    if dtype is None:
        return None
    if hasattr(dtype, "dtype"):
        maybe = dtype.dtype
        if isinstance(maybe, str):
            return _DTYPE_BY_NAME.get(maybe, dtype)
    if dtype in _PYTYPE_TO_DTYPE:
        return _PYTYPE_TO_DTYPE[dtype]
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[key]
    for candidate in (getattr(dtype, "name", None), getattr(dtype, "__name__", None), str(dtype)):
        if not isinstance(candidate, str):
            continue
        key = candidate.strip().lower()
        if key.startswith("mlx.core."):
            key = key.removeprefix("mlx.core.")
        if key.startswith("<class '") and key.endswith("'>"):
            key = key[8:-2]
            if "." in key:
                key = key.rsplit(".", 1)[-1]
        if key in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[key]
    return dtype


def _default_dtype_for_python_scalar(value: Any) -> Any | None:
    if type(value) in _PYTYPE_TO_DTYPE:
        return _PYTYPE_TO_DTYPE[type(value)]
    return None


_SIGNED_INT_DTYPES = frozenset({int8, int16, int32, int64})
_UNSIGNED_INT_DTYPES = frozenset({uint8, uint16, uint32, uint64})
_FLOATING_DTYPES = frozenset({float16, bfloat16, float32, float64})
_COMPLEX_DTYPES = frozenset(
    {
        dt
        for dt in (
            complex64,
            getattr(mx, "complex128", None),
        )
        if dt is not None
    },
)


def _dtype_for_checks(dt: Any) -> Any:
    return _resolve_dtype(dt)


def _is_bool_dtype(dt: Any) -> bool:
    return _dtype_for_checks(dt) is bool_


def _is_unsigned_int_dtype(dt: Any) -> bool:
    return _dtype_for_checks(dt) in _UNSIGNED_INT_DTYPES


def _is_signed_int_dtype(dt: Any) -> bool:
    return _dtype_for_checks(dt) in _SIGNED_INT_DTYPES


def _is_integer_dtype(dt: Any) -> bool:
    return _is_signed_int_dtype(dt) or _is_unsigned_int_dtype(dt)


def _is_floating_or_bfloat_dtype(dt: Any) -> bool:
    return _dtype_for_checks(dt) in _FLOATING_DTYPES


def _is_complex_dtype(dt: Any) -> bool:
    return _dtype_for_checks(dt) in _COMPLEX_DTYPES


def _is_inexact_dtype(dt: Any) -> bool:
    return _is_floating_or_bfloat_dtype(dt) or _is_complex_dtype(dt)


def _float_eps_for_dtype(dtype: Any) -> float:
    dt = _dtype_for_checks(dtype)
    complex128_dtype = getattr(mx, "complex128", None)
    if dt is complex64:
        dt = float32
    elif complex128_dtype is not None and dt is complex128_dtype:
        dt = float64
    eps_by_dtype = {
        float16: 9.765625e-04,
        bfloat16: 7.8125e-03,
        float32: 1.1920928955078125e-07,
        float64: 2.220446049250313e-16,
    }
    if dt in eps_by_dtype:
        return eps_by_dtype[dt]
    msg = f"Unsupported floating dtype for epsilon: {dtype!r}"
    raise TypeError(msg)


def _needs_cpu_default_device_for_dtype(dt: Any | None) -> bool:
    return dt in {_NUMPY_DEFAULT_INT_DTYPE, _NUMPY_DEFAULT_UINT_DTYPE, _NUMPY_DEFAULT_FLOAT_DTYPE}


@contextmanager
def _cpu_default_device_for_dtype(dt: Any | None):
    if not _needs_cpu_default_device_for_dtype(dt) or mx.default_device() != mx.gpu:
        yield
        return
    previous = mx.default_device()
    mx.set_default_device(cast("Any", mx.cpu))
    try:
        yield
    finally:
        mx.set_default_device(previous)


@contextmanager
def _cpu_default_device_for_dtypes(*dtypes: Any | None):
    if not builtins.any(_needs_cpu_default_device_for_dtype(dt) for dt in dtypes) or mx.default_device() != mx.gpu:
        yield
        return
    previous = mx.default_device()
    mx.set_default_device(cast("Any", mx.cpu))
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _infer_default_dtype_from_python_sequence(obj: Any) -> Any | None:
    has_bool = False
    has_int = False
    has_float = False
    has_complex = False

    def _visit(value: Any) -> bool:
        nonlocal has_bool, has_int, has_float, has_complex
        scalar_dtype = _default_dtype_for_python_scalar(value)
        if scalar_dtype is bool_:
            has_bool = True
            return True
        if scalar_dtype is _NUMPY_DEFAULT_INT_DTYPE:
            has_int = True
            return True
        if scalar_dtype is _NUMPY_DEFAULT_FLOAT_DTYPE:
            has_float = True
            return True
        if scalar_dtype is _NUMPY_DEFAULT_COMPLEX_DTYPE:
            has_complex = True
            return True
        if isinstance(value, (list, tuple)):
            return builtins.all(_visit(item) for item in value)
        return False

    if not isinstance(obj, (list, tuple)):
        return None
    if not _visit(obj):
        return None
    if has_complex:
        return _NUMPY_DEFAULT_COMPLEX_DTYPE
    if has_float:
        return _NUMPY_DEFAULT_FLOAT_DTYPE
    if has_int:
        return _NUMPY_DEFAULT_INT_DTYPE
    if has_bool:
        return bool_
    return None


def _infer_default_array_dtype_like_numpy(obj: Any) -> Any | None:
    if isinstance(obj, mx.array):
        return obj.dtype

    scalar_dtype = _default_dtype_for_python_scalar(obj)
    if scalar_dtype is not None:
        return scalar_dtype

    obj_dtype = getattr(obj, "dtype", None)
    if obj_dtype is not None:
        resolved = _resolve_dtype(obj_dtype)
        if resolved is not None:
            return resolved

    seq_dtype = _infer_default_dtype_from_python_sequence(obj)
    if seq_dtype is not None:
        return seq_dtype

    if isinstance(obj, (list, tuple)):
        try:
            np_mod = _numpy()
        except (NotImplementedError, FallbackDisabledError):
            return None
        try:
            inferred = np_mod.asarray(obj).dtype
        except (TypeError, ValueError):
            return None
        resolved = _resolve_dtype(inferred)
        if resolved is not None:
            return resolved
    return None


def _infer_arange_default_dtype(start: Any, stop: Any | None, step: Any | None) -> Any:
    args = [arg for arg in (start, stop, step) if arg is not None]
    has_complex = False
    has_float = False
    for arg in args:
        scalar_dtype = _default_dtype_for_python_scalar(arg)
        dt = scalar_dtype if scalar_dtype is not None else _resolve_dtype(getattr(arg, "dtype", None))
        if dt is None:
            continue
        if _is_complex_dtype(dt):
            has_complex = True
            break
        if _is_floating_or_bfloat_dtype(dt):
            has_float = True
    if has_complex:
        return _NUMPY_DEFAULT_COMPLEX_DTYPE
    if has_float:
        return _NUMPY_DEFAULT_FLOAT_DTYPE
    return _NUMPY_DEFAULT_INT_DTYPE


def dtype(spec: Any) -> Any:
    if hasattr(spec, "dtype"):
        dt = spec.dtype
        name = str(dt)
        if name.startswith("mlx.core."):
            return _DTYPE_BY_NAME.get(name.removeprefix("mlx.core."), dt)
        return dt
    return _resolve_dtype(spec)


def finfo(dtype_spec: Any) -> Any:
    np_mod = bridge.numpy_module()
    np_dtype = _numpy_dtype(dtype_spec)
    if np_dtype is None:
        return np_mod.finfo(dtype_spec)
    return np_mod.finfo(np_dtype)


def iinfo(dtype_spec: Any) -> Any:
    np_mod = bridge.numpy_module()
    np_dtype = _numpy_dtype(dtype_spec)
    if np_dtype is None:
        return np_mod.iinfo(dtype_spec)
    return np_mod.iinfo(np_dtype)


def _asarray(a: Any, dtype_: Any | None = None) -> mx.array:
    return mx.asarray(a, dtype=_resolve_dtype(dtype_))


def _normalize_shape_arg(shape: Any) -> Any:
    if isinstance(shape, list):
        return tuple(shape)
    return shape


def _normalize_axes(axes: int | Sequence[int] | None, ndim_: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim_))
    if isinstance(axes, int):
        axes = (axes,)
    result: list[int] = []
    for axis in axes:
        ax = axis + ndim_ if axis < 0 else axis
        if not 0 <= ax < ndim_:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim_}"
            raise ValueError(msg)
        if ax in result:
            msg = "repeated axis"
            raise ValueError(msg)
        result.append(ax)
    return tuple(result)


def _scalar_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, mx.array):
        return bool(x.item())
    if hasattr(x, "item"):
        return bool(x.item())
    return bool(x)


def _check_order(order: str | None) -> None:
    if order is None:
        return
    if order not in {"C", "F", "A", "K"}:
        msg = f"Unsupported order {order!r}"
        raise ValueError(msg)


def _tolist_1d_list(arr: mx.array) -> list[Any]:
    return cast("list[Any]", arr.tolist())


def _narrow_quantile_method(method: str) -> _QuantileMethod:
    if method not in {"linear", "lower", "higher", "nearest", "midpoint"}:
        msg = f"unsupported quantile method {method!r}"
        raise ValueError(msg)
    return cast("_QuantileMethod", method)


def _normalize_roll_args(
    shift: int | Sequence[int],
    axis: int | Sequence[int] | None,
) -> tuple[int | tuple[int, ...], int | tuple[int, ...] | None]:
    if isinstance(shift, int):
        shift_arg: int | tuple[int, ...] = shift
    else:
        shift_arg = tuple(int(s) for s in shift)
    if axis is None:
        axis_arg: int | tuple[int, ...] | None = None
    elif isinstance(axis, int):
        axis_arg = axis
    else:
        axis_arg = tuple(int(a) for a in axis)
    return shift_arg, axis_arg


def _prefix_dims(arr: mx.array, ndmin: int) -> mx.array:
    if arr.ndim >= ndmin:
        return arr
    new_shape = (1,) * (ndmin - arr.ndim) + tuple(arr.shape)
    return mx.reshape(arr, new_shape)


def _prod_int(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return int(out)


def _normalize_axis_index(axis: int, ndim_: int) -> int:
    return _normalize_axes((axis,), ndim_)[0]


def _python_scalar_key(value: Any) -> Any:
    # Preserve NumPy-like set semantics for NaN values when using Python maps/sets.
    if isinstance(value, float):
        if math.isnan(value):
            return ("float_nan",)
        return ("float", value)
    if isinstance(value, complex):
        r = ("float_nan",) if math.isnan(value.real) else ("float", value.real)
        i = ("float_nan",) if math.isnan(value.imag) else ("float", value.imag)
        return ("complex", r, i)
    return ("other", value)


def _slice_along_axis(arr: mx.array, axis: int, start: int | None = None, stop: int | None = None) -> mx.array:
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(start, stop)
    return arr[tuple(idx)]


def _item_from_scalar_like(value: Any) -> Any:
    if isinstance(value, mx.array):
        if value.ndim != 0:
            return _NO_VALUE
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return _NO_VALUE
    return _NO_VALUE


def _coerce_exact_scalar_int(value: Any, *, allow_bool: bool) -> int | None:
    if isinstance(value, bool):
        return int(value) if allow_bool else None
    if isinstance(value, int):
        return int(value)
    item = _item_from_scalar_like(value)
    if item is _NO_VALUE:
        return None
    if isinstance(item, bool):
        return int(item) if allow_bool else None
    if isinstance(item, int):
        return int(item)
    return None


def _equal_with_nan(a: Any, b: Any) -> mx.array:
    eq = equal(a, b)
    try:
        return logical_or(eq, logical_and(isnan(a), isnan(b)))
    except (TypeError, ValueError):
        return eq


def _can_broadcast_to_shape(src_shape: Sequence[int], target_shape: Sequence[int]) -> bool:
    src = tuple(int(s) for s in src_shape)
    tgt = tuple(int(s) for s in target_shape)
    if len(src) > len(tgt):
        return False
    src = (1,) * (len(tgt) - len(src)) + src
    return builtins.all(s in (1, t) for s, t in zip(src, tgt, strict=False))


def _ensure_dtype_for_reduction(a: Any, dtype_: Any | None) -> mx.array:
    arr = _asarray(a)
    if dtype_ is not None:
        resolved = _resolve_dtype(dtype_)
        if resolved is not None:
            arr = arr.astype(resolved)
    return arr


def _default_sum_prod_dtype(arr: mx.array, dtype_: Any | None) -> Any | None:
    resolved = _resolve_dtype(dtype_)
    if resolved is not None:
        return resolved
    dt = arr.dtype
    if _is_bool_dtype(dt) or _is_signed_int_dtype(dt):
        return _NUMPY_DEFAULT_INT_DTYPE
    if _is_unsigned_int_dtype(dt) and dt is not _NUMPY_DEFAULT_UINT_DTYPE:
        return _NUMPY_DEFAULT_UINT_DTYPE
    return None


def _default_mean_var_dtype(arr: mx.array, dtype_: Any | None) -> Any | None:
    resolved = _resolve_dtype(dtype_)
    if resolved is not None:
        return resolved
    dt = arr.dtype
    if _is_bool_dtype(dt) or _is_integer_dtype(dt):
        return _NUMPY_DEFAULT_FLOAT_DTYPE
    return None


def _default_float_output_dtype() -> Any:
    return _NUMPY_DEFAULT_FLOAT_DTYPE


def _default_int_output_dtype() -> Any:
    return _NUMPY_DEFAULT_INT_DTYPE


def _default_uint_output_dtype() -> Any:
    return _NUMPY_DEFAULT_UINT_DTYPE


def _coerce_public_output_dtype(arr: mx.array, out_dtype: Any) -> mx.array:
    if arr.dtype is out_dtype:
        return arr
    with _cpu_default_device_for_dtypes(arr.dtype, out_dtype):
        return arr.astype(out_dtype)


def _coerce_public_default_float_output(arr: mx.array) -> mx.array:
    return _coerce_public_output_dtype(arr, _NUMPY_DEFAULT_FLOAT_DTYPE)


def _coerce_public_default_int_output(arr: mx.array) -> mx.array:
    return _coerce_public_output_dtype(arr, _NUMPY_DEFAULT_INT_DTYPE)


def _coerce_public_default_complex_output(arr: mx.array) -> mx.array:
    return _coerce_public_output_dtype(arr, _NUMPY_DEFAULT_COMPLEX_DTYPE)


def _count_dtype_for_nan_stats(arr_dtype: Any) -> Any:
    if arr_dtype is mx.float64:
        return mx.float64
    return mx.float32


def _nan_scalar_for_dtype(arr_dtype: Any) -> mx.array:
    return mx.array(float("nan"), dtype=arr_dtype)


def _prepare_split_indices(length: int, sections: int) -> list[int]:
    if sections <= 0:
        msg = "number of sections must be >= 1"
        raise ValueError(msg)
    q, r = builtins.divmod(length, sections)
    indices: list[int] = []
    running = 0
    for i in range(sections - 1):
        running += q + (1 if i < r else 0)
        indices.append(running)
    return indices


def _slices_for_diff(axis: int, ndim_: int) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    first = [slice(None)] * ndim_
    second = [slice(None)] * ndim_
    first[axis] = slice(1, None)
    second[axis] = slice(None, -1)
    return tuple(first), tuple(second)


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _FALLBACK_ATTR_CACHE:
        return _FALLBACK_ATTR_CACHE[name]
    try:
        np_mod = bridge.numpy_module()
    except (NotImplementedError, FallbackDisabledError):
        np_mod = None
    if hasattr(mx, name):
        obj = getattr(mx, name)
        if (
            np_mod is not None
            and callable(obj)
            and not isinstance(obj, type)
            and hasattr(np_mod, name)
            and callable(getattr(np_mod, name))
            and not isinstance(getattr(np_mod, name), type)
        ):
            wrapped = bridge.hybrid_callable(name, obj, getattr(np_mod, name), namespace="core")
            _FALLBACK_ATTR_CACHE[name] = wrapped
            return wrapped
        _FALLBACK_ATTR_CACHE[name] = obj
        return obj
    if np_mod is not None and hasattr(np_mod, name):
        obj = getattr(np_mod, name)
        if callable(obj) and not isinstance(obj, type):
            wrapped = bridge.numpy_callable(name, obj, namespace="core")
            _FALLBACK_ATTR_CACHE[name] = wrapped
            return wrapped
        _FALLBACK_ATTR_CACHE[name] = obj
        return obj
    msg = f"module 'mumpy' has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    names = set(__all__) | {n for n in dir(mx) if not n.startswith("_")}
    with suppress(NotImplementedError, FallbackDisabledError):
        names |= {n for n in dir(bridge.numpy_module()) if not n.startswith("_")}
    return sorted(names)


def array(
    obj: Any,
    dtype: Any | None = None,
    copy: bool = True,
    order: str = "C",
    ndmin: int = 0,
) -> mx.array:
    _check_order(order)
    dtype_resolved = _resolve_dtype(dtype) or _infer_default_array_dtype_like_numpy(obj)
    if order in {"F", "A", "K"}:
        bridge.record_fallback(f"core.array:order_{order}")
        np_mod = _numpy()
        arr_np = np_mod.array(obj, dtype=_numpy_dtype(dtype_resolved), copy=copy, order=cast("Any", order), ndmin=ndmin)
        return mx.array(arr_np)
    arr = mx.array(obj, dtype=dtype_resolved) if copy else mx.asarray(obj, dtype=dtype_resolved)
    return _prefix_dims(arr, ndmin)


def asarray(a: Any, dtype: Any | None = None, order: str | None = None) -> mx.array:
    _check_order(order)
    dtype_resolved = _resolve_dtype(dtype) or _infer_default_array_dtype_like_numpy(a)
    if order in {"F", "A", "K"}:
        bridge.record_fallback(f"core.asarray:order_{order}")
        np_mod = _numpy()
        return mx.array(np_mod.asarray(a, dtype=_numpy_dtype(dtype_resolved), order=cast("Any", order)))
    return mx.asarray(a, dtype=dtype_resolved)


def asanyarray(a: Any, dtype: Any | None = None, order: str | None = None) -> mx.array:
    return asarray(a, dtype=dtype, order=order)


def ascontiguousarray(a: Any, dtype: Any | None = None) -> mx.array:
    return asarray(a, dtype=dtype)


def asfortranarray(a: Any, dtype: Any | None = None) -> mx.array:
    return asarray(a, dtype=dtype, order="F")


def asarray_chkfinite(a: Any, dtype: Any | None = None, order: str | None = None) -> mx.array:
    arr = asarray(a, dtype=dtype, order=order)
    if not _scalar_bool(all(isfinite(arr))):
        msg = "array must not contain infs or NaNs"
        raise ValueError(msg)
    return arr


def copy(a: Any) -> mx.array:
    # MLX arrays are immutable, but NumPy exposes a copy helper; re-materializing
    # the array preserves the expectation of a new object.
    return mx.array(_asarray(a))


def astype(a: Any, dtype: Any, copy: bool = True) -> mx.array:
    arr = _asarray(a)
    dtype_ = _resolve_dtype(dtype)
    if not copy and arr.dtype == dtype_:
        return arr
    if dtype_ is None:
        msg = "dtype must not be None"
        raise TypeError(msg)
    with _cpu_default_device_for_dtypes(arr.dtype, dtype_):
        return arr.astype(dtype_)


def zeros(shape: Any, dtype: Any | None = None) -> mx.array:
    out_dtype = _resolve_dtype(dtype) or _NUMPY_DEFAULT_FLOAT_DTYPE
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.zeros(_normalize_shape_arg(shape), dtype=out_dtype)


def ones(shape: Any, dtype: Any | None = None) -> mx.array:
    out_dtype = _resolve_dtype(dtype) or _NUMPY_DEFAULT_FLOAT_DTYPE
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.ones(_normalize_shape_arg(shape), dtype=out_dtype)


def full(shape: Any, fill_value: Any, dtype: Any | None = None) -> mx.array:
    out_dtype = _resolve_dtype(dtype) or _infer_default_array_dtype_like_numpy(fill_value)
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.full(_normalize_shape_arg(shape), fill_value, dtype=out_dtype)


def empty(shape: Any, dtype: Any | None = None) -> mx.array:
    # MLX does not expose uninitialized allocation. Use zeros as a deterministic fallback.
    return zeros(shape, dtype=dtype)


def zeros_like(a: Any, dtype: Any | None = None, order: str = "K", shape: Any | None = None) -> mx.array:
    _check_order(None if order == "K" else order)
    arr = _asarray(a)
    out_shape = arr.shape if shape is None else _normalize_shape_arg(shape)
    out_dtype = _resolve_dtype(dtype) or arr.dtype
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.zeros(out_shape, dtype=out_dtype)


def ones_like(a: Any, dtype: Any | None = None, order: str = "K", shape: Any | None = None) -> mx.array:
    _check_order(None if order == "K" else order)
    arr = _asarray(a)
    out_shape = arr.shape if shape is None else _normalize_shape_arg(shape)
    out_dtype = _resolve_dtype(dtype) or arr.dtype
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.ones(out_shape, dtype=out_dtype)


def full_like(
    a: Any,
    fill_value: Any,
    dtype: Any | None = None,
    order: str = "K",
    shape: Any | None = None,
) -> mx.array:
    _check_order(None if order == "K" else order)
    arr = _asarray(a)
    out_shape = arr.shape if shape is None else _normalize_shape_arg(shape)
    out_dtype = _resolve_dtype(dtype) or arr.dtype
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.full(out_shape, fill_value, dtype=out_dtype)


def empty_like(a: Any, dtype: Any | None = None, order: str = "K", shape: Any | None = None) -> mx.array:
    return zeros_like(a, dtype=dtype, order=order, shape=shape)


def arange(
    start: Any,
    stop: Any | None = None,
    step: Any | None = None,
    dtype: Any | None = None,
) -> mx.array:
    dtype_ = _resolve_dtype(dtype) or _infer_arange_default_dtype(start, stop, step)
    with _cpu_default_device_for_dtype(dtype_):
        if stop is None:
            return mx.arange(start, step=step, dtype=dtype_)
        return mx.arange(start, stop, step, dtype=dtype_)


@overload
def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: Any | None = None,
) -> mx.array: ...


@overload
def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: Literal[True] = True,
    dtype: Any | None = None,
) -> tuple[mx.array, mx.array]: ...


def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: Any | None = None,
) -> mx.array | tuple[mx.array, mx.array]:
    if num < 0:
        msg = "Number of samples, num, must be non-negative"
        raise ValueError(msg)
    dtype_ = _resolve_dtype(dtype) or _NUMPY_DEFAULT_FLOAT_DTYPE
    with _cpu_default_device_for_dtype(dtype_):
        if endpoint:
            arr = mx.linspace(start, stop, num=num, dtype=dtype_)
            step_val = (stop - start) / (num - 1) if num > 1 else mx.nan
        elif num == 0:
            arr = mx.linspace(0, 0, num=0, dtype=dtype_)
            step_val = mx.nan
        else:
            step_val = (stop - start) / num
            effective_stop = start + step_val * (num - 1)
            arr = mx.linspace(start, effective_stop, num=num, dtype=dtype_)
    if retstep:
        return arr, mx.array(step_val, dtype=dtype_)
    return arr


def logspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    base: Any = 10.0,
    dtype: Any | None = None,
) -> mx.array:
    exponents = linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)
    with _cpu_default_device_for_dtype(getattr(exponents, "dtype", None)):
        return power(base, exponents)


def geomspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    dtype: Any | None = None,
    axis: int = 0,
) -> mx.array:
    if num < 0:
        msg = "Number of samples, num, must be non-negative"
        raise ValueError(msg)
    dtype_res = _resolve_dtype(dtype)
    start_b, stop_b = broadcast_arrays(_asarray(start), _asarray(stop))
    if _scalar_bool(any(equal(start_b, 0))) or _scalar_bool(any(equal(stop_b, 0))):
        msg = "Geometric sequence cannot include zero"
        raise ValueError(msg)
    out_ndim = start_b.ndim + 1
    ax = axis + out_ndim if axis < 0 else axis
    if not 0 <= ax < out_ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {out_ndim}"
        raise ValueError(msg)

    need_complex = _is_complex_dtype(start_b.dtype) or _is_complex_dtype(stop_b.dtype)
    if dtype_res is not None and _is_complex_dtype(dtype_res):
        need_complex = True
    if need_complex:
        if not _is_complex_dtype(start_b.dtype):
            start_b = start_b.astype(mx.complex64)
        if not _is_complex_dtype(stop_b.dtype):
            stop_b = stop_b.astype(mx.complex64)

    target_dtype = dtype_res
    if target_dtype is None and not need_complex:
        target_dtype = _NUMPY_DEFAULT_FLOAT_DTYPE

    t_dtype = mx.float64 if target_dtype is mx.float64 else mx.float32
    with _cpu_default_device_for_dtype(target_dtype):
        t = linspace(0.0, 1.0, num=num, endpoint=endpoint, dtype=t_dtype)
        t_shape = [1] * out_ndim
        t_shape[ax] = num
        t = reshape(t, t_shape)
        start_e = expand_dims(start_b, ax)
        stop_e = expand_dims(stop_b, ax)
        out = start_e * power(stop_e / start_e, t)
        if target_dtype is not None and out.dtype != target_dtype:
            out = out.astype(target_dtype)
        return out


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: Any | None = None,
) -> mx.array:
    m_val = N if M is None else M
    out_dtype = _resolve_dtype(dtype) or _NUMPY_DEFAULT_FLOAT_DTYPE
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.eye(N, m=m_val, k=k, dtype=out_dtype)


def identity(n: int, dtype: Any | None = None) -> mx.array:
    out_dtype = _resolve_dtype(dtype) or _NUMPY_DEFAULT_FLOAT_DTYPE
    with _cpu_default_device_for_dtype(out_dtype):
        return mx.identity(n, dtype=out_dtype)


def tri(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: Any | None = None,
) -> mx.array:
    m_val = N if M is None else M
    out_dtype = _resolve_dtype(dtype) or _NUMPY_DEFAULT_FLOAT_DTYPE
    with _cpu_default_device_for_dtype(out_dtype):
        return tril(ones((N, m_val), dtype=out_dtype), k=k)


def diag(v: Any, k: int = 0) -> mx.array:
    arr = _asarray(v)
    with _cpu_default_device_for_dtype(arr.dtype):
        return mx.diag(arr, k=k)


def diagflat(v: Any, k: int = 0) -> mx.array:
    arr = flatten(_asarray(v))
    n = int(arr.size)
    size_ = n + builtins.abs(k)
    out = zeros((size_, size_), dtype=arr.dtype)
    if n == 0:
        return out
    if k >= 0:
        rows = arange(0, n, dtype=int32)
        cols = rows + k
    else:
        cols = arange(0, n, dtype=int32)
        rows = cols + (-k)
    out[rows, cols] = arr
    return out


def reshape(a: Any, newshape: Any, order: str = "C") -> mx.array:
    _check_order(order)
    if order != "C":
        bridge.record_fallback(f"core.reshape:order_{order}")
        np_mod = _numpy()
        result = np_mod.reshape(np_mod.asarray(a), _normalize_shape_arg(newshape), order=cast("Any", order))
        return mx.array(result)
    return mx.reshape(_asarray(a), _normalize_shape_arg(newshape))


def transpose(a: Any, axes: Sequence[int] | None = None) -> mx.array:
    return mx.transpose(_asarray(a), axes=axes)


def matrix_transpose(x: Any) -> mx.array:
    arr = _asarray(x)
    if arr.ndim < 2:
        msg = "Input array must be at least 2-dimensional"
        raise ValueError(msg)
    axes = list(range(arr.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return transpose(arr, axes)


def permute_dims(a: Any, axes: Sequence[int] | None = None) -> mx.array:
    return mx.permute_dims(_asarray(a), axes=axes)


def swapaxes(a: Any, axis1: int, axis2: int) -> mx.array:
    return mx.swapaxes(_asarray(a), axis1, axis2)


def moveaxis(a: Any, source: int | Sequence[int], destination: int | Sequence[int]) -> mx.array:
    arr = _asarray(a)
    if isinstance(source, int) and isinstance(destination, int):
        return mx.moveaxis(arr, source, destination)
    if isinstance(source, int) or isinstance(destination, int):
        msg = "source and destination must both be integers or both be sequences"
        raise TypeError(msg)
    if len(source) != len(destination):
        msg = "source and destination arguments must have the same number of elements"
        raise ValueError(msg)
    src = _normalize_axes(tuple(source), arr.ndim)
    dst = _normalize_axes(tuple(destination), arr.ndim)
    order = [ax for ax in range(arr.ndim) if ax not in src]
    for d, s in sorted(zip(dst, src, strict=False), key=lambda x: x[0]):
        order.insert(d, s)
    return transpose(arr, order)


def rollaxis(a: Any, axis: int, start: int = 0) -> mx.array:
    arr = _asarray(a)
    ax = _normalize_axis_index(axis, arr.ndim)
    if start < 0:
        start += arr.ndim
    if start < 0 or start > arr.ndim:
        msg = "start out of range"
        raise ValueError(msg)
    if ax < start:
        start -= 1
    return moveaxis(arr, ax, start)


def expand_dims(a: Any, axis: int | Sequence[int]) -> mx.array:
    arr = _asarray(a)
    if isinstance(axis, int):
        return mx.expand_dims(arr, axis)
    out = arr
    ndim_target = arr.ndim + len(axis)
    normalized = []
    for ax in axis:
        pos = ax + ndim_target if ax < 0 else ax
        if not 0 <= pos <= ndim_target - 1:
            msg = "axis out of bounds"
            raise ValueError(msg)
        normalized.append(pos)
    for ax in sorted(normalized):
        out = mx.expand_dims(out, ax)
    return out


def squeeze(a: Any, axis: int | Sequence[int] | None = None) -> mx.array:
    return mx.squeeze(_asarray(a), axis=axis)


def flatten(a: Any, order: str = "C") -> mx.array:
    _check_order(order)
    if order != "C":
        bridge.record_fallback(f"core.flatten:order_{order}")
        np_mod = _numpy()
        return mx.array(np_mod.asarray(a).flatten(order=cast("Any", order)))
    return mx.flatten(_asarray(a))


def ravel(a: Any, order: str = "C") -> mx.array:
    _check_order(order)
    if order != "C":
        bridge.record_fallback(f"core.ravel:order_{order}")
        np_mod = _numpy()
        return mx.array(np_mod.ravel(np_mod.asarray(a), order=cast("Any", order)))
    return flatten(a, order=order)


def atleast_1d(*arys: Any) -> mx.array | list[mx.array]:
    outs = []
    for ary in arys:
        arr = _asarray(ary)
        if arr.ndim == 0:
            arr = reshape(arr, (1,))
        outs.append(arr)
    return outs[0] if len(outs) == 1 else outs


def atleast_2d(*arys: Any) -> mx.array | list[mx.array]:
    outs = []
    for ary in arys:
        arr = _asarray(ary)
        if arr.ndim == 0:
            arr = reshape(arr, (1, 1))
        elif arr.ndim == 1:
            arr = reshape(arr, (1, arr.shape[0]))
        outs.append(arr)
    return outs[0] if len(outs) == 1 else outs


def atleast_3d(*arys: Any) -> mx.array | list[mx.array]:
    outs = []
    for ary in arys:
        arr = _asarray(ary)
        if arr.ndim == 0:
            arr = reshape(arr, (1, 1, 1))
        elif arr.ndim == 1:
            arr = reshape(arr, (1, arr.shape[0], 1))
        elif arr.ndim == 2:
            arr = reshape(arr, (*arr.shape, 1))
        outs.append(arr)
    return outs[0] if len(outs) == 1 else outs


def _atleast_1d_one(a: Any) -> mx.array:
    out = atleast_1d(a)
    if isinstance(out, list):
        return out[0]
    return out


def _atleast_2d_one(a: Any) -> mx.array:
    out = atleast_2d(a)
    if isinstance(out, list):
        return out[0]
    return out


def concatenate(arrays: Sequence[Any], axis: int | None = 0, dtype: Any | None = None) -> mx.array:
    if not arrays:
        msg = "need at least one array to concatenate"
        raise ValueError(msg)
    converted: list[mx.array] = [_asarray(a, dtype_=dtype) if dtype is not None else _asarray(a) for a in arrays]
    if axis is None:
        converted = [flatten(a) for a in converted]
        axis = 0
    return mx.concatenate(converted, axis=axis)


def stack(arrays: Sequence[Any], axis: int = 0, dtype: Any | None = None) -> mx.array:
    if not arrays:
        msg = "need at least one array to stack"
        raise ValueError(msg)
    converted: list[mx.array] = [_asarray(a, dtype_=dtype) if dtype is not None else _asarray(a) for a in arrays]
    return mx.stack(converted, axis=axis)


def hstack(tup: Sequence[Any]) -> mx.array:
    arrs: list[mx.array] = [_atleast_1d_one(a) for a in tup]
    if not arrs:
        msg = "need at least one array to stack"
        raise ValueError(msg)
    axis = 0 if arrs[0].ndim == 1 else 1
    return concatenate(arrs, axis=axis)


def vstack(tup: Sequence[Any]) -> mx.array:
    arrs: list[mx.array] = [_atleast_2d_one(a) for a in tup]
    if not arrs:
        msg = "need at least one array to stack"
        raise ValueError(msg)
    return concatenate(arrs, axis=0)


def row_stack(tup: Sequence[Any]) -> mx.array:
    return vstack(tup)


def dstack(tup: Sequence[Any]) -> mx.array:
    arrs = [atleast_3d(a) for a in tup]
    if not arrs:
        msg = "need at least one array to stack"
        raise ValueError(msg)
    return concatenate(arrs, axis=2)


def column_stack(tup: Sequence[Any]) -> mx.array:
    arrs = []
    for a in tup:
        arr = _asarray(a)
        if arr.ndim == 1:
            arr = reshape(arr, (arr.shape[0], 1))
        arrs.append(arr)
    if not arrs:
        msg = "need at least one array to stack"
        raise ValueError(msg)
    return concatenate(arrs, axis=1)


def append(arr: Any, values: Any, axis: int | None = None) -> mx.array:
    return concatenate([arr, values], axis=axis)


def insert(arr: Any, obj: Any, values: Any, axis: int | None = None) -> mx.array:
    out = _asarray(arr)
    if axis is None:
        out = ravel(out)
        ax = 0
    else:
        ax = _normalize_axis_index(axis, out.ndim)
    n = int(out.shape[ax])

    scalar_obj = False
    if isinstance(obj, slice):
        raw_positions = list(range(*obj.indices(n)))
    elif isscalar(obj) or (isinstance(obj, mx.array) and obj.ndim == 0):
        scalar_obj = True
        raw_positions = [int(_asarray(obj).item())]
    else:
        obj_arr = ravel(_asarray(obj))
        if obj_arr.dtype == mx.bool_:
            raw_positions = [i for i, flag in enumerate(_tolist_1d_list(obj_arr)) if flag]
        else:
            raw_positions = [int(v) for v in _tolist_1d_list(obj_arr)]

    positions: list[int] = []
    for pos in raw_positions:
        adj_pos = pos
        if pos < 0:
            adj_pos += n
        positions.append(builtins.min(builtins.max(adj_pos, 0), n))

    vals = _asarray(values)
    if scalar_obj:
        pos = positions[0]
        if out.ndim == 1:
            vals_block = ravel(vals)
        else:
            target_shape = list(out.shape)
            slice_shape = tuple(int(out.shape[i]) for i in range(out.ndim) if i != ax)
            if vals.ndim == 0:
                target_shape[ax] = 1
                vals_block = full(tuple(target_shape), vals.item(), dtype=vals.dtype)
            elif vals.ndim == out.ndim - 1 and tuple(int(s) for s in vals.shape) == slice_shape:
                vals_block = expand_dims(vals, ax)
            elif vals.ndim == out.ndim:
                target_shape[ax] = int(vals.shape[ax])
                if not _can_broadcast_to_shape(tuple(int(s) for s in vals.shape), tuple(target_shape)):
                    msg = "could not broadcast values for insert"
                    raise ValueError(msg)
                vals_block = broadcast_to(vals, tuple(target_shape))
            else:
                msg = "could not broadcast values for insert"
                raise ValueError(msg)
        parts: list[mx.array] = []
        if pos > 0:
            parts.append(_slice_along_axis(out, ax, 0, pos))
        parts.append(vals_block)
        if pos < n:
            parts.append(_slice_along_axis(out, ax, pos, None))
        if len(parts) == 1:
            return parts[0]
        return concatenate(parts, axis=ax)

    if not positions:
        return out
    m = len(positions)
    if out.ndim == 1:
        vals_seq = ravel(vals)
        if vals_seq.ndim == 0 or vals_seq.shape[0] == 1:
            vals_seq = broadcast_to(ravel(vals_seq), (m,))
        elif int(vals_seq.shape[0]) != m:
            msg = "could not broadcast values for insert"
            raise ValueError(msg)
    else:
        target_shape = list(out.shape)
        target_shape[ax] = m
        target_shape_t = tuple(int(s) for s in target_shape)
        slice_shape = tuple(int(out.shape[i]) for i in range(out.ndim) if i != ax)
        if vals.ndim == 0:
            vals_seq = full(target_shape_t, vals.item(), dtype=vals.dtype)
        elif vals.ndim == out.ndim and _can_broadcast_to_shape(tuple(int(s) for s in vals.shape), target_shape_t):
            vals_seq = broadcast_to(vals, target_shape_t)
        elif vals.ndim == out.ndim - 1 and tuple(int(s) for s in vals.shape) == slice_shape:
            vals_seq = broadcast_to(expand_dims(vals, ax), target_shape_t)
        elif vals.ndim == out.ndim and int(vals.shape[0]) == m and tuple(int(s) for s in vals.shape[1:]) == slice_shape:
            vals_seq = moveaxis(vals, 0, ax)
        else:
            msg = "could not broadcast values for insert"
            raise ValueError(msg)

    buckets: list[list[mx.array]] = [[] for _ in range(n + 1)]
    for i, pos in enumerate(positions):
        buckets[pos].append(_slice_along_axis(vals_seq, ax, i, i + 1))

    parts2: list[mx.array] = []
    for pos in range(n + 1):
        parts2.extend(buckets[pos])
        if pos < n:
            parts2.append(_slice_along_axis(out, ax, pos, pos + 1))
    if not parts2:
        return out
    return concatenate(parts2, axis=ax)


def delete(arr: Any, obj: Any, axis: int | None = None) -> mx.array:
    out = _asarray(arr)
    if axis is None:
        out = ravel(out)
        ax = 0
    else:
        ax = _normalize_axis_index(axis, out.ndim)
    n = int(out.shape[ax])

    remove = [False] * n
    if isinstance(obj, slice):
        for i in range(*obj.indices(n)):
            remove[i] = True
    elif isscalar(obj) or (isinstance(obj, mx.array) and obj.ndim == 0):
        idx = int(_asarray(obj).item())
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            msg = f"index {idx} is out of bounds for axis {ax} with size {n}"
            raise IndexError(msg)
        remove[idx] = True
    else:
        obj_arr = ravel(_asarray(obj))
        if obj_arr.dtype == mx.bool_:
            if int(obj_arr.shape[0]) != n:
                msg = "boolean array argument obj to delete must match the axis length"
                raise ValueError(msg)
            remove = [bool(v) for v in _tolist_1d_list(obj_arr)]
        else:
            for v in _tolist_1d_list(obj_arr):
                idx = int(v)
                if idx < 0:
                    idx += n
                if idx < 0 or idx >= n:
                    msg = f"index {idx} is out of bounds for axis {ax} with size {n}"
                    raise IndexError(msg)
                remove[idx] = True
    keep = [i for i in range(n) if not remove[i]]
    return take(out, array(keep, dtype=mx.int32), axis=ax)


def split(ary: Any, indices_or_sections: int | Sequence[int], axis: int = 0) -> list[mx.array]:
    return list(mx.split(_asarray(ary), indices_or_sections, axis=axis))


def array_split(ary: Any, indices_or_sections: int | Sequence[int], axis: int = 0) -> list[mx.array]:
    arr = _asarray(ary)
    if isinstance(indices_or_sections, int):
        indices = _prepare_split_indices(arr.shape[axis], indices_or_sections)
        return split(arr, indices, axis=axis)
    return split(arr, indices_or_sections, axis=axis)


def vsplit(ary: Any, indices_or_sections: int | Sequence[int]) -> list[mx.array]:
    arr = _asarray(ary)
    if arr.ndim < 2:
        msg = "vsplit only works on arrays of 2 or more dimensions"
        raise ValueError(msg)
    return split(arr, indices_or_sections, axis=0)


def hsplit(ary: Any, indices_or_sections: int | Sequence[int]) -> list[mx.array]:
    arr = _asarray(ary)
    if arr.ndim == 0:
        msg = "hsplit only works on arrays of 1 or more dimensions"
        raise ValueError(msg)
    axis = 0 if arr.ndim == 1 else 1
    return split(arr, indices_or_sections, axis=axis)


def dsplit(ary: Any, indices_or_sections: int | Sequence[int]) -> list[mx.array]:
    arr = _asarray(ary)
    if arr.ndim < 3:
        msg = "dsplit only works on arrays of 3 or more dimensions"
        raise ValueError(msg)
    return split(arr, indices_or_sections, axis=2)


def broadcast_to(a: Any, shape: Sequence[int]) -> mx.array:
    return mx.broadcast_to(_asarray(a), shape)


def broadcast_arrays(*args: Any) -> list[mx.array]:
    return list(mx.broadcast_arrays(*[_asarray(a) for a in args]))


def broadcast_shapes(*args: Sequence[int]) -> tuple[int, ...]:
    return tuple(mx.broadcast_shapes(*args))


def take(a: Any, indices: Any, axis: int | None = None) -> mx.array:
    idx = indices if isinstance(indices, int) else _asarray(indices)
    return mx.take(_asarray(a), idx, axis=axis)


def take_along_axis(arr: Any, indices: Any, axis: int) -> mx.array:
    return mx.take_along_axis(_asarray(arr), _asarray(indices), axis)


def repeat(a: Any, repeats: int, axis: int | None = None) -> mx.array:
    return mx.repeat(_asarray(a), repeats, axis=axis)


def tile(a: Any, reps: int | Sequence[int]) -> mx.array:
    return mx.tile(_asarray(a), reps)


def concat(arrays: Sequence[Any], axis: int = 0) -> mx.array:
    return concatenate(arrays, axis=axis)


def where(condition: Any, x: Any, y: Any) -> mx.array:
    x_val = x if isscalar(x) else _asarray(x)
    y_val = y if isscalar(y) else _asarray(y)
    return mx.where(_asarray(condition), x_val, y_val)


def choose(a: Any, choices: Sequence[Any], out: None = None, mode: str = "raise") -> mx.array:
    if out is not None:
        msg = "out is not currently supported"
        raise NotImplementedError(msg)
    if len(choices) == 0:
        msg = "choices list cannot be empty"
        raise ValueError(msg)
    if mode not in {"raise", "wrap", "clip"}:
        msg = "mode must be 'raise', 'wrap', or 'clip'"
        raise ValueError(msg)
    bcast = broadcast_arrays(_asarray(a), *[_asarray(c) for c in choices])
    idx = bcast[0].astype(mx.int64)
    choice_arrs = bcast[1:]
    n_choices = len(choice_arrs)
    if mode == "raise":
        bad = logical_or(less(idx, 0), greater_equal(idx, n_choices))
        if _scalar_bool(any(bad)):
            msg = "invalid entry in choice array"
            raise ValueError(msg)
        idx_use = idx
    elif mode == "clip":
        idx_use = clip(idx, 0, n_choices - 1).astype(mx.int64)
    else:
        idx_use = (idx % n_choices).astype(mx.int64)
    stacked = stack(choice_arrs, axis=0)
    gathered = take_along_axis(stacked, expand_dims(idx_use.astype(mx.int32), 0), axis=0)
    return squeeze(gathered, axis=0)


def select(condlist: Sequence[Any], choicelist: Sequence[Any], default: Any = 0) -> mx.array:
    if len(condlist) != len(choicelist):
        msg = "list of cases must be same length as list of conditions"
        raise ValueError(msg)
    if len(condlist) == 0:
        return _asarray(default)
    cond_arrs = [_asarray(c).astype(mx.bool_) for c in condlist]
    choice_arrs = [_asarray(c) for c in choicelist]
    default_arr = _asarray(default)
    shape = broadcast_shapes(
        *[tuple(int(s) for s in c.shape) for c in cond_arrs],
        *[tuple(int(s) for s in c.shape) for c in choice_arrs],
        tuple(int(s) for s in default_arr.shape),
    )
    result = broadcast_to(default_arr, shape)
    cond_b = [broadcast_to(c, shape) for c in cond_arrs]
    choice_b = [broadcast_to(c, shape) for c in choice_arrs]
    for cond, choice in zip(reversed(cond_b), reversed(choice_b), strict=False):
        result = where(cond, choice, result)
    return result


def piecewise(x: Any, condlist: Sequence[Any], funclist: Sequence[Any], *args: Any, **kw: Any) -> mx.array:
    arr = _asarray(x)
    x_shape = tuple(int(s) for s in arr.shape)
    conds = list(condlist) if isinstance(condlist, (list, tuple)) else [condlist]
    funcs = list(funclist) if isinstance(funclist, (list, tuple)) else [funclist]

    if len(funcs) == len(conds):
        default_term = 0
    elif len(funcs) == len(conds) + 1:
        default_term = funcs[-1]
        funcs = funcs[:-1]
    else:
        msg = "with N conditions, functions must be N or N+1"
        raise ValueError(msg)

    cond_arrs: list[mx.array] = []
    for cond in conds:
        c = _asarray(cond).astype(mx.bool_)
        c_shape = tuple(int(s) for s in c.shape)
        if c_shape != x_shape:
            if not _can_broadcast_to_shape(c_shape, x_shape):
                msg = "condition arrays must be broadcastable to x"
                raise ValueError(msg)
            c = broadcast_to(c, x_shape)
        cond_arrs.append(c)

    class _PiecewiseFallbackError(Exception):
        pass

    def _eval_term(term: Any) -> Any:
        val = term(arr, *args, **kw) if callable(term) else term
        if isscalar(val):
            return val
        val_arr = _asarray(val)
        v_shape = tuple(int(s) for s in val_arr.shape)
        if v_shape != x_shape:
            if not _can_broadcast_to_shape(v_shape, x_shape):
                # NumPy callable piecewise semantics pass x[cond] subsets. If the
                # callable is not vectorized/broadcastable, preserve correctness via fallback.
                raise _PiecewiseFallbackError
            val_arr = broadcast_to(val_arr, x_shape)
        return val_arr

    try:
        result = _eval_term(default_term)
        for cond, term in zip(cond_arrs, funcs, strict=False):
            term_val = _eval_term(term)
            if not isinstance(term_val, mx.array) and isscalar(term_val):
                result = where(cond, term_val, result)
                continue
            result = where(cond, term_val, result)
        if isinstance(result, mx.array):
            return result
        return full(x_shape, result)
    except _PiecewiseFallbackError:
        bridge.record_fallback("core.piecewise:nonvectorized_callable")
        np_mod = _numpy()
        np_result = np_mod.piecewise(np_mod.asarray(x), [np_mod.asarray(c) for c in conds], funclist, *args, **kw)
        return _numpy_wrap_value(np_result)


def clip(a: Any, a_min: Any, a_max: Any) -> mx.array:
    return mx.clip(_asarray(a), a_min, a_max)


def _pad_width_scalar(value: Any) -> int | None:
    return _coerce_exact_scalar_int(value, allow_bool=True)


def _pad_width_pair(value: Any) -> tuple[int, int] | None:
    scalar = _pad_width_scalar(value)
    if scalar is not None:
        return (scalar, scalar)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    seq = list(value)
    if len(seq) == 1:
        scalar = _pad_width_scalar(seq[0])
        if scalar is not None:
            return (scalar, scalar)
        return None
    if len(seq) != 2:
        return None
    before = _pad_width_scalar(seq[0])
    after = _pad_width_scalar(seq[1])
    if before is None or after is None:
        return None
    return (before, after)


def _normalize_pad_widths(pad_width: Any, ndim_: int) -> list[tuple[int, int]]:
    if ndim_ == 0:
        return []
    pair = _pad_width_pair(pad_width)
    if pair is not None:
        out = [pair] * ndim_
    else:
        if isinstance(pad_width, (str, bytes)) or not isinstance(pad_width, Sequence):
            msg = "Unable to create correctly shaped tuple from pad_width"
            raise ValueError(msg)
        seq = list(pad_width)
        if len(seq) != ndim_:
            msg = "Unable to create correctly shaped tuple from pad_width"
            raise ValueError(msg)
        out = []
        for item in seq:
            pair_i = _pad_width_pair(item)
            if pair_i is None:
                msg = "Unable to create correctly shaped tuple from pad_width"
                raise ValueError(msg)
            out.append(pair_i)
    for before, after in out:
        if before < 0 or after < 0:
            msg = "index can't contain negative values"
            raise ValueError(msg)
    return out


def _pad_axis_indices(length: int, before: int, after: int, mode: str) -> mx.array:
    total = before + length + after
    if length == 0:
        if before or after:
            msg = "can't extend empty axis 0 using modes other than 'constant' or 'empty'"
            raise ValueError(msg)
        return arange(0, dtype=mx.int32)
    if before == 0 and after == 0:
        return arange(length, dtype=mx.int32)
    if length == 1:
        return zeros((total,), dtype=mx.int32)
    offset = arange(total, dtype=mx.int64) - before
    if mode == "reflect":
        period = 2 * (length - 1)
        folded = offset % period
        idx = where(less(folded, length), folded, period - folded)
        return idx.astype(mx.int32)
    if mode == "symmetric":
        period = 2 * length
        folded = offset % period
        idx = where(less(folded, length), folded, (2 * length - 1) - folded)
        return idx.astype(mx.int32)
    msg = f"unsupported mode {mode!r}"
    raise ValueError(msg)


def _pad_reflect_like(arr: mx.array, pad_widths: Sequence[tuple[int, int]], mode: str) -> mx.array:
    out = arr
    for axis, (before, after) in enumerate(pad_widths):
        if before == 0 and after == 0:
            continue
        idx = _pad_axis_indices(int(out.shape[axis]), before, after, mode)
        out = take(out, idx, axis=axis)
    return out


def pad(
    array: Any,
    pad_width: Any,
    mode: str = "constant",
    **kwargs: Any,
) -> mx.array:
    mode_name = str(mode)
    mode_key = mode_name.lower()
    if mode_key in {"constant", "edge"}:
        constant_values = kwargs.pop("constant_values", 0)
        if not kwargs:
            return mx.pad(_asarray(array), pad_width, mode=mode_key, constant_values=constant_values)
        kwargs["constant_values"] = constant_values
    elif mode_key in {"reflect", "symmetric"}:
        reflect_type = kwargs.pop("reflect_type", "even")
        if kwargs or reflect_type != "even":
            if reflect_type != "even":
                kwargs["reflect_type"] = reflect_type
            bridge.record_fallback(f"core.pad:mode_{mode_key}")
        else:
            arr = _asarray(array)
            return _pad_reflect_like(arr, _normalize_pad_widths(pad_width, arr.ndim), mode_key)
    else:
        bridge.record_fallback(f"core.pad:mode_{mode_key}")
    np_mod = _numpy()
    result = np_mod.pad(np_mod.asarray(array), pad_width, mode=cast("Any", mode_name), **kwargs)
    return _numpy_wrap_value(result)


def roll(a: Any, shift: int | Sequence[int], axis: int | Sequence[int] | None = None) -> mx.array:
    shift_arg, axis_arg = _normalize_roll_args(shift, axis)
    return mx.roll(_asarray(a), cast("Any", shift_arg), cast("Any", axis_arg))


def flip(m: Any, axis: int | Sequence[int] | None = None) -> mx.array:
    arr = _asarray(m)
    if axis is None:
        flipped = flatten(arr)[::-1]
        return reshape(flipped, arr.shape)
    axes = _normalize_axes(axis, arr.ndim) if not isinstance(axis, int) else _normalize_axes((axis,), arr.ndim)
    index = [slice(None)] * arr.ndim
    for ax in axes:
        index[ax] = slice(None, None, -1)
    return arr[tuple(index)]


def fliplr(m: Any) -> mx.array:
    arr = _asarray(m)
    if arr.ndim < 2:
        msg = "Input must be >= 2-d."
        raise ValueError(msg)
    return flip(arr, axis=1)


def flipud(m: Any) -> mx.array:
    arr = _asarray(m)
    if arr.ndim < 1:
        msg = "Input must be >= 1-d."
        raise ValueError(msg)
    return flip(arr, axis=0)


def rot90(m: Any, k: int = 1, axes: tuple[int, int] = (0, 1)) -> mx.array:
    arr = _asarray(m)
    ax1, ax2 = axes
    ndim_ = arr.ndim
    ax1 = ax1 + ndim_ if ax1 < 0 else ax1
    ax2 = ax2 + ndim_ if ax2 < 0 else ax2
    if ax1 == ax2:
        msg = "Axes must be different"
        raise ValueError(msg)
    k_mod = k % 4
    if k_mod == 0:
        return arr
    if k_mod == 2:
        return flip(flip(arr, axis=ax1), axis=ax2)
    if k_mod == 1:
        return swapaxes(flip(arr, axis=ax2), ax1, ax2)
    return flip(swapaxes(arr, ax1, ax2), axis=ax2)


def diagonal(a: Any, offset: int = 0, axis1: int = 0, axis2: int = 1) -> mx.array:
    return mx.diagonal(_asarray(a), offset=offset, axis1=axis1, axis2=axis2)


def trace(a: Any, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Any | None = None) -> mx.array:
    arr = _asarray(a)
    trace_dtype = _default_sum_prod_dtype(arr, dtype)
    with _cpu_default_device_for_dtypes(arr.dtype, trace_dtype):
        return mx.trace(arr, offset=offset, axis1=axis1, axis2=axis2, dtype=trace_dtype)


def tril(m: Any, k: int = 0) -> mx.array:
    return mx.tril(_asarray(m), k)


def triu(m: Any, k: int = 0) -> mx.array:
    return mx.triu(_asarray(m), k)


def add(x1: Any, x2: Any) -> mx.array:
    return mx.add(_asarray(x1), _asarray(x2))


def subtract(x1: Any, x2: Any) -> mx.array:
    return mx.subtract(_asarray(x1), _asarray(x2))


def multiply(x1: Any, x2: Any) -> mx.array:
    return mx.multiply(_asarray(x1), _asarray(x2))


def divide(x1: Any, x2: Any) -> mx.array:
    return mx.divide(_asarray(x1), _asarray(x2))


def true_divide(x1: Any, x2: Any) -> mx.array:
    return divide(x1, x2)


def floor_divide(x1: Any, x2: Any) -> mx.array:
    return mx.floor_divide(_asarray(x1), _asarray(x2))


def remainder(x1: Any, x2: Any) -> mx.array:
    return mx.remainder(_asarray(x1), _asarray(x2))


def mod(x1: Any, x2: Any) -> mx.array:
    return remainder(x1, x2)


def fmod(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    return a - trunc(divide(a, b)) * b


def divmod(x1: Any, x2: Any) -> tuple[mx.array, mx.array]:
    return floor_divide(x1, x2), remainder(x1, x2)


def power(x1: Any, x2: Any) -> mx.array:
    return mx.power(_asarray(x1), _asarray(x2))


def square(x: Any) -> mx.array:
    return mx.square(_asarray(x))


def float_power(x1: Any, x2: Any) -> mx.array:
    out_dtype = _NUMPY_DEFAULT_FLOAT_DTYPE
    with _cpu_default_device_for_dtype(out_dtype):
        return power(asarray(x1, dtype=out_dtype), asarray(x2, dtype=out_dtype))


def reciprocal(x: Any) -> mx.array:
    return mx.reciprocal(_asarray(x))


def negative(x: Any) -> mx.array:
    return mx.negative(_asarray(x))


def positive(x: Any) -> mx.array:
    return _asarray(x)


def abs(x: Any) -> mx.array:
    return mx.abs(_asarray(x))


def absolute(x: Any) -> mx.array:
    return abs(x)


def fabs(x: Any) -> mx.array:
    return abs(x)


def sign(x: Any) -> mx.array:
    return mx.sign(_asarray(x))


def exp(x: Any) -> mx.array:
    return mx.exp(_asarray(x))


def exp2(x: Any) -> mx.array:
    return power(2.0, x)


def expm1(x: Any) -> mx.array:
    return mx.expm1(_asarray(x))


def log(x: Any) -> mx.array:
    return mx.log(_asarray(x))


def log1p(x: Any) -> mx.array:
    return mx.log1p(_asarray(x))


def log2(x: Any) -> mx.array:
    return mx.log2(_asarray(x))


def log10(x: Any) -> mx.array:
    return mx.log10(_asarray(x))


def logaddexp2(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1).astype(mx.float32)
    b = _asarray(x2).astype(mx.float32)
    m = maximum(a, b)
    return m + log2(exp2(a - m) + exp2(b - m))


def sqrt(x: Any) -> mx.array:
    return mx.sqrt(_asarray(x))


def rsqrt(x: Any) -> mx.array:
    return mx.rsqrt(_asarray(x))


def sin(x: Any) -> mx.array:
    return mx.sin(_asarray(x))


def cos(x: Any) -> mx.array:
    return mx.cos(_asarray(x))


def tan(x: Any) -> mx.array:
    return mx.tan(_asarray(x))


def asin(x: Any) -> mx.array:
    return arcsin(x)


def acos(x: Any) -> mx.array:
    return arccos(x)


def atan(x: Any) -> mx.array:
    return arctan(x)


def arcsin(x: Any) -> mx.array:
    return mx.arcsin(_asarray(x))


def arccos(x: Any) -> mx.array:
    return mx.arccos(_asarray(x))


def arctan(x: Any) -> mx.array:
    return mx.arctan(_asarray(x))


def acosh(x: Any) -> mx.array:
    return mx.arccosh(_asarray(x))


def asinh(x: Any) -> mx.array:
    return mx.arcsinh(_asarray(x))


def atanh(x: Any) -> mx.array:
    return mx.arctanh(_asarray(x))


def atan2(x1: Any, x2: Any) -> mx.array:
    return mx.arctan2(_asarray(x1), _asarray(x2))


def sinh(x: Any) -> mx.array:
    return mx.sinh(_asarray(x))


def cosh(x: Any) -> mx.array:
    return mx.cosh(_asarray(x))


def tanh(x: Any) -> mx.array:
    return mx.tanh(_asarray(x))


def sinc(x: Any) -> mx.array:
    arr = _asarray(x).astype(mx.float32)
    pix = pi * arr
    return where(equal(arr, 0), 1.0, sin(pix) / pix)


def degrees(x: Any) -> mx.array:
    return mx.degrees(_asarray(x))


def radians(x: Any) -> mx.array:
    return mx.radians(_asarray(x))


def deg2rad(x: Any) -> mx.array:
    return radians(x)


def rad2deg(x: Any) -> mx.array:
    return degrees(x)


def floor(x: Any) -> mx.array:
    return mx.floor(_asarray(x))


def ceil(x: Any) -> mx.array:
    return mx.ceil(_asarray(x))


def round(a: Any, decimals: int = 0) -> mx.array:
    return mx.round(_asarray(a), decimals)


def around(a: Any, decimals: int = 0) -> mx.array:
    return round(a, decimals=decimals)


def trunc(x: Any) -> mx.array:
    arr = _asarray(x)
    return where(arr >= 0, floor(arr), ceil(arr))


def rint(x: Any) -> mx.array:
    return round(x, decimals=0)


def fix(x: Any) -> mx.array:
    return trunc(x)


def maximum(x1: Any, x2: Any) -> mx.array:
    return mx.maximum(_asarray(x1), _asarray(x2))


def minimum(x1: Any, x2: Any) -> mx.array:
    return mx.minimum(_asarray(x1), _asarray(x2))


def fmax(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    return where(isnan(a), b, where(isnan(b), a, maximum(a, b)))


def fmin(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    return where(isnan(a), b, where(isnan(b), a, minimum(a, b)))


def equal(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    with _cpu_default_device_for_dtypes(a.dtype, b.dtype):
        return mx.equal(a, b)


def not_equal(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    with _cpu_default_device_for_dtypes(a.dtype, b.dtype):
        return mx.not_equal(a, b)


def greater(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    with _cpu_default_device_for_dtypes(a.dtype, b.dtype):
        return mx.greater(a, b)


def greater_equal(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    with _cpu_default_device_for_dtypes(a.dtype, b.dtype):
        return mx.greater_equal(a, b)


def less(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    with _cpu_default_device_for_dtypes(a.dtype, b.dtype):
        return mx.less(a, b)


def less_equal(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1)
    b = _asarray(x2)
    with _cpu_default_device_for_dtypes(a.dtype, b.dtype):
        return mx.less_equal(a, b)


def logical_and(x1: Any, x2: Any) -> mx.array:
    return mx.logical_and(_asarray(x1), _asarray(x2))


def logical_or(x1: Any, x2: Any) -> mx.array:
    return mx.logical_or(_asarray(x1), _asarray(x2))


def logical_xor(x1: Any, x2: Any) -> mx.array:
    a = _asarray(x1).astype(mx.bool_)
    b = _asarray(x2).astype(mx.bool_)
    return mx.not_equal(a, b)


def logical_not(x: Any) -> mx.array:
    return mx.logical_not(_asarray(x))


def bitwise_and(x1: Any, x2: Any) -> mx.array:
    return mx.bitwise_and(_asarray(x1), _asarray(x2))


def bitwise_or(x1: Any, x2: Any) -> mx.array:
    return mx.bitwise_or(_asarray(x1), _asarray(x2))


def bitwise_xor(x1: Any, x2: Any) -> mx.array:
    return mx.bitwise_xor(_asarray(x1), _asarray(x2))


def bitwise_not(x: Any) -> mx.array:
    return ~_asarray(x)


def bitwise_invert(x: Any) -> mx.array:
    return bitwise_not(x)


def left_shift(x1: Any, x2: Any) -> mx.array:
    return mx.left_shift(_asarray(x1), _asarray(x2))


def right_shift(x1: Any, x2: Any) -> mx.array:
    return mx.right_shift(_asarray(x1), _asarray(x2))


def bitwise_left_shift(x1: Any, x2: Any) -> mx.array:
    return left_shift(x1, x2)


def bitwise_right_shift(x1: Any, x2: Any) -> mx.array:
    return right_shift(x1, x2)


def isnan(x: Any) -> mx.array:
    arr = _asarray(x)
    with _cpu_default_device_for_dtype(arr.dtype):
        return mx.isnan(arr)


def isinf(x: Any) -> mx.array:
    arr = _asarray(x)
    with _cpu_default_device_for_dtype(arr.dtype):
        return mx.isinf(arr)


def isfinite(x: Any) -> mx.array:
    arr = _asarray(x)
    with _cpu_default_device_for_dtype(arr.dtype):
        return mx.isfinite(arr)


def signbit(x: Any) -> mx.array:
    arr = _asarray(x)
    if _is_complex_dtype(arr.dtype):
        msg = "ufunc 'signbit' not supported for the input types"
        raise TypeError(msg)
    neg = less(arr, 0)
    if _is_floating_or_bfloat_dtype(arr.dtype):
        neg_zero = logical_and(equal(arr, 0), mx.isneginf(divide(1.0, arr)))
        return logical_or(neg, neg_zero)
    return neg


def copysign(x1: Any, x2: Any) -> mx.array:
    mag = abs(x1)
    return where(signbit(x2), -mag, mag)


def hypot(x1: Any, x2: Any) -> mx.array:
    a = abs(x1).astype(mx.float32)
    b = abs(x2).astype(mx.float32)
    hi = maximum(a, b)
    lo = minimum(a, b)
    ratio = where(equal(hi, 0), 0.0, lo / hi)
    return where(equal(hi, 0), 0.0, hi * sqrt(1.0 + ratio * ratio))


def heaviside(x1: Any, x2: Any) -> mx.array:
    x = _asarray(x1)
    return where(less(x, 0), 0, where(greater(x, 0), 1, x2))


def nan_to_num(
    x: Any,
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> mx.array:
    arr = _asarray(x)
    out = mx.nan_to_num(arr, nan=nan, posinf=posinf, neginf=neginf)
    if copy:
        return out
    return out


def isclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> mx.array:
    return mx.isclose(_asarray(a), _asarray(b), rtol=rtol, atol=atol, equal_nan=equal_nan)


def allclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> bool:
    return _scalar_bool(mx.allclose(_asarray(a), _asarray(b), rtol=rtol, atol=atol, equal_nan=equal_nan))


def array_equal(a: Any, b: Any, equal_nan: bool = False) -> bool:
    return _scalar_bool(mx.array_equal(_asarray(a), _asarray(b), equal_nan=equal_nan))


def array_equiv(a: Any, b: Any) -> bool:
    try:
        x, y = broadcast_arrays(a, b)
    except (TypeError, ValueError):
        return False
    return array_equal(x, y)


def isreal(x: Any) -> mx.array:
    arr = _asarray(x)
    if not _is_complex_dtype(arr.dtype):
        return ones(arr.shape, dtype=mx.bool_)
    return equal(imag(arr), 0)


def iscomplex(x: Any) -> mx.array:
    arr = _asarray(x)
    if not _is_complex_dtype(arr.dtype):
        return zeros(arr.shape, dtype=mx.bool_)
    return not_equal(imag(arr), 0)


def isrealobj(x: Any) -> bool:
    dt = getattr(_asarray(x), "dtype", None) if isinstance(x, mx.array) else getattr(x, "dtype", None)
    if dt is None:
        try:
            dt = _asarray(x).dtype
        except (TypeError, ValueError):
            return not isinstance(x, complex)
    return not _is_complex_dtype(dt)


def iscomplexobj(x: Any) -> bool:
    return not isrealobj(x)


def sum(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
) -> mx.array:
    arr = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(arr, dtype)
    target_dtype = reduce_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if reduce_dtype is not None and arr.dtype != reduce_dtype:
            arr = arr.astype(reduce_dtype)
        return mx.sum(arr, axis=axis, keepdims=keepdims)


def prod(a: Any, axis: int | Sequence[int] | None = None, dtype: Any | None = None, keepdims: bool = False) -> mx.array:
    arr = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(arr, dtype)
    target_dtype = reduce_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if reduce_dtype is not None and arr.dtype != reduce_dtype:
            arr = arr.astype(reduce_dtype)
        return mx.prod(arr, axis=axis, keepdims=keepdims)


def mean(a: Any, axis: int | Sequence[int] | None = None, dtype: Any | None = None, keepdims: bool = False) -> mx.array:
    arr = _asarray(a)
    reduce_dtype = _default_mean_var_dtype(arr, dtype)
    target_dtype = reduce_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if reduce_dtype is not None and arr.dtype != reduce_dtype:
            arr = arr.astype(reduce_dtype)
        return mx.mean(arr, axis=axis, keepdims=keepdims)


def std(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> mx.array:
    arr = _asarray(a)
    reduce_dtype = _default_mean_var_dtype(arr, dtype)
    target_dtype = reduce_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if reduce_dtype is not None and arr.dtype != reduce_dtype:
            arr = arr.astype(reduce_dtype)
        return mx.std(arr, axis=axis, keepdims=keepdims, ddof=ddof)


def var(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> mx.array:
    arr = _asarray(a)
    reduce_dtype = _default_mean_var_dtype(arr, dtype)
    target_dtype = reduce_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if reduce_dtype is not None and arr.dtype != reduce_dtype:
            arr = arr.astype(reduce_dtype)
        return mx.var(arr, axis=axis, keepdims=keepdims, ddof=ddof)


def median(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return mx.median(_asarray(a), axis=axis, keepdims=keepdims)


def _quantile_prepare_rows(
    a: Any,
    axis: int | Sequence[int] | None,
) -> tuple[mx.array, mx.array, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    arr = _asarray(a)
    if axis is None:
        axes = tuple(range(arr.ndim))
    elif isinstance(axis, int):
        axes = (_normalize_axis_index(axis, arr.ndim),)
    else:
        axes = _normalize_axes(axis, arr.ndim)
    keep_axes = tuple(i for i in range(arr.ndim) if i not in axes)
    perm = keep_axes + axes
    arr_t = transpose(arr, perm) if perm != tuple(range(arr.ndim)) else arr
    outer_shape = tuple(int(arr.shape[i]) for i in keep_axes)
    n_outer = _prod_int(outer_shape) if outer_shape else 1
    red_shape = tuple(int(arr.shape[i]) for i in axes)
    n_red = _prod_int(red_shape) if red_shape else 1
    rows = reshape(arr_t, (n_outer, n_red))
    return arr, rows, axes, keep_axes, outer_shape


def _quantile_pack_rows(
    row_results: list[mx.array],
    q_shape: tuple[int, ...] | None,
    _arr_ndim: int,
    axes: tuple[int, ...],
    outer_shape: tuple[int, ...],
    keepdims: bool,
) -> mx.array:
    shaped: list[mx.array] = []
    for row in row_results:
        tmp = reshape(row, outer_shape or ())
        if keepdims:
            for ax in sorted(axes):
                tmp = expand_dims(tmp, ax)
        shaped.append(tmp)
    if q_shape is None:
        return shaped[0]
    out = stack(shaped, axis=0)
    return reshape(out, q_shape + tuple(int(s) for s in shaped[0].shape))


def _quantile_rows_impl(
    rows: mx.array,
    q: Any,
    method: str,
    nan_policy: bool,
) -> tuple[list[mx.array], tuple[int, ...] | None]:
    q_arr = _asarray(q).astype(mx.float32)
    q_shape = None if q_arr.ndim == 0 else tuple(int(s) for s in q_arr.shape)
    q_vals = [float(v) for v in _tolist_1d_list(ravel(q_arr))]
    for qv in q_vals:
        if qv < 0.0 or qv > 1.0:
            msg = "Quantiles must be in the range [0, 1]"
            raise ValueError(msg)
    method_ = _narrow_quantile_method(method)

    n_cols = int(rows.shape[1])
    interp_rows = rows
    if _is_bool_dtype(rows.dtype) or _is_integer_dtype(rows.dtype):
        interp_rows = rows.astype(mx.float32)

    if not nan_policy:
        if n_cols == 0:
            msg = "cannot do a non-empty take from an empty axes."
            raise IndexError(msg)
        sorted_rows = sort(rows, axis=1)
        sorted_interp = sort(interp_rows, axis=1) if interp_rows is not rows else sorted_rows
        outputs: list[mx.array] = []
        for qv in q_vals:
            pos = qv * (n_cols - 1)
            if method_ == "nearest":
                idx = int(round(pos))
                outputs.append(sorted_rows[:, idx])
                continue
            lo = math.floor(pos)
            hi = math.ceil(pos)
            if method_ == "lower":
                outputs.append(sorted_rows[:, lo])
                continue
            if method_ == "higher":
                outputs.append(sorted_rows[:, hi])
                continue
            v0 = sorted_interp[:, lo]
            v1 = sorted_interp[:, hi]
            if method_ == "midpoint":
                outputs.append((v0 + v1) * 0.5)
            else:
                frac = pos - lo
                outputs.append(v0 + (v1 - v0) * frac)
        return outputs, q_shape

    if _is_complex_dtype(rows.dtype):
        # Keep complex nanquantile parity via NumPy for now; MLX sort ordering for complex is not guaranteed.
        bridge.record_fallback("core.nanquantile:complex")
        np_mod = _numpy()
        np_rows = np_mod.asarray(rows)
        fn = np_mod.nanquantile
        np_out = [mx.array(fn(np_rows, qv, axis=1, method=method_)) for qv in q_vals]
        return np_out, q_shape

    mask = logical_not(isnan(rows))
    counts = sum(mask.astype(mx.int32), axis=1)
    valid = greater(counts, 0)
    sorted_rows = sort(where(mask, rows, mx.inf), axis=1)
    sorted_interp = sort(where(mask, interp_rows, mx.inf), axis=1) if interp_rows is not rows else sorted_rows
    counts_f = counts.astype(mx.float32)
    outputs2: list[mx.array] = []
    for qv in q_vals:
        pos = where(valid, (counts_f - 1.0) * qv, 0.0)
        if method_ == "nearest":
            idx = mx.round(pos).astype(mx.int32)
            val = squeeze(take_along_axis(sorted_rows, expand_dims(idx, 1), axis=1), axis=1)
            outputs2.append(where(valid, val, nan))
            continue
        lo = mx.floor(pos).astype(mx.int32)
        hi = mx.ceil(pos).astype(mx.int32)
        if method_ == "lower":
            val = squeeze(take_along_axis(sorted_rows, expand_dims(lo, 1), axis=1), axis=1)
            outputs2.append(where(valid, val, nan))
            continue
        if method_ == "higher":
            val = squeeze(take_along_axis(sorted_rows, expand_dims(hi, 1), axis=1), axis=1)
            outputs2.append(where(valid, val, nan))
            continue
        v0 = squeeze(take_along_axis(sorted_interp, expand_dims(lo, 1), axis=1), axis=1)
        v1 = squeeze(take_along_axis(sorted_interp, expand_dims(hi, 1), axis=1), axis=1)
        if method_ == "midpoint":
            val = (v0 + v1) * 0.5
        else:
            frac = pos - lo.astype(mx.float32)
            val = v0 + (v1 - v0) * frac
        outputs2.append(where(valid, val, nan))
    return outputs2, q_shape


def quantile(
    a: Any,
    q: Any,
    axis: int | Sequence[int] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> mx.array:
    if out is not None:
        msg = "out is not currently supported"
        raise NotImplementedError(msg)
    if overwrite_input:
        msg = "overwrite_input=True is not currently supported"
        raise NotImplementedError(msg)
    arr, rows, axes, _keep_axes, outer_shape = _quantile_prepare_rows(a, axis)
    row_results, q_shape = _quantile_rows_impl(rows, q, method=method, nan_policy=False)
    return _quantile_pack_rows(row_results, q_shape, arr.ndim, axes, outer_shape, keepdims)


def percentile(
    a: Any,
    q: Any,
    axis: int | Sequence[int] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> mx.array:
    if out is not None:
        msg = "out is not currently supported"
        raise NotImplementedError(msg)
    if overwrite_input:
        msg = "overwrite_input=True is not currently supported"
        raise NotImplementedError(msg)
    return quantile(a, _asarray(q).astype(mx.float32) / 100.0, axis=axis, method=method, keepdims=keepdims)


def nanquantile(
    a: Any,
    q: Any,
    axis: int | Sequence[int] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> mx.array:
    if out is not None:
        msg = "out is not currently supported"
        raise NotImplementedError(msg)
    if overwrite_input:
        msg = "overwrite_input=True is not currently supported"
        raise NotImplementedError(msg)
    arr = _asarray(a)
    if not _is_inexact_dtype(arr.dtype):
        return quantile(arr, q, axis=axis, method=method, keepdims=keepdims)
    arr0, rows, axes, _keep_axes, outer_shape = _quantile_prepare_rows(arr, axis)
    row_results, q_shape = _quantile_rows_impl(rows, q, method=method, nan_policy=True)
    return _quantile_pack_rows(row_results, q_shape, arr0.ndim, axes, outer_shape, keepdims)


def nanpercentile(
    a: Any,
    q: Any,
    axis: int | Sequence[int] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> mx.array:
    if out is not None:
        msg = "out is not currently supported"
        raise NotImplementedError(msg)
    if overwrite_input:
        msg = "overwrite_input=True is not currently supported"
        raise NotImplementedError(msg)
    return nanquantile(a, _asarray(q).astype(mx.float32) / 100.0, axis=axis, method=method, keepdims=keepdims)


def nansum(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
) -> mx.array:
    base = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(base, dtype)
    target_dtype = reduce_dtype or base.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        arr = _ensure_dtype_for_reduction(base, dtype if dtype is not None else reduce_dtype)
        return sum(where(isnan(arr), 0, arr), axis=axis, dtype=dtype, keepdims=keepdims)


def nanprod(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
) -> mx.array:
    base = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(base, dtype)
    target_dtype = reduce_dtype or base.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        arr = _ensure_dtype_for_reduction(base, dtype if dtype is not None else reduce_dtype)
        return prod(where(isnan(arr), 1, arr), axis=axis, dtype=dtype, keepdims=keepdims)


def nanmean(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
) -> mx.array:
    arr = _asarray(a)
    reduce_dtype = _default_mean_var_dtype(arr, dtype)
    target_dtype = reduce_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if reduce_dtype is not None and arr.dtype != reduce_dtype:
            arr = arr.astype(reduce_dtype)
        mask = logical_not(isnan(arr))
        total = sum(where(mask, arr, 0), axis=axis, dtype=arr.dtype, keepdims=keepdims)
        count_dtype = _count_dtype_for_nan_stats(arr.dtype)
        count = sum(mask.astype(count_dtype), axis=axis, dtype=count_dtype, keepdims=keepdims)
        nan_value = _nan_scalar_for_dtype(getattr(total, "dtype", arr.dtype))
        return where(greater(count, 0), total / count, nan_value)


def nanstd(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> mx.array:
    return sqrt(nanvar(a, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof))


def nanvar(
    a: Any,
    axis: int | Sequence[int] | None = None,
    dtype: Any | None = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> mx.array:
    arr = _asarray(a)
    reduce_dtype = _default_mean_var_dtype(arr, dtype)
    target_dtype = reduce_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if reduce_dtype is not None and arr.dtype != reduce_dtype:
            arr = arr.astype(reduce_dtype)
        count_dtype = _count_dtype_for_nan_stats(arr.dtype)
        zero = mx.array(0, dtype=arr.dtype)
        zero_count = mx.array(0, dtype=count_dtype)
        nan_value = _nan_scalar_for_dtype(arr.dtype)
        mask = logical_not(isnan(arr))
        count_keep = sum(mask.astype(count_dtype), axis=axis, dtype=count_dtype, keepdims=True)
        total_keep = sum(where(mask, arr, zero), axis=axis, dtype=arr.dtype, keepdims=True)
        mean_keep = where(greater(count_keep, 0), total_keep / count_keep, zero)
        centered = where(mask, arr - mean_keep, zero)
        ss_keep = sum(centered * centered, axis=axis, dtype=arr.dtype, keepdims=True)
        denom_keep = count_keep - mx.array(float(ddof), dtype=count_dtype)
        var_keep = where(greater(denom_keep, zero_count), ss_keep / denom_keep, nan_value)
    if keepdims:
        return var_keep
    if axis is None:
        return var_keep
    return squeeze(var_keep, axis=axis)


def nanmedian(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return nanquantile(a, 0.5, axis=axis, keepdims=keepdims)


def average(
    a: Any,
    axis: int | Sequence[int] | None = None,
    weights: Any | None = None,
    returned: bool = False,
    keepdims: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    arr = _asarray(a)
    work_dtype = _default_mean_var_dtype(arr, None)
    target_dtype = work_dtype or arr.dtype
    with _cpu_default_device_for_dtype(target_dtype):
        if work_dtype is not None and arr.dtype != work_dtype:
            arr = arr.astype(work_dtype)
        axes = tuple(range(arr.ndim)) if axis is None else _normalize_axes(axis, arr.ndim)
        axis_arg: int | tuple[int, ...] | None
        if axis is None:
            axis_arg = None
        elif isinstance(axis, int):
            axis_arg = axis
        else:
            axis_arg = tuple(axis)

        if weights is None:
            avg = mean(arr, axis=axis_arg, dtype=arr.dtype, keepdims=keepdims)
            if not returned:
                return avg
            count = float(_prod_int([arr.shape[ax] for ax in axes]))
            sw_shape = avg.shape if isinstance(avg, mx.array) else ()
            sw_dtype = getattr(avg, "dtype", _NUMPY_DEFAULT_FLOAT_DTYPE)
            sw = full(sw_shape, count, dtype=sw_dtype)
            return avg, sw

        w = _asarray(weights)
        if _is_complex_dtype(w.dtype) and not _is_complex_dtype(arr.dtype):
            arr = arr.astype(_NUMPY_DEFAULT_COMPLEX_DTYPE)
        elif (_is_integer_dtype(w.dtype) or _is_bool_dtype(w.dtype)) and (
            arr.dtype != _NUMPY_DEFAULT_FLOAT_DTYPE and (_is_integer_dtype(arr.dtype) or _is_bool_dtype(arr.dtype))
        ):
            arr = arr.astype(_NUMPY_DEFAULT_FLOAT_DTYPE)
        if w.dtype != arr.dtype:
            w = w.astype(arr.dtype)

        if axis is None:
            if tuple(w.shape) != tuple(arr.shape):
                msg = "Axis must be specified when shapes of a and weights differ."
                raise TypeError(msg)
            w_b = w
        elif tuple(w.shape) == tuple(arr.shape):
            w_b = w
        elif tuple(int(s) for s in w.shape) == tuple(int(arr.shape[ax]) for ax in axes):
            shape = [1] * arr.ndim
            for i, ax in enumerate(axes):
                shape[ax] = int(w.shape[i])
            w_b = reshape(w, shape)
        elif len(axes) == 1 and w.ndim == 1 and w.shape[0] == arr.shape[axes[0]]:
            shape = [1] * arr.ndim
            shape[axes[0]] = w.shape[0]
            w_b = reshape(w, shape)
        elif w.ndim == arr.ndim and _can_broadcast_to_shape(
            tuple(int(s) for s in w.shape),
            tuple(int(s) for s in arr.shape),
        ):
            w_b = broadcast_to(w, tuple(int(s) for s in arr.shape))
        else:
            msg = "Axis must be specified when shapes of a and weights differ."
            raise TypeError(msg)

        sw = sum(w_b, axis=axis_arg, dtype=arr.dtype, keepdims=keepdims)
        if _scalar_bool(any(equal(sw, 0))):
            msg = "Weights sum to zero, can't be normalized"
            raise ZeroDivisionError(msg)
        avg = sum(arr * w_b, axis=axis_arg, dtype=arr.dtype, keepdims=keepdims) / sw
        if tuple(sw.shape) != tuple(avg.shape):
            sw = broadcast_to(sw, avg.shape)
        if returned:
            return avg, sw
        return avg


def cov(
    m: Any,
    y: Any | None = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: Any | None = None,
    aweights: Any | None = None,
    dtype: Any | None = None,
) -> mx.array:
    def _to_2d_obs(x: Any) -> mx.array:
        arr = _asarray(x)
        if arr.ndim > 2:
            msg = "m has more than 2 dimensions"
            raise ValueError(msg)
        if arr.ndim == 0:
            arr = reshape(arr, (1, 1))
        elif arr.ndim == 1:
            arr = reshape(arr, (1, int(arr.shape[0])))
        if not rowvar and int(arr.shape[0]) != 1:
            arr = transpose(arr)
        return arr

    x = _to_2d_obs(m)
    if y is not None:
        x = concatenate([x, _to_2d_obs(y)], axis=0)

    dt = _resolve_dtype(dtype)
    if dt is not None:
        x = x.astype(dt)
    elif not _is_inexact_dtype(x.dtype):
        x = x.astype(mx.float32)
    obs = int(x.shape[1])

    fw: mx.array | None = None
    aw: mx.array | None = None
    if fweights is not None:
        fw = ravel(_asarray(fweights)).astype(mx.float32)
        if int(fw.shape[0]) != obs:
            msg = "incompatible numbers of samples and fweights"
            raise RuntimeError(msg)
        if _scalar_bool(any(less(fw, 0))):
            msg = "fweights cannot be negative"
            raise ValueError(msg)
        if _scalar_bool(any(not_equal(fw, floor(fw)))):
            msg = "fweights must be integer"
            raise TypeError(msg)
    if aweights is not None:
        aw = ravel(_asarray(aweights)).astype(mx.float32)
        if int(aw.shape[0]) != obs:
            msg = "incompatible numbers of samples and aweights"
            raise RuntimeError(msg)
        if _scalar_bool(any(less(aw, 0))):
            msg = "aweights cannot be negative"
            raise ValueError(msg)

    ddof_i = int(ddof) if ddof is not None else (0 if bias else 1)
    if fw is None and aw is None:
        avg = mean(x, axis=1, keepdims=True)
        xc = x - avg
        fact = float(obs - ddof_i)
        out = matmul(xc, swapaxes(conjugate(xc), -1, -2)) / fact
    else:
        w = None
        if fw is not None and aw is not None:
            w = fw * aw
        elif fw is not None:
            w = fw
        else:
            w = aw
        if w is None:
            msg = "Internal error: expected weights to be set"
            raise RuntimeError(msg)
        w = w.astype(mx.float32) if _is_complex_dtype(x.dtype) else w.astype(x.dtype)
        w_row = reshape(w, (1, obs))
        w_sum = sum(w)
        avg = sum(x * w_row, axis=1, keepdims=True) / w_sum
        xc = x - avg
        out = matmul(xc * w_row, swapaxes(conjugate(xc), -1, -2))
        if aw is None:
            fact = w_sum - float(ddof_i)
        else:
            awv = aw.astype(w.dtype)
            fact = w_sum - float(ddof_i) * sum(w * awv) / w_sum
        out = out / fact

    if tuple(int(s) for s in out.shape) == (1, 1):
        return out[0, 0]
    return out


def corrcoef(
    x: Any,
    y: Any | None = None,
    rowvar: bool = True,
    dtype: Any | None = None,
) -> mx.array:
    c = cov(x, y=y, rowvar=rowvar, dtype=dtype)
    if isinstance(c, mx.array) and c.ndim == 0:
        return c / c
    d = diagonal(c)
    d_real = real(d) if _is_complex_dtype(d.dtype) else d
    denom = outer(sqrt(d_real), sqrt(d_real))
    return c / denom


def min(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return mx.min(_asarray(a), axis=axis, keepdims=keepdims)


def max(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return mx.max(_asarray(a), axis=axis, keepdims=keepdims)


def amin(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return min(a, axis=axis, keepdims=keepdims)


def amax(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return max(a, axis=axis, keepdims=keepdims)


def nanmin(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    arr = _asarray(a)
    if _is_complex_dtype(arr.dtype):
        bridge.record_fallback("core.nanmin:complex")
        np_mod = _numpy()
        return mx.array(np_mod.nanmin(np_mod.asarray(a), axis=axis, keepdims=keepdims))
    if not _is_floating_or_bfloat_dtype(arr.dtype):
        return min(arr, axis=axis, keepdims=keepdims)
    mask = isnan(arr)
    replaced = where(mask, inf, arr)
    result = min(replaced, axis=axis, keepdims=keepdims)
    all_nan = all(mask, axis=axis, keepdims=keepdims)
    return where(all_nan, nan, result)


def nanmax(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    arr = _asarray(a)
    if _is_complex_dtype(arr.dtype):
        bridge.record_fallback("core.nanmax:complex")
        np_mod = _numpy()
        return mx.array(np_mod.nanmax(np_mod.asarray(a), axis=axis, keepdims=keepdims))
    if not _is_floating_or_bfloat_dtype(arr.dtype):
        return max(arr, axis=axis, keepdims=keepdims)
    mask = isnan(arr)
    replaced = where(mask, -inf, arr)
    result = max(replaced, axis=axis, keepdims=keepdims)
    all_nan = all(mask, axis=axis, keepdims=keepdims)
    return where(all_nan, nan, result)


def ptp(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    arr = _asarray(a)
    return mx.max(arr, axis=axis, keepdims=keepdims) - mx.min(arr, axis=axis, keepdims=keepdims)


def argmin(a: Any, axis: int | None = None, keepdims: bool = False) -> mx.array:
    return mx.argmin(_asarray(a), axis=axis, keepdims=keepdims)


def argmax(a: Any, axis: int | None = None, keepdims: bool = False) -> mx.array:
    return mx.argmax(_asarray(a), axis=axis, keepdims=keepdims)


def nanargmin(a: Any, axis: int | None = None) -> mx.array:
    arr = _asarray(a)
    mask = isnan(arr)
    if _scalar_bool(any(all(mask, axis=axis))):
        msg = "All-NaN slice encountered"
        raise ValueError(msg)
    return argmin(where(mask, inf, arr), axis=axis)


def nanargmax(a: Any, axis: int | None = None) -> mx.array:
    arr = _asarray(a)
    mask = isnan(arr)
    if _scalar_bool(any(all(mask, axis=axis))):
        msg = "All-NaN slice encountered"
        raise ValueError(msg)
    return argmax(where(mask, -inf, arr), axis=axis)


def all(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return mx.all(_asarray(a), axis=axis, keepdims=keepdims)


def any(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    return mx.any(_asarray(a), axis=axis, keepdims=keepdims)


def count_nonzero(a: Any, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> mx.array:
    mask = not_equal(_asarray(a), 0).astype(mx.int32)
    return mx.sum(mask, axis=axis, keepdims=keepdims)


def nonzero(a: Any) -> tuple[mx.array, ...]:
    arr = _asarray(a)
    flat_idx = flatnonzero(arr)
    if arr.ndim == 0:
        return (flat_idx.astype(mx.int64),)
    return tuple(idx.astype(mx.int64) for idx in unravel_index(flat_idx, arr.shape))


def flatnonzero(a: Any) -> mx.array:
    flat = ravel(_asarray(a))
    idx = arange(flat.shape[0], dtype=mx.int64)
    mask = not_equal(flat, 0).astype(mx.int64)
    if flat.shape[0] == 0:
        return idx
    keys = where(equal(mask, 1), idx, idx + int(flat.shape[0]))
    perm = argsort(keys)
    count = int(sum(mask).item())
    return perm[:count].astype(mx.int64)


def argwhere(a: Any) -> mx.array:
    arr = _asarray(a)
    if arr.ndim == 0:
        return zeros((1 if _scalar_bool(not_equal(arr, 0)) else 0, 0), dtype=mx.int64)
    coords = nonzero(arr)
    if arr.ndim == 1:
        return expand_dims(coords[0], 1).astype(mx.int64)
    return stack(coords, axis=1).astype(mx.int64)


def cumsum(a: Any, axis: int | None = None, dtype: Any | None = None) -> mx.array:
    arr_in = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(arr_in, dtype)
    with _cpu_default_device_for_dtypes(arr_in.dtype, reduce_dtype):
        arr = arr_in if reduce_dtype is None else arr_in.astype(reduce_dtype)
        return mx.cumsum(arr, axis=axis)


def cumulative_sum(a: Any, axis: int | None = None, dtype: Any | None = None) -> mx.array:
    return cumsum(a, axis=axis, dtype=dtype)


def cumprod(a: Any, axis: int | None = None, dtype: Any | None = None) -> mx.array:
    arr_in = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(arr_in, dtype)
    with _cpu_default_device_for_dtypes(arr_in.dtype, reduce_dtype):
        arr = arr_in if reduce_dtype is None else arr_in.astype(reduce_dtype)
        return mx.cumprod(arr, axis=axis)


def cumulative_prod(a: Any, axis: int | None = None, dtype: Any | None = None) -> mx.array:
    return cumprod(a, axis=axis, dtype=dtype)


def nancumsum(a: Any, axis: int | None = None, dtype: Any | None = None) -> mx.array:
    arr_in = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(arr_in, dtype)
    with _cpu_default_device_for_dtypes(arr_in.dtype, reduce_dtype):
        arr = arr_in if reduce_dtype is None else arr_in.astype(reduce_dtype)
        clean = where(isnan(arr), 0, arr)
        return mx.cumsum(clean, axis=axis)


def nancumprod(a: Any, axis: int | None = None, dtype: Any | None = None) -> mx.array:
    arr_in = _asarray(a)
    reduce_dtype = _default_sum_prod_dtype(arr_in, dtype)
    with _cpu_default_device_for_dtypes(arr_in.dtype, reduce_dtype):
        arr = arr_in if reduce_dtype is None else arr_in.astype(reduce_dtype)
        clean = where(isnan(arr), 1, arr)
        return mx.cumprod(clean, axis=axis)


def diff(
    a: Any,
    n: int = 1,
    axis: int = -1,
    prepend: Any = _NO_VALUE,
    append: Any = _NO_VALUE,
) -> mx.array:
    if n < 0:
        msg = f"order must be non-negative but got {n}"
        raise ValueError(msg)
    out = _asarray(a)
    if n == 0:
        return out
    ax = axis + out.ndim if axis < 0 else axis
    if not 0 <= ax < out.ndim:
        msg = "axis out of bounds"
        raise ValueError(msg)
    if prepend is not _NO_VALUE or append is not _NO_VALUE:
        parts: list[mx.array] = []

        def _edge_block(edge: Any) -> mx.array:
            edge_arr = _asarray(edge)
            if edge_arr.ndim == 0:
                shape = list(out.shape)
                shape[ax] = 1
                return full(tuple(shape), edge_arr.item(), dtype=edge_arr.dtype)
            if edge_arr.ndim != out.ndim:
                msg = "array dimensions for prepend/append must match input"
                raise ValueError(msg)
            for i in range(out.ndim):
                if i == ax:
                    continue
                if int(edge_arr.shape[i]) != int(out.shape[i]):
                    msg = "array dimensions for prepend/append must match input"
                    raise ValueError(msg)
            return edge_arr

        if prepend is not _NO_VALUE:
            parts.append(_edge_block(prepend))
        parts.append(out)
        if append is not _NO_VALUE:
            parts.append(_edge_block(append))
        out = concatenate(parts, axis=ax)
    for _ in range(n):
        s1, s0 = _slices_for_diff(ax, out.ndim)
        a1 = out[s1]
        a0 = out[s0]
        out = mx.not_equal(a1, a0) if out.dtype == mx.bool_ else mx.subtract(a1, a0)
    return out


def ediff1d(ary: Any, to_end: Any | None = None, to_begin: Any | None = None) -> mx.array:
    base = diff(ravel(_asarray(ary)))
    parts: list[mx.array] = []
    if to_begin is not None:
        parts.append(ravel(_asarray(to_begin)))
    parts.append(base)
    if to_end is not None:
        parts.append(ravel(_asarray(to_end)))
    if len(parts) == 1:
        return parts[0]
    return concatenate(parts)


def gradient(
    f: Any,
    *varargs: Any,
    axis: int | Sequence[int] | None = None,
    edge_order: int = 1,
) -> mx.array | list[mx.array]:
    if edge_order not in {1, 2}:
        msg = "edge_order greater than 2 not supported"
        raise ValueError(msg)
    arr = _asarray(f)
    if arr.ndim == 0:
        msg = "Shape of array too small to calculate a numerical gradient"
        raise ValueError(msg)
    if _is_bool_dtype(arr.dtype) or _is_integer_dtype(arr.dtype):
        arr = arr.astype(mx.float32)

    axes = (
        tuple(range(arr.ndim))
        if axis is None
        else (
            _normalize_axes(axis, arr.ndim) if not isinstance(axis, int) else (_normalize_axis_index(axis, arr.ndim),)
        )
    )
    if len(varargs) == 0:
        spacings: list[Any] = [1.0] * len(axes)
    elif len(varargs) == 1:
        v0 = _asarray(varargs[0])
        if v0.ndim == 0:
            spacings = [varargs[0]] * len(axes)
        elif len(axes) == 1:
            spacings = [varargs[0]]
        else:
            msg = "invalid number of spacing arguments"
            raise TypeError(msg)
    elif len(varargs) == len(axes):
        spacings = list(varargs)
    else:
        msg = "invalid number of spacing arguments"
        raise TypeError(msg)

    def _grad_axis(x: mx.array, ax: int, spacing: Any) -> mx.array:
        n_ax = int(x.shape[ax])
        if n_ax < edge_order + 1:
            msg = "Shape of array too small to calculate a numerical gradient"
            raise ValueError(msg)
        sp_arr = _asarray(spacing)
        if sp_arr.ndim == 0:
            h = sp_arr.astype(mx.float32)
            if edge_order == 1:
                first = (_slice_along_axis(x, ax, 1, 2) - _slice_along_axis(x, ax, 0, 1)) / h
                last = (_slice_along_axis(x, ax, n_ax - 1, n_ax) - _slice_along_axis(x, ax, n_ax - 2, n_ax - 1)) / h
            else:
                first = (
                    -3.0 * _slice_along_axis(x, ax, 0, 1)
                    + 4.0 * _slice_along_axis(x, ax, 1, 2)
                    - _slice_along_axis(x, ax, 2, 3)
                ) / (2.0 * h)
                last = (
                    3.0 * _slice_along_axis(x, ax, n_ax - 1, n_ax)
                    - 4.0 * _slice_along_axis(x, ax, n_ax - 2, n_ax - 1)
                    + _slice_along_axis(x, ax, n_ax - 3, n_ax - 2)
                ) / (2.0 * h)
            if n_ax == 2:
                return concatenate([first, last], axis=ax)
            interior = (_slice_along_axis(x, ax, 2, None) - _slice_along_axis(x, ax, None, -2)) / (2.0 * h)
            return concatenate([first, interior, last], axis=ax)

        coords = ravel(sp_arr).astype(mx.float32)
        if int(coords.shape[0]) != n_ax:
            msg = "distances must match the length of the corresponding dimension"
            raise ValueError(msg)
        if edge_order == 1:
            first = (_slice_along_axis(x, ax, 1, 2) - _slice_along_axis(x, ax, 0, 1)) / (coords[1] - coords[0])
            last = (_slice_along_axis(x, ax, n_ax - 1, n_ax) - _slice_along_axis(x, ax, n_ax - 2, n_ax - 1)) / (
                coords[n_ax - 1] - coords[n_ax - 2]
            )
        else:
            dx1 = coords[1] - coords[0]
            dx2 = coords[2] - coords[1]
            a0 = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
            b0 = (dx1 + dx2) / (dx1 * dx2)
            c0 = -dx1 / (dx2 * (dx1 + dx2))
            first = (
                a0 * _slice_along_axis(x, ax, 0, 1)
                + b0 * _slice_along_axis(x, ax, 1, 2)
                + c0 * _slice_along_axis(x, ax, 2, 3)
            )
            dx1e = coords[n_ax - 2] - coords[n_ax - 3]
            dx2e = coords[n_ax - 1] - coords[n_ax - 2]
            ae = dx2e / (dx1e * (dx1e + dx2e))
            be = -(dx1e + dx2e) / (dx1e * dx2e)
            ce = (2.0 * dx2e + dx1e) / (dx2e * (dx1e + dx2e))
            last = (
                ae * _slice_along_axis(x, ax, n_ax - 3, n_ax - 2)
                + be * _slice_along_axis(x, ax, n_ax - 2, n_ax - 1)
                + ce * _slice_along_axis(x, ax, n_ax - 1, n_ax)
            )

        if n_ax == 2:
            return concatenate([first, last], axis=ax)

        dx1 = coords[1:-1] - coords[:-2]
        dx2 = coords[2:] - coords[1:-1]
        shape = [1] * x.ndim
        shape[ax] = n_ax - 2
        dx1b = reshape(dx1, shape)
        dx2b = reshape(dx2, shape)
        aa = -dx2b / (dx1b * (dx1b + dx2b))
        bb = (dx2b - dx1b) / (dx1b * dx2b)
        cc = dx1b / (dx2b * (dx1b + dx2b))
        interior = (
            aa * _slice_along_axis(x, ax, None, -2)
            + bb * _slice_along_axis(x, ax, 1, -1)
            + cc * _slice_along_axis(x, ax, 2, None)
        )
        return concatenate([first, interior, last], axis=ax)

    grads = [_grad_axis(arr, ax, sp) for ax, sp in zip(axes, spacings, strict=False)]
    return grads[0] if len(grads) == 1 else grads


def trapezoid(y: Any, x: Any | None = None, dx: float = 1.0, axis: int = -1) -> mx.array:
    arr = _asarray(y)
    ax = _normalize_axis_index(axis, arr.ndim)
    s1, s0 = _slices_for_diff(ax, arr.ndim)
    y1 = arr[s1]
    y0 = arr[s0]
    if x is None:
        d = dx
    else:
        x_arr = _asarray(x)
        if x_arr.ndim == 1:
            d1 = diff(x_arr)
            shape = [1] * arr.ndim
            shape[ax] = d1.shape[0]
            d = reshape(d1, shape)
        else:
            d = diff(x_arr, axis=ax)
    return sum((y1 + y0) * 0.5 * d, axis=ax)


def trapz(y: Any, x: Any | None = None, dx: float = 1.0, axis: int = -1) -> mx.array:
    return trapezoid(y, x=x, dx=dx, axis=axis)


def sort(a: Any, axis: int | None = -1) -> mx.array:
    return mx.sort(_asarray(a), axis=axis)


def argsort(a: Any, axis: int | None = -1) -> mx.array:
    return mx.argsort(_asarray(a), axis=axis)


def _coerce_partition_kth(kth: Any) -> int | None:
    if isinstance(kth, (list, tuple)):
        return None
    if hasattr(kth, "shape") and not isinstance(kth, (int, bool)):
        item = _item_from_scalar_like(kth)
        if item is _NO_VALUE:
            return None
        try:
            return int(item)
        except (TypeError, ValueError, OverflowError):
            return None
    exact = _coerce_exact_scalar_int(kth, allow_bool=True)
    if exact is not None:
        return exact
    try:
        return int(kth)
    except (TypeError, ValueError, OverflowError):
        return None


def partition(a: Any, kth: Any, axis: int = -1) -> mx.array:
    kth_i = _coerce_partition_kth(kth)
    if kth_i is not None:
        with suppress(TypeError, ValueError, RuntimeError, NotImplementedError):
            return mx.partition(_asarray(a), kth_i, axis=axis)
    bridge.record_fallback("core.partition:numpy")
    np_mod = _numpy()
    result = np_mod.partition(np_mod.asarray(a), kth, axis=axis)
    return mx.array(result)


def argpartition(a: Any, kth: Any, axis: int = -1) -> mx.array:
    kth_i = _coerce_partition_kth(kth)
    if kth_i is not None:
        with suppress(TypeError, ValueError, RuntimeError, NotImplementedError):
            return mx.argpartition(_asarray(a), kth_i, axis=axis).astype(mx.int64)
    bridge.record_fallback("core.argpartition:numpy")
    np_mod = _numpy()
    result = np_mod.argpartition(np_mod.asarray(a), kth, axis=axis)
    return mx.array(result)


def lexsort(keys: Sequence[Any], axis: int = -1) -> mx.array:
    bridge.record_fallback("core.lexsort:numpy")
    np_mod = _numpy()
    result = np_mod.lexsort(tuple(np_mod.asarray(k) for k in keys), axis=axis)
    return mx.array(result)


def sort_complex(a: Any) -> mx.array:
    bridge.record_fallback("core.sort_complex:numpy")
    np_mod = _numpy()
    result = np_mod.sort_complex(np_mod.asarray(a))
    return mx.array(result)


def searchsorted(a: Any, v: Any, side: str = "left", sorter: Any | None = None) -> mx.array:
    if side not in {"left", "right"}:
        msg = "side must be 'left' or 'right'"
        raise ValueError(msg)
    arr = ravel(_asarray(a))
    if sorter is not None:
        arr = take(arr, _asarray(sorter), axis=0)
    vv = _asarray(v)
    comp = less(arr, expand_dims(vv, -1)) if side == "left" else less_equal(arr, expand_dims(vv, -1))
    return sum(comp.astype(mx.int64), axis=-1).astype(mx.int64)


def digitize(x: Any, bins: Any, right: bool = False) -> mx.array:
    bins_arr = ravel(_asarray(bins))
    if bins_arr.ndim != 1:
        msg = "bins must be 1-D"
        raise ValueError(msg)
    if bins_arr.shape[0] == 0:
        return zeros_like(_asarray(x), dtype=mx.int64)
    if bins_arr.shape[0] == 1:
        cmp = less(_asarray(x), bins_arr[0]) if not right else less_equal(_asarray(x), bins_arr[0])
        return where(cmp, 0, 1).astype(mx.int64)
    inc = _scalar_bool(all(greater_equal(bins_arr[1:], bins_arr[:-1])))
    dec = _scalar_bool(all(less_equal(bins_arr[1:], bins_arr[:-1])))
    if not (inc or dec):
        msg = "bins must be monotonically increasing or decreasing"
        raise ValueError(msg)
    if inc:
        side = "left" if right else "right"
        return searchsorted(bins_arr, x, side=side)
    rev = bins_arr[::-1]
    side = "right" if right else "left"
    return mx.array(bins_arr.shape[0], dtype=mx.int64) - searchsorted(rev, x, side=side)


def _hist_scalar_int(value: Any) -> int | None:
    return _coerce_exact_scalar_int(value, allow_bool=False)


def _hist_bin_edges_1d(values: mx.array, bins: Any, range_: tuple[float, float] | None) -> mx.array:
    bins_count = _hist_scalar_int(bins)
    if bins_count is not None:
        if bins_count < 1:
            msg = "`bins` must be positive"
            raise ValueError(msg)
        if range_ is None:
            if int(values.shape[0]) == 0:
                lo = 0.0
                hi = 1.0
            else:
                if _is_complex_dtype(values.dtype):
                    msg = "Complex data not supported for histogram"
                    raise TypeError(msg)
                lo = float(min(values).item())
                hi = float(max(values).item())
        else:
            lo, hi = float(range_[0]), float(range_[1])
            if not math.isfinite(lo) or not math.isfinite(hi):
                msg = "supplied range is not finite"
                raise ValueError(msg)
            if hi < lo:
                msg = "max must be larger than min in range parameter"
                raise ValueError(msg)
        if hi == lo:
            lo -= 0.5
            hi += 0.5
        edge_dtype = values.dtype if _is_floating_or_bfloat_dtype(values.dtype) else _NUMPY_DEFAULT_FLOAT_DTYPE
        return linspace(lo, hi, num=bins_count + 1, endpoint=True, dtype=edge_dtype)

    edges = ravel(_asarray(bins))
    if edges.ndim != 1:
        msg = "`bins` must be 1D"
        raise ValueError(msg)
    if int(edges.shape[0]) < 2:
        msg = "`bins` must contain at least 2 edges"
        raise ValueError(msg)
    if _scalar_bool(any(less(_slice_along_axis(edges, 0, 1, None), _slice_along_axis(edges, 0, None, -1)))):
        msg = "`bins` must increase monotonically"
        raise ValueError(msg)
    return edges


def _hist_bin_indices(values: mx.array, edges: mx.array) -> tuple[mx.array, mx.array]:
    n_bins = int(edges.shape[0]) - 1
    with _cpu_default_device_for_dtypes(values.dtype, edges.dtype):
        idx = searchsorted(edges, values, side="right").astype(mx.int64) - 1
        if n_bins > 0:
            idx = where(equal(values, edges[-1]), n_bins - 1, idx).astype(mx.int64)
        valid = logical_and(greater_equal(idx, 0), less(idx, n_bins))
    return idx, valid


def _histogramdd_points(sample: Any) -> mx.array:
    if isinstance(sample, (list, tuple)):
        if len(sample) == 0:
            msg = "sample must contain at least one coordinate array"
            raise ValueError(msg)
        cols = [ravel(_asarray(col)) for col in sample]
        n = int(cols[0].shape[0])
        for col in cols:
            if col.ndim != 1:
                msg = "sample coordinate arrays must be 1D"
                raise ValueError(msg)
            if int(col.shape[0]) != n:
                msg = "sample coordinate arrays must be the same length"
                raise ValueError(msg)
        return stack(cols, axis=1)
    pts = _asarray(sample)
    if pts.ndim == 0:
        return reshape(pts, (1, 1))
    if pts.ndim == 1:
        return reshape(pts, (int(pts.shape[0]), 1))
    if pts.ndim != 2:
        msg = "sample must be a 2D array"
        raise ValueError(msg)
    return pts


def bincount(x: Any, weights: Any | None = None, minlength: int = 0) -> mx.array:
    arr = _asarray(x)
    if arr.ndim != 1:
        msg = "object too deep for desired array"
        raise ValueError(msg)
    ml = int(minlength)
    if ml < 0:
        msg = "minlength must be non-negative"
        raise ValueError(msg)

    if _is_bool_dtype(arr.dtype) or _is_integer_dtype(arr.dtype):
        idx = arr.astype(mx.int64)
    else:
        msg = "bincount only supports integer arrays"
        raise TypeError(msg)

    if _scalar_bool(any(less(idx, 0))):
        msg = "x must be non-negative"
        raise ValueError(msg)

    n = int(arr.shape[0])
    out_len = ml if n == 0 else builtins.max(ml, int(max(idx).item()) + 1)

    if weights is None:
        out = zeros((out_len,), dtype=mx.int32)
        if n == 0:
            return out.astype(mx.int64)
        vals = ones((n,), dtype=mx.int32)
        return out.at[idx.astype(mx.int32)].add(vals).astype(mx.int64)

    w = _asarray(weights)
    if w.ndim != 1 or int(w.shape[0]) != n:
        msg = "weights should have the same shape as x"
        raise ValueError(msg)
    if _is_complex_dtype(w.dtype):
        msg = "complex weights are not supported"
        raise TypeError(msg)
    vals_w = w if _is_floating_or_bfloat_dtype(w.dtype) else w.astype(mx.float32)
    out = zeros((out_len,), dtype=vals_w.dtype)
    if n == 0:
        return out
    return out.at[idx.astype(mx.int32)].add(vals_w)


def histogram(
    a: Any,
    bins: int | Sequence[float] = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
    weights: Any | None = None,
) -> tuple[mx.array, mx.array]:
    if isinstance(bins, str):
        bridge.record_fallback("core.histogram:str_bins")
        np_mod = _numpy()
        weights_np = None if weights is None else np_mod.asarray(weights)
        hist, bin_edges = np_mod.histogram(
            np_mod.asarray(a),
            bins=bins,
            range=range,
            density=density,
            weights=weights_np,
        )
        return mx.array(hist), mx.array(bin_edges)

    vals = ravel(_asarray(a))
    edges = _hist_bin_edges_1d(vals, bins, range)
    idx, valid = _hist_bin_indices(vals, edges)
    sel = flatnonzero(valid)
    idx_valid = take(idx, sel, axis=0).astype(mx.int64)

    weights_valid = None
    if weights is not None:
        w = ravel(_asarray(weights))
        if tuple(int(s) for s in w.shape) != tuple(int(s) for s in vals.shape):
            msg = "weights should have the same shape as a"
            raise ValueError(msg)
        weights_valid = take(w, sel, axis=0)

    hist = bincount(idx_valid, weights=weights_valid, minlength=int(edges.shape[0]) - 1)
    if density:
        hist_f = hist.astype(mx.float32)
        widths = diff(edges).astype(hist_f.dtype)
        total = sum(hist_f)
        denom = widths * total
        nan_fill = full(hist_f.shape, nan, dtype=hist_f.dtype)
        hist = where(equal(total, 0), nan_fill, hist_f / denom)
    return hist, edges


def histogram2d(
    x: Any,
    y: Any,
    bins: Any = 10,
    range: Sequence[tuple[float, float]] | None = None,
    density: bool = False,
    weights: Any | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    if _hist_scalar_int(bins) is not None:
        bins_xy = (bins, bins)
    elif isinstance(bins, (list, tuple)):
        bins_seq = list(bins)
        bins_xy = (bins_seq[0], bins_seq[1]) if len(bins_seq) == 2 else (bins, bins)
    else:
        bins_xy = (bins, bins)

    sample = stack([ravel(_asarray(x)), ravel(_asarray(y))], axis=1)
    hist, edges = histogramdd(sample, bins=bins_xy, range=range, density=density, weights=weights)
    return hist, edges[0], edges[1]


def histogramdd(
    sample: Any,
    bins: Any = 10,
    range: Sequence[tuple[float, float]] | None = None,
    density: bool = False,
    weights: Any | None = None,
) -> tuple[mx.array, list[mx.array]]:
    pts = _histogramdd_points(sample)
    n_samples = int(pts.shape[0])
    n_dims = int(pts.shape[1])

    bins_scalar = _hist_scalar_int(bins)
    if bins_scalar is not None:
        bins_spec = [bins_scalar] * n_dims
    elif isinstance(bins, (list, tuple)):
        bins_seq = list(bins)
        if len(bins_seq) != n_dims:
            msg = "The dimension of bins must be equal to the dimension of the sample"
            raise ValueError(msg)
        bins_spec = bins_seq
    else:
        if n_dims != 1:
            msg = "bins must be an int or a sequence of bins per dimension"
            raise ValueError(msg)
        bins_spec = [bins]

    if range is None:
        ranges = [None] * n_dims
    else:
        ranges = list(range)
        if len(ranges) != n_dims:
            msg = "range must be a sequence of length equal to the sample dimension"
            raise ValueError(msg)

    edges: list[mx.array] = []
    n_bins_each: list[int] = []
    idx_cols: list[mx.array] = []
    valid_all = ones((n_samples,), dtype=mx.bool_)

    for dim in builtins.range(n_dims):
        values_d = pts[:, dim]
        if _is_complex_dtype(values_d.dtype):
            msg = "Complex data not supported for histogramdd"
            raise TypeError(msg)
        edges_d = _hist_bin_edges_1d(values_d, bins_spec[dim], ranges[dim])
        idx_d, valid_d = _hist_bin_indices(values_d, edges_d)
        edges.append(edges_d)
        n_bins_each.append(int(edges_d.shape[0]) - 1)
        idx_cols.append(idx_d.astype(mx.int64))
        valid_all = logical_and(valid_all, valid_d)

    sel = flatnonzero(valid_all)
    flat_size = _prod_int(n_bins_each)

    weights_valid = None
    if weights is not None:
        w = ravel(_asarray(weights))
        if w.ndim != 1 or int(w.shape[0]) != n_samples:
            msg = "weights should have the same length as sample"
            raise ValueError(msg)
        weights_valid = take(w, sel, axis=0)

    lin = zeros((int(sel.shape[0]),), dtype=mx.int64)
    stride = 1
    for dim in builtins.range(n_dims - 1, -1, -1):
        idx_valid = take(idx_cols[dim], sel, axis=0)
        lin = lin + idx_valid * stride
        stride *= n_bins_each[dim]

    hist = reshape(bincount(lin, weights=weights_valid, minlength=flat_size), tuple(n_bins_each))

    if density:
        hist_f = hist.astype(mx.float32)
        total = sum(hist_f)
        widths_prod: mx.array | float = 1.0
        for dim, edge in enumerate(edges):
            widths = diff(edge).astype(hist_f.dtype)
            shape = [1] * n_dims
            shape[dim] = int(widths.shape[0])
            widths_prod = widths_prod * reshape(widths, shape)
        nan_fill = full(hist_f.shape, nan, dtype=hist_f.dtype)
        hist = where(equal(total, 0), nan_fill, hist_f / (total * widths_prod))

    return hist, edges


def interp(
    x: Any,
    xp: Any,
    fp: Any,
    left: float | None = None,
    right: float | None = None,
    period: float | None = None,
) -> mx.array:
    if period is not None:
        p = float(period)
        if p == 0.0:
            msg = "period must be a non-zero value"
            raise ValueError(msg)
        p = abs(p)
        x_arr = _asarray(x).astype(mx.float32)
        xp_arr = ravel(_asarray(xp)).astype(mx.float32)
        fp_arr = ravel(_asarray(fp))
        if xp_arr.ndim != 1 or fp_arr.ndim != 1:
            msg = "xp and fp must be 1-D sequences"
            raise ValueError(msg)
        if int(xp_arr.shape[0]) != int(fp_arr.shape[0]):
            msg = "fp and xp are not of the same length"
            raise ValueError(msg)
        n = int(xp_arr.shape[0])
        if n == 0:
            msg = "array of sample points is empty"
            raise ValueError(msg)
        xp_mod = xp_arr % p
        x_mod = x_arr % p
        order = argsort(xp_mod, axis=0).astype(mx.int32)
        xp_sorted = take(xp_mod, order, axis=0)
        fp_sorted = take(fp_arr, order, axis=0)
        xp_ext = concatenate([xp_sorted[-1:] - p, xp_sorted, xp_sorted[:1] + p], axis=0)
        fp_ext = concatenate([fp_sorted[-1:], fp_sorted, fp_sorted[:1]], axis=0)
        return interp(x_mod, xp_ext, fp_ext)

    x_arr = _asarray(x).astype(mx.float32)
    xp_arr = ravel(_asarray(xp)).astype(mx.float32)
    fp_arr = ravel(_asarray(fp))
    if xp_arr.ndim != 1 or fp_arr.ndim != 1:
        msg = "xp and fp must be 1-D sequences"
        raise ValueError(msg)
    if int(xp_arr.shape[0]) != int(fp_arr.shape[0]):
        msg = "fp and xp are not of the same length"
        raise ValueError(msg)
    n = int(xp_arr.shape[0])
    if n == 0:
        msg = "array of sample points is empty"
        raise ValueError(msg)
    left_val = fp_arr[0] if left is None else _asarray(left)
    right_val = fp_arr[-1] if right is None else _asarray(right)
    if n == 1:
        base = full(
            tuple(int(s) for s in x_arr.shape),
            fp_arr[0].item() if hasattr(fp_arr[0], "item") else fp_arr[0],
            dtype=fp_arr.dtype,
        )
        base = where(less(x_arr, xp_arr[0]), left_val, base)
        return where(greater(x_arr, xp_arr[0]), right_val, base)

    idx = searchsorted(xp_arr, x_arr, side="left").astype(mx.int32)
    idx_hi = clip(idx, 1, n - 1).astype(mx.int32)
    idx_lo = (idx_hi - 1).astype(mx.int32)
    x0 = take(xp_arr, idx_lo, axis=0)
    x1 = take(xp_arr, idx_hi, axis=0)
    y0 = take(fp_arr, idx_lo, axis=0)
    y1 = take(fp_arr, idx_hi, axis=0)
    denom = x1 - x0
    slope = where(equal(denom, 0), 0.0, (y1 - y0) / denom)
    out = where(equal(denom, 0), y0, y0 + (x_arr - x0) * slope)
    out = where(less(x_arr, xp_arr[0]), left_val, out)
    return where(greater(x_arr, xp_arr[-1]), right_val, out)


def unique(ar: Any) -> mx.array:
    flat = ravel(_asarray(ar))
    if int(flat.shape[0]) <= 1:
        return flat
    s = sort(flat, axis=0)
    neq = logical_not(_equal_with_nan(_slice_along_axis(s, 0, 1, None), _slice_along_axis(s, 0, None, -1)))
    mask = concatenate([array([True], dtype=mx.bool_), neq], axis=0)
    return compress(mask, s, axis=0)


def in1d(ar1: Any, ar2: Any, assume_unique: bool = False, invert: bool = False) -> mx.array:
    return ravel(isin(ravel(_asarray(ar1)), ar2, assume_unique=assume_unique, invert=invert))


def isin(element: Any, test_elements: Any, assume_unique: bool = False, invert: bool = False) -> mx.array:
    elem = _asarray(element)
    tests = ravel(_asarray(test_elements))
    if int(tests.shape[0]) == 0:
        out = zeros(elem.shape, dtype=mx.bool_)
        return logical_not(out) if invert else out
    tests_u = tests if assume_unique else unique(tests)
    tests_s = sort(tests_u, axis=0)
    flat = ravel(elem)
    n = int(tests_s.shape[0])
    pos = searchsorted(tests_s, flat, side="left").astype(mx.int64)
    pos_clip = clip(pos, 0, n - 1).astype(mx.int32)
    cand = take(tests_s, pos_clip, axis=0)
    mask_flat = logical_and(less(pos, n), _equal_with_nan(cand, flat))
    out = reshape(mask_flat, elem.shape)
    return logical_not(out) if invert else out


def intersect1d(ar1: Any, ar2: Any, assume_unique: bool = False, return_indices: bool = False) -> Any:
    x1 = ravel(_asarray(ar1))
    x2 = ravel(_asarray(ar2))
    u1 = x1 if assume_unique else unique(x1)
    u2 = x2 if assume_unique else unique(x2)
    if int(u1.shape[0]) == 0 or int(u2.shape[0]) == 0:
        vals = x1[:0]
        if return_indices:
            empty_idx = array([], dtype=mx.int64)
            return vals, empty_idx, empty_idx
        return vals
    joined = concatenate([u1, u2], axis=0)
    s = sort(joined, axis=0)
    dup = _equal_with_nan(_slice_along_axis(s, 0, 1, None), _slice_along_axis(s, 0, None, -1))
    vals = unique(compress(dup, _slice_along_axis(s, 0, 1, None), axis=0))
    if not return_indices:
        return vals

    idx_map1: dict[Any, int] = {}
    idx_map2: dict[Any, int] = {}
    for i, v in enumerate(_tolist_1d_list(x1)):
        idx_map1.setdefault(_python_scalar_key(v), i)
    for i, v in enumerate(_tolist_1d_list(x2)):
        idx_map2.setdefault(_python_scalar_key(v), i)
    vals_list = _tolist_1d_list(vals)
    i1 = [idx_map1[_python_scalar_key(v)] for v in vals_list]
    i2 = [idx_map2[_python_scalar_key(v)] for v in vals_list]
    return vals, array(i1, dtype=mx.int64), array(i2, dtype=mx.int64)


def union1d(ar1: Any, ar2: Any) -> mx.array:
    return unique(concatenate([ravel(_asarray(ar1)), ravel(_asarray(ar2))], axis=0))


def setdiff1d(ar1: Any, ar2: Any, assume_unique: bool = False) -> mx.array:
    x1 = ravel(_asarray(ar1))
    x2 = ravel(_asarray(ar2))
    if not assume_unique:
        x1 = unique(x1)
        x2 = unique(x2)
    return compress(isin(x1, x2, assume_unique=True, invert=True), x1, axis=0)


def setxor1d(ar1: Any, ar2: Any, assume_unique: bool = False) -> mx.array:
    x1 = ravel(_asarray(ar1))
    x2 = ravel(_asarray(ar2))
    if not assume_unique:
        x1 = unique(x1)
        x2 = unique(x2)
    joined = concatenate([x1, x2], axis=0)
    if int(joined.shape[0]) <= 1:
        return sort(joined, axis=0)
    s = sort(joined, axis=0)
    eq_adj = _equal_with_nan(_slice_along_axis(s, 0, 1, None), _slice_along_axis(s, 0, None, -1))
    eq_prev = concatenate([array([False], dtype=mx.bool_), eq_adj], axis=0)
    eq_next = concatenate([eq_adj, array([False], dtype=mx.bool_)], axis=0)
    singles = logical_not(logical_or(eq_prev, eq_next))
    return compress(singles, s, axis=0)


def compress(condition: Any, a: Any, axis: int | None = None) -> mx.array:
    cond = ravel(_asarray(condition)).astype(mx.bool_)
    arr = _asarray(a)
    if axis is None:
        flat = ravel(arr)
        usable = cond[: flat.shape[0]]
        return take(flat, flatnonzero(usable), axis=0)
    ax = _normalize_axis_index(axis, arr.ndim)
    usable = cond[: arr.shape[ax]]
    return take(arr, flatnonzero(usable), axis=ax)


def extract(condition: Any, arr: Any) -> mx.array:
    cond = ravel(_asarray(condition)).astype(mx.bool_)
    flat = ravel(_asarray(arr))
    usable = cond[: flat.shape[0]]
    return take(flat, flatnonzero(usable), axis=0)


def dot(a: Any, b: Any) -> mx.array:
    x = _asarray(a)
    y = _asarray(b)
    if x.ndim == 0 or y.ndim == 0:
        return multiply(x, y)
    if x.ndim == 1 and y.ndim == 1:
        return mx.sum(x * y)
    if x.ndim == 2 and y.ndim == 2:
        return mx.matmul(x, y)
    if y.ndim == 1:
        return mx.tensordot(x, y, axes=[[x.ndim - 1], [0]])
    return mx.tensordot(x, y, axes=[[x.ndim - 1], [builtins.max(y.ndim - 2, 0)]])


def vdot(a: Any, b: Any) -> mx.array:
    x = flatten(_asarray(a))
    y = flatten(_asarray(b))
    return mx.sum(mx.conjugate(x) * y)


def vecdot(a: Any, b: Any, axis: int = -1) -> mx.array:
    x = _asarray(a)
    y = _asarray(b)
    ax = _normalize_axis_index(axis, x.ndim)
    ay = _normalize_axis_index(axis, y.ndim)
    if x.shape[ax] != y.shape[ay]:
        msg = "Input sizes over selected axes do not match"
        raise ValueError(msg)
    if ax != x.ndim - 1:
        x = moveaxis(x, ax, -1)
    if ay != y.ndim - 1:
        y = moveaxis(y, ay, -1)
    return sum(conjugate(x) * y, axis=-1)


def inner(a: Any, b: Any) -> mx.array:
    return mx.inner(_asarray(a), _asarray(b))


def outer(a: Any, b: Any) -> mx.array:
    return mx.outer(_asarray(a), _asarray(b))


def matmul(a: Any, b: Any) -> mx.array:
    a_arr = _asarray(a)
    b_arr = _asarray(b)
    with _cpu_default_device_for_dtypes(a_arr.dtype, b_arr.dtype):
        return mx.matmul(a_arr, b_arr)


def matvec(a: Any, b: Any) -> mx.array:
    return matmul(a, b)


def vecmat(a: Any, b: Any) -> mx.array:
    return squeeze(matmul(expand_dims(a, -2), b), axis=-2)


def _normalize_tensordot_axes(
    axes: int | Sequence[Sequence[int]] | Sequence[int],
    x_ndim: int,
    y_ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(axes, int):
        n = int(axes)
        if n < 0:
            msg = "axes must be non-negative"
            raise ValueError(msg)
        if n > builtins.min(x_ndim, y_ndim):
            msg = "shape-mismatch for sum"
            raise ValueError(msg)
        return tuple(range(x_ndim - n, x_ndim)), tuple(range(n))
    if isinstance(axes, (str, bytes)) or not isinstance(axes, Sequence):
        msg = "axes must be an int or a pair of sequences"
        raise TypeError(msg)
    axes_list = list(axes)
    if len(axes_list) != 2:
        msg = "axes must consist of two sequences"
        raise ValueError(msg)
    ax_a_raw, ax_b_raw = axes_list
    if isinstance(ax_a_raw, int):
        ax_a_seq = (ax_a_raw,)
    else:
        if isinstance(ax_a_raw, (str, bytes)) or not isinstance(ax_a_raw, Sequence):
            msg = "axes entries must be ints or sequences of ints"
            raise TypeError(msg)
        ax_a_seq = tuple(int(ax) for ax in ax_a_raw)
    if isinstance(ax_b_raw, int):
        ax_b_seq = (ax_b_raw,)
    else:
        if isinstance(ax_b_raw, (str, bytes)) or not isinstance(ax_b_raw, Sequence):
            msg = "axes entries must be ints or sequences of ints"
            raise TypeError(msg)
        ax_b_seq = tuple(int(ax) for ax in ax_b_raw)
    if len(ax_a_seq) != len(ax_b_seq):
        msg = "shape-mismatch for sum"
        raise ValueError(msg)
    ax_a = _normalize_axes(ax_a_seq, x_ndim)
    ax_b = _normalize_axes(ax_b_seq, y_ndim)
    return ax_a, ax_b


def tensordot(a: Any, b: Any, axes: int | Sequence[Sequence[int]] = 2) -> mx.array:
    x = _asarray(a)
    y = _asarray(b)
    ax_a, ax_b = _normalize_tensordot_axes(axes, x.ndim, y.ndim)
    for xa, yb in zip(ax_a, ax_b, strict=False):
        if int(x.shape[xa]) != int(y.shape[yb]):
            msg = "shape-mismatch for sum"
            raise ValueError(msg)
    mx_axes: int | list[Sequence[int]]
    np_axes: int | tuple[tuple[int, ...], tuple[int, ...]]
    if isinstance(axes, int):
        mx_axes = axes
        np_axes = axes
    else:
        mx_axes = [ax_a, ax_b]
        np_axes = (ax_a, ax_b)
    try:
        return mx.tensordot(x, y, axes=mx_axes)
    except Exception:  # noqa: BLE001
        bridge.record_fallback("core.tensordot:mx_exception")
        np_mod = _numpy()
        return mx.array(np_mod.tensordot(np_mod.asarray(a), np_mod.asarray(b), axes=np_axes))


def einsum(subscripts: str, *operands: Any) -> mx.array:
    return mx.einsum(subscripts, *[_asarray(op) for op in operands])


def kron(a: Any, b: Any) -> mx.array:
    return mx.kron(_asarray(a), _asarray(b))


def cross(a: Any, b: Any, axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: int | None = None) -> mx.array:
    aa = _asarray(a)
    bb = _asarray(b)
    if axis is not None:
        axisa = axisb = axisc = axis
    axa = _normalize_axis_index(axisa, aa.ndim)
    axb = _normalize_axis_index(axisb, bb.ndim)
    if axa != aa.ndim - 1:
        aa = moveaxis(aa, axa, -1)
    if axb != bb.ndim - 1:
        bb = moveaxis(bb, axb, -1)
    na = int(aa.shape[-1])
    nb = int(bb.shape[-1])
    if na not in (2, 3) or nb not in (2, 3):
        msg = "incompatible dimensions for cross product (dimension must be 2 or 3)"
        raise ValueError(msg)

    ax0 = aa[..., 0]
    ay0 = aa[..., 1]
    az0 = aa[..., 2] if na == 3 else 0
    bx0 = bb[..., 0]
    by0 = bb[..., 1]
    bz0 = bb[..., 2] if nb == 3 else 0

    if na == 2 and nb == 2:
        return ax0 * by0 - ay0 * bx0

    cx = ay0 * bz0 - az0 * by0
    cy = az0 * bx0 - ax0 * bz0
    cz = ax0 * by0 - ay0 * bx0
    out = stack([cx, cy, cz], axis=-1)
    axc = _normalize_axis_index(axisc, out.ndim)
    if axc != out.ndim - 1:
        out = moveaxis(out, -1, axc)
    return out


def meshgrid(*xi: Any, copy: bool = True, sparse: bool = False, indexing: str = "xy") -> list[mx.array]:
    _ = copy
    return list(mx.meshgrid(*[_asarray(x) for x in xi], sparse=sparse, indexing=indexing))


def indices(dimensions: Sequence[int], dtype: Any = int, sparse: bool = False) -> mx.array | tuple[mx.array, ...]:
    dims = tuple(int(d) for d in dimensions)
    if builtins.any(d < 0 for d in dims):
        msg = "negative dimensions are not allowed"
        raise ValueError(msg)
    dt = _resolve_dtype(dtype) or mx.int32
    sparse_axes: list[mx.array] = []
    for i, d in enumerate(dims):
        shape = [1] * len(dims)
        shape[i] = d
        sparse_axes.append(reshape(arange(d, dtype=dt), shape))
    if sparse:
        return tuple(sparse_axes)
    if not dims:
        return zeros((0,), dtype=dt)
    return stack(broadcast_arrays(*sparse_axes), axis=0)


def ix_(*args: Any) -> tuple[mx.array, ...]:
    if len(args) == 0:
        return ()
    out: list[mx.array] = []
    n = len(args)
    for i, arg in enumerate(args):
        arr = ravel(_asarray(arg))
        if arr.ndim != 1:
            msg = "Cross index must be 1 dimensional"
            raise ValueError(msg)
        if arr.dtype == mx.bool_:
            arr = flatnonzero(arr)
        shape = [1] * n
        shape[i] = arr.shape[0]
        out.append(reshape(arr, shape))
    return tuple(out)


def unravel_index(indices_: Any, shape_: Sequence[int], order: str = "C") -> tuple[mx.array, ...]:
    dims = tuple(int(d) for d in shape_)
    if order not in {"C", "F"}:
        msg = "order must be 'C' or 'F'"
        raise ValueError(msg)
    if len(dims) == 0:
        return ()
    if builtins.any(d <= 0 for d in dims):
        msg = "dimensions must be positive"
        raise ValueError(msg)
    total = _prod_int(dims)
    idx = _asarray(indices_).astype(mx.int64)
    if _scalar_bool(any(logical_or(less(idx, 0), greater_equal(idx, total)))):
        msg = "invalid entry in index array"
        raise ValueError(msg)
    coords: list[mx.array] = []
    if order == "C":
        strides: list[int] = []
        running = 1
        for d in reversed(dims[1:]):
            running *= d
            strides.append(running)
        strides = [*list(reversed(strides)), 1]
        rem = idx
        for stride, dim in zip(strides, dims, strict=False):
            coord = floor_divide(rem, stride)
            rem = remainder(rem, stride)
            coords.append(coord.astype(mx.int64))
            if dim != 1:
                pass
    else:
        rem = idx
        for dim in dims:
            coords.append(remainder(rem, dim).astype(mx.int64))
            rem = floor_divide(rem, dim)
    return tuple(coords)


def ravel_multi_index(
    multi_index: Sequence[Any],
    dims: Sequence[int],
    mode: str | tuple[str, ...] = "raise",
    order: str = "C",
) -> mx.array:
    shape = tuple(int(d) for d in dims)
    if order not in {"C", "F"}:
        msg = "order must be 'C' or 'F'"
        raise ValueError(msg)
    if len(multi_index) != len(shape):
        msg = "parameter multi_index must be a sequence of length len(dims)"
        raise ValueError(msg)
    if builtins.any(d <= 0 for d in shape):
        msg = "invalid dims"
        raise ValueError(msg)
    if isinstance(mode, str):
        modes = (mode,) * len(shape)
    else:
        modes = tuple(mode)
        if len(modes) != len(shape):
            msg = "mode must be a string or sequence with one entry per dimension"
            raise ValueError(msg)
    if len(shape) == 0:
        return mx.array(0, dtype=mx.int64)
    idxs = broadcast_arrays(*[_asarray(v).astype(mx.int64) for v in multi_index])
    fixed: list[mx.array] = []
    for idx, dim, m in zip(idxs, shape, modes, strict=False):
        if m == "raise":
            bad = logical_or(less(idx, 0), greater_equal(idx, dim))
            if _scalar_bool(any(bad)):
                msg = "invalid entry in coordinates array"
                raise ValueError(msg)
            fixed.append(idx)
        elif m == "wrap":
            fixed.append(remainder(idx, dim).astype(mx.int64))
        elif m == "clip":
            fixed.append(clip(idx, 0, dim - 1).astype(mx.int64))
        else:
            msg = "invalid mode"
            raise ValueError(msg)
    strides: list[int] = []
    if order == "C":
        running = 1
        for d in reversed(shape[1:]):
            running *= d
            strides.append(running)
        strides = [*list(reversed(strides)), 1]
    else:
        running = 1
        for d in shape[:-1]:
            strides.append(running)
            running *= d
        strides.append(running)
    out = zeros_like(fixed[0], dtype=mx.int64)
    for idx, stride in zip(fixed, strides, strict=False):
        out = out + idx * stride
    return out.astype(mx.int64)


def diag_indices(n: int, ndim: int = 2) -> tuple[mx.array, ...]:
    idx = arange(n, dtype=mx.int64)
    return tuple(idx for _ in range(ndim))


def diag_indices_from(arr: Any) -> tuple[mx.array, ...]:
    a = _asarray(arr)
    if a.ndim < 2:
        msg = "input array must be at least 2-d"
        raise ValueError(msg)
    if not builtins.all(dim == a.shape[0] for dim in a.shape):
        msg = "All dimensions of input must be of equal length"
        raise ValueError(msg)
    return diag_indices(int(a.shape[0]), ndim=a.ndim)


def triu_indices(n: int, k: int = 0, m: int | None = None) -> tuple[mx.array, mx.array]:
    m = n if m is None else int(m)
    rows = broadcast_to(reshape(arange(n, dtype=mx.int64), (n, 1)), (n, m))
    cols = broadcast_to(reshape(arange(m, dtype=mx.int64), (1, m)), (n, m))
    return cast("tuple[mx.array, mx.array]", nonzero(greater_equal(cols - rows, k)))


def triu_indices_from(arr: Any, k: int = 0) -> tuple[mx.array, mx.array]:
    a = _asarray(arr)
    if a.ndim != 2:
        msg = "input array must be 2-d"
        raise ValueError(msg)
    return triu_indices(int(a.shape[0]), k=k, m=int(a.shape[1]))


def tril_indices(n: int, k: int = 0, m: int | None = None) -> tuple[mx.array, mx.array]:
    m = n if m is None else int(m)
    rows = broadcast_to(reshape(arange(n, dtype=mx.int64), (n, 1)), (n, m))
    cols = broadcast_to(reshape(arange(m, dtype=mx.int64), (1, m)), (n, m))
    return cast("tuple[mx.array, mx.array]", nonzero(less_equal(cols - rows, k)))


def tril_indices_from(arr: Any, k: int = 0) -> tuple[mx.array, mx.array]:
    a = _asarray(arr)
    if a.ndim != 2:
        msg = "input array must be 2-d"
        raise ValueError(msg)
    return tril_indices(int(a.shape[0]), k=k, m=int(a.shape[1]))


def shape(a: Any) -> tuple[int, ...]:
    return tuple(_asarray(a).shape)


def ndim(a: Any) -> int:
    return int(_asarray(a).ndim)


def size(a: Any, axis: int | None = None) -> int:
    arr = _asarray(a)
    if axis is None:
        return int(arr.size)
    return int(arr.shape[axis])


def item(a: Any, *args: Any) -> Any:
    arr = _asarray(a)
    if args:
        return arr[args].item()
    return arr.item()


def tolist(a: Any) -> Any:
    return _asarray(a).tolist()


def isscalar(x: Any) -> bool:
    if isinstance(x, (mx.array, list, tuple, dict, set)):
        return False
    if hasattr(x, "ndim"):
        try:
            return int(x.ndim) == 0
        except (TypeError, ValueError):
            return False
    return True


def conj(x: Any) -> mx.array:
    return mx.conj(_asarray(x))


def conjugate(x: Any) -> mx.array:
    return mx.conjugate(_asarray(x))


def real(x: Any) -> mx.array:
    return mx.real(_asarray(x))


def imag(x: Any) -> mx.array:
    return mx.imag(_asarray(x))


def real_if_close(a: Any, tol: float = 100) -> mx.array:
    arr = _asarray(a)
    if not _is_complex_dtype(arr.dtype):
        return arr
    eps = _float_eps_for_dtype(arr.dtype)
    thresh = tol * eps if tol > 1 else tol
    if _scalar_bool(all(less_equal(abs(imag(arr)), thresh))):
        return real(arr)
    return arr


def vander(x: Any, N: int | None = None, increasing: bool = False) -> mx.array:
    arr = ravel(_asarray(x))
    if arr.ndim != 1:
        msg = "x must be a one-dimensional array or sequence."
        raise ValueError(msg)
    ncols = arr.shape[0] if N is None else int(N)
    if ncols < 0:
        msg = "N must be nonnegative"
        raise ValueError(msg)
    powers = arange(ncols, dtype=mx.int32)
    if not increasing:
        powers = powers[::-1]
    return power(expand_dims(arr, -1), powers)


def trim_zeros(filt: Any, trim: str = "fb") -> mx.array:
    arr = ravel(_asarray(filt))
    if arr.ndim != 1:
        msg = "trim_zeros only works on 1-D arrays"
        raise ValueError(msg)
    trim_lower = trim.lower()
    if trim_lower not in {"f", "b", "fb", "bf"}:
        msg = "trim must be 'f', 'b', or 'fb'"
        raise ValueError(msg)
    nz = flatnonzero(arr)
    if nz.shape[0] == 0:
        return arr[:0]
    start = int(nz[0].item()) if "f" in trim_lower else 0
    stop = int(nz[-1].item()) + 1 if "b" in trim_lower else int(arr.shape[0])
    return arr[start:stop]


def unstack(x: Any, /, *, axis: int = 0) -> tuple[mx.array, ...]:
    arr = _asarray(x)
    ax = _normalize_axis_index(axis, arr.ndim)
    parts = split(arr, int(arr.shape[ax]), axis=ax)
    return tuple(squeeze(p, axis=ax) for p in parts)


def unwrap(p: Any, discont: float | None = None, axis: int = -1, period: float = 2 * builtins.float(pi)) -> mx.array:
    arr = _asarray(p).astype(mx.float32)
    if arr.ndim == 0:
        return arr
    ax = _normalize_axis_index(axis, arr.ndim)
    dd = diff(arr, axis=ax)
    half = period / 2.0
    ddmod = remainder(dd + half, period) - half
    ddmod = where(logical_and(equal(ddmod, -half), greater(dd, 0)), half, ddmod)
    threshold = builtins.max(half, half if discont is None else builtins.float(discont))
    ph_corr = ddmod - dd
    ph_corr = where(less(abs(dd), threshold), 0.0, ph_corr)
    pad_shape = list(ph_corr.shape)
    pad_shape[ax] = 1
    corr = concatenate([zeros(tuple(pad_shape), dtype=ph_corr.dtype), cumsum(ph_corr, axis=ax)], axis=ax)
    return arr + corr


def may_share_memory(a: Any, b: Any, max_work: Any | None = None) -> bool:
    if max_work is not None:
        _ = max_work
    return isinstance(a, mx.array) and isinstance(b, mx.array) and a is b


def shares_memory(a: Any, b: Any, max_work: Any | None = None) -> bool:
    return may_share_memory(a, b, max_work=max_work)


def eval(*arrays: Any) -> None:
    mx.eval(*[_asarray(a) for a in arrays])


__all__ = [
    "FallbackDisabledError",
    "abs",
    "absolute",
    "add",
    "all",
    "allclose",
    "amax",
    "amin",
    "any",
    "append",
    "arange",
    "arccos",
    "arcsin",
    "arctan",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "argwhere",
    "array",
    "array_equal",
    "array_split",
    "asanyarray",
    "asarray",
    "ascontiguousarray",
    "astype",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "average",
    "bfloat16",
    "bincount",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bool_",
    "broadcast_arrays",
    "broadcast_to",
    "ceil",
    "choose",
    "clip",
    "column_stack",
    "complex64",
    "compress",
    "concatenate",
    "conj",
    "conjugate",
    "copy",
    "corrcoef",
    "cos",
    "cosh",
    "count_nonzero",
    "cov",
    "cross",
    "cumprod",
    "cumsum",
    "deg2rad",
    "degrees",
    "delete",
    "diag",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "diff",
    "digitize",
    "divide",
    "dot",
    "dsplit",
    "dstack",
    "dtype",
    "e",
    "ediff1d",
    "einsum",
    "empty",
    "empty_like",
    "equal",
    "eval",
    "exp",
    "expand_dims",
    "expm1",
    "extract",
    "eye",
    "finfo",
    "flatnonzero",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "float16",
    "float32",
    "float64",
    "floor",
    "floor_divide",
    "full",
    "full_like",
    "gradient",
    "greater",
    "greater_equal",
    "histogram",
    "histogram2d",
    "histogramdd",
    "hsplit",
    "hstack",
    "identity",
    "iinfo",
    "imag",
    "in1d",
    "indices",
    "inf",
    "inner",
    "insert",
    "int8",
    "int16",
    "int32",
    "int64",
    "interp",
    "intersect1d",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isscalar",
    "item",
    "ix_",
    "kron",
    "less",
    "less_equal",
    "lexsort",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logspace",
    "matmul",
    "max",
    "maximum",
    "mean",
    "median",
    "meshgrid",
    "min",
    "minimum",
    "mod",
    "moveaxis",
    "multiply",
    "nan",
    "nan_to_num",
    "nanargmax",
    "nanargmin",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    "ndarray",
    "ndim",
    "negative",
    "newaxis",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "outer",
    "pad",
    "partition",
    "percentile",
    "permute_dims",
    "pi",
    "piecewise",
    "positive",
    "power",
    "prod",
    "ptp",
    "quantile",
    "rad2deg",
    "radians",
    "ravel",
    "ravel_multi_index",
    "real",
    "reciprocal",
    "remainder",
    "repeat",
    "reshape",
    "roll",
    "rot90",
    "round",
    "row_stack",
    "rsqrt",
    "searchsorted",
    "select",
    "set_strict_fallbacks",
    "setdiff1d",
    "setxor1d",
    "shape",
    "sign",
    "sin",
    "sinh",
    "size",
    "sort",
    "sort_complex",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "strict_fallbacks_enabled",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "tolist",
    "trace",
    "transpose",
    "trapezoid",
    "trapz",
    "tri",
    "tril",
    "tril_indices",
    "tril_indices_from",
    "triu",
    "triu_indices",
    "triu_indices_from",
    "true_divide",
    "trunc",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "union1d",
    "unique",
    "unravel_index",
    "var",
    "vdot",
    "vsplit",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]
