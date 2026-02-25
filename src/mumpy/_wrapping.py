"""Wrapper types and shared public boundary conversion helpers."""

# ruff: noqa: SLF001

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_RAW_ARRAY_TYPE = mx.array

_SCALAR_RESULT_NAMES = {
    "sum",
    "prod",
    "mean",
    "std",
    "var",
    "median",
    "quantile",
    "percentile",
    "nanquantile",
    "nanpercentile",
    "average",
    "nansum",
    "nanprod",
    "nanmean",
    "nanstd",
    "nanvar",
    "nanmedian",
    "min",
    "max",
    "amin",
    "amax",
    "nanmin",
    "nanmax",
    "ptp",
    "argmin",
    "argmax",
    "nanargmin",
    "nanargmax",
    "all",
    "any",
    "count_nonzero",
    "trace",
    "dot",
    "vdot",
    "inner",
    "tensordot",
    "det",
    "cond",
    "matrix_rank",
    "norm",
    "vector_norm",
    "matrix_norm",
    "float_power",
    "add",
    "subtract",
    "multiply",
    "true_divide",
    "floor_divide",
    "mod",
    "power",
    "matmul",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "negative",
    "positive",
    "abs",
    "bitwise_not",
}

_ARRAY_METHOD_DISPATCH_NAMES = {
    "abs",
    "all",
    "any",
    "argmax",
    "argmin",
    "astype",
    "conj",
    "cumprod",
    "cumsum",
    "diag",
    "diagonal",
    "flatten",
    "imag",
    "max",
    "mean",
    "min",
    "moveaxis",
    "prod",
    "real",
    "reshape",
    "round",
    "squeeze",
    "std",
    "sum",
    "swapaxes",
    "tolist",
    "transpose",
    "var",
}

_ARRAY_OPERATOR_FUNCTIONS = {
    "__add__": "add",
    "__radd__": "add",
    "__sub__": "subtract",
    "__rsub__": "subtract",
    "__mul__": "multiply",
    "__rmul__": "multiply",
    "__truediv__": "true_divide",
    "__rtruediv__": "true_divide",
    "__floordiv__": "floor_divide",
    "__rfloordiv__": "floor_divide",
    "__mod__": "mod",
    "__rmod__": "mod",
    "__pow__": "power",
    "__rpow__": "power",
    "__matmul__": "matmul",
    "__rmatmul__": "matmul",
    "__and__": "bitwise_and",
    "__rand__": "bitwise_and",
    "__or__": "bitwise_or",
    "__ror__": "bitwise_or",
    "__xor__": "bitwise_xor",
    "__rxor__": "bitwise_xor",
    "__eq__": "equal",
    "__ne__": "not_equal",
    "__lt__": "less",
    "__le__": "less_equal",
    "__gt__": "greater",
    "__ge__": "greater_equal",
}

_UNARY_OPERATOR_FUNCTIONS = {
    "__neg__": "negative",
    "__pos__": "positive",
    "__abs__": "abs",
    "__invert__": "bitwise_not",
}

_ARRAY_INPLACE_OPERATOR_DUNDERS = (
    "__iadd__",
    "__isub__",
    "__imul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__imatmul__",
    "__iand__",
    "__ior__",
    "__ixor__",
)


class MumPyAtIndexer:
    """Wrapper around MLX's immutable update indexer object."""

    __slots__ = ("_at",)

    def __init__(self, at_obj: Any) -> None:
        self._at = at_obj

    def __getitem__(self, key: Any) -> MumPyAtIndexer:
        return MumPyAtIndexer(self._at[unwrap_public_input(key)])

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._at, name)
        if not callable(attr):
            return attr

        return _wrap_native_callable_attr(attr, api_name=name)

    def __repr__(self) -> str:
        return repr(self._at)


class _MumPyValueBase:
    __slots__ = ("_mx",)
    __array_priority__ = 1000

    @property
    def mx(self) -> mx.array:
        return self._mx

    @property
    def dtype(self) -> Any:
        return self._mx.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self._mx.shape)

    @property
    def ndim(self) -> int:
        return int(self._mx.ndim)

    @property
    def size(self) -> int:
        return int(self._mx.size)

    @property
    def itemsize(self) -> int:
        return int(self._mx.itemsize)

    @property
    def nbytes(self) -> int:
        return int(self._mx.nbytes)

    @property
    def T(self) -> Any:  # noqa: N802
        return self.transpose()

    @property
    def real(self) -> Any:
        return _call_mumpy_function("real", self)

    @property
    def imag(self) -> Any:
        return _call_mumpy_function("imag", self)

    @property
    def at(self) -> MumPyAtIndexer:
        return MumPyAtIndexer(self._mx.at)

    def item(self) -> Any:
        return self._mx.item()

    def tolist(self) -> Any:
        return self._mx.tolist()

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> Any:
        arr = np.asarray(self._mx, dtype=dtype)
        if copy is True:
            return np.array(arr, copy=True)
        return arr

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        return self._mx.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self) -> Any:
        return self._mx.__dlpack_device__()

    def __iter__(self) -> Iterator[Any]:
        from . import _core as core  # noqa: PLC0415

        with core._cpu_default_device_for_dtype(self._mx.dtype):
            for v in self._mx:
                if isinstance(v, _RAW_ARRAY_TYPE) and v.ndim == 0:
                    yield MumPyScalar(v)
                else:
                    yield wrap_public_result(v)

    def __len__(self) -> int:
        return len(self._mx)

    def __getitem__(self, key: Any) -> Any:
        from . import _core as core  # noqa: PLC0415

        normalized_key = unwrap_public_input(key)
        with core._cpu_default_device_for_dtype(self._mx.dtype):
            result = self._mx[normalized_key]
        if isinstance(result, _RAW_ARRAY_TYPE) and result.ndim == 0:
            return MumPyScalar(result)
        return wrap_public_result(result)

    def __repr__(self) -> str:
        return repr(self._mx)

    def __str__(self) -> str:
        return str(self._mx)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if "out" in kwargs:
            return NotImplemented
        from . import _bridge as bridge  # noqa: PLC0415

        coerced_inputs, coerced_kwargs = _coerce_numpy_protocol_args_kwargs(bridge, inputs, kwargs)
        fn = getattr(ufunc, method)
        result = fn(*coerced_inputs, **coerced_kwargs)
        return bridge.wrap_from_numpy(result, api_name=getattr(ufunc, "__name__", None))

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        if kwargs is None:
            kwargs = {}
        from . import _bridge as bridge  # noqa: PLC0415

        coerced_args, coerced_kwargs = _coerce_numpy_protocol_args_kwargs(bridge, tuple(args), kwargs)
        try:
            result = func(*coerced_args, **coerced_kwargs)
        except TypeError:
            return NotImplemented
        return bridge.wrap_from_numpy(result, api_name=getattr(func, "__name__", None))

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._mx, name)
        if not callable(attr):
            return wrap_public_result(attr, api_name=name)

        if name in _ARRAY_METHOD_DISPATCH_NAMES:

            def dispatched_method(*args: Any, **kwargs: Any) -> Any:
                return _call_mumpy_function(name, self, *args, **kwargs)

            return dispatched_method

        return _wrap_native_callable_attr(attr, api_name=name)


class MumPyArray(_MumPyValueBase):
    """Public MumPy array wrapper around ``mlx.core.array``."""

    __slots__ = ()
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __init__(self, value: Any) -> None:
        if isinstance(value, MumPyArray):
            self._mx = value._mx
            return
        if isinstance(value, MumPyScalar):
            self._mx = value._mx
            return
        if not isinstance(value, _RAW_ARRAY_TYPE):
            msg = f"MumPyArray requires mlx.core.array, got {type(value)!r}"
            raise TypeError(msg)
        self._mx = value

    def __bool__(self) -> bool:
        if self._mx.ndim != 0 and self._mx.size != 1:
            msg = "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()."
            raise ValueError(msg)
        return bool(self._mx.item())

    def reshape(self, *shape: Any, order: str = "C") -> Any:
        newshape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
        return _call_mumpy_function("reshape", self, newshape, order=order)

    def view(self, dtype: Any | None = None, type: Any | None = None) -> Any:
        return _call_mumpy_function("view", self, dtype=dtype, type=type)

    def transpose(self, *axes: Any) -> Any:
        if not axes:
            axes_arg = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes_arg = axes[0]
        else:
            axes_arg = tuple(axes)
        return _call_mumpy_function("transpose", self, axes_arg)


class MumPyScalar(_MumPyValueBase):
    """NumPy-scalar-like wrapper for scalar MumPy results."""

    __slots__ = ()

    def __init__(self, value: Any) -> None:
        if isinstance(value, MumPyScalar):
            self._mx = value._mx
            return
        if isinstance(value, MumPyArray):
            if value._mx.ndim != 0:
                msg = "MumPyScalar requires a scalar array"
                raise TypeError(msg)
            self._mx = value._mx
            return
        if isinstance(value, _RAW_ARRAY_TYPE):
            if value.ndim != 0:
                msg = "MumPyScalar requires a 0-d mlx.core.array"
                raise TypeError(msg)
            self._mx = value
            return
        self._mx = mx.array(value)
        if self._mx.ndim != 0:
            msg = "MumPyScalar requires a scalar value"
            raise TypeError(msg)

    def __bool__(self) -> bool:
        return bool(self._mx.item())

    def __int__(self) -> int:
        return int(self._mx.item())

    def __index__(self) -> int:
        if self.dtype in {mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.uint16, mx.uint32, mx.uint64}:
            return int(self._mx.item())
        msg = f"{self.dtype} cannot be interpreted as an integer index"
        raise TypeError(msg)

    def __float__(self) -> float:
        return float(self._mx.item())

    def __complex__(self) -> complex:
        return complex(self._mx.item())

    def __hash__(self) -> int:
        return hash(self._mx.item())

    def __repr__(self) -> str:
        item_repr = repr(self._mx.item())
        return f"MumPyScalar({item_repr}, dtype={self.dtype})"


for _dunder, _fn_name in _UNARY_OPERATOR_FUNCTIONS.items():

    def _make_unary(fn_name: str):
        def _op(self: Any):
            return _call_mumpy_function(fn_name, self)

        return _op

    setattr(MumPyArray, _dunder, _make_unary(_fn_name))
    setattr(MumPyScalar, _dunder, _make_unary(_fn_name))


for _dunder, _fn_name in _ARRAY_OPERATOR_FUNCTIONS.items():

    def _make_binary(fn_name: str, reverse: bool):
        def _op(self: Any, other: Any):
            if reverse:
                return _call_mumpy_function(fn_name, other, self)
            return _call_mumpy_function(fn_name, self, other)

        return _op

    _reverse = _dunder.startswith("__r") and _dunder not in {"__repr__", "__reduce__"}
    setattr(MumPyArray, _dunder, _make_binary(_fn_name, _reverse))
    setattr(MumPyScalar, _dunder, _make_binary(_fn_name, _reverse))


def is_mumpy_array(value: Any) -> bool:
    return isinstance(value, MumPyArray)


def is_mumpy_scalar(value: Any) -> bool:
    return isinstance(value, MumPyScalar)


def unwrap_mx(value: Any) -> Any:
    if isinstance(value, (MumPyArray, MumPyScalar)):
        return value._mx
    return value


def unwrap_public_input(value: Any) -> Any:
    if isinstance(value, (MumPyArray, MumPyScalar)):
        return value._mx
    if isinstance(value, tuple):
        return tuple(unwrap_public_input(v) for v in value)
    if isinstance(value, list):
        return [unwrap_public_input(v) for v in value]
    if isinstance(value, dict):
        return {k: unwrap_public_input(v) for k, v in value.items()}
    return value


def _invoke_with_unwrapped_inputs(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return func(
        *(unwrap_public_input(arg) for arg in args),
        **{k: unwrap_public_input(v) for k, v in kwargs.items()},
    )


def _coerce_numpy_protocol_args_kwargs(
    bridge: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    coerced_args = tuple(bridge.coerce_to_numpy(v) for v in args)
    coerced_kwargs = {k: bridge.coerce_to_numpy(v) for k, v in kwargs.items()}
    return coerced_args, coerced_kwargs


def _wrap_native_callable_attr(attr: Any, *, api_name: str) -> Any:
    @wraps(attr)
    def method(*args: Any, **kwargs: Any) -> Any:
        result = _invoke_with_unwrapped_inputs(attr, args, kwargs)
        return wrap_public_result(result, api_name=api_name)

    return method


def _prefer_scalar_for_api(api_name: str | None) -> bool:
    return api_name in _SCALAR_RESULT_NAMES if api_name is not None else False


def wrap_public_result(value: Any, *, api_name: str | None = None) -> Any:
    if isinstance(value, (MumPyArray, MumPyScalar)):
        return value
    if isinstance(value, tuple):
        return tuple(wrap_public_result(v) for v in value)
    if isinstance(value, list):
        return [wrap_public_result(v) for v in value]
    if isinstance(value, dict):
        return {k: wrap_public_result(v) for k, v in value.items()}
    if isinstance(value, _RAW_ARRAY_TYPE):
        if value.ndim == 0 and _prefer_scalar_for_api(api_name):
            return MumPyScalar(value)
        return MumPyArray(value)
    return value


def wrap_public_callable(func: Callable[..., Any], *, api_name: str | None = None) -> Callable[..., Any]:
    name = api_name or getattr(func, "__name__", None)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = _invoke_with_unwrapped_inputs(func, args, kwargs)
        return wrap_public_result(result, api_name=name)

    return wrapper


def wrap_dynamic_attr_value(name: str, value: Any) -> Any:
    if callable(value) and not isinstance(value, type):
        return wrap_public_callable(value, api_name=name)
    return wrap_public_result(value, api_name=name)


def _call_mumpy_function(name: str, *args: Any, **kwargs: Any) -> Any:
    import mumpy as mp  # noqa: PLC0415

    fn = getattr(mp, name)
    return fn(*args, **kwargs)


def _raise_immutable_inplace_update(self: Any, other: Any) -> Any:
    del self, other
    msg = "MumPyArray is immutable; use x = x + y or a functional update instead"
    raise TypeError(msg)


for _dunder in _ARRAY_INPLACE_OPERATOR_DUNDERS:
    setattr(MumPyArray, _dunder, _raise_immutable_inplace_update)
