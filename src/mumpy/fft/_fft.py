from collections.abc import Sequence
from typing import Any

import mlx.core as mx

from .. import _bridge as bridge
from .. import _core as core
from .. import _dynamic_exports as dynamic_exports

_FALLBACK_ATTR_CACHE: dict[str, Any] = {}
_NUMPY_EXPORT_EXCEPTIONS = (NotImplementedError, bridge.FallbackDisabledError)


def _asarray(a: Any) -> mx.array:
    return mx.asarray(a)


def _normalize_fft_axes_for_shift(axes: int | tuple[int, ...] | None) -> Sequence[int] | None:
    if axes is None:
        return None
    if isinstance(axes, int):
        return (axes,)
    return axes


def __getattr__(name: str) -> Any:
    return dynamic_exports.resolve_module_fallback_attr(
        name,
        cache=_FALLBACK_ATTR_CACHE,
        native_namespace=mx.fft,
        numpy_namespace_getter=lambda: bridge.numpy_module().fft,
        bridge_module=bridge,
        bridge_namespace="fft",
        public_module_name="mumpy.fft",
    )


def fft(a: Any, n: int | None = None, axis: int = -1) -> mx.array:
    return mx.fft.fft(_asarray(a), n=n, axis=axis)


def ifft(a: Any, n: int | None = None, axis: int = -1) -> mx.array:
    return mx.fft.ifft(_asarray(a), n=n, axis=axis)


def fft2(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = (-2, -1)) -> mx.array:
    return mx.fft.fft2(_asarray(a), s=s, axes=axes)


def ifft2(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = (-2, -1)) -> mx.array:
    return mx.fft.ifft2(_asarray(a), s=s, axes=axes)


def fftn(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None) -> mx.array:
    if s is not None and axes is None:
        axes = tuple(range(-len(s), 0))
    return mx.fft.fftn(_asarray(a), s=s, axes=axes)


def ifftn(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None) -> mx.array:
    if s is not None and axes is None:
        axes = tuple(range(-len(s), 0))
    return mx.fft.ifftn(_asarray(a), s=s, axes=axes)


def rfft(a: Any, n: int | None = None, axis: int = -1) -> mx.array:
    return mx.fft.rfft(_asarray(a), n=n, axis=axis)


def irfft(a: Any, n: int | None = None, axis: int = -1) -> mx.array:
    return mx.fft.irfft(_asarray(a), n=n, axis=axis)


def rfft2(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = (-2, -1)) -> mx.array:
    return mx.fft.rfft2(_asarray(a), s=s, axes=axes)


def irfft2(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = (-2, -1)) -> mx.array:
    return mx.fft.irfft2(_asarray(a), s=s, axes=axes)


def rfftn(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None) -> mx.array:
    if s is not None and axes is None:
        axes = tuple(range(-len(s), 0))
    return mx.fft.rfftn(_asarray(a), s=s, axes=axes)


def irfftn(a: Any, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None) -> mx.array:
    if s is not None and axes is None:
        axes = tuple(range(-len(s), 0))
    return mx.fft.irfftn(_asarray(a), s=s, axes=axes)


def fftshift(x: Any, axes: int | tuple[int, ...] | None = None) -> mx.array:
    return mx.fft.fftshift(_asarray(x), axes=_normalize_fft_axes_for_shift(axes))


def ifftshift(x: Any, axes: int | tuple[int, ...] | None = None) -> mx.array:
    return mx.fft.ifftshift(_asarray(x), axes=_normalize_fft_axes_for_shift(axes))


def fftfreq(n: int, d: float = 1.0) -> mx.array:
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    val = 1.0 / (n * d)
    if n % 2 == 0:
        pos = mx.arange(0, n // 2, dtype=mx.float32)
        neg = mx.arange(-(n // 2), 0, dtype=mx.float32)
    else:
        pos = mx.arange(0, (n - 1) // 2 + 1, dtype=mx.float32)
        neg = mx.arange(-((n - 1) // 2), 0, dtype=mx.float32)
    out = mx.concatenate([pos, neg]) * val
    return core._coerce_public_default_float_output(out)  # noqa: SLF001


def rfftfreq(n: int, d: float = 1.0) -> mx.array:
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    val = 1.0 / (n * d)
    out = mx.arange(0, n // 2 + 1, dtype=mx.float32) * val
    return core._coerce_public_default_float_output(out)  # noqa: SLF001


def hfft(a: Any, n: int | None = None, axis: int = -1) -> mx.array:
    arr = _asarray(a)
    n_out = (arr.shape[axis] - 1) * 2 if n is None else int(n)
    return mx.real(mx.fft.irfft(mx.conjugate(arr), n=n_out, axis=axis)) * float(n_out)


def ihfft(a: Any, n: int | None = None, axis: int = -1) -> mx.array:
    arr = _asarray(a)
    n_in = arr.shape[axis] if n is None else int(n)
    return mx.conjugate(mx.fft.rfft(arr, n=n_in, axis=axis)) / float(n_in)


__all__ = [
    "fft",
    "fft2",
    "fftfreq",
    "fftn",
    "fftshift",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "irfft",
    "irfft2",
    "irfftn",
    "rfft",
    "rfft2",
    "rfftfreq",
    "rfftn",
]


def __dir__() -> list[str]:
    return dynamic_exports.dynamic_dir(
        __all__,
        mx.fft,
        numpy_namespace_getter=lambda: bridge.numpy_module().fft,
        numpy_exceptions=_NUMPY_EXPORT_EXCEPTIONS,
    )


__all__ = dynamic_exports.extend_all_with_fallback_names(  # pyright: ignore[reportUnsupportedDunderAll]
    __all__,
    mx.fft,
    numpy_namespace_getter=lambda: bridge.numpy_module().fft,
    numpy_exceptions=_NUMPY_EXPORT_EXCEPTIONS,
)
