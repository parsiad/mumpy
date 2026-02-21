import os
import threading
from functools import wraps
from typing import Any

import mlx.core as mx


class FallbackDisabledError(RuntimeError):
    pass


_STRICT_FALLBACKS_STATE = {
    "enabled": str(os.environ.get("MUMPY_STRICT_NO_FALLBACK", "")).strip().lower() in {"1", "true", "yes", "on"},
}
_FALLBACK_COUNTS_LOCK = threading.Lock()
_FALLBACK_COUNTS: dict[str, int] = {}


def strict_fallbacks_enabled() -> bool:
    return bool(_STRICT_FALLBACKS_STATE["enabled"])


def set_strict_fallbacks(enabled: bool) -> bool:
    previous = bool(_STRICT_FALLBACKS_STATE["enabled"])
    _STRICT_FALLBACKS_STATE["enabled"] = bool(enabled)
    return previous


def record_fallback(site: str) -> None:
    key = str(site)
    with _FALLBACK_COUNTS_LOCK:
        _FALLBACK_COUNTS[key] = _FALLBACK_COUNTS.get(key, 0) + 1


def get_recorded_fallback_counts() -> dict[str, int]:
    with _FALLBACK_COUNTS_LOCK:
        return dict(_FALLBACK_COUNTS)


def reset_recorded_fallback_counts() -> None:
    with _FALLBACK_COUNTS_LOCK:
        _FALLBACK_COUNTS.clear()


def numpy_module():
    if strict_fallbacks_enabled():
        msg = "NumPy fallback is disabled. Clear MUMPY_STRICT_NO_FALLBACK or call set_strict_fallbacks(False)."
        raise FallbackDisabledError(msg)
    try:
        import numpy as np
    except ImportError as exc:
        msg = "This feature currently requires numpy to be installed"
        raise NotImplementedError(msg) from exc
    return np


def coerce_to_numpy(value: Any) -> Any:
    from . import _wrapping as wrapping

    np_mod = numpy_module()
    if wrapping.is_mumpy_array(value) or wrapping.is_mumpy_scalar(value):
        return np_mod.asarray(wrapping.unwrap_mx(value))
    if isinstance(value, mx.array):
        return np_mod.asarray(value)
    if isinstance(value, tuple):
        return tuple(coerce_to_numpy(v) for v in value)
    if isinstance(value, list):
        return [coerce_to_numpy(v) for v in value]
    if isinstance(value, dict):
        return {k: coerce_to_numpy(v) for k, v in value.items()}
    return value


def _mlx_dtype_from_numpy_dtype(np_dtype: Any) -> Any | None:
    name = str(getattr(np_dtype, "name", np_dtype)).lower()
    return getattr(mx, name, None)


def wrap_from_numpy(value: Any, *, api_name: str | None = None) -> Any:
    from . import _wrapping as wrapping

    np_mod = numpy_module()
    if isinstance(value, tuple):
        return tuple(wrap_from_numpy(v, api_name=api_name) for v in value)
    if isinstance(value, list):
        return [wrap_from_numpy(v, api_name=api_name) for v in value]
    if isinstance(value, dict):
        return {k: wrap_from_numpy(v, api_name=api_name) for k, v in value.items()}
    if isinstance(value, np_mod.ndarray):
        try:
            mlx_dtype = _mlx_dtype_from_numpy_dtype(value.dtype)
            if mlx_dtype is None:
                return wrapping.wrap_public_result(mx.array(value), api_name=api_name)
            return wrapping.wrap_public_result(mx.array(value, dtype=mlx_dtype), api_name=api_name)
        except Exception:  # noqa: BLE001
            return value
    if isinstance(value, np_mod.generic):
        try:
            mlx_dtype = _mlx_dtype_from_numpy_dtype(value.dtype)
            if mlx_dtype is not None:
                return wrapping.wrap_public_result(mx.array(value.item(), dtype=mlx_dtype), api_name=api_name)
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def _fallback_site(namespace: str | None, name: str, site: str | None) -> str:
    if site:
        return site
    if namespace:
        return f"{namespace}.dynamic:{name}"
    return name


def _set_wrapper_bridge_metadata(wrapper: Any, kind: str, fallback_site: str) -> None:
    for attr_name, value in (
        ("__mumpy_bridge_kind__", kind),
        ("__mumpy_fallback_site__", fallback_site),
        ("__mumpy_bridge_dynamic__", True),
    ):
        setattr(wrapper, attr_name, value)


def numpy_callable(
    name: str,
    func: Any,
    *,
    site: str | None = None,
    namespace: str | None = None,
    record: bool = True,
) -> Any:
    fallback_site = _fallback_site(namespace, name, site)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if strict_fallbacks_enabled():
            msg = (
                f"NumPy fallback for {name!r} is disabled. Clear MUMPY_STRICT_NO_FALLBACK or call "
                "set_strict_fallbacks(False)."
            )
            raise FallbackDisabledError(msg)
        if record:
            record_fallback(fallback_site)
        result = func(
            *(coerce_to_numpy(arg) for arg in args),
            **{k: coerce_to_numpy(v) for k, v in kwargs.items()},
        )
        return wrap_from_numpy(result)

    wrapper.__name__ = name
    wrapper.__doc__ = getattr(func, "__doc__", None)
    _set_wrapper_bridge_metadata(wrapper, "numpy", fallback_site)
    return wrapper


def hybrid_callable(
    name: str,
    mx_func: Any,
    np_func: Any,
    *,
    site: str | None = None,
    namespace: str | None = None,
) -> Any:
    fallback_site = _fallback_site(namespace, name, site)
    np_wrapper = numpy_callable(name, np_func, site=fallback_site, namespace=namespace, record=True)

    @wraps(mx_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return mx_func(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return np_wrapper(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__doc__ = getattr(mx_func, "__doc__", None) or getattr(np_func, "__doc__", None)
    _set_wrapper_bridge_metadata(wrapper, "hybrid", fallback_site)
    return wrapper
