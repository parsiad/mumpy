"""Helpers for dynamic module attribute and export surfaces."""

from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any


def _public_names(namespace: Any) -> list[str]:
    return [name for name in dir(namespace) if not name.startswith("_")]


def _optional_public_names(
    numpy_namespace_getter: Callable[[], Any] | None,
    numpy_exceptions: tuple[type[BaseException], ...],
) -> list[str]:
    if numpy_namespace_getter is None:
        return []
    if not numpy_exceptions:
        return _public_names(numpy_namespace_getter())
    with suppress(*numpy_exceptions):
        return _public_names(numpy_namespace_getter())
    return []


def resolve_module_fallback_attr(
    name: str,
    *,
    cache: dict[str, Any],
    native_namespace: Any,
    numpy_namespace_getter: Callable[[], Any],
    bridge_module: Any,
    bridge_namespace: str,
    public_module_name: str,
) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    if name in cache:
        return cache[name]
    if hasattr(native_namespace, name):
        obj = getattr(native_namespace, name)
        cache[name] = obj
        return obj
    numpy_namespace = numpy_namespace_getter()
    if hasattr(numpy_namespace, name):
        obj = getattr(numpy_namespace, name)
        if callable(obj) and not isinstance(obj, type):
            wrapped = bridge_module.numpy_callable(name, obj, namespace=bridge_namespace)
            cache[name] = wrapped
            return wrapped
        cache[name] = obj
        return obj
    msg = f"module '{public_module_name}' has no attribute {name!r}"
    raise AttributeError(msg)


def dynamic_dir(
    explicit_names: Sequence[str],
    native_namespace: Any,
    *,
    numpy_namespace_getter: Callable[[], Any] | None = None,
    numpy_exceptions: tuple[type[BaseException], ...] = (),
) -> list[str]:
    names = set(explicit_names)
    names.update(_public_names(native_namespace))
    names.update(_optional_public_names(numpy_namespace_getter, numpy_exceptions))
    return sorted(names)


def extend_all_with_fallback_names(
    explicit_names: Sequence[str],
    native_namespace: Any,
    *,
    numpy_namespace_getter: Callable[[], Any] | None = None,
    numpy_exceptions: tuple[type[BaseException], ...] = (),
) -> list[str]:
    merged = list(explicit_names)
    merged.extend(_public_names(native_namespace))
    merged.extend(_optional_public_names(numpy_namespace_getter, numpy_exceptions))
    seen = set()
    out: list[str] = []
    for name in merged:
        if name.startswith("_") or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out
