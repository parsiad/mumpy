"""Random sampling public exports."""

from __future__ import annotations

from functools import wraps
from typing import Any

from .._wrapping import wrap_dynamic_attr_value, wrap_public_callable, wrap_public_result  # noqa: TID252
from . import _random as impl
from ._random import (
    bernoulli,
    beta,
    binomial,
    chisquare,
    choice,
    dirichlet,
    exponential,
    f,
    gamma,
    gumbel,
    integers,
    key,
    laplace,
    multinomial,
    multivariate_normal,
    negative_binomial,
    normal,
    permutation,
    permuted,
    poisson,
    rand,
    randint,
    randn,
    random,
    random_sample,
    ranf,
    sample,
    seed,
    shuffle,
    split,
    standard_gamma,
    standard_normal,
    standard_t,
    state,
    truncated_normal,
    uniform,
)


def _wrap_generator_result(value: Any) -> Any:
    if isinstance(value, impl.Generator):
        return Generator(value)
    if isinstance(value, tuple):
        return tuple(_wrap_generator_result(v) for v in value)
    if isinstance(value, list):
        return [_wrap_generator_result(v) for v in value]
    if isinstance(value, dict):
        return {k: _wrap_generator_result(v) for k, v in value.items()}
    return wrap_public_result(value)


class Generator:
    """Public random Generator wrapper that preserves MumPy return semantics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, Generator):
                self._impl = arg._impl  # noqa: SLF001
                return
            if isinstance(arg, impl.Generator):
                self._impl = arg
                return
        self._impl = impl.Generator(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped implementation generator."""
        attr = getattr(self._impl, name)
        if callable(attr) and not isinstance(attr, type):
            wrapped_callable = wrap_public_callable(attr, api_name=name)

            @wraps(wrapped_callable)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return _wrap_generator_result(wrapped_callable(*args, **kwargs))

            self.__dict__[name] = wrapped
            return wrapped
        return _wrap_generator_result(attr)

    def __dir__(self) -> list[str]:
        """Return a curated directory listing for the wrapped generator."""
        return sorted({n for n in dir(self._impl) if not n.startswith("_")} | {"_impl"})

    def __repr__(self) -> str:
        """Return the representation of the wrapped implementation generator."""
        return repr(self._impl)


def default_rng(seed: Any | None = None) -> Generator:
    return Generator(impl.default_rng(seed))


__all__ = (
    "Generator",
    "bernoulli",
    "beta",
    "binomial",
    "chisquare",
    "choice",
    "default_rng",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "gumbel",
    "integers",
    "key",
    "laplace",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "normal",
    "permutation",
    "permuted",
    "poisson",
    "rand",
    "randint",
    "randn",
    "random",
    "random_sample",
    "ranf",
    "sample",
    "seed",
    "shuffle",
    "split",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "state",
    "truncated_normal",
    "uniform",
)

for _name in __all__:
    _obj = globals()[_name]
    if callable(_obj) and not isinstance(_obj, type):
        globals()[_name] = wrap_public_callable(_obj, api_name=_name)


def __getattr__(name: str):
    value = getattr(impl, name)
    value = wrap_dynamic_attr_value(name, value)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
