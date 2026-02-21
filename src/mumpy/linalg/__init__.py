"""Linear algebra public exports."""

from .._wrapping import wrap_dynamic_attr_value, wrap_public_callable  # noqa: TID252
from . import _linalg as impl
from ._linalg import (
    cholesky,
    cond,
    cross,
    det,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    inv,
    lstsq,
    matrix_power,
    matrix_rank,
    multi_dot,
    norm,
    pinv,
    qr,
    slogdet,
    solve,
    svd,
)

__all__ = (
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
