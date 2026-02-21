"""FFT public exports."""

from .._wrapping import wrap_dynamic_attr_value, wrap_public_callable  # noqa: TID252
from . import _fft as impl
from ._fft import (
    fft,
    fft2,
    fftfreq,
    fftn,
    fftshift,
    ifft,
    ifft2,
    ifftn,
    ifftshift,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfftfreq,
    rfftn,
)

__all__ = (
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
