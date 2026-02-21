"""Completeness checks for the wrapper method dispatch manifest."""

import mumpy as mp
from mumpy import _wrapping

_PROPERTY_NAMES = {"real", "imag"}


def test_declared_wrapper_dispatch_names_exist_on_mumpy_array() -> None:
    x = mp.arange(6).reshape(2, 3)

    missing = [name for name in _wrapping._ARRAY_METHOD_DISPATCH_NAMES if not hasattr(x, name)]  # noqa: SLF001
    assert not missing, f"missing wrapper-dispatched methods/properties: {sorted(missing)}"

    for name in _wrapping._ARRAY_METHOD_DISPATCH_NAMES:  # noqa: SLF001
        attr = getattr(x, name)
        if name not in _PROPERTY_NAMES:
            assert callable(attr), f"wrapper dispatch entry {name!r} is not callable"
