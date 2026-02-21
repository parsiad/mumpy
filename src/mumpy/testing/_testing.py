from __future__ import annotations

from typing import Any

import numpy as np  # Intentionally do not use the NumPy bridge here


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def assert_allclose(
    actual: Any,
    desired: Any,
    rtol: float = 1e-7,
    atol: float = 0.0,
    equal_nan: bool = False,
    err_msg: str = "",
) -> None:
    np.testing.assert_allclose(
        _to_numpy(actual),
        _to_numpy(desired),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
    )


def assert_array_equal(actual: Any, desired: Any, err_msg: str = "") -> None:
    np.testing.assert_array_equal(_to_numpy(actual), _to_numpy(desired), err_msg=err_msg)


def assert_equal(actual: Any, desired: Any, err_msg: str = "") -> None:
    np.testing.assert_equal(_to_numpy(actual), _to_numpy(desired), err_msg=err_msg)


__all__ = ["assert_allclose", "assert_array_equal", "assert_equal"]
