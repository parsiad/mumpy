"""Generic default-dtype parity checks driven by the audit manifest."""

from typing import Any

import pytest

import mumpy as mp

from .default_dtype_cases import DTYPE_CASES, DtypeCase


def _dtype_name(value: Any) -> str:
    dt = getattr(value, "dtype", value)
    return str(dt).removeprefix("mlx.core.").lower()


def _return_kind(value: Any) -> str:
    if isinstance(value, mp.MumPyArray):
        return "array"
    if isinstance(value, mp.MumPyScalar):
        return "scalar"
    return "python"


def _expected_return_kind(case: DtypeCase) -> str:
    if case.expected_return_kind != "infer":
        return case.expected_return_kind
    if case.expected_shape == ():
        return "scalar"
    return "array"


@pytest.mark.parametrize("case", DTYPE_CASES, ids=[case.case_id for case in DTYPE_CASES])
def test_default_dtype_parity_matrix(case: DtypeCase) -> None:
    got = case.mumpy_factory()
    assert _return_kind(got) == _expected_return_kind(case)
    if case.expected_shape is not None:
        assert getattr(got, "shape", None) == case.expected_shape

    if case.comparison == "numpy":
        assert case.numpy_factory is not None
        expected = case.numpy_factory()
        expected_dtype_name = _dtype_name(expected)
    else:
        assert case.expected_dtype_name is not None
        expected_dtype_name = case.expected_dtype_name

    assert _dtype_name(got) == expected_dtype_name
