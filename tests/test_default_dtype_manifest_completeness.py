"""Ensure explicit public APIs are accounted for in the default-dtype audit manifest."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from .default_dtype_cases import (
    CASE_NAMES_BY_SURFACE,
    DTYPE_RELEVANT_EXPLICIT_NAMES_BY_SURFACE,
    classify_explicit_name,
)


def _load_coverage_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_api_coverage.py"
    spec = spec_from_file_location("mumpy_generate_api_coverage", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _explicit_names_by_surface(coverage_data: dict[str, list[tuple[str, str]]], module: Any) -> dict[str, set[str]]:
    explicit_categories = {
        module._CAT_NATIVE,  # noqa: SLF001
        module._CAT_EXPLICIT_FALLBACK,  # noqa: SLF001
    }
    out: dict[str, set[str]] = {}
    for surface, rows in coverage_data.items():
        out[surface] = {name for name, category in rows if category in explicit_categories}
    return out


def test_default_dtype_manifest_cases_reference_explicit_public_callables() -> None:
    cov_module = _load_coverage_script_module()
    coverage = cov_module._build_coverage_data()  # noqa: SLF001
    explicit = _explicit_names_by_surface(coverage, cov_module)

    for surface, names in CASE_NAMES_BY_SURFACE.items():
        missing = names - explicit.get(surface, set())
        assert not missing, f"dtype cases reference non-explicit names for {surface}: {sorted(missing)}"


def test_default_dtype_relevant_explicit_names_are_covered_or_documented() -> None:
    cov_module = _load_coverage_script_module()
    coverage = cov_module._build_coverage_data()  # noqa: SLF001
    explicit = _explicit_names_by_surface(coverage, cov_module)

    for surface, names in DTYPE_RELEVANT_EXPLICIT_NAMES_BY_SURFACE.items():
        missing_from_explicit = names - explicit.get(surface, set())
        assert not missing_from_explicit, (
            f"dtype-relevant manifest names are not explicit on {surface}: {sorted(missing_from_explicit)}"
        )
        uncovered = [
            name for name in names if classify_explicit_name(surface, name) not in {"tested", "documented-exception"}
        ]
        msg = f"dtype-relevant explicit names are not tested/documented on {surface}: {sorted(uncovered)}"
        assert not uncovered, msg


def test_all_explicit_public_callables_are_classified_for_dtype_audit() -> None:
    cov_module = _load_coverage_script_module()
    coverage = cov_module._build_coverage_data()  # noqa: SLF001
    explicit = _explicit_names_by_surface(coverage, cov_module)

    valid_categories = {"tested", "documented-exception", "non-numeric-output", "no-default-dtype"}
    for surface, names in explicit.items():
        for name in names:
            category = classify_explicit_name(surface, name)
            assert category in valid_categories, f"unclassified explicit name: {surface}.{name}"
