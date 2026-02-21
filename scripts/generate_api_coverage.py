#!/usr/bin/env python
"""Generate the checked-in MumPy coverage summary document."""

import ast
import importlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CAT_NATIVE = "Native explicit"
_CAT_EXPLICIT_FALLBACK = "Explicit (may fallback)"
_CAT_DYNAMIC = "Dynamic NumPy bridge"
_CAT_PASSTHROUGH = "Non-callable passthrough"
_CATEGORIES = [_CAT_NATIVE, _CAT_EXPLICIT_FALLBACK, _CAT_DYNAMIC, _CAT_PASSTHROUGH]
_MISSING_ATTR = object()

_FALLBACK_CALL_NAMES = {
    "_numpy",
    "_numpy_callable",
    "_hybrid_mx_numpy_callable",
    "_numpy_wrap_value",
    "_numpy_coerce_value",
}
_BRIDGE_CALL_NAMES = {
    "numpy_module",
    "numpy_callable",
    "hybrid_callable",
    "record_fallback",
    "wrap_from_numpy",
    "coerce_to_numpy",
}


@dataclass(frozen=True)
class _SurfaceSpec:
    name: str
    kind: str
    module_name: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _function_contains_fallback(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    found = False

    class Finder(ast.NodeVisitor):
        def visit_ClassDef(self, _node: ast.ClassDef) -> None:
            return

        def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal found
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FALLBACK_CALL_NAMES:
                found = True
                return
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "bridge"
                and fn.attr in _BRIDGE_CALL_NAMES
            ):
                found = True
                return
            self.generic_visit(node)

    finder = Finder()
    for stmt in func_node.body:
        if found:
            break
        finder.visit(stmt)
    return found


def _scan_explicit_fallback_functions(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    explicit_fallbacks: set[str] = set()

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            qual = ".".join([*self.class_stack, node.name]) if self.class_stack else node.name
            if _function_contains_fallback(node):
                explicit_fallbacks.add(qual)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            for child in node.body:
                self.visit(child)
            self.class_stack.pop()

    Collector().visit(tree)
    return explicit_fallbacks


def _defining_callable(obj: Any) -> Any:
    raw = getattr(obj, "__func__", obj)
    seen_ids: set[int] = set()
    while True:
        if getattr(raw, "__mumpy_bridge_kind__", None) in {"numpy", "hybrid"}:
            return raw
        wrapped = getattr(raw, "__wrapped__", None)
        if wrapped is None:
            return raw
        wrapped = getattr(wrapped, "__func__", wrapped)
        wrapped_id = id(wrapped)
        if wrapped_id in seen_ids:
            return raw
        seen_ids.add(wrapped_id)
        raw = wrapped


def _public_names(surface_obj: Any, surface_kind: str) -> list[str]:
    if surface_kind != "module":
        return sorted({n for n in dir(surface_obj) if not n.startswith("_")})

    # For module surfaces, report the broader compatibility surface (including dynamic
    # names) even when __all__ is intentionally curated in package __init__.py files.
    names: set[str] = set(getattr(surface_obj, "__all__", [])) or set(dir(surface_obj))

    impl_mod = getattr(surface_obj, "_impl", None)
    if impl_mod is not None:
        try:
            names.update(n for n in impl_mod.__dir__() if not n.startswith("_"))
        except Exception:  # noqa: BLE001
            names.update(n for n in dir(impl_mod) if not n.startswith("_"))

    core_mod = getattr(surface_obj, "_core_module", None)
    if core_mod is not None:
        try:
            names.update(n for n in core_mod.__dir__() if not n.startswith("_"))
        except Exception:  # noqa: BLE001
            names.update(n for n in dir(core_mod) if not n.startswith("_"))

    return sorted(n for n in names if not n.startswith("_"))


def _load_surface_object(mp_mod: Any, spec: _SurfaceSpec) -> Any:
    if spec.kind == "module":
        return importlib.import_module(spec.module_name)
    if spec.name == "mumpy.random.Generator":
        return mp_mod.random.default_rng(0)
    msg = f"unsupported surface {spec}"
    raise ValueError(msg)


def _source_path_for_module(module_name: str) -> Path | None:
    mod = importlib.import_module(module_name)
    path = getattr(mod, "__file__", None)
    if not path:
        return None
    p = Path(path).resolve()
    root = _repo_root().resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p


def _classify_name(name: str, value: Any, fallback_qualnames_by_source: dict[Path, set[str]]) -> str:
    if not callable(value):
        return _CAT_PASSTHROUGH
    raw = _defining_callable(value)
    if getattr(raw, "__mumpy_bridge_kind__", None) in {"numpy", "hybrid"}:
        return _CAT_DYNAMIC

    module_name = getattr(raw, "__module__", None)
    qualname = getattr(raw, "__qualname__", name)
    if isinstance(module_name, str):
        source_path = _source_path_for_module(module_name)
        if source_path is not None:
            fallback_names = fallback_qualnames_by_source.get(source_path, set())
            if qualname in fallback_names:
                return _CAT_EXPLICIT_FALLBACK
    return _CAT_NATIVE


def _build_coverage_data() -> dict[str, list[tuple[str, str]]]:
    mp = importlib.import_module("mumpy")

    specs = [
        _SurfaceSpec("mumpy", "module", "mumpy"),
        _SurfaceSpec("mumpy.linalg", "module", "mumpy.linalg"),
        _SurfaceSpec("mumpy.fft", "module", "mumpy.fft"),
        _SurfaceSpec("mumpy.random", "module", "mumpy.random"),
        _SurfaceSpec("mumpy.random.Generator", "object", "mumpy.random"),
    ]

    fallback_scan_modules = {
        "mumpy._core",
        "mumpy.linalg._linalg",
        "mumpy.fft._fft",
        "mumpy.random._random",
    }
    fallback_qualnames_by_source: dict[Path, set[str]] = {}
    for module_name in fallback_scan_modules:
        source_path = _source_path_for_module(module_name)
        if source_path is None:
            continue
        fallback_qualnames_by_source[source_path] = _scan_explicit_fallback_functions(source_path)

    coverage: dict[str, list[tuple[str, str]]] = {}
    for spec in specs:
        surface_obj = _load_surface_object(mp, spec)
        rows: list[tuple[str, str]] = []
        for name in _public_names(surface_obj, spec.kind):
            try:
                value = getattr(surface_obj, name)
            except Exception:  # noqa: BLE001
                value = _MISSING_ATTR
            if value is _MISSING_ATTR:
                continue
            category = _classify_name(name, value, fallback_qualnames_by_source)
            rows.append((name, category))
        coverage[spec.name] = rows
    return coverage


def _format_markdown_table(headers: list[str], rows: list[list[str]], right_align_cols: set[int]) -> list[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i in right_align_cols:
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    def fmt_sep() -> str:
        parts = []
        for i, width in enumerate(widths):
            width_value = max(width, 3)
            if i in right_align_cols:
                parts.append("-" * max(width_value - 1, 2) + ":")
            else:
                parts.append("-" * width_value)
        return "| " + " | ".join(parts) + " |"

    return [fmt_row(headers), fmt_sep(), *[fmt_row(row) for row in rows]]


def _generate_coverage_section() -> str:
    coverage = _build_coverage_data()
    lines: list[str] = []
    lines.append("Category meanings:")
    lines.append("")
    lines.append(
        "- Native explicit: Explicitly implemented by MumPy without direct NumPy bridge calls in the function body.",
    )
    lines.append(
        "- Explicit (may fallback): Explicitly implemented by MumPy and includes one or more NumPy fallback paths.",
    )
    lines.append(
        (
            "- Dynamic NumPy bridge: Provided dynamically via module `__getattr__` "
            "using the NumPy bridge (compatibility surface)."
        ),
    )
    lines.append(
        "- Non-callable passthrough: Public non-callable objects (constants/types/helpers) surfaced on the module.",
    )
    lines.append("")
    table_headers = [
        "Surface",
        "Native explicit",
        "Explicit (may fallback)",
        "Dynamic NumPy bridge",
        "Non-callable passthrough",
    ]

    per_surface_counters: dict[str, Counter[str]] = {}
    per_surface_names: dict[str, dict[str, list[str]]] = {}
    table_rows: list[list[str]] = []
    for surface, rows in coverage.items():
        counter = Counter(category for _, category in rows)
        per_surface_counters[surface] = counter
        per_surface_names[surface] = {cat: sorted([name for name, c in rows if c == cat]) for cat in _CATEGORIES}
        table_rows.append(
            [
                f"`{surface}`",
                str(counter[_CAT_NATIVE]),
                str(counter[_CAT_EXPLICIT_FALLBACK]),
                str(counter[_CAT_DYNAMIC]),
                str(counter[_CAT_PASSTHROUGH]),
            ],
        )
    lines.extend(_format_markdown_table(table_headers, table_rows, right_align_cols={1, 2, 3, 4}))
    lines.append("")

    detail_categories = [_CAT_EXPLICIT_FALLBACK, _CAT_DYNAMIC]
    for category in detail_categories:
        lines.append(f"## {category}")
        lines.append("")
        any_names = False
        for surface in coverage:
            names = per_surface_names[surface][category]
            if not names:
                continue
            any_names = True
            lines.append(f"- `{surface}`")
            lines.extend(f"  - `{name}`" for name in names)
        if not any_names:
            lines.append("_None_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _generate_coverage_document() -> str:
    lines = [
        "# Coverage",
        "",
        "This file is generated by `scripts/generate_api_coverage.py`.",
        "",
    ]
    return "\n".join(lines) + "\n" + _generate_coverage_section()


def _main() -> int:
    sys.stdout.write(_generate_coverage_document())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
