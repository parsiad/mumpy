"""Tests for export-boundary packaging conventions."""

from pathlib import Path

import mumpy as mp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_MUMPY = REPO_ROOT / "src" / "mumpy"
PUBLIC_SUBMODULES = ("fft", "linalg", "random", "fallbacks", "testing")


def test_public_submodules_are_packages_with_init_and_no_flat_files():
    for name in PUBLIC_SUBMODULES:
        pkg_dir = SRC_MUMPY / name
        flat_file = SRC_MUMPY / f"{name}.py"
        assert pkg_dir.is_dir(), f"expected package directory: {pkg_dir}"
        assert (pkg_dir / "__init__.py").is_file(), f"missing __init__.py: {pkg_dir / '__init__.py'}"
        assert not flat_file.exists(), f"unexpected flat public module remains: {flat_file}"


def test_top_level_python_files_are_init_or_private():
    disallowed = []
    for path in SRC_MUMPY.glob("*.py"):
        if path.name == "__init__.py":
            continue
        if path.name.startswith("_"):
            continue
        disallowed.append(path.name)
    assert disallowed == [], f"unexpected top-level public .py files: {sorted(disallowed)}"


def test_public_submodule_files_are_package_inits():
    modules = [mp.fft, mp.linalg, mp.random, mp.fallbacks, mp.testing]
    for mod in modules:
        path = getattr(mod, "__file__", None)
        assert isinstance(path, str), f"{mod.__name__} missing __file__: {path!r}"
        assert path.endswith("__init__.py"), f"{mod.__name__} not package-backed: {path!r}"


def test_package_split_preserves_dynamic_getattr_but_not_curated_star_exports():
    assert callable(mp.linalg.tensorinv)
    assert callable(mp.fft.hfft)
    assert callable(mp.random.logseries)

    assert "tensorinv" not in mp.linalg.__all__
    assert "hfft" not in mp.fft.__all__
    assert "logseries" not in mp.random.__all__

    assert "tensorinv" not in dir(mp.linalg)
    assert "hfft" not in dir(mp.fft)
    assert "logseries" not in dir(mp.random)

    fft_ns: dict[str, object] = {}
    exec("from mumpy.fft import *", fft_ns, fft_ns)
    assert "fft" in fft_ns
    assert "hfft" not in fft_ns

    linalg_ns: dict[str, object] = {}
    exec("from mumpy.linalg import *", linalg_ns, linalg_ns)
    assert "solve" in linalg_ns
    assert "tensorinv" not in linalg_ns

    random_ns: dict[str, object] = {}
    exec("from mumpy.random import *", random_ns, random_ns)
    assert "chisquare" in random_ns
    assert "logseries" not in random_ns
