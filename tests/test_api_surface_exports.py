"""API surface export and star-import behavior tests."""

import numpy as np

import mumpy as mp

from .conftest import assert_array_equal


def test_init_all_is_curated_tuple_and_members_exist():
    modules = [
        ("mumpy", mp),
        ("mumpy.linalg", mp.linalg),
        ("mumpy.fft", mp.fft),
        ("mumpy.random", mp.random),
        ("mumpy.fallbacks", mp.fallbacks),
        ("mumpy.testing", mp.testing),
    ]
    for label, mod in modules:
        exported = getattr(mod, "__all__", None)
        assert isinstance(exported, tuple), f"{label}.__all__ should be a tuple"
        assert len(exported) == len(set(exported)), f"{label}.__all__ should not contain duplicates"
        for name in exported:
            if name != "__version__":
                assert not name.startswith("_"), f"{label} exports private name {name!r}"
            assert hasattr(mod, name), f"{label}.__all__ contains missing name {name!r}"


def test_dynamic_fallback_names_are_accessible_but_not_exported_from_modules():
    assert callable(mp.array2string)
    assert "array2string" not in mp.__all__
    assert "array2string" not in dir(mp)
    assert mp.array2string(mp.array([1, 2, 3])) == np.array2string(np.array([1, 2, 3]))

    assert callable(mp.linalg.tensorinv)
    assert "tensorinv" not in mp.linalg.__all__
    assert "tensorinv" not in dir(mp.linalg)

    assert callable(mp.fft.hfft)
    assert "hfft" not in mp.fft.__all__
    assert "hfft" not in dir(mp.fft)

    assert callable(mp.random.logseries)
    assert "logseries" not in mp.random.__all__
    assert "logseries" not in dir(mp.random)

    g = mp.random.default_rng(0)
    for name in ("beta", "logseries", "gamma", "multinomial"):
        assert name in dir(g)
        assert callable(getattr(g, name))


def test_star_import_uses_curated_all_only():
    top_ns: dict[str, object] = {}
    exec("from mumpy import *", top_ns, top_ns)
    assert "array" in top_ns
    assert "acos" not in top_ns
    assert "array2string" not in top_ns

    linalg_ns: dict[str, object] = {}
    exec("from mumpy.linalg import *", linalg_ns, linalg_ns)
    assert "solve" in linalg_ns
    assert "tensorinv" not in linalg_ns

    fft_ns: dict[str, object] = {}
    exec("from mumpy.fft import *", fft_ns, fft_ns)
    assert "fft" in fft_ns
    assert "hfft" not in fft_ns

    random_ns: dict[str, object] = {}
    exec("from mumpy.random import *", random_ns, random_ns)
    assert "chisquare" in random_ns
    assert "logseries" not in random_ns


def test_star_import_does_not_shadow_builtins_used_by_core_helpers():
    ns: dict[str, object] = {}
    exec("from mumpy import *", ns, ns)

    mp_parts = mp.array_split(mp.arange(10), 3)
    np_parts = np.array_split(np.arange(10), 3)
    assert [p.tolist() for p in mp_parts] == [p.tolist() for p in np_parts]

    assert_array_equal(mp.diagflat([1, 2], k=-1), np.diagflat(np.array([1, 2]), k=-1))
