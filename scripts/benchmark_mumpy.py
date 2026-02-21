#!/usr/bin/env python
"""Benchmark runner for MumPy performance smoke checks."""

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _stdout_line(line: str = "") -> None:
    sys.stdout.write(f"{line}\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _BenchCase:
    name: str
    func: Callable[[], Any]
    notes: str = ""


def _collect_mx_arrays(value: Any, out: list[Any], mx_array_type: Any) -> None:
    if isinstance(value, mx_array_type):
        out.append(value)
        return
    mx_value = getattr(value, "mx", None)
    if isinstance(mx_value, mx_array_type):
        out.append(mx_value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_mx_arrays(item, out, mx_array_type)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_mx_arrays(item, out, mx_array_type)


def _force_eval(value: Any, *, mx_mod: Any) -> None:
    arrays: list[Any] = []
    _collect_mx_arrays(value, arrays, mx_mod.array)
    if arrays:
        mx_mod.eval(*arrays)


def _device_info(mx_mod: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        dev = mx_mod.default_device()
    except Exception:  # noqa: BLE001
        dev = None
    if dev is not None:
        info["default_device"] = str(dev)
        for attr in ("platform", "type"):
            if hasattr(dev, attr):
                with suppress(Exception):
                    info[f"default_device_{attr}"] = str(getattr(dev, attr))
    with suppress(Exception):
        info["gpu_available"] = bool(getattr(mx_mod, "gpu", None))
    return info


def _build_cases(mp: Any, np: Any) -> list[_BenchCase]:
    rng = np.random.default_rng(0)

    part_src = rng.integers(0, 100_000, size=50_000, dtype=np.int32)
    part_k = 12_345

    bc_x = rng.integers(0, 2048, size=50_000, dtype=np.int32)
    bc_w = rng.random(50_000, dtype=np.float32)

    hist_x = (rng.standard_normal(80_000).astype(np.float32) * 2.0) + 1.0
    hist_w = rng.random(80_000, dtype=np.float32)

    h2_x = rng.random(50_000, dtype=np.float32) * 3.0
    h2_y = rng.random(50_000, dtype=np.float32) * 4.0

    hdd = np.stack(
        [
            rng.random(30_000, dtype=np.float32) * 3.0,
            rng.random(30_000, dtype=np.float32) * 4.0,
            rng.random(30_000, dtype=np.float32) * 5.0,
        ],
        axis=1,
    )

    interp_x = (rng.random(80_000, dtype=np.float32) * 20.0) - 5.0
    xp = np.linspace(0.0, 10.0, 2048, dtype=np.float32)
    fp = np.sin(xp).astype(np.float32, copy=False)

    pad_src = rng.standard_normal((192, 224)).astype(np.float32)
    pad_width = ((16, 20), (12, 10))

    td_a = rng.standard_normal((8, 16, 32)).astype(np.float32)
    td_b = rng.standard_normal((32, 24, 16)).astype(np.float32)
    td_c = rng.standard_normal((8, 32)).astype(np.float32)
    td_d = rng.standard_normal((32, 64)).astype(np.float32)

    return [
        _BenchCase(
            "partition",
            lambda: mp.partition(part_src, part_k),
            notes="MLX path when supported; NumPy fallback otherwise",
        ),
        _BenchCase(
            "argpartition",
            lambda: mp.argpartition(part_src, part_k),
            notes="MLX path when supported; NumPy fallback otherwise",
        ),
        _BenchCase("bincount", lambda: mp.bincount(bc_x, weights=bc_w)),
        _BenchCase("histogram", lambda: mp.histogram(hist_x, bins=128, range=(-5.0, 7.0), weights=hist_w)),
        _BenchCase("histogram2d", lambda: mp.histogram2d(h2_x, h2_y, bins=(48, 64), range=((0.0, 3.0), (0.0, 4.0)))),
        _BenchCase(
            "histogramdd",
            lambda: mp.histogramdd(hdd, bins=(12, 16, 20), range=((0.0, 3.0), (0.0, 4.0), (0.0, 5.0))),
        ),
        _BenchCase("interp_periodic", lambda: mp.interp(interp_x, xp, fp, period=10.0)),
        _BenchCase("pad_reflect", lambda: mp.pad(pad_src, pad_width, mode="reflect")),
        _BenchCase("pad_symmetric", lambda: mp.pad(pad_src, pad_width, mode="symmetric")),
        _BenchCase("tensordot_axes_int", lambda: mp.tensordot(td_c, td_d, axes=1)),
        _BenchCase("tensordot_axes_tuple", lambda: mp.tensordot(td_a, td_b, axes=([2, 1], [0, 2]))),
    ]


def _time_case(case: _BenchCase, *, iterations: int, warmup: int, mp: Any, mx_mod: Any) -> dict[str, Any]:
    for _ in range(max(warmup, 0)):
        _force_eval(case.func(), mx_mod=mx_mod)

    timings: list[float] = []
    with mp.fallbacks.capture_counts(reset=True) as cap:
        for _ in range(iterations):
            t0 = time.perf_counter()
            value = case.func()
            _force_eval(value, mx_mod=mx_mod)
            timings.append(time.perf_counter() - t0)

    mean_s = statistics.fmean(timings) if timings else 0.0
    median_s = statistics.median(timings) if timings else 0.0
    min_s = min(timings) if timings else 0.0
    max_s = max(timings) if timings else 0.0

    return {
        "name": case.name,
        "iterations": iterations,
        "warmup": warmup,
        "timings_s": timings,
        "mean_s": mean_s,
        "median_s": median_s,
        "min_s": min_s,
        "max_s": max_s,
        "fallback_delta": cap.delta,
        "notes": case.notes,
    }


def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _repo_root() / ".benchmarks" / f"mumpy_bench_{ts}.json"


def _run_benchmarks(*, iterations: int, warmup: int) -> dict[str, Any]:
    import mlx  # pyright: ignore[reportMissingImports]
    import mlx.core as mx  # pyright: ignore[reportMissingImports]
    import numpy as np

    import mumpy as mp

    cases = _build_cases(mp, np)
    rows = [_time_case(case, iterations=iterations, warmup=warmup, mp=mp, mx_mod=mx) for case in cases]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "iterations": iterations,
            "warmup": warmup,
            "case_count": len(rows),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mumpy_version": getattr(mp, "__version__", None),
            "numpy_version": getattr(np, "__version__", None),
            "mlx_version": getattr(mlx, "__version__", None),
            **_device_info(mx),
        },
        "cases": rows,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MumPy micro-benchmarks and emit a JSON report.")
    parser.add_argument("--iterations", type=int, default=12, help="Timed iterations per case (default: 12)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per case (default: 2)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: .benchmarks/mumpy_bench_<timestamp>.json)",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")
    args = parser.parse_args(argv)

    if args.iterations <= 0:
        parser.error("--iterations must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    report = _run_benchmarks(iterations=args.iterations, warmup=args.warmup)
    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=args.indent, sort_keys=True) + "\n")

    _stdout_line(f"Wrote benchmark report: {output_path}")
    for row in report["cases"]:
        fallback_total = int(sum(int(v) for v in row.get("fallback_delta", {}).values()))
        _stdout_line(
            (
                f"{row['name']:>20s}  mean={row['mean_s']:.6f}s  "
                f"median={row['median_s']:.6f}s  fallback_hits={fallback_total}"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
