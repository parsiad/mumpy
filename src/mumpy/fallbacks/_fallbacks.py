from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .. import _bridge as bridge


@dataclass
class FallbackCountCapture:
    before: dict[str, int]
    after: dict[str, int] = field(default_factory=dict)
    delta: dict[str, int] = field(default_factory=dict)


def get_counts() -> dict[str, int]:
    return bridge.get_recorded_fallback_counts()


def reset_counts() -> None:
    bridge.reset_recorded_fallback_counts()


@contextmanager
def capture_counts(reset: bool = False) -> Iterator[FallbackCountCapture]:
    if reset:
        bridge.reset_recorded_fallback_counts()
    before = bridge.get_recorded_fallback_counts()
    capture = FallbackCountCapture(before=before)
    try:
        yield capture
    finally:
        after = bridge.get_recorded_fallback_counts()
        capture.after = after
        keys = set(before) | set(after)
        capture.delta = {k: diff for k in sorted(keys) if (diff := after.get(k, 0) - before.get(k, 0)) > 0}


__all__ = ["FallbackCountCapture", "capture_counts", "get_counts", "reset_counts"]
