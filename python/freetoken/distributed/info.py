from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


@contextlib.contextmanager
def tp_view(rank: int, size: int):
    """Temporarily adopt one rank's TP view in a single-process tool.

    set_tp_info is the engine's one-shot init; a tool that inspects every rank's
    geometry (ft bench bw) scopes the view instead, restoring the previous one.
    """
    global _TP_INFO
    prev = _TP_INFO
    _TP_INFO = DistributedInfo(rank, size)
    try:
        yield
    finally:
        _TP_INFO = prev


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info", "tp_view"]
