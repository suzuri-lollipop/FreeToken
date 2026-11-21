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


_PP_INFO: DistributedInfo | None = None


def set_pp_info(rank: int, size: int) -> None:
    global _PP_INFO
    if _PP_INFO is not None:
        raise RuntimeError("PP info has been set")
    _PP_INFO = DistributedInfo(rank, size)


def get_pp_info() -> DistributedInfo:
    if _PP_INFO is None:
        raise RuntimeError("PP info has not been set")
    return _PP_INFO


def try_get_pp_info() -> DistributedInfo | None:
    return _PP_INFO


def pp_layer_range(num_layers: int) -> tuple[int, int]:
    """This PP rank's [start, end) layer slice; the full range when PP is off.
    Contiguous split, remainder spread over the leading ranks (Maya's GPU0:0-19 /
    GPU1:20-41 for 42 layers / 2 ranks)."""
    pp = _PP_INFO
    if pp is None or pp.size <= 1:
        return (0, num_layers)
    base, rem = divmod(num_layers, pp.size)
    start = pp.rank * base + min(pp.rank, rem)
    end = start + base + (1 if pp.rank < rem else 0)
    return (start, end)


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


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info", "tp_view",
           "set_pp_info", "get_pp_info", "try_get_pp_info", "pp_layer_range"]
