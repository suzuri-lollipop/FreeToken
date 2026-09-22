"""Row-parallel linear layers under tensor parallelism (CPU, no process group).

A row-parallel layer keeps the output full width on every rank, so each rank's GEMM adds the
same bias and the all-reduce counts it ``tp_size`` times. The text towers never see this (their
``o_proj`` / ``down_proj`` carry no bias); the Qwen VL tower's projections all do.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F
from freetoken.distributed import DistributedInfo, info
from freetoken.layers import LinearRowParallel

IN, OUT = 64, 32
WHOLE = torch.randn(OUT, IN)
BIAS = torch.randn(OUT)
X = torch.randn(8, IN)


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    saved = info._TP_INFO
    info._TP_INFO = DistributedInfo(0, 1)
    yield
    info._TP_INFO = saved


@contextlib.contextmanager
def _rank(rank: int, world: int):
    """Build inside a rank's TP info so the layer declares its own local shapes, then restore."""
    saved = info._TP_INFO
    info._TP_INFO = DistributedInfo(rank, world)
    try:
        yield
    finally:
        info._TP_INFO = saved


class _PassThrough:
    """The test sums the partials itself, so the collective is a pass-through."""

    def all_reduce(self, x):
        return x


def _rank_layer(rank: int, world: int) -> LinearRowParallel:
    """This rank's layer over its slice of the input columns, holding the weight as loaded."""
    with _rank(rank, world):
        layer = LinearRowParallel(IN, OUT, has_bias=True)
    cols = slice(rank * (IN // world), (rank + 1) * (IN // world))
    with torch.no_grad():
        layer.weight.copy_(WHOLE[:, cols])
        layer.bias.copy_(BIAS)
    return layer


def test_a_row_parallel_bias_is_added_once_across_ranks():
    torch.manual_seed(0)
    whole = F.linear(X, WHOLE, BIAS)
    partials = [
        layer.quant_method.apply(layer, X[:, cols])
        for layer, cols in (
            (_rank_layer(0, 2), slice(0, IN // 2)),
            (_rank_layer(1, 2), slice(IN // 2, IN)),
        )
    ]
    summed = sum(partials)
    assert not torch.allclose(summed, whole, atol=1e-5), "the double-counted bias went unnoticed"
    first = _rank_layer(0, 2)
    first._comm = _PassThrough()
    assert torch.allclose(first._reduce(summed), whole, atol=1e-5, rtol=1e-5)


def test_a_single_rank_row_parallel_layer_is_untouched():
    torch.manual_seed(0)
    layer = _rank_layer(0, 1)
    assert torch.allclose(layer.forward(X), F.linear(X, WHOLE, BIAS), atol=1e-5, rtol=1e-5)


def test_fp8_decode_candidate_selects_only_tall_merged_columns():
    """fp8 decode GEMM opt-in: the measured wins (docs/qwen38_flash_next_optimizations.md)
    are exactly the tall merged column projections; row/o_proj/replicated shapes lose more to
    the per-step activation quantization than the GEMM saves, so they must stay bf16 no matter
    how large they are."""
    from freetoken.layers import LinearColParallelMerged, LinearOProj, LinearReplicated
    from freetoken.layers.quantization.linear.unquantized import fp8_decode_candidate

    with _rank(0, 1):
        col = LinearColParallelMerged(IN, [OUT, OUT], has_bias=False)
        row = LinearRowParallel(IN, OUT, has_bias=False)
        oproj = LinearOProj(IN, OUT, has_bias=False)
        rep = LinearReplicated(IN, OUT, has_bias=False)

    # tiny weights sit under the default size floor for every class
    assert not fp8_decode_candidate(col)
    # with the floor lifted, only the opted-in class is a candidate
    assert fp8_decode_candidate(col, min_elements=1)
    for layer in (row, oproj, rep):
        assert not fp8_decode_candidate(layer, min_elements=1)
