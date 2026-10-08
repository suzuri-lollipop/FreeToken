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


def test_w8a16_candidate_gates():
    """W8A16 opt-in: the projection classes take it, the size floor keeps tiny GEMVs
    bf16, and routers never quantize (their logits pick the expert set)."""
    from freetoken.layers import LinearColParallelMerged, LinearOProj, LinearReplicated
    from freetoken.layers.quantization.linear.unquantized import w8a16_decode_candidate

    with _rank(0, 1):
        col = LinearColParallelMerged(IN, [OUT, OUT], has_bias=False)
        row = LinearRowParallel(IN, OUT, has_bias=False)
        oproj = LinearOProj(IN, OUT, has_bias=False)
        rep = LinearReplicated(IN, OUT, has_bias=False)
        gate = LinearReplicated(IN, OUT, has_bias=False, prefix="model.layers.3.mlp.gate")

    # tiny weights sit under the default size floor for every class
    for layer in (col, row, oproj, rep, gate):
        assert not w8a16_decode_candidate(layer)
    # with the floor lifted, the opted-in classes are candidates...
    for layer in (col, row, oproj, rep):
        assert w8a16_decode_candidate(layer, min_elements=1)
    # ...but a router stays full precision at any size
    assert not w8a16_decode_candidate(gate, min_elements=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A16 kernel needs a GPU")
def test_w8a16_finalize_replaces_weight_and_matches_reference(monkeypatch):
    """finalize() swaps the bf16 master for per-channel fp8 + scale; decode (M<=max)
    runs the triton kernel, prefill dequantizes on the fly, and both track the bf16
    reference within weight-only fp8 error."""
    from freetoken.layers.quantization.linear import unquantized as unq

    monkeypatch.setattr(unq, "_W8A16_ENABLED", True)
    monkeypatch.setattr(unq, "_W8A16_MIN_ELEMENTS", 1000)
    torch.manual_seed(5)
    K, N = 1024, 512
    with _rank(0, 1):
        layer = LinearRowParallel(K, N, has_bias=False)
    w_bf16 = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
    layer.weight = w_bf16.clone()
    layer.quant_method.finalize(layer)
    assert layer.weight is None, "the bf16 master must be dropped (its VRAM is the win)"
    assert layer._w8a16_weight.shape == (N, K)
    assert layer._w8a16_weight.dtype == torch.float8_e4m3fn
    assert layer._w8a16_scale.shape == (N,)

    # tolerance = inherent per-channel e4m3 weight error (dot-product relative error
    # ~= per-element relative error ~2.7%, K-independent); strictly tighter than the
    # W8A8 path production already accepts for in_proj/qkv (weight AND activation fp8),
    # and far inside this NVFP4-expert checkpoint's noise budget.
    ref = w_bf16.float()
    for M in (1, 4, 8, 16, 24, 32):  # decode kernel path (BLOCK_M 16 then 32)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        y = layer.forward(x)
        rel = ((y.float() - x.float() @ ref.T).norm() / (x.float() @ ref.T).norm()).item()
        assert rel < 0.04, f"M={M}: rel err {rel}"
    x = torch.randn(64, K, device="cuda", dtype=torch.bfloat16)  # prefill dequant path
    y = layer.forward(x)
    rel = ((y.float() - x.float() @ ref.T).norm() / (x.float() @ ref.T).norm()).item()
    assert rel < 0.04, f"prefill: rel err {rel}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A16 kernel needs a GPU")
def test_w8a16_disabled_keeps_bf16(monkeypatch):
    from freetoken.layers.quantization.linear import unquantized as unq

    monkeypatch.setattr(unq, "_W8A16_ENABLED", False)
    monkeypatch.setattr(unq, "_W8A16_MIN_ELEMENTS", 1)
    monkeypatch.setattr(unq, "_FP8_ENABLED", False)
    with _rank(0, 1):
        layer = LinearRowParallel(IN, OUT, has_bias=False)
    w = torch.randn(OUT, IN).to(torch.bfloat16).cuda()
    layer.weight = w.clone()
    layer.quant_method.finalize(layer)
    assert layer.weight is not None and getattr(layer, "_w8a16_weight", None) is None
    x = torch.randn(2, IN, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(layer.forward(x), F.linear(x, w))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A16 kernel needs a GPU")
def test_lm_head_w8a16_replaces_untied_head_only(monkeypatch):
    """The vocab-parallel lm_head opts into W8A16: it is the largest bf16 decode
    read per rank, and the freed VRAM goes to the expert slot cache. A tied head
    shares the input embedding's weight and must stay bf16."""
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.layers.quantization.linear.unquantized import w8a16_decode_candidate

    head = ParallelLMHead(4096, 1024, quant_config=NoQuantConfig(), prefix="lm_head")
    assert head.w8a16_decode_ok
    assert w8a16_decode_candidate(head)

    embed = VocabParallelEmbedding(4096, 1024)
    tied = ParallelLMHead(4096, 1024, tie_word_embeddings=True,
                          tied_embedding=embed, prefix="lm_head")
    assert not tied.w8a16_decode_ok

    monkeypatch.setenv("FREETOKEN_W8A16_LM_HEAD", "0")
    off = ParallelLMHead(4096, 1024, quant_config=NoQuantConfig(), prefix="lm_head")
    assert not off.w8a16_decode_ok

    # finalize swaps in the per-channel fp8 weight; the head's own projection still serves
    # every row count: the decode kernel (M=20 -> the BLOCK_M=32 path) and the prefill
    # dequant branch (M=64) the speculative head pass runs on. Both track the bf16
    # reference within weight-only fp8 error.
    torch.manual_seed(7)
    w_bf16 = (torch.randn(4096, 1024) * 0.02).to(torch.bfloat16).cuda()
    head.weight = w_bf16.clone()
    head.quant_method.finalize(head)
    assert head.weight is None
    assert head._w8a16_weight.dtype == torch.float8_e4m3fn
    for rows in (20, 64):
        x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
        y = head.shard_logits(x)  # through the head, which is what the spec path calls
        ref = x.float() @ w_bf16.float().T
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        assert rel < 0.04, f"M={rows}: rel err {rel}"
