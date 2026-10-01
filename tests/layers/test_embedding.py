"""W8A16-in for the input embedding (vocab-parallel table -> per-row fp8-e4m3).

The input embedding is a gather, not a GEMM: its forward reads one row per token, so the
fp8 win is NOT a faster decode read (a row is ~2 KiB) but the half-the-table VRAM that the
dropped bf16 master frees into the expert slot cache. The quantizer is the same per-"row"
e4m3 the linear W8A16 path uses -- for a [vocab, hidden] table the row axis is the hidden
dim, exactly matching its per-output-channel semantics.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from freetoken.distributed import DistributedInfo, info
from freetoken.layers.embedding import VocabParallelEmbedding


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    saved = info._TP_INFO
    info._TP_INFO = DistributedInfo(0, 1)
    yield
    info._TP_INFO = saved


def _cuda_embed(vocab: int, dim: int) -> VocabParallelEmbedding:
    with _rank(0, 1):
        embed = VocabParallelEmbedding(vocab, dim)
    embed.weight = (torch.randn(vocab, dim) * 0.02).to(torch.bfloat16).cuda()
    return embed


@contextlib.contextmanager
def _rank(rank: int, world: int):
    saved = info._TP_INFO
    info._TP_INFO = DistributedInfo(rank, world)
    try:
        yield
    finally:
        info._TP_INFO = saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A16 embed needs a GPU")
def test_embed_w8a16_finalize_replaces_weight_and_matches_reference(monkeypatch):
    torch.manual_seed(7)
    vocab, dim = 1000, 256
    embed = _cuda_embed(vocab, dim)
    ref = embed.weight.float().clone()

    monkeypatch.setattr(embed, "w8a16_embed_ok", True)
    embed.finalize()

    assert embed.weight is None, "the bf16 master must be dropped (the VRAM is the win)"
    assert embed._w8a16_weight.shape == (vocab, dim)
    assert embed._w8a16_weight.dtype == torch.float8_e4m3fn
    assert embed._w8a16_scale.shape == (vocab,)

    # per-row e4m3 weight error (~2.7% relative) sets the bound; a gather is exact, so
    # the only error source is the weight quantization itself.
    for bs in (1, 4, 16, 32):  # decode/prefill batch sizes, incl. the 16-way target
        idx = torch.randint(0, vocab, (bs,), device="cuda")
        y = embed.forward(idx)
        exp = ref[idx].to(torch.bfloat16)
        rel = ((y.float() - exp.float()).norm() / exp.float().norm()).item()
        assert rel < 0.04, f"bs={bs}: rel err {rel}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A16 embed needs a GPU")
def test_embed_w8a16_disabled_keeps_bf16():
    torch.manual_seed(11)
    embed = _cuda_embed(512, 128)
    w = embed.weight.clone()
    embed.w8a16_embed_ok = False  # the --no-embed-fp8 knob landed on the module default
    embed.finalize()
    assert embed.weight is not None and embed._w8a16_weight is None
    idx = torch.randint(0, 512, (8,), device="cuda")
    assert torch.equal(embed.forward(idx), w[idx])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A16 embed needs a GPU")
def test_embed_w8a16_forward_matches_reference_directly():
    """The forward() fp8 branch (gather rows + gather scales + dequant) must track the
    bf16 reference gather within per-row e4m3 error across the forward's own path."""
    torch.manual_seed(17)
    vocab, dim = 2000, 128
    embed = _cuda_embed(vocab, dim)
    ref = embed.weight.float().clone()
    embed.w8a16_embed_ok = True
    embed.finalize()

    idx = torch.randint(0, vocab, (32,), device="cuda")
    y = embed.forward(idx)
    exp = ref[idx].to(torch.bfloat16)
    rel = ((y.float() - exp.float()).norm() / exp.float().norm()).item()
    assert rel < 0.04, f"rel err {rel}"