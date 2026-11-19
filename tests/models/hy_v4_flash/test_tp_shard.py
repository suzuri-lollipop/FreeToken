"""The TP reader cuts: head-sharded attention, intermediate-sharded FFNs, vocab
ceil-sharding, and replication of everything the ranks must agree on (latent
projections, router, indexer, engram, iHC). Simulates both ranks without a
process group; the end-to-end TP parity is the server gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import requires_cuda, write_tiny_checkpoint
from .test_reference_parity import REAL_DIMS

pytestmark = requires_cuda


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    from freetoken.models.hy_v4_flash.config import parse_config
    from freetoken.utils.hf import cached_load_hf_config
    from safetensors.torch import load_file

    folder = tmp_path_factory.mktemp("hy4-tp-shard")
    write_tiny_checkpoint(str(folder), seed=5, indexer_weight_scale=1.0, **REAL_DIMS)
    tensors = load_file(str(folder) + "/model.safetensors")
    config = parse_config(cached_load_hf_config(str(folder)))
    return tensors, config


def _cut(name, tensor, config, rank, world, monkeypatch):
    import freetoken.distributed as dist
    from freetoken.models.hy_v4_flash.weight import _tp_cut

    monkeypatch.setattr(dist, "get_tp_info", lambda: SimpleNamespace(rank=rank, size=world))
    return _tp_cut(name, tensor, config)


@pytest.mark.parametrize("rank", [0, 1])
def test_head_sharded_attention(fixture, rank, monkeypatch):
    tensors, config = fixture
    args = config.hy4_args
    world = 2
    for L in (1, 2):  # one indexed (32 heads), one SWA (64 heads)
        heads = args.heads_of(L)
        local = heads // world
        p = f"model.layers.{L}.self_attn"
        cases = [
            (f"{p}.q_b_proj.weight", 0, heads * args.qk_head_dim),
            (f"{p}.kv_b_proj.weight", 0, heads * (args.qk_nope_head_dim + args.v_head_dim)),
            (f"{p}.linear_gate.weight", 0, heads * args.v_head_dim),
            (f"{p}.o_proj.weight", 1, heads * args.v_head_dim),
            (f"{p}.learnable_sink_param", 0, heads),
        ]
        for name, axis, units in cases:
            t = tensors[name]
            got = _cut(name, t, config, rank, world, monkeypatch)
            want = t.narrow(axis, rank * (units // world), units // world)
            assert torch.equal(got, want), name
        # the block scale narrows by its own 128 ratio
        name = f"{p}.q_b_proj.weight_scale_inv"
        if name in tensors:
            t = tensors[name]
            got = _cut(name, t, config, rank, world, monkeypatch)
            rows = heads * args.qk_head_dim // 128
            want = t.narrow(0, rank * (rows // world), rows // world)
            assert torch.equal(got, want), name


@pytest.mark.parametrize("rank", [0, 1])
def test_ffn_and_vocab_shards(fixture, rank, monkeypatch):
    tensors, config = fixture
    args = config.hy4_args
    world = 2
    inter = args.intermediate_size
    shared = args.moe_inter_dim * args.n_shared_experts
    cases = [
        ("model.layers.0.mlp.gate_proj.weight", 0, inter),
        ("model.layers.0.mlp.up_proj.weight", 0, inter),
        ("model.layers.0.mlp.down_proj.weight", 1, inter),
        ("model.layers.1.mlp.shared_experts.gate_proj.weight", 0, shared),
        ("model.layers.1.mlp.shared_experts.up_proj.weight", 0, shared),
        ("model.layers.1.mlp.shared_experts.down_proj.weight", 1, shared),
        ("lm_head.weight", 0, args.vocab_size),
        ("model.embed_tokens.weight", 0, args.vocab_size),
    ]
    for name, axis, units in cases:
        t = tensors[name]
        got = _cut(name, t, config, rank, world, monkeypatch)
        want = t.narrow(axis, rank * (units // world), units // world)
        assert torch.equal(got, want), name


def test_replicated_modules(fixture, monkeypatch):
    tensors, config = fixture
    replicated = [
        "model.layers.1.self_attn.q_a_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.1.self_attn.kv_b_proj.weight_scale_inv" if "model.layers.1.self_attn.kv_b_proj.weight_scale_inv" in tensors else "model.layers.1.self_attn.q_a_layernorm.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.routed_expert_down_proj.weight",
        "model.layers.1.mlp.routed_expert_up_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.self_attn.compressed_indexer.query_projection.weight",
        "model.layers.1.engram_layer.value_proj.weight",
        "model.layers.1.hc_attn_layer.hc_pre.hc_fn",
        "model.hc_head.hc_head_fn",
        "model.norm.weight",
    ]
    for name in replicated:
        t = tensors[name]
        for rank in (0, 1):
            got = _cut(name, t, config, rank, 2, monkeypatch)
            assert got is t or torch.equal(got, t), name
