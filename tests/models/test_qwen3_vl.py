"""Qwen VL on the CPU side: parse_config, the engine's encoder decision, the registry invariants, the text-side image scatter, the tower's rank shards (no checkpoint)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo, info
from freetoken.engine.config import EngineConfig
from freetoken.mm import MM_PAD_SHIFT_VALUE
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl import deepstack_add, parse_config
from freetoken.models.qwen3_vl.config import parse_vision_config
from freetoken.models.qwen3_vl.vision import (
    Qwen3VLVisionModel,
    _TP_LEAF_AXIS,
    vision_module_leaf,
    vision_tp_shard,
)

ROPE = {"rope_theta": 5_000_000, "rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True}


def _hf_config(num_experts=0, arch="Qwen3VLForConditionalGeneration"):
    text = SimpleNamespace(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, rms_norm_eps=1e-6, hidden_act="silu", vocab_size=1000,
        max_position_embeddings=4096, rope_parameters=ROPE, rope_scaling=ROPE, model_type="qwen3_vl_text",
        num_experts=num_experts, num_experts_per_tok=2, moe_intermediate_size=32, norm_topk_prob=True,
    )
    vision = SimpleNamespace(
        hidden_size=32, depth=3, num_heads=4, intermediate_size=64, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=64, in_channels=3,
        deepstack_visual_indexes=[0, 1],
    )
    return SimpleNamespace(
        text_config=text, vision_config=vision, image_token_id=151655, tie_word_embeddings=False,
        model_type="qwen3_vl", architectures=[arch],
    )


def _engine_config(monkeypatch, hf, processor, **overrides):
    import freetoken.engine.config as engine_config
    import freetoken.mm.processor as mm_processor

    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: hf)
    monkeypatch.setattr(engine_config, "checkpoint_quant_config", lambda *args: None)
    monkeypatch.setattr(mm_processor, "get_mm_processor", lambda path, mm=None: processor)
    return EngineConfig(
        model_path="/fake", tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16, **overrides
    )


def test_vision_carries_mrope_and_deepstack():
    c = parse_config(_hf_config())
    assert c.is_multimodal and c.model_is_mrope
    assert c.rotary_config.mrope_section == [24, 20, 20] and c.rotary_config.mrope_layout == "interleaved"
    assert c.rotary_config.base == 5_000_000 and c.rotary_config.scaling is None
    assert c.vision_config.deepstack_visual_indexes == (0, 1) and c.image_token_id == 151655
    assert c.num_layers == 4 and c.num_experts == 0 and c.architectures == ["Qwen3VLForConditionalGeneration"]
    assert [g.attn_type.name for g in c.kv_cache_group_specs()] == ["FULL"]


def test_no_vision_section_is_a_plain_qwen3():
    hf = _hf_config()
    hf.vision_config = None
    c = parse_config(hf)
    assert not c.is_multimodal and not c.model_is_mrope and c.rotary_config.mrope_section is None


def test_engine_builds_the_tower_for_a_registered_family(monkeypatch):
    c = _engine_config(monkeypatch, _hf_config(), object())
    assert [e.kind for e in c.active_encoders] == ["vision"] and c.served_modalities == {"image"}
    assert c.model_config.is_multimodal and c.model_config.model_is_mrope


def test_text_model_only_hands_the_parser_a_config_without_vision(monkeypatch):
    hf = _hf_config()
    c = _engine_config(monkeypatch, hf, object(), mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)))
    assert not c.served_modalities
    assert not c.model_config.is_multimodal and not c.model_config.model_is_mrope
    assert hf.vision_config is not None  # the cached checkpoint config is left alone


@pytest.mark.parametrize("kinds, served", [({"vision"}, set()), ({"audio"}, {"image"})])
def test_mm_disable_drops_only_the_named_tower(monkeypatch, kinds, served):
    c = _engine_config(monkeypatch, _hf_config(), object(), mm=MultimodalConfig(disabled_encoders=frozenset(kinds)))
    assert c.served_modalities == served and not c.mm.text_model_only
    assert c.model_config.is_multimodal == bool(served) and c.model_config.model_is_mrope == bool(served)


def test_registered_encoders_come_with_a_processor_and_the_model_hooks():
    from freetoken.models.blocks import SupportsMultimodal
    from freetoken.models.register import _MODEL_REGISTRY, _load_attr

    for arch, spec in _MODEL_REGISTRY.items():
        assert (spec.mm_processor is None) == (not spec.encoders), arch
        for e in spec.encoders:
            assert e.kind in ENCODER_KINDS and e.modalities, arch
        if spec.encoders:
            assert issubclass(_load_attr(spec.module, spec.model_cls), SupportsMultimodal), arch


def test_a_family_without_registered_encoders_is_served_text_only(monkeypatch):
    import freetoken.engine.config as engine_config
    from dataclasses import replace
    from freetoken.models.register import get_model_spec

    monkeypatch.setattr(engine_config, "get_model_spec", lambda arch: replace(get_model_spec(arch), mm_processor=None, encoders=()))
    c = _engine_config(monkeypatch, _hf_config(), None)
    assert not c.active_encoders and not c.served_modalities and not c.model_config.is_multimodal


def test_tower_roots_make_the_fp8_ignore_list_match():
    from freetoken.layers.quantization import NameMap, QuantConfig
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("Qwen3VLForConditionalGeneration")
    # quantized Qwen VL exports list every tower module under its stored name and keep it bf16
    hf = SimpleNamespace(
        architectures=["Qwen3VLForConditionalGeneration"],
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "lm_head",
                "model.visual.blocks.0.attn.qkv",
                "model.visual.blocks.0.attn.proj",
                "model.visual.blocks.0.mlp.linear_fc1",
                "model.visual.blocks.0.mlp.linear_fc2",
                "model.visual.merger.linear_fc1",
                "model.visual.merger.linear_fc2",
            ],
        },
    )
    with_roots = QuantConfig.from_hf(hf, name_map=NameMap(roots=spec.checkpoint_roots, packed=spec.packed_modules_mapping))
    assert with_roots.scheme_for("model.layers.0.self_attn.qkv_proj") is not None
    assert with_roots.scheme_for("visual.blocks.0.attn.qkv") is None
    assert with_roots.scheme_for("visual.merger.linear_fc2") is None
    # without the visual -> model.visual root the tower would be built fp8 against bf16 tensors
    without = QuantConfig.from_hf(hf, name_map=NameMap(roots=(), packed=spec.packed_modules_mapping))
    assert without.scheme_for("visual.blocks.0.attn.qkv") is not None


def test_image_rows_take_the_leading_columns_and_deepstack_adds_the_next_block():
    H = 4
    table = torch.arange(10 * H, dtype=torch.float32).view(10, H)
    embed = SimpleNamespace(forward=lambda ids: table[ids], num_embeddings=10)
    assert torch.equal(embed_input_ids(embed, torch.tensor([1, 2]), SimpleNamespace(mm_embeds=None)), table[[1, 2]])
    ids = torch.tensor([1, MM_PAD_SHIFT_VALUE + 7, 2, MM_PAD_SHIFT_VALUE + 7])
    mm = torch.arange(2 * 3 * H, dtype=torch.float32).view(2, 3 * H)
    rows = torch.tensor([1, 3])
    x = embed_input_ids(embed, ids, SimpleNamespace(mm_embeds=mm, mm_rows=rows))
    assert torch.equal(x[0], table[1]) and torch.equal(x[2], table[2])
    assert torch.equal(x[1], mm[0, :H]) and torch.equal(x[3], mm[1, :H])
    deepstack_add(x, rows, mm, level=1, hidden_size=H)
    assert torch.equal(x[1], mm[0, :H] + mm[0, 2 * H : 3 * H]) and torch.equal(x[0], table[1])


# ------------------------------------------------------------------ tensor-parallel tower load
@contextlib.contextmanager
def _tp_rank(rank: int, world: int):
    """Run the body as one rank of a TP group, then restore the session's single-rank default."""
    saved = info._TP_INFO
    info._TP_INFO = DistributedInfo(rank, world)
    try:
        yield
    finally:
        info._TP_INFO = saved


def _tower():
    """The parsed vision section of the toy config above, with the full-width tensors to slice.

    The keys carry the reader's ``visual.`` prefix: that is the namespace a family's reader and
    the wrapper's state dict meet in, while the tower module itself names its buffers below it.
    """
    vc = parse_vision_config(_hf_config())
    with _tp_rank(0, 1):
        with torch.device("meta"):
            buffers = Qwen3VLVisionModel(vc).state_dict()
        full = {
            "visual." + name: torch.arange(int(tensor.numel()), dtype=torch.float32).reshape(tuple(tensor.shape))
            for name, tensor in buffers.items()
        }
    return vc, list(full), full


def test_the_tower_leaf_names_the_module_that_owns_the_cut():
    assert vision_module_leaf("visual.blocks.3.attn.qkv.weight") == "attn.qkv"
    assert vision_module_leaf("visual.blocks.3.attn.proj.bias") == "attn.proj"
    assert vision_module_leaf("visual.blocks.3.mlp.linear_fc1.weight") == "mlp.linear_fc1"
    # a merger's fc1 splits by the merged width, not by the block's intermediate size
    assert vision_module_leaf("visual.merger.linear_fc1.weight") == "merger.linear_fc1"
    assert vision_module_leaf("visual.deepstack_merger_list.1.linear_fc2.weight") == "deepstack_merger_list.linear_fc2"
    for replicated in ("visual.patch_embed.proj.weight", "visual.pos_embed.weight",
                       "visual.blocks.3.norm2.weight", "visual.merger.norm.bias"):
        assert vision_module_leaf(replicated) not in _TP_LEAF_AXIS, replicated


def test_each_rank_reads_its_heads_of_every_qkv_run():
    vc, _, full = _tower()
    hidden, qkv = vc.hidden_size, full["visual.blocks.0.attn.qkv.weight"]
    local = hidden // 2
    with _tp_rank(1, 2):
        cut = vision_tp_shard(vc).tensor("attn.qkv", qkv)
    assert torch.equal(cut, torch.cat([qkv[lo + local : lo + 2 * local] for lo in (0, hidden, 2 * hidden)]))


def test_slicing_the_tower_fills_the_buffers_the_rank_builds():
    """Every buffer the model declares for a rank is exactly what the slicer hands it: the loader asserts on this shape."""
    vc, names, full = _tower()
    for world in (2, 4):
        for rank in range(world):
            with _tp_rank(rank, world):
                shard = vision_tp_shard(vc)
                got = {name: shard.tensor(vision_module_leaf(name), full[name]) for name in names}
                with torch.device("meta"):
                    want = {"visual." + name: t for name, t in Qwen3VLVisionModel(vc).state_dict().items()}
            assert set(got) == set(want)
            for name, tensor in want.items():
                assert tuple(got[name].shape) == tuple(tensor.shape), f"world {world} rank {rank}: {name}"


def test_the_ranks_partition_the_sharded_leaves_without_overlap():
    """One rank holds each row and column of a sharded buffer, which is what the row-parallel all-reduce adds back."""
    vc, _, full = _tower()
    sharded = [n for n, t in full.items() if _TP_LEAF_AXIS.get(vision_module_leaf(n)) is not None]
    assert "visual.blocks.0.attn.qkv.weight" in sharded and "visual.patch_embed.proj.weight" not in sharded
    for name in sharded:
        whole = full[name]
        axis = _TP_LEAF_AXIS[vision_module_leaf(name)]
        runs = 3 if name.endswith(("qkv.weight", "qkv.bias")) else 1
        pieces = []
        for rank in range(2):
            with _tp_rank(rank, 2):
                pieces.append(vision_tp_shard(vc).tensor(vision_module_leaf(name), whole))
        if whole.dim() <= axis:
            # a row-parallel layer adds its bias after the all-reduce, so every rank holds it whole
            assert all(torch.equal(piece, whole) for piece in pieces), name
            continue
        for run in range(runs):
            per_rank, per_run = pieces[0].shape[axis] // runs, whole.shape[axis] // runs
            rebuilt = torch.cat([p.narrow(axis, run * per_rank, per_rank) for p in pieces], dim=axis)
            assert torch.equal(rebuilt, whole.narrow(axis, run * per_run, per_run)), f"{name} run {run}"
