"""Config-time resolution for inkling2_loop: the SWA + conv-state combination runs on
the naive prefix cache, and the MoE fetch overlap is disabled because the loop revisits
expert layers out of the sequential order the prefetch assumes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.inkling2.config import parse_config

from .test_inkling2 import _hf_dict


def _config(**overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-inkling2-test",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )
    object.__setattr__(config, "model_config", parse_config(SimpleNamespace(**_hf_dict())))
    return config


def test_swa_plus_conv_state_forces_naive_cache():
    from freetoken.engine.engine import _adjust_config

    config = _config()
    _adjust_config(config)
    assert config.cache_type == "naive"
    assert config.page_size == 1
    assert config.attention_backend == "triton"


def test_loop_disables_moe_fetch_overlap():
    from freetoken.engine.engine import _adjust_config

    config = _config(moe_strategy="offload")
    assert config.moe_prefill_overlap and config.decode_fetch_overlap
    _adjust_config(config)
    assert not config.moe_prefill_overlap
    assert not config.decode_fetch_overlap


def test_visit_schedule_and_kv_groups():
    mc = parse_config(SimpleNamespace(**_hf_dict()))
    args = mc.inkling2_args
    # 8 physical layers, core [2, 6) twice: 2 + 4*2 + 2 = 12 visits
    assert mc.num_layers == 12
    assert mc.moe_layer_count == 8
    assert args.visit_physical == (0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7)
    swa, full, linear = mc.attention_groups
    assert len(swa.layer_ids) + len(full.layer_ids) == 12
    assert linear.layer_ids == ()
    assert mc.slot_states[0].layer_ids == tuple(range(12))


# ---------------------------------------------------------------- dspark (the official drafter)

def _mini_hf_dict() -> dict:
    """The released Inkling-2-Mini text geometry (the drafter serves only this shape)."""
    d = _hf_dict()
    d["text_config"] = SimpleNamespace(
        model_type="inkling2_loop_text",
        hidden_size=2304, num_hidden_layers=24, vocab_size=201024, unpadded_vocab_size=200058,
        num_attention_heads=24, num_key_value_heads=8, head_dim=128,
        swa_num_attention_heads=24, swa_num_key_value_heads=8, swa_head_dim=128,
        d_rel=16, rel_extent=1024, sliding_window_size=1024,
        qk_l2_clip_threshold=22.627416997969522, rel_l2_clip_threshold=8.0,
        log_scaling_n_floor=8192, log_scaling_alpha=0.1,
        sconv_kernel_size=4, rms_norm_eps=1e-6,
        local_layer_ids=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
        n_layers_per_attn_res_block=6,
        n_routed_experts=128, num_experts_per_tok=8, n_shared_experts=0,
        intermediate_size=1152, mlp_gate_softcap=4.0, mlp_up_softcap=25.0,
        route_scale=8.0, use_gate_bias=True, final_logit_softcapping=30.0,
        loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=False),
    )
    return d


def _mini_config(**overrides):
    config = _config(**overrides)
    object.__setattr__(config, "model_config", parse_config(SimpleNamespace(**_mini_hf_dict())))
    return config


def _drafter_file(tmp_path):
    import safetensors.torch

    p = tmp_path / "mtp.safetensors"
    safetensors.torch.save_file({
        "fuse_weight": torch.zeros(4, 4), "norm.weight": torch.zeros(2304),
        "mask_embed": torch.zeros(2304), "markov_head.markov_w1.weight": torch.zeros(201024, 256),
        "markov_head.rnn_proj.weight": torch.zeros(768, 5120),
        "markov_head.hidden_out.weight": torch.zeros(2304, 256),
    }, str(p))
    return str(p)


def test_dspark_needs_the_mini_geometry(tmp_path):
    from freetoken.engine.engine import _adjust_config

    # the miniature inkling2 config (width 64, 12 visits) is inkling2_loop but not Mini
    config = _config(speculative_algorithm="dspark", speculative_draft_model_path=_drafter_file(tmp_path))
    with pytest.raises(ValueError, match="Mini geometry"):
        _adjust_config(config)


def test_dspark_rejects_other_models(tmp_path):
    from freetoken.spec.dspark.proposer import DSparkProposer

    config = _config(speculative_algorithm="dspark", speculative_draft_model_path=_drafter_file(tmp_path))
    object.__setattr__(config, "model_config", SimpleNamespace(inkling2_args=None, slot_states=()))
    with pytest.raises(ValueError, match="Inkling-2-Mini"):
        DSparkProposer.adjust_config(config, lambda a, v: None)


def test_dspark_resolves_on_mini_and_lifts_the_swa_ban(tmp_path):
    from freetoken.engine.engine import _adjust_config

    config = _mini_config(speculative_algorithm="dspark", speculative_draft_model_path=_drafter_file(tmp_path))
    # SWA + spec used to be refused outright; inkling2's committed-frontier eviction lifts it
    _adjust_config(config)
    assert config.speculative_num_steps == 7
    assert config.decode_rows_per_req == 8
    assert [s.name for s in config.spec_slot_states] == ["ik2_draft_ctx_local", "ik2_draft_ctx_full", "ik2_draft_frnn"]
    assert config.cache_type == "naive"


def test_dspark_slot_shapes_and_budget(tmp_path, monkeypatch):
    from freetoken.spec.dspark.proposer import DSparkProposer

    config = _mini_config(speculative_algorithm="dspark", speculative_draft_model_path=_drafter_file(tmp_path))
    local, full, frnn = DSparkProposer.slot_states(config)
    assert local.shape == (2, 2048, 1024) and local.layer_ids == (0, 2) and local.dtype == torch.float32
    assert full.shape == (2, 8192, 1024) and full.layer_ids == (1,) and full.dtype == torch.float16
    assert frnn.shape == (2304,) and frnn.dtype == torch.float32
    fixed, per_token = DSparkProposer.budget(config)
    assert per_token == 0  # the context rides the slots, not paged KV
    assert fixed > 600 * 2**20  # the drafter's weights + fold + tap buffers
    monkeypatch.setenv("FREETOKEN_IK2_DRAFT_CTX", "4096")
    assert DSparkProposer.slot_states(config)[1].shape == (2, 4096, 1024)
    monkeypatch.setenv("FREETOKEN_IK2_DRAFT_CTX", "512")
    assert DSparkProposer.slot_states(config)[1].shape == (2, 2048, 1024)  # Maya's floor
