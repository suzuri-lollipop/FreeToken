"""MTP head module (models/qwen4_exp/mtp.py), CPU-buildable pieces.

The head mirrors one main decoder layer over the hyper-connection streams; these
tests pin the dense-parameter naming the loader must map onto (checkpoint names,
with the loader-side fusions the main model already uses: q|k|v -> qkv_proj,
down+inject -> input_mix_weight_down_block_inject), the toy-geometry shapes, and
the front-end wiring switch. The fp8-block experts and the TP2 placement are
Phase 2 of _scratch/mtp_design.md; the attention forward needs the QSA backend
and is exercised from Phase 1 onward.
"""

from types import SimpleNamespace

import pytest
import torch

from .common import fill_weights, parsed_config, toy_hf_config


def _mtp_block(theta=None):
    # theta None -> the config carries no mtp rope_theta; the head then shares the
    # main rotary config unconditionally (the released checkpoint sets it equal).
    return SimpleNamespace(
        hybrid=True,
        layer_types=["full_attention"],
        num_hidden_layers=1,
        rope_theta=theta,
        mtp_use_hidden_state_from_layer=None,
    )


def _head(wiring="norm_mix_fc", **text_kw):
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead

    config = parsed_config(mtp=_mtp_block(**text_kw) if "mtp" not in text_kw else text_kw.pop("mtp"), **text_kw)
    return Qwen4ExpMTPHead(config, layer_id=config.num_layers, wiring=wiring), config


def test_dense_parameter_names_match_the_checkpoint_fused_forms():
    head, _ = _head()
    keys = set(head.state_dict().keys())
    expected_dense = {
        "fc_embedding.weight",
        "fc_hidden.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "hyper_connection_mixer.hc_norm.weight",
        "hyper_connection_mixer.input_mix_weight_down.weight",
        "hyper_connection_mixer.input_mix_weight_up.weight",
        "layers.0.attn_hyper_connection.hc_norm.weight",
        "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        "layers.0.attn_hyper_connection.input_mix_weight_up.weight",
        "layers.0.mlp_hyper_connection.hc_norm.weight",
        "layers.0.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
        "layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.indexer.index_qk_proj.weight",
        "layers.0.self_attn.indexer.q_layernorm.weight",
        "layers.0.self_attn.indexer.k_layernorm.weight",
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.shared_expert.gate_up_proj.weight",  # loader fuses gate_proj+up_proj
        "layers.0.mlp.shared_expert.down_proj.weight",
        "layers.0.mlp.shared_expert_gate.weight",
    }
    missing = expected_dense - keys
    assert not missing, f"missing dense keys: {sorted(missing)}"
    # every other key must belong to the routed experts (phase 2 territory)
    extras = {k for k in keys if k not in expected_dense and ".mlp.experts" not in k}
    assert not extras, f"unexpected non-expert keys: {sorted(extras)}"


def test_shapes_follow_the_toy_geometry():
    head, config = _head()
    args = config.qwen4_args
    h, hc, lr = args.hidden_size, args.hc_count, args.hc_lowrank
    width = hc * h
    sd = head.state_dict()
    assert sd["fc_embedding.weight"].shape == (h, h)
    assert sd["fc_hidden.weight"].shape == (h, h)
    assert sd["pre_fc_norm_embedding.weight"].shape == (h,)
    assert sd["pre_fc_norm_hidden.weight"].shape == (width,)
    assert sd["hyper_connection_mixer.input_mix_weight_down.weight"].shape == (lr, width)
    assert sd["hyper_connection_mixer.input_mix_weight_up.weight"].shape == (width, lr)
    pad = (-(lr + hc)) % 16
    merged = sd["layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"]
    assert merged.shape == (lr + hc + pad, width)
    nq, nkv, hd = config.num_qo_heads, config.num_kv_heads, config.head_dim
    assert sd["layers.0.self_attn.qkv_proj.weight"].shape == (nq * hd * 2 + 2 * nkv * hd, h)
    assert sd["layers.0.self_attn.o_proj.weight"].shape == (h, nq * hd)
    assert head.layers.op_list[0].self_attn.layer_id == config.num_layers


def test_front_end_runs_on_cpu_and_repeats_to_the_stream_width():
    head, config = _head()
    fill_weights(head, seed=3, device=torch.device("cpu"))
    args = config.qwen4_args
    r = torch.randn(5, args.ple_state_width)
    e = torch.randn(5, args.hidden_size)
    out = head.fuse_input(r, e)
    assert out.shape == (5, args.ple_state_width)
    # the four streams are copies of one fused [T, hidden] vector
    folded = out.unflatten(-1, (args.hc_count, args.hidden_size))
    assert torch.equal(folded[:, 0], folded[:, 1]) and torch.equal(folded[:, 0], folded[:, 3])


def test_wiring_switch_changes_the_front_end():
    head_a, config = _head(wiring="norm_mix_fc")
    head_b, _ = _head(wiring="norm_mixfrom_fc")
    for h in (head_a, head_b):
        fill_weights(h, seed=7, device=torch.device("cpu"))
    # same seeds -> identical weights; only the mix path differs (mix applies the
    # mixer's hc_norm internally, mix_from_normed consumes the pre-normed R).
    # mix_from_normed's kernels are CUDA-only, so compare construction here; the
    # numeric A/B runs in the GPU wiring probe (design doc section 4).
    assert head_a.wiring != head_b.wiring
    assert head_a.hyper_connection_mixer.use_combine is False


def test_rejects_unknown_wiring_and_missing_mtp_block():
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead

    with pytest.raises(ValueError, match="unknown MTP wiring"):
        _head(wiring="bogus")
    config = parsed_config()  # toy config without an mtp block
    with pytest.raises(ValueError, match="no MTP block"):
        Qwen4ExpMTPHead(config, layer_id=config.num_layers)


def test_rejects_mtp_rope_theta_mismatch():
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead

    config = parsed_config(mtp=_mtp_block(theta=12345.0))
    with pytest.raises(NotImplementedError, match="rope_theta"):
        Qwen4ExpMTPHead(config, layer_id=config.num_layers)


def test_head_experts_stay_resident_under_an_offload_engine():
    # The offload cache is keyed by main-decoder layer ids, so the head's MoE must not
    # become an OffloadMoELayer at the synthetic index -- it builds resident whatever
    # the engine's strategy says (TP2 placement of the fp8-block experts is Phase 2).
    from dataclasses import replace

    from freetoken.layers import OffloadMoELayer
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead

    config = replace(parsed_config(mtp=_mtp_block()), moe_strategy="offload")
    head = Qwen4ExpMTPHead(config, layer_id=config.num_layers)
    assert not isinstance(head.layers.op_list[0].mlp.experts, OffloadMoELayer)


def test_model_attaches_the_head_only_when_enabled():
    from freetoken.models.qwen4_exp.config import extend_config_for_mtp
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    base = parsed_config(mtp=_mtp_block())
    assert Qwen4ExpForCausalLM(base).mtp is None  # recorded, not enabled

    model = Qwen4ExpForCausalLM(extend_config_for_mtp(base))
    assert model.mtp is not None
    # the strict-load naming: the head hangs under `mtp`, matching the checkpoint keys
    keys = set(model.state_dict())
    assert any(k.startswith("mtp.fc_embedding") for k in keys)
    assert any(k.startswith("mtp.layers.0.mlp.experts.") for k in keys)
