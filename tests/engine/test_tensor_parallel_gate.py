"""Config-time tensor-parallelism gate.

``--tp-size`` splits heads, MLP widths and the KV pool across ranks, so a configuration
that cannot be sharded has to say so before any rank builds a CUDA context or reads a
checkpoint: the family needs a TP-aware weight reader, the geometry needs to divide, and
the experts have to sit somewhere a rank can price (the offload cache is not TP-aware).
Backend resolution is covered here too -- only backends that read TP-local heads may be
selected once tp_size > 1.
"""

from types import SimpleNamespace

import pytest
import torch


def _model_config(**overrides):
    mc = SimpleNamespace(
        model_type="test_model",
        single_stream_only=False,
        is_moe=False,
        moe_enabled=False,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=False,
        num_layers=4,
        num_qo_heads=16,
        num_kv_heads=8,
        head_dim=128,
        attention_groups=(),
        intermediate_size=4864,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        rotary_config=SimpleNamespace(max_position=1024),
    )
    for name, value in overrides.items():
        setattr(mc, name, value)
    return mc


def _config(**kwargs):
    """EngineConfig with a duck-typed model config/spec, so no checkpoint is read."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    tp = kwargs.pop("tp", 1)
    model_path = kwargs.pop("path", "/tmp/freetoken-test-model")
    model_config = _model_config(**kwargs.pop("model", {}))
    model_spec = kwargs.pop("spec", None) or SimpleNamespace(tp_supported=True)
    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=tp),
        dtype=torch.bfloat16,
        **kwargs,
    )
    object.__setattr__(config, "model_config", model_config)
    object.__setattr__(config, "model_spec", model_spec)
    return config


def _preflight(config):
    from freetoken.engine.config import tp_preflight_error

    return tp_preflight_error(config)


def test_tp_one_is_never_gated():
    config = _config(tp=1, spec=SimpleNamespace(tp_supported=False))
    assert _preflight(config) is None


def test_family_without_a_tp_reader_is_rejected():
    from freetoken.engine.engine import _adjust_config

    config = _config(tp=2, spec=SimpleNamespace(tp_supported=False))
    with pytest.raises(ValueError, match="does not shard its checkpoint"):
        _adjust_config(config)


def test_an_ftw_directory_is_rejected_under_tp(tmp_path):
    """FTW stores the tensors already fused at full width, so its replay has no rank slice."""
    from freetoken.checkpoint.ftw import INDEX_NAME

    (tmp_path / INDEX_NAME).write_text("{}")
    error = _preflight(_config(tp=2, path=str(tmp_path)))
    assert error is not None and "FTW checkpoint" in error
    assert _preflight(_config(tp=1, path=str(tmp_path))) is None


@pytest.mark.parametrize(
    "tp, model, expected",
    [
        # 16 qo / 8 kv / 4864 intermediate: splits cleanly to 8 ranks
        (8, {}, None),
        (3, {}, "query heads are not divisible by 3"),
        # 3 KV heads neither split nor replicate over 2 ranks (div_even's allow_replicate)
        (2, {"num_qo_heads": 16, "num_kv_heads": 3}, "KV heads neither split"),
        # TP beyond the KV head count replicates them, which is legal
        (4, {"num_kv_heads": 2}, None),
        (6, {"num_kv_heads": 4}, "KV heads neither split"),
        (2, {"intermediate_size": 4865}, "intermediate size 4865"),
        # GDN heads ride the same split/replicate rule as KV heads
        (
            3,
            {
                "attention_groups": (
                    SimpleNamespace(
                        name="linear", num_key_heads=16, num_value_heads=32, num_kv_heads=None
                    ),
                )
            },
            "key heads (16) neither split",
        ),
    ],
)
def test_geometry_gate(tp, model, expected):
    from freetoken.engine.config import tp_shard_error

    error = tp_shard_error(_model_config(**model), tp)
    if expected is None:
        assert error is None
    else:
        assert expected in error



def test_moe_expert_shard_must_stay_on_its_scale_group():
    from freetoken.engine.config import tp_shard_error

    moe = dict(
        is_moe=True,
        moe_enabled=True,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
    )
    # 640 / (8 * 16) == 5 whole nvfp4 scale groups per rank
    assert tp_shard_error(_model_config(expert_quant="nvfp4", **moe), 8) is None
    # 16 ranks would slice the intermediate below one 16-column group
    assert "nvfp4 experts" in tp_shard_error(_model_config(expert_quant="nvfp4", **moe), 16)
    # block-fp8 carries 128-wide blocks along the same axis
    assert "fp8_block experts" in tp_shard_error(_model_config(expert_quant="fp8_block", **moe), 4)
    deepseek = {**moe, "moe_intermediate_size": 2048}
    assert tp_shard_error(_model_config(expert_quant="fp8_block", **deepseek), 8) is None
    assert "fp8_block experts" in tp_shard_error(_model_config(expert_quant="fp8_block", **deepseek), 3)
    # a shared expert that cannot divide is reported on its own
    error = tp_shard_error(
        _model_config(
            is_moe=True,
            moe_enabled=True,
            moe_intermediate_size=640,
            shared_expert_intermediate_size=641,
        ),
        2,
    )
    assert "shared-expert" in error


@pytest.mark.parametrize("strategy", ["auto", "offload", "cpu", "hybrid"])
def test_offload_family_is_rejected_under_tp(strategy):
    config = _config(
        tp=2,
        moe_strategy=strategy,
        model=dict(is_moe=True, moe_enabled=True, moe_intermediate_size=640),
    )
    error = _preflight(config)
    assert error is not None and "--moe-strategy fused" in error


def test_a_family_routed_by_model_type_is_still_gated():
    # qwen3_moe never sets moe_enabled; is_moe comes from its model_type. Reading only the
    # flag would let its ranks stream full-size expert banks through the TP-unaware cache.
    moe_model = dict(model_type="qwen3_moe", is_moe=True, moe_intermediate_size=768)
    error = _preflight(_config(tp=2, model=moe_model))
    assert error is not None and "--moe-strategy fused" in error
    # the resident expert path stays open for it
    assert _preflight(_config(tp=2, moe_strategy="fused", model=moe_model)) is None


def test_resident_experts_pass_the_gate():
    config = _config(
        tp=2,
        moe_strategy="fused",
        model=dict(is_moe=True, moe_enabled=True, moe_intermediate_size=640),
    )
    assert _preflight(config) is None


def _patch_env(monkeypatch, *, major=9, flashinfer=True, sgl=True):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: major == 10)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: major == 9)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: flashinfer)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: sgl)


def test_tp_one_still_resolves_the_fastest_backend(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)
    config = _config(tp=1, attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == "trtllm"


@pytest.mark.parametrize("major", [10, 9])
def test_auto_drops_backends_that_do_not_shard_heads(monkeypatch, major):
    from freetoken.engine.engine import _adjust_config

    # Same machine at TP=2: trtllm (sm_100) and fa (sm_90) both read global head counts.
    _patch_env(monkeypatch, major=major)
    config = _config(tp=2, attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == "fi"


def test_explicit_non_sharding_backend_is_rejected_under_tp(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)
    config = _config(tp=2, attention_backend="trtllm")
    with pytest.raises(ValueError, match="does not shard attention heads"):
        _adjust_config(config)


def test_no_sharding_backend_available_reports_tp(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10, flashinfer=False)
    config = _config(tp=2, attention_backend="auto")
    with pytest.raises(RuntimeError, match="tensor parallelism over 2 ranks"):
        _adjust_config(config)
