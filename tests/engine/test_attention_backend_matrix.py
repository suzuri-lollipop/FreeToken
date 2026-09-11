"""Type x backend capability matrix: config-time resolution and rejection.

Golden table for the unified validation that replaced the per-model gates: every
model family stub declares its attention types through the group-spec walk, auto
must resolve to the historical winner, and every illegal explicit combination must
be rejected before weights load (in particular DSV4 x generic backends, which used
to pass config and crash post-load).
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.models.config import (
    DSV4AttentionGroupConfig,
    FullAttentionGroupConfig,
    KVCacheGroupSpec,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)


def _spec(name, attn_type, *, mla=False, sliding_window=None, index_head_dim=0, index_ratio=1):
    return KVCacheGroupSpec(
        name=name,
        layer_ids=(0, 1),
        num_kv_heads=1,
        head_dim=64,
        sliding_window=sliding_window,
        mla=mla,
        index_head_dim=index_head_dim,
        num_index_layers=2 if index_head_dim else 0,
        index_ratio=index_ratio,
        attn_type=attn_type,
    )


def _model_config(kind):
    mc = SimpleNamespace(
        model_type=kind,
        single_stream_only=False,
        is_moe=False,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=False,
        num_layers=4,
        rotary_config=SimpleNamespace(max_position=1024),
    )
    if kind == "full":
        specs = (_spec("full", AttnType.FULL),)
    elif kind == "swa":
        mc.has_swa_attention = True
        specs = (
            _spec("full", AttnType.FULL),
            _spec("swa", AttnType.SWA, sliding_window=128),
        )
    elif kind == "mla":
        specs = (_spec("full", AttnType.MLA, mla=True),)
    elif kind == "dsa":
        specs = (_spec("full", AttnType.DSA, mla=True, index_head_dim=128),)
    elif kind == "dsv4":
        mc.dsv4_args = SimpleNamespace(window_size=128)
        specs = (_spec("dsv4", AttnType.DSV4, sliding_window=128),)
    elif kind == "bsa":
        # MiniMax-M3 shape: one FULL-family group, mla=False + index dims -> BSA.
        specs = (_spec("full", AttnType.BSA, index_head_dim=128),)
    elif kind == "qsa":
        # Qwen3.8-Flash-Next shape: hybrid-linear + one FULL-family group whose index keys
        # are compressed index_ratio:1 -> QSA.
        mc.has_linear_attention = True
        specs = (_spec("full", AttnType.QSA, index_head_dim=128, index_ratio=4),)
    elif kind == "linear_hybrid":
        mc.has_linear_attention = True
        specs = (_spec("full", AttnType.FULL),)
    else:
        raise AssertionError(kind)
    mc.kv_cache_group_specs = lambda: specs
    return mc


def _config(kind, **overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )
    object.__setattr__(config, "model_config", _model_config(kind))
    return config


def _patch_env(monkeypatch, *, major=9, flashinfer=True, sgl=True):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: major == 10)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: major == 9)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: flashinfer)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: sgl)


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("full", "fa,fi"),  # sm90 tree
        ("linear_hybrid", "fa,fi"),  # LINEAR never constrains backend choice
        ("swa", "triton"),
        ("mla", "dsa"),  # plain latent MLA
        ("dsa", "dsa"),  # MLA + DSA indexer (GLM-5.2 shape)
        ("dsv4", "dsv4_sparse"),
        ("bsa", "m3_sparse"),  # MiniMax-M3 block-sparse GQA
        ("qsa", "qsa_sparse"),  # Qwen3.8-Flash-Next compressed-block sparse
    ],
)
def test_auto_resolves_per_type(monkeypatch, kind, expected):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config(kind, attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == expected


def test_auto_bsa_sets_block_page_size(monkeypatch):
    # m3_sparse declares page_sizes=(128,): one KV page == one sparse block, and
    # config-time resolution must coerce the page size to match.
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("bsa", attention_backend="auto")
    _adjust_config(config)
    assert config.page_size == 128


def test_bsa_rejects_float32_dtype(monkeypatch):
    # --dtype float32 used to pass config validation and die on the pool's
    # itemsize==2 assert only after the model was resident.
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("bsa", attention_backend="auto")
    object.__setattr__(config, "dtype", torch.float32)
    with pytest.raises(ValueError, match="16-bit"):
        _adjust_config(config)


def test_auto_qsa_sets_page_size_64(monkeypatch):
    # qsa_sparse registers page_sizes=(64,) (a 4-token compress group must never straddle
    # a page); the generic backend page-size coercion takes the default 1 to 64.
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("qsa", attention_backend="auto")
    assert config.page_size == 1
    _adjust_config(config)
    assert config.page_size == 64


def test_qsa_coerces_explicit_page_size(monkeypatch):
    # Same policy as m3_sparse (page_sizes=(128,)): an unsupported explicit value is
    # coerced to the backend's page size with a warning, not rejected.
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("qsa", attention_backend="auto", page_size=16)
    _adjust_config(config)
    assert config.page_size == 64


def test_qsa_rejects_float32_dtype(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("qsa", attention_backend="auto")
    object.__setattr__(config, "dtype", torch.float32)
    with pytest.raises(ValueError, match="16-bit"):
        _adjust_config(config)


def test_auto_dsv4_sets_window_page_size(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("dsv4", attention_backend="auto")
    _adjust_config(config)
    assert config.page_size == 128


@pytest.mark.parametrize(
    "kind, backend",
    [
        # reverse gates: type-specific backends on models without the type
        ("full", "dsa"),
        ("full", "dsv4_sparse"),
        ("full", "m3_sparse"),
        ("swa", "dsa"),
        ("full", "qsa_sparse"),
        # forward gates: generic backends on the BSA-locked model
        ("bsa", "fi"),
        ("bsa", "triton"),
        # forward gates: generic and neighbouring sparse backends on the QSA-locked model
        ("qsa", "fi"),
        ("qsa", "fa"),
        ("qsa", "triton"),
        ("qsa", "m3_sparse"),
        ("bsa", "qsa_sparse"),
        # forward gates: generic backends on type-locked models
        ("mla", "fi"),
        ("mla", "triton"),
        ("dsa", "fi"),
        ("dsa", "triton"),
        ("dsv4", "fi"),  # used to pass config and crash after weights load
        ("dsv4", "fa"),
        ("dsv4", "triton"),
        ("dsv4", "trtllm"),
        ("swa", "fa"),
        ("swa", "fi"),
        # comma pairs are validated per part
        ("swa", "fa,triton"),
        ("swa", "triton,fa"),
        ("dsv4", "dsv4_sparse,fi"),
        ("mla", "dsa,triton"),
    ],
)
def test_illegal_combinations_rejected_at_config_time(monkeypatch, kind, backend):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)  # every package/arch gate satisfied
    config = _config(kind, attention_backend=backend)
    with pytest.raises(ValueError, match="does not support"):
        _adjust_config(config)


@pytest.mark.parametrize(
    "kind, backend",
    [
        ("mla", "dsa"),
        ("dsa", "dsa"),
        ("dsv4", "dsv4_sparse"),
        ("qsa", "qsa_sparse"),
        ("swa", "triton"),
        ("full", "triton"),
        ("full", "fa,fi"),
    ],
)
def test_legal_explicit_combinations_pass(monkeypatch, kind, backend):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config(kind, attention_backend=backend)
    _adjust_config(config)
    assert config.attention_backend == backend


@pytest.mark.parametrize("kind", ["mla", "dsa"])
def test_mla_auto_adjusts_page_size(monkeypatch, kind):
    """Any --page-size is auto-adjusted (with a warning) to the layout's
    required size: 1 for plain latent-KV, 64 for the kpool variant."""
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config(kind, attention_backend="auto", page_size=16)
    _adjust_config(config)
    assert config.page_size == 1


def test_trtllm_page_size_coercion_is_part_aware(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)
    config = _config("full", attention_backend="fi,trtllm", page_size=1)
    _adjust_config(config)
    assert config.page_size == 64


def test_validate_rejects_more_than_two_parts():
    from argparse import ArgumentTypeError

    from freetoken.attention import validate_attn_backend

    with pytest.raises(ArgumentTypeError, match="At most two"):
        validate_attn_backend("fa,fi,triton")


def test_duck_typed_config_without_spec_walk_defaults_to_full(monkeypatch):
    from freetoken.engine.engine import _required_attn_types

    assert _required_attn_types(SimpleNamespace()) == frozenset({AttnType.FULL})
    assert _required_attn_types(SimpleNamespace(dsv4_args=object())) == frozenset(
        {AttnType.DSV4}
    )


def _rotary():
    return RotaryConfig(head_dim=64, rotary_dim=64, max_position=1024, base=1e4, scaling=None)


def _real_hybrid_model_config():
    kwargs = dict(
        num_layers=4,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=512,
        vocab_size=1000,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        rotary_config=_rotary(),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="test",
        architectures=["Test"],
    )
    return ModelConfig(
        **kwargs,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full", layer_ids=(0,), num_kv_heads=2, head_dim=64,
                rotary_config=_rotary(),
            ),
            SWAAttentionGroupConfig(
                name="swa", layer_ids=(1,), num_kv_heads=2, head_dim=64,
                rotary_config=_rotary(), sliding_window=128,
            ),
            LinearGatedDeltaGroupConfig(
                name="linear", layer_ids=(2,), num_key_heads=2, num_value_heads=2,
                key_head_dim=64, value_head_dim=64, conv_kernel_dim=4, output_gate=False,
            ),
            DSV4AttentionGroupConfig(
                name="dsv4", layer_ids=(3,), num_kv_heads=1, head_dim=512,
                sliding_window=128,
            ),
        ),
    )


def test_attn_type_for_layer_and_spec_derivation():
    mc = _real_hybrid_model_config()
    assert mc.attn_type_for_layer(0) is AttnType.FULL
    assert mc.attn_type_for_layer(1) is AttnType.SWA
    assert mc.attn_type_for_layer(2) is AttnType.LINEAR
    assert mc.attn_type_for_layer(3) is AttnType.DSV4

    by_name = {s.name: s for s in mc.kv_cache_group_specs()}
    # linear groups own no paged KV spec; the other three map onto their types
    assert set(by_name) == {"full", "swa", "dsv4"}
    assert by_name["full"].attn_type is AttnType.FULL
    assert by_name["swa"].attn_type is AttnType.SWA
    assert by_name["dsv4"].attn_type is AttnType.DSV4
    # a DSV4 spec must never read as SWA or MLA to generic spec walkers
    assert by_name["dsv4"].is_swa is False and by_name["dsv4"].mla is False


def test_mla_spec_derivation_from_full_group():
    mc = _real_hybrid_model_config()
    mla_group = FullAttentionGroupConfig(
        name="full", layer_ids=(0,), num_kv_heads=1, head_dim=576,
        rotary_config=_rotary(), mla=True,
    )
    mc2 = ModelConfig(
        **{
            **{f: getattr(mc, f) for f in (
                "num_layers", "num_qo_heads", "num_kv_heads", "head_dim", "hidden_size",
                "vocab_size", "intermediate_size", "rms_norm_eps", "rotary_config",
                "hidden_act", "tie_word_embeddings", "num_experts", "num_experts_per_tok",
                "moe_intermediate_size", "norm_topk_prob", "model_type", "architectures",
            )},
        },
        attention_groups=(mla_group,),
    )
    (spec,) = mc2.kv_cache_group_specs()
    assert spec.attn_type is AttnType.MLA and spec.mla is True
    assert mc2.attn_type_for_layer(0) is AttnType.MLA

    # mla + index slab (GLM-5.2 shape) derives DSA, mirroring the DSAKVCache split
    dsa_group = FullAttentionGroupConfig(
        name="full", layer_ids=(0,), num_kv_heads=1, head_dim=576,
        rotary_config=_rotary(), mla=True, index_head_dim=128, num_index_layers=1,
    )
    mc3 = dataclasses_replace_groups(mc2, (dsa_group,))
    (spec3,) = mc3.kv_cache_group_specs()
    assert spec3.attn_type is AttnType.DSA
    assert mc3.attn_type_for_layer(0) is AttnType.DSA


def dataclasses_replace_groups(mc, groups):
    import dataclasses

    return dataclasses.replace(mc, attention_groups=groups)


def test_linear_attention_defaults_to_hybrid_radix():
    from freetoken.engine.engine import _resolve_cache_type

    assert _resolve_cache_type(True, "radix") == "hybrid_radix"
    assert _resolve_cache_type(True, "naive") == "naive"


# --- --kv-cache-dtype x backend capability -------------------------------------------
# A quantized pool is only readable by a backend that applies its descales: ignoring the
# scales yields wrong logits, not slow ones, so the combination is refused at config time.
# fa and trtllm carry the code path but are not hardware-validated yet (see the registry
# comments), so both report supports_fp8_kv=False until they are. Model families whose KV
# pool cannot even hold a quantized buffer are refused one step earlier, by the pool check
# that runs before the backend search.


@pytest.mark.parametrize("major", [9, 10])
@pytest.mark.parametrize("kind, expected", [("full", "fi"), ("swa", "triton")])
def test_auto_skips_backends_that_cannot_descale(monkeypatch, major, kind, expected):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=major)
    config = _config(kind, attention_backend="auto", kv_cache_dtype="fp8_e4m3")
    _adjust_config(config)
    assert config.attention_backend == expected


def test_auto_without_quantization_keeps_the_unquantized_winner(monkeypatch):
    """Control for the test above: the same config without --kv-cache-dtype still
    resolves to fa,fi on sm90/sm100, so the skip really is caused by the quantized pool."""
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=9)
    config = _config("full", attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == "fa,fi"


@pytest.mark.parametrize("backend", ["fa", "trtllm", "fa,fi", "fi,trtllm"])
def test_explicit_backend_without_descale_support_is_rejected(monkeypatch, backend):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)  # every package/arch gate satisfied
    config = _config("full", attention_backend=backend, kv_cache_dtype="fp8_e4m3")
    with pytest.raises(RuntimeError, match="quantized KV cache"):
        _adjust_config(config)


@pytest.mark.parametrize("backend", ["triton", "fi", "fi,triton"])
def test_descale_capable_backends_accept_a_quantized_cache(monkeypatch, backend):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)
    config = _config("full", attention_backend=backend, kv_cache_dtype="fp8_e4m3")
    _adjust_config(config)
    assert config.attention_backend == backend


def test_only_the_quantized_dtype_gates_the_backends(monkeypatch):
    """--kv-cache-dtype auto/none (i.e. no quantization) resolves like an unset flag."""
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=9)
    for dtype in ("auto", "none"):
        config = _config("full", attention_backend="auto", kv_cache_dtype=dtype)
        _adjust_config(config)
        assert config.attention_backend == "fa,fi"
        assert config.kv_quant is None


@pytest.mark.parametrize("kind", ["mla", "dsa", "dsv4", "bsa"])
def test_quantized_cache_is_refused_by_the_pool_before_the_backend(monkeypatch, kind):
    """Latent/index-key pools cannot descale their own reads, so the POOL is what rejects
    --kv-cache-dtype for these families. It must say so: the backend search one step later
    can only report "no backend on this machine", which sends the user hunting for a
    hardware or package problem instead of dropping the flag."""
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)  # every package/arch gate satisfied
    config = _config(kind, attention_backend="auto", kv_cache_dtype="fp8_e4m3")
    with pytest.raises(RuntimeError, match=r"is not supported by the \w+ pool"):
        _adjust_config(config)


def test_qsa_accepts_a_quantized_cache(monkeypatch):
    """QSA is the one index-key family that does descale: its paged K/V goes through the
    in-tree Triton kernel, which takes the pool's scale pair. Only that slab quantizes, so
    the family stays on the accepted list even though its index tiers do not."""
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)
    config = _config("qsa", attention_backend="auto", kv_cache_dtype="fp8_e4m3")
    _adjust_config(config)
    assert config.attention_backend == "qsa_sparse"
    assert config.kv_quant is not None


def test_auto_names_the_backends_a_quantized_cache_blocked(monkeypatch):
    """Control for the test above: when the pool is fine and only the descale capability is
    missing, the failure names the blocked candidates and the way out, rather than blaming
    the machine."""
    from dataclasses import replace

    from freetoken.attention import attention_backend_info
    from freetoken.engine import engine

    _patch_env(monkeypatch, major=10)
    monkeypatch.setattr(
        engine,
        "attention_backend_info",
        lambda name: replace(attention_backend_info(name), supports_fp8_kv=False),
    )
    config = _config("full", attention_backend="auto", kv_cache_dtype="fp8_e4m3")
    with pytest.raises(RuntimeError) as excinfo:
        engine._adjust_config(config)
    msg = str(excinfo.value)
    assert "triton" in msg and "fi" in msg and "trtllm" in msg
    assert "drop --kv-cache-dtype" in msg
