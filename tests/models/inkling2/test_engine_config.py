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
