"""Inkling-2-Mini weight reader against a synthetic checkpoint in the release layout:
model.llm.* dense names, stacked NVFP4 experts (amax / 2688 globals) on the leading
layers and MXFP8 experts on the trailing ones, served dequantized to bf16.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from freetoken.layers.quantization import QuantKind, set_quant_config
from freetoken.models.inkling2.config import parse_config
from freetoken.models.inkling2.model import Inkling2LoopForConditionalGeneration
from freetoken.models.inkling2.weight import iter_expert_pieces, iter_weights
from freetoken.models.register import checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config

from .test_inkling2 import (
    BLOCK, CORE_END, CORE_START, D_REL, E, EXTENT, H, HD, INTER, KERNEL, NH, NK, PHYS,
    RANK, SWA_PHYS, UNPADDED, VOCAB, WINDOW, _hf_dict,
)

KV_DIM = NK * HD
NVFP4_LAYERS = list(range(6))
MXFP8_LAYERS = [6, 7]


def _checkpoint(path):
    torch.manual_seed(1)
    bf16 = lambda *s: (torch.randn(*s) * 0.1).to(torch.bfloat16)
    sd = {
        "model.llm.embed.weight": bf16(VOCAB, H),
        "model.llm.embed_norm.weight": bf16(H),
        "model.llm.norm.weight": bf16(H),
        "model.llm.unembed.weight": bf16(VOCAB, H),
        "model.llm.attn_res.output_query": torch.randn(H),
        "model.llm.middle_cycle_cross_pass_queries": torch.randn(2 * (CORE_END - CORE_START), H),
        "model.llm.middle_cycle_cross_pass_kv_queries": torch.randn(2 * (CORE_END - CORE_START), H),
    }
    for i in range(PHYS - 1):
        q = torch.randn(3, H) if (i + 1) not in SWA_PHYS else torch.randn(H)
        sd[f"model.llm.attn_res.attn_queries.{i}"] = q
    for i in range(PHYS):
        sd[f"model.llm.attn_res.mlp_queries.{i}"] = torch.randn(H)
    for p in range(PHYS):
        pre = f"model.llm.layers.{p}"
        a = f"{pre}.attn"
        swa = p in SWA_PHYS
        for leaf, rows in (("wq_du", NH * HD), ("wk_dv", KV_DIM), ("wv_dv", KV_DIM), ("wr_du", NH * D_REL), ("gate_attn", NH * HD), ("attn_sink_proj", NH)):
            sd[f"{a}.{leaf}.weight"] = bf16(rows, H)
        sd[f"{a}.wo_ud.weight"] = bf16(H, NH * HD)
        sd[f"{a}.q_l2_clip.scale"] = bf16(NH, HD)
        sd[f"{a}.rel_l2_clip.scale"] = bf16(NH, D_REL)
        sd[f"{a}.rel_logits_proj.proj"] = bf16(D_REL, WINDOW if swa else EXTENT)
        for name in ("k_sconv", "v_sconv"):
            if swa:
                sd[f"{a}.{name}.weight"] = bf16(KV_DIM, 1, KERNEL)
            else:
                sd[f"{a}.{name}.static_w"] = bf16(KERNEL, KV_DIM)
                sd[f"{a}.{name}.weight_proj"] = bf16(NK * KERNEL, H)
        if swa:
            sd[f"{pre}.attn_norm.weight"] = bf16(H)
        else:
            for n in ("q", "k", "v"):
                sd[f"{pre}.{n}_input_norm.weight"] = bf16(H)
        sd[f"{pre}.attn_sconv.weight"] = bf16(H, 1, KERNEL)
        sd[f"{pre}.mlp_sconv.weight"] = bf16(H, 1, KERNEL)
        sd[f"{pre}.mlp.gate.weight"] = bf16(E, H)
        sd[f"{pre}.mlp.gate.bias"] = torch.randn(E)
        sd[f"{pre}.mlp_norm.weight"] = bf16(H)
        if CORE_START <= p < CORE_END:
            for kind in ("attn", "k", "v", "mlp"):
                sd[f"{pre}.controllers.{kind}.down_weight"] = bf16(RANK, 2 * H)
                sd[f"{pre}.controllers.{kind}.gain_weight"] = bf16(H, RANK)
                sd[f"{pre}.controllers.{kind}.forget_weight"] = bf16(H, RANK)
        ex = f"{pre}.mlp.experts"
        if p in NVFP4_LAYERS:
            sd[f"{ex}.w13_weight"] = torch.randint(0, 255, (E, 2 * INTER, H // 2), dtype=torch.uint8)
            sd[f"{ex}.w13_weight_scale"] = torch.ones(E, 2 * INTER, H // 16).to(torch.float8_e4m3fn)
            sd[f"{ex}.w13_weight_amax"] = torch.rand(E) + 0.1
            sd[f"{ex}.w13_weight_outer_scale_denominator"] = torch.tensor([2688.0])
            sd[f"{ex}.w2_weight"] = torch.randint(0, 255, (E, H, INTER // 2), dtype=torch.uint8)
            sd[f"{ex}.w2_weight_scale"] = torch.ones(E, H, INTER // 16).to(torch.float8_e4m3fn)
            sd[f"{ex}.w2_weight_amax"] = torch.rand(E) + 0.1
            sd[f"{ex}.w2_weight_outer_scale_denominator"] = torch.tensor([2688.0])
        else:
            sd[f"{ex}.w13_weight"] = (torch.randn(E, 2 * INTER, H) * 0.1).to(torch.float8_e4m3fn)
            sd[f"{ex}.w13_weight_scale"] = torch.full((E, 2 * INTER, H // 32), 127, dtype=torch.uint8)
            sd[f"{ex}.w2_weight"] = (torch.randn(E, H, INTER) * 0.1).to(torch.float8_e4m3fn)
            sd[f"{ex}.w2_weight_scale"] = torch.full((E, H, INTER // 32), 127, dtype=torch.uint8)
    safetensors.torch.save_file(sd, path / "model.safetensors")
    hf = _hf_dict()
    hf["text_config"] = vars(hf["text_config"])
    (path / "config.json").write_text(json.dumps(hf | {
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }))
    groups = []
    for layers, dtype in ((NVFP4_LAYERS, "nvfp4"), (MXFP8_LAYERS, "mxfp8")):
        groups.append({
            "compute_dtype": dtype,
            "tensors": [f"model.llm.layers.{p}.mlp.experts.w13_weight" for p in layers],
        })
    (path / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": groups}}))
    return sd


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the NVFP4 MoE method probes the device")
def test_dense_pass_fills_the_model_and_experts_stay_out(tmp_path, monkeypatch):
    from freetoken.distributed import info
    from freetoken.layers.quantization import NameMap
    from freetoken.layers.quantization.configs.base import quantization_config_of
    from freetoken.layers.quantization.configs.modelopt import ModelOptConfig
    from freetoken.models.register import _INKLING2_ROOTS, _INKLING2_SEGMENTS, _INKLING2_UNQUANTIZED

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    stored = _checkpoint(tmp_path)
    hf = cached_load_hf_config(str(tmp_path))
    spec = get_model_spec("Inkling2LoopForConditionalGeneration")
    # the release globs keep the real checkpoint's MXFP8 layers (20-23) bf16; the synthetic
    # checkpoint puts them at 6-7, so swap that one glob for the test layout
    from freetoken.layers.quantization.names import name_set

    assert name_set(_INKLING2_UNQUANTIZED)("model.llm.layers.20.mlp.experts")
    assert not name_set(_INKLING2_UNQUANTIZED)("model.llm.layers.0.mlp.experts")
    unquantized = tuple(g for g in _INKLING2_UNQUANTIZED if "2[0-3]" not in g) + ("model.llm.layers.[67].mlp.experts*",)
    quant = ModelOptConfig(
        quantization_config_of(hf), hf,
        name_map=NameMap(roots=_INKLING2_ROOTS, segments=_INKLING2_SEGMENTS),
        unquantized=unquantized,
    )
    set_quant_config(quant)
    # only the leading experts are NVFP4; the MXFP8 layers and every dense module stay bf16
    assert quant.scheme_for("model.layers.0.mlp.experts").kind is QuantKind.NVFP4
    assert quant.scheme_for("model.layers.6.mlp.experts") is None
    assert quant.scheme_for("model.layers.0.self_attn.q_proj") is None
    assert quant.scheme_for("lm_head") is None
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Inkling2LoopForConditionalGeneration(replace(parse_config(hf), quant=quant))
        got = dict(iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    finally:
        torch.set_default_dtype(prev)
    want = model.state_dict()
    assert set(got) == set(want)
    assert all(got[k].shape == want[k].shape and got[k].dtype == want[k].dtype for k in want)
    # conv weights land fp32 in lag order; the router weight fp32
    a = "model.layers.0.self_attn"
    assert torch.equal(got[f"{a}.k_conv.weight"], stored["model.llm.layers.0.attn.k_sconv.weight"].squeeze(1).flip(-1).float())
    assert torch.equal(got["model.layers.5.self_attn.k_conv.weight"], stored["model.llm.layers.5.attn.k_sconv.static_w"].transpose(0, 1).float())
    assert torch.equal(got["model.layers.0.mlp.gate.weight"], stored["model.llm.layers.0.mlp.gate.weight"].float())
    # the cross-pass queries stack [q[2i], kv[2i], kv[2i+1], q[2i+1]] per core layer
    q = stored["model.llm.middle_cycle_cross_pass_queries"]
    kv = stored["model.llm.middle_cycle_cross_pass_kv_queries"]
    assert torch.equal(got["model.layers.2.loop_msg_query"], torch.stack([q[0], kv[0], kv[1], q[1]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mxfp8 dequant kernel")
def test_expert_pieces_split_nvfp4_and_dequantize_mxfp8(tmp_path):
    stored = _checkpoint(tmp_path)
    hf = cached_load_hf_config(str(tmp_path))
    config = parse_config(hf)
    nvfp4 = list(iter_expert_pieces(str(tmp_path), config, QuantKind.NVFP4))
    assert len(nvfp4) == len(NVFP4_LAYERS) * E
    layer, e0, e1, piece = nvfp4[0]
    assert (layer, e0, e1) == (0, 0, 1)
    ex = "model.llm.layers.0.mlp.experts"
    # the checkpoint stores gate|up interleaved; the pieces deinterleave into halves
    assert torch.equal(piece["gate"][0], stored[f"{ex}.w13_weight"][0, 0::2])
    assert torch.equal(piece["up"][0], stored[f"{ex}.w13_weight"][0, 1::2])
    assert torch.equal(piece["down_scale"][0], stored[f"{ex}.w2_weight_scale"][0])
    want_global = (stored[f"{ex}.w13_weight_amax"][0] / 2688.0).float()
    assert piece["gate_global"].item() == pytest.approx(want_global.item())
    # the MXFP8 layers come through as bf16 gate_up / down, dequantized with the ue8m0 codes
    bf16 = list(iter_expert_pieces(str(tmp_path), config, QuantKind.NONE))
    assert len(bf16) == len(MXFP8_LAYERS) * E
    from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant

    layer, e0, e1, piece = bf16[0]
    assert layer == MXFP8_LAYERS[0]
    ex = f"model.llm.layers.{layer}.mlp.experts"
    deq = mxfp8_dequant(stored[f"{ex}.w13_weight"][0], stored[f"{ex}.w13_weight_scale"][0], dtype=torch.bfloat16)
    want = torch.cat([deq[0::2], deq[1::2]], dim=0)
    assert torch.equal(piece["gate_up"][0], want)
    # other kinds are not claimed by the family hook
    assert iter_expert_pieces(str(tmp_path), config, QuantKind.FP8_TENSOR) is None
