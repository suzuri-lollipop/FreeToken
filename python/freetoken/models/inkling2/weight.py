"""Inkling-2-Mini weight reader: bf16 dense pass plus per-format expert pieces.

The checkpoint (modelopt "inkling2" sidecar) quantizes ONLY the routed experts: layers
0-19 NVFP4 (packed e2m1 + fp8 block scales + per-expert amax / 2688 global), layers 20-23
MXFP8 (fp8 + ue8m0 block-32 scales, served dequantized to bf16). Everything else -- the
projections, router, controllers, queries, convs, vision tower -- is bf16; the conv
weights are upcast to fp32 and flipped into lag order, the router weight to fp32.

The dense pass renames ``model.llm.*`` / ``model.visual.*`` onto the module tree; the
expert banks are filled through ``iter_expert_pieces`` (the offload family's hook), never
through this pass. The MTP drafter (mtp.safetensors) is not a target weight;
``--speculative-algorithm dspark`` loads it on its own (freetoken.models.inkling2.draft).
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import drop_page_cache, safetensors_weight_map
from freetoken.utils import download_hf_weight
from freetoken.utils.hf import optional_hf_file
from tqdm import tqdm

_EXPERT_LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.")
_ATTN_LEAVES = {
    "wq_du": "q_proj", "wk_dv": "k_proj", "wv_dv": "v_proj", "wo_ud": "o_proj",
    "wr_du": "r_proj", "gate_attn": "gate_proj", "attn_sink_proj": "sink_proj",
}
_CTRL_KINDS = {"attn": "0", "k": "1", "v": "2", "mlp": "3"}


def _quant_layer_sets(model_path: str) -> tuple[set[int], set[int]]:
    """(nvfp4 layers, mxfp8 layers) from the checkpoint's hf_quant_config.json groups."""
    sidecar = optional_hf_file(model_path, "hf_quant_config.json")
    if sidecar is None:
        raise ValueError("inkling2_loop: the checkpoint ships no hf_quant_config.json")
    with open(sidecar, encoding="utf-8") as f:
        groups = json.load(f)["model"]["parameter_groups"]
    nvfp4: set[int] = set()
    mxfp8: set[int] = set()
    for group in groups:
        dtype = group.get("compute_dtype")
        if dtype not in ("nvfp4", "mxfp8"):
            continue
        target = nvfp4 if dtype == "nvfp4" else mxfp8
        for name in group.get("tensors", []):
            match = _EXPERT_LAYER.match(name)
            if match:
                target.add(int(match.group(1)))
    return nvfp4, mxfp8


def _conv_lag(tensor: torch.Tensor, *, static_w: bool) -> torch.Tensor:
    """Checkpoint conv weight -> fp32 lag order [C, K] (lag 0 weights the current token)."""
    if static_w:
        # full-attention static_w [K, C] is already indexed by backward distance
        return tensor.transpose(0, 1).float().contiguous()
    # conv1d [C, 1, K]: index 0 is the oldest tap (lag K-1)
    return tensor.squeeze(1).flip(-1).float().contiguous()


def _dense_rename(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    if name.startswith("model.visual."):
        out = name[len("model.") :]
        out = out.replace(".ls_1.gamma", ".ls_1_gamma").replace(".ls_2.gamma", ".ls_2_gamma")
        if out.endswith(".position_embedding"):
            out += ".weight"
        return out, tensor
    if not name.startswith("model.llm."):
        return None
    rest = name[len("model.llm.") :]
    if rest == "embed.weight":
        return "model.embed_tokens.weight", tensor
    if rest == "embed_norm.weight":
        return "model.embed_norm.weight", tensor
    if rest == "norm.weight":
        return "model.norm.weight", tensor
    if rest == "unembed.weight":
        return "lm_head.weight", tensor
    if rest == "attn_res.output_query":
        return "model.output_res_query", tensor.unsqueeze(0)
    if rest.startswith("attn_res.attn_queries."):
        layer = int(rest.rsplit(".", 1)[1]) + 1
        return f"model.layers.{layer}.attn_res_query", tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
    if rest.startswith("attn_res.mlp_queries."):
        layer = int(rest.rsplit(".", 1)[1])
        return f"model.layers.{layer}.ffn_res_query", tensor.unsqueeze(0)
    if rest.startswith("layers."):
        parts = rest.split(".")
        layer = parts[1]
        leaf = ".".join(parts[2:])
        pre = f"model.layers.{layer}"
        if leaf.startswith("attn."):
            sub = leaf[len("attn.") :]
            if sub.endswith(".weight") and sub[: -len(".weight")] in _ATTN_LEAVES:
                return f"{pre}.self_attn.{_ATTN_LEAVES[sub[:-len('.weight')]]}.weight", tensor
            if sub == "q_l2_clip.scale":
                return f"{pre}.self_attn.q_clip_scale", tensor
            if sub == "rel_l2_clip.scale":
                return f"{pre}.self_attn.rel_clip_scale", tensor
            if sub == "rel_logits_proj.proj":
                return f"{pre}.self_attn.rel_bias", tensor
            if sub in ("k_sconv.weight", "v_sconv.weight"):
                return f"{pre}.self_attn.{sub[0]}_conv.weight", _conv_lag(tensor, static_w=False)
            if sub in ("k_sconv.static_w", "v_sconv.static_w"):
                return f"{pre}.self_attn.{sub[0]}_conv.weight", _conv_lag(tensor, static_w=True)
            if sub in ("k_sconv.weight_proj", "v_sconv.weight_proj"):
                return f"{pre}.self_attn.{sub[0]}_conv.conv_proj.weight", tensor
            raise ValueError(f"inkling2: unknown attention tensor {name}")
        if leaf == "attn_norm.weight":
            return f"{pre}.self_attn.input_layernorm.weight", tensor
        if leaf in ("q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"):
            return f"{pre}.self_attn.{leaf}", tensor
        if leaf == "attn_sconv.weight":
            return f"{pre}.self_attn.attn_post_conv.weight", _conv_lag(tensor, static_w=False)
        if leaf == "mlp_sconv.weight":
            return f"{pre}.ffn_post_conv.weight", _conv_lag(tensor, static_w=False)
        if leaf == "mlp.gate.weight":
            return f"{pre}.mlp.gate.weight", tensor.float()
        if leaf == "mlp.gate.bias":
            return f"{pre}.mlp.gate_bias", tensor
        if leaf == "mlp_norm.weight":
            return f"{pre}.mlp_norm.weight", tensor
        if leaf.startswith("controllers."):
            _, kind, weight = leaf.split(".")
            return f"{pre}.controllers.{_CTRL_KINDS[kind]}.{weight}", tensor
        raise ValueError(f"inkling2: unknown layer tensor {name}")
    raise ValueError(f"inkling2: unknown tensor {name}")


def iter_weights(
    model_path: str, device: torch.device, *, include_moe_experts: bool, include_non_moe: bool, include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return
    assert not include_moe_experts, (
        "inkling2_loop serves routed experts from the offload banks (NVFP4 / bf16-dequantized MXFP8); "
        "the expert reader is iter_expert_pieces"
    )
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    loop_queries = {
        "model.llm.middle_cycle_cross_pass_queries",
        "model.llm.middle_cycle_cross_pass_kv_queries",
    }
    by_file: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if _EXPERT_LAYER.match(name) or name in loop_queries:
            continue
        if not include_vision and name.startswith("model.visual."):
            continue
        by_file.setdefault(shard, []).append(name)
    for shard in tqdm(sorted(by_file), desc="Loading inkling2 weights"):
        path = os.path.join(folder, shard)
        drop_page_cache(path)
        with safetensors.safe_open(path, framework="pt", device=str(device)) as f:
            for name in by_file[shard]:
                renamed = _dense_rename(name, f.get_tensor(name))
                if renamed is not None:
                    yield renamed
        drop_page_cache(path)
    yield from _loop_query_pieces(folder, weight_map, device, _core_start(model_path))


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it (the FTW encoder-only reader)."""
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    by_file: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith("model.visual."):
            by_file.setdefault(shard, []).append(name)
    for shard in sorted(by_file):
        path = os.path.join(folder, shard)
        with safetensors.safe_open(path, framework="pt", device=str(device)) as f:
            for name in by_file[shard]:
                renamed = _dense_rename(name, f.get_tensor(name))
                if renamed is not None:
                    yield renamed


def _core_start(model_path: str) -> int:
    from freetoken.utils import cached_load_hf_config

    text = getattr(cached_load_hf_config(model_path), "text_config", None)
    loop = getattr(text, "loop_config", None)
    return int(loop["core_start"] if isinstance(loop, dict) else loop.core_start)


def _loop_query_pieces(folder: str, weight_map: dict[str, str], device: torch.device, core_start: int):
    """The cross-pass message queries: core layer start+i mixes q/k/v/mlp with [q[2i], kv[2i], kv[2i+1], q[2i+1]]."""
    q_name = "model.llm.middle_cycle_cross_pass_queries"
    kv_name = "model.llm.middle_cycle_cross_pass_kv_queries"
    with safetensors.safe_open(os.path.join(folder, weight_map[q_name]), framework="pt", device=str(device)) as f:
        q = f.get_tensor(q_name)
        kv = f.get_tensor(kv_name)
    for i in range(q.shape[0] // 2):
        yield f"model.layers.{core_start + i}.loop_msg_query", torch.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])


def iter_expert_pieces(
    model_path: str, config, kind: QuantKind, *, layers=None, parallel: bool = False, workers: int = 8, chunk: int = 8 << 20,
):
    """Expert pieces per format: NVFP4 layers 0-19 as packed codes + scales + amax/2688
    globals; MXFP8 layers 20-23 dequantized to bf16 gate_up/down. None for other kinds."""
    if kind not in (QuantKind.NVFP4, QuantKind.NONE):
        return None
    nvfp4_layers, mxfp8_layers = _quant_layer_sets(model_path)
    wanted = nvfp4_layers if kind is QuantKind.NVFP4 else mxfp8_layers
    keep = wanted if layers is None else (wanted & set(layers))
    if not keep:
        return iter(())
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    num_experts = config.num_experts
    intermediate = config.moe_intermediate_size

    def _gen():
        for layer in sorted(keep):
            base = f"model.llm.layers.{layer}.mlp.experts"
            shard = os.path.join(folder, weight_map[f"{base}.w13_weight"])
            drop_page_cache(shard)
            with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                w13 = f.get_tensor(f"{base}.w13_weight")
                w2 = f.get_tensor(f"{base}.w2_weight")
                if kind is QuantKind.NVFP4:
                    s13 = f.get_tensor(f"{base}.w13_weight_scale")
                    s2 = f.get_tensor(f"{base}.w2_weight_scale")
                    g13 = (f.get_tensor(f"{base}.w13_weight_amax") / f.get_tensor(f"{base}.w13_weight_outer_scale_denominator")).float()
                    g2 = (f.get_tensor(f"{base}.w2_weight_amax") / f.get_tensor(f"{base}.w2_weight_outer_scale_denominator")).float()
                    for e in range(num_experts):
                        # the checkpoint stores gate|up INTERLEAVED rows (g0,u0,g1,u1,...);
                        # the banks read [gate; up] halves (the converter deinterleaves the same way)
                        piece = {
                            "gate": w13[e, 0::2].unsqueeze(0),
                            "up": w13[e, 1::2].unsqueeze(0),
                            "gate_scale": s13[e, 0::2].unsqueeze(0),
                            "up_scale": s13[e, 1::2].unsqueeze(0),
                            "gate_global": g13[e].reshape(1, 1),
                            "up_global": g13[e].reshape(1, 1),
                            "down": w2[e].unsqueeze(0),
                            "down_scale": s2[e].unsqueeze(0),
                            "down_global": g2[e].reshape(1, 1),
                        }
                        yield layer, e, e + 1, piece
                else:
                    from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant

                    s13 = f.get_tensor(f"{base}.w13_weight_scale")
                    s2 = f.get_tensor(f"{base}.w2_weight_scale")
                    w13 = mxfp8_dequant(w13, s13, dtype=torch.bfloat16)
                    w2 = mxfp8_dequant(w2, s2, dtype=torch.bfloat16)
                    for e in range(num_experts):
                        piece = {
                            "gate_up": torch.cat([w13[e, 0::2], w13[e, 1::2]], dim=0).unsqueeze(0),
                            "down": w2[e].unsqueeze(0),
                        }
                        yield layer, e, e + 1, piece
            drop_page_cache(shard)

    return _gen()


__all__ = ["iter_expert_pieces", "iter_weights"]
