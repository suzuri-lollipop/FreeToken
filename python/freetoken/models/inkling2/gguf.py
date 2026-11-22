"""Inkling-2-Mini from a mainline llama.cpp ``inkling2_loop`` GGUF (unsloth's
Mainline_Support files, ggml-org/llama.cpp#32694 + #32704).

The converter interleaves each expert's gate|up rows for llama.cpp's fused glu and
flips the short-conv weights into conv1d order; the loader deinterleaves the banks
(the ggml MoE kernel reads [gate; up] halves) and flips the convs back into lag order.
The relative bias table is stored transposed. An ``inkling2v`` mmproj beside the text
file adds the vision tower; its metadata becomes the HF vision keys.

The root unsloth GGUFs use Unsloth's own looped layout (arch ``inkling2``) and are NOT
this layout; use the Mainline_Support files (rename .gguf_file -> .gguf).
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any, Iterator

import torch

from freetoken.models.gguf.mmproj import Mmproj, find_mmproj, log_mmproj, mmproj_tensor
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, load_gguf_metadata
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix

from .config import IMAGE_TOKEN_ID, parse_config

_ARCH = "inkling2_loop"
_PROJECTOR = "inkling2v"
_IMAGE_TOKEN = "<|unused_200054|>"
_ROPE_THETA = 10000.0

_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.r_proj": "attn_rel",
    "self_attn.gate_proj": "attn_gate",
    "self_attn.sink_proj": "attn_sink_proj",
    "self_attn.k_conv.conv_proj": "attn_k_conv_proj",
    "self_attn.v_conv.conv_proj": "attn_v_conv_proj",
    "mlp.gate": "ffn_gate_inp",
    # the merged gate_up bank answers both the gate and the up probe (the gemma4 precedent)
    "mlp.experts.0.gate_proj": "ffn_gate_up_exps",
    "mlp.experts.0.up_proj": "ffn_gate_up_exps",
    "mlp.experts.0.down_proj": "ffn_down_exps",
}
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_CTRL_KINDS = {"0": "q", "1": "k", "2": "v", "3": "ffn"}


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    if name in _TOP_LEAVES:
        return (_TOP_LEAVES[name],)
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    leaf = _LAYER_LEAVES.get(m.group(2))
    return (f"blk.{m.group(1)}.{leaf}",) if leaf else ()


def _head_count(value) -> int:
    if isinstance(value, list):
        if len(set(value)) != 1:
            raise NotImplementedError("inkling2_loop GGUF: per-layer head counts must match")
        return int(value[0])
    return int(value)


def _hf_config(shim) -> SimpleNamespace:
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    if shim.model_path:
        check_scales(shim.model_path)
    layers = int(kv("block_count"))
    pattern = kv("attention.sliding_window_pattern")
    if isinstance(pattern, list):
        swa = [bool(p) for p in pattern]
    else:
        swa = [i % 6 != 5 for i in range(layers)]
    if len(swa) != layers:
        raise ValueError(f"{shim.model_path}: sliding_window_pattern has {len(swa)} entries for {layers} layers")
    if bool(md.get(f"{_ARCH}.loop.varying", False)):
        raise NotImplementedError("inkling2_loop GGUF: the variable-pass research variant is not served")
    return SimpleNamespace(
        model_type="inkling2_loop",
        architectures=list(shim.architectures),
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=layers,
        num_attention_heads=_head_count(kv("attention.head_count")),
        num_key_value_heads=_head_count(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
        swa_num_attention_heads=_head_count(kv("attention.head_count")),
        swa_num_key_value_heads=_head_count(kv("attention.head_count_kv")),
        d_rel=int(kv("attention.relative_length")),
        rel_extent=int(kv("attention.relative_extent")),
        sliding_window_size=int(kv("attention.sliding_window")),
        qk_l2_clip_threshold=float(kv("attention.qk_clip")),
        rel_l2_clip_threshold=float(kv("attention.relative_clip")),
        log_scaling_n_floor=int(kv("attention.log_scale_floor")),
        log_scaling_alpha=float(kv("attention.log_scale_alpha")),
        sconv_kernel_size=int(kv("shortconv.l_cache")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        local_layer_ids=[i for i, s in enumerate(swa) if s],
        n_layers_per_attn_res_block=int(kv("attn_res.block_size")),
        n_routed_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        n_shared_experts=0,
        intermediate_size=int(kv("expert_feed_forward_length")),
        mlp_gate_softcap=float(kv("activation.situ_beta")),
        mlp_up_softcap=float(kv("activation.situ_linear_beta")),
        route_scale=float(kv("expert_weights_scale")),
        use_gate_bias=True,
        final_logit_softcapping=float(kv("final_logit_softcapping")),
        vocab_size=int(shim.vocab_size),
        unpadded_vocab_size=int(kv("vocab_size_unpadded")),
        loop_config={
            "core_start": int(kv("loop.start")),
            "core_end": int(kv("loop.end")),
            "num_passes": int(kv("loop.count")),
            "gate_rank": int(kv("loop.gate_rank")),
            "is_varloop": False,
        },
        vision_config=getattr(shim, "vision_config", None),
        image_token_id=getattr(shim, "image_token_id", IMAGE_TOKEN_ID),
    )


def parse_gguf_config(shim):
    from dataclasses import replace

    return replace(parse_config(_hf_config(shim)), expert_quant="gguf")


# ----------------------------------------------------------------------------------------------
# Vision tower (an inkling2v mmproj)
# ----------------------------------------------------------------------------------------------

_V_TOP = {
    "v.class_embd": "backbone.embeddings.class_embedding",
    "v.class_position_embd.weight": "backbone.embeddings.cls_position_embedding",
    "v.position_embd.weight": "backbone.embeddings.position_embedding.weight",
    "v.pre_ln.weight": "backbone.ln_pre.weight",
    "v.pre_ln.bias": "backbone.ln_pre.bias",
    "mm.1.weight": "proj_in.weight",
    "mm.2.weight": "proj_out.weight",
}
_V_BLOCK = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ffn_up": "mlp.c_fc",
    "ffn_down": "mlp.c_proj",
    "ln1": "ln_1",
    "ln2": "ln_2",
}


def gguf_mm_config(text_md: dict[str, Any], mmproj: Mmproj, vocab_size: int, *, model_path: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    if mmproj.projector != _PROJECTOR:
        raise ValueError(f"mmproj {mmproj.path} is a {mmproj.projector!r} tower; Inkling-2 needs {_PROJECTOR!r}")
    md, index = load_gguf_metadata(mmproj.path), gguf_tensor_index(mmproj.path)

    def kv(key: str):
        return md[f"clip.vision.{key}"]

    text_hidden = int(text_md[f"{_ARCH}.embedding_length"])
    if int(kv("projection_dim")) != text_hidden:
        raise ValueError(f"mmproj {mmproj.path} projects to {int(kv('projection_dim'))} but the text model is {text_hidden} wide")
    tokens = text_md["tokenizer.ggml.tokens"]
    if _IMAGE_TOKEN not in tokens:
        raise ValueError(f"the text GGUF tokenizer has no {_IMAGE_TOKEN} token")
    hidden, patch, merge = int(kv("embedding_length")), int(kv("patch_size")), int(kv("spatial_merge_size"))
    rows = int(index["v.position_embd.weight"].ne[1])
    side = math.isqrt(rows)
    if side * side != rows:
        raise ValueError(f"mmproj {mmproj.path}: the position table has {rows} rows, not a square")
    vision_config = {
        "hidden_size": hidden,
        "intermediate_size": int(kv("feed_forward_length")),
        "num_hidden_layers": int(kv("block_count")),
        "num_attention_heads": int(kv("attention.head_count")),
        "native_patch_size": patch,
        "adapter_merge_size": merge,
        "posemb_grid_size": side,
        "projection_hidden_dim": int(index["mm.1.weight"].ne[1]),
        "projector_bias": False,
        "text_hidden_size": text_hidden,
        "layer_norm_eps": float(kv("attention.layer_norm_epsilon")),
        "rope_parameters": {"rope_theta": _ROPE_THETA},
    }
    hf_fields = {"vision_config": vision_config, "image_token_id": tokens.index(_IMAGE_TOKEN)}
    preprocessor = {
        "image_processor_type": "Inkling2ImageProcessor",
        "do_convert_rgb": True,
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": True,
        "image_mean": [float(v) for v in kv("image_mean")],
        "image_std": [float(v) for v in kv("image_std")],
        "patch_size": patch,
        "merge_size": merge,
    }
    return hf_fields, preprocessor


def iter_mmproj_weights(mmproj: Mmproj) -> Iterator[tuple[str, torch.Tensor]]:
    index = gguf_tensor_index(mmproj.path)
    n_blocks = 1 + max(int(n.split(".")[2]) for n in index if n.startswith("v.blk."))
    # the converter stores the linear patch weight as [out, c, kh, kw] (ggml ne [kw, kh, c, out]);
    # the tower's linear reads (kh, kw, c) rows, so permute back before flattening
    patch = mmproj_tensor(index["v.patch_embd.weight"])
    yield "visual.backbone.embeddings.patch_embedding.weight", patch.permute(0, 2, 3, 1).flatten(1)
    for gguf_name, name in _V_TOP.items():
        yield f"visual.{name}", mmproj_tensor(index[gguf_name])
    consumed = {"v.patch_embd.weight", *_V_TOP}
    for layer in range(n_blocks):
        blk, dst = f"v.blk.{layer}", f"visual.backbone.layers.{layer}"
        for part in ("weight", "bias"):
            for src, name in _V_BLOCK.items():
                yield f"{dst}.{name}.{part}", mmproj_tensor(index[f"{blk}.{src}.{part}"])
                consumed.add(f"{blk}.{src}.{part}")
        for ls in ("ls1", "ls2"):
            yield f"{dst}.ls_{ls[-1]}_gamma", mmproj_tensor(index[f"{blk}.{ls}.weight"])
            consumed.add(f"{blk}.{ls}.weight")
    leftovers = sorted(set(index) - consumed)
    if leftovers:
        raise ValueError(f"mmproj {mmproj.path}: unexpected tensors {leftovers[:5]}")


def _conv_lag(info: GgufTensorInfo) -> torch.Tensor:
    # the converter stores conv1d order (index 0 = oldest tap); the module reads lag order
    return float_tensor(info).flip(-1).contiguous()


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return
    from freetoken.distributed import info as _di
    from freetoken.models.loader import shard_tensor
    from freetoken.utils import cached_load_hf_config

    import os as _os
    tp = _di.DistributedInfo(rank=0, size=1) if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1" else (_di._TP_INFO if _di._TP_INFO is not None else _di.DistributedInfo(rank=0, size=1))
    hf = _hf_config(cached_load_hf_config(model_path))
    vision = include_vision and hf.vision_config is not None
    log_mmproj(model_path, vision)
    index = gguf_tensor_index(model_path)
    hidden = hf.hidden_size
    core_start = hf.loop_config["core_start"]
    core_end = hf.loop_config["core_end"]

    def tp_shard(name: str, t: torch.Tensor) -> torch.Tensor:
        if tp.size == 1:
            return t
        # standard projections: shard_tensor handles q/k/v/gate/o/embed/lm_head
        if any(sub in name for sub in (".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".o_proj", "lm_head", "embed_tokens")):
            if t.dtype == torch.uint8:
                # fused adds ggml pad rows at the end; strip them, shard the DATA evenly
                # along dim 0 (output rows), then re-add 1 pad row per shard.
                # o_proj shards along dim 1 (input cols) — handle separately.
                if ".o_proj" in name:
                    # row-parallel: split dim 1 (the packed input dimension)
                    sh = shard_tensor(name, t, rank=tp.rank, world_size=tp.size, num_kv_heads=hf.num_key_value_heads)
                    return sh
                data_rows = t.shape[0] - 1  # strip the fused pad
                per = (data_rows + tp.size - 1) // tp.size
                lo = tp.rank * per
                hi = min(lo + per, data_rows)
                sh = t[lo:hi].clone()
                sh = torch.cat([sh, sh.new_zeros(per - sh.shape[0] + 1, *sh.shape[1:])])
                return sh
            return shard_tensor(name, t, rank=tp.rank, world_size=tp.size, num_kv_heads=hf.num_key_value_heads)
        # r_proj, sink_proj: split output dim (per q-head)
        if any(sub in name for sub in (".r_proj", ".sink_proj")):
            if t.dtype == torch.uint8:
                data_rows = t.shape[0] - 1
                per = (data_rows + tp.size - 1) // tp.size
                lo = tp.rank * per
                hi = min(lo + per, data_rows)
                sh = t[lo:hi].clone()
                return torch.cat([sh, sh.new_zeros(per - sh.shape[0] + 1, *sh.shape[1:])])
            return t.chunk(tp.size, dim=0)[tp.rank].clone()
        # per-head clip scales: [nh, hd] or [nh, d_rel]
        if "clip_scale" in name:
            return t.chunk(tp.size, dim=0)[tp.rank].clone()
        # k/v conv weights and conv_proj: split channel dim
        if any(sub in name for sub in (".k_conv.weight", ".v_conv.weight", ".k_conv.conv_proj", ".v_conv.conv_proj")):
            if t.dtype == torch.uint8:
                data_rows = t.shape[0] - 1
                per = (data_rows + tp.size - 1) // tp.size
                lo = tp.rank * per
                hi = min(lo + per, data_rows)
                sh = t[lo:hi].clone()
                return torch.cat([sh, sh.new_zeros(per - sh.shape[0] + 1, *sh.shape[1:])])
            return t.chunk(tp.size, dim=0)[tp.rank].clone()
        return t

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def linear(module: str, gguf_name: str, k: int):
        info = t(gguf_name)
        for name, tensor in fused(module, [info], [matrix(info)], k):
            yield name, tp_shard(name, tensor)

    yield from linear("model.embed_tokens", "token_embd", hidden)
    if "output.weight" in index:
        yield from linear("lm_head", "output", hidden)
    yield "model.embed_norm.weight", float_tensor(t("token_embd_norm"))
    yield "model.norm.weight", float_tensor(t("output_norm"))
    yield "model.output_res_query", float_tensor(t("output_res_query")).unsqueeze(0)

    for p in range(hf.num_hidden_layers):
        blk, pre = f"blk.{p}", f"model.layers.{p}"
        sa = f"{pre}.self_attn"
        swa = p in hf.local_layer_ids
        for leaf in ("q_proj", "k_proj", "v_proj", "o_proj", "r_proj", "gate_proj", "sink_proj"):
            info = t(f"{blk}.{_LAYER_LEAVES[f'self_attn.{leaf}']}")
            yield from linear(f"{sa}.{leaf}", info.name.removesuffix(".weight"), int(info.ne[0]))
        yield f"{sa}.q_clip_scale", tp_shard(f"{sa}.q_clip_scale", float_tensor(t(f"{blk}.attn_q_clip")))
        yield f"{sa}.rel_clip_scale", tp_shard(f"{sa}.rel_clip_scale", float_tensor(t(f"{blk}.attn_rel_clip")))
        yield f"{sa}.rel_bias", float_tensor(t(f"{blk}.attn_rel_bias")).transpose(0, 1).contiguous()
        yield f"{sa}.k_conv.weight", tp_shard(f"{sa}.k_conv.weight", _conv_lag(t(f"{blk}.attn_k_conv")))
        yield f"{sa}.v_conv.weight", tp_shard(f"{sa}.v_conv.weight", _conv_lag(t(f"{blk}.attn_v_conv")))
        yield f"{sa}.attn_post_conv.weight", _conv_lag(t(f"{blk}.attn_post_conv"))
        yield f"{pre}.ffn_post_conv.weight", _conv_lag(t(f"{blk}.ffn_post_conv"))
        if swa:
            yield f"{sa}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        else:
            for qkv in ("q", "k", "v"):
                yield f"{sa}.{qkv}_input_norm.weight", float_tensor(t(f"{blk}.attn_{qkv}_inp_norm"))
            yield from linear(f"{sa}.k_conv.conv_proj", f"{blk}.attn_k_conv_proj", hidden)
            yield from linear(f"{sa}.v_conv.conv_proj", f"{blk}.attn_v_conv_proj", hidden)
        yield f"{pre}.mlp_norm.weight", float_tensor(t(f"{blk}.ffn_norm"))
        yield f"{pre}.mlp.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp")).float()
        yield f"{pre}.mlp.gate_bias", float_tensor(index[f"{blk}.exp_probs_b.bias"]).float()
        if p > 0:
            q = float_tensor(t(f"{blk}.attn_res_query"))
            yield f"{pre}.attn_res_query", q.unsqueeze(0) if q.dim() == 1 else q
        yield f"{pre}.ffn_res_query", float_tensor(t(f"{blk}.ffn_res_query")).unsqueeze(0)
        if core_start <= p < core_end:
            yield f"{pre}.loop_msg_query", float_tensor(t(f"{blk}.loop_msg_query"))
            for idx, kind in _CTRL_KINDS.items():
                for w in ("down", "gain", "forget"):
                    yield f"{pre}.controllers.{idx}.{w}_weight", float_tensor(t(f"{blk}.loop_{kind}_ctrl_{w}"))

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


__all__ = ["gguf_mm_config", "iter_gguf_weights", "iter_mmproj_weights", "parse_gguf_config", "to_gguf"]
