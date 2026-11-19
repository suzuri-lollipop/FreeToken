"""Weight loading for Hy4-flash.

Dense tensors stream through under their checkpoint names (the model's state dict
mirrors them); the router bias is remapped ``mlp.gate.e_score_correction_bias ->
mlp.e_score_correction_bias``. Skipped: the routed experts (NVFP4 offload banks or
the stacked bf16 path), the engram embedding tables (``load_host_tables`` streams
them from disk), the DFly2 draft layers and the vision tower (text-only serving).

The NVFP4 release stores experts PER-EXPERT in the ModelOpt layout
(``weight`` uint8 packed | ``weight_scale`` fp8 block | ``weight_scale_2`` fp32
dequant-side global); the FP8 source's stacked block-fp8 experts are dequantized
to bf16 by the conversion script, not served directly.
"""

from __future__ import annotations

import re
from typing import Iterator

import torch
from freetoken.models.loader import ShardReader, safetensors_weight_map
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config

_EXPERT_KEY_RE = (
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_INFIX = ".mlp.experts."
_TABLE_SUFFIX = ".engram_layer.multi_head_embedding.embedding.weight"
_SKIP_PREFIXES = ("model.dfly_layers.", "vit.")


def _rename(name: str) -> str:
    if name.endswith(".mlp.gate.e_score_correction_bias"):
        return name.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
    return name


def _tp_cut(name: str, tensor: torch.Tensor, config) -> torch.Tensor:
    """This rank's slice of a sharded buffer; everything else stays replicated.

    TP shards the attention heads (q_b/kv_b/linear_gate rows, o_proj cols, sinks),
    the dense/shared-expert intermediate, and the vocab. The latent KV projections,
    router, indexer, engram, iHC and norms replicate: every rank must select the
    same tokens and carry the same residual streams. The routed experts replicate
    here too -- the offload bank pack slices their intermediate per rank.
    """
    from freetoken.distributed import get_tp_info
    from freetoken.models.tp_shard import COLS, ROWS, TpShard

    tp = get_tp_info()
    if tp.size == 1:
        return tensor
    args = config.hy4_args
    parts = name.split(".")
    leaf = axis = units = None
    if parts[0] == "model" and parts[1] == "layers":
        L = int(parts[2])
        heads = args.heads_of(L)
        mod = ".".join(parts[3:-1]) if parts[-1] in ("weight", "weight_scale_inv") else ".".join(parts[3:])
        if mod == "self_attn.q_b_proj":
            leaf, axis, units = mod, ROWS, heads * args.qk_head_dim
        elif mod == "self_attn.kv_b_proj":
            leaf, axis, units = mod, ROWS, heads * (args.qk_nope_head_dim + args.v_head_dim)
        elif mod == "self_attn.linear_gate":
            leaf, axis, units = mod, ROWS, heads * args.v_head_dim
        elif mod == "self_attn.o_proj":
            leaf, axis, units = mod, COLS, heads * args.v_head_dim
        elif mod == "self_attn.learnable_sink_param":
            leaf, axis, units = mod, ROWS, heads
        elif mod == "mlp.shared_experts.gate_proj" or mod == "mlp.shared_experts.up_proj":
            leaf, axis, units = mod, ROWS, args.moe_inter_dim * args.n_shared_experts
        elif mod == "mlp.shared_experts.down_proj":
            leaf, axis, units = mod, COLS, args.moe_inter_dim * args.n_shared_experts
        elif L < args.first_k_dense_replace and mod in ("mlp.gate_proj", "mlp.up_proj"):
            leaf, axis, units = mod, ROWS, args.intermediate_size
        elif L < args.first_k_dense_replace and mod == "mlp.down_proj":
            leaf, axis, units = mod, COLS, args.intermediate_size
    elif name in ("lm_head.weight", "model.embed_tokens.weight"):
        leaf = "lm_head" if name.startswith("lm_head") else "embed_tokens"
        axis, units = ROWS, args.vocab_size
    if leaf is None:
        return tensor
    return TpShard(tp.rank, tp.size, {leaf: axis}, {leaf: units}).tensor(leaf, tensor)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    if config.expert_quant == "nvfp4":
        assert not include_moe_experts, (
            "Hy4-flash NVFP4 experts serve from the offload banks; the dense reader never reads them"
        )
    reader = ShardReader(model_path, device)
    try:
        names = []
        for n in weight_map:
            if n.startswith(_SKIP_PREFIXES) or n.endswith(_TABLE_SUFFIX):
                continue
            is_expert = _EXPERT_INFIX in n
            if is_expert and not include_moe_experts:
                continue
            if not is_expert and not include_non_moe:
                continue
            names.append(n)
        for name in tqdm(names, desc="Loading Hy4-flash weights"):
            tensor = reader.get_tensor(name)
            if _EXPERT_INFIX in name:
                # stacked bf16 experts (unquantized release / test fixture): the generic
                # bf16 bank reader consumes them under their packed names
                assert tensor.dtype == torch.bfloat16, f"{name}: stacked experts must be bf16, got {tensor.dtype}"
                yield name, tensor
                continue
            yield _rename(name), _tp_cut(name, tensor, config)
    finally:
        reader.close()


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    dense = config.first_k_dense_replace
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, cfg: layer - dense if layer >= dense else None,
        desc="Hy4-flash NVFP4 experts",
    )


# ======================================================================================
# Block-FP8 routed experts (tencent/Hy4-flash-FP8): stacked per layer, one piece per
# layer carrying the fused gate_up / down codes and their 128x128 block scales.
# ======================================================================================

_FP8_STACKED_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)(?P<scale>_scale_inv)?$"
)


def iter_expert_pieces(model_path, config, kind, *, layers=None, parallel: bool = False, workers: int = 8, chunk: int = 8 << 20):
    """Stacked block-fp8 experts as one piece per MoE layer: ``gate_up`` / ``down``
    fp8 codes plus ``gate_up_scale`` / ``down_scale`` (the checkpoint's
    ``*_scale_inv``). Other expert kinds use the generic readers."""
    from freetoken.layers.quantization import QuantKind, get_quant_config

    if kind is not QuantKind.FP8_BLOCK:
        return None
    L, E, dense = config.num_moe_layers, config.num_experts, config.first_k_dense_replace
    keep = None if layers is None else set(layers)
    scale_suffix = get_quant_config().stored_tensors(QuantKind.FP8_BLOCK)["weight_scale_inv"].name[len("weight"):]

    def locate(name):
        m = _FP8_STACKED_RE.match(name)
        if m is None:
            return None
        li = int(m["layer"]) - dense
        if not 0 <= li < L:
            raise ValueError(f"unexpected routed-expert layer in {name}")
        if keep is not None and li not in keep:
            return None
        role = "gate_up" if m["proj"] == "gate_up_proj" else "down"
        return li, role + ("_scale" if m["scale"] else "")

    def _pieces(tensors):
        pending: dict[int, dict] = {}
        for name, tensor in tensors:
            where = locate(name)
            if where is None:
                continue
            li, role = where
            piece = pending.setdefault(li, {})
            piece[role] = tensor
            if len(piece) == 4:
                del pending[li]
                yield li, 0, E, piece
        if pending:
            raise ValueError(f"incomplete fp8 expert layers: {sorted(pending)}")

    if parallel:
        from freetoken.models.weight import iter_expert_tensors_parallel

        return _pieces(iter_expert_tensors_parallel(model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk))

    def _serial():
        reader = ShardReader(model_path, torch.device("cpu"))
        try:
            for li in range(L):
                if keep is not None and li not in keep:
                    continue
                base = f"model.layers.{dense + li}.mlp.experts"
                for proj in ("gate_up_proj", "down_proj"):
                    yield f"{base}.{proj}", reader.get_tensor(f"{base}.{proj}")
                    yield f"{base}.{proj}{scale_suffix}", reader.get_tensor(f"{base}.{proj}{scale_suffix}")
        finally:
            reader.close()

    return _pieces(_serial())


__all__ = ["iter_weights", "nvfp4_expert_spec", "iter_expert_pieces"]
