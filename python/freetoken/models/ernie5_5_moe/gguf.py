"""ERNIE 5.5 from a Strata-convention GGUF (HF-name-preserving, no llama.cpp).

Unlike the llama.cpp-converted families, this GGUF keeps the checkpoint's HF
tensor names (``model.layers.N.self_attn.linear_q_up_proj.weight`` ...) and the
HF config as a JSON metadata blob, the way Strata's own reader/writer
(``terminal/tools/gguf_reader.py`` / ``gguf_writer.py``) carries an artifact:
no converter transforms to undo (no zero-centered ``w+1`` norms, no tiled
heads, no interleaved expert rows). Dense projections are stored dequantized
(BF16); the routed MXFP4 experts ride the offload GGUF banks. The Strata reader
is the reference for the quant types gguf-py cannot represent (MXFP4 = 39).

The name map is therefore the identity on HF names: ``to_gguf`` appends the
role suffix, and the dense reader yields the same model buffers the HF
``iter_weights`` yields, reading BF16 rows straight from the GGUF index.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Iterator

import torch

from freetoken.models.config import ModelConfig
from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import float_tensor, matrix

from .args import load_args
from .config import model_config_from_args

_ARCH = "ernie5_5_moe"
# The HF config.json travels as one JSON string KV; parse_gguf_config rebuilds
# the args from it exactly as the HF path does (single source of geometry).
HF_CONFIG_KV = f"{_ARCH}.hf_config"


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors that store it. HF-name-preserving:
    the module's own HF leaf plus ``.weight`` (and the block scale companion when the
    source carries one). Strata convention keeps the names, so this is the identity."""
    out = []
    for suffix in (".weight", ".weight_scale_inv", ".bias"):
        if f"{name}{suffix}" in tensors or name.endswith(suffix):
            out.append(name if name.endswith(suffix) else f"{name}{suffix}")
    if not out and name in tensors:
        out.append(name)
    return tuple(dict.fromkeys(out))


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    md = shim.metadata
    raw = md.get(HF_CONFIG_KV)
    if raw is None:
        raise ValueError(
            f"{shim.model_path}: ERNIE 5.5 GGUF lacks {HF_CONFIG_KV!r} (the HF config JSON); "
            "reconvert with the Strata-convention writer"
        )
    hf = json.loads(raw) if isinstance(raw, str) else dict(raw)
    text = hf.get("text_config") if isinstance(hf.get("text_config"), dict) else hf
    args = load_args(SimpleNamespace(**text), max_batch_size=1)
    max_position = int(text.get("max_position_embeddings", 0) or 0) or int(args.original_seq_len * args.rope_factor)
    config = model_config_from_args(args, max_position=max_position)
    # The Strata-convention GGUF carries the HF quantization_config (or none for a
    # dequantized BF16 artifact): the expert method follows it, so the offload
    # dispatch reads the HF-name expert tensors via iter_expert_pieces, not the
    # llama.cpp-name blk.N.ffn_*_exps loader.
    from dataclasses import replace
    if not text.get("quantization_config"):
        config = replace(config, expert_quant="none")
    return config


class _GgufReader:
    """Duck-type of the HF ``_ShardReader`` over the GGUF index: ``get`` returns a
    dequantized torch tensor in checkpoint orientation, ``has`` tests presence."""

    def __init__(self, model_path: str):
        self._index = gguf_tensor_index(model_path)

    def has(self, name: str) -> bool:
        return name in self._index

    def get(self, name: str) -> torch.Tensor:
        info = self._index[name]
        return matrix(info)

    def close(self) -> None:
        pass


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense state dict from the Strata-convention GGUF (BF16, HF names). Routed
    experts load as offload GGUF banks, so ``include_moe_experts`` must be False
    (the HF path's contract)."""
    if include_moe_experts:
        raise ValueError(
            "ERNIE 5.5 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return
    from .weight import _iter_dense_from_reader

    shim = _shim(model_path)
    args = load_args(SimpleNamespace(**_hf_text(shim)), max_batch_size=1)
    reader = _GgufReader(model_path)
    try:
        yield from _iter_dense_from_reader(reader, args)
    finally:
        reader.close()


def _shim(model_path: str) -> GgufConfigShim:
    from freetoken.models.gguf.config import build_gguf_shim
    return build_gguf_shim(model_path)


def _hf_text(shim: GgufConfigShim) -> dict:
    raw = shim.metadata.get(HF_CONFIG_KV)
    hf = json.loads(raw) if isinstance(raw, str) else dict(raw)
    return hf.get("text_config") if isinstance(hf.get("text_config"), dict) else hf


__all__ = ["to_gguf", "parse_gguf_config", "iter_gguf_weights"]
