"""Weight loading for ERNIE 5.5 Flash (engine path).

- :func:`iter_weights` streams resident (non-expert) tensors keyed by the model's
  attribute paths. Checkpoint transforms:
  - mHC ``mapping_proj`` / ``hc_head_fn`` are stored ``[in, out]`` -> transposed;
    ``alpha_pre/post/res`` stack into ``hc_{attn,ffn}_scale[3]``.
  - the two split-feature routers (``gate.weight`` / ``gate.weight_1``) stack
    into one fp32 ``[2E, H]`` GEMM weight.
  - ``linear_o_group_proj`` (HCA grouped wo_a) and ``kv_b_proj`` (DSA MLA
    absorption) are block-fp8 in the checkpoint, dequantized to bf16 for the
    bf16 einsum / bmm paths (DSV4 / GLM-DSA precedent).
  - ``multimax_ranges`` / ``multimax_ts`` upcast bf16 -> fp32.
  - the DSA ``softmax_offset`` IS the attention sink.
- :func:`iter_expert_pieces` streams the routed MXFP4 experts (e2m1 pairs +
  e8m0 per-32 scales) for the offload cache's banks; ``model.mtp.*`` (the
  bundled DSpark draft) is skipped -- the draft is not served yet.
- the optional FlashIndexer head-routing gates load from the checkpoint's
  ``flashindexer/`` sidecar safetensors (outside the main weight index).
"""

from __future__ import annotations

import json
import os
import re

import safetensors
import torch
from tqdm import tqdm

from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import drop_page_cache

from .args import Ernie55Args, load_args


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize square-block FP8 (e4m3) with F32 scales to bf16.

    ERNIE's dense ``weight_scale_inv`` are plain floats (the config's
    ``scale_fmt: ue8m0`` describes the expert scales, not these).
    """
    n, k = weight.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _hc_scale(reader: _ShardReader, base: str) -> torch.Tensor:
    return torch.cat(
        [reader.get(f"{base}.alpha_{a}").float() for a in ("pre", "post", "res")]
    )


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
    include_vision: bool = True,
):
    """Stream resident weights as ``(model attribute path, tensor)``.

    Routed MXFP4 experts come from the offload cache, so ``include_moe_experts``
    must be False (ERNIE runs ``--moe-strategy offload``, the DSV4 precedent).
    """
    if include_moe_experts:
        raise ValueError(
            "ERNIE 5.5 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(_hf_config(model_path), max_batch_size=1)
    reader = _ShardReader(model_path, _weight_map(model_path), device)

    try:
        yield from _iter_dense_from_reader(reader, args)
        if args.enable_flashindexer:
            yield from _iter_flashindexer(model_path, args, device)
    finally:
        reader.close()


def _vocab_shard(tensor: torch.Tensor, args) -> torch.Tensor:
    """This rank's vocab rows of the embedding / LM head (VocabParallelEmbedding /
    ParallelLMHead shard the vocab, rounded up like the CEIL_LEAVES cut)."""
    from freetoken.distributed import try_get_tp_info
    tp = try_get_tp_info()
    if tp is None or tp.size <= 1:
        return tensor
    from freetoken.models.tp_shard import ROWS, TpShard
    # materialize the full tensor before the rank cut: the reader may hand back a
    # mmap-backed view whose pages get dropped, and a narrow+clone of the far rank's
    # rows would then read garbage (the TP2 embed NaN).
    return TpShard(tp.rank, tp.size, {"lm_head": ROWS}, {"lm_head": args.vocab_size}).tensor("lm_head", tensor.clone())


def _mlp_shard(name: str, tensor: torch.Tensor, args) -> torch.Tensor:
    """This rank's slice of a SiTU-GLU MLP leaf (dense layer 0 / shared expert): gate/up
    rows, down columns, the block-fp8 scale by its 128 grain. The two MLP kinds have
    different intermediates (dense = intermediate_size, shared = moe_inter_dim*n_shared),
    so the shard is built per instance. Attention / latent projections stay replicated
    (MQA shared KV + VHA cross-head mix); the routed experts shard in the bank pack."""
    from freetoken.distributed import try_get_tp_info
    tp = try_get_tp_info()
    if tp is None or tp.size <= 1:
        return tensor
    if name.endswith(".weight_scale_inv"):
        base = name[: -len(".weight_scale_inv")]
    elif name.endswith(".weight"):
        base = name[: -len(".weight")]
    else:
        return tensor
    leaf = base.rsplit(".", 1)[-1]
    if leaf not in ("gate_proj", "up_proj", "down_proj"):
        return tensor
    if ".shared_experts." in name:
        inter = args.moe_inter_dim * args.n_shared_experts
    elif ".ffn." in name:
        inter = args.intermediate_size
    else:
        return tensor
    from freetoken.models.tp_shard import COLS, ROWS, TpShard
    axis = COLS if leaf == "down_proj" else ROWS
    return TpShard(tp.rank, tp.size, {leaf: axis}, {leaf: inter}).tensor(leaf, tensor)


def _iter_dense_from_reader(reader, args):
    """Dense checkpoint->model name map shared by the HF reader and the Strata-convention
    GGUF reader: both expose ``get``/``has`` over HF-named tensors, so the GGUF path needs
    no converter-transform undo (BF16 rows carry no ``weight_scale_inv``)."""
    def linear(src: str, dst: str):
        yield f"{dst}.weight", _mlp_shard(f"{dst}.weight", reader.get(f"{src}.weight"), args)
        if reader.has(f"{src}.weight_scale_inv"):
            yield f"{dst}.weight_scale_inv", _mlp_shard(f"{dst}.weight_scale_inv", reader.get(f"{src}.weight_scale_inv"), args)

    yield "model.embed.weight", _vocab_shard(reader.get("model.embed_tokens.weight"), args)
    yield "model.norm.weight", reader.get("model.norm.weight")
    yield "model.head.weight", _vocab_shard(reader.get("lm_head.weight"), args)
    yield "model.multimax_ranges", reader.get("model.lm_head.multimax_ranges").float()
    yield "model.multimax_ts", reader.get("model.lm_head.multimax_ts").float()
    yield "model.hc_head_fn", reader.get("model.mhc_contract.hc_head_fn").t().contiguous()
    yield "model.hc_head_base", reader.get("model.mhc_contract.hc_head_base")
    yield "model.hc_head_scale", reader.get("model.mhc_contract.hc_head_scale")

    for L in range(args.n_layers):
        ck = f"model.layers.{L}"
        m = f"model.layers.{L}"
        yield f"{m}.attn_norm.weight", reader.get(f"{ck}.input_layernorm.weight")
        yield f"{m}.ffn_norm.weight", reader.get(f"{ck}.post_attention_layernorm.weight")
        for kind, hc in (("attn", "self_attention"), ("ffn", "mlp")):
            base = f"{ck}.{hc}_hyper_connection"
            yield f"{m}.hc_{kind}_fn", reader.get(f"{base}.mapping_proj.weight").t().contiguous()
            yield f"{m}.hc_{kind}_base", reader.get(f"{base}.bias")
            yield f"{m}.hc_{kind}_scale", _hc_scale(reader, base)

        a = f"{ck}.self_attn"
        ma = f"{m}.attn"
        yield f"{ma}.vha.vha_U", reader.get(f"{a}.vha_postmix_U")
        yield f"{ma}.vha.vha_V", reader.get(f"{a}.vha_postmix_V")
        yield from linear(f"{a}.gate_proj", f"{ma}.o_gate")
        if args.is_dsa_layer(L):
            yield from linear(f"{a}.q_a_proj", f"{ma}.q_a_proj")
            yield f"{ma}.q_a_layernorm.weight", reader.get(f"{a}.q_a_layernorm.weight")
            yield from linear(f"{a}.q_b_proj", f"{ma}.q_b_proj")
            # kv_a: pad the weight rows to the block-fp8 multiple of 128 (the
            # checkpoint's scale rows are already padded; the model slices back)
            kv_a_w = reader.get(f"{a}.kv_a_proj_with_mqa.weight")
            pad = -(-kv_a_w.shape[0] // 128) * 128 - kv_a_w.shape[0]
            if pad:
                kv_a_w = torch.nn.functional.pad(kv_a_w, (0, 0, 0, pad))
            yield f"{ma}.kv_a_proj_with_mqa.weight", kv_a_w
            if reader.has(f"{a}.kv_a_proj_with_mqa.weight_scale_inv"):
                yield f"{ma}.kv_a_proj_with_mqa.weight_scale_inv", reader.get(f"{a}.kv_a_proj_with_mqa.weight_scale_inv")
            yield f"{ma}.kv_a_layernorm.weight", reader.get(f"{a}.kv_a_layernorm.weight")
            # MLA absorption reads kv_b as bf16 bmm operands (GLM-DSA precedent);
            # a bf16 checkpoint stores it unquantized already
            if reader.has(f"{a}.kv_b_proj.weight_scale_inv"):
                yield f"{ma}.kv_b_proj.weight", _dequant_fp8_block(
                    reader.get(f"{a}.kv_b_proj.weight"),
                    reader.get(f"{a}.kv_b_proj.weight_scale_inv"),
                )
            else:
                yield f"{ma}.kv_b_proj.weight", reader.get(f"{a}.kv_b_proj.weight")
            yield from linear(f"{a}.o_proj", f"{ma}.o_proj")
            # the checkpoint's softmax_offset is the per-head attention sink
            yield f"{ma}.attn_sink", reader.get(f"{a}.core_attention.softmax_offset").float()
            idx = f"{a}.core_attention.indexer"
            yield from linear(f"{idx}.wq_b", f"{ma}.indexer.wq_b")
            yield f"{ma}.indexer.wk.weight", reader.get(f"{idx}.wk.weight")
            yield f"{ma}.indexer.k_norm.weight", reader.get(f"{idx}.k_norm.weight")
            yield f"{ma}.indexer.k_norm.bias", reader.get(f"{idx}.k_norm.bias")
            yield f"{ma}.indexer.weights_proj.weight", reader.get(f"{idx}.weights_proj.weight")
        else:
            yield from linear(f"{a}.linear_q_down_proj", f"{ma}.wq_a")
            yield f"{ma}.q_norm.weight", reader.get(f"{a}.q_layernorm.weight")
            yield from linear(f"{a}.linear_q_up_proj", f"{ma}.wq_b")
            yield from linear(f"{a}.linear_kv_proj", f"{ma}.wkv")
            yield f"{ma}.kv_norm.weight", reader.get(f"{a}.kv_layernorm.weight")
            # grouped wo_a: fp8 in the checkpoint, bf16 einsum at runtime
            # (the tensor carries no .weight suffix; bf16 checkpoints store it raw)
            if reader.has(f"{a}.linear_o_group_proj_scale_inv"):
                yield f"{ma}.wo_a", _dequant_fp8_block(
                    reader.get(f"{a}.linear_o_group_proj"),
                    reader.get(f"{a}.linear_o_group_proj_scale_inv"),
                )
            else:
                yield f"{ma}.wo_a", reader.get(f"{a}.linear_o_group_proj")
            yield from linear(f"{a}.o_proj", f"{ma}.wo_b")
            yield f"{ma}.attn_sink", reader.get(f"{a}.core_attention.attn_sink")
            c = f"{a}.core_attention.compressor"
            yield f"{ma}.compressor.ape", reader.get(f"{c}.ape")
            yield f"{ma}.compressor.wkv.weight", reader.get(f"{c}.linear_wkv.weight")
            yield f"{ma}.compressor.wgate.weight", reader.get(f"{c}.linear_wgate.weight")
            yield f"{ma}.compressor.norm.weight", reader.get(f"{c}.norm.weight")

        if L < args.first_k_dense_replace:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield from linear(f"{ck}.mlp.{proj}", f"{m}.ffn.{proj}")
        else:
            g = f"{ck}.block_sparse_moe"
            mg = f"{m}.ffn"
            yield f"{mg}.gate_weight", torch.cat(
                [reader.get(f"{g}.gate.weight"), reader.get(f"{g}.gate.weight_1")]
            ).float()
            yield f"{mg}.e_score_correction_bias", reader.get(f"{g}.e_score_correction_bias")
            yield f"{mg}.routed_scaling_factor_param", reader.get(f"{g}.gate.routed_scaling_factor_param")
            yield from linear(f"{g}.fc1_latent_proj", f"{mg}.fc1_latent_proj")
            yield from linear(f"{g}.fc2_latent_proj", f"{mg}.fc2_latent_proj")
            yield f"{mg}.latent_norm.weight", reader.get(f"{g}.latent_norm.weight")
            for proj, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                yield from linear(f"{g}.shared_experts.{proj}", f"{mg}.shared_experts.{dst}")


def _hf_config(model_path: str):
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def _iter_flashindexer(model_path: str, args: Ernie55Args, device):
    """The head-routing gates from the ``flashindexer/`` sidecar (own safetensors,
    outside the main index). Missing sidecar -> loud error, never a random gate."""
    path = os.path.join(model_path, "flashindexer", "flashindexer.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"enable_flashindexer is on but {path} is missing; drop the override "
            "to score with all indexer heads"
        )
    with safetensors.safe_open(path, framework="pt", device=str(device)) as f:
        for L in args.dsa_layer_ids:
            base = f"model.layers.{L}.self_attn.core_attention.indexer"
            yield f"model.layers.{L}.attn.indexer.linear_gate.weight", f.get_tensor(f"{base}.linear_gate.weight")
            yield f"model.layers.{L}.attn.indexer.flashindexer_bias", f.get_tensor(f"{base}.flashindexer_bias").float()


# --------------------------------------------------------------------------------------
# Routed MXFP4 expert pieces (offload-cache banks).
# --------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|weight_scale_inv)$"
)
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_KIND_SUFFIX = {"weight": "", "weight_scale_inv": "_scale"}


def iter_expert_pieces(model_path: str, config, kind: QuantKind, *, layers=None, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Routed experts, one piece per expert: ``{gate, up, down}`` e2m1 pairs and
    their e8m0 ``_scale`` companions (MXFP4), or plain bf16 weights (the resident
    parity path). The bundled DSpark draft (``model.mtp.*``) and the dense layer 0
    carry no routed experts and are skipped."""
    if kind not in (QuantKind.MXFP4, QuantKind.NONE):
        return None
    # The reader hands over FULL experts at any TP: the MXFP4 kernel's pack banks this
    # rank's intermediate slice (mxfp4.py layout/pack), and the MoE layer all-reduces the
    # per-rank down partials. No reader-side shard here.
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import bank_layer_of, per_expert_pieces
    from freetoken.models.gguf.reader import is_gguf_path

    if is_gguf_path(model_path):
        return _iter_gguf_expert_pieces(model_path, config, kind, layers=layers)

    args = load_args(_hf_config(model_path), max_batch_size=1)
    moe_layers = list(range(args.first_k_dense_replace, args.n_layers))
    keep = None if layers is None else set(layers)
    bf16 = kind is QuantKind.NONE

    def locate(raw_name: str):
        m = _EXPERT_RE.match(raw_name)
        if m is None or int(m["layer"]) not in moe_layers:
            return None
        bank_layer = bank_layer_of(config, int(m["layer"]))
        if bank_layer is None or (keep is not None and bank_layer not in keep):
            return None
        if bf16 and m["kind"] != "weight":
            return None
        return bank_layer, int(m["expert"]), _PROJ_ROLE[m["proj"]] + ("" if bf16 else _KIND_SUFFIX[m["kind"]])

    if parallel and not bf16:
        tensors = iter_expert_tensors_parallel(model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk)
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = _ShardReader(model_path, _weight_map(model_path), torch.device("cpu"))
        try:
            for li in tqdm(moe_layers if keep is None else sorted(keep), desc="Loading ERNIE 5.5 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(args.n_routed_experts):
                    base = f"model.layers.{li}.block_sparse_moe.experts.{e}"
                    for proj in ("w1", "w3", "w2"):
                        kinds = ("weight",) if bf16 else ("weight", "weight_scale_inv")
                        for kind_ in kinds:
                            name = f"{base}.{proj}.{kind_}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=3 if bf16 else 6)


class _GgufExpertReader:
    """HF-name expert tensors from a Strata-convention GGUF: dequantized BF16
    experts (stored F32/BF16) read as floats; raw packed MXFP4 / e8m0 scale rows
    (stored I8) read as their raw bytes, the same layout the HF checkpoint ships."""

    def __init__(self, model_path: str):
        from freetoken.models.gguf.reader import gguf_tensor_index
        self._index = gguf_tensor_index(model_path)

    def get(self, name: str) -> torch.Tensor:
        from freetoken.models.gguf.tensors import float_tensor
        info = self._index[name]
        if info.ggml_type in (0, 1, 30):      # F32 / F16 / BF16
            return float_tensor(info)
        raw = torch.from_numpy(info.data().copy()).view(*reversed(info.ne))
        # raw I8 rows: the e2m1 packed weight is int8, the e8m0 scale companion is
        # float8_e8m0fnu -- the same dtypes the HF checkpoint ships, so the ds_fp4
        # kernel reads the identical layout.
        if name.endswith("weight_scale_inv"):
            return raw.view(torch.float8_e8m0fnu)
        return raw.view(torch.int8)

    def close(self) -> None:
        pass


def _iter_gguf_expert_pieces(model_path, config, kind, *, layers=None):
    """Strata-convention GGUF experts under HF names (no llama.cpp blk.N.ffn_*_exps):
    the same per-expert pieces the HF path yields, read from the GGUF index."""
    from freetoken.moe.expert_pieces import bank_layer_of, per_expert_pieces

    args = config.dsv4_args
    moe_layers = list(range(args.first_k_dense_replace, args.n_layers))
    keep = None if layers is None else set(layers)
    bf16 = kind is QuantKind.NONE
    reader = _GgufExpertReader(model_path)

    def locate(raw_name: str):
        m = _EXPERT_RE.match(raw_name)
        if m is None or int(m["layer"]) not in moe_layers:
            return None
        bank_layer = bank_layer_of(config, int(m["layer"]))
        if bank_layer is None or (keep is not None and bank_layer not in keep):
            return None
        if bf16 and m["kind"] != "weight":
            return None
        return bank_layer, int(m["expert"]), _PROJ_ROLE[m["proj"]] + ("" if bf16 else _KIND_SUFFIX[m["kind"]])

    def _serial():
        for li in moe_layers if keep is None else sorted(keep):
            for e in range(args.n_routed_experts):
                base = f"model.layers.{li}.block_sparse_moe.experts.{e}"
                for proj in ("w1", "w3", "w2"):
                    kinds = ("weight",) if bf16 else ("weight", "weight_scale_inv")
                    for kind_ in kinds:
                        name = f"{base}.{proj}.{kind_}"
                        yield name, reader.get(name)

    return per_expert_pieces(_serial(), locate, tensors_per_expert=3 if bf16 else 6)


__all__ = ["iter_weights", "iter_expert_pieces"]
