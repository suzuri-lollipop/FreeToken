"""Convert tencent/Hy4-flash-FP8 to an NVFP4 checkpoint FreeToken serves.

The routed experts (stacked 128x128 block-fp8 in the source) become per-expert
ModelOpt NVFP4: packed e2m1 pairs (low nibble first), fp8-e4m3 scales per 16 input
elements, one fp32 dequant-side global per projection (``weight_scale_2``). The
block-fp8 attention projections are dequantized to bf16 (the MLA absorption needs
an unquantized kv_b anyway, and the engine's dense path reads bf16). Engram tables,
norms, iHC parameters, the router and the vision tower copy through unchanged; the
DFly2 draft layers are dropped (not served).

Usage:
    python scripts/convert_hy4_fp8_to_nvfp4.py \
        --source ~/models/tencent--Hy4-flash-FP8 \
        --out ~/models/tencent--Hy4-flash-NVFP4 [--shard-gib 8] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

FP8_MAX = 448.0
E2M1_MAX = 6.0
# e2m1 magnitudes by code (0..7); code 8..15 are the negatives
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round to the e2m1 grid, ties to even code; returns the signed 4-bit codes."""
    mag = x.abs()
    mid = E2M1_MIDPOINTS.to(mag.device)
    code = torch.searchsorted(mid, mag, right=True)
    # exact midpoints round to the EVEN code (searchsorted right lands on the upper)
    tie = (code > 0) & (mag == mid[(code - 1).clamp_min(0)])
    code = torch.where(tie & (code % 2 == 1), code - 1, code)
    return code.to(torch.uint8) | torch.where(x < 0, torch.tensor(8, dtype=torch.uint8, device=x.device), torch.tensor(0, dtype=torch.uint8, device=x.device))


def quantize_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[OUT, IN]`` bf16/fp32 -> (packed uint8 [OUT, IN//2], scale fp8 [OUT, IN//16], global fp32 scalar).

    Dequant convention (the engine's nvfp4 kernels): ``w = e2m1(code) * scale * global``.
    The global is the dequant-side per-tensor scale ``amax / (448 * 6)``; the block
    scale is the e4m3 value minimizing the block's post-rounding MSE (chosen among
    the nearest-rounded scale and its e4m3 code neighbors), codes round-to-nearest-even.
    """
    assert w.dim() == 2 and w.shape[1] % 16 == 0, w.shape
    wf = w.float()
    amax = wf.abs().amax().clamp(min=1e-12)
    g = (amax / (FP8_MAX * E2M1_MAX)).reshape(())
    blocks = wf.reshape(w.shape[0], w.shape[1] // 16, 16)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    s0 = (block_amax / (E2M1_MAX * g)).clamp(max=FP8_MAX)
    sf = s0.to(torch.float8_e4m3fn)
    # e4m3fn positive codes are monotonic: neighbors in code space are the adjacent
    # representable scales (0x7E = 448 is the last finite code)
    c0 = sf.view(torch.uint8).to(torch.int16)
    lut = torch.cat([E2M1_VALUES, -E2M1_VALUES]).to(w.device)
    best_scale = None
    best_err = None
    best_codes = None
    for delta in (0, -1, 1, -2, 2):
        cand = (c0 + delta).clamp(0, 0x7E).to(torch.uint8).view(torch.float8_e4m3fn).float()
        denom = (cand * g).clamp(min=1e-30)
        codes = _e2m1_codes((blocks / denom).clamp(-E2M1_MAX, E2M1_MAX))
        deq = lut[codes.to(torch.int32)] * denom
        err = ((deq - blocks) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_scale, best_err, best_codes = cand, err, codes
        else:
            take = err < best_err
            best_scale = torch.where(take, cand, best_scale)
            best_err = torch.where(take, err, best_err)
            best_codes = torch.where(take, codes, best_codes)
    scale = best_scale.squeeze(-1).to(torch.float8_e4m3fn)
    code = best_codes.reshape(w.shape)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale.contiguous(), g.to(torch.float32)


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Reference dequant for verification (matches kernel/triton/nvfp4_dequant.py)."""
    lut = torch.cat([E2M1_VALUES, -E2M1_VALUES]).to(packed.device)
    codes = packed.to(torch.int32)
    vals = torch.stack([lut[codes & 0xF], lut[(codes >> 4) & 0xF]], dim=-1).flatten(1)
    return vals * scale.float().repeat_interleave(16, dim=1) * g.float()


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor, block: int = 128) -> torch.Tensor:
    """DeepSeek-style block fp8: ``w[i, j] * scale_inv[i // block, j // block]`` -> bf16."""
    wf = w.float()
    if wf.dim() == 3:  # stacked experts [E, OUT, IN]
        e, out, inn = wf.shape
        s = scale_inv.float().repeat_interleave(block, dim=1)[:, :out].repeat_interleave(block, dim=2)[:, :, :inn]
        return (wf * s).to(torch.bfloat16)
    out, inn = wf.shape
    s = scale_inv.float().repeat_interleave(block, dim=0)[:out].repeat_interleave(block, dim=1)[:, :inn]
    return (wf * s).to(torch.bfloat16)


# modelopt ignore list: everything except the per-expert NVFP4 projections.
# The family's checkpoint_roots map queries to the HF wrapper spelling
# (language_model.model.<internal>), so literal / anchored patterns carry both
# spellings; the "*.foo*" globs match either already.
_IGNORE_INTERNAL = [
    "lm_head", "model.embed_tokens", "model.norm", "model.hc_head*",
    "*.self_attn*", "*.hc_attn_layer*", "*.hc_mlp_layer*", "*.engram_layer*",
    "*.mlp.gate*", "*.mlp.shared_experts*", "*.mlp.routed_expert*",
    "*.mlp.combine_output_norm*", "*.mlp.post_fc2_norm*",
    "model.layers.0.mlp*", "*.input_layernorm*", "*.post_attention_layernorm*",
    "vit*", "model.dfly_layers*",
]
IGNORE = _IGNORE_INTERNAL + [
    "language_model.model." + p for p in _IGNORE_INTERNAL if p.startswith(("model.", "lm_head"))
]


def convert(source: str, out: str, *, shard_gib: float = 8.0, device: str = "cuda", max_layers: int | None = None, expert_dtype: str = "nvfp4") -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)
    text = config["text_config"]
    n_layers = text["num_hidden_layers"]
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)
    E = text["n_routed_experts"]
    inter = text["moe_intermediate_size"]
    latent = text.get("routed_expert_hidden_size") or text["hidden_size"]
    dense_layers = text.get("first_k_dense_replace", 1)

    with open(os.path.join(source, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = weight_map[name]
        h = handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(source, shard), framework="pt", device="cpu").__enter__()
            handles[shard] = h
        return h.get_tensor(name)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    shard_limit = int(shard_gib * (1 << 30))
    shard_idx = 0
    shard_tensors: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    new_map: dict[str, str] = {}
    stats = {"experts": 0, "dequant_fp8": 0, "copied": 0, "skipped": 0}

    def emit(name: str, tensor: torch.Tensor) -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        tensor = tensor.contiguous().cpu()
        nbytes = tensor.numel() * tensor.element_size()
        if shard_bytes + nbytes > shard_limit and shard_tensors:
            flush_shard()
        shard_tensors[name] = tensor
        shard_bytes += nbytes
        new_map[name] = f"model-{shard_idx + 1:05d}.safetensors"

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        fname = f"model-{shard_idx + 1:05d}.safetensors"
        save_file(shard_tensors, os.path.join(out, fname), metadata={"format": "pt"})
        shard_idx += 1
        shard_tensors = {}
        shard_bytes = 0

    def quantize_expert(name: str, w: torch.Tensor) -> None:
        packed, scale, g = quantize_nvfp4(w.to(dev))
        emit(name + ".weight", packed)
        emit(name + ".weight_scale", scale)
        emit(name + ".weight_scale_2", g.reshape(1))
        stats["experts"] += 1

    try:
        # ---- routed experts: stacked block-fp8 -> per-expert NVFP4 (or stacked bf16) ----
        # dequantize in expert chunks to bound the fp32 transient on the device
        chunk = 64
        for layer in tqdm(range(dense_layers, n_layers), desc="experts"):
            m = f"model.layers.{layer}.mlp"
            gu_fp8 = get(f"{m}.experts.gate_up_proj")
            gu_s = get(f"{m}.experts.gate_up_proj_scale_inv")
            dn_fp8 = get(f"{m}.experts.down_proj")
            dn_s = get(f"{m}.experts.down_proj_scale_inv")
            gu_parts, dn_parts = [], []
            for e0 in range(0, E, chunk):
                e1 = min(e0 + chunk, E)
                gu = dequant_fp8_block(gu_fp8[e0:e1].to(dev), gu_s[e0:e1].to(dev))
                dn = dequant_fp8_block(dn_fp8[e0:e1].to(dev), dn_s[e0:e1].to(dev))
                if expert_dtype == "bf16":
                    gu_parts.append(gu.cpu())
                    dn_parts.append(dn.cpu())
                else:
                    for e in range(e0, e1):
                        base = f"{m}.experts.{e}"
                        quantize_expert(f"{base}.gate_proj", gu[e - e0, :inter])
                        quantize_expert(f"{base}.up_proj", gu[e - e0, inter:])
                        quantize_expert(f"{base}.down_proj", dn[e - e0])
                del gu, dn
            if expert_dtype == "bf16":
                emit(f"{m}.experts.gate_up_proj", torch.cat(gu_parts))
                emit(f"{m}.experts.down_proj", torch.cat(dn_parts))
        # ---- dense tensors ----
        dense_names = [n for n in weight_map if ".mlp.experts." not in n]
        for name in tqdm(dense_names, desc="dense"):
            if name.startswith("model.dfly_layers."):
                stats["skipped"] += 1
                continue
            parts = name.split(".")
            if parts[0] == "model" and parts[1] == "layers" and int(parts[2]) >= n_layers:
                stats["skipped"] += 1
                continue
            if name.endswith("_scale_inv"):
                stats["skipped"] += 1
                continue  # consumed with its fp8 weight below
            t = get(name)
            if t.dtype == torch.float8_e4m3fn:
                base = name[: -len(".weight")]
                t = dequant_fp8_block(t, get(base + ".weight_scale_inv"))
                stats["dequant_fp8"] += 1
            emit(name, t)
            stats["copied"] += 1
        flush_shard()
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)

    # ---- index + config + tokenizer ----
    total = sum(os.path.getsize(os.path.join(out, f)) for f in set(new_map.values()))
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))}, f, indent=1)
    # top-level: detect_expert_quant and the wrapper-config readers both look there first
    text.pop("quantization_config", None)
    config.pop("quantization_config", None)
    if expert_dtype == "nvfp4":
        config["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "W4A16_NVFP4",
            "ignore": IGNORE,
        }
    if max_layers is not None:
        text["num_hidden_layers"] = n_layers
        text["layer_types"] = text["layer_types"][:n_layers]
        text["mlp_layer_types"] = text["mlp_layer_types"][:n_layers]
        text["engram_layer_ids"] = [i for i in text["engram_layer_ids"] if i <= n_layers]
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    for aux in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json", "LICENSE"):
        src = os.path.join(source, aux)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, aux))
    with open(os.path.join(out, "conversion_environment.json"), "w") as f:
        json.dump({
            "tool": "freetoken scripts/convert_hy4_fp8_to_nvfp4.py",
            "source": "tencent/Hy4-flash-FP8",
            "pytorch": torch.__version__,
            "experts": "block-fp8 128x128 -> per-expert ModelOpt NVFP4 (W4A16)",
            "attention": "block-fp8 -> bf16",
            "engram": "bf16 tables copied unchanged (disk-streamed at serve time)",
            "dropped": "model.dfly_layers.* (DFly2 draft, not served)",
        }, f, indent=1)
    print(json.dumps(stats, indent=1))
    print(f"wrote {shard_idx} shards, {total / 2**30:.1f} GiB -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gib", type=float, default=8.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-layers", type=int, default=None, help="dev: convert only the first N layers")
    ap.add_argument("--expert-dtype", choices=("nvfp4", "bf16"), default="nvfp4",
                    help="bf16 keeps the experts dequantized (stacked) for A/B quality checks")
    args = ap.parse_args()
    convert(args.source, args.out, shard_gib=args.shard_gib, device=args.device, max_layers=args.max_layers, expert_dtype=args.expert_dtype)


if __name__ == "__main__":
    sys.exit(main())
