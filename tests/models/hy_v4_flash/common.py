"""Shared fixtures for the hy_v4_flash tests: a tiny Hy4-flash-shaped config (2 indexed +
3 SWA + 1 indexed layer, one dense FFN, 8 latent experts, engram at layer 1) and a synthetic
bf16 checkpoint in the real tensor naming. Holds no tests itself."""

from __future__ import annotations

import json
import os

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB = 512
DIM = 256
N_LAYERS = 6
LAYER_TYPES = ["indexed_attention", "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "indexed_attention"]
HEADS = 4
SWA_HEADS = 8
QK_NOPE = 32
QK_ROPE = 64  # the engine's rope kernels serve head sizes 64/128/256/512
V_HEAD = 32
Q_LORA = 64
KV_LORA = 64
INTER = 128
MOE_INTER = 32
LATENT = 32
EXPERTS = 8
TOP_K = 2
WINDOW = 16
RATIO = 4
GROUP_TOPK = 4  # candidate_topk = 16: the sparse path engages past 16 tokens
INDEX_HEADS = 16
INDEX_DIM = 128  # partial rope on the trailing QK_ROPE channels, as the real indexer
HC = 4
ENGRAM_LAYERS_1BASED = [2]  # zero-based layer 1
ENGRAM_HEADS = 2  # per order
ENGRAM_HEAD_DIM = 16
ENGRAM_PRIMES = [1031, 1033, 1039, 1041]  # order-2 heads, order-3 heads
ENGRAM_ROWS = sum(ENGRAM_PRIMES)
LOOKUP_SIZE = 512
MAX_POS = 4096


def tiny_text_config(**over) -> dict:
    text = dict(
        model_type="hy_v4_flash",
        vocab_size=VOCAB, hidden_size=DIM, intermediate_size=INTER, moe_intermediate_size=MOE_INTER,
        num_hidden_layers=N_LAYERS, num_attention_heads=HEADS, swa_num_attention_heads=SWA_HEADS,
        q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE, qk_rope_head_dim=QK_ROPE, v_head_dim=V_HEAD,
        swa_qk_nope_head_dim=QK_NOPE, swa_qk_rope_head_dim=QK_ROPE, swa_v_head_dim=V_HEAD,
        hidden_act="silu", rms_norm_eps=1e-5, max_position_embeddings=MAX_POS,
        layer_types=list(LAYER_TYPES),
        mlp_layer_types=["dense"] + ["sparse"] * (N_LAYERS - 1),
        first_k_dense_replace=1,
        n_routed_experts=EXPERTS, num_experts_per_tok=TOP_K, n_shared_experts=1,
        routed_scaling_factor=3.998, norm_topk_prob=True, n_group=1, topk_group=1,
        routed_expert_hidden_size=LATENT, swiglu_limit=10.0,
        sliding_window=WINDOW,
        rope_parameters={
            "indexed_attention": {"rope_type": "default", "rope_theta": 1e7, "mrope_interleaved": True, "mrope_section": [12, 10, 10]},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4, "mrope_interleaved": True, "mrope_section": [12, 10, 10]},
        },
        compressed_indexer_compress_ratio=RATIO, compressed_indexer_num_attention_heads=INDEX_HEADS,
        compressed_indexer_head_dim=INDEX_DIM, compressed_indexer_group_topk=GROUP_TOPK,
        hc_mult=HC, hc_magnitude=2.0, hc_eps=1e-6,
        engram_layer_ids=list(ENGRAM_LAYERS_1BASED), engram_max_ngram_size=3,
        engram_n_head_per_ngram=ENGRAM_HEADS, engram_n_embed_per_ngram=ENGRAM_HEADS * ENGRAM_HEAD_DIM,
        engram_layer_head_vocab_sizes={"2": [ENGRAM_PRIMES[:2], ENGRAM_PRIMES[2:]]},
        engram_pad_id=2, engram_seed=0, engram_lookup_table_size=LOOKUP_SIZE, engram_fp8_embedding=True,
        learnable_sink_init=0.0, tie_word_embeddings=False,
    )
    text.update(over)
    return text


def tiny_hf_config(**over) -> dict:
    return dict(architectures=["HYV4FlashForCausalLM"], model_type="hy_v4_flash", text_config=tiny_text_config(**over))


def checkpoint_tensors(text: dict, seed: int = 0, indexer_weight_scale: float = 0.02) -> dict[str, torch.Tensor]:
    """Random bf16 weights under the checkpoint's names; hc / sink / router-bias / indexer
    pooling weights in fp32 as shipped. Experts are stacked bf16 (the unquantized form)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.02, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    dim = text["hidden_size"]
    qk = text["qk_nope_head_dim"] + text["qk_rope_head_dim"]
    E, inter, latent = text["n_routed_experts"], text["moe_intermediate_size"], text["routed_expert_hidden_size"]
    hc = text["hc_mult"]
    ih, idim = text["compressed_indexer_num_attention_heads"], text["compressed_indexer_head_dim"]
    t: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rnd(text["vocab_size"], dim, scale=1.0),
        "model.norm.weight": torch.ones(dim, dtype=torch.bfloat16),
        "lm_head.weight": rnd(text["vocab_size"], dim),
        "model.hc_head.hc_head_fn": rnd(hc, hc * dim, scale=0.05, dtype=torch.float32),
        "model.hc_head.hc_head_base": rnd(hc, scale=0.3, dtype=torch.float32),
        "model.hc_head.hc_head_scale": (torch.rand(1, generator=g) + 0.5).to(torch.float32),
    }
    for L in range(text["num_hidden_layers"]):
        swa = text["layer_types"][L] == "sliding_attention"
        heads = text["swa_num_attention_heads"] if swa else text["num_attention_heads"]
        a = f"model.layers.{L}.self_attn"
        t[f"{a}.q_a_proj.weight"] = rnd(text["q_lora_rank"], dim)
        t[f"{a}.q_a_layernorm.weight"] = torch.ones(text["q_lora_rank"], dtype=torch.bfloat16)
        t[f"{a}.q_b_proj.weight"] = rnd(heads * qk, text["q_lora_rank"])
        t[f"{a}.kv_a_proj_with_mqa.weight"] = rnd(text["kv_lora_rank"] + text["qk_rope_head_dim"], dim)
        t[f"{a}.kv_a_layernorm.weight"] = torch.ones(text["kv_lora_rank"], dtype=torch.bfloat16)
        t[f"{a}.kv_b_proj.weight"] = rnd(heads * (text["qk_nope_head_dim"] + text["v_head_dim"]), text["kv_lora_rank"])
        t[f"{a}.linear_gate.weight"] = rnd(heads * text["v_head_dim"], dim)
        t[f"{a}.o_proj.weight"] = rnd(dim, heads * text["v_head_dim"])
        t[f"{a}.post_proj_norm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
        t[f"{a}.learnable_sink_param"] = rnd(heads, scale=0.5, dtype=torch.float32)
        if not swa:
            c = f"{a}.compressed_indexer"
            t[f"{c}.query_projection.weight"] = rnd(ih * idim, text["q_lora_rank"])
            t[f"{c}.head_weight_projection.weight"] = rnd(ih, dim, scale=indexer_weight_scale)
            t[f"{c}.token_key_projection.weight"] = rnd(idim, dim)
            t[f"{c}.token_key_norm.weight"] = torch.ones(idim, dtype=torch.bfloat16)
            t[f"{c}.key_value_projection.weight"] = rnd(idim, idim, dtype=torch.float32)
            t[f"{c}.key_gate_projection.weight"] = rnd(idim, idim, dtype=torch.float32)
            t[f"{c}.key_norm.weight"] = torch.ones(idim, dtype=torch.bfloat16)
            t[f"{c}.intra_group_ape"] = rnd(text["compressed_indexer_compress_ratio"], idim, scale=0.1, dtype=torch.float32)
        p = f"model.layers.{L}"
        t[f"{p}.input_layernorm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
        t[f"{p}.post_attention_layernorm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
        for nm in ("hc_attn_layer", "hc_mlp_layer"):
            t[f"{p}.{nm}.hc_pre.hc_fn"] = rnd(2 * hc, hc * dim, scale=0.05, dtype=torch.float32)
            t[f"{p}.{nm}.hc_pre.hc_base"] = rnd(2 * hc, scale=0.3, dtype=torch.float32)
            t[f"{p}.{nm}.hc_pre.hc_scale"] = (torch.rand(2, generator=g) + 0.5).to(torch.float32)
        m = f"{p}.mlp"
        if text["mlp_layer_types"][L] == "dense":
            for proj, shape in (("gate_proj", (text["intermediate_size"], dim)), ("up_proj", (text["intermediate_size"], dim)), ("down_proj", (dim, text["intermediate_size"]))):
                t[f"{m}.{proj}.weight"] = rnd(*shape)
            t[f"{m}.post_fc2_norm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
        else:
            t[f"{m}.gate.weight"] = rnd(E, dim)
            t[f"{m}.gate.e_score_correction_bias"] = rnd(E, scale=0.1, dtype=torch.float32)
            t[f"{m}.routed_expert_down_proj.weight"] = rnd(latent, dim)
            t[f"{m}.routed_expert_up_proj.weight"] = rnd(dim, latent)
            t[f"{m}.experts.gate_up_proj"] = rnd(E, 2 * inter, latent)
            t[f"{m}.experts.down_proj"] = rnd(E, latent, inter)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                shape = (inter, dim) if proj != "down_proj" else (dim, inter)
                t[f"{m}.shared_experts.{proj}.weight"] = rnd(*shape)
            t[f"{m}.combine_output_norm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
        if (L + 1) in text["engram_layer_ids"]:
            e = f"{p}.engram_layer"
            head_dim = text["engram_n_embed_per_ngram"] // text["engram_n_head_per_ngram"]
            width = (text["engram_max_ngram_size"] - 1) * text["engram_n_embed_per_ngram"]
            rows = sum(s for order in text["engram_layer_head_vocab_sizes"][str(L + 1)] for s in order)
            # A compression map (max id < table length - 1), like the real checkpoint's
            # normalized-vocab map: an identity arange would hide the multiplier-bound bug
            # (table length vs max id + 1 coincide for identity).
            size = text["engram_lookup_table_size"]
            t[f"{e}.lookup_table"] = (torch.arange(size, dtype=torch.int32) % (size - 1))
            t[f"{e}.multi_head_embedding.embedding.weight"] = rnd(rows, head_dim, scale=1.0)
            t[f"{e}.value_proj.weight"] = rnd(dim, width)
            t[f"{e}.value_proj.bias"] = rnd(dim, scale=0.1)
            for h in range(hc):
                t[f"{e}.key_projs.{h}.weight"] = rnd(dim, width)
                t[f"{e}.key_projs.{h}.bias"] = rnd(dim, scale=0.1)
                t[f"{e}.norm_key.{h}.weight"] = torch.ones(dim, dtype=torch.bfloat16)
                t[f"{e}.norm_query.{h}.weight"] = torch.ones(dim, dtype=torch.bfloat16)
                t[f"{e}.unified_per_hc_scales.{h}"] = rnd(dim, scale=0.1)
            for h in range((text["engram_max_ngram_size"] - 1) * text["engram_n_head_per_ngram"]):
                t[f"{e}.lookup_per_head_norms.{h}.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
            t[f"{e}.unified_pre_gamma"] = torch.ones(dim, dtype=torch.bfloat16)
    # present in a real checkpoint, dropped by the reader
    t["model.dfly_layers.0.mlp.gate_proj.weight"] = rnd(8, dim)
    t["vit.embeddings.patch_embedding.weight"] = rnd(8, 3, 4, 4)
    return t


def write_tiny_checkpoint(folder: str, seed: int = 0, indexer_weight_scale: float = 0.02, **over) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    text = tiny_text_config(**over)
    tensors = checkpoint_tensors(text, seed, indexer_weight_scale=indexer_weight_scale)
    os.makedirs(folder, exist_ok=True)
    save_file(tensors, os.path.join(folder, "model.safetensors"))
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(tiny_hf_config(**over), f)
    return tensors


def quant_block_fp8(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-style square-block fp8 quantization (the format definition, independent
    of the engine's kernels): amax/448 per block, bf16 scale_inv."""
    out, inn = w.shape[-2], w.shape[-1]
    assert out % block == 0 and inn % block == 0, w.shape
    wf = w.float().reshape(*w.shape[:-2], out // block, block, inn // block, block)
    amax = wf.abs().amax(dim=(-3, -1))
    scale = (amax / 448.0).clamp(min=1e-12).to(torch.bfloat16)
    q = (wf / scale.float()[..., :, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(w.shape), scale


def dequant_block_fp8(q: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    out, inn = q.shape[-2], q.shape[-1]
    s = scale.float().repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)
    return (q.float() * s[..., :out, :inn]).to(torch.bfloat16)


def write_tiny_fp8_checkpoint(folder: str, seed: int = 0, indexer_weight_scale: float = 0.02, **over) -> dict[str, torch.Tensor]:
    """The bf16 fixture with the FP8 release's quantization layout: stacked block-fp8
    experts + block-fp8 q_b/o/linear_gate on the MoE layers, everything else bf16,
    modules_to_not_convert in the HF wrapper spelling. Returns the DEQUANTIZED bf16
    tensors (what an eager reference computes on)."""
    from safetensors.torch import save_file

    text = tiny_text_config(**over)
    bf16 = checkpoint_tensors(text, seed, indexer_weight_scale=indexer_weight_scale)
    dense = text["first_k_dense_replace"]
    fp8_targets = []
    for L in range(text["num_hidden_layers"]):
        if text["mlp_layer_types"][L] == "sparse":
            fp8_targets += [f"model.layers.{L}.mlp.experts.gate_up_proj", f"model.layers.{L}.mlp.experts.down_proj"]
        if L >= dense:
            fp8_targets += [f"model.layers.{L}.self_attn.{p}.weight" for p in ("q_b_proj", "o_proj", "linear_gate")]
    stored, dequantized, not_convert = {}, {}, []
    for name, t in bf16.items():
        if name in fp8_targets:
            q, s = quant_block_fp8(t)
            stored[name] = q
            stored[name + "_scale_inv"] = s
            dequantized[name] = dequant_block_fp8(q, s)
        else:
            stored[name] = t
            dequantized[name] = t
            module = name[: -len(".weight")] if name.endswith(".weight") else name
            not_convert.append("language_model.model." + module)
    os.makedirs(folder, exist_ok=True)
    save_file(stored, os.path.join(folder, "model.safetensors"))
    cfg = tiny_hf_config(**over)
    cfg["text_config"]["quantization_config"] = {
        "quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
        "weight_block_size": [128, 128], "modules_to_not_convert": not_convert,
    }
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(cfg, f)
    return dequantized


__all__ = [
    "requires_cuda", "tiny_text_config", "tiny_hf_config", "checkpoint_tensors", "write_tiny_checkpoint",
    "write_tiny_fp8_checkpoint", "quant_block_fp8", "dequant_block_fp8",
    "VOCAB", "DIM", "N_LAYERS", "ENGRAM_ROWS", "ENGRAM_HEAD_DIM", "ENGRAM_PRIMES", "LOOKUP_SIZE",
    "WINDOW", "RATIO", "GROUP_TOPK", "EXPERTS", "LATENT", "MOE_INTER",
]
