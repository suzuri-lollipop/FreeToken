"""Tiny synthetic ERNIE 5.5 checkpoints for the family's tests.

Writes the checkpoint's HF naming (``model.layers.*.self_attn.linear_*`` /
``block_sparse_moe`` / ``mhc_contract`` / ``lm_head.multimax_*``) at a geometry
small enough to run a full forward in a test, in bf16 (parity reference runs on
the same weights) or block-fp8 + MXFP4 experts (the serving quant path).
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file, save_file

# Tiny geometry (kernel constraints kept: index heads >= 16, pow-2 index dim,
# HCA head_dim 512, fp8/MXFP4 dims multiples of 128 / 32).
GEO = dict(
    vocab_size=64,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=16,
    num_key_value_heads=1,
    head_dim=512,
    q_lora_rank=128,
    kv_lora_rank=512,
    qk_nope_head_dim=448,
    qk_rope_head_dim=64,
    o_groups=4,
    o_lora_rank=64,
    sliding_window=128,
    compress_ratios=[128, 128, -2, 128],
    hybrid_mla_num_attention_heads=16,
    hybrid_mla_q_lora_rank=128,
    hybrid_mla_kv_lora_rank=128,
    hybrid_mla_qk_nope_head_dim=128,
    hybrid_mla_qk_rope_head_dim=64,
    hybrid_mla_v_head_dim=128,
    index_n_heads=16,
    index_head_dim=128,
    index_topk=32,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    intermediate_size=256,
    moe_intermediate_size=256,
    moe_latent_size=256,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=4,
    first_k_dense_replace=1,
    routed_scaling_factor=3.0,
    norm_topk_prob=True,
    hidden_act="situ",
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    multimax="lm_head",
    rms_norm_eps=1e-6,
    rope_theta=8000000.0,
    compress_rope_theta=160000.0,
    rope_scaling={
        "rope_type": "yarn",
        "factor": 4.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 512,
    },
    max_position_embeddings=2048,
    tie_word_embeddings=False,
    model_type="ernie5_5_moe",
    architectures=["Ernie5_5_MoeForCausalLM"],
    dspark_block_size=7,
    dspark_markov_rank=256,
    dspark_noise_token_id=1024,
    dspark_target_layer_ids=[2, 3],
)


def _bf16(*shape, scale=0.02, generator=None):
    return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)


def _fp8_block(w: torch.Tensor, block: int = 128):
    """Quantize a bf16 weight to e4m3 + F32 square-block scales (the checkpoint's
    dense form); returns (weight_fp8, scale_inv)."""
    n, k = w.shape
    wf = w.float()
    nb, kb = -(-n // block), -(-k // block)
    padded = torch.zeros(nb * block, kb * block)
    padded[:n, :k] = wf
    blocks = padded.view(nb, block, kb, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-10)
    scale = amax / 448.0
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(nb * block, kb * block)[:n, :k].contiguous(), scale.contiguous()


def _mxfp4(w: torch.Tensor):
    """Quantize a bf16 weight to OCP MXFP4 (e2m1 pairs int8 + e8m0 per-32 scales)."""
    n, k = w.shape
    assert k % 32 == 0
    wf = w.float().view(n, k // 32, 32)
    amax = wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    # e8m0 scale: 2^ceil(log2(amax / 6))
    exp = torch.ceil(torch.log2(amax / 6.0))
    scale = torch.exp2(exp)
    q = (wf / scale).clamp(-6, 6)
    # round to the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = q.sign()
    mag = q.abs()
    idx = (mag.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    codes = idx.to(torch.uint8) | ((sign < 0).to(torch.uint8) << 3)
    codes = codes.view(n, k)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.int8)
    e8m0 = (exp.view(n, k // 32) + 127).clamp(0, 255).to(torch.uint8).view(torch.float8_e8m0fnu)
    return packed, e8m0


def _dequant_fp8_block(q, s, block=128, dtype=torch.float32):
    n, k = q.shape
    sf = s.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    return (q.float() * sf).to(dtype)


_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _dequant_mxfp4(packed, e8m0):
    codes = packed.view(torch.uint8)
    lo, hi = codes & 0xF, codes >> 4
    w = torch.stack([_E2M1_GRID[lo.long()], _E2M1_GRID[hi.long()]], dim=-1).reshape(codes.shape[0], -1)
    s = torch.exp2(e8m0.view(torch.uint8).float() - 127.0)
    return (w * s.repeat_interleave(32, dim=1)).to(torch.bfloat16)


def make_dequantized_checkpoint(folder: str, *, seed: int = 0) -> str:
    """A bf16 checkpoint holding the EXACT values the quantized checkpoint's
    weights dequantize to -- the oracle that isolates kernel correctness from
    quantization error (the quantized engine must reproduce this model up to
    activation quant + accumulation order)."""
    import shutil

    qfolder = folder + "_qsrc"
    make_checkpoint(qfolder, quantized=True, seed=seed)
    os.makedirs(folder, exist_ok=True)
    shutil.copy(os.path.join(qfolder, "config.json"), os.path.join(folder, "config.json"))
    # drop the quantization_config: this checkpoint serves bf16
    cfg = json.load(open(os.path.join(folder, "config.json")))
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w"))
    wm = json.load(open(os.path.join(qfolder, "model.safetensors.index.json")))["weight_map"]
    shard = next(iter(wm.values()))
    t = load_file(os.path.join(qfolder, shard))
    out = {}
    # weights the engine serves through the fp8 GEMM keep fp32 dequant precision
    # (the kernel applies the float scale in fp32); wo_a / kv_b are dequantized to
    # bf16 by the reader itself, so their oracle stays bf16.
    bf16_dequant = {"kv_b_proj.weight"}
    for name, w in t.items():
        if name.endswith(".weight_scale_inv"):
            continue
        scale_name = name[: -len(".weight")] + ".weight_scale_inv"
        if scale_name in t and w.dtype == torch.float8_e4m3fn:
            dt = torch.bfloat16 if name.rsplit(".", 1)[-1] in bf16_dequant or name.endswith("kv_b_proj.weight") else torch.float32
            out[name] = _dequant_fp8_block(w, t[scale_name], dtype=dt)
        elif scale_name in t and w.dtype == torch.int8:
            out[name] = _dequant_mxfp4(w, t[scale_name])
        else:
            out[name] = w
    # wo_a carries no .weight suffix; the reader dequantizes it to bf16
    for layer in range(cfg["num_hidden_layers"]):
        raw = f"model.layers.{layer}.self_attn.linear_o_group_proj"
        if raw in t and t[raw].dtype == torch.float8_e4m3fn:
            out[raw] = _dequant_fp8_block(t[raw], t[raw + "_scale_inv"], dtype=torch.bfloat16)
    save_file(out, os.path.join(folder, shard))
    json.dump({"metadata": {"total_size": 0}, "weight_map": {k: shard for k in out}},
              open(os.path.join(folder, "model.safetensors.index.json"), "w"))
    shutil.rmtree(qfolder)
    return folder


def write_flashindexer_sidecar(folder: str, cfg: dict, *, seed: int = 0) -> None:
    """The head-routing gates in the checkpoint's flashindexer/ subfolder."""
    g = torch.Generator().manual_seed(seed + 100)
    os.makedirs(os.path.join(folder, "flashindexer"), exist_ok=True)
    t = {}
    for layer, ratio in enumerate(cfg["compress_ratios"]):
        if ratio > 0:
            continue
        base = f"model.layers.{layer}.self_attn.core_attention.indexer"
        t[base + ".linear_gate.weight"] = _bf16(cfg["index_n_heads"], cfg["hidden_size"], scale=0.05, generator=g)
        t[base + ".flashindexer_bias"] = torch.randn(cfg["index_n_heads"], generator=g) * 0.1
    save_file(t, os.path.join(folder, "flashindexer", "flashindexer.safetensors"))


def make_checkpoint(folder: str, *, quantized: bool = False, seed: int = 0, geo_overrides: dict | None = None, fan_in: bool = False) -> str:
    """Write config.json + one safetensors shard + index; returns the folder.
    ``geo_overrides`` patches the tiny geometry (e.g. a larger moe_intermediate_size
    so a TP shard still satisfies the decode GEMV's K//2 % 128 floor). ``fan_in``
    scales each compute weight by ``fan_in**-0.5`` (the verify-contract lever): it
    keeps activations O(1) so a verify-vs-decode ulp does not grow through the mHC
    residual into a non-tie argmax flip."""
    os.makedirs(folder, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    cfg = dict(GEO)
    if geo_overrides:
        cfg.update(geo_overrides)
    if quantized:
        cfg["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "expert_dtype": "mxfp4",
            "scale_fmt": "ue8m0",
        }
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(cfg, f)

    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    HD = cfg["head_dim"]
    QL = cfg["q_lora_rank"]
    NH = cfg["num_attention_heads"]
    OG, OL = cfg["o_groups"], cfg["o_lora_rank"]
    E = cfg["n_routed_experts"]
    MI = cfg["moe_intermediate_size"]
    ML = cfg["moe_latent_size"]
    MNH = cfg["hybrid_mla_num_attention_heads"]
    MQL = cfg["hybrid_mla_q_lora_rank"]
    MKV = cfg["hybrid_mla_kv_lora_rank"]
    MNOPE = cfg["hybrid_mla_qk_nope_head_dim"]
    MROPE = cfg["hybrid_mla_qk_rope_head_dim"]
    MV = cfg["hybrid_mla_v_head_dim"]
    IH, ID = cfg["index_n_heads"], cfg["index_head_dim"]
    HC = cfg["hc_mult"]
    MIX = (2 + HC) * HC

    t: dict[str, torch.Tensor] = {}

    def _fanin(w):
        return w * (w.shape[-1] ** -0.5) if fan_in else w

    def linear(name, w):
        w = _fanin(w)
        if quantized:
            q, s = _fp8_block(w)
            t[name + ".weight"] = q
            t[name + ".weight_scale_inv"] = s
        else:
            t[name + ".weight"] = w

    def expert(name, w):
        w = _fanin(w)
        if quantized:
            p, s = _mxfp4(w)
            t[name + ".weight"] = p
            t[name + ".weight_scale_inv"] = s
        else:
            t[name + ".weight"] = w

    t["model.embed_tokens.weight"] = _bf16(V, H, generator=g)
    t["lm_head.weight"] = _bf16(V, H, generator=g)
    t["model.norm.weight"] = (torch.ones(H) + _bf16(H, scale=0.01, generator=g).float()).to(torch.bfloat16)
    t["model.lm_head.multimax_ranges"] = torch.tensor([15.625, 15.9375, 8.75, 0.3671875]).to(torch.bfloat16)
    t["model.lm_head.multimax_ts"] = torch.tensor([-0.1337890625, 0.279296875, -0.09765625, -0.0301513671875]).to(torch.bfloat16)
    t["model.mhc_contract.hc_head_fn"] = _bf16(HC * H, HC, scale=0.01, generator=g).float()
    t["model.mhc_contract.hc_head_base"] = torch.randn(HC, generator=g) * 0.1
    t["model.mhc_contract.hc_head_scale"] = torch.tensor([0.14])

    for layer in range(L):
        p = f"model.layers.{layer}"
        t[p + ".input_layernorm.weight"] = (torch.ones(H) + _bf16(H, scale=0.01, generator=g).float()).to(torch.bfloat16)
        t[p + ".post_attention_layernorm.weight"] = (torch.ones(H) + _bf16(H, scale=0.01, generator=g).float()).to(torch.bfloat16)
        for kind in ("self_attention", "mlp"):
            hc = f"{p}.{kind}_hyper_connection"
            t[hc + ".mapping_proj.weight"] = _bf16(HC * H, MIX, scale=0.01, generator=g).float()
            t[hc + ".bias"] = torch.randn(MIX, generator=g) * 0.1
            t[hc + ".alpha_pre"] = torch.tensor([0.01])
            t[hc + ".alpha_post"] = torch.tensor([0.006])
            t[hc + ".alpha_res"] = torch.tensor([5.3])
        a = p + ".self_attn"
        t[a + ".vha_postmix_U"] = _bf16(NH, 16, scale=0.05, generator=g)
        t[a + ".vha_postmix_V"] = _bf16(NH, 16, scale=0.05, generator=g)
        linear(a + ".gate_proj", _bf16(OG * OL if cfg["compress_ratios"][layer] > 0 else MNH * MV, QL, generator=g))
        if cfg["compress_ratios"][layer] <= 0:
            linear(a + ".q_a_proj", _bf16(MQL, H, generator=g))
            t[a + ".q_a_layernorm.weight"] = (torch.ones(MQL) + _bf16(MQL, scale=0.01, generator=g).float()).to(torch.bfloat16)
            linear(a + ".q_b_proj", _bf16(MNH * (MNOPE + MROPE), MQL, generator=g))
            linear(a + ".kv_a_proj_with_mqa", _bf16(MKV + MROPE, H, generator=g))
            t[a + ".kv_a_layernorm.weight"] = (torch.ones(MKV) + _bf16(MKV, scale=0.01, generator=g).float()).to(torch.bfloat16)
            linear(a + ".kv_b_proj", _bf16(MNH * (MNOPE + MV), MKV, generator=g))
            linear(a + ".o_proj", _bf16(H, MNH * MV, generator=g))
            t[a + ".core_attention.softmax_offset"] = torch.randn(MNH, generator=g) * 0.5
            idx = a + ".core_attention.indexer"
            linear(idx + ".wq_b", _bf16(IH * ID, MQL, generator=g))
            t[idx + ".wk.weight"] = _bf16(ID, H, generator=g)
            t[idx + ".k_norm.weight"] = (torch.ones(ID) + _bf16(ID, scale=0.01, generator=g).float()).to(torch.bfloat16)
            t[idx + ".k_norm.bias"] = _bf16(ID, scale=0.01, generator=g)
            t[idx + ".weights_proj.weight"] = _bf16(IH, H, generator=g)
        else:
            linear(a + ".linear_q_down_proj", _bf16(QL, H, generator=g))
            t[a + ".q_layernorm.weight"] = (torch.ones(QL) + _bf16(QL, scale=0.01, generator=g).float()).to(torch.bfloat16)
            linear(a + ".linear_q_up_proj", _bf16(NH * HD, QL, generator=g))
            linear(a + ".linear_kv_proj", _bf16(HD, H, generator=g))
            t[a + ".kv_layernorm.weight"] = (torch.ones(HD) + _bf16(HD, scale=0.01, generator=g).float()).to(torch.bfloat16)
            # the checkpoint stores wo_a WITHOUT the .weight suffix (raw tensor name)
            w_oa = _bf16(OG * OL, NH * HD // OG, generator=g)
            if quantized:
                q, s = _fp8_block(w_oa)
                t[a + ".linear_o_group_proj"] = q
                t[a + ".linear_o_group_proj_scale_inv"] = s
            else:
                t[a + ".linear_o_group_proj"] = w_oa
            linear(a + ".o_proj", _bf16(H, OG * OL, generator=g))
            t[a + ".core_attention.attn_sink"] = torch.randn(NH, generator=g) * 0.5
            c = a + ".core_attention.compressor"
            t[c + ".ape"] = torch.randn(cfg["sliding_window"], HD, generator=g) * 0.05
            t[c + ".linear_wkv.weight"] = _bf16(HD, H, generator=g)
            t[c + ".linear_wgate.weight"] = _bf16(HD, H, generator=g)
            t[c + ".norm.weight"] = (torch.ones(HD) + _bf16(HD, scale=0.01, generator=g).float()).to(torch.bfloat16)
        if layer < cfg["first_k_dense_replace"]:
            for proj, shape in (("gate_proj", (cfg["intermediate_size"], H)), ("up_proj", (cfg["intermediate_size"], H)), ("down_proj", (H, cfg["intermediate_size"]))):
                linear(f"{p}.mlp.{proj}", _bf16(*shape, generator=g))
        else:
            m = p + ".block_sparse_moe"
            t[m + ".gate.weight"] = _bf16(E, H, generator=g)
            t[m + ".gate.weight_1"] = _bf16(E, H, generator=g)
            t[m + ".gate.routed_scaling_factor_param"] = torch.rand(E, generator=g) + 2.5
            t[m + ".e_score_correction_bias"] = torch.randn(E, generator=g) * 0.1
            linear(m + ".fc1_latent_proj", _bf16(ML, H, generator=g))
            linear(m + ".fc2_latent_proj", _bf16(H, ML, generator=g))
            t[m + ".latent_norm.weight"] = (torch.ones(ML) + _bf16(ML, scale=0.01, generator=g).float()).to(torch.bfloat16)
            for proj, shape in (("w1", (MI, H)), ("w3", (MI, H)), ("w2", (H, MI))):
                linear(f"{m}.shared_experts.{proj}", _bf16(*shape, generator=g))
            for e in range(E):
                expert(f"{m}.experts.{e}.w1", _bf16(MI, ML, generator=g))
                expert(f"{m}.experts.{e}.w3", _bf16(MI, ML, generator=g))
                expert(f"{m}.experts.{e}.w2", _bf16(ML, MI, generator=g))

    # the bundled DSpark drafter (model.mtp.*): a tiny 3-layer draft over the contracted
    # hidden C=H, GQA 2/1 head_dim 128, Markov rank R, taps = target layers + final hidden
    if cfg.get("dspark", True):
        C = H
        DL = cfg.get("dspark_num_layers", 3)
        DHD = cfg.get("dspark_head_dim", 128)
        DH, DKV = 2, 1
        R = cfg.get("dspark_markov_rank", 32)
        DF = cfg.get("dspark_intermediate_size", 256)
        taps = len(cfg.get("dspark_target_layer_ids", (1, 3))) + 1
        mtp = "model.mtp."
        t[mtp + "norm.weight"] = (torch.ones(C) + _bf16(C, scale=0.01, generator=g).float()).to(torch.bfloat16)
        t[mtp + "mask_embed"] = _bf16(C, scale=0.05, generator=g).float()
        t[mtp + "fuse_weight"] = torch.randn(DL + 1, taps, generator=g) * 0.1
        for i in range(DL):
            b = f"{mtp}layers.{i}."
            a = f"{b}self_attn."
            for nm in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
                t[b + nm + ".weight"] = (torch.ones(C) + _bf16(C, scale=0.01, generator=g).float()).to(torch.bfloat16)
            t[a + "q_proj.weight"] = _bf16(DH * DHD, C, scale=0.02, generator=g)
            t[a + "k_proj.weight"] = _bf16(DKV * DHD, C, scale=0.02, generator=g)
            t[a + "v_proj.weight"] = _bf16(DKV * DHD, C, scale=0.02, generator=g)
            t[a + "o_proj.weight"] = _bf16(C, DH * DHD, scale=0.02, generator=g)
            t[a + "k_proj_ctx.weight"] = _bf16(DKV * DHD, C, scale=0.02, generator=g)
            t[a + "v_proj_ctx.weight"] = _bf16(DKV * DHD, C, scale=0.02, generator=g)
            for nm in ("q_norm", "k_norm", "k_norm_ctx"):
                t[a + nm + ".weight"] = (torch.ones(DHD) + _bf16(DHD, scale=0.01, generator=g).float()).to(torch.bfloat16)
            t[b + "mlp.gate_proj.weight"] = _bf16(DF, C, scale=0.02, generator=g)
            t[b + "mlp.up_proj.weight"] = _bf16(DF, C, scale=0.02, generator=g)
            t[b + "mlp.down_proj.weight"] = _bf16(C, DF, scale=0.02, generator=g)
        mh = mtp + "markov_head."
        t[mh + "dflare_norm.weight"] = (torch.ones(C) + _bf16(C, scale=0.01, generator=g).float()).to(torch.bfloat16)
        t[mh + "markov_w1.weight"] = _bf16(V, R, scale=0.02, generator=g)
        t[mh + "rnn_proj.weight"] = _bf16(3 * R, 2 * R + 2 * C, scale=0.02, generator=g)
        t[mh + "rnn_proj.bias"] = torch.randn(3 * R, generator=g) * 0.05
        t[mh + "hidden_out.weight"] = _bf16(C, R, scale=0.02, generator=g)
        t[mh + "hidden_out.bias"] = _bf16(C, scale=0.02, generator=g)

    shard = "model-00001-of-00001.safetensors"
    save_file(t, os.path.join(folder, shard))
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": {k: shard for k in t}}, f)
    return folder
