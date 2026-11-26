"""Independent pure-torch ERNIE 5.5 reference for the parity tests.

Transcribed from baidu/vLLM-ERNIE (Apache-2.0): ``ernie5_5/nvidia/model.py``
(mHC / multimax / weight layout), ``attention.py`` (HCA gate + VHA, DSA indexer
window exclusion + sink), ``moe.py`` (split-feature routing, latent MoE),
``mhc.py`` / ``tests/models/ernie5_5/test_mhc.py`` (the eps-outside RMS factor),
and vLLM's ``SituAndMul``. No FreeToken kernels or layers are imported -- this
file is the oracle the engine path is checked against.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


# ----- small math ---------------------------------------------------------------------


def yarn_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device):
    def correction_dim(num_rotations):
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    smooth = 1 - ramp
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32)
    return torch.polar(torch.ones(seqlen, dim // 2), torch.outer(t, freqs)).to(device)


def plain_freqs(dim, seqlen, base, device):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    return torch.polar(torch.ones(seqlen, dim // 2), torch.outer(t, freqs)).to(device)


def rope_interleaved(x, freqs, inverse=False):
    """x [..., rd] (rd even), freqs [T, rd//2] complex broadcast over leading dims."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    f = freqs.conj() if inverse else freqs
    if xc.ndim == 3 and f.ndim == 2:
        f = f.unsqueeze(1)
    return torch.view_as_real(xc * f).flatten(-2).to(x.dtype)


def rope_neox_partial(x, positions, rotary_dim, base):
    """NeoX half-split rope on the FIRST rotary_dim dims (the indexer convention)."""
    rd = rotary_dim
    inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=x.device) / rd))
    freqs = torch.outer(positions.float(), inv)
    cos, sin = freqs.cos(), freqs.sin()
    xr = x[..., :rd].float()
    half = rd // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    # positions may broadcast over a head axis: cos/sin [T, half] -> [..., T, half]
    while cos.ndim < x1.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([rot.to(x.dtype), x[..., rd:]], dim=-1)


def fp8_roundtrip(x, block):
    """ue8m0 per-block e4m3 quant+dequant (the reference act_quant inplace=True)."""
    shape = x.shape
    xf = x.float().reshape(-1, shape[-1] // block, block)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(shape).to(x.dtype)


def rmsnorm(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def situ_and_mul(gate_up, beta, linear_beta):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(gate_up.dtype)


def multimax(logits, r, t):
    return (
        logits
        + t[0] * (r[0] - logits).relu()
        + t[1] * (logits - r[1]).relu()
        + t[2] * (r[2] - logits).relu().square()
        + t[3] * (logits - r[3]).relu().square()
    )


def mhc_pre_ref(residual, fn, scale, base, rms_eps, hc_eps, iters):
    """vLLM-ERNIE test_mhc._mhc_pre_ref: eps OUTSIDE the sqrt, post alpha 2.0."""
    t, n, h = residual.shape
    x = residual.view(t, -1).float()
    mixes = (x @ fn.t()) / (x.square().mean(dim=-1, keepdim=True).sqrt() + rms_eps)
    pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n])
    post = torch.sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n]) * 2.0
    comb = mixes[:, 2 * n :] * scale[2] + base[2 * n :]
    comb = torch.softmax(comb.view(-1, n, n), dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    layer_input = ((pre + hc_eps).unsqueeze(-1) * residual.float()).sum(dim=1)
    return post.unsqueeze(-1), comb, layer_input.to(residual.dtype)


def mhc_post_ref(x, residual, post, comb):
    mixed = torch.einsum("tij,tih->tjh", comb.float(), residual.float())
    return (mixed + post.float() * x.unsqueeze(-2).float()).to(residual.dtype)


def softmax_with_sink(scores, sink):
    """scores [..., K], sink [H] broadcast on the head axis; null key, zero value."""
    s = torch.cat([scores, sink.unsqueeze(-1).expand(scores.shape[:-1] + (1,))], dim=-1)
    p = torch.softmax(s.float(), dim=-1)
    return p[..., :-1]


# ----- the model ----------------------------------------------------------------------


class Ernie55Reference:
    def __init__(self, checkpoint: str, device="cuda", act_quant: bool = False):
        import json
        import os

        self.device = device
        # W8A8/W4A8 serving numerics: fp8 (ue8m0, 1x128) activation round-trip
        # before every quantized GEMM (the checkpoint's activation_scheme).
        self.act_quant = act_quant
        cfg = json.load(open(os.path.join(checkpoint, "config.json")))
        self.cfg = cfg
        wm = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))["weight_map"]
        shards = {}
        for name, shard in wm.items():
            if shard not in shards:
                shards[shard] = load_file(os.path.join(checkpoint, shard), device=device)
        self.t = {k: shards[s][k] for k, s in wm.items()}
        self.H = cfg["hidden_size"]
        self.n = cfg["hc_mult"]
        self.eps = cfg["rms_norm_eps"]
        self.hc_eps = cfg["hc_eps"]
        self.iters = cfg["hc_sinkhorn_iters"]
        self.window = cfg["sliding_window"]
        self.beta = cfg["activation_situ_beta"]
        self.linear_beta = cfg["activation_situ_linear_beta"]

    def g(self, name):
        return self.t[name]

    def linear(self, x, name):
        if self.act_quant:
            x = fp8_roundtrip(x, 128)
        w = self.g(name + ".weight")
        if w.dtype == torch.float32:
            # fp8-GEMM oracle: the kernel applies the float scale in fp32
            return (x.float() @ w.t()).to(x.dtype)
        return x @ w.t()

    def mlp(self, x, prefix, names=("gate_proj", "up_proj", "down_proj")):
        gate_up = torch.cat([self.linear(x, f"{prefix}.{names[0]}"), self.linear(x, f"{prefix}.{names[1]}")], dim=-1)
        return self.linear(situ_and_mul(gate_up, self.beta, self.linear_beta), f"{prefix}.{names[2]}")

    def moe(self, x, prefix):
        cfg = self.cfg
        E = cfg["n_routed_experts"]
        logits = x.float() @ torch.cat([self.g(prefix + ".gate.weight"), self.g(prefix + ".gate.weight_1")]).float().t()
        scores = logits[:, :E].sigmoid() + logits[:, E:].sigmoid()
        bias = self.g(prefix + ".e_score_correction_bias")
        scale = self.g(prefix + ".gate.routed_scaling_factor_param")
        ids = torch.topk(scores + bias, cfg["num_experts_per_tok"], dim=-1, sorted=False).indices
        w = scores.gather(1, ids)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
        w = w * scale[ids]
        latent = self.linear(x, prefix + ".fc1_latent_proj")
        out = torch.zeros_like(latent)
        for e in range(E):
            mask = ids == e
            if not mask.any():
                continue
            rows = mask.nonzero(as_tuple=True)
            xe = latent[rows[0]]
            if self.act_quant:
                xe = fp8_roundtrip(xe, 128)
            gu = torch.cat([xe @ self.g(f"{prefix}.experts.{e}.w1.weight").t(),
                            xe @ self.g(f"{prefix}.experts.{e}.w3.weight").t()], dim=-1)
            he = situ_and_mul(gu, self.beta, self.linear_beta)
            if self.act_quant:
                he = fp8_roundtrip(he, 128)
            ye = he @ self.g(f"{prefix}.experts.{e}.w2.weight").t()
            out[rows[0]] += ye * w[rows[0], rows[1]].unsqueeze(-1).to(ye.dtype)
        out = self.linear(rmsnorm(out, self.g(prefix + ".latent_norm.weight"), self.eps), prefix + ".fc2_latent_proj")
        shared = self.mlp(x, prefix + ".shared_experts", names=("w1", "w3", "w2"))
        return out + shared

    def hc_pre(self, x, kind, layer):
        p = f"model.layers.{layer}.{kind}_hyper_connection"
        fn = self.g(p + ".mapping_proj.weight").t().contiguous()
        base = self.g(p + ".bias")
        scale = torch.cat([self.g(p + f".alpha_{a}") for a in ("pre", "post", "res")])
        return mhc_pre_ref(x, fn, scale, base, self.eps, self.hc_eps, self.iters)

    def vha(self, o, layer):
        a = f"model.layers.{layer}.self_attn"
        U, V = self.g(a + ".vha_postmix_U"), self.g(a + ".vha_postmix_V")
        m = torch.eye(U.shape[0], dtype=U.dtype, device=U.device) + V @ U.t()
        return torch.matmul(m, o)

    def hca_attention(self, x, layer, freqs):
        cfg = self.cfg
        a = f"model.layers.{layer}.self_attn"
        T = x.shape[0]
        NH, HD, RD = cfg["num_attention_heads"], cfg["head_dim"], cfg["qk_rope_head_dim"]
        QL = cfg["q_lora_rank"]
        qr = rmsnorm(self.linear(x, a + ".linear_q_down_proj"), self.g(a + ".q_layernorm.weight"), self.eps)
        q = self.linear(qr, a + ".linear_q_up_proj").view(T, NH, HD)
        q = rmsnorm(q, None, self.eps)
        q = torch.cat([q[..., : HD - RD], rope_interleaved(q[..., HD - RD :], freqs)], dim=-1)
        gate = torch.sigmoid(self.linear(qr, a + ".gate_proj"))
        kv = rmsnorm(self.linear(x, a + ".linear_kv_proj"), self.g(a + ".kv_layernorm.weight"), self.eps)
        kv = torch.cat([kv[..., : HD - RD], rope_interleaved(kv[..., HD - RD :], freqs)], dim=-1)
        kv = torch.cat([fp8_roundtrip(kv[..., : HD - RD], 64), kv[..., HD - RD :]], dim=-1)

        # compressor: gated pooling over each full 128-block
        c = a + ".core_attention.compressor"
        ratio = self.window
        comp = []
        for b in range(T // ratio):
            xb = x[b * ratio : (b + 1) * ratio].float()
            kvb = xb @ self.g(c + ".linear_wkv.weight").float().t()
            score = xb @ self.g(c + ".linear_wgate.weight").float().t() + self.g(c + ".ape").float()
            pooled = (kvb * score.softmax(dim=0)).sum(dim=0)
            pooled = rmsnorm(pooled.to(x.dtype), self.g(c + ".norm.weight"), self.eps)
            pr = rope_interleaved(pooled[..., HD - RD :], freqs[b * ratio])
            pooled = torch.cat([pooled[..., : HD - RD], pr], dim=-1)
            pooled = torch.cat([fp8_roundtrip(pooled[..., : HD - RD], 64), pooled[..., HD - RD :]], dim=-1)
            comp.append(pooled)
        comp = torch.stack(comp) if comp else x.new_zeros(0, HD)

        sink = self.g(a + ".core_attention.attn_sink").float()
        scale = HD ** -0.5
        o = x.new_zeros(T, NH, HD)
        for p in range(T):
            wlo = max(0, p - self.window + 1)
            kc = [kv[wlo : p + 1]]
            nblk = (p + 1) // ratio
            if nblk:
                kc.append(comp[:nblk])
            k = torch.cat(kc, dim=0).float()
            scores = (q[p].float() @ k.t()) * scale
            probs = softmax_with_sink(scores, sink)
            o[p] = (probs @ k).to(x.dtype)
        o = torch.cat([o[..., : HD - RD], rope_interleaved(o[..., HD - RD :], freqs, inverse=True)], dim=-1)
        o = self.vha(o, layer)
        OG, OL = cfg["o_groups"], cfg["o_lora_rank"]
        wo_a = self.g(a + ".linear_o_group_proj").view(OG, OL, NH * HD // OG)
        z = torch.einsum("tgd,grd->tgr", o.reshape(T, OG, -1), wo_a).reshape(T, OG * OL)
        return self.linear(z * gate, a + ".o_proj")

    def dsa_attention(self, x, layer, positions):
        cfg = self.cfg
        a = f"model.layers.{layer}.self_attn"
        T = x.shape[0]
        NH = cfg["hybrid_mla_num_attention_heads"]
        NOPE, RD = cfg["hybrid_mla_qk_nope_head_dim"], cfg["hybrid_mla_qk_rope_head_dim"]
        V = cfg["hybrid_mla_v_head_dim"]
        KV = cfg["hybrid_mla_kv_lora_rank"]
        QK = NOPE + RD
        theta = cfg["rope_theta"]
        q_a = rmsnorm(self.linear(x, a + ".q_a_proj"), self.g(a + ".q_a_layernorm.weight"), self.eps)
        gate = torch.sigmoid(self.linear(q_a, a + ".gate_proj"))
        q = self.linear(q_a, a + ".q_b_proj").view(T, NH, QK)
        q_nope, q_rope = q.split([NOPE, RD], dim=-1)
        kv = self.linear(x, a + ".kv_a_proj_with_mqa")
        c_kv, k_pe = kv.split([KV, RD], dim=-1)
        c_kv = rmsnorm(c_kv, self.g(a + ".kv_a_layernorm.weight"), self.eps)
        freqs = plain_freqs(RD, T, theta, x.device)
        q_rope = rope_interleaved(q_rope, freqs)
        k_pe = rope_interleaved(k_pe, freqs)

        # indexer: NeoX partial rope, relu head-reduced scores, window exclusion
        idx = a + ".core_attention.indexer"
        IH, ID = cfg["index_n_heads"], cfg["index_head_dim"]
        qi = self.linear(q_a, idx + ".wq_b").view(T, IH, ID)
        ki = F.layer_norm(self.linear(x, idx + ".wk"), (ID,), self.g(idx + ".k_norm.weight"), self.g(idx + ".k_norm.bias"), 1e-6)
        qi = rope_neox_partial(qi, positions, RD, theta)
        ki = rope_neox_partial(ki, positions, RD, theta)
        w = self.linear(x, idx + ".weights_proj").float() * (ID ** -0.5 * IH ** -0.5)
        topk = cfg["index_topk"]
        sink = self.g(a + ".core_attention.softmax_offset").float()

        kv_b = self.g(a + ".kv_b_proj.weight").view(NH, NOPE + V, KV)
        w_uk, w_uv = kv_b[:, :NOPE, :], kv_b[:, NOPE:, :]
        scale = QK ** -0.5
        o = x.new_zeros(T, NH, V)
        for p in range(T):
            scored_end = max(0, p + 1 - self.window)
            cand = list(range(max(0, p + 1 - self.window), p + 1))
            if scored_end > 0:
                s = torch.einsum("hd,jd->hj", qi[p].float(), ki[:scored_end].float())
                s = s.clamp_min(0) * w[p][:, None]
                s = s.sum(0)
                k = min(topk, scored_end)
                cand = torch.topk(s, k).indices.tolist() + cand
            ckv = c_kv[cand].float()
            kpe = k_pe[cand].float()
            kn = torch.einsum("jl,hlv->hjv", ckv, w_uk.float())  # [H, K, NOPE]
            sc = torch.einsum("hn,hkn->hk", q_nope[p].float(), kn) + q_rope[p].float() @ kpe.t()
            probs = softmax_with_sink(sc * scale, sink)
            vv = torch.einsum("jl,hlv->hjv", ckv, w_uv.float().transpose(1, 2))
            o[p] = torch.einsum("hk,hkv->hv", probs, vv).to(x.dtype)
        o = self.vha(o, layer)
        return self.linear(o.reshape(T, NH * V) * gate, a + ".o_proj")

    def forward(self, tokens: list[int]) -> torch.Tensor:
        """Full-sequence forward; returns fp32 logits [T, vocab] (multimax applied)."""
        cfg = self.cfg
        dev = self.device
        T = len(tokens)
        ids = torch.tensor(tokens, dtype=torch.long, device=dev)
        positions = torch.arange(T, device=dev)
        h = self.g("model.embed_tokens.weight")[ids]
        h = h.unsqueeze(1).repeat(1, self.n, 1)
        sc = cfg["rope_scaling"]
        hca_freqs = yarn_freqs(
            cfg["qk_rope_head_dim"], T, sc["original_max_position_embeddings"],
            cfg["compress_rope_theta"], sc["factor"], sc["beta_fast"], sc["beta_slow"], dev,
        )
        for layer in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{layer}"
            residual = h
            post, comb, x = self.hc_pre(h, "self_attention", layer)
            x = rmsnorm(x, self.g(p + ".input_layernorm.weight"), self.eps)
            if cfg["compress_ratios"][layer] <= 0:
                x = self.dsa_attention(x, layer, positions)
            else:
                x = self.hca_attention(x, layer, hca_freqs)
            h = mhc_post_ref(x, residual, post, comb)
            residual = h
            post, comb, x = self.hc_pre(h, "mlp", layer)
            x = rmsnorm(x, self.g(p + ".post_attention_layernorm.weight"), self.eps)
            if layer < cfg["first_k_dense_replace"]:
                x = self.mlp(x, p + ".mlp")
            else:
                x = self.moe(x, p + ".block_sparse_moe")
            h = mhc_post_ref(x, residual, post, comb)
        # head contract: upstream RMS factor (eps inside)
        xf = h.view(T, -1).float()
        mixes = (xf @ self.g("model.mhc_contract.hc_head_fn")) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        pre = torch.sigmoid(mixes * self.g("model.mhc_contract.hc_head_scale") + self.g("model.mhc_contract.hc_head_base")) + self.hc_eps
        h = (pre.unsqueeze(-1) * xf.view(T, self.n, self.H)).sum(1).to(h.dtype)
        h = rmsnorm(h, self.g("model.norm.weight"), self.eps)
        logits = (h @ self.g("lm_head.weight").t()).float()
        return multimax(logits, self.g("model.lm_head.multimax_ranges").float(), self.g("model.lm_head.multimax_ts").float())
