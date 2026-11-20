"""Standalone eager reference for Hy4-flash, transcribed from the transformers PR
(huggingface/transformers#50903, modeling/modular_hy_v4_flash.py) and its HYV4 base.

Pure torch: per-head K/V attention (no weight absorption), brute-force compressed
indexer selection, dynamic latent cache, fp32 iHC, host engram hash. Independent of
the engine's kernels and pools -- the parity expectation comes from here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# diagnostics switches: the engine's kernels rope and softmax in fp32; the PR's
# eager path runs both in the activation dtype. Set True to match the engine
# convention when comparing against it at widths where bf16 conventions flip
# near-tied indexer scores.
SOFTMAX_FP32 = False
ROPE_FP32 = False


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(dtype)


def unweighted_rms_inv(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # GLM interleaved: adjacent (even, odd) pairs, matching the PR's rotate_half
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if ROPE_FP32:
        cos = cos.unsqueeze(1).float()
        sin = sin.unsqueeze(1).float()
        qf, kf = q.float(), k.float()
        return (qf * cos + rotate_half(qf) * sin).to(q.dtype), (kf * cos + rotate_half(kf) * sin).to(k.dtype)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def partial_rope_trailing(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
    """Rotate the TRAILING rope_dim channels (indexer convention), GLM interleaved."""
    passthrough, rotary = torch.split(x, [x.shape[-1] - rope_dim, rope_dim], dim=-1)
    if ROPE_FP32:
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        rf = rotary.float()
        rotary = (rf * cos + rotate_half(rf) * sin).to(x.dtype)
    else:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        rotary = rotary * cos + rotate_half(rotary) * sin
    return torch.cat((passthrough, rotary), dim=-1)


class RopeTable:
    def __init__(self, rope_dim: int, theta: float, max_pos: int, device):
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float, device=device) / rope_dim))
        freqs = torch.arange(max_pos, dtype=torch.float, device=device)[:, None] * inv_freq[None, :]
        # interleaved layout: each frequency repeated for its (even, odd) pair
        emb = freqs.repeat_interleave(2, dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()

    def at(self, positions: torch.Tensor, dtype):
        return self.cos[positions].to(dtype), self.sin[positions].to(dtype)


class Cache:
    def __init__(self):
        self.k_pass: dict[int, torch.Tensor] = {}
        self.k_rot: dict[int, torch.Tensor] = {}
        self.token_keys: dict[int, torch.Tensor] = {}
        self.positions: torch.Tensor | None = None

    def update_positions(self, positions: torch.Tensor) -> torch.Tensor:
        self.positions = positions if self.positions is None else torch.cat([self.positions, positions], dim=1)
        return self.positions

    def update(self, k_pass, k_rot, layer):
        if layer in self.k_pass:
            k_pass = torch.cat([self.k_pass[layer], k_pass], dim=2)
            k_rot = torch.cat([self.k_rot[layer], k_rot], dim=2)
        self.k_pass[layer], self.k_rot[layer] = k_pass, k_rot
        return k_pass, k_rot

    def update_indexer(self, keys, layer):
        if layer in self.token_keys:
            keys = torch.cat([self.token_keys[layer], keys], dim=1)
        self.token_keys[layer] = keys
        return keys


def draw_multipliers(seed: int, layer_1based: int, max_ngram: int, compressed_vocab: int) -> torch.Tensor:
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
    gen = np.random.default_rng(seed + 10007 * layer_1based)
    drawn = gen.integers(0, half_bound, size=max_ngram, dtype=np.int64)
    return torch.tensor((drawn * 2 + 1).tolist(), dtype=torch.int64)


def fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    quantized = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return quantized


class Reference:
    def __init__(self, tensors: dict[str, torch.Tensor], text: dict, device="cuda"):
        # a _lazy provider (streaming full-depth diagnostics) is kept as-is; a plain
        # dict is moved to the device up front
        self.t = tensors if getattr(tensors, "_lazy", False) else {k: v.to(device) for k, v in tensors.items()}
        self.c = text
        self.device = device
        self.dim = text["hidden_size"]
        self.eps = text["rms_norm_eps"]
        self.hc = text["hc_mult"]
        self.rope_dim = text["qk_rope_head_dim"]
        self.qk = text["qk_nope_head_dim"] + self.rope_dim
        self.v_head = text["v_head_dim"]
        self.kv_lora = text["kv_lora_rank"]
        self.ratio = text["compressed_indexer_compress_ratio"]
        self.group_topk = text["compressed_indexer_group_topk"]
        self.candidate_topk = self.group_topk * self.ratio
        self.window = text["sliding_window"]
        self.limit = text["swiglu_limit"]
        self.max_ngram = text["engram_max_ngram_size"]
        self.engram_layers = {i - 1 for i in text["engram_layer_ids"]}
        rope_params = text["rope_parameters"]
        indexed_params = rope_params.get("indexed_attention") or rope_params.get("deepseek_sparse_attention")
        self.rope = {
            "indexed_attention": RopeTable(self.rope_dim, indexed_params["rope_theta"], text["max_position_embeddings"], device),
            "sliding_attention": RopeTable(self.rope_dim, rope_params["sliding_attention"]["rope_theta"], text["max_position_embeddings"], device),
        }
        # engram hash state per layer
        self.engram = {}
        for L in self.engram_layers:
            lookup = self.t[f"model.layers.{L}.engram_layer.lookup_table"].long()
            sizes = [s for order in text["engram_layer_head_vocab_sizes"][str(L + 1)] for s in order]
            n_heads = text["engram_n_head_per_ngram"]
            self.engram[L] = dict(
                lookup=lookup,
                table_size=lookup.numel(),
                pad_compressed=int(lookup[text["engram_pad_id"]].item()),
                multipliers=draw_multipliers(text["engram_seed"], L + 1, self.max_ngram, int(lookup.max().item()) + 1).to(device),
                moduli=torch.tensor(sizes, dtype=torch.int64, device=device),
                offsets=torch.tensor([0] + sizes[:-1], dtype=torch.int64, device=device).cumsum(0),
                n_heads=n_heads,
                head_dim=text["engram_n_embed_per_ngram"] // n_heads,
                table=self.t[f"model.layers.{L}.engram_layer.multi_head_embedding.embedding.weight"],
            )

    def _get(self, name):
        return self.t[name]

    def _linear(self, x, name, bias=False):
        w = self.t[name + ".weight"]
        b = self.t[name + ".bias"] if bias else None
        return F.linear(x, w, b)

    # ----- engram ---------------------------------------------------------------------
    def build_ngrams(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """[B, T, max_ngram]: the token and its predecessors, pad past the doc start."""
        pad = self.c["engram_pad_id"]
        offsets = positions  # text-only: positions restart at 0 per document
        shifted = [input_ids]
        for off in range(1, self.max_ngram):
            previous = F.pad(input_ids, (off, 0), value=pad)[:, : input_ids.shape[1]]
            shifted.append(torch.where(offsets < off, pad, previous))
        return torch.stack(shifted, dim=-1)

    def engram_forward(self, L: int, streams: torch.Tensor, ngram_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.engram[L]
        p = f"model.layers.{L}.engram_layer"
        table_size = e["table_size"]
        compressed = e["lookup"][ngram_tokens.clamp_max(table_size - 1)]
        compressed = torch.where(ngram_tokens < table_size, compressed, torch.tensor(e["pad_compressed"], device=self.device))
        mixed = compressed.long() * e["multipliers"]
        ids = []
        for order in range(2, self.max_ngram + 1):
            mix = mixed[..., 0]
            for off in range(1, order):
                mix = mix ^ mixed[..., off]
            start = (order - 2) * e["n_heads"]
            ids.append(mix.unsqueeze(-1) % e["moduli"][start : start + e["n_heads"]])
        hash_ids = torch.cat(ids, dim=-1)
        emb = e["table"][hash_ids + e["offsets"]].to(streams.dtype)
        if self.c["engram_fp8_embedding"]:
            emb = fp8_round_trip(emb)
        emb = torch.stack(
            [rms_norm(emb[:, :, h], self._get(f"{p}.lookup_per_head_norms.{h}.weight"), 1e-5) for h in range(hash_ids.shape[-1])],
            dim=2,
        )
        emb = emb.flatten(start_dim=-2)
        gates = []
        for hc in range(self.hc):
            key = rms_norm(self._linear(emb, f"{p}.key_projs.{hc}", bias=True), self._get(f"{p}.norm_key.{hc}.weight"), 1e-5)
            query = rms_norm(streams[:, :, hc], self._get(f"{p}.norm_query.{hc}.weight"), 1e-5)
            gate = (key * query).sum(dim=-1) * self.dim**-0.5
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim=2).unsqueeze(-1)
        value = self._linear(emb, f"{p}.value_proj", bias=True)
        main = self._get(f"{p}.unified_pre_gamma") * (value.float() * unweighted_rms_inv(value, 1e-5)).to(value.dtype)
        branch = torch.stack([self._get(f"{p}.unified_per_hc_scales.{hc}") * main for hc in range(self.hc)], dim=2)
        out = gates * main.unsqueeze(2) + branch
        return out.masked_fill(~mask[:, :, None, None], 0.0)

    # ----- indexer ----------------------------------------------------------------------
    def indexer_mask(self, L, x, q_resid, positions, cache, kv_len, attn_mask):
        p = f"model.layers.{L}.self_attn.compressed_indexer"
        B, T, _ = x.shape
        rope = self.rope["indexed_attention"]
        token_keys = rms_norm(self._linear(x, f"{p}.token_key_projection"), self._get(f"{p}.token_key_norm.weight"), self.eps)
        token_keys = cache.update_indexer(token_keys, L)
        if kv_len <= self.candidate_topk:
            return attn_mask
        idim = self.c["compressed_indexer_head_dim"]
        ih = self.c["compressed_indexer_num_attention_heads"]
        n_groups = token_keys.shape[1] // self.ratio
        grouped = token_keys[:, : n_groups * self.ratio].reshape(B, n_groups, self.ratio, idim).float()
        values = F.linear(grouped, self._get(f"{p}.key_value_projection.weight"))
        gates = F.linear(grouped, self._get(f"{p}.key_gate_projection.weight")) + self._get(f"{p}.intra_group_ape")
        pooled = (values * gates.softmax(dim=2)).sum(dim=2)
        group_keys = rms_norm(pooled, self._get(f"{p}.key_norm.weight"), self.eps).to(token_keys.dtype)
        query = self._linear(q_resid, f"{p}.query_projection").view(B, T, ih, idim)
        qcos, qsin = rope.at(positions, query.dtype)
        query = partial_rope_trailing(query, qcos, qsin, self.rope_dim)
        group_positions = cache.positions[:, : n_groups * self.ratio : self.ratio]
        gcos, gsin = rope.at(group_positions, group_keys.dtype)
        keys = partial_rope_trailing(group_keys.unsqueeze(2), gcos, gsin, self.rope_dim).squeeze(2)
        head_weights = self._linear(x, f"{p}.head_weight_projection").float() * (ih**-0.5 * idim**-0.5)
        scores = F.relu(torch.matmul(query.float(), keys.float().unsqueeze(1).transpose(-1, -2)))
        logits = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)
        query_slots = torch.arange(kv_len - T, kv_len, device=x.device)
        query_group_index = (query_slots // self.ratio).expand(B, -1)
        group_ids = torch.arange(n_groups, device=x.device)
        logits = logits.masked_fill(group_ids >= query_group_index.unsqueeze(-1), float("-inf"))
        complete_topk = min(self.group_topk - 1, n_groups)
        recalled = logits.topk(complete_topk, dim=-1).indices
        recalled = torch.cat((recalled, query_group_index.unsqueeze(-1)), dim=-1)
        offsets = torch.arange(self.ratio, device=x.device)
        tokens = (recalled.unsqueeze(-1) * self.ratio + offsets).flatten(-2).clamp(max=kv_len - 1)
        unrecalled = attn_mask.new_ones((B, T, kv_len), dtype=torch.bool)
        unrecalled.scatter_(-1, tokens, False)
        return attn_mask.masked_fill(unrecalled.unsqueeze(1), torch.finfo(attn_mask.dtype).min)

    # ----- attention ----------------------------------------------------------------------
    def attention(self, L, x, positions, cache, attn_mask):
        a = f"model.layers.{L}.self_attn"
        swa = self.c["layer_types"][L] == "sliding_attention"
        heads = self.c["swa_num_attention_heads"] if swa else self.c["num_attention_heads"]
        B, T, _ = x.shape
        rope = self.rope["sliding_attention" if swa else "indexed_attention"]
        gate_states = self._linear(x, f"{a}.linear_gate").view(B, T, heads, self.v_head)
        q_resid = rms_norm(self._linear(x, f"{a}.q_a_proj"), self._get(f"{a}.q_a_layernorm.weight"), self.eps)
        q = self._linear(q_resid, f"{a}.q_b_proj").view(B, T, heads, self.qk).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [self.c["qk_nope_head_dim"], self.rope_dim], dim=-1)
        kv = self._linear(x, f"{a}.kv_a_proj_with_mqa")
        kv_pass, k_rot = torch.split(kv, [self.kv_lora, self.rope_dim], dim=-1)
        k_pass = rms_norm(kv_pass, self._get(f"{a}.kv_a_layernorm.weight"), self.eps).view(B, 1, T, self.kv_lora)
        k_rot = k_rot.view(B, 1, T, self.rope_dim)
        cos, sin = rope.at(positions, x.dtype)
        q_rot, k_rot = apply_rope(q_rot, k_rot, cos, sin)
        k_pass, k_rot = cache.update(k_pass, k_rot, L)
        q = torch.cat((q_pass, q_rot), dim=-1)
        kv_b = self._linear(k_pass, f"{a}.kv_b_proj").view(B, k_pass.shape[2], heads, self.c["qk_nope_head_dim"] + self.v_head).transpose(1, 2)
        k_nope, v = torch.split(kv_b, [self.c["qk_nope_head_dim"], self.v_head], dim=-1)
        k = torch.cat((k_nope, k_rot.expand(-1, heads, -1, -1)), dim=-1)
        kv_len = k.shape[2]
        if not swa:
            attn_mask = self.indexer_mask(L, x, q_resid, positions, cache, kv_len, attn_mask)
        scores = torch.matmul(q, k.transpose(2, 3)) * (self.qk**-0.5)
        scores = scores + attn_mask
        sinks = self._get(f"{a}.learnable_sink_param").reshape(1, heads, 1, 1).expand(B, heads, T, 1)
        combined = torch.cat([scores, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1, dtype=torch.float32 if SOFTMAX_FP32 else combined.dtype)[..., :-1]
        out = torch.matmul(probs.to(v.dtype), v).transpose(1, 2).contiguous()
        out = out * torch.sigmoid(gate_states)
        out = out.reshape(B, T, heads * self.v_head)
        return rms_norm(self._linear(out, f"{a}.o_proj"), self._get(f"{a}.post_proj_norm.weight"), self.eps)

    # ----- MoE ----------------------------------------------------------------------------
    def mlp(self, L, x):
        m = f"model.layers.{L}.mlp"
        if self.c["mlp_layer_types"][L] == "dense":
            y = self._linear(F.silu(self._linear(x, f"{m}.gate_proj")) * self._linear(x, f"{m}.up_proj"), f"{m}.down_proj")
            return rms_norm(y, self._get(f"{m}.post_fc2_norm.weight"), self.eps)
        orig_shape = x.shape
        xf = x.reshape(-1, x.shape[-1]).float()
        logits = F.linear(xf, self._get(f"{m}.gate.weight").float())
        scores = logits.sigmoid()
        choice = scores + self._get(f"{m}.gate.e_score_correction_bias")
        k = self.c["num_experts_per_tok"]
        topk_ids = choice.topk(k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, topk_ids)
        if self.c["norm_topk_prob"]:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.c["routed_scaling_factor"]
        latent = self._linear(x.reshape(-1, x.shape[-1]), f"{m}.routed_expert_down_proj")
        E = self.c["n_routed_experts"]
        gate_up = self._get(f"{m}.experts.gate_up_proj")
        down = self._get(f"{m}.experts.down_proj")
        routed = torch.zeros_like(latent)
        flat_ids = topk_ids.reshape(-1)
        flat_w = weights.reshape(-1)
        flat_x = latent.unsqueeze(1).expand(-1, k, -1).reshape(-1, latent.shape[-1])
        for e in range(E):
            hit = (flat_ids == e).nonzero(as_tuple=True)[0]
            if hit.numel() == 0:
                continue
            gu = F.linear(flat_x[hit], gate_up[e])
            g, u = gu.chunk(2, dim=-1)
            g = g.clamp(max=self.limit)
            u = u.clamp(min=-self.limit, max=self.limit)
            y = F.linear(F.silu(g) * u, down[e]) * flat_w[hit, None]
            routed.index_add_(0, hit // k, y.to(routed.dtype))
        routed = self._linear(routed, f"{m}.routed_expert_up_proj").reshape(orig_shape)
        shared = self._linear(
            F.silu(self._linear(x, f"{m}.shared_experts.gate_proj")) * self._linear(x, f"{m}.shared_experts.up_proj"),
            f"{m}.shared_experts.down_proj",
        )
        return rms_norm(routed + shared, self._get(f"{m}.combine_output_norm.weight"), self.eps)

    # ----- iHC ----------------------------------------------------------------------------
    def ihc(self, streams, fn, base, scale):
        flat = streams.flatten(2).float()
        mixes = F.linear(flat, fn.float()) * unweighted_rms_inv(flat, self.eps)
        pre_b, post_b = base.split(self.hc)
        pre_scale, post_scale = scale.unbind(0)
        pre_logits, post_logits = mixes.split(self.hc, dim=-1)
        pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + self.c["hc_eps"]
        post = self.c["hc_magnitude"] * torch.sigmoid(post_logits * post_scale + post_b) + self.c["hc_eps"]
        out = (pre.unsqueeze(-1) * streams).sum(dim=2)
        return post, out.to(streams.dtype)

    def hc_head(self, streams):
        flat = streams.flatten(2).float()
        mixes = F.linear(flat, self._get("model.hc_head.hc_head_fn").float()) * unweighted_rms_inv(flat, self.eps)
        pre = torch.sigmoid(mixes * self._get("model.hc_head.hc_head_scale").float() + self._get("model.hc_head.hc_head_base").float()) + self.c["hc_eps"]
        return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)

    # ----- forward --------------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, cache: Cache, ngram_tokens=None, engram_mask=None):
        B, T = input_ids.shape
        kv_len = (cache.k_pass[0].shape[2] if 0 in cache.k_pass else 0) + T
        cache.update_positions(positions)
        dtype = self._get("model.embed_tokens.weight").dtype
        h = F.embedding(input_ids, self._get("model.embed_tokens.weight"))
        if ngram_tokens is None:
            ngram_tokens = self.build_ngrams(input_ids, positions)
            engram_mask = input_ids != self.c["engram_pad_id"]
        # causal mask over the full history (+ window on SWA layers)
        q_pos = positions
        k_pos = torch.arange(kv_len, device=input_ids.device).expand(B, -1)
        causal = k_pos.unsqueeze(1) <= q_pos.unsqueeze(2)  # [B, T, kv]
        base_mask = torch.zeros(B, 1, T, kv_len, dtype=dtype, device=input_ids.device)
        base_mask = base_mask.masked_fill(~causal.unsqueeze(1), torch.finfo(dtype).min)
        swa_mask = base_mask.masked_fill(
            (k_pos.unsqueeze(1) < (q_pos.unsqueeze(2) - self.window + 1)).unsqueeze(1), torch.finfo(dtype).min
        )
        streams = h.unsqueeze(2).expand(-1, -1, self.hc, -1).contiguous()
        for L in range(self.c["num_hidden_layers"]):
            p = f"model.layers.{L}"
            swa = self.c["layer_types"][L] == "sliding_attention"
            if L in self.engram_layers:
                streams = streams + self.engram_forward(L, streams, ngram_tokens, engram_mask)
            residual = streams
            post, x = self.ihc(streams, self._get(f"{p}.hc_attn_layer.hc_pre.hc_fn"), self._get(f"{p}.hc_attn_layer.hc_pre.hc_base"), self._get(f"{p}.hc_attn_layer.hc_pre.hc_scale"))
            x = rms_norm(x, self._get(f"{p}.input_layernorm.weight"), self.eps)
            y = self.attention(L, x, positions, cache, swa_mask if swa else base_mask)
            streams = (post.float().unsqueeze(-1) * y.float().unsqueeze(-2) + residual.float()).to(dtype)
            residual = streams
            post, x = self.ihc(streams, self._get(f"{p}.hc_mlp_layer.hc_pre.hc_fn"), self._get(f"{p}.hc_mlp_layer.hc_pre.hc_base"), self._get(f"{p}.hc_mlp_layer.hc_pre.hc_scale"))
            x = rms_norm(x, self._get(f"{p}.post_attention_layernorm.weight"), self.eps)
            y = self.mlp(L, x)
            streams = (post.float().unsqueeze(-1) * y.float().unsqueeze(-2) + residual.float()).to(dtype)
        hidden = rms_norm(self.hc_head(streams), self._get("model.norm.weight"), self.eps)
        return F.linear(hidden.float(), self._get("lm_head.weight").float())

    def generate_step(self, token: torch.Tensor, position: int, cache: Cache, full_ids: torch.Tensor):
        """One decode step: token [B, 1] at `position`; n-grams hashed over the full history."""
        positions = torch.full((token.shape[0], 1), position, device=token.device, dtype=torch.long)
        ngrams = self.build_ngrams(full_ids, torch.arange(full_ids.shape[1], device=token.device).expand(token.shape[0], -1))[:, -1:]
        mask = full_ids[:, -1:] != self.c["engram_pad_id"]
        return self.forward(token, positions, cache, ngram_tokens=ngrams, engram_mask=mask)


__all__ = ["Reference", "Cache"]
