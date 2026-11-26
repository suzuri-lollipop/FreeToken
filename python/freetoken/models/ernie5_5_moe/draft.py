"""ERNIE 5.5's bundled DSpark drafter (the checkpoint's ``model.mtp.*`` tensors, BF16).

The same DSpark math as the inkling2 drafter (see models/inkling2/draft.py): the target
leaves taps a position (the contracted mHC stream at each ``dspark_target_layer_ids``
layer and the final normed hidden); each draft layer softmax-mixes them (fuse_weight),
RMS-norms (dflare_norm) and projects K/V (k_proj_ctx / v_proj_ctx, per-head k_norm_ctx,
NeoX RoPE) into a per-request context cache. A block of rows -- the anchor (the pending
token, embedded and normed as the target embeds) then mask_embed rows -- runs the dense
draft layers, the block's rows seeing each other and the context window; the non-anchor
rows of the final norm through the target's head are the base logits. A GRU-like Markov
head over [state; markov_w1[prev]; hidden; f_rnn] then drafts the tokens one by one, its
latent's bias on the logits folded through the head once (lm_head . hidden_out).

Drafts only: the target verifies every token, so the bf16 gemm roundings here never reach
an answer. Pure torch, device-agnostic: the proposer owns the per-request state (the
context rings and f_rnn ride the linear-state slots); this module is the weights and math.
Unlike inkling2's separate mtp.safetensors, ERNIE bundles the drafter under model.mtp.* in
the main checkpoint, so load() reads through the family's shard reader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

EPS = 1e-6


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(x.to(torch.bfloat16), w).float()


@dataclass
class DraftLayer:
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    dflare: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    k_norm_ctx: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    wkc: torch.Tensor
    wvc: torch.Tensor
    wgu: torch.Tensor  # gate rows then up rows
    wd: torch.Tensor
    local: bool  # a windowed ring vs the whole kept context


class Ernie55DSparkDraft:
    """The drafter's weights and forwards; load() reads model.mtp.* through the reader."""

    def __init__(self, C: int, n_vocab: int, layers: List[DraftLayer], *,
                 alpha: torch.Tensor, norm: torch.Tensor, mask: torch.Tensor,
                 mk_dflare: torch.Tensor, w1: torch.Tensor, rnn_w: torch.Tensor,
                 rnn_b: torch.Tensor, hout_w: torch.Tensor, hout_b: torch.Tensor,
                 block: int, head_dim: int, theta: float, window: int, ring: int):
        self.C = C
        self.n_vocab = n_vocab
        self.layers = layers
        self.alpha = alpha  # [layers+1, taps] fp32, softmax over the taps
        self.norm = norm
        self.mask = mask
        self.mk_dflare = mk_dflare
        self.w1 = w1  # [V, R] bf16
        self.rnn_w = rnn_w  # [3R, 2R+2C] bf16
        self.rnn_b = rnn_b  # [3R] fp32
        self.hout_w = hout_w  # [C, R] bf16, the fold's source
        self.hout_b = hout_b  # [C] bf16
        self.R = w1.shape[1]
        self.HEAD_DIM = head_dim
        self.THETA = theta
        self.BLOCK = block  # anchor + (block-1) mask rows
        self.WINDOW = window
        self.RING = ring
        self.H = layers[0].wq.shape[0] // head_dim
        self.KV = layers[0].wk.shape[0] // head_dim
        self.F = layers[0].wgu.shape[0] // 2
        self.V = w1.shape[0]  # the head's padded rows
        self.N_TAPS = alpha.shape[1]

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, reader, args, n_vocab: int) -> "Ernie55DSparkDraft":
        """Read model.mtp.* through the family's shard reader (a .get(name) callable)."""
        get = reader.get if hasattr(reader, "get") else reader.__getitem__
        b0 = "model.mtp."
        C = get(b0 + "norm.weight").shape[0]
        n_layers = args.dspark_num_layers
        bf = lambda t: t.to(torch.bfloat16)
        f32 = lambda t: t.float()
        # the windowed draft layers: all but the middle full-context one (inkling2 parity)
        local = lambda i: i != n_layers // 2
        layers = []
        for i in range(n_layers):
            b = f"{b0}layers.{i}."
            a = f"{b}self_attn."
            layers.append(DraftLayer(
                in_norm=f32(get(b + "input_layernorm.weight")),
                post_norm=f32(get(b + "post_attention_layernorm.weight")),
                dflare=f32(get(b + "dflare_norm.weight")),
                q_norm=f32(get(a + "q_norm.weight")),
                k_norm=f32(get(a + "k_norm.weight")),
                k_norm_ctx=f32(get(a + "k_norm_ctx.weight")),
                wq=bf(get(a + "q_proj.weight")),
                wk=bf(get(a + "k_proj.weight")),
                wv=bf(get(a + "v_proj.weight")),
                wo=bf(get(a + "o_proj.weight")),
                wkc=bf(get(a + "k_proj_ctx.weight")),
                wvc=bf(get(a + "v_proj_ctx.weight")),
                wgu=bf(torch.cat([get(b + "mlp.gate_proj.weight"), get(b + "mlp.up_proj.weight")], dim=0)),
                wd=bf(get(b + "mlp.down_proj.weight")),
                local=local(i),
            ))
        return cls(
            C, n_vocab, layers,
            alpha=torch.softmax(f32(get(b0 + "fuse_weight")), dim=-1),
            norm=f32(get(b0 + "norm.weight")),
            mask=f32(get(b0 + "mask_embed")),
            mk_dflare=f32(get(b0 + "markov_head.dflare_norm.weight")),
            w1=bf(get(b0 + "markov_head.markov_w1.weight")),
            rnn_w=bf(get(b0 + "markov_head.rnn_proj.weight")),
            rnn_b=f32(get(b0 + "markov_head.rnn_proj.bias")),
            hout_w=bf(get(b0 + "markov_head.hidden_out.weight")),
            hout_b=bf(get(b0 + "markov_head.hidden_out.bias")),
            block=args.dspark_block_size + 1,
            head_dim=args.dspark_head_dim,
            theta=args.dspark_rope_theta,
            window=args.dspark_window,
            ring=args.dspark_ring,
        )

    def to(self, device: torch.device) -> "Ernie55DSparkDraft":
        for t in self._tensors():
            t.data = t.data.to(device)
        return self

    def _tensors(self):
        yield from (self.alpha, self.norm, self.mask, self.mk_dflare, self.w1, self.rnn_w, self.rnn_b, self.hout_w, self.hout_b)
        for l in self.layers:
            yield from (t for t in vars(l).values() if isinstance(t, torch.Tensor))

    def weight_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._tensors())

    # ------------------------------------------------------------------ context

    def _rope_neox(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hd = self.HEAD_DIM
        i = torch.arange(hd // 2, device=x.device, dtype=torch.float32)
        inv = self.THETA ** (-2.0 * i / hd)
        ang = positions.float().unsqueeze(-1) * inv
        sn, cs = ang.sin(), ang.cos()
        while sn.dim() < x.dim():
            sn, cs = sn.unsqueeze(-2), cs.unsqueeze(-2)
        a, b = x[..., : hd // 2], x[..., hd // 2 :]
        return torch.cat([a * cs - b * sn, b * cs + a * sn], dim=-1)

    def mix_taps(self, taps: torch.Tensor, row: int) -> torch.Tensor:
        mixed = torch.einsum("mtc,t->mc", taps.float(), self.alpha[row])
        return _rms(mixed, self.layers[row].dflare if row < len(self.layers) else self.mk_dflare)

    def context_rows(self, mixed: torch.Tensor, layer: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l = self.layers[layer]
        k = _gemm(mixed, l.wkc).view(-1, self.KV, self.HEAD_DIM)
        v = _gemm(mixed, l.wvc).view(-1, self.KV, self.HEAD_DIM)
        k = _rms(k, l.k_norm_ctx)
        k = self._rope_neox(k, positions)
        return k.reshape(-1, self.KV * self.HEAD_DIM), v.reshape(-1, self.KV * self.HEAD_DIM)

    def f_rnn(self, taps_row: torch.Tensor) -> torch.Tensor:
        return self.mix_taps(taps_row.unsqueeze(0), len(self.layers))[0]

    # ------------------------------------------------------------------ the block

    def _attn(self, l: DraftLayer, h: torch.Tensor, positions: torch.Tensor,
              ctx_k: torch.Tensor, ctx_v: torch.Tensor) -> torch.Tensor:
        T = h.shape[0]
        hd = self.HEAD_DIM
        q = _gemm(h, l.wq).view(T, self.H, hd)
        k = _gemm(h, l.wk).view(T, self.KV, hd)
        v = _gemm(h, l.wv).view(T, self.KV, hd)
        q = self._rope_neox(_rms(q, l.q_norm), positions)
        k = self._rope_neox(_rms(k, l.k_norm), positions)
        keys = torch.cat([ctx_k.view(-1, self.KV, hd), k], dim=0)
        vals = torch.cat([ctx_v.view(-1, self.KV, hd), v], dim=0)
        rep = self.H // self.KV
        qg = q.view(T, self.KV, rep, hd).permute(1, 0, 2, 3).reshape(self.KV, T * rep, hd)
        scores = torch.bmm(qg, keys.permute(1, 0, 2).transpose(1, 2)) / math.sqrt(hd)
        probs = torch.softmax(scores, dim=-1)
        og = torch.bmm(probs, vals.permute(1, 0, 2))
        o = og.view(self.KV, T, rep, hd).permute(1, 0, 2, 3).reshape(T, self.H * hd)
        return _gemm(o, l.wo)

    def block_hidden(self, anchor_h: torch.Tensor, P: int,
                     ctx_k: List[torch.Tensor], ctx_v: List[torch.Tensor]) -> torch.Tensor:
        device = anchor_h.device
        res = torch.stack([anchor_h.float()] + [self.mask] * (self.BLOCK - 1))
        positions = torch.arange(P, P + self.BLOCK, device=device)
        x = res
        for i, l in enumerate(self.layers):
            if i > 0:
                res = x + res
            h = _rms(res, l.in_norm)
            x = self._attn(l, h, positions, ctx_k[i], ctx_v[i])
            res = x + res
            h = _rms(res, l.post_norm)
            gu = _gemm(h, l.wgu)
            g, u = gu[:, : self.F], gu[:, self.F :]
            act = (g / (1.0 + torch.exp(-g)) * u).to(torch.bfloat16)
            x = _gemm(act, l.wd)
        res = x + res
        return _rms(res, self.norm)

    # ------------------------------------------------------------------ one proposal

    def propose(self, anchor: int, anchor_h: torch.Tensor, P: int,
                ctx_k: List[torch.Tensor], ctx_v: List[torch.Tensor], f_rnn: torch.Tensor,
                head_logits: Callable[[torch.Tensor], torch.Tensor],
                fold_w: torch.Tensor, fold_b: torch.Tensor, k: int | None = None) -> torch.Tensor:
        if k is None:
            k = self.BLOCK - 1
        hidden = self.block_hidden(anchor_h, P, ctx_k, ctx_v)
        base = head_logits(hidden[1 : k + 1].to(torch.bfloat16)).float()[:, : self.V]
        state = torch.zeros(self.R, device=hidden.device, dtype=torch.float32)
        toks = torch.empty(k + 1, dtype=torch.int64, device=hidden.device)
        toks[0] = anchor
        for t in range(k):
            rnn_in = torch.cat([state, self.w1[toks[t]].float(), hidden[t + 1], f_rnn]).to(torch.bfloat16)
            proj = _gemm(rnn_in.unsqueeze(0), self.rnn_w)[0] + self.rnn_b
            gr, cr, orw = proj[: self.R], proj[self.R : 2 * self.R], proj[2 * self.R :]
            gate = torch.sigmoid(gr)
            state = gate * state + (1.0 - gate) * torch.tanh(cr)
            latent = torch.tanh(orw)
            bias = torch.mv(fold_w, latent.to(fold_w.dtype)).float() + fold_b
            scores = base[t] + bias
            toks[t + 1] = scores[: self.n_vocab].argmax()
        return toks[1:]

    def fold(self, head_logits: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.cat([head_logits(c).float()
                       for c in self.hout_w.t().to(torch.bfloat16).split(32)], dim=0)  # [R, V]
        b = head_logits(self.hout_b.unsqueeze(0).to(torch.bfloat16)).float()[0]  # [V]
        return w.t().to(torch.float16).contiguous(), b.to(torch.float16)


__all__ = ["Ernie55DSparkDraft", "DraftLayer"]