"""Inkling-2-Mini's DSpark drafter (the official checkpoint's mtp.safetensors, BF16).

Transcription of maya-ik2's drafter (src/ik2/ik2_draft.cu, v1.1.73): the target leaves
4 taps a position (the stream at the start of visits 5, 23, 41 and the final normed
hidden); each draft layer softmax-mixes them (fuse_weight), RMS-norms (dflare_norm) and
projects K/V (k_proj_ctx / v_proj_ctx, per-head k_norm_ctx, NeoX RoPE theta 1e6) into a
per-request context cache. A block of 8 rows -- the anchor (the pending token, embedded
and normed as the target embeds) then 7 mask_embed rows -- runs the 3 dense layers
(Qwen3-like: per-head q/k norm, GQA 24/8, SiLU MLP), the block's rows seeing each other
and the context window; rows 1..7 of the final norm through the target's head are the
base logits. A GRU-like Markov head over [state; markov_w1[prev]; hidden; f_rnn] then
drafts the 7 tokens one by one, its latent's bias on the logits folded through the head
once (lm_head . hidden_out). Drafts only: the target verifies every token, so the bf16
gemm roundings here (Maya keeps F32 between its BF16 gemms) never reach an answer.

Pure torch, device-agnostic: the proposer owns the per-request state (the context rings
and f_rnn ride the linear-state slots); this module is the weights and the math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

# the tap visits in execution order (a looped model's sequential visit id, not its slot id)
TAP_VISITS = (5, 23, 41)
N_TAPS = 4  # the three visit streams + the final normed hidden
BLOCK = 8  # anchor + 7 mask rows
WINDOW = 1024  # the local layers' context window
RING = 2048  # the local layers' ring rows (Maya: wrap headroom)
EPS = 1e-6
THETA = 1e6
HEAD_DIM = 128

# the Markov head's 7 steps replay as one CUDA graph (Maya's IK2_DRAFT_GRAPH); =0 keeps eager
import os as _os

_DRAFT_GRAPH = _os.environ.get("FREETOKEN_IK2_DRAFT_GRAPH", "1") != "0"
# the graphed block pads each layer's context to a fixed capacity and masks the unused columns
# through a device buffer (a captured scalar comparison would bake the capture-time length)
_DRAFT_BLOCK_GRAPH = _os.environ.get("FREETOKEN_IK2_DRAFT_BLOCK_GRAPH", "1") != "0"


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 RMS norm with a learned weight (Maya's k::rms)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _rope_neox(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """NeoX half-split RoPE on [..., heads, 128] fp32, theta 1e6 (Maya's rope_neox)."""
    i = torch.arange(HEAD_DIM // 2, device=x.device, dtype=torch.float32)
    inv = THETA ** (-2.0 * i / HEAD_DIM)
    ang = positions.float().unsqueeze(-1) * inv  # [T, 64]
    sn, cs = ang.sin(), ang.cos()
    while sn.dim() < x.dim():
        sn, cs = sn.unsqueeze(-2), cs.unsqueeze(-2)
    a, b = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    return torch.cat([a * cs - b * sn, b * cs + a * sn], dim=-1)


def _gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """BF16 inputs, fp32 sums (Maya's gemm_bf); torch rounds the product to bf16 once."""
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
    local: bool  # a 1024-position window (ring) vs the whole kept context


class DSparkDraft:
    """The drafter's weights and forwards; load() reads mtp.safetensors onto ``device``."""

    def __init__(self, C: int, n_vocab: int, layers: List[DraftLayer], *,
                 alpha: torch.Tensor, norm: torch.Tensor, mask: torch.Tensor,
                 mk_dflare: torch.Tensor, w1: torch.Tensor, rnn_w: torch.Tensor,
                 rnn_b: torch.Tensor, hout_w: torch.Tensor, hout_b: torch.Tensor):
        self.C = C
        self.n_vocab = n_vocab  # the argmax range (the head's unpadded rows)
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
        self.H = layers[0].wq.shape[0] // HEAD_DIM
        self.KV = layers[0].wk.shape[0] // HEAD_DIM
        self.F = layers[0].wgu.shape[0] // 2
        self.V = w1.shape[0]  # the head's padded rows
        self._mk_graph = None
        self._mk_k = None
        self._mk_fold = None
        self._blk_graph = None
        self._ctx_caps = None

    # ------------------------------------------------------------------ load

    @staticmethod
    def check_source(path: str) -> None:
        """Fail at config time on a file that cannot give the drafter its weights."""
        import os

        from safetensors import safe_open

        if os.path.isdir(path):
            path = os.path.join(path, "mtp.safetensors")
        if not os.path.isfile(path):
            raise ValueError(f"the DSpark drafter needs mtp.safetensors (the official Inkling-2-Mini drafter); not found at {path}")
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
        need = {"fuse_weight", "norm.weight", "mask_embed", "markov_head.markov_w1.weight",
                "markov_head.rnn_proj.weight", "markov_head.hidden_out.weight"}
        if missing := sorted(need - keys):
            raise ValueError(f"{path} is not an Inkling-2 DSpark drafter: missing {missing}")

    @classmethod
    def load(cls, path: str, device: torch.device, n_vocab: int) -> "DSparkDraft":
        import os

        import safetensors.torch

        if os.path.isdir(path):
            path = os.path.join(path, "mtp.safetensors")
        sd = safetensors.torch.load_file(path, device=str(device))
        C = sd["norm.weight"].shape[0]
        n_layers = sum(1 for k in sd if k.startswith("layers.") and k.endswith(".input_layernorm.weight"))
        taps = sd["fuse_weight"].shape[1]
        if (n_layers, taps) not in ((3, 4), (5, 8)):
            raise ValueError(f"{path}: unsupported DSpark geometry ({n_layers} layers, {taps} taps)")
        # Mini (3 layers): layers 0 and 2 windowed, layer 1 sees the whole kept context;
        # the 5-layer full Inkling-2: layers 0, 1, 3, 4 windowed (Maya's local_ids)
        local = (lambda i: i != 1) if n_layers == 3 else (lambda i: i in (0, 1, 3, 4))
        bf = lambda t: t.to(torch.bfloat16)
        f32 = lambda t: t.float()
        layers = []
        for i in range(n_layers):
            b = f"layers.{i}."
            a = f"{b}self_attn."
            layers.append(DraftLayer(
                in_norm=f32(sd[b + "input_layernorm.weight"]),
                post_norm=f32(sd[b + "post_attention_layernorm.weight"]),
                dflare=f32(sd[b + "dflare_norm.weight"]),
                q_norm=f32(sd[a + "q_norm.weight"]),
                k_norm=f32(sd[a + "k_norm.weight"]),
                k_norm_ctx=f32(sd[a + "k_norm_ctx.weight"]),
                wq=bf(sd[a + "q_proj.weight"]),
                wk=bf(sd[a + "k_proj.weight"]),
                wv=bf(sd[a + "v_proj.weight"]),
                wo=bf(sd[a + "o_proj.weight"]),
                wkc=bf(sd[a + "k_proj_ctx.weight"]),
                wvc=bf(sd[a + "v_proj_ctx.weight"]),
                wgu=bf(torch.cat([sd[b + "mlp.gate_proj.weight"], sd[b + "mlp.up_proj.weight"]], dim=0)),
                wd=bf(sd[b + "mlp.down_proj.weight"]),
                local=local(i),
            ))
        return cls(
            C, n_vocab, layers,
            alpha=torch.softmax(f32(sd["fuse_weight"]), dim=-1),
            norm=f32(sd["norm.weight"]),
            mask=f32(sd["mask_embed"]),
            mk_dflare=f32(sd["markov_head.dflare_norm.weight"]),
            w1=bf(sd["markov_head.markov_w1.weight"]),
            rnn_w=bf(sd["markov_head.rnn_proj.weight"]),
            rnn_b=f32(sd["markov_head.rnn_proj.bias"]),
            hout_w=bf(sd["markov_head.hidden_out.weight"]),
            hout_b=bf(sd["markov_head.hidden_out.bias"]),
        )

    def to(self, device: torch.device) -> "DSparkDraft":
        for t in self._tensors():
            t.data = t.data.to(device)
        return self

    def _tensors(self):
        yield from (self.alpha, self.norm, self.mask, self.mk_dflare, self.w1, self.rnn_w, self.rnn_b, self.hout_w, self.hout_b)
        for l in self.layers:
            # vars() also carries the `local` flag; only the weights move and count
            yield from (t for t in vars(l).values() if isinstance(t, torch.Tensor))

    def weight_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._tensors())

    # ------------------------------------------------------------------ context

    def mix_taps(self, taps: torch.Tensor, row: int) -> torch.Tensor:
        """[m, taps, C] -> [m, C] fp32: the row's softmax mix, RMS-normed (Maya's mix_rms)."""
        mixed = torch.einsum("mtc,t->mc", taps.float(), self.alpha[row])
        return _rms(mixed, self.layers[row].dflare if row < len(self.layers) else self.mk_dflare)

    def context_rows(self, mixed: torch.Tensor, layer: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The context K/V rows of [m, C] mixed taps: bf16 projections, per-head K norm, RoPE."""
        l = self.layers[layer]
        k = _gemm(mixed, l.wkc).view(-1, self.KV, HEAD_DIM)
        v = _gemm(mixed, l.wvc).view(-1, self.KV, HEAD_DIM)
        k = _rms(k, l.k_norm_ctx)
        k = _rope_neox(k, positions)
        return k.reshape(-1, self.KV * HEAD_DIM), v.reshape(-1, self.KV * HEAD_DIM)

    def f_rnn(self, taps_row: torch.Tensor) -> torch.Tensor:
        """[taps, C] -> [C]: the Markov head's mix of the last committed position's taps."""
        return self.mix_taps(taps_row.unsqueeze(0), len(self.layers))[0]

    # ------------------------------------------------------------------ the block

    def _attn(self, l: DraftLayer, h: torch.Tensor, positions: torch.Tensor,
              ctx_k: torch.Tensor, ctx_v: torch.Tensor) -> torch.Tensor:
        """The block's 8 rows attend the context rows and each other (no causal mask)."""
        T = h.shape[0]
        q = _gemm(h, l.wq).view(T, self.H, HEAD_DIM)
        k = _gemm(h, l.wk).view(T, self.KV, HEAD_DIM)
        v = _gemm(h, l.wv).view(T, self.KV, HEAD_DIM)
        q = _rope_neox(_rms(q, l.q_norm), positions)
        k = _rope_neox(_rms(k, l.k_norm), positions)
        keys = torch.cat([ctx_k.view(-1, self.KV, HEAD_DIM), k], dim=0)
        vals = torch.cat([ctx_v.view(-1, self.KV, HEAD_DIM), v], dim=0)
        # grouped bmm: the rep queries of a KV head share its keys, no expanded copy
        rep = self.H // self.KV
        qg = q.view(T, self.KV, rep, HEAD_DIM).permute(1, 0, 2, 3).reshape(self.KV, T * rep, HEAD_DIM)
        scores = torch.bmm(qg, keys.permute(1, 0, 2).transpose(1, 2)) / math.sqrt(HEAD_DIM)
        probs = torch.softmax(scores, dim=-1)
        og = torch.bmm(probs, vals.permute(1, 0, 2))
        o = og.view(self.KV, T, rep, HEAD_DIM).permute(1, 0, 2, 3).reshape(T, self.H * HEAD_DIM)
        return _gemm(o, l.wo)

    def block_hidden(self, anchor_h: torch.Tensor, P: int,
                     ctx_k: List[torch.Tensor], ctx_v: List[torch.Tensor]) -> torch.Tensor:
        """[8, C] fp32 draft hidden: the anchor row then 7 mask rows through the dense layers."""
        device = anchor_h.device
        res = torch.stack([anchor_h.float()] + [self.mask] * (BLOCK - 1))
        positions = torch.arange(P, P + BLOCK, device=device)
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
            act = (g / (1.0 + torch.exp(-g)) * u).to(torch.bfloat16)  # Maya's silu_mul_bf16
            x = _gemm(act, l.wd)
        res = x + res
        return _rms(res, self.norm)

    # ------------------------------------------------------------------ graphed block

    def set_ctx_caps(self, caps: List[int]) -> None:
        """Per-layer context buffer capacity the graphed block pads to (the proposer's rings)."""
        self._ctx_caps = list(caps)

    def warm_graphs(self, device, fold_w, fold_b, k: int = BLOCK - 1) -> None:
        """Capture the block and Markov graphs at startup (dummy inputs) so runtime only replays."""
        if device.type != "cuda" or not _DRAFT_GRAPH:
            return
        caps = getattr(self, "_ctx_caps", None)
        if caps is not None and _DRAFT_BLOCK_GRAPH:
            kv = self.KV * HEAD_DIM
            a = torch.zeros(self.C, dtype=torch.float32, device=device)
            ck = [torch.zeros(min(8, c), kv, dtype=torch.float32, device=device) for c in caps]
            cv = [torch.zeros(min(8, c), kv, dtype=torch.float32, device=device) for c in caps]
            self._block(a, 8, ck, cv)
        hidden = torch.zeros(BLOCK, self.C, dtype=torch.float32, device=device)
        base = torch.zeros(k, self.V, dtype=torch.float32, device=device)
        frnn = torch.zeros(self.C, dtype=torch.float32, device=device)
        self._markov(hidden, base, frnn, 0, fold_w, fold_b, k)

    def _attn_buf(self, l, h, positions, ck, cv, mask, cap):
        """_attn over a padded fixed-capacity context buffer; ``mask`` [cap] is 0 / -inf per column."""
        T = h.shape[0]
        q = _gemm(h, l.wq).view(T, self.H, HEAD_DIM)
        k = _gemm(h, l.wk).view(T, self.KV, HEAD_DIM)
        v = _gemm(h, l.wv).view(T, self.KV, HEAD_DIM)
        q = _rope_neox(_rms(q, l.q_norm), positions)
        k = _rope_neox(_rms(k, l.k_norm), positions)
        keys = torch.cat([ck.view(cap, self.KV, HEAD_DIM), k], dim=0)
        vals = torch.cat([cv.view(cap, self.KV, HEAD_DIM), v], dim=0)
        rep = self.H // self.KV
        qg = q.view(T, self.KV, rep, HEAD_DIM).permute(1, 0, 2, 3).reshape(self.KV, T * rep, HEAD_DIM)
        scores = torch.bmm(qg, keys.permute(1, 0, 2).transpose(1, 2)) / math.sqrt(HEAD_DIM)
        scores[:, :, :cap] = scores[:, :, :cap] + mask.view(1, 1, cap)
        probs = torch.softmax(scores, dim=-1)
        og = torch.bmm(probs, vals.permute(1, 0, 2))
        o = og.view(self.KV, T, rep, HEAD_DIM).permute(1, 0, 2, 3).reshape(T, self.H * HEAD_DIM)
        return _gemm(o, l.wo)

    def _block_body(self):
        """The block forward on persistent buffers (graph-capturable; P and the masks on device)."""
        device = self._blk_anchor.device
        res = torch.stack([self._blk_anchor[0]] + [self.mask] * (BLOCK - 1))
        positions = self._blk_P.to(torch.int64) + torch.arange(BLOCK, device=device)
        x = res
        for i, l in enumerate(self.layers):
            if i > 0:
                res = x + res
            h = _rms(res, l.in_norm)
            x = self._attn_buf(l, h, positions, self._blk_ck[i], self._blk_cv[i],
                               self._blk_mask[i], self._ctx_caps[i])
            res = x + res
            h = _rms(res, l.post_norm)
            gu = _gemm(h, l.wgu)
            g, u = gu[:, : self.F], gu[:, self.F :]
            act = (g / (1.0 + torch.exp(-g)) * u).to(torch.bfloat16)
            x = _gemm(act, l.wd)
        res = x + res
        self._blk_out.copy_(_rms(res, self.norm))

    def _block(self, anchor_h, P, ctx_k, ctx_v):
        """block_hidden, CUDA-graph replay on the device when armed, eager elsewhere."""
        device = anchor_h.device
        caps = getattr(self, "_ctx_caps", None)
        if device.type != "cuda" or not _DRAFT_BLOCK_GRAPH or caps is None:
            return self.block_hidden(anchor_h, P, ctx_k, ctx_v)
        if self._blk_graph is None:
            kv = self.KV * HEAD_DIM
            self._blk_anchor = torch.empty(1, self.C, dtype=torch.float32, device=device)
            self._blk_P = torch.empty((), dtype=torch.int32, device=device)
            self._blk_ck = [torch.zeros(c, kv, dtype=torch.float32, device=device) for c in caps]
            self._blk_cv = [torch.zeros(c, kv, dtype=torch.float32, device=device) for c in caps]
            self._blk_mask = [torch.zeros(c, dtype=torch.float32, device=device) for c in caps]
            self._blk_out = torch.empty(BLOCK, self.C, dtype=torch.float32, device=device)
            self._blk_fill(anchor_h, P, ctx_k, ctx_v)
            # capture on the current stream: a side-stream capture picks a different cuBLAS
            # workspace and mis-rounds some context lengths (the drafts drift, accept drops)
            cur = torch.cuda.current_stream(device)
            for _ in range(3):
                self._block_body()
            cur.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._block_body()
            self._blk_graph = g
        self._blk_fill(anchor_h, P, ctx_k, ctx_v)
        self._blk_graph.replay()
        return self._blk_out.clone()

    def _blk_fill(self, anchor_h, P, ctx_k, ctx_v):
        self._blk_anchor.copy_(anchor_h.view(1, self.C))
        self._blk_P.fill_(P)
        for i in range(len(self.layers)):
            n = ctx_k[i].shape[0]
            self._blk_ck[i][:n].copy_(ctx_k[i])
            self._blk_cv[i][:n].copy_(ctx_v[i])
            # the column mask is data the graph reads, not a captured scalar comparison
            self._blk_mask[i].fill_(float("-inf"))
            self._blk_mask[i][:n].zero_()

    # ------------------------------------------------------------------ one proposal

    def propose(self, anchor: int, anchor_h: torch.Tensor, P: int,
                ctx_k: List[torch.Tensor], ctx_v: List[torch.Tensor], f_rnn: torch.Tensor,
                head_logits: Callable[[torch.Tensor], torch.Tensor],
                fold_w: torch.Tensor, fold_b: torch.Tensor, k: int = BLOCK - 1) -> torch.Tensor:
        """k draft tokens after the anchor at position P; head_logits reads the target's head
        (fp32 [rows, V] over its padded rows), fold_w [V, R] / fold_b [V] the folded Markov bias."""
        hidden = self._block(anchor_h, P, ctx_k, ctx_v)
        # Maya's to_bf16 before the head pass: an fp32 activation would upcast / dequantize
        # the whole head weight per step (a full-vocab transient, OOM on a 24 GiB rank)
        base = head_logits(hidden[1 : k + 1].to(torch.bfloat16)).float()[:, : self.V]
        toks = self._markov(hidden, base, f_rnn, anchor, fold_w, fold_b, k)
        return toks

    def _markov_body(self, hidden, base, f_rnn, toks, state, fold_w, fold_b, k):
        """The 7-step Markov chain on persistent buffers (toks[0] preset); state zeroed here.

        Identical arithmetic to the eager loop: the GRU-like cell over [state; w1[prev];
        hidden; f_rnn], the folded head bias, argmax over the real vocab. Graph-capturable
        (fixed shapes, device-only token feedback, no host sync)."""
        state.zero_()
        for t in range(k):
            # index_select (not w1[toks[t]]): the 0-dim advanced index is not graph-capturable
            prev = torch.index_select(self.w1, 0, toks[t : t + 1])[0].float()
            rnn_in = torch.cat([state, prev, hidden[t + 1], f_rnn]).to(torch.bfloat16)
            proj = _gemm(rnn_in.unsqueeze(0), self.rnn_w)[0] + self.rnn_b
            gr, cr, orw = proj[: self.R], proj[self.R : 2 * self.R], proj[2 * self.R :]
            gate = torch.sigmoid(gr)
            state = gate * state + (1.0 - gate) * torch.tanh(cr)
            latent = torch.tanh(orw)
            bias = torch.mv(fold_w, latent.to(fold_w.dtype)).float() + fold_b
            scores = base[t] + bias
            toks[t + 1] = scores[: self.n_vocab].argmax()

    def _markov(self, hidden, base, f_rnn, anchor, fold_w, fold_b, k):
        """k draft tokens after the anchor; CUDA-graph replay on the device, eager elsewhere."""
        device = hidden.device
        if device.type != "cuda" or not _DRAFT_GRAPH:
            state = torch.zeros(self.R, device=device, dtype=torch.float32)
            toks = torch.empty(k + 1, dtype=torch.int64, device=device)
            toks[0] = anchor
            self._markov_body(hidden, base, f_rnn, toks, state, fold_w, fold_b, k)
            return toks[1:]
        g = self._mk_graph
        if g is None or self._mk_k != k or self._mk_fold is not fold_w:
            self._mk_hidden = torch.empty(BLOCK, self.C, dtype=torch.float32, device=device)
            self._mk_base = torch.empty(k, self.V, dtype=torch.float32, device=device)
            self._mk_frnn = torch.empty(self.C, dtype=torch.float32, device=device)
            self._mk_toks = torch.empty(k + 1, dtype=torch.int64, device=device)
            self._mk_state = torch.empty(self.R, dtype=torch.float32, device=device)
            self._mk_k = k
            self._mk_fold = fold_w
            self._mk_hidden.copy_(hidden)
            self._mk_base.copy_(base)
            self._mk_frnn.copy_(f_rnn)
            self._mk_toks[0] = anchor
            # warm up on a side stream so cuBLAS workspaces / autotune settle before capture
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._markov_body(self._mk_hidden, self._mk_base, self._mk_frnn,
                                      self._mk_toks, self._mk_state, fold_w, fold_b, k)
            torch.cuda.current_stream(device).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                self._markov_body(self._mk_hidden, self._mk_base, self._mk_frnn,
                                  self._mk_toks, self._mk_state, fold_w, fold_b, k)
            self._mk_graph = g
        self._mk_hidden.copy_(hidden)
        self._mk_base.copy_(base)
        self._mk_frnn.copy_(f_rnn)
        self._mk_toks[0] = anchor
        g.replay()
        return self._mk_toks[1:].clone()

    def fold(self, head_logits: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(fold_w [V, R] fp16, fold_b [V] fp16): the Markov bias through the target's head,
        computed on the head's own (possibly quantized) projection, Maya's chunked F32 fold."""
        # row chunks keep the pass inside the W8A16 head's fused decode window (M <= 32):
        # one R-row call dequantizes the whole head (a full-vocab fp32 transient, OOM on
        # a 24 GiB rank). The projection is row-independent, so the fold is unchanged.
        w = torch.cat([head_logits(c).float()
                       for c in self.hout_w.t().to(torch.bfloat16).split(32)], dim=0)  # [R, V]
        b = head_logits(self.hout_b.unsqueeze(0).to(torch.bfloat16)).float()[0]  # [V]
        return w.t().to(torch.float16).contiguous(), b.to(torch.float16)


__all__ = ["DSparkDraft", "DraftLayer", "TAP_VISITS", "N_TAPS", "BLOCK", "WINDOW", "RING"]
