"""ERNIE 5.5 attention: HCA (DSV4 window+compressed MQA) and DSA (token-level MLA).

HCA layers are the DSV4 ratio-128 attention (same compressor, same paged pools,
same YaRN rope at ``compress_rope_theta``) plus the ERNIE output path: a sigmoid
output gate from the q_lora, the VHA head post-mix ``o' = (I + V U^T) o``, the
grouped ``wo_a`` bmm and the gated ``wo_b``.

DSA layers are latent-KV MLA with kv_b absorption (GLM-DSA precedent) over the
pool's full-history latent slabs; the lightning indexer scores tokens older
than the trailing 128-window and the backend always appends the window to the
selection. ``softmax_offset`` in the checkpoint is the per-head attention sink.
The indexer's rope is NeoX half-split on the FIRST 64 of the 128 index dims at
plain ``rope_theta`` (the main MLA rope is interleaved); the reference's
Hadamard + FP8/FP4 indexer quant are orthonormal / quantization rewrites, so
bf16 scoring here is the exact-math baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.kernel.triton.dsv4.norm import rms_norm
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.quantization import without_online
from freetoken.layers.rotary import get_rope
from freetoken.models.deepseek_v4.compress import Compressor
from freetoken.models.deepseek_v4.layers import get_compress_topk_idxs, get_window_topk_idxs
from freetoken.models.deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode, get_freqs_cis

from .args import Ernie55Args


class _VHAPostMix(BaseOP):
    """Head mixing ``o[h] <- sum_g (I + V U^T)[h, g] * o[g]`` (rank-16 residual).

    U / V load from the checkpoint; the folded matrix is derived post-load
    (``prepare_for_runtime``) and recomputed from U/V so reloads stay idempotent.
    """

    def __init__(self, n_heads: int, rank: int = 16):
        self.n_heads = n_heads
        self.vha_U = torch.empty(n_heads, rank, dtype=torch.bfloat16)
        self.vha_V = torch.empty(n_heads, rank, dtype=torch.bfloat16)
        self._m: torch.Tensor | None = None

    def fold(self) -> None:
        eye = torch.eye(self.n_heads, dtype=self.vha_U.dtype, device=self.vha_U.device)
        m = torch.addmm(eye, self.vha_V, self.vha_U.t())
        if self._m is None:
            self._m = m
        else:
            self._m.copy_(m)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        if self._m is None:
            self.fold()
        return torch.matmul(self._m, o)


class Ernie55HCAAttention(BaseOP):
    """Heavily Compressed Attention: MQA latent K==V over window + compressed blocks."""

    def __init__(self, layer_id: int, args: Ernie55Args, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = torch.empty(self.n_heads, dtype=torch.float32)
        self.wq_a = LinearReplicated(self.dim, self.q_lora_rank, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_a")
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = LinearReplicated(self.q_lora_rank, self.n_heads * self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b")
        self.wkv = LinearReplicated(self.dim, self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # grouped wo_a: one [o_lora_rank, K] matrix per output group, bf16 einsum
        # (the reader dequantizes the checkpoint's block-fp8; DSV4 precedent).
        wo_a_rows = self.n_groups * args.o_lora_rank
        wo_a_k = self.n_heads * self.head_dim // self.n_groups
        self.wo_a = torch.empty(wo_a_rows, wo_a_k, dtype=torch.bfloat16)
        self.wo_b = LinearReplicated(self.n_groups * args.o_lora_rank, self.dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wo_b")
        self.o_gate = LinearReplicated(self.q_lora_rank, self.n_groups * args.o_lora_rank, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.o_gate")
        self.vha = _VHAPostMix(self.n_heads)
        self.softmax_scale = self.head_dim ** -0.5

        # compressor weights ship bf16; no quant method (the checkpoint's fp8 config
        # does not cover them -- DSV4 serves them through without_online, same effect)
        self.compressor = Compressor(args, self.compress_ratio, self.head_dim, quant_config=None, prefix=f"{prefix}.compressor")
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, args.original_seq_len,
            args.compress_rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, pool, device: torch.device) -> None:
        self._freqs_cis = get_freqs_cis(*self._freqs_params, device)
        self.compressor.bind_paged(pool, self.layer_id, self._freqs_cis, device, tier="attn")

    def reset(self) -> None:
        self.compressor.reset()

    def prepare_for_runtime(self) -> None:
        self.vha.fold()

    def _wo(self, o: torch.Tensor, gate: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        z = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b.forward(z * gate)

    def forward_ragged(self, x, segments, flat_positions):
        """Ragged batched prefill over the flat [1, T, dim] stream (DSV4 contract,
        text-only: no image spans)."""
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        _, T, _ = x.size()
        if len(segments) == 1:
            freqs = self._freqs_cis[segments[0][3]:segments[0][3] + T]
        else:
            freqs = self._freqs_cis.index_select(0, flat_positions)

        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rms_norm(q, None, self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)
        gate = torch.sigmoid(self.o_gate.forward(qr))

        kv = self.wkv.forward(x)
        kv = self.kv_norm.forward(kv)
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv[..., :-rd], 64)

        win_parts: list[torch.Tensor] = []
        cmp_parts: list[torch.Tensor] = []
        max_c = 0
        for off, n, ti, start_pos in segments:
            win_global, cmp_global = self._prefill_segment(
                x[:, off:off + n], kv[0, off:off + n], ti, start_pos, n
            )
            win_parts.append(win_global)
            cmp_parts.append(cmp_global)
            max_c = max(max_c, cmp_global.shape[-1])

        n_window = win if len(segments) > 1 else win_parts[0].shape[-1]
        flat = []
        for i, (off, n, ti, _start) in enumerate(segments):
            wg = win_parts[i]
            if wg.shape[-1] < n_window:
                wg = F.pad(wg, (0, n_window - wg.shape[-1]), value=-1)
            cg = cmp_parts[i]
            if cg.shape[-1] < max_c:
                cg = F.pad(cg, (0, max_c - cg.shape[-1]), value=-1)
            flat.append(torch.cat([wg, cg], dim=-1))
        topk_idxs = (flat[0] if len(flat) == 1 else torch.cat(flat, dim=1)).int()

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            has_compression=True,
        )
        o = self.vha.forward(o)
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, gate, 1, T)

    def _prefill_segment(self, x_seg, kv_seg, ti: int, start_pos: int, n: int):
        win, ratio, device = self.window_size, self.compress_ratio, x_seg.device
        end = start_pos + n
        slots = self.attn.window_slots_of(ti, start_pos, end)
        self.attn.store_window(kv_seg, self.layer_id, slots)
        if start_pos == 0:
            win_cols = get_window_topk_idxs(win, 1, n, 0).to(device)
            win_global = self.attn.win_cols_to_global(win_cols, slots)
            self.reset()
            blocks = get_compress_topk_idxs(ratio, 1, n, 0, 0).to(device)
            self.compressor.forward(x_seg, 0, slots, ti=ti)
        else:
            w_lo = max(0, start_pos - win + 1)
            ws_pool = self.attn.window_slots_of(ti, w_lo, end)
            abs_p = start_pos + torch.arange(n, device=device).unsqueeze(1)
            cand = (abs_p - win + 1).clamp(min=w_lo) + torch.arange(win, device=device)
            win_cols = torch.where(cand > abs_p, -1, cand - w_lo).unsqueeze(0)
            win_global = self.attn.win_cols_to_global(win_cols, ws_pool)
            tail_ws = int(self.attn.window_slots_of(ti, start_pos - 1, start_pos).item())
            blocks = self._compress_topk_extend(n, start_pos, end, device)
            self.compressor.forward(x_seg, start_pos, slots, tail_window_slot=tail_ws, ti=ti)
        return win_global, self.attn.blocks_to_global(blocks, ratio, ti=ti)

    def _compress_topk_extend(self, seqlen, start_pos, end, device):
        ratio = self.compress_ratio
        n_blocks = end // ratio
        blk = torch.arange(n_blocks, device=device).unsqueeze(0)
        abs_p1 = (start_pos + torch.arange(seqlen, device=device) + 1).unsqueeze(1)
        valid = blk < (abs_p1 // ratio)
        return torch.where(valid, blk, -1).unsqueeze(0)

    def decode_step(self, x, pos, rows, cmp_stage_cap, wctx=None):
        """Batched single-token attention (eager / graph), DSV4 decode contract."""
        B = x.size(0)
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        n_cmp_stage = (cmp_stage_cap + 1) // ratio
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        window_slots, prev_window_slots, window_slots_topk = wctx
        freqs_t = self._freqs_cis.index_select(0, pos)

        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rms_norm(q, None, self.eps)
        apply_rotary_emb_decode(q[..., -rd:], freqs_t)
        gate = torch.sigmoid(self.o_gate.forward(qr))

        kv = self.wkv.forward(x)
        kv = self.kv_norm.forward(kv)
        apply_rotary_emb_decode(kv[..., -rd:], freqs_t)
        act_quant_fp8_inplace(kv[..., :-rd], 64)
        self.attn.store_window(kv.view(B, -1), self.layer_id, window_slots)

        # The rolling carry is sequential: verify row w reads the carry row w-1 just wrote.
        # A batched k+1-row verify would read all carries before any write, so step the
        # compressor one verify row at a time (each call is bs co-tenant rows, disjoint pages).
        width = get_global_ctx().batch.rows_per_req or 1
        if width > 1:
            bs = B // width
            for w in range(width):
                idx = torch.arange(bs, device=device) * width + w
                self.compressor.decode_step(x[idx], pos[idx], prev_window_slots[idx], window_slots[idx], rows[idx])
            # save the verify rows' compressor inputs so commit_verify can roll the carry
            # ring back to the accepted frontier (rejected drafts overwrote it)
            self._verify_carry = (x.detach().clone(), pos, prev_window_slots, window_slots, rows, width)
        else:
            self.compressor.decode_step(x, pos, prev_window_slots, window_slots, rows)
        valid = (pos + 1) // ratio
        blk = torch.arange(n_cmp_stage, device=device)
        col_valid = blk[None, :] < valid[:, None]
        blocks = torch.where(col_valid, blk[None, :], -1).view(B, 1, n_cmp_stage)
        compress_idxs = self.attn.blocks_to_global(blocks, ratio, rows=rows)
        topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1).int()
        cmp_counts = valid.clamp(max=compress_idxs.shape[-1]).to(torch.int32).view(B, 1)

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, win, self.attn_sink, self.softmax_scale,
            cmp_counts=cmp_counts, has_compression=True,
        )
        o = self.vha.forward(o)
        apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)
        return self._wo(o, gate, B, 1)

    def rollback_carry(self, counts) -> None:
        """After a verify, re-run the compressor for each request's ACCEPTED rows so the
        rolling carry ring ends at the committed frontier (the rejected draft rows' carry
        writes overwrote it). Row 0 reads the pre-verify carry (its prev page was not touched
        by the verify), and the accepted rows chain forward from there."""
        saved = getattr(self, "_verify_carry", None)
        if saved is None:
            return
        x, pos, prev_window_slots, window_slots, rows, width = saved
        bs = x.size(0) // width
        # restore the pre-verify carry ring first: the rejected draft rows' writes overwrote
        # the shared per-page block, so replaying accepted rows must start from the saved block
        cb = getattr(self.compressor, "_verify_carry_block", None)
        if cb is not None:
            block, cb_prev = cb
            self.attn.write_carry_blocks(self.layer_id, self.compressor.tier, cb_prev, self.compressor.ring_size, block)
        for i in range(bs):
            n = int(counts[i])
            for w in range(n):
                idx = i * width + w
                self.compressor.decode_step(
                    x[idx:idx + 1], pos[idx:idx + 1], prev_window_slots[idx:idx + 1],
                    window_slots[idx:idx + 1], rows[idx:idx + 1],
                )
        self._verify_carry = None


class _IdxLayerNorm(BaseOP):
    """LayerNorm with bias (the indexer's k_norm; RMSNorm everywhere else)."""

    def __init__(self, size: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.empty(size, dtype=torch.bfloat16)
        self.bias = torch.empty(size, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class Ernie55Indexer(BaseOP):
    """Lightning indexer scoring the tokens older than the sliding window."""

    def __init__(self, args: Ernie55Args, *, quant_config=None, prefix: str = ""):
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.dim = args.dim
        self.wq_b = LinearReplicated(args.mla_q_lora_rank, self.n_heads * self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b")
        self.wk = LinearReplicated(self.dim, self.head_dim, has_bias=False)
        self.k_norm = _IdxLayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = LinearReplicated(self.dim, self.n_heads, has_bias=False)
        # NeoX half-split rope on the FIRST qk_rope dims of the 128-dim index heads,
        # plain rope_theta (no YaRN -- that is HCA-only).
        self._rope = get_rope(
            head_dim=self.head_dim,
            rotary_dim=args.mla_qk_rope_head_dim,
            max_position=args.max_seq_len,
            base=args.rope_theta,
            is_neox=True,
        )
        # Optional FlashIndexer head routing (top-8 of 64, raw-gate reweight).
        self.linear_gate: LinearReplicated | None = None
        self.flashindexer_bias: torch.Tensor | None = None
        if args.enable_flashindexer:
            self.linear_gate = LinearReplicated(self.dim, self.n_heads, has_bias=False)
            self.flashindexer_bias = torch.empty(self.n_heads, dtype=torch.float32)
        self._topk_heads = args.flashindexer_topk_heads

    def compute(self, x, q_resid, positions):
        from freetoken.attention.dsa import DSAIndexerInputs

        t = x.shape[0]
        q = self.wq_b.forward(q_resid).view(t, self.n_heads * self.head_dim)
        k = self.k_norm.forward(self.wk.forward(x)).view(t, self.head_dim)
        q, k = self._rope.forward(positions, q, k)
        q = q.view(t, self.n_heads, self.head_dim)
        w = self.weights_proj.forward(x).float() * (self.n_heads ** -0.5)
        if self.linear_gate is not None:
            q, w = self._select_heads(x, q, w)
        return DSAIndexerInputs(q=q, k=k, w=w)

    def _select_heads(self, x, q, w):
        """FlashIndexer: rank heads by raw gate + bias, keep top-k, reweight by the
        RAW gate value. Padded to 16 heads (the scoring kernel's tl.dot minimum)
        with zero weights -- an exact no-op contribution."""
        gates = self.linear_gate.forward(x).float()
        scores = gates + self.flashindexer_bias
        k = self._topk_heads
        idx = scores.topk(k, dim=-1).indices  # [T, k]
        q_sel = q.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        w_sel = w.gather(1, idx) * gates.gather(1, idx)
        pad = 16 - k
        if pad > 0:
            q_sel = F.pad(q_sel, (0, 0, 0, pad))
            w_sel = F.pad(w_sel, (0, pad))
        return q_sel, w_sel


class Ernie55DSAAttention(BaseOP):
    """Sparse MLA with a token-level indexer, attention sink, VHA and output gate."""

    def __init__(self, layer_id: int, args: Ernie55Args, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.dim = args.dim
        self.num_heads = args.mla_n_heads
        self.qk_nope_head_dim = args.mla_qk_nope_head_dim
        self.qk_rope_head_dim = args.mla_qk_rope_head_dim
        self.qk_head_dim = args.mla_qk_head_dim
        self.v_head_dim = args.mla_v_head_dim
        self.kv_lora_rank = args.mla_kv_lora_rank
        self.q_lora_rank = args.mla_q_lora_rank
        self.eps = args.norm_eps

        self.q_a_proj = LinearReplicated(self.dim, self.q_lora_rank, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.q_a_proj")
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.q_b_proj = LinearReplicated(self.q_lora_rank, self.num_heads * self.qk_head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.q_b_proj")
        self.kv_a_proj_with_mqa = LinearReplicated(
            self.dim, self._kv_a_padded(), has_bias=False, quant_config=quant_config, prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=self.eps)
        # MLA absorption reads kv_b as bf16 bmm operands (GLM-DSA precedent)
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False, quant_config=without_online(quant_config), prefix=f"{prefix}.kv_b_proj",
        )
        self.kv_b_proj.w8a16_decode_ok = False
        self.o_proj = LinearReplicated(self.num_heads * self.v_head_dim, self.dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.o_gate = LinearReplicated(self.q_lora_rank, self.num_heads * self.v_head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.o_gate")
        self.vha = _VHAPostMix(self.num_heads)
        self.attn_sink = torch.empty(self.num_heads, dtype=torch.float32)
        self.indexer = Ernie55Indexer(args, quant_config=quant_config, prefix=f"{prefix}.indexer")
        # Interleaved plain rope (no YaRN) on the 64-dim rope halves.
        self.rope = get_rope(
            head_dim=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=args.max_seq_len,
            base=args.rope_theta,
            is_neox=False,
        )
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_a_padded(self) -> int:
        # block-fp8 needs N divisible by 128; the checkpoint pads the SCALE rows
        # (ceil(576/128)=5) but stores 576 weight rows -- pad the weight at load
        # and slice the forward output back to the latent width.
        return -(-(self.kv_lora_rank + self.qk_rope_head_dim) // 128) * 128

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            self._w_uk = w[:, : self.qk_nope_head_dim, :].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None
        self.vha.fold()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flat [T, dim] forward -- prefill and decode alike (the backend owns the
        per-phase addressing)."""
        ctx = get_global_ctx()
        t = x.shape[0]
        w_uk, w_uv = self._kv_b()

        q_a_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        gate = torch.sigmoid(self.o_gate.forward(q_a_resid))
        q = self.q_b_proj.forward(q_a_resid).view(t, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa.forward(x)[..., : self.kv_lora_rank + self.qk_rope_head_dim]
        c_kv, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm.forward(c_kv)

        rope_dim = self.qk_rope_head_dim
        q_rope = q_rope.reshape(t, self.num_heads * rope_dim)
        k_rope = k_rope.reshape(t, rope_dim)
        q_rope, k_rope = self.rope.forward(ctx.batch.positions, q_rope, k_rope)
        q_rope = q_rope.view(t, self.num_heads, rope_dim)

        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(0, 1)
        indexer_inputs = self.indexer.compute(x, q_a_resid, ctx.batch.positions)

        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(), q_rope.contiguous(), c_kv.contiguous(),
            k_rope.contiguous(), self.layer_id, ctx.batch,
            indexer_inputs=indexer_inputs, attn_sink=self.attn_sink,
        )  # [T, H, kv_lora_rank]
        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        o = self.vha.forward(o)
        return self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim) * gate)


__all__ = ["Ernie55HCAAttention", "Ernie55DSAAttention", "Ernie55Indexer"]
