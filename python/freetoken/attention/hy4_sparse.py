"""Hy4-flash DSA backend: compressed-indexer selection + SWA windows over the MLA pool.

Extends the glm5_next kpool backend (same shadow-slab / tail-ring storage, same
graph-safe write plan) with Hy4's differences:

* pooling reads the model's VALUE/GATE projections of the normed token keys, norms
  the pooled group key (fp32, before the bf16 store) and rotates its trailing 64
  channels at the group's FIRST token position -- the reference rotates at scoring
  time, the same value once per group instead of once per score.
* selection: a query recalls the ``group_topk - 1`` (511) best COMPLETE groups
  strictly before its own group, plus every existing token of its own group
  (1..4 tokens, the reference's forced own-group inclusion). Width is exactly
  ``candidate_topk`` (2048); short requests select every live token, so the sparse
  layout equals dense attention there and no regime switch is needed on decode.
* SWA layers recall the last ``sliding_window`` (512) tokens instead.
* learned per-head sinks seed the sparse kernel's softmax denominator.
* indexed layers run 32 heads, SWA layers 64, over the same 576-wide latent rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx

from .dsa import DSAMetadata
from .dsa_indexer_kpool import Glm5NextDSABackend

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models import ModelConfig


class Hy4DSABackend(Glm5NextDSABackend):
    def __init__(self, config: "ModelConfig") -> None:
        self._current_sinks: torch.Tensor | None = None
        super().__init__(config)

    # ----- model-family hooks -----------------------------------------------------------
    def _model_args(self, config: "ModelConfig"):
        args = config.hy4_args
        assert args is not None, "hy4 dsa backend needs ModelConfig.hy4_args"
        self.kpool = args.index_compress_ratio
        self.select_k = args.index_group_topk - 1
        self._hy4 = args
        return args

    def _build_index_slots(self, args, config: "ModelConfig") -> None:
        # every indexed layer owns its indexer and is its own leader (no IndexShare)
        for lid in range(config.num_layers):
            if not args.swa_layers[lid]:
                self._idx_slot[lid] = len(self._idx_slot)
                self._leader[lid] = lid

    def _heads(self, layer_id: int) -> int:
        # TP shards the query heads; the latent rows and selection stay replicated
        from freetoken.distributed import get_tp_info

        return self._hy4.heads_of(layer_id) // get_tp_info().size

    # ----- store: kpool plan + norm + group-start rope ------------------------------------
    def _store_index(self, inputs, batch: "Batch", layer_id: int) -> None:
        from freetoken.kernel.triton.kpool_compress import kpool_compress_store, kpool_rope_group_rows
        from freetoken.kernel.triton.qsa import qsa_store_rows

        md = batch.attn_metadata
        assert isinstance(md, DSAMetadata)
        assert inputs.gate is not None and inputs.ape is not None
        k, gate = inputs.k, inputs.gate.to(inputs.k.dtype)
        slot = self._idx_slot[layer_id]
        tail_k, tail_g = self.kvcache.tail_k(slot), self.kvcache.tail_gate(slot)
        plan = self._plan_kpool_writes(md, batch, slot)
        kpool_compress_store(
            k, gate,
            tail_k.view(-1, k.shape[-1]), tail_g.view(-1, k.shape[-1]),
            inputs.ape,
            plan.ring_slots, plan.token_to_req, plan.cu_seqlens, batch.positions,
            self.kvcache.index_k_cache(slot), plan.cmp_rows,
            self.kpool,
            norm_weight=inputs.norm_weight, norm_eps=self._hy4.norm_eps,
        )
        qsa_store_rows(tail_k, plan.ring_rows, k)
        qsa_store_rows(tail_g, plan.ring_rows, gate)
        if inputs.rope_cache is not None:
            kpool_rope_group_rows(
                self.kvcache.index_k_cache(slot), plan.cmp_rows, batch.positions,
                inputs.rope_cache, self.kpool, self._hy4.qk_rope_head_dim,
            )

    # ----- attend (sinks) ------------------------------------------------------------------
    def _attend(self, q_cat, layer_id, sel, cnt, out=None, sinks=None):
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

        return glm_dsa_sparse_attn(
            q_cat, self.kvcache.latent_rows(layer_id), sel, self.sm_scale,
            counts=cnt, d_v=self.kv_lora_rank, out=out,
            kv_format=self.kvcache.kv_format, scale=self.kvcache.latent_scale(layer_id),
            sinks=sinks,
        )

    def mla_forward(self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs=None, sinks=None):
        self._current_sinks = sinks
        md = batch.attn_metadata
        # multi-row verify extend (DFly): graph-capturable device-side path
        if getattr(md, "verify_rows_per_req", 1) > 1 and not md.is_decode:
            self.kvcache.store_kv(c_kv, k_rope, batch.out_loc, layer_id)
            if self.dsa_enabled and indexer_inputs is not None:
                self._store_index(indexer_inputs, batch, layer_id)
            return self._verify(md, layer_id, q_nope, q_pe, indexer_inputs)
        return super().mla_forward(q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs=indexer_inputs)

    # ----- multi-row verify (CUDA-graph capturable, DFly) -------------------------------------
    def prepare_metadata(self, batch) -> None:
        super().prepare_metadata(batch)
        if getattr(batch, "phase", None) == "verify":
            reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
            batch.attn_metadata.verify_rows_per_req = reqs[0].extend_len if reqs else 1

    def init_capture_graph(self, max_seq_len: int, bs_list, rows_per_req: int = 1) -> None:
        if rows_per_req == 1:
            super().init_capture_graph(max_seq_len, bs_list, rows_per_req)
            self._verify_rows_per_req = 1
            return
        # multi-row verify: static buffers shaped [bs, m, ...] (qsa_sparse pattern)
        self.max_seq_len = max_seq_len
        self.capture_bs = sorted(bs_list)
        self._verify_rows_per_req = rows_per_req
        max_bs = max(bs_list)
        width = get_global_ctx().page_table.shape[1]
        m = rows_per_req
        self._v_rows_buf = torch.full((max_bs, m, width), -1, dtype=torch.int32, device=self.device)
        self._v_kvlen_buf = torch.zeros(max_bs, m, dtype=torch.int32, device=self.device)
        self._v_pos_buf = torch.zeros(max_bs, m, dtype=torch.int32, device=self.device)
        self._v_max_pools = max_seq_len // self.kpool

    def _stage_verify(self, batch, bs: int) -> None:
        try:
            self._stage_verify_inner(batch, bs)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _stage_verify_inner(self, batch, bs: int) -> None:
        md = batch.attn_metadata
        m = self._verify_rows_per_req
        reqs = batch.padded_reqs
        page_table = get_global_ctx().page_table
        rows = torch.stack([page_table[r.table_idx] for r in reqs])[:bs]      # [bs, W]
        kvlen = md.kv_len_cpu.to(self.device, non_blocking=True)[:bs]         # [bs]
        pos = getattr(batch, "positions", None)
        if pos is None:  # dummy capture batch: replays restage live positions
            pos = torch.zeros(bs * m, dtype=torch.int32, device=self.device)
        pos = pos.view(bs, m)
        self._v_rows_buf[:bs].copy_(rows.unsqueeze(1).expand(bs, m, -1))
        self._v_kvlen_buf[:bs].copy_(kvlen.unsqueeze(1).expand(bs, m))
        self._v_pos_buf[:bs].copy_(pos)
        md.rows = self._v_rows_buf[:bs]
        md.kvlen = self._v_kvlen_buf[:bs]
        md.positions = self._v_pos_buf[:bs]

    def prepare_for_capture(self, batch) -> None:
        if getattr(self, "_verify_rows_per_req", 1) > 1:
            self.prepare_metadata(batch)
            self._stage_verify(batch, batch.size)
        else:
            super().prepare_for_capture(batch)

    def prepare_for_replay(self, batch) -> None:
        if getattr(self, "_verify_rows_per_req", 1) > 1:
            self._stage_verify(batch, batch.padded_size)
        else:
            super().prepare_for_replay(batch)

    def reset_capture(self) -> None:
        super().reset_capture()
        self._v_rows_buf = self._v_kvlen_buf = self._v_pos_buf = None
        self._verify_rows_per_req = 1

    def _verify(self, md, layer_id, q_nope, q_pe, inputs):
        try:
            return self._verify_inner(md, layer_id, q_nope, q_pe, inputs)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _verify_inner(self, md, layer_id, q_nope, q_pe, inputs):
        """Device-side batched multi-row verify attention (m = k+1 rows/request)."""
        rows, kvlen, positions = md.rows, md.kvlen, md.positions   # [bs, m, W], [bs, m], [bs, m]
        bs, m = positions.shape
        heads = self._heads(layer_id)
        if self._hy4.swa_layers[layer_id]:
            sel, cnt = self._window_select(rows, positions, kvlen)
        elif not self.dsa_enabled or inputs is None:
            sel = rows
            cnt = (positions + 1).clamp_max(rows.shape[-1]).to(torch.int32)
        else:
            q_idx = inputs.q.view(bs, m, self._hy4.index_n_heads, self._hy4.index_head_dim)
            w = inputs.w.view(bs, m, self._hy4.index_n_heads)
            # per-request page-table rows [bs, W] (identical across the m query rows)
            sel, cnt, pos = self._select_verify_indexed(
                self._idx_slot[layer_id], q_idx, w, rows[:, 0, :], positions, kvlen[:, 0], self._v_max_pools
            )
            md.sel[layer_id] = (sel, cnt)
            md.sel_pos[layer_id] = pos      # token positions for the DFly draft reuse snapshot
        if q_pe.shape[-1] == 0:
            q_cat = q_nope.view(bs, m, heads, self.kv_lora_rank)
        else:
            q_cat = torch.cat([q_nope, q_pe], dim=-1).view(bs, m, heads, self.latent_dim)
        o = self._attend(q_cat, layer_id, sel, cnt, sinks=self._current_sinks)
        return o.view(bs * m, heads, self.kv_lora_rank)

    # ----- DFly draft: attend over the paired TARGET layer's KV ------------------------------
    def dfly_mla_forward(self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, sinks=None):
        """Draft layer ``layer_id`` writes its query-block latent rows into the paired
        target layer's pool and attends over that pool with the reused selection
        (``md.dfly_sel[layer_id]``, built by the DFly proposer from the target
        indexer snapshot + the draft query block). No private draft KV chain."""
        from freetoken.models.hy_v4_flash.dfly import DFLY_TARGET_LAYERS

        md = batch.attn_metadata
        assert isinstance(md, DSAMetadata)
        target_layer = DFLY_TARGET_LAYERS[layer_id]
        self._current_sinks = sinks
        self.kvcache.store_kv(c_kv, k_rope, batch.out_loc, target_layer)
        sel, cnt = md.dfly_sel[layer_id]
        # the paired target layers are all indexed (sparse): the draft's indexed head
        # count matches them; _heads(target_layer) would index the draft's 5-layer swa table
        from freetoken.distributed import get_tp_info

        heads = self._hy4.n_heads // get_tp_info().size
        bs, m = sel.shape[0], sel.shape[1]
        if q_pe.shape[-1] == 0:
            q_cat = q_nope.view(bs, m, heads, self.kv_lora_rank)
        else:
            q_cat = torch.cat([q_nope, q_pe], dim=-1).view(bs, m, heads, self.latent_dim)
        o = self._attend(q_cat, target_layer, sel, cnt, sinks=sinks)
        return o.view(bs * m, heads, self.kv_lora_rank)

    # ----- selection helpers ---------------------------------------------------------------
    def _expand_groups(self, picks, rows, q_pos, kvlen):
        """Picked complete groups + the query's own group tokens (up to kpool),
        sentinels compacted to the tail. ``picks [B, m, select_k]`` (-1 sentinel),
        ``q_pos [B, m]`` query positions, ``kvlen [B, m]`` live lengths ->
        (sel [B, m, kpool + select_k*kpool] physical rows, cnt [B, m],
        pos [B, m, width] token positions -- the DFly snapshot reuses positions)."""
        kp = self.kpool
        b, m, k_sel = picks.shape
        device = picks.device
        own_start = (q_pos // kp) * kp
        toffs = torch.arange(kp, device=device)
        own_pos = own_start.unsqueeze(-1) + toffs
        own_pos = torch.where(own_pos < kvlen.unsqueeze(-1), own_pos, own_pos.new_full((), -1))
        offs = torch.arange(kp, device=device)
        hist_pos = picks.unsqueeze(-1) * kp + offs
        hist_pos = torch.where(picks.unsqueeze(-1) < 0, hist_pos.new_full((), -1), hist_pos).view(b, m, k_sel * kp)
        pos = torch.cat([hist_pos, own_pos], dim=-1)
        # The kernel reads the FIRST cnt entries, so every valid row must precede
        # every -1 sentinel: a query with fewer live groups than k_sel has invalid
        # picks (-1) interleaved before the own-group tokens. Stable-sort the
        # sentinels to the tail (graph-safe: static shapes, device-only).
        valid = pos >= 0
        order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
        pos = pos.gather(-1, order)
        rows_b = rows if rows.dim() == 3 else rows.unsqueeze(1).expand(b, m, -1)
        sel = rows_b.gather(-1, pos.clamp_min(0).long()).to(torch.int32)
        sel = torch.where(pos < 0, sel.new_full((), -1), sel)
        cnt = valid.sum(dim=-1).to(torch.int32)
        return sel, cnt, pos.to(torch.int32)

    def _window_select(self, rows, q_pos, kvlen):
        """Last ``sliding_window`` positions per query; rows [B, W] or [B, m, W]."""
        w = self._hy4.sliding_window
        b, m = q_pos.shape
        start = (q_pos - (w - 1)).clamp_min(0)
        pos = start.unsqueeze(-1) + torch.arange(w, device=q_pos.device)
        pos = torch.where(pos <= q_pos.unsqueeze(-1), pos, pos.new_full((), -1))
        rows_b = rows if rows.dim() == 3 else rows.unsqueeze(1).expand(b, m, -1)
        sel = rows_b.gather(-1, pos.clamp_min(0).long()).to(torch.int32)
        sel = torch.where(pos < 0, sel.new_full((), -1), sel)
        cnt = kvlen.clamp_max(w).to(torch.int32)
        return sel, cnt

    # ----- decode (CUDA-graph capturable) ----------------------------------------------------
    def _decode(self, md, layer_id, q_nope, q_pe, inputs):
        bs = q_nope.shape[0]
        heads = self._heads(layer_id)
        rows, kvlen = md.rows, md.kvlen
        if self._hy4.swa_layers[layer_id]:
            sel, cnt = self._window_select(rows, (kvlen - 1).view(bs, 1), kvlen.view(bs, 1))
            sel, cnt = sel.view(bs, 1, -1), cnt.view(bs, 1)
        elif not self.dsa_enabled or inputs is None:
            sel, cnt = rows.view(bs, 1, -1), kvlen.view(bs, 1)
        else:
            kp = self.kpool
            g_q = (kvlen - 1) // kp  # the query's own group; candidates are strictly before it
            pool_rows = (rows[:, kp - 1 :: kp] // kp).contiguous()
            s = self.dsa_decode_scores(inputs.q, inputs.w, self._idx_slot[layer_id], pool_rows, g_q)
            k_sel = min(self.select_k, s.shape[-1])
            picks = self.indexer_select_decode(
                s.view(bs, 1, -1), valid=g_q, topk=k_sel, offset=0
            )[:, 0]  # [bs, k_sel]
            sel, cnt, _ = self._expand_groups(
                picks.unsqueeze(1), rows.unsqueeze(1), (kvlen - 1).view(bs, 1), kvlen.view(bs, 1)
            )
            sel, cnt = sel.view(bs, 1, -1), cnt.view(bs, 1)
            md.sel.clear()
            md.sel[layer_id] = (sel, cnt)
        if q_pe.shape[-1] == 0:
            q_cat = q_nope.unsqueeze(1)
        else:
            q_cat = torch.cat([q_nope, q_pe], dim=-1).view(bs, 1, heads, self.latent_dim)
        o = self._attend(q_cat, layer_id, sel, cnt, sinks=self._current_sinks)
        return o.view(bs, heads, self.kv_lora_rank)

    # ----- prefill / extend (eager) -----------------------------------------------------------
    def _select_prefill_indexed(self, slot, q_idx, w, rows, positions, kv_len):
        """Per-request causal group top-k expanded to token rows (host start_pos-free:
        positions are the device query positions)."""
        kp = self.kpool
        n_pools = kv_len // kp
        m = q_idx.shape[0]
        k_sel = min(self.select_k, max(n_pools, 1))
        width = kp + k_sel * kp
        if n_pools == 0:
            picks = torch.full((1, m, k_sel), -1, dtype=torch.int32, device=self.device)
            sel, cnt, _ = self._expand_groups(picks, rows.view(1, -1), positions.view(1, -1), (positions + 1).view(1, -1))
            return sel, cnt
        pool_rows = rows[kp - 1 :: kp] // kp
        k_pool = self.kvcache.index_k_cache(slot).index_select(0, pool_rows.long())
        sel = torch.empty(m, width, dtype=torch.int32, device=self.device)
        cnt = torch.empty(m, dtype=torch.int32, device=self.device)
        chunk = 512
        for s0 in range(0, m, chunk):
            s1 = min(s0 + chunk, m)
            scores = self.dsa_prefill_logits(q_idx[s0:s1], k_pool, w[s0:s1])
            p = positions[s0:s1]
            live = p // kp  # candidate groups: [0, live)
            cols = torch.arange(scores.shape[-1], device=self.device)
            scores = scores + torch.where(cols >= live.unsqueeze(1), float("-inf"), 0.0)
            picks = scores.topk(min(k_sel, n_pools), dim=-1)[1]
            picks = torch.where(picks >= live.unsqueeze(1), -1, picks)
            sel_c, cnt_c, _ = self._expand_groups(
                picks.unsqueeze(0), rows.view(1, -1), p.view(1, -1), (p + 1).view(1, -1)
            )
            sel[s0:s1] = sel_c[0]
            cnt[s0:s1] = cnt_c[0]
        return sel.view(1, m, width), cnt.view(1, m)

    def _select_verify_indexed(self, slot, q_idx, w, rows, positions, kv_len, max_pools):
        """Device-side, static-shape causal pool top-k for a verify/extend block
        (``m`` query rows per request). Batched generalization of
        ``_select_prefill_indexed`` with no per-request host loop, so it captures
        into a multi-row verify graph (``max_pools`` is the capture-time constant
        ``max_seq_len // kpool``). ``q_idx [bs, m, H, D]``, ``w [bs, m, H]``,
        ``rows [bs, W]`` position-ordered physical rows, ``positions [bs, m]``,
        ``kv_len [bs]``. Returns ``(sel [bs, m, width], cnt [bs, m])``."""
        from freetoken.attention.dfly_reuse import verify_select_pools
        from freetoken.kernel.triton.dsv4.indexer import indexer_logits

        kp = self.kpool
        bs, m = positions.shape
        device = rows.device
        pool_rows_full = rows[:, kp - 1 :: kp] // kp           # [bs, <=max_pools]
        pr = torch.zeros(bs, max_pools, dtype=torch.int64, device=device)
        n = min(pool_rows_full.shape[1], max_pools)
        pr[:, :n] = pool_rows_full[:, :n].to(torch.int64)
        k_pool = self.kvcache.index_k_cache(slot).index_select(0, pr.view(-1)).view(bs, max_pools, -1)
        scores = indexer_logits(q_idx, k_pool, (w * self.index_scale).to(torch.float32))  # [bs, m, max_pools]
        n_pools = (kv_len // kp).to(torch.int64)
        pool_idx = torch.arange(max_pools, device=device)
        scores = scores + torch.where(pool_idx[None, None, :] >= n_pools[:, None, None], float("-inf"), 0.0)
        picks = verify_select_pools(scores, positions, kp, self.select_k)   # [bs, m, select_k]
        return self._expand_groups(picks, rows, positions, kv_len[:, None].expand(bs, m))

    def _prefill(self, md, layer_id, q_nope, q_pe, batch, inputs):
        t = q_nope.shape[0]
        heads = self._heads(layer_id)
        if q_pe.shape[-1] == 0:
            q_cat = q_nope
        else:
            q_cat = torch.cat([q_nope, q_pe], dim=-1)
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = self._page_table()
        qo = md.qo_indptr_cpu.tolist()
        kv_lens = md.kv_len_cpu.tolist()
        is_swa = self._hy4.swa_layers[layer_id]
        sparse = self.dsa_enabled and not is_swa and max(kv_lens) > self.index_topk
        o = q_cat.new_empty(t, heads, self.kv_lora_rank)
        for i, r in enumerate(reqs):
            m = qo[i + 1] - qo[i]
            if m == 0:
                continue
            rows = page_table[r.table_idx, : r.device_len]
            positions = batch.positions[qo[i] : qo[i + 1]]
            if is_swa:
                sel, cnt = self._window_select(
                    rows.view(1, -1), positions.view(1, -1), (positions + 1).view(1, -1)
                )
            elif sparse and inputs is not None:
                sel, cnt = self._select_prefill_indexed(
                    self._idx_slot[layer_id],
                    inputs.q[qo[i] : qo[i + 1]], inputs.w[qo[i] : qo[i + 1]],
                    rows, positions, kv_lens[i],
                )
            else:
                sel = rows.view(1, 1, -1).to(torch.int32)
                cnt = (positions + 1).to(torch.int32).view(1, m)
            self._attend(
                q_cat[qo[i] : qo[i + 1]].view(1, m, heads, self.latent_dim),
                layer_id, sel, cnt,
                out=o[qo[i] : qo[i + 1]].view(1, m, heads, self.kv_lora_rank),
                sinks=self._current_sinks,
            )
        return o

    @staticmethod
    def _page_table():
        from freetoken.core import get_global_ctx

        return get_global_ctx().page_table


__all__ = ["Hy4DSABackend"]
