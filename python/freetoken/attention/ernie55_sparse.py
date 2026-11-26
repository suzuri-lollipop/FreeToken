"""ERNIE 5.5 sparse-attention backend: DSV4 HCA addressing + token-level DSA.

The 37 HCA layers drive the inherited DSV4 vocabulary verbatim (window ring,
compressed blocks, compressor carry, ``attend`` over ``[window | compressed]``
global slots with the per-head sink). The six DSA layers call :meth:`mla_forward`
(GLM-DSA contract: kv_b-absorbed q over latent rows) with two ERNIE deltas:

* the indexer scores only tokens OLDER than the trailing 128-window; the window
  is always appended to the selection (window-first, so one device-side count
  ``wlen + picked_valid`` bounds the kernel exactly);
* the sparse MLA softmax carries the per-head attention sink
  (``softmax_offset`` in the checkpoint).

One merged metadata object serves both layer families; the decode snapshot /
static-buffer staging follows the DSV4 overlap-safety contract (the captured
graph reads staged rows, never the live page table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx

from .dsv4_sparse import DSV4AttnMetadata, DSV4SparseAttnBackend

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
# Prefill scoring transient budget (the fp32 logits tile is [chunk, kv_len]).
_DSA_SCORE_BYTES = 128 << 20
_DSA_SCORE_CHUNK = 512


@dataclass
class Ernie55AttnMetadata(DSV4AttnMetadata):
    """DSV4 addressing + the DSA layer family's per-forward fields."""

    is_decode: bool = False
    qo_indptr_cpu: torch.Tensor | None = None
    kv_len_cpu: torch.Tensor | None = None
    # DSA decode addressing: per-request row snapshot + live lengths (device-read).
    dsa_rows: torch.Tensor | None = None
    dsa_kvlen: torch.Tensor | None = None
    # layer -> (sel, cnt) for the current forward (prefill: per-request list).
    dsa_sel: dict = field(default_factory=dict)


class Ernie55SparseAttnBackend(DSV4SparseAttnBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        args = config.dsv4_args
        self.dsa_layer_ids = frozenset(args.dsa_layer_ids)
        self.index_topk = args.index_topk
        self.dsa_window = args.window_size
        self.index_scale = args.index_head_dim**-0.5
        self.dsa_sm_scale = args.mla_qk_head_dim**-0.5
        self.kv_lora_rank = args.mla_kv_lora_rank
        self.dsa_latent_dim = args.mla_latent_dim
        self.num_qo_heads = args.mla_n_heads
        self._dsa_rows_buf: torch.Tensor | None = None
        self._dsa_kvlen_buf: torch.Tensor | None = None

    # ----- metadata -------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        super().prepare_metadata(batch)
        md = batch.attn_metadata
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        batch.attn_metadata = Ernie55AttnMetadata(
            last_indices=md.last_indices,
            segments=md.segments,
            full_snap=md.full_snap,
            table_rows=md.table_rows,
            window_ar=md.window_ar,
            is_decode=getattr(batch, "phase", None) == "decode",
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
        )

    # ----- DSA forward (the GLM mla_forward contract + window exclusion + sink) --------
    def mla_forward(
        self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs=None, attn_sink=None
    ) -> torch.Tensor:
        """Store this forward's latent + index rows and attend over the paged history.

        ``q_nope`` [T, H, kv_lora_rank] (kv_b-absorbed), ``q_pe`` [T, H, rope_dim],
        ``c_kv`` [T, kv_lora_rank] / ``k_rope`` [T, rope_dim]. Returns [T, H, kv_lora_rank].
        """
        md = batch.attn_metadata
        assert isinstance(md, Ernie55AttnMetadata)
        pool = self.pool
        pool.store_dsa_kv(c_kv, k_rope, batch.out_loc, layer_id)
        if indexer_inputs is not None:
            pool.store_dsa_index_k(indexer_inputs.k, batch.out_loc, layer_id)
        if md.is_decode:
            return self._dsa_decode(md, layer_id, q_nope, q_pe, indexer_inputs, attn_sink)
        return self._dsa_prefill(md, layer_id, q_nope, q_pe, batch, indexer_inputs, attn_sink)

    def _dsa_decode_rows(self, md: Ernie55AttnMetadata, batch: Batch) -> None:
        assert md.table_rows is not None
        md.dsa_rows = get_global_ctx().page_table.index_select(0, md.table_rows).to(torch.int32)
        md.dsa_kvlen = md.kv_len_cpu.to(self.device, non_blocking=True)

    def _dsa_window_rows(self, rows: torch.Tensor, kvlen: torch.Tensor) -> torch.Tensor:
        """The trailing-window rows per decode row, ``[bs, window]`` (-1 padded).

        Window slot j holds position ``kvlen - wlen + j`` with ``wlen = min(kvlen,
        window)``: sequences shorter than the window start at position 0 (the old
        ``kvlen - window`` offset went negative and clamped every slot to row 0).
        """
        bs = rows.shape[0]
        j = torch.arange(self.dsa_window, device=self.device)
        wlen = kvlen.clamp(max=self.dsa_window)
        wpos = kvlen[:, None] - wlen[:, None] + j[None, :]
        valid = j[None, :] < wlen[:, None]
        wr = rows.gather(1, wpos.clamp(min=0, max=rows.shape[-1] - 1).long()).to(torch.int32)
        return torch.where(valid, wr, torch.full_like(wr, -1))

    def _dsa_decode(self, md, layer_id, q_nope, q_pe, inputs, sink) -> torch.Tensor:
        bs = q_nope.shape[0]
        if md.dsa_rows is None:
            self._dsa_decode_rows(md, get_global_ctx().batch)
        rows, kvlen = md.dsa_rows, md.dsa_kvlen
        window = self.dsa_window
        K = min(self.index_topk, rows.shape[1])
        wlen = kvlen.clamp(max=window)
        win_rows = self._dsa_window_rows(rows, kvlen)
        scored_len = (kvlen - window).clamp_min(0)
        if inputs is not None:
            q_idx, w = inputs.q, inputs.w
            s = self.dsa_decode_scores(q_idx, w, layer_id, rows, scored_len)
            picks = self.indexer_select_decode(
                s.view(bs, 1, -1), valid=scored_len, topk=K, offset=0
            )[:, 0]
            sel_picked = self.dsa_map_rows(picks, rows)
            pv = scored_len.clamp(max=K)
        else:
            sel_picked = torch.full((bs, K), -1, dtype=torch.int32, device=self.device)
            pv = torch.zeros_like(kvlen)
        # Window FIRST: its wlen valid rows are a prefix, the picked half's pv valid
        # rows follow at column `window`, so one count bounds the kernel exactly.
        sel = torch.cat([win_rows, sel_picked], dim=-1).view(bs, 1, -1)
        cnt = (wlen + pv).to(torch.int32).view(bs, 1)
        q_cat = torch.cat([q_nope, q_pe], dim=-1).view(bs, 1, self.num_qo_heads, self.dsa_latent_dim)
        o = self._dsa_attend(q_cat, layer_id, sel, cnt, sink)
        return o.view(bs, self.num_qo_heads, self.kv_lora_rank)

    def _dsa_attend(self, q_cat, layer_id, sel, cnt, sink, out=None) -> torch.Tensor:
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

        return glm_dsa_sparse_attn(
            q_cat, self.pool.dsa_latent_rows(layer_id), sel, self.dsa_sm_scale,
            counts=cnt, d_v=self.kv_lora_rank, sink=sink, out=out,
            kv_format=self.pool.kv_format, scale=self.pool.dsa_latent_scale(layer_id),
        )

    def dsa_decode_scores(self, q_idx, w, layer_id, rows, kvlen) -> torch.Tensor:
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_decode_logits

        return glm_dsa_decode_logits(
            q_idx, w * self.index_scale, self.pool.dsa_index_rows(layer_id), rows, kvlen
        )

    def dsa_prefill_logits(self, q_idx, k_all, w) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.indexer import indexer_logits

        return indexer_logits(
            q_idx.unsqueeze(0), k_all.unsqueeze(0),
            (w * self.index_scale).to(torch.float32).unsqueeze(0),
        )[0]

    @staticmethod
    def dsa_map_rows(picks: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        sel = rows.gather(-1, picks.clamp_min(0).long()).to(torch.int32)
        return torch.where(picks < 0, sel.new_full((), -1), sel)

    def _dsa_prefill(self, md, layer_id, q_nope, q_pe, batch, inputs, sink) -> torch.Tensor:
        t = q_nope.shape[0]
        q_cat = torch.cat([q_nope, q_pe], dim=-1)
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = get_global_ctx().page_table
        qo = md.qo_indptr_cpu.tolist()
        kv_lens = md.kv_len_cpu.tolist()
        window = self.dsa_window
        K = self.index_topk
        # topk covers every scored token (kv - window <= topk) -> selection == all
        # tokens: the identity path, exact and scoring-free.
        dense = max(kv_lens) <= K + window
        o = q_cat.new_empty(t, self.num_qo_heads, self.kv_lora_rank)
        for i, r in enumerate(reqs):
            m = qo[i + 1] - qo[i]
            if m == 0:
                continue
            positions = batch.positions[qo[i] : qo[i + 1]]
            rows = page_table[r.table_idx, : r.device_len].to(torch.int32)
            if dense or inputs is None:
                sel = rows.view(1, 1, -1)
                cnt = (positions + 1).to(torch.int32).view(1, m)
            else:
                sel, cnt = self._dsa_select_prefill(
                    layer_id, inputs, rows, positions, kv_lens[i] - m, K, window
                )
            self._dsa_attend(
                q_cat[qo[i] : qo[i + 1]].view(1, m, self.num_qo_heads, self.dsa_latent_dim),
                layer_id, sel, cnt, sink,
                out=o[qo[i] : qo[i + 1]].view(1, m, self.num_qo_heads, self.kv_lora_rank),
            )
        return o

    def _dsa_select_prefill(self, layer_id, inputs, rows, positions, start_pos, K, window):
        """Per-query [window | top-k scored] selection for one request's chunk."""
        kv_len = rows.numel()
        k_all = self.pool.dsa_index_rows(layer_id).index_select(0, rows.long())
        q_idx, w = inputs.q, inputs.w
        m = q_idx.shape[0]
        k_sel = min(K, kv_len)
        picked = torch.empty(m, k_sel, dtype=torch.int32, device=self.device)
        chunk = max(16, min(_DSA_SCORE_CHUNK, _DSA_SCORE_BYTES // max(kv_len * 4, 1)))
        for s0 in range(0, m, chunk):
            s1 = min(s0 + chunk, m)
            scores = self.dsa_prefill_logits(q_idx[s0:s1], k_all, w[s0:s1])
            # Window exclusion: query p scores only [0, p+1-window). Passing
            # start_pos - window to the shared causal select makes its live count
            # exactly that bound (and its -1 conversion cover the masked columns).
            picks = self.indexer_select_prefill(
                scores.unsqueeze(0), start_pos=start_pos + s0 - window,
                seqlen=s1 - s0, ratio=1, topk=k_sel, offset=0,
            )[0]
            picked[s0:s1] = self.dsa_map_rows(picks, rows.view(1, -1).expand(s1 - s0, -1))
        # Per-query window rows: positions [p+1-wlen, p+1), -1 past the sequence start.
        j = torch.arange(window, device=self.device)
        p1 = positions + 1
        wlen = p1.clamp(max=window)
        wpos = p1[:, None] - wlen[:, None] + j[None, :]
        valid = j[None, :] < wlen[:, None]
        win_rows = rows.view(1, -1).expand(m, -1).gather(1, wpos.clamp(min=0, max=rows.shape[-1] - 1).long()).to(torch.int32)
        win_rows = torch.where(valid, win_rows, torch.full_like(win_rows, -1))
        sel = torch.cat([win_rows, picked], dim=-1).view(1, m, -1)
        pv = (p1 - window).clamp(min=0, max=k_sel)
        cnt = (wlen + pv).to(torch.int32).view(1, m)
        return sel, cnt

    # ----- CUDA graph (decode): DSV4 snapshot + DSA static buffers ---------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], rows_per_req: int = 1) -> None:
        super().init_capture_graph(max_seq_len, bs_list, rows_per_req)
        max_bs = max(bs_list) * rows_per_req
        width = get_global_ctx().page_table.shape[1]
        self._dsa_rows_buf = torch.full((max_bs, width), -1, dtype=torch.int32, device=self.device)
        self._dsa_kvlen_buf = torch.zeros(max_bs, dtype=torch.int32, device=self.device)

    def _point_to_capture(self, batch: Batch, bs: int, rows_ti: torch.Tensor) -> None:
        assert self.capture is not None and bs <= self.max_graph_bs
        cap = self.capture
        src = self.pool.full_loc_map.index_select(0, rows_ti)
        w = min(src.shape[1], cap.full_snap.shape[1])
        cap.full_snap[:bs, :w].copy_(src[:bs, :w])
        md = getattr(batch, "attn_metadata", None)
        kv_len_cpu = md.kv_len_cpu if isinstance(md, Ernie55AttnMetadata) else None
        if kv_len_cpu is None:
            reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
            kv_len_cpu = torch.tensor([r.device_len for r in reqs], **_CPU_PINNED)
            # a spec verify stages rows_per_req query rows per request
            kv_len_cpu = kv_len_cpu.repeat_interleave(batch.rows_per_req or 1)
        self._dsa_rows_buf[:bs].copy_(src.to(torch.int32))
        self._dsa_kvlen_buf[:bs].copy_(kv_len_cpu.to(self.device, non_blocking=True))
        batch.attn_metadata = Ernie55AttnMetadata(
            last_indices=cap.last_indices[:bs],
            full_snap=cap.full_snap[:bs],
            window_ar=self._window_ar,
            is_decode=True,
            kv_len_cpu=kv_len_cpu,
            dsa_rows=self._dsa_rows_buf[:bs],
            dsa_kvlen=self._dsa_kvlen_buf[:bs],
        )

    def reset_capture(self) -> None:
        super().reset_capture()
        self._dsa_rows_buf = None
        self._dsa_kvlen_buf = None


__all__ = ["Ernie55SparseAttnBackend", "Ernie55AttnMetadata"]
