"""DFly draft sparse-index reuse (port of vLLM PR #68296 ``dfly_inputs.py``).

A DFly draft layer attends over its paired target layer's selected history plus
the draft query block, instead of running its own indexer. Per draft query row
the index layout is::

    [ target top-k history (rows < anchor - local_window) |
      local window [anchor - local_window, anchor)        |
      causal draft block [anchor, anchor + query_offset]  | -1 pad ]

width ``round_up(target_topk + local_window + query_width, 128)``. The target
history comes from a snapshot of the target indexer's selected rows (one snapshot
slot per draft layer); ``-1`` sentinels are masked by the sparse-MLA kernel.

This is the vectorized, CUDA-graph-safe torch form (static shapes, device-only
arithmetic, one small loop over the 5 draft layers); the vLLM original is a fused
triton launch. Correctness is pinned by ``tests/models/hy_v4_flash/test_dfly.py``.
"""

from __future__ import annotations

import torch


def dfly_out_width(target_topk: int, local_window: int, query_width: int, align: int = 128) -> int:
    return ((target_topk + local_window + query_width + align - 1) // align) * align


def build_dfly_reuse_indices(
    out: torch.Tensor,            # [num_layers, padded_batch * query_width, out_width] int32
    snapshots: torch.Tensor,      # [num_index_layers, max_rows, target_topk] int32 target selected rows
    layer_snapshot_slots: torch.Tensor,  # [num_layers] int -> snapshot slot per draft layer
    active: torch.Tensor,         # [padded_batch] bool
    source_rows: torch.Tensor,    # [padded_batch] int -> target query row whose selection to reuse
    anchors: torch.Tensor,        # [padded_batch] int -> first draft position (target last pos + 1)
    target_topk: int,
    local_window: int,
    query_width: int,
) -> None:
    num_layers, rows, out_width = out.shape
    device = out.device
    req = torch.arange(rows, device=device) // query_width
    qoff = torch.arange(rows, device=device) % query_width
    act = active[req].bool()
    anchor = anchors[req].to(torch.int64)
    src = source_rows[req].to(torch.int64)
    window_start = anchor - local_window
    cols = torch.arange(out_width, device=device)
    is_history = cols < target_topk
    window_col = cols - target_topk
    is_window = (window_col >= 0) & (window_col < local_window)
    block_col = window_col - local_window
    hist_cols = cols.clamp(max=target_topk - 1).expand(rows, -1)
    for layer in range(num_layers):
        slot = layer_snapshot_slots[layer].to(torch.int64)
        hist = snapshots[slot].index_select(0, src)[:, :target_topk]      # [rows, target_topk]
        hist = torch.where((hist >= 0) & (hist < window_start[:, None]), hist, hist.new_full((), -1))
        hist_vals = torch.where(is_history[None, :], hist.gather(1, hist_cols), hist.new_full((), -1))
        window_vals = (window_start[:, None] + window_col[None, :]).clamp_min(-1)
        is_causal = (block_col[None, :] >= 0) & (block_col[None, :] <= qoff[:, None])
        causal_vals = anchor[:, None] + block_col[None, :]
        vals = torch.where(
            is_history[None, :], hist_vals,
            torch.where(is_window[None, :], window_vals, torch.where(is_causal, causal_vals, -1)),
        )
        vals = torch.where(act[:, None], vals, vals.new_full((), -1))
        out[layer] = vals.to(torch.int32)


def verify_select_pools(
    scores: torch.Tensor,    # [bs, m, n_blocks] fp32 indexer pool scores
    positions: torch.Tensor,  # [bs, m] int absolute query positions
    ratio: int,
    topk: int,
) -> torch.Tensor:
    """Device-side, static-shape causal pool top-k for a verify/extend block.

    Batched generalization of ``indexer_select_prefill`` (dsv4_indexer): query row
    at position ``p`` attends pools ``[0, (p + 1) // ratio)``; pools past that score
    ``-inf`` and any pick still past the live count returns ``-1``. All arithmetic
    is device-only with fixed shapes, so it captures into a multi-row verify graph
    (the hy4 dsa backend's missing ``uniform_extend_graph`` capability). Returns
    ``[bs, m, topk]`` int pool ids (-1 sentinel)."""
    n_blocks = scores.shape[-1]
    live = ((positions + 1) // ratio).unsqueeze(-1)            # [bs, m, 1]
    blk = torch.arange(n_blocks, device=scores.device)         # [n_blocks]
    masked = scores + torch.where(blk >= live, float("-inf"), 0.0)
    picks = masked.topk(min(topk, n_blocks), dim=-1)[1]        # [bs, m, topk]
    return torch.where(picks >= live, -1, picks).to(torch.int32)


__all__ = ["build_dfly_reuse_indices", "dfly_out_width", "verify_select_pools"]
