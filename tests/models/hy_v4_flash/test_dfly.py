"""DFly2 draft components: the sparse-index reuse builder (Phase 2).

Pins ``build_dfly_reuse_indices`` against a naive scalar transcription of the
vLLM ``_build_dfly_reuse_indices_kernel`` so the vectorized graph-safe form cannot
drift from the reference layout (history | local window | causal draft block).
"""

from __future__ import annotations

import torch

from freetoken.attention.dfly_reuse import build_dfly_reuse_indices, dfly_out_width, verify_select_pools


def _naive(snapshots, layer_snapshot_slots, active, source_rows, anchors,
           target_topk, local_window, query_width, padded_batch, out_width):
    num_layers = snapshots.shape[0]
    out = torch.full((num_layers, padded_batch * query_width, out_width), -1, dtype=torch.int32)
    for layer in range(num_layers):
        slot = int(layer_snapshot_slots[layer])
        for req in range(padded_batch):
            if not bool(active[req]):
                continue
            a = int(anchors[req])
            src = int(source_rows[req])
            ws = a - local_window
            hist = snapshots[slot, src, :target_topk].clone()
            hist = torch.where((hist >= 0) & (hist < ws), hist, torch.full_like(hist, -1))
            for qoff in range(query_width):
                row = req * query_width + qoff
                out[layer, row, :target_topk] = hist
                for wc in range(local_window):
                    out[layer, row, target_topk + wc] = max(ws + wc, -1)
                for bc in range(qoff + 1):
                    out[layer, row, target_topk + local_window + bc] = a + bc
    return out


def test_dfly_out_width():
    assert dfly_out_width(2048, 0, 6) == 2176
    assert dfly_out_width(8, 2, 3) == 128
    assert dfly_out_width(128, 0, 1) == 256


def test_build_dfly_reuse_indices_matches_naive():
    torch.manual_seed(0)
    target_topk, local_window, query_width = 8, 3, 4
    num_layers, padded_batch, max_rows = 3, 5, 6
    out_width = dfly_out_width(target_topk, local_window, query_width)
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        snapshots = torch.randint(-2, 40, (num_layers, max_rows, target_topk), generator=g, dtype=torch.int32)
        layer_snapshot_slots = torch.arange(num_layers, dtype=torch.int32)
        active = torch.randint(0, 2, (padded_batch,), generator=g).bool()
        source_rows = torch.randint(0, max_rows, (padded_batch,), generator=g)
        anchors = torch.randint(0, 30, (padded_batch,), generator=g)
        out = torch.empty((num_layers, padded_batch * query_width, out_width), dtype=torch.int32)
        build_dfly_reuse_indices(
            out, snapshots, layer_snapshot_slots, active, source_rows, anchors,
            target_topk, local_window, query_width,
        )
        want = _naive(snapshots, layer_snapshot_slots, active, source_rows, anchors,
                      target_topk, local_window, query_width, padded_batch, out_width)
        assert torch.equal(out, want), f"seed {seed} mismatch"


def test_build_dfly_reuse_indices_zero_window():
    # hy4-flash: context_local_window is None -> 0; layout is history | causal block
    target_topk, local_window, query_width = 16, 0, 6
    num_layers, padded_batch, max_rows = 5, 2, 4
    out_width = dfly_out_width(target_topk, local_window, query_width)
    g = torch.Generator().manual_seed(3)
    snapshots = torch.randint(-1, 50, (num_layers, max_rows, target_topk), generator=g, dtype=torch.int32)
    layer_snapshot_slots = torch.arange(num_layers, dtype=torch.int32)
    active = torch.ones(padded_batch, dtype=torch.bool)
    source_rows = torch.tensor([0, 3])
    anchors = torch.tensor([20, 7])
    out = torch.empty((num_layers, padded_batch * query_width, out_width), dtype=torch.int32)
    build_dfly_reuse_indices(out, snapshots, layer_snapshot_slots, active, source_rows, anchors,
                             target_topk, local_window, query_width)
    want = _naive(snapshots, layer_snapshot_slots, active, source_rows, anchors,
                  target_topk, local_window, query_width, padded_batch, out_width)
    assert torch.equal(out, want)
    # the causal block of the last query row carries anchor..anchor+query_width-1
    last = out[0, query_width - 1]
    assert last[target_topk + query_width - 1].item() == 20 + query_width - 1


def test_verify_select_pools_matches_scalar_prefill():
    # batched device-side causal pool top-k must equal the per-row scalar rule
    # (indexer_select_prefill): live = (p+1)//ratio, pools >= live masked / -1.
    torch.manual_seed(0)
    ratio, topk = 4, 5
    bs, m, n_blocks = 3, 6, 20
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        scores = torch.randn(bs, m, n_blocks, generator=g)
        positions = torch.randint(0, 60, (bs, m), generator=g)
        got = verify_select_pools(scores, positions, ratio, topk)
        want = torch.full((bs, m, min(topk, n_blocks)), -1, dtype=torch.int32)
        blk = torch.arange(n_blocks)
        for b in range(bs):
            for r in range(m):
                live = (int(positions[b, r]) + 1) // ratio
                s = scores[b, r].clone()
                s = s + torch.where(blk >= live, float("-inf"), 0.0)
                picks = s.topk(min(topk, n_blocks))[1]
                want[b, r] = torch.where(picks >= live, -1, picks).to(torch.int32)
        assert torch.equal(got, want), f"seed {seed}"
