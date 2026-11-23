"""rollback_tail after a widened decode step: the request keeps exactly the pages of its committed tokens, the rest go back, and the next step allocates no page twice."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req
from freetoken.scheduler.cache import CacheManager
from freetoken.utils import div_ceil

K = 3
NUM_PAGES = 64


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cached", [8, 9, 11])
@pytest.mark.parametrize("n", range(1, K + 2))
def test_rollback_keeps_committed_pages_only(page_size, cached, n):
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, page_size, page_table, "naive")
    req = Req(input_ids=torch.zeros(cached, dtype=torch.int32), table_idx=0, cached_len=0, output_len=32,
              uid=0, sampling_params=SimpleNamespace(), cache_handle=None)
    cm.allocate_paged([req])
    req.complete_one()  # cached_len = cached, the pending token sits at position `cached`

    for _ in range(2):
        entry = req.cached_len
        req.widen_pending(K)
        cm.allocate_paged([req])
        req.commit_pending(n)
        with cm.lazy_free_region():
            cm.rollback_tail(req, req.cached_len, entry + K + 1)
        kept = div_ceil(req.cached_len, page_size)
        assert len(cm.free_slots) == NUM_PAGES - kept
        assert not set(cm.free_slots.tolist()) & set(cm.page_table[0, : kept * page_size : page_size].tolist())


def test_rollback_returns_the_rejected_tails_swa_slots():
    """On a SWA-mapped pool the rejected verify tail's swa slots go back with its full pages
    (the committed rows keep theirs), and the next verify re-charges the span cleanly."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8, sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8, sliding_window=8),
    )
    pool = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=NUM_PAGES, page_size=1,
                            dtype=torch.bfloat16, device=torch.device("cpu"), num_swa_tokens=32)
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, 1, page_table, "naive", swa_pool=pool, sliding_window_size=8)
    assert cm.swa_paged
    req = Req(input_ids=torch.zeros(8, dtype=torch.int32), table_idx=0, cached_len=0, output_len=32,
              uid=0, sampling_params=SimpleNamespace(), cache_handle=None)
    cm.allocate_paged([req])
    req.complete_one()
    entry = req.cached_len
    req.widen_pending(K)
    cm.allocate_paged([req])
    charged = pool.swa_available_size()
    req.commit_pending(1)
    with cm.lazy_free_region():
        cm.rollback_tail(req, req.cached_len, entry + K + 1)
    kept = req.cached_len
    assert len(cm.free_slots) == NUM_PAGES - kept
    assert pool.swa_available_size() == charged + K  # the rejected rows, not the bonus token's row
    tail = page_table[0, kept : entry + K + 1].long()
    assert (pool.full_to_swa_index_mapping[tail] == 0).all()
    assert (pool.full_to_swa_index_mapping[page_table[0, :kept].long()] > 0).all()
    # the next verify re-charges exactly that span
    req.widen_pending(K)
    cm.allocate_paged([req])
    assert pool.swa_available_size() == charged + K - (K + 1)
    assert (pool.full_to_swa_index_mapping[page_table[0, kept : req.device_len].long()] > 0).all()
