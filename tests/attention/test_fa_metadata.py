"""FlashAttention backend host-side metadata and capture bookkeeping
(attention/fa.py), CPU-only.

The FA3/FA4 kernels read exactly what prepare_metadata lays out: a wrong
cu_seqlens entry, a page table that forgot the page-size floor, or a capture
buffer that replays stale bounds still runs to completion and returns a
plausible tensor over whatever rows the layout happens to name. The forward
stays out of scope (that parity gate lives in the kernel suite); everything here
is the silent-failure host arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.attention import fa as fa_mod
from freetoken.attention.fa import FAMetadata, FlashAttentionBackend
from freetoken.distributed import set_tp_info, try_get_tp_info


class StubKVCache:
    def __init__(self, page_size, quant=None):
        self.device = torch.device("cpu")
        self.page_size = page_size
        self.quant = quant

    def k_cache(self, layer_id):
        raise AssertionError("forward is out of scope here")

    def v_cache(self, layer_id):
        raise AssertionError("forward is out of scope here")

    def store_kv(self, k, v, out_loc, layer_id):
        raise AssertionError("forward is out of scope here")


class StubBatch:
    def __init__(self, reqs, attn_metadata=None, **kw):
        self.padded_reqs = reqs
        self.attn_metadata = attn_metadata
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _fresh_ctx():
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(fa_mod, "is_arch_supported", lambda *a: False)
    monkeypatch.setattr(fa_mod, "is_sm100_supported", lambda: False)
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    def build(page_size, quant=None, n_tables=4, table_len=16):
        ctx = core.Context(page_size=page_size)
        ctx.kv_cache = StubKVCache(page_size, quant)
        seq = torch.arange(n_tables * table_len, dtype=torch.int32)
        ctx.page_table = seq.view(n_tables, table_len)
        core.set_global_ctx(ctx)
        config = SimpleNamespace(head_dim=128, num_kv_heads=4)
        return FlashAttentionBackend(config), ctx

    return build


def _req(extend_len, device_len, cached_len, table_idx):
    return SimpleNamespace(
        extend_len=extend_len, device_len=device_len, cached_len=cached_len, table_idx=table_idx
    )


def test_decode_metadata_uses_arange_cu_seqlens_q(backend):
    fa_backend, ctx = backend(page_size=1)
    reqs = [_req(extend_len=1, device_len=5, cached_len=4, table_idx=1)]
    batch = StubBatch(reqs)
    fa_backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert isinstance(meta, FAMetadata)
    assert meta.max_seqlen_q == 1
    assert meta.max_seqlen_k == 5
    assert meta.cu_seqlens_q.tolist() == [0, 1]  # decode: arange, independent of history
    assert meta.cu_seqlens_k.tolist() == [0, 5]
    assert meta.cache_seqlens.tolist() == [5]
    # page_size=1: the request's table row, first max_seqlen_k entries
    assert meta.page_table.tolist() == [[16, 17, 18, 19, 20]]
    assert meta.get_last_indices(bs=1).tolist() == [0]


def test_prefill_without_cache_hit_shares_cu_seqlens(backend):
    fa_backend, ctx = backend(page_size=1)
    batch = StubBatch(
        [_req(extend_len=8, device_len=8, cached_len=0, table_idx=0),
         _req(extend_len=2, device_len=2, cached_len=0, table_idx=3)]
    )
    fa_backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.max_seqlen_q == 8
    assert meta.cu_seqlens_q.tolist() == [0, 8, 10]  # == cu_seqlens_k (no partial hit)
    assert meta.cu_seqlens_k.tolist() == [0, 8, 10]
    assert meta.cache_seqlens.tolist() == [8, 2]
    assert meta.get_last_indices(bs=2).tolist() == [7, 9]


def test_prefill_with_partial_hit_builds_own_cumsum(backend):
    fa_backend, ctx = backend(page_size=1)
    batch = StubBatch([_req(extend_len=3, device_len=8, cached_len=5, table_idx=0)])
    fa_backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.cu_seqlens_q.tolist() == [0, 3]  # only the NEW portion
    assert meta.cu_seqlens_k.tolist() == [0, 8]  # full history
    assert meta.cache_seqlens.tolist() == [8]


def test_page_table_divides_by_page_size(backend):
    fa_backend, ctx = backend(page_size=2)
    batch = StubBatch([_req(extend_len=1, device_len=5, cached_len=4, table_idx=1)])
    fa_backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    # stride 2 over the table row [16..20] -> [16, 18, 20], floored by page_size
    assert meta.page_table.tolist() == [[8, 9, 10]]


def test_fp8_query_and_descaled_buffers_grow_on_demand(backend):
    quant = SimpleNamespace(
        k_scale=torch.tensor(448.0), v_scale=torch.tensor(2.0)
    )
    fa_backend, ctx = backend(page_size=1, quant=quant)

    q = torch.tensor([448.0, -448.0, 0.0, 224000.0])
    q_fp8, q_d, k_d, v_d = fa_backend._fp8_query(q, bs=2)
    assert q_fp8.dtype == torch.float8_e4m3fn
    # 224000/448 = 500 clamps to the e4m3 max (448); 1 and 0 are exact
    assert q_fp8.float().tolist() == [1.0, -1.0, 0.0, 448.0]
    assert list(fa_backend._descales[0].shape) == [64, 4]  # rows = max(bs, 64)
    assert torch.all(q_d == 448.0) and torch.all(k_d == 448.0) and torch.all(v_d == 2.0)

    first = fa_backend._descales
    fa_backend._fp8_query(q, bs=100)  # forces a regrow past the 64-row floor
    assert fa_backend._descales is not first
    assert list(fa_backend._descales[0].shape) == [100, 4]


def test_capture_init_and_prepare_replay_roundtrip(backend):
    fa_backend, ctx = backend(page_size=2)
    fa_backend.init_capture_graph(max_seq_len=16, bs_list=[1, 4, 2])

    assert fa_backend.max_graph_bs == 4
    assert fa_backend.capture_bs == [1, 2, 4]
    assert list(fa_backend.capture.page_table.shape) == [4, 8]  # max_seq_len // page_size

    # capture-side metadata for bs=2: slices of the static capture buffers
    batch = StubBatch(reqs=[], size=2)
    fa_backend.prepare_for_capture(batch)
    misshaped = batch.attn_metadata
    assert misshaped.cu_seqlens_k.tolist() == [0, 1, 2]
    assert misshaped.max_seqlen_k == 16  # page_table cols * page_size

    # a bs the capture list does not contain must never slide through
    with pytest.raises(AssertionError):
        fa_backend.prepare_for_capture(StubBatch(reqs=[], size=3))

    # replay copies the live metadata into the capture buffers in place
    fake = FAMetadata(
        cu_seqlens_k=torch.tensor([0, 5, 9]),
        cu_seqlens_q=torch.tensor([0, 1, 2]),
        cache_seqlens=torch.tensor([5, 4]),
        max_seqlen_k=9,
        max_seqlen_q=1,
        page_table=torch.tensor([[1, 2, 3], [4, 5, 6]]),
    )
    fa_backend.prepare_for_replay(StubBatch(reqs=[], padded_size=2, attn_metadata=fake))
    assert fa_backend.capture.cu_seqlens_k[:3].tolist() == [0, 5, 9]
    assert fa_backend.capture.seq_lens[:2].tolist() == [5, 4]
    assert fa_backend.capture.page_table[:2, :3].tolist() == [[1, 2, 3], [4, 5, 6]]