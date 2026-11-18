"""The fused multi-bank ``copy_missing`` path must move exactly the same bytes as the
legacy per-bank ``fast_index_copy_jit`` loop, for every miss count (including the
zero-copy case), across banks of differing per-row sizes.
"""

from __future__ import annotations

import time

import pytest
import torch

from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# mxfp4_triton 6-bank schema with mixed 16B-aligned per-row sizes (bytes), >=256 so the
# legacy per-bank kernel's vectorized template is valid. Sizes not covered by a model in
# kernel/aot_models.py must be listed in its TEST_FEATURE_SIZES so the per-bank kernels
# stay prebuilt under FREETOKEN_DISABLE_JIT=1.
FEATS = [8192, 512, 256, 4096, 512, 256]


def _build_cache(num_layers, num_experts, cache_size):
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=dev, cache_policy="lru", prefill_overlap=False, quant_format="mxfp4_triton",
    )
    schema = _BANK_SCHEMAS["mxfp4_triton"]
    # Views into one flat tensor: only per-layer addressing matters here, not
    # independent allocations.
    sources = {
        name: list(torch.randint(0, 256, (num_layers * num_experts, feat), dtype=torch.uint8, device=dev)
                   .split(num_experts))
        for name, feat in zip(schema, FEATS)
    }
    cache.set_bank_sources(sources)  # also builds the fused-copy descriptor
    return cache


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("num_indices", [0, 1, 4, 8])
def test_fused_copy_matches_per_bank(num_indices):
    num_layers, num_experts, cache_size = 8, 8, 32
    layer_id = 3  # exercise a non-zero per-layer source selection, not just layer 0
    cache = _build_cache(num_layers, num_experts, cache_size)
    assert cache._copy_fused_ok, "fused copy should activate for 16B-aligned banks"
    # copy_missing resolves the per-layer source through this (normally set by
    # ensure_experts/materialize_layer); poked directly here since this test drives
    # evict_slots/src_indices/num_indices by hand.
    cache._pending_src_layer = layer_id

    cache.num_indices.fill_(num_indices)
    if num_indices:
        dev = torch.device("cuda")
        cache.evict_slots[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % cache_size
        # src_indices are layer-local expert rows (0..num_experts) under the new contract.
        cache.src_indices[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % num_experts

    # reference: legacy per-bank loop
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = False
    cache.copy_missing()
    torch.cuda.synchronize()
    ref = [c.clone() for _, c in cache.banks]

    # fused multi-bank launch
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = True
    cache.copy_missing()
    torch.cuda.synchronize()

    for b, (r, (_, c)) in enumerate(zip(ref, cache.banks)):
        assert torch.equal(r, c), f"bank {b} (feat={FEATS[b]}) fused != per-bank at num_indices={num_indices}"


@CUDA
def test_fused_copy_identity_rows_and_wait_flag():
    # without src_indices row i copies source row i; with wait_flag the copy holds until the flag is set (the locked-layer staging)
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr

    dev = torch.device("cuda")
    feats = [512, 256]
    src = [alloc_pinned_tensor(4 * f, dtype=torch.uint8) for f in feats]
    for t in src:
        t.random_(0, 256)
    dst = [torch.zeros(16 * f, dtype=torch.uint8, device=dev) for f in feats]
    dst_ptrs = torch.tensor([t.data_ptr() for t in dst], dtype=torch.int64, device=dev)
    src_ptrs = torch.tensor([device_ptr(t) for t in src], dtype=torch.int64, device=dev)
    feat_bytes = torch.tensor(feats, dtype=torch.int64, device=dev)
    slots = torch.tensor([5, 2, 9, 0], dtype=torch.int32, device=dev)
    num = torch.tensor([3], dtype=torch.int64, device=dev)
    flag = alloc_pinned_tensor(1, dtype=torch.int64)
    flag.fill_(0)

    fast_index_copy_multi_jit(dst_ptrs, src_ptrs, feat_bytes, slots, None, num, wait_flag=device_ptr(flag))
    time.sleep(0.2)
    assert not torch.cuda.current_stream().query(), "the copy ran before its flag was set"
    flag.fill_(1)
    torch.cuda.synchronize()
    for d, s, f in zip(dst, src, feats):
        for i, slot in enumerate([5, 2, 9]):
            assert torch.equal(d[slot * f:(slot + 1) * f].cpu(), s[i * f:(i + 1) * f])
        assert not d[0:f].any(), "num_indices bounds the rows"


# The prefill miss gather's grid width is a PCIe queue-depth policy, not a copy
# correctness one: a wide grid silently re-queues concurrent decode fetches behind
# the chunk (the piped-prefill ITL regression), so pin the resolution here.
@pytest.mark.parametrize(
    "device_name,expected",
    [("RTX PRO 4000 Blackwell", 2), ("H100", 2), ("GH200", 32), ("GB200", 32)],
)
def test_prefill_h2d_width_stays_at_the_link_knee(monkeypatch, device_name, expected):
    from freetoken.kernel import fast_index_copy as fic

    monkeypatch.delenv("FREETOKEN_PREFILL_H2D_BPB", raising=False)
    monkeypatch.delenv("FREETOKEN_H2D_BLOCKS_PER_BANK", raising=False)
    monkeypatch.setattr(fic.torch.cuda, "get_device_name", lambda device: device_name)
    assert fic.prefill_h2d_blocks_per_bank(torch.device("cuda")) == expected


def test_prefill_h2d_width_env_override(monkeypatch):
    from freetoken.kernel import fast_index_copy as fic

    monkeypatch.setenv("FREETOKEN_PREFILL_H2D_BPB", "8")
    assert fic.prefill_h2d_blocks_per_bank(torch.device("cuda")) == 8
    monkeypatch.setenv("FREETOKEN_PREFILL_H2D_BPB", "1")
    assert fic.prefill_h2d_blocks_per_bank(torch.device("cuda")) == 1
    monkeypatch.setenv("FREETOKEN_PREFILL_H2D_BPB", "0")
    with pytest.raises(AssertionError):
        fic.prefill_h2d_blocks_per_bank(torch.device("cuda"))
