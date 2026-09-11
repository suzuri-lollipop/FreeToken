"""The e4m3 read path: an fp8 pool read through the Triton attention kernels must match
the same kernel run on a cache holding the dequantized values, which is the quantization
error and nothing else."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.attention import (
    decode_paged_attention,
    extend_paged_attention,
    paged_attention,
)
from freetoken.kernel.triton.kv_quant import store_kv_e4m3
from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

K_SCALE, V_SCALE = 0.5, 4.0


def _setup(lens, q_heads=6, kv_heads=2, head_dim=64, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    total = sum(lens)
    starts = [0]
    for length in lens[:-1]:
        starts.append(starts[-1] + length)
    indices = torch.cat(
        [torch.arange(s, s + n, dtype=torch.int32, device=device) for s, n in zip(starts, lens)]
    )
    indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device)
    k_src = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16) * 3
    v_src = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16) * 2
    k8 = torch.zeros(total, kv_heads * head_dim, device=device, dtype=torch.float8_e4m3fn)
    v8 = torch.zeros(total, kv_heads * head_dim, device=device, dtype=torch.float8_e4m3fn)
    store_kv_e4m3(k8, v8, indices, k_src, v_src, K_SCALE, V_SCALE)
    k_fp8 = k8.view(total, kv_heads, head_dim)
    v_fp8 = v8.view(total, kv_heads, head_dim)
    # The reference cache holds exactly what the fp8 pool holds, one dtype wider: the
    # difference against it is the kernel's handling of the scales, not the quantization.
    k_wide = (k_fp8.float() * K_SCALE).to(torch.bfloat16)
    v_wide = (v_fp8.float() * V_SCALE).to(torch.bfloat16)
    q = torch.randn(len(lens), q_heads, head_dim, device=device, dtype=torch.bfloat16)
    return dict(
        lens=lens,
        indices=indices,
        indptr=indptr,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        k_wide=k_wide,
        v_wide=v_wide,
        q=q,
        sm_scale=head_dim**-0.5,
    )


def _assert_matches(fp8_out, wide_out):
    assert not bool(fp8_out.isnan().any())
    # bf16 rounding at the output magnitude, not a scale bug (a missed descale is off by
    # the scale factor itself, i.e. 2x/4x here).
    torch.testing.assert_close(fp8_out.float(), wide_out.float(), atol=2e-2, rtol=2e-2)


def test_store_kernel_quantizes_clamps_and_leaves_other_rows():
    device = "cuda"
    rows, heads, dim = 6, 2, 64
    torch.manual_seed(1)
    k = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16) * 10
    v = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16) * 10
    k[0, 0, 0] = 1e5  # outlier: clamps to the grid edge, never the NaN code
    k_cache = torch.zeros(32, heads * dim, device=device, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros(32, heads * dim, device=device, dtype=torch.float8_e4m3fn)
    idx = torch.tensor([5, 1, 30, 0, 7, 2], device=device, dtype=torch.int32)
    store_kv_e4m3(k_cache, v_cache, idx, k, v, K_SCALE, V_SCALE)
    expect = (k.reshape(rows, -1).float() / K_SCALE).clamp(-448.0, 448.0)
    assert torch.equal(k_cache[idx.long()].float(), expect.to(torch.float8_e4m3fn).float())
    assert not bool(k_cache.float().isnan().any())
    untouched = torch.ones(32, heads * dim, device=device, dtype=torch.bool)
    untouched[idx.long()] = False
    assert torch.equal(k_cache[untouched].float(), torch.zeros_like(k_cache[untouched].float()))


def test_paged_attention_reads_e4m3_cache():
    s = _setup([48, 17, 1])
    q_to_req = torch.arange(len(s["lens"]), dtype=torch.int32, device="cuda")
    q_positions = torch.tensor([n - 1 for n in s["lens"]], dtype=torch.int32, device="cuda")
    fp8 = paged_attention(
        q=s["q"], k_cache=s["k_fp8"], v_cache=s["v_fp8"], indptr=s["indptr"],
        indices=s["indices"], q_to_req=q_to_req, q_positions=q_positions,
        sm_scale=s["sm_scale"], k_scale=K_SCALE, v_scale=V_SCALE,
    )
    wide = paged_attention(
        q=s["q"], k_cache=s["k_wide"], v_cache=s["v_wide"], indptr=s["indptr"],
        indices=s["indices"], q_to_req=q_to_req, q_positions=q_positions,
        sm_scale=s["sm_scale"],
    )
    _assert_matches(fp8, wide)


def test_decode_attention_reads_e4m3_cache():
    s = _setup([96, 33, 5])
    bs = len(s["lens"])
    device = "cuda"
    head_dim = s["q"].shape[-1]
    logits = torch.empty(bs, s["q"].shape[1], 8, head_dim, device=device, dtype=torch.float32)
    lse = torch.empty(bs, s["q"].shape[1], 8, device=device, dtype=torch.float32)
    splits = torch.full((bs,), 8, dtype=torch.int32, device=device)
    q_positions = torch.tensor([n - 1 for n in s["lens"]], dtype=torch.int32, device=device)
    kwargs = dict(
        q=s["q"], indptr=s["indptr"], indices=s["indices"], q_positions=q_positions,
        attn_logits=logits, attn_lse=lse, num_kv_splits=splits, max_kv_splits=8,
        sm_scale=s["sm_scale"],
    )
    fp8 = decode_paged_attention(
        k_cache=s["k_fp8"], v_cache=s["v_fp8"], k_scale=K_SCALE, v_scale=V_SCALE, **kwargs
    )
    wide = decode_paged_attention(k_cache=s["k_wide"], v_cache=s["v_wide"], **kwargs)
    _assert_matches(fp8, wide)


def test_extend_attention_descales_only_the_cached_half():
    """The current step's k/v are still in the model dtype: only the cached prefix carries
    a descale. Both extend kernels (the plain one and the extend-aware split one) check."""
    s = _setup([64, 25, 9])
    bs = len(s["lens"])
    device = "cuda"
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(s["lens"], dtype=torch.int32, device=device)
    new_k = torch.randn(bs, s["k_fp8"].shape[1], s["k_fp8"].shape[2], device=device,
                        dtype=torch.bfloat16)
    new_v = torch.randn_like(new_k)
    for k_extend, v_extend in ((None, None), (new_k, new_v)):
        kwargs = dict(
            q=s["q"], qo_indptr=qo_indptr, kv_indptr=s["indptr"], kv_indices=s["indices"],
            prefix_lens=prefix_lens, max_q_len=1, sm_scale=s["sm_scale"],
            k_extend=k_extend, v_extend=v_extend,
        )
        fp8 = extend_paged_attention(
            k_cache=s["k_fp8"], v_cache=s["v_fp8"], k_scale=K_SCALE, v_scale=V_SCALE, **kwargs
        )
        wide = extend_paged_attention(
            k_cache=s["k_wide"], v_cache=s["v_wide"], **kwargs
        )
        _assert_matches(fp8, wide)


def test_a_missing_descale_is_a_visible_error_not_a_silent_one():
    """Sanity check on the test itself: reading the fp8 pool as if it needed no descale
    moves the output by the scale factor, which the tolerance above would catch."""
    s = _setup([48, 17, 1])
    q_to_req = torch.arange(len(s["lens"]), dtype=torch.int32, device="cuda")
    q_positions = torch.tensor([n - 1 for n in s["lens"]], dtype=torch.int32, device="cuda")
    common = dict(
        q=s["q"], k_cache=s["k_fp8"], v_cache=s["v_fp8"], indptr=s["indptr"],
        indices=s["indices"], q_to_req=q_to_req, q_positions=q_positions,
        sm_scale=s["sm_scale"],
    )
    right = paged_attention(**common, k_scale=K_SCALE, v_scale=V_SCALE)
    wrong = paged_attention(**common)  # scales default to 1.0
    assert (right.float() - wrong.float()).abs().max() > 0.1


# ---- QSA sparse attend ------------------------------------------------------------------
#
# The QSA kernel reads its K/V through a page table instead of a flat index list, and its
# split-K partials are stored normalized, so the descale rides the partial rather than every
# tile. Both the split=1 direct store and the split-K + merge path are checked.

PAGE_SIZE = 16
PAGES_PER_REQ = 4


def _qsa_setup(topk, head_dim=64, num_rows=3, q_heads=6, kv_heads=2, seed=0):
    """Two requests' worth of pages in fp8 plus the same rows one dtype wider, with a
    selection that spans several pages of each request. Columns past a row's visible length
    are -1, the padding the QSA selection emits for tokens not generated yet."""
    torch.manual_seed(seed)
    device = "cuda"
    num_reqs = 2
    total = num_reqs * PAGES_PER_REQ * PAGE_SIZE
    k_src = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16) * 3
    v_src = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16) * 2
    k8 = torch.zeros(total, kv_heads * head_dim, device=device, dtype=torch.float8_e4m3fn)
    v8 = torch.zeros(total, kv_heads * head_dim, device=device, dtype=torch.float8_e4m3fn)
    slots = torch.arange(total, dtype=torch.int32, device=device)
    store_kv_e4m3(k8, v8, slots, k_src, v_src, K_SCALE, V_SCALE)
    pages = (num_reqs * PAGES_PER_REQ, PAGE_SIZE, kv_heads, head_dim)
    k_fp8, v_fp8 = k8.view(*pages), v8.view(*pages)
    k_wide = (k_fp8.float() * K_SCALE).to(torch.bfloat16)
    v_wide = (v_fp8.float() * V_SCALE).to(torch.bfloat16)
    tokens_per_req = PAGES_PER_REQ * PAGE_SIZE
    block_table = torch.arange(
        num_reqs * PAGES_PER_REQ, dtype=torch.int32, device=device
    ).view(num_reqs, PAGES_PER_REQ)
    indices = torch.full((num_rows, topk), -1, dtype=torch.int32)
    for row in range(num_rows):
        visible = max(1, min(topk, 5 + 7 * row))
        picked = (torch.arange(visible) % tokens_per_req) + (row % num_reqs) * tokens_per_req
        indices[row, :visible] = picked.to(torch.int32)
    return dict(
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        k_wide=k_wide,
        v_wide=v_wide,
        indices=indices.to(device),
        block_table=block_table,
        token_to_req=(torch.arange(num_rows) % num_reqs).to(device=device, dtype=torch.int32),
        q=torch.randn(num_rows, q_heads, head_dim, device=device, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    "topk, head_dim",
    [
        (16, 64),  # one tile -> the split=1 kernel that stores the output directly
        (64, 64),  # four tiles -> split-K
        (96, 64),  # a selection that does not divide into equal splits
        (64, 256),  # the shipping QSA head_dim
    ],
)
def test_qsa_sparse_attention_reads_e4m3_cache(topk, head_dim):
    s = _qsa_setup(topk, head_dim=head_dim)
    selection = (s["indices"], s["block_table"], s["token_to_req"])
    fp8 = qsa_sparse_paged_attention(
        s["q"], s["k_fp8"], s["v_fp8"], *selection, k_scale=K_SCALE, v_scale=V_SCALE
    )
    wide = qsa_sparse_paged_attention(s["q"], s["k_wide"], s["v_wide"], *selection)
    _assert_matches(fp8, wide)
