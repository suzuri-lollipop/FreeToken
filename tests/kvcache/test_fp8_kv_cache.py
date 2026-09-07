"""Tests for FP8 KV cache quantization."""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fp8_supported():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


requires_fp8 = pytest.mark.skipif(not _fp8_supported(), reason="FP8 requires SM89+")


# ---- MHAKVCache FP8 properties ----


def test_mha_pool_fp8_creation():
    _init_tp()
    from freetoken.kvcache.mha_pool import MHAKVCache

    pool = MHAKVCache(
        num_kv_heads=4, num_layers=2, head_dim=64,
        num_pages=8, page_size=16,
        dtype=torch.float8_e4m3fn, device=_device(),
    )
    assert pool.is_fp8
    assert pool.dtype == torch.float8_e4m3fn
    assert pool._k_scales.shape == (2,)
    assert pool._v_scales.shape == (2,)
    assert pool._k_scales.dtype == torch.float32


def test_mha_pool_bf16_is_not_fp8():
    _init_tp()
    from freetoken.kvcache.mha_pool import MHAKVCache

    pool = MHAKVCache(
        num_kv_heads=4, num_layers=2, head_dim=64,
        num_pages=8, page_size=16,
        dtype=torch.bfloat16, device=_device(),
    )
    assert not pool.is_fp8
    assert not hasattr(pool, "_k_scales")


def test_mha_pool_fp8_scale_accessors():
    _init_tp()
    from freetoken.kvcache.mha_pool import MHAKVCache

    pool = MHAKVCache(
        num_kv_heads=4, num_layers=3, head_dim=64,
        num_pages=8, page_size=16,
        dtype=torch.float8_e4m3fn, device=_device(),
    )
    k_s = pool.k_scale(0)
    v_s = pool.v_scale(2)
    assert k_s.dtype == torch.float32
    assert v_s.dtype == torch.float32
    assert k_s.item() == 1.0
    assert v_s.item() == 1.0


def test_mha_pool_fp8_unit_bytes_halved():
    _init_tp()
    from freetoken.kvcache.mha_pool import MHAKVCache

    bf16_pool = MHAKVCache(
        num_kv_heads=4, num_layers=2, head_dim=64,
        num_pages=8, page_size=16,
        dtype=torch.bfloat16, device=_device(),
    )
    fp8_pool = MHAKVCache(
        num_kv_heads=4, num_layers=2, head_dim=64,
        num_pages=8, page_size=16,
        dtype=torch.float8_e4m3fn, device=_device(),
    )
    bf16_bytes, _ = bf16_pool.unit_bytes()
    fp8_bytes, _ = fp8_pool.unit_bytes()
    assert fp8_bytes == bf16_bytes // 2


# ---- FP8 quantization kernel ----


@requires_fp8
def test_compute_fp8_scale():
    from freetoken.kernel.triton.fp8_kv_cache import FP8_E4M3_MAX, compute_fp8_scale

    x = torch.randn(128, 64, dtype=torch.bfloat16, device=_device())
    scale = torch.zeros(1, dtype=torch.float32, device=_device())
    compute_fp8_scale(x, scale)
    expected = x.abs().max().item() / FP8_E4M3_MAX
    assert abs(scale.item() - expected) < 1e-6


@requires_fp8
def test_quantize_fp8_roundtrip():
    from freetoken.kernel.triton.fp8_kv_cache import FP8_E4M3_MAX, compute_fp8_scale, quantize_fp8

    x = torch.randn(64, 32, dtype=torch.bfloat16, device=_device())
    scale = torch.zeros(1, dtype=torch.float32, device=_device())
    compute_fp8_scale(x, scale)
    x_fp8 = quantize_fp8(x, scale)
    assert x_fp8.dtype == torch.float8_e4m3fn
    x_dequant = x_fp8.to(torch.float32) * scale
    max_err = (x.float() - x_dequant).abs().max().item()
    rel_err = max_err / x.abs().max().item()
    assert rel_err < 0.1


@requires_fp8
def test_compute_fp8_scale_all_zeros_no_nan():
    """An all-zero tensor must not produce scale=0 (which would give 0/0=NaN)."""
    from freetoken.kernel.triton.fp8_kv_cache import (
        MIN_SCALE,
        compute_fp8_scale,
        quantize_fp8,
    )

    x = torch.zeros(64, 32, dtype=torch.bfloat16, device=_device())
    scale = torch.zeros(1, dtype=torch.float32, device=_device())
    compute_fp8_scale(x, scale)
    # fp32 rounds 1e-6 to 9.999999974752427e-07; assert it floored, not exact.
    assert scale.item() > 0.0
    assert scale.item() >= MIN_SCALE * 0.999
    x_fp8 = quantize_fp8(x, scale)
    x_dequant = x_fp8.to(torch.float32) * scale
    assert not torch.isnan(x_dequant).any()
    assert torch.equal(x_dequant, torch.zeros_like(x_dequant))


@requires_fp8
def test_compute_fp8_scale_tiny_values_finite():
    """Subnormal-magnitude activations stay finite through the round-trip."""
    from freetoken.kernel.triton.fp8_kv_cache import compute_fp8_scale, quantize_fp8

    x = (torch.randn(64, 32, dtype=torch.bfloat16, device=_device()) * 1e-5)
    scale = torch.zeros(1, dtype=torch.float32, device=_device())
    compute_fp8_scale(x, scale)
    x_fp8 = quantize_fp8(x, scale)
    x_dequant = x_fp8.to(torch.float32) * scale
    assert not torch.isnan(x_dequant).any()
    assert not torch.isinf(x_dequant).any()


# ---- spec_kv_bytes_per_token with FP8 ----


def test_spec_kv_bytes_per_token_fp8():
    from dataclasses import dataclass

    from freetoken.kvcache.base import spec_kv_bytes_per_token

    @dataclass
    class FakeSpec:
        head_dim: int = 128
        num_kv_heads: int = 8
        num_layers: int = 2
        mla: bool = False
        attn_type: object = None
        index_head_dim: int = 0
        num_index_layers: int = 0
        index_ratio: int = 1
        is_swa: bool = False
        name: str = "full"

        def kv_cache_group_specs(self_):
            return [self_]

    @dataclass
    class FakeConfig:
        dtype: torch.dtype = torch.bfloat16
        kv_cache_itemsize: int = 1
        tp_info: object = None

    from freetoken.distributed import DistributedInfo
    cfg = FakeConfig(tp_info=DistributedInfo(rank=0, size=1))
    spec = FakeSpec()
    result = spec_kv_bytes_per_token(spec, cfg)
    expected = 2 * 128 * 8 * 1 * 2  # 2(K+V) * head_dim * kv_heads * itemsize * layers
    assert result == expected


# ---- _resolve_kv_cache_dtype ----


def test_resolve_kv_cache_dtype_auto():
    from freetoken.engine.engine import _resolve_kv_cache_dtype

    class FakeConfig:
        kv_cache_dtype = "auto"

    assert _resolve_kv_cache_dtype(FakeConfig(), torch.bfloat16) == torch.bfloat16


def test_resolve_kv_cache_dtype_fp8():
    from freetoken.engine.engine import _resolve_kv_cache_dtype

    class FakeConfig:
        kv_cache_dtype = "fp8"

    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9):
        assert _resolve_kv_cache_dtype(FakeConfig(), torch.bfloat16) == torch.float8_e4m3fn
    else:
        with pytest.raises(RuntimeError, match="SM89"):
            _resolve_kv_cache_dtype(FakeConfig(), torch.bfloat16)


def test_resolve_kv_cache_dtype_invalid():
    from freetoken.engine.engine import _resolve_kv_cache_dtype

    class FakeConfig:
        kv_cache_dtype = "int4"

    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_kv_cache_dtype(FakeConfig(), torch.bfloat16)


# ---- FP8-aware auto backend resolution ----


def test_auto_backend_fp8_prefers_triton_for_full():
    from freetoken.attention import AttnType
    from freetoken.engine.engine import _resolve_auto_attention_backend

    # Without fp8, FULL resolves to the hardware-preferred backend (fi/fa/trtllm/triton).
    # With fp8, only the FP8-capable triton survives the filter.
    assert _resolve_auto_attention_backend(frozenset({AttnType.FULL}), fp8=True) == "triton"


def test_auto_backend_fp8_qsa_picks_qsa_sparse():
    from freetoken.attention import AttnType
    from freetoken.engine.engine import _resolve_auto_attention_backend

    assert _resolve_auto_attention_backend(frozenset({AttnType.QSA}), fp8=True) == "qsa_sparse"


def test_auto_backend_fp8_rejects_mla():
    from freetoken.attention import AttnType
    from freetoken.engine.engine import _resolve_auto_attention_backend

    # MLA/DSA/BSA/DSV4 have no FP8-capable backend; the resolver must say so clearly.
    with pytest.raises(RuntimeError, match="FP8-capable"):
        _resolve_auto_attention_backend(frozenset({AttnType.MLA}), fp8=True)


# ---- QSA pool FP8 ----


def test_qsa_pool_fp8_creation():
    """QSA pool with FP8 KV storage keeps the index slab in bf16."""
    _init_tp()
    from freetoken.kvcache.qsa_pool import QSAKVCache

    pool = QSAKVCache(
        num_kv_heads=4, num_layers=2, head_dim=64,
        num_pages=8, page_size=64,
        dtype=torch.float8_e4m3fn, device=_device(),
        index_head_dim=64, num_index_layers=1, index_ratio=4,
        num_req_slots=4,
    )
    assert pool.is_fp8
    assert pool.dtype == torch.float8_e4m3fn
    assert pool._index_dtype == torch.bfloat16
    assert pool._cmp_k_buffer.dtype == torch.bfloat16
    assert pool._k_scales.shape == (2,)


# ---- End-to-end: FP8 store -> Triton paged attention read ----


def _ref_attention(q, k, v, group, sm_scale):
    """Float32 reference causal-decode attention. q:[1,H,D], k/v:[S,KH,D]."""
    q = q[0].float()  # [H, D]
    k = k.float()     # [S, KH, D]
    v = v.float()     # [S, KH, D]
    H, D = q.shape
    S, KH, _ = k.shape
    out = torch.empty(H, D, dtype=torch.float32, device=q.device)
    for h in range(H):
        kvh = h // group
        scores = (q[h] @ k[:, kvh].T) * sm_scale  # [S]
        p = torch.softmax(scores, dim=0)
        out[h] = p @ v[:, kvh]  # [D]
    return out


@requires_fp8
def test_fp8_paged_attention_matches_bf16_reference():
    """Full path: store_kv quantizes to fp8, paged_attention dequantizes on read.

    The fp8 output must track the float32 reference within fp8 tolerance.
    K/V are 2D [tokens, kv_heads*head_dim] as the real model passes them.
    """
    _init_tp()
    from freetoken.kernel.triton.attention import paged_attention
    from freetoken.kvcache.mha_pool import MHAKVCache

    torch.manual_seed(0)
    num_q_heads, num_kv_heads, head_dim, seq_len = 8, 4, 64, 16
    group = num_q_heads // num_kv_heads
    kv_dim = num_kv_heads * head_dim
    sm_scale = head_dim ** -0.5
    dev = _device()

    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(seq_len, kv_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(seq_len, kv_dim, dtype=torch.bfloat16, device=dev)

    pool = MHAKVCache(
        num_kv_heads=num_kv_heads, num_layers=1, head_dim=head_dim,
        num_pages=seq_len + 1, page_size=1,
        dtype=torch.float8_e4m3fn, device=dev,
    )
    out_loc = torch.arange(seq_len, dtype=torch.int32, device=dev)
    pool.store_kv(k, v, out_loc, layer_id=0)

    k_cache = pool.k_cache(0).view(-1, num_kv_heads, head_dim)
    v_cache = pool.v_cache(0).view(-1, num_kv_heads, head_dim)
    indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
    indices = torch.arange(seq_len, dtype=torch.int32, device=dev)
    q_to_req = torch.zeros(1, dtype=torch.int32, device=dev)
    q_positions = torch.tensor([seq_len - 1], dtype=torch.int64, device=dev)

    out_fp8 = paged_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        indptr=indptr, indices=indices,
        q_to_req=q_to_req, q_positions=q_positions,
        sm_scale=sm_scale,
        k_scale=pool.k_scale(0), v_scale=pool.v_scale(0),
    )

    k3 = k.view(seq_len, num_kv_heads, head_dim)
    v3 = v.view(seq_len, num_kv_heads, head_dim)
    ref = _ref_attention(q, k3, v3, group, sm_scale)
    # fp8-e4m3 has ~2 decimal digits; relative error per element stays modest.
    rel_err = (out_fp8[0].float() - ref).abs().max().item() / ref.abs().max().item()
    assert not torch.isnan(out_fp8).any()
    assert rel_err < 0.15, f"fp8 attention rel_err {rel_err} too high"


@requires_fp8
def test_fp8_paged_attention_zero_kv_no_nan():
    """All-zero K/V (e.g. padding) must not NaN the attention output."""
    _init_tp()
    from freetoken.kernel.triton.attention import paged_attention
    from freetoken.kvcache.mha_pool import MHAKVCache

    num_q_heads, num_kv_heads, head_dim, seq_len = 4, 2, 64, 8
    kv_dim = num_kv_heads * head_dim
    sm_scale = head_dim ** -0.5
    dev = _device()

    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.zeros(seq_len, kv_dim, dtype=torch.bfloat16, device=dev)
    v = torch.zeros(seq_len, kv_dim, dtype=torch.bfloat16, device=dev)

    pool = MHAKVCache(
        num_kv_heads=num_kv_heads, num_layers=1, head_dim=head_dim,
        num_pages=seq_len + 1, page_size=1,
        dtype=torch.float8_e4m3fn, device=dev,
    )
    out_loc = torch.arange(seq_len, dtype=torch.int32, device=dev)
    pool.store_kv(k, v, out_loc, layer_id=0)

    k_cache = pool.k_cache(0).view(-1, num_kv_heads, head_dim)
    v_cache = pool.v_cache(0).view(-1, num_kv_heads, head_dim)
    out = paged_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        indptr=torch.tensor([0, seq_len], dtype=torch.int32, device=dev),
        indices=torch.arange(seq_len, dtype=torch.int32, device=dev),
        q_to_req=torch.zeros(1, dtype=torch.int32, device=dev),
        q_positions=torch.tensor([seq_len - 1], dtype=torch.int64, device=dev),
        sm_scale=sm_scale,
        k_scale=pool.k_scale(0), v_scale=pool.v_scale(0),
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@requires_fp8
def test_qsa_sparse_fp8_kernel_compiles_at_real_dims():
    """Qwen3.8-Flash-Next QSA dims (24 q / 2 kv / head_dim 256, group 12 -> BLOCK_M 16).

    The float32-dequant version overflowed the 101KB shared-memory limit here
    (OutOfResources: Required 180224). Dequantizing to bf16 keeps the dot operands
    at their original size, so the kernel must compile and run.
    """
    from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention

    torch.manual_seed(0)
    num_q_heads, num_kv_heads, head_dim = 24, 2, 256
    page_size, num_blocks, topk = 64, 2, 64
    dev = _device()

    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k_cache = torch.randn(num_blocks, page_size, num_kv_heads, head_dim, device=dev).to(
        torch.float8_e4m3fn
    )
    v_cache = torch.randn(num_blocks, page_size, num_kv_heads, head_dim, device=dev).to(
        torch.float8_e4m3fn
    )
    # All selected tokens live in physical page 0 (block_table[0,0]=0).
    logical_indices = torch.arange(topk, dtype=torch.int32, device=dev).view(1, topk)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=dev)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=dev)
    k_scale = torch.full((1,), 0.01, dtype=torch.float32, device=dev)
    v_scale = torch.full((1,), 0.01, dtype=torch.float32, device=dev)

    out = qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req,
        k_scale=k_scale, v_scale=v_scale,
    )
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@requires_fp8
def test_fp8_decode_attention_compiles_at_head_dim_256():
    """The split-K decode kernel (triton backend, FULL attention) must not overflow
    shared memory with fp8 dequant at head_dim 256 -- the same float32-operand
    regression that hit the QSA kernel."""
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(0)
    num_q_heads, num_kv_heads, head_dim, seq_len = 24, 2, 256, 32
    max_kv_splits = 8
    dev = _device()

    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k_cache = torch.randn(seq_len, num_kv_heads, head_dim, device=dev).to(torch.float8_e4m3fn)
    v_cache = torch.randn(seq_len, num_kv_heads, head_dim, device=dev).to(torch.float8_e4m3fn)
    attn_logits = torch.empty(1, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=dev)
    attn_lse = torch.empty(1, num_q_heads, max_kv_splits, dtype=torch.float32, device=dev)
    num_kv_splits = torch.full((1,), max_kv_splits, dtype=torch.int32, device=dev)
    k_scale = torch.full((1,), 0.01, dtype=torch.float32, device=dev)
    v_scale = torch.full((1,), 0.01, dtype=torch.float32, device=dev)

    out = decode_paged_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        indptr=torch.tensor([0, seq_len], dtype=torch.int32, device=dev),
        indices=torch.arange(seq_len, dtype=torch.int32, device=dev),
        q_positions=torch.tensor([seq_len - 1], dtype=torch.int64, device=dev),
        attn_logits=attn_logits, attn_lse=attn_lse,
        num_kv_splits=num_kv_splits, max_kv_splits=max_kv_splits,
        sm_scale=head_dim ** -0.5,
        k_scale=k_scale, v_scale=v_scale,
    )
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
