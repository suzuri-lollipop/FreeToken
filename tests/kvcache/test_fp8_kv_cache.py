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
