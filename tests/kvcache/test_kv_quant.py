"""Quantized KV storage: the dtype/scale resolution, the byte cost it implies, and the
pool families that may carry it. GPU round-trips skip themselves without CUDA."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache import check_kv_quant
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.kv_quant import KVQuant, parse_kv_cache_dtype, resolve_kv_quant

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _engine_config(dtype=torch.bfloat16, kv_cache_dtype="auto", scale=1.0):
    return SimpleNamespace(
        dtype=dtype, kv_cache_dtype=kv_cache_dtype, kv_cache_quant_scale=scale
    )


def test_parse_kv_cache_dtype():
    assert parse_kv_cache_dtype(None) is None
    assert parse_kv_cache_dtype("auto") is None
    assert parse_kv_cache_dtype("fp8_e4m3") is torch.float8_e4m3fn
    # the spelling users type coming from other engines resolves to the same storage
    for alias in ("fp8", "fp8e4m3", "e4m3", "float8_e4m3fn"):
        assert parse_kv_cache_dtype(alias) is torch.float8_e4m3fn
    with pytest.raises(ValueError, match="unknown kv-cache dtype"):
        parse_kv_cache_dtype("int8")


def test_kvquant_validates_its_pair():
    quant = KVQuant(dtype=torch.float8_e4m3fn, k_scale=2.0, v_scale=0.5)
    assert quant.itemsize == 1
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            KVQuant(dtype=torch.float8_e4m3fn, k_scale=bad)
    with pytest.raises(ValueError, match="unsupported kv-cache quant dtype"):
        KVQuant(dtype=torch.bfloat16)


def test_resolve_kv_quant_auto_keeps_the_model_dtype():
    assert resolve_kv_quant(_engine_config(), "auto", 1.0) is None


def test_resolve_kv_quant_needs_native_fp8(monkeypatch):
    import freetoken.kvcache.kv_quant as kv_quant

    monkeypatch.setattr(kv_quant, "has_native_e4m3", lambda: False)
    with pytest.raises(RuntimeError, match="native fp8e4nv"):
        resolve_kv_quant(_engine_config(), "fp8_e4m3", 1.0)


def test_resolve_kv_quant_rejects_a_storage_that_saves_nothing(monkeypatch):
    import freetoken.kvcache.kv_quant as kv_quant

    monkeypatch.setattr(kv_quant, "has_native_e4m3", lambda: True)
    with pytest.raises(RuntimeError, match="stores no smaller"):
        resolve_kv_quant(_engine_config(dtype=torch.float8_e4m3fn), "fp8_e4m3", 1.0)


def test_resolve_kv_quant_records_the_compute_dtype(monkeypatch):
    import freetoken.kvcache.kv_quant as kv_quant

    monkeypatch.setattr(kv_quant, "has_native_e4m3", lambda: True)
    quant = resolve_kv_quant(_engine_config(dtype=torch.float16), "fp8_e4m3", 4.0)
    assert quant.dtype is torch.float8_e4m3fn
    assert (quant.k_scale, quant.v_scale) == (4.0, 4.0)
    # plan()-time backends need the query's dtype, not just the cache's.
    assert quant.compute_dtype is torch.float16


def _spec(**kwargs):
    from freetoken.models.config import KVCacheGroupSpec

    return KVCacheGroupSpec(
        name="full",
        layer_ids=(0, 1, 2, 3),
        num_kv_heads=2,
        head_dim=64,
        sliding_window=None,
        **kwargs,
    )


def test_spec_kv_bytes_per_token_follows_the_quant_dtype():
    _init_tp()
    spec = _spec()
    plain = SimpleNamespace(dtype=torch.bfloat16, kv_quant=None, tp_info=SimpleNamespace(size=1))
    quant = SimpleNamespace(
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        kv_quant=KVQuant(dtype=torch.float8_e4m3fn),
    )
    # 2 slabs x head_dim x kv heads x bytes/elem x layers
    assert spec_kv_bytes_per_token(spec, plain) == 2 * 64 * 2 * 2 * 4
    assert spec_kv_bytes_per_token(spec, quant) == 2 * 64 * 2 * 1 * 4


def test_duck_typed_config_without_kv_quant_still_sizes():
    # Older/duck-typed configs (and the sizing tests' stand-ins) carry no kv_quant at all.
    _init_tp()
    legacy = SimpleNamespace(dtype=torch.bfloat16, tp_info=SimpleNamespace(size=1))
    assert spec_kv_bytes_per_token(_spec(), legacy) == 2 * 64 * 2 * 2 * 4


def _mha_pool(dtype=torch.bfloat16, quant=None, pages=64, page_size=8, device="cpu"):
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    return MHAKVCache(
        num_kv_heads=2,
        num_layers=4,
        head_dim=64,
        num_pages=pages,
        page_size=page_size,
        dtype=dtype,
        device=torch.device(device),
        quant=quant,
    )


def test_mha_pool_stores_in_the_quant_dtype():
    plain = _mha_pool()
    quant = _mha_pool(quant=KVQuant(dtype=torch.float8_e4m3fn))
    assert plain.dtype is torch.bfloat16 and plain.quant is None
    assert quant.dtype is torch.float8_e4m3fn
    assert quant.quant.k_scale == 1.0
    # Half the bytes per cached token, and unit_bytes (what the cache sliders and the
    # capacity report denominate in) has to say so.
    assert quant.unit_bytes() == (plain.unit_bytes()[0] // 2, 0)


def test_mha_pool_rebuild_keeps_the_quant_dtype():
    pool = _mha_pool(quant=KVQuant(dtype=torch.float8_e4m3fn), pages=64)
    pool.rebuild(32)
    assert pool.dtype is torch.float8_e4m3fn
    assert pool.k_cache(0).shape[-1] == 64


def test_hybrid_swa_pool_quantizes_both_tiers():
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    _init_tp()
    groups = (
        KVCacheGroupSpec(
            name="full", layer_ids=(0, 2), num_kv_heads=2, head_dim=64, sliding_window=None
        ),
        KVCacheGroupSpec(
            name="swa", layer_ids=(1, 3), num_kv_heads=2, head_dim=64, sliding_window=128
        ),
    )

    def build(quant=None):
        return HybridSWAKVCache(
            groups=groups,
            num_layers=4,
            num_full_pages=4,
            page_size=8,
            num_swa_tokens=16,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            quant=quant,
        )

    plain, quant = build(), build(KVQuant(dtype=torch.float8_e4m3fn))
    assert quant.dtype is torch.float8_e4m3fn and quant.quant.k_scale == 1.0
    assert quant.k_cache(0).dtype is torch.float8_e4m3fn  # full tier
    assert quant.k_cache(1).dtype is torch.float8_e4m3fn  # window tier
    plain_kv, plain_swa = plain.unit_bytes()
    quant_kv, quant_swa = quant.unit_bytes()
    assert (quant_kv, quant_swa) == (plain_kv // 2, plain_swa // 2)


def test_check_kv_quant_rejects_pools_without_a_descale_path(monkeypatch):
    import freetoken.kvcache as kvcache_pkg

    class MLAStub:
        pass

    monkeypatch.setattr(kvcache_pkg, "resolve_pool_class", lambda model_config: MLAStub)
    with pytest.raises(RuntimeError, match="MLAStub"):
        check_kv_quant(KVQuant(dtype=torch.float8_e4m3fn), SimpleNamespace())
    # No quantization -> nothing to check, even for a family that could not carry it.
    check_kv_quant(None, SimpleNamespace())


@needs_cuda
def test_pool_store_kv_roundtrips_through_e4m3():
    from freetoken.kvcache.kv_quant import E4M3_MAX

    _init_tp()
    quant = KVQuant(dtype=torch.float8_e4m3fn, k_scale=0.5, v_scale=8.0)
    heads, dim = 2, 64  # _mha_pool's geometry
    pool = _mha_pool(quant=quant, pages=8, page_size=4, device="cuda")
    assert pool.k_cache(0).dtype is torch.float8_e4m3fn
    k = torch.randn(5, heads, dim, device="cuda", dtype=torch.bfloat16) * 20
    v = torch.randn(5, heads, dim, device="cuda", dtype=torch.bfloat16) * 200
    k[0, 0, 0] = 1e4  # far outside the grid: must clamp, never NaN
    out_loc = torch.arange(5, device="cuda", dtype=torch.int32)
    pool.store_kv(k, v, out_loc, 0)

    got_k = pool.k_cache(0).view(-1, heads, dim)[out_loc.long()].float()
    got_v = pool.v_cache(0).view(-1, heads, dim)[out_loc.long()].float()
    assert not bool(got_k.isnan().any())
    expect_k = (k.float() / quant.k_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    expect_v = (v.float() / quant.v_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    assert torch.equal(got_k, expect_k.float())
    assert torch.equal(got_v, expect_v.float())
