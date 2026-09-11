"""Quantized KV storage: the dtype/scale resolution, the byte cost it implies, and the
pool families that may carry it. GPU round-trips skip themselves without CUDA."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
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
