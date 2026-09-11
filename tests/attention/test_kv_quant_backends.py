"""Per-backend descale plumbing + config-time gating for --kv-cache-dtype.

The numerics live in tests/kernels/test_kv_quant_attention.py; what these tests pin is
that each backend actually hands the pool's dtype and scale pair to its library, and that
a backend which cannot is refused before weights load instead of misreading the cache."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.kv_quant import KVQuant

QUANT = KVQuant(dtype=torch.float8_e4m3fn, k_scale=0.5, v_scale=4.0, compute_dtype=torch.bfloat16)


class FakeKVCache:
    """Minimal pool stand-in: an fp8 (or bf16) slab, a no-op store, and the quant pair."""

    def __init__(self, quant=None, dtype=torch.float8_e4m3fn, dims=(4, 1, 1, 4)):
        self.device = torch.device("cpu")
        self.dtype = dtype
        self.quant = quant
        self._k = torch.zeros(dims, dtype=dtype)
        self._v = torch.zeros(dims, dtype=dtype)

    def store_kv(self, k, v, out_loc, layer_id):
        pass

    def k_cache(self, layer_id):
        return self._k

    def v_cache(self, layer_id):
        return self._v


def _triton_metadata(is_decode=False, dtype=torch.float32):
    from freetoken.attention.triton import TritonMetadata

    return TritonMetadata(
        cu_seqlens_q_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        indptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        q_to_req=torch.tensor([0, 1], dtype=torch.int32),
        q_positions=torch.tensor([1, 1], dtype=torch.int64),
        is_decode=is_decode,
        prefix_lens=torch.tensor([1, 1], dtype=torch.int32),
        max_q_len=1,
        attn_logits=torch.zeros(2, 1, 2, 4, dtype=torch.float32) if is_decode else None,
        attn_lse=torch.zeros(2, 1, 2, dtype=torch.float32) if is_decode else None,
        num_kv_splits=torch.full((2,), 2, dtype=torch.int32) if is_decode else None,
    )


def _triton_batch(metadata, n_tokens=2):
    return SimpleNamespace(
        attn_metadata=metadata,
        out_loc=torch.arange(n_tokens, dtype=torch.int32),
        padded_size=n_tokens,
        size=n_tokens,
    )


def _triton_backend(monkeypatch, pool):
    from freetoken.attention.triton import TritonAttentionBackend

    monkeypatch.setattr(
        "freetoken.attention.triton.get_global_ctx", lambda: SimpleNamespace(kv_cache=pool)
    )
    return TritonAttentionBackend(SimpleNamespace())


@pytest.mark.parametrize(
    "wrapper, is_decode, dtype",
    [
        ("decode_paged_attention", True, torch.bfloat16),
        ("extend_paged_attention", False, torch.bfloat16),
        ("paged_attention", False, torch.float32),
    ],
)
def test_triton_backend_forwards_the_pool_scales(monkeypatch, wrapper, is_decode, dtype):
    import freetoken.kernel.triton.attention as attn_kernels

    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(attn_kernels, wrapper, fake)
    pool = FakeKVCache(quant=QUANT)
    backend = _triton_backend(monkeypatch, pool)
    q = torch.zeros(2, 1, 4, dtype=dtype)
    backend.forward(q, q.clone(), q.clone(), 0, _triton_batch(_triton_metadata(is_decode)))
    assert captured["k_scale"] == QUANT.k_scale
    assert captured["v_scale"] == QUANT.v_scale


def test_triton_backend_defaults_to_one_for_a_plain_pool(monkeypatch):
    import freetoken.kernel.triton.attention as attn_kernels

    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(attn_kernels, "paged_attention", fake)
    pool = FakeKVCache(quant=None, dtype=torch.bfloat16)
    backend = _triton_backend(monkeypatch, pool)
    q = torch.zeros(2, 1, 4, dtype=torch.float32)
    backend.forward(q, q.clone(), q.clone(), 0, _triton_batch(_triton_metadata()))
    assert captured["k_scale"] == 1.0 and captured["v_scale"] == 1.0


# --- FlashInfer: plan() learns the two dtypes, run() learns the two scales ------------


class _StubWrapper:
    def __init__(self, *args, **kwargs):
        self._int_workspace_buffer = None
        self.plans: list[dict] = []
        self.runs: list[dict] = []

    def plan(self, **kwargs):
        self.plans.append(kwargs)

    def run(self, **kwargs):
        self.runs.append(kwargs)
        return kwargs["q"]


class _StubDecode(_StubWrapper):
    pass


class _StubPrefill(_StubWrapper):
    pass


def _stub_flashinfer(monkeypatch) -> None:
    module = types.ModuleType("flashinfer")
    module.BatchDecodeWithPagedKVCacheWrapper = _StubDecode
    module.BatchPrefillWithPagedKVCacheWrapper = _StubPrefill
    module.CUDAGraphBatchDecodeWithPagedKVCacheWrapper = _StubDecode
    monkeypatch.setitem(sys.modules, "flashinfer", module)


def _fi_backend(monkeypatch, pool):
    from freetoken.attention.fi import FlashInferBackend
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    _stub_flashinfer(monkeypatch)
    monkeypatch.setattr(
        "freetoken.attention.fi.get_global_ctx", lambda: SimpleNamespace(kv_cache=pool)
    )
    return FlashInferBackend(SimpleNamespace(num_qo_heads=4, num_kv_heads=2, head_dim=64))


def _fi_metadata(wrapper, dtype=torch.bfloat16):
    from freetoken.attention.fi import FIMetadata

    return FIMetadata(
        cu_seqlens_q_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        cu_seqlens_q_gpu=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        indices=torch.arange(6, dtype=torch.int32, device="cuda"),
        last_page_len_cpu=torch.ones(2, dtype=torch.int32),
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
        page_size=1,
        pos_encoding_mode="NONE",
        seq_lens_cpu=torch.tensor([3, 3], dtype=torch.int32),
        dtype=dtype,
        wrapper=wrapper,
    )


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@needs_cuda
def test_flashinfer_plans_two_dtypes_and_runs_two_scales(monkeypatch):
    pool = FakeKVCache(quant=QUANT, dtype=torch.float8_e4m3fn, dims=(6, 1, 2, 64))
    backend = _fi_backend(monkeypatch, pool)
    # The pool's dtype is not the query's: plan() has to be told about both.
    assert backend.kv_dtype is torch.float8_e4m3fn
    assert backend.q_dtype is torch.bfloat16
    assert backend.scale_kwargs == {"k_scale": QUANT.k_scale, "v_scale": QUANT.v_scale}

    for wrapper in (_StubDecode(), _StubPrefill()):
        metadata = _fi_metadata(wrapper)
        backend._initialize_metadata_once(metadata)
        plan = wrapper.plans[0]
        assert plan["kv_data_type"] is torch.float8_e4m3fn
        assert plan["q_data_type"] is torch.bfloat16
        assert "k_scale" not in plan  # flashinfer descales on run(), not plan()

        batch = SimpleNamespace(attn_metadata=metadata, out_loc=torch.arange(2, dtype=torch.int32))
        q = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
        backend.forward(q, q.clone(), q.clone(), 0, batch)
        run = wrapper.runs[0]
        assert run["k_scale"] == QUANT.k_scale and run["v_scale"] == QUANT.v_scale


@needs_cuda
def test_flashinfer_keeps_its_plain_call_for_an_unquantized_pool(monkeypatch):
    pool = FakeKVCache(quant=None, dtype=torch.bfloat16, dims=(6, 1, 2, 64))
    backend = _fi_backend(monkeypatch, pool)
    assert backend.kv_dtype is backend.q_dtype is torch.bfloat16
    assert backend.scale_kwargs == {}
    wrapper = _StubPrefill()
    backend._initialize_metadata_once(_fi_metadata(wrapper))
    plan = wrapper.plans[0]
    assert plan["kv_data_type"] is torch.bfloat16 and plan["q_data_type"] is torch.bfloat16


# --- FA3: the query joins the cache in e4m3, descales ride as tensors -----------------


def _stub_sgl_kernel(monkeypatch, captured: dict) -> None:
    def flash_attn_with_kvcache(**kwargs):
        captured.update(kwargs)
        return kwargs["q"]

    submodule = types.ModuleType("sgl_kernel.flash_attn")
    submodule.flash_attn_with_kvcache = flash_attn_with_kvcache
    package = types.ModuleType("sgl_kernel")
    package.flash_attn = submodule
    monkeypatch.setitem(sys.modules, "sgl_kernel", package)
    monkeypatch.setitem(sys.modules, "sgl_kernel.flash_attn", submodule)


def _fa_backend(monkeypatch, pool):
    from freetoken.attention.fa import FAMetadata, FlashAttentionBackend
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr("freetoken.attention.fa.is_arch_supported", lambda *a: False)
    monkeypatch.setattr(
        "freetoken.attention.fa.get_global_ctx",
        lambda: SimpleNamespace(kv_cache=pool, page_size=1),
    )
    backend = FlashAttentionBackend(SimpleNamespace(head_dim=64, num_kv_heads=2))
    backend._metadata = FAMetadata(  # kept on the backend so the test stays short
        cu_seqlens_k=torch.tensor([0, 3, 6], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cache_seqlens=torch.tensor([3, 3], dtype=torch.int32),
        max_seqlen_k=3,
        max_seqlen_q=1,
        page_table=torch.arange(6, dtype=torch.int32).view(2, 3),
    )
    return backend


def _fa_forward(backend, dtype):
    batch = SimpleNamespace(
        attn_metadata=backend._metadata, out_loc=torch.arange(2, dtype=torch.int32)
    )
    q = torch.randn(2, 2, 64, dtype=dtype)
    backend.forward(q, q.clone(), q.clone(), 0, batch)
    return q


def test_fa3_takes_the_query_in_e4m3_with_descale_tensors(monkeypatch):
    captured: dict = {}
    _stub_sgl_kernel(monkeypatch, captured)
    pool = FakeKVCache(quant=QUANT, dtype=torch.float8_e4m3fn)
    _fa_forward(_fa_backend(monkeypatch, pool), torch.bfloat16)
    assert captured["q"].dtype is torch.float8_e4m3fn
    # FA3 wants one fp32 descale per (batch row, kv head), not one per row.
    assert captured["q_descale"].shape == (2, 2)
    # q_descale * k_descale restores the score, v_descale the output (kv_quant.py).
    assert torch.allclose(captured["q_descale"], torch.full((2, 2), QUANT.k_scale))
    assert torch.allclose(captured["k_descale"], torch.full((2, 2), QUANT.k_scale))
    assert torch.allclose(captured["v_descale"], torch.full((2, 2), QUANT.v_scale))


def test_fa3_leaves_the_query_alone_for_an_unquantized_pool(monkeypatch):
    captured: dict = {}
    _stub_sgl_kernel(monkeypatch, captured)
    pool = FakeKVCache(quant=None, dtype=torch.bfloat16)
    _fa_forward(_fa_backend(monkeypatch, pool), torch.bfloat16)
    assert captured["q"].dtype is torch.bfloat16
    assert captured["q_descale"] is captured["k_descale"] is captured["v_descale"] is None


# --- trtllm-gen: the scales fold into the two GEMM scales ----------------------------


@pytest.mark.parametrize(
    "quant, expect",
    [(QUANT, (64**-0.5 * QUANT.k_scale, QUANT.v_scale)), (None, (64**-0.5, 1.0))],
)
def test_trtllm_folds_the_pool_scales_into_bmm_scales(monkeypatch, quant, expect):
    from freetoken.attention.trtllm import TensorRTLLMBackend

    pool = FakeKVCache(
        quant=quant, dtype=torch.float8_e4m3fn if quant else torch.bfloat16
    )
    monkeypatch.setattr(
        "freetoken.attention.trtllm.get_global_ctx",
        lambda: SimpleNamespace(kv_cache=pool, page_size=1),
    )
    backend = TensorRTLLMBackend(SimpleNamespace(head_dim=64))
    assert backend.bmm1_scale == pytest.approx(expect[0])
    assert backend.bmm2_scale == pytest.approx(expect[1])


@needs_cuda
def test_a_real_fp8_pool_reads_back_through_the_backend(monkeypatch):
    """End of the pipe: a real quantized MHAKVCache, stored through its own kernel and
    read by the real Triton backend, must reproduce the same output as a plain pool that
    holds exactly what the fp8 buffer recorded. Anything lost between the two is the
    descale plumbing, not the quantization error."""
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(0)
    lens, heads, dim = [8, 4], 2, 64
    total, qo_heads = sum(lens), 4
    device = torch.device("cuda")

    def pool(quant):
        return MHAKVCache(
            num_kv_heads=heads,
            num_layers=2,
            head_dim=dim,
            num_pages=4,
            page_size=4,
            dtype=torch.bfloat16,
            device=device,
            quant=quant,
        )

    fp8_pool, ref_pool = pool(QUANT), pool(None)
    row = heads * dim
    k = torch.randn(total, row, device=device, dtype=torch.bfloat16) * 3
    v = torch.randn(total, row, device=device, dtype=torch.bfloat16) * 2
    slots = torch.arange(total, dtype=torch.int32, device=device)
    fp8_pool.store_kv(k, v, slots, 0)
    stored_k = (fp8_pool.k_cache(0).view(-1, row)[:total].float() * QUANT.k_scale).to(
        torch.bfloat16
    )
    stored_v = (fp8_pool.v_cache(0).view(-1, row)[:total].float() * QUANT.v_scale).to(
        torch.bfloat16
    )
    ref_pool.store_kv(stored_k, stored_v, slots, 0)
    assert not torch.equal(stored_k, k), "the fp8 store should not be exact at scale 0.5"

    q = torch.randn(len(lens), qo_heads, dim, device=device, dtype=torch.bfloat16)
    # The backend always stores what it is given; park one row in a slot the attention
    # below never indexes so both pools get the same (ignored) write.
    spare = torch.full((1,), total, dtype=torch.int32, device=device)
    dummy = torch.zeros(1, row, device=device, dtype=torch.bfloat16)
    outs = []
    for candidate in (fp8_pool, ref_pool):
        monkeypatch.setattr(
            "freetoken.attention.triton.get_global_ctx",
            lambda p=candidate: SimpleNamespace(kv_cache=p),
        )
        backend = TritonAttentionBackend(
            SimpleNamespace(num_qo_heads=qo_heads, num_kv_heads=heads, head_dim=dim)
        )
        metadata = TritonMetadata(
            cu_seqlens_q_gpu=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
            indptr=torch.tensor([0, 8, 12], dtype=torch.int32, device=device),
            indices=slots,
            q_to_req=torch.tensor([0, 1], dtype=torch.int32, device=device),
            q_positions=torch.tensor([7, 3], dtype=torch.int32, device=device),
            is_decode=True,
            prefix_lens=torch.tensor([7, 3], dtype=torch.int32, device=device),
            max_q_len=1,
        )
        batch = SimpleNamespace(
            attn_metadata=metadata,
            out_loc=spare,
            padded_size=len(lens),
            size=len(lens),
        )
        outs.append(backend.forward(q, dummy, dummy, 0, batch))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    # A couple of bf16 ulps: the two pools hold the same numbers, so anything larger would
    # be a descale that never reached the kernel (e4m3 alone costs ~6% here).
    assert diff < 0.01 * float(outs[1].float().abs().max()), (
        f"fp8 pool read diverged from its dequantized twin: {diff}"
    )
