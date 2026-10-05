from contextlib import contextmanager

import os

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16_offload_layer(layer_id: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int):
    """A bf16 offload layer on the fused kernel."""
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig

    return OffloadMoELayer(
        layer_id, num_experts, top_k, hidden_size, intermediate_size,
        quant_config=NoQuantConfig(), prefix=f"model.layers.{layer_id}.mlp.experts",
    )


def _make_layer_and_cache():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    layer = _bf16_offload_layer(0, 4, 2, 8, 16)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    layer.offload_cache = cache
    return layer, cache


def test_dummy_expert_banks_follow_the_kernel_layout(monkeypatch):

    from freetoken.kernel import backend
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
    from freetoken.moe.expert_banks import build_expert_banks

    _init_tp()
    L, E, H, I = 3, 4, 64, 32
    monkeypatch.setattr(backend, "device_capability", lambda: (0, 0))
    monkeypatch.setattr(backend, "is_vllm_installed", lambda: False)
    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: False)

    def _bound(quant):
        layer = _bf16_offload_layer(0, E, 2, H, I) if quant is None else None
        if layer is None:
            from freetoken.layers.moe import OffloadMoELayer

            layer = OffloadMoELayer(0, E, 2, H, I, quant_config=quant, prefix="model.layers.0.mlp.experts")
        return layer

    bf16 = _bound(None)
    banks = build_expert_banks(bf16.quant_method, L, None, device=torch.device("cpu"), dummy=True)
    assert banks.kind is QuantKind.NONE and set(banks.sources) == {"gate_up", "down"}
    assert len(banks.sources["gate_up"]) == L and all(t.shape == (E, 2 * I, H) for t in banks.sources["gate_up"])
    assert all(t.shape == (E, H, I) for t in banks.sources["down"])

    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
    nvfp4 = _bound(quant)
    banks = build_expert_banks(nvfp4.quant_method, L, None, device=torch.device("cpu"), dummy=True)
    assert banks.kind is QuantKind.NVFP4 and banks.kernel == "triton"
    assert {len(layers) for layers in banks.sources.values()} == {L}
    assert {t.shape[0] for layers in banks.sources.values() for t in layers} == {E}
    assert torch.all(banks.sources["gate_up_scale"][0].float() == 1.0)
    assert torch.all(banks.sources["gate_up_global"][0].float() > 0)


def test_offload_moe_layer_prefill_forward_uses_single_layer_cache_view(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(cache, "materialize_layer", lambda layer_id: calls.setdefault("layer_id", layer_id))
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", fake_fused)

    out = layer.prefill_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["layer_id"] == 0
    assert calls["copied"] is True
    assert calls["w1"].shape[0] == layer.num_experts
    assert calls["w2"].shape[0] == layer.num_experts
    assert calls["w1"].data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert calls["w2"].data_ptr() == cache.bank_caches["down"].data_ptr()
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    # slot == expert id after materialize, so the routing ids pass through unmapped
    assert calls["topk_ids"].tolist() == [[2, 1]]


def test_offload_moe_layer_prefill_overlap_prefetches_layers_into_two_buffers(monkeypatch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    num_layers = 3
    num_experts = 4
    layers = [_bf16_offload_layer(layer_id, num_experts, 2, 8, 16) for layer_id in range(num_layers)]
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})
    for layer in layers:
        layer.offload_cache = cache

    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, num_experts)
    fused_calls = []

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (
            topk_weights,
            topk_ids.clone(),
        ),
    )

    def unexpected_fast_index_copy(*args, **kwargs):
        raise AssertionError("prefill overlap should use direct async copy")

    monkeypatch.setattr("freetoken.kernel.fast_index_copy_jit", unexpected_fast_index_copy)

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        layer_id = len(fused_calls)
        fused_calls.append(
            {
                "w1_ptr": w1.data_ptr(),
                "w2_ptr": w2.data_ptr(),
                "w1": w1.clone(),
                "w2": w2.clone(),
                "topk_weights": got_topk_weights,
                "topk_ids": got_topk_ids.clone(),
            }
        )
        return hidden_states + layer_id

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", fake_fused)

    out = hidden_states
    for layer in layers:
        out = layer.prefill_forward(out, router_logits)

    assert torch.allclose(out, hidden_states + 3)
    for layer_id in range(num_layers):
        assert fused_calls[layer_id]["topk_weights"] is topk_weights
        assert fused_calls[layer_id]["topk_ids"].tolist() == [[2, 1]]
        assert torch.equal(fused_calls[layer_id]["w1"], gate_up_source[layer_id])
        assert torch.equal(fused_calls[layer_id]["w2"], down_source[layer_id])

    assert fused_calls[0]["w1_ptr"] == fused_calls[2]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] == fused_calls[2]["w2_ptr"]
    assert fused_calls[0]["w1_ptr"] != fused_calls[1]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] != fused_calls[1]["w2_ptr"]
    prefill_gate_up_buffer, prefill_down_buffer = cache.prefill_bank_buffers
    assert prefill_gate_up_buffer.data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert prefill_down_buffer.data_ptr() == cache.bank_caches["down"].data_ptr()


def test_offload_moe_cache_prefill_overlap_requires_two_layer_slots():
    from freetoken.moe.offload_cache import OffloadMoeCache

    with pytest.raises(AssertionError):
        OffloadMoeCache(
            num_layers=3,
            num_experts=4,
            cache_size=7,
            device=torch.device("cpu"),
            prefill_overlap=True,
        )


def test_offload_moe_cache_marlin_rejects_slot_count_beyond_kernel_limit():
    from freetoken.moe.offload_cache import OffloadMoeCache

    with pytest.raises(ValueError, match="992"):
        OffloadMoeCache(
            num_layers=2,
            num_experts=8,
            cache_size=1024,
            device=torch.device("cpu"),
            quant_format="nvfp4_marlin",
        )


def test_prefill_overlap_prefetch_invalidates_borrowed_unified_cache_slots():
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_layers = 3
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    old_layers = torch.tensor([2, 2, 1, 1], dtype=torch.int32)
    old_experts = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cache.id_of_slot[:num_experts] = old_layers * num_experts + old_experts
    cache.usage[:num_experts] = torch.arange(1, num_experts + 1, dtype=torch.int64)
    for slot, (layer_id, expert_id) in enumerate(zip(old_layers.tolist(), old_experts.tolist())):
        cache.slot_for_id[layer_id, expert_id] = slot

    cache.prefetch_prefill_layer(0)

    assert cache.id_of_slot[:num_experts].tolist() == [-1] * num_experts
    # the borrowed slots are stamped step+1 (not zero) so decode's LRU victim pick
    # prefers regular slots -- see test_interleaved_decode_keeps_residency below
    assert cache.usage[:num_experts].tolist() == [int(cache.step) + 1] * num_experts
    for layer_id, expert_id in zip(old_layers.tolist(), old_experts.tolist()):
        assert int(cache.slot_for_id[layer_id, expert_id].item()) == -1
    assert torch.equal(cache.bank_caches["gate_up"][:num_experts], gate_up_source[0])
    assert torch.equal(cache.bank_caches["down"][:num_experts], down_source[0])


def test_interleaved_decode_keeps_residency_across_chunk_invalidation():
    # Mixed-workload regression: a prefill chunk plan invalidates the borrowed buffer
    # slots [0, 2E). When invalidation zeroed their usage, every decode fetch picked a
    # buffer slot as its LRU victim (usage 0 = oldest), so the next chunk wiped the row
    # and decode hit rate stayed at ~0 -- each interleaved decode step re-fetched in
    # full over PCIe. With the step+1 stamp, decode lands in regular slots that chunk
    # plans never touch, and the same expert hits on the next interleaved step.
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.offload_kernels import ensure_experts_hybrid

    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=16,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )

    def chunk_pass():
        # one prefill chunk stages every layer through both buffers by parity
        cache._invalidate_prefill_buffer(0)
        cache._invalidate_prefill_buffer(1)

    def decode_step(expert: int) -> int:
        ids = torch.tensor([[expert]], dtype=torch.int32)
        ensure_experts_hybrid(cache, 0, ids, 4, 0.0)  # CPU reference path
        return int(cache.num_indices.item())

    chunk_pass()
    misses = decode_step(0)
    slot = int(cache.slot_for_id[0, 0].item())
    assert misses == 1
    assert slot >= 2 * cache.num_experts, "decode fetch landed in a borrowed buffer slot"

    chunk_pass()  # an interleaved chunk wipes both buffers
    misses = decode_step(0)
    assert misses == 0, "decode lost its row to the chunk invalidation"
    assert int(cache.slot_for_id[0, 0].item()) == slot


def test_prefill_overlap_waits_for_previous_prefill_release_after_begin(monkeypatch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_layers = 2
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.zeros(num_layers * num_experts, 32, 8).split(num_experts))
    down_source = list(torch.zeros(num_layers * num_experts, 8, 16).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    class FakeStream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event.name)

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream=None):
            pass

    @contextmanager
    def fake_cuda_stream(stream):
        yield

    copy_stream = FakeStream()
    cache.prefill_copy_stream = copy_stream
    cache.prefill_begin_event = FakeEvent("begin")
    cache.prefill_ready_events = [FakeEvent("ready0"), FakeEvent("ready1")]
    cache.prefill_release_events = [FakeEvent("release0"), FakeEvent("release1")]
    monkeypatch.setattr("torch.cuda.stream", fake_cuda_stream)
    monkeypatch.setattr("torch.cuda.current_stream", lambda device=None: object())

    cache.prefetch_prefill_layer(0)
    cache.release_prefill_layer(0)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)

    # begin_prefill fences the copy stream behind the compute stream (so a prefetch
    # cannot race the preceding decode batch), then the buffer reuse waits on the
    # previous prefill's release event.
    assert copy_stream.waited == ["begin", "release0"]


def test_offload_moe_layer_decode_forward_uses_remapped_slot_ids(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )

    def fake_ensure(layer_id, expert_ids):
        calls["ensure_layer_id"] = layer_id
        calls["ensure_expert_ids"] = expert_ids.clone()
        expert_ids.copy_(torch.tensor([[5, 0]], dtype=torch.int32))

    monkeypatch.setattr(cache, "ensure_experts", fake_ensure)
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused_decode(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_decode_impl", fake_fused_decode)

    out = layer.decode_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["ensure_layer_id"] == 0
    assert calls["ensure_expert_ids"].tolist() == [[2, 1]]
    assert calls["copied"] is True
    assert calls["w1"] is cache.bank_caches["gate_up"]
    assert calls["w2"] is cache.bank_caches["down"]
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    assert calls["topk_ids"].tolist() == [[5, 0]]



def test_lru_gpu_cache_assigns_unique_slots_for_large_miss_batch():
    import pytest
    from freetoken.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    cache = OffloadMoeCache(
        num_layers=40,
        num_experts=256,
        cache_size=1664,
        device=torch.device("cuda"),
    )
    expert_ids = torch.arange(256, dtype=torch.int32, device="cuda").view(32, 8)

    cache.ensure_experts(0, expert_ids)
    torch.cuda.synchronize()

    assert int(cache.num_indices.item()) == 256
    assert expert_ids.min().item() >= 0
    assert expert_ids.max().item() < cache.cache_size
    evict_slots = cache.evict_slots[:256]
    assert evict_slots.min().item() >= 0
    assert evict_slots.max().item() < cache.cache_size
    assert torch.unique(evict_slots).numel() == evict_slots.numel()
    assert cache.src_indices[:256].tolist() == list(range(256))


def test_adjust_config_converts_moe_cache_rate_to_cache_size(monkeypatch):
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    import freetoken.engine.engine as engine_module
    from freetoken.engine.engine import _adjust_config

    # This test exercises the discrete-GPU offload path regardless of the host
    # running the suite (GB10 reports cudaDevAttrIntegrated=1).
    monkeypatch.setattr(engine_module, "_is_unified_memory_gpu", lambda index=None: False)

    # triton: the flashinfer probe must not refuse the config ahead of the sizing gate.
    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        attention_backend="triton",
        moe_cache_rate=0.3,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="none",
            moe_strategy="auto",
        ),
    )

    _adjust_config(config)

    from freetoken.moe import is_offload_moe_strategy

    assert config.moe_cache_size == 24
    # Family, not member: a box with a benchbw profile resolves bf16 experts to hybrid.
    assert is_offload_moe_strategy(config.moe_strategy)


def test_graph_capture_reuses_warm_offload_cache_before_capture(monkeypatch):
    import freetoken.core as core
    from freetoken.core import Context, Req, get_global_ctx
    from freetoken.engine.graph import GraphRunner

    events = []
    _init_tp()
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))

    class FakeGraph:
        def pool(self):
            return "pool"

    @contextmanager
    def fake_cuda_graph(graph, pool=None, stream=None):
        events.append("graph_enter")
        yield
        events.append("graph_exit")

    class FakeAttnBackend:
        def init_capture_graph(self, max_seq_len, bs_list):
            pass

        def prepare_for_capture(self, batch):
            pass

    class FakeModel:
        def forward(self):
            events.append("forward")
            batch = get_global_ctx().batch
            return torch.zeros(batch.size, 3)

    class FakeOffloadCache:
        def reset(self):
            events.append("reset")

    monkeypatch.setattr("torch.cuda.CUDAGraph", FakeGraph)
    monkeypatch.setattr("torch.cuda.graph", fake_cuda_graph)
    monkeypatch.setattr("torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 1024)

    dummy_req = Req(
        input_ids=torch.tensor([0], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=None,
        cache_handle=None,
    )
    GraphRunner(
        stream=None,
        device=torch.device("cpu"),
        model=FakeModel(),
        attn_backend=FakeAttnBackend(),
        cuda_graph_bs=[1],
        cuda_graph_max_bs=None,
        free_memory=1024,
        max_seq_len=1,
        vocab_size=3,
        dummy_req=dummy_req,
        moe_offload_cache=FakeOffloadCache(),
    )

    assert events == [
        "reset",
        "forward",
        "graph_enter",
        "forward",
        "graph_exit",
        "reset",
        "reset",
    ]


def test_nvfp4_materialize_keeps_bookkeeping_consistent_across_requests():
    """Regression: a full-layer prefill loads the layer's experts into slots [0, E).
    If that overwrite does not invalidate the previous owners' mappings, a later
    decode "hits" a stale slot_for_id entry and silently reads another expert's
    weights. materialize_layer must keep bookkeeping == slot contents."""
    import pytest
    from freetoken.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    L, E, S = 2, 8, 8
    OUT, IN = 64, 512  # keep rows >= 128B so the fast_index_copy JIT has a kernel
    dev = torch.device("cuda")

    def bank(out, inner, dtype):
        # one independently allocated [E, out, inner] tensor per layer (the per-layer host
        # bank contract); row idx within layer l keeps the old flat fingerprint l*E+idx.
        layers = []
        for l in range(L):
            t = torch.zeros(E, out, inner, dtype=dtype)
            for e in range(E):
                t[e].view(torch.uint8).fill_(l * E + e)
            layers.append(t)
        return layers

    def pinned(layers):
        return [t.pin_memory() for t in layers]

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=dev, quant_format="nvfp4"
    )
    cache.set_bank_sources(
        {
            "gate_up_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "gate_up_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "gate_up_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
            "down_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "down_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "down_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
        }
    )
    cache.reset()

    def fingerprint(slot):  # which source row's bytes live in this slot?
        return int(cache.bank_caches["gate_up_packed"][slot].view(torch.uint8).flatten()[0].item())

    # Request A, decode: layer 0 loads experts 3 and 5 somewhere in the cache.
    ids = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids.tolist()] == [3, 5]

    # Request B, prefill: layer 1 is materialized into slots [0, E), overwriting
    # every slot (S == E), including the ones decode A used.
    cache.materialize_layer(1)
    cache.copy_missing()
    torch.cuda.synchronize()
    # The layer's experts fill slots [0, E) bijectively and the bookkeeping agrees.
    assert [fingerprint(s) for s in range(E)] == [E + e for e in range(E)]
    assert cache.slot_for_id[1].tolist() == list(range(E))

    # Request B, decode: layer 0 routes to experts 3/5 again. Their old slots were
    # overwritten, so this must be a miss + reload -- never a stale hit serving
    # layer-1 bytes.
    ids2 = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids2)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids2.tolist()] == [3, 5]

    # The prefilled layer's own experts still resolve to correct bytes (S == E, so
    # the layer-0 reload above evicted two layer-1 slots -- hit or miss, the
    # bookkeeping must never serve another expert's bytes).
    ids3 = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    cache.ensure_experts(1, ids3)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids3.tolist()] == [E + 1, E + 2]


def test_offload_cache_rebuild_resizes_and_preserves_sources():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    gate_up = torch.randn(4, 32, 8)
    down = torch.randn(4, 8, 16)
    cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})

    cache.rebuild(10)

    assert cache.cache_size == 10
    # host sources preserved (same objects, not reloaded)
    assert cache.bank_sources["gate_up"][0] is gate_up
    assert cache.bank_sources["down"][0] is down
    # GPU slot caches resized to the new cache_size, row shape unchanged
    assert cache.bank_caches["gate_up"].shape == (10, 32, 8)
    assert cache.bank_caches["down"].shape == (10, 8, 16)
    # bookkeeping resized + reset
    assert cache.id_of_slot.shape == (10,)
    assert cache.usage.shape == (10,)
    assert torch.all(cache.slot_for_id == -1)
    assert torch.all(cache.id_of_slot == -1)


def test_offload_cache_rebuild_disables_prefill_overlap_when_too_small():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    assert cache.prefill_overlap is True

    cache.rebuild(5)  # 5 < 2*num_experts (8) -> overlap must auto-disable

    assert cache.cache_size == 5
    assert cache.prefill_overlap is False
    assert cache.prefill_bank_buffers == []


def test_offload_cache_rebuild_keeps_overlap_at_boundary():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    cache.rebuild(8)  # exactly 2*num_experts -> overlap stays on
    assert cache.prefill_overlap is True
    assert cache.cache_size == 8


def test_offload_cache_validate_rebuild_enforces_marlin_cap_and_floor():
    # The constructor caps nvfp4_marlin slots at 992; a runtime rebuild must enforce the
    # same upper cap (and the num_experts floor), else marlin decode kernels later break.
    from freetoken.moe.offload_cache import MARLIN_MAX_CACHE_SIZE, OffloadMoeCache

    _init_tp()
    marlin = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=16,
        device=torch.device("cpu"), quant_format="nvfp4_marlin",
    )
    with pytest.raises(ValueError, match="992"):
        marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE + 1)
    marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE)  # exactly at the cap: allowed

    bf16 = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="num_experts"):
        bf16.validate_rebuild(3)  # below the num_experts floor


def _make_split_cache(num_layers=2, locked=(1,), prefill_overlap=False, device="cpu"):
    """A [gate_up, down] bf16 cache with the given layers LOCKED (rest pinned)."""
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    dev = torch.device(device)
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=4, cache_size=8,
        device=dev, prefill_overlap=prefill_overlap,
    )
    cache.cpu_layer_ids = frozenset(locked)
    src_dev = dev if dev.type == "cuda" else torch.device("cpu")
    sources = {
        # CUDA-resident pinned-layer sources keep _build_copy_plan's device_ptr happy in the CUDA variant; locked layers stay host tensors (never translated)
        "gate_up": [
            torch.randn(4, 32, 8, device=torch.device("cpu") if i in locked else src_dev)
            for i in range(num_layers)
        ],
        "down": [
            torch.randn(4, 8, 16, device=torch.device("cpu") if i in locked else src_dev)
            for i in range(num_layers)
        ],
    }
    residency = [
        HostResidency.LOCKED.value if i in locked else HostResidency.PINNED.value
        for i in range(num_layers)
    ]
    cache.set_bank_sources(sources, layer_residency=residency)
    return cache, sources


def test_set_bank_sources_locked_layer_requires_cpu_layer_ids():
    # a layer without a device address can only decode on the CPU executor; labeling it LOCKED outside cpu_layer_ids is a wiring bug and must fail loudly
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
    )
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="cpu_layer_ids"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.LOCKED.value],
        )


def test_set_bank_sources_locked_layer_rejects_prefill_overlap():
    # prefill overlap DMAs from registered banks; a LOCKED layer cannot feed it
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.cpu_layer_ids = frozenset({1})
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="[Pp]refill overlap"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.LOCKED.value],
        )


def test_locked_layer_prefill_materialize_copies_whole_layer_pageable():
    # the only movement a LOCKED layer needs: copy_missing's pageable branch copies the whole layer into slots [0, E) with position == expert id
    # stage the state materialize_layer would (its kernel is CUDA-only; the fixture cache lives on the CPU)
    cache, sources = _make_split_cache(num_layers=2, locked=(1,))

    cache._pending_src_layer = 1
    cache._pending_whole_layer = True
    cache.copy_missing()

    gate_up_cache, down_cache = (c for _, c in cache.banks)
    assert torch.equal(gate_up_cache[:4], sources["gate_up"][1])
    assert torch.equal(down_cache[:4], sources["down"][1])
    # (The pinned layers' staged JIT path is covered by the mocked tests above.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_copy_plan_skips_locked_layers_and_keeps_fused_path():
    # _build_copy_plan must not resolve a device alias for a LOCKED layer; its descriptor row stays a 0 placeholder while the pinned layers keep the fused path
    cache, _ = _make_split_cache(num_layers=2, locked=(1,), device="cuda")

    assert cache._copy_fused_ok
    assert (cache._copy_src_ptrs[1] == 0).all(), "locked layer row must stay 0"
    assert (cache._copy_src_ptrs[0] != 0).all(), "pinned layer rows must resolve"


def test_locked_layer_copy_missing_rejects_ensure_experts_staging():
    # the pageable branch presumes materialize_layer's position == expert id; staging via ensure_experts (LRU slot remap) on a locked layer must fail loudly, not gather other experts' weights
    # stage the state ensure_experts would (its kernel is CUDA-only; the fixture cache lives on the CPU)
    cache, _ = _make_split_cache(num_layers=2, locked=(1,))

    cache._pending_src_layer = 1
    cache._pending_whole_layer = False
    with pytest.raises(RuntimeError, match="unpinned"):
        cache.copy_missing()


def test_requested_residency_routes_layer_settles(monkeypatch):
    # the ambient plan installed by load_expert_banks must route each layer's banks by label at both slow-path settle points (PinPipeline layer sink, list-valued pin_banks) and record that it was consulted
    # without a plan everything pins
    import freetoken.moe.host_banks as hb

    settled = []
    monkeypatch.setattr(hb.HostBank, "pin", lambda self: settled.append("pin"))
    monkeypatch.setattr(hb.HostBank, "lock", lambda self: settled.append("lock"))
    banks = {
        "gate_up": [hb.HostBank((4,), torch.uint8) for _ in range(3)],
        "down": [hb.HostBank((4,), torch.uint8) for _ in range(3)],
    }
    labels = [
        hb.HostResidency.PINNED.value,
        hb.HostResidency.LOCKED.value,
        hb.HostResidency.PAGEABLE.value,
    ]

    with hb.requested_residency(labels) as plan:
        with hb.PinPipeline() as pins:
            for layer_id in range(3):
                pins(layer_id, {name: per[layer_id] for name, per in banks.items()})
    # the single drain thread settles FIFO: layer 0 pins, layer 1 locks, layer 2 passes
    assert settled == ["pin", "pin", "lock", "lock"]
    assert plan.applied

    settled.clear()
    with hb.requested_residency(labels) as plan:
        hb.pin_banks(banks)
    assert settled == ["pin", "lock", "pin", "lock"]  # per name: layer 0 pin, 1 lock, 2 skip
    assert plan.applied

    settled.clear()
    hb.pin_banks(banks)  # no ambient plan -> every layer pins
    assert settled == ["pin"] * 6


def test_echo_residency_stamps_honored_requests_only():
    # load_expert_banks stamps the request onto the provider's ExpertBanks only when a settle point consulted the plan; an unconsulted plan keeps None (the engine's degrade signal)
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency
    from freetoken.moe.host_banks import HostResidency, _ResidencyPlan

    labels = [HostResidency.PINNED.value, HostResidency.LOCKED.value]
    banks = ExpertBanks("bf16", {"gate_up": [], "down": []})

    plan = _ResidencyPlan(labels)
    plan.residency_for(1)  # a settle point consulted the plan
    assert _echo_residency(banks, labels, plan).layer_residency == labels

    stale = _ResidencyPlan(labels)  # never consulted -> keep None + warn
    assert _echo_residency(banks, labels, stale).layer_residency is None
    assert _echo_residency(banks, None, None) is banks


def test_lock_failure_downgrades_echoed_residency(monkeypatch):
    # a failed mlock leaves the bank pageable; the plan and the echoed labels must report that instead of the requested LOCKED
    import freetoken.moe.host_banks as hb
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency

    def boom(addr, nbytes):
        raise OSError(12, "mlock denied")

    monkeypatch.setattr(hb, "_os_lock", boom)
    monkeypatch.setattr(hb, "_os_lock_failed", False)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # keep the pinned layer off CUDA
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.LOCKED.value]

    banks = {"gate_up": [hb.HostBank((4,), torch.uint8) for _ in range(2)]}
    with hb.requested_residency(labels) as plan:
        hb.pin_banks(banks)
    assert plan.actual == {1: hb.HostResidency.PAGEABLE.value}
    echoed = _echo_residency(ExpertBanks("bf16", {}), labels, plan)
    assert echoed.layer_residency == [
        hb.HostResidency.PINNED.value, hb.HostResidency.PAGEABLE.value,
    ]

    monkeypatch.setattr(hb, "_os_lock_failed", False)
    with hb.requested_residency(labels) as plan2:
        with hb.PinPipeline() as pins:
            pins(1, {"gate_up": hb.HostBank((4,), torch.uint8)})
    assert plan2.actual == {1: hb.HostResidency.PAGEABLE.value}


@pytest.mark.skipif(os.name != "nt", reason="VirtualLock is the Windows branch of _os_lock")
def test_windows_lock_reports_a_quota_not_an_import_failure(monkeypatch):
    """The bug was silent: ``_os_lock`` raised ImportError from ``import resource``, ``lock()``
    swallowed it, and every LOCKED layer settled PAGEABLE under a Linux quota message.

    How much a host may page-lock is policy (the 'Lock pages in memory' right), so either answer
    from the OS is fine here -- neither of them is an ImportError, which is why ``_os_lock`` is
    called directly rather than through the swallowing ``lock()``.
    """
    import freetoken.moe.host_banks as hb

    monkeypatch.setattr(hb, "_os_locked_total", 0)
    bank = hb.HostBank((2 << 20,), torch.uint8)
    bank.tensor.fill_(0)  # lock after fill, the order the loader keeps
    try:
        hb._os_lock(bank.addr, bank.nbytes)
    except OSError as exc:
        assert "Lock pages in memory" in str(exc)
    else:
        assert hb._os_locked_total == bank.nbytes


@pytest.mark.skipif(os.name != "nt", reason="the quota is the Windows page-lock ceiling")
def test_windows_quota_request_covers_the_live_working_set():
    """VirtualLock charges its quota against the whole working set, not the bytes locked so far.

    The old sizing (locked bytes + one bank) let a server that is already tens of GiB resident
    ask for a maximum barely above a bank, which is the refusal that read as a missing privilege.
    """
    from freetoken.moe.host_banks import _nt_quota_request

    GiB = 1 << 30
    want = _nt_quota_request(nbytes=1 * GiB, locked_total=2 * GiB, working_set=65 * GiB)
    assert want >= 68 * GiB, "a bank-sized raise can never cover a resident model"
    assert _nt_quota_request(1 * GiB, 0, 0) > 1 * GiB, "the raise needs headroom, not the exact bytes"


@pytest.mark.skipif(os.name != "nt", reason="the quota is the Windows page-lock ceiling")
def test_windows_quota_request_is_sized_on_the_planned_footprint():
    """The working set at the first lock is a snapshot of a load still in progress, not the footprint.

    Seen on a 63 GiB boot: the first bank settled at 2% read, so the raise asked 8.0 GiB -- enough for
    that one bank, and the next would be refused once the experts were in. What the run says it will
    hold therefore floors the ask."""
    from freetoken.moe.host_banks import _nt_quota_request

    GiB = 1 << 30
    early = 6 * GiB  # resident when the first bank settles, against a 64 GiB bank set
    want = _nt_quota_request(nbytes=1 * GiB, locked_total=0, working_set=early, planned=64 * GiB)
    assert want > 60 * GiB, "sized on the snapshot, the next bank is refused once the load finishes"
    assert _nt_quota_request(1 * GiB, 0, 65 * GiB, 64 * GiB) == _nt_quota_request(1 * GiB, 0, 65 * GiB)


@pytest.mark.skipif(os.name != "nt", reason="the quota is the Windows page-lock ceiling")
def test_windows_refusal_names_the_granted_ceiling(monkeypatch):
    """A refusal must say how much was granted against how much the resident banks need."""
    import freetoken.moe.host_banks as hb

    GiB = 1 << 30
    monkeypatch.setattr(hb, "_nt_quota_job_capped", True)
    msg = hb._nt_lock_refusal(1453, nbytes=1 * GiB, ceiling=1 * GiB, want=80 * GiB)
    assert "WinError 1453" in msg
    assert "1.0 GiB of the 80.0 GiB" in msg
    assert "imposed by the job" in msg
    assert "Lock pages in memory" in msg


@pytest.mark.skipif(os.name != "nt", reason="the quota is the Windows page-lock ceiling")
def test_windows_refusal_names_the_ceiling_after_a_successful_raise(monkeypatch):
    """A refusal that only says 'missing privilege' leaves the reader guessing whether the raise ran.

    Reporting the granted maximum even when it is the request or larger is what separates a job
    object that will not stretch from a box whose only problem is the right."""
    import freetoken.moe.host_banks as hb

    GiB = 1 << 30
    monkeypatch.setattr(hb, "_nt_quota_job_capped", True)  # imposed, yet no longer the binding limit
    msg = hb._nt_lock_refusal(1453, nbytes=1 * GiB, ceiling=72 * GiB, want=72 * GiB)
    assert "raised to 72.0 GiB" in msg
    assert "imposed by the job" not in msg


@pytest.mark.skipif(os.name != "nt", reason="the quota is the Windows page-lock ceiling")
def test_windows_quota_raise_reports_what_the_os_granted(monkeypatch):
    """A job object can hold the maximum below the request, so the grant is what must be reported."""
    import freetoken.moe.host_banks as hb

    GiB = 1 << 30
    granted = [2 * GiB]  # the job caps us far below the 68 GiB ask

    def limits():
        return 0, granted[0], 0x2 | 0x8  # maximum valid, and held down by a job object

    def set_max(minimum, maximum):
        granted[0] = min(maximum, 4 * GiB)  # the job refuses the rest
        return True

    monkeypatch.setattr(hb, "_nt_working_set_limits", limits)
    monkeypatch.setattr(hb, "_nt_set_working_set_max", set_max)
    monkeypatch.setattr(hb, "_nt_quota_ceiling", 0)
    assert hb._nt_raise_working_set_quota(68 * GiB) == 4 * GiB
    assert hb._nt_quota_job_capped is True


@pytest.mark.skipif(os.name != "nt", reason="the counters come from psapi")
def test_working_set_query_answers_on_windows():
    """A wrong PROCESS_MEMORY_COUNTERS layout is refused with ERROR_INSUFFICIENT_BUFFER, and a
    None here quietly downgrades the whole quota sizing to 'assume nothing is resident'."""
    import freetoken.moe.host_banks as hb

    assert hb._working_set_bytes() > 0


# ---- --expert-load auto: the low-RAM veto is sized by the experts, not the whole checkpoint ----


def _write_fake_shard(path, tensors: dict[str, int]) -> int:
    """A safetensors-shaped file (u64 header length + JSON header + zero-filled region)."""
    import json
    import struct

    header, end = {}, 0
    for name, nbytes in tensors.items():
        header[name] = {"dtype": "U8", "shape": [nbytes], "data_offsets": [end, end + nbytes]}
        end += nbytes
    blob = json.dumps(header).encode() + b" " * 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * end)
    return 8 + len(blob) + end


def _fake_checkpoint(tmp_path, *, expert_tensor_bytes: int, experts: int, side_bytes: int) -> dict[str, int]:
    """Many SMALL expert tensors, a dense shard, and a far bigger side file listed in the index --
    the Qwen3.8-Flash-Next shape, whose PLE/MTP file the expert reader never opens."""
    import json

    projs = ("gate", "up", "down")
    expert_tensors = {
        f"model.language_model.layers.{i // 3}.mlp.experts.{i}.{projs[i % 3]}_proj.weight": expert_tensor_bytes
        for i in range(experts)
    }
    side_experts = 8
    shards = {
        "model-00001-of-00003.safetensors": expert_tensors,
        "model-00002-of-00003.safetensors": {f"model.language_model.layers.{i}.mlp.gate_proj.weight": 2 << 20 for i in range(2)},
        # Qwen3.8-Flash-Next shape: a huge side file whose expert tensors belong to the speculative
        # head the loader drops, so the reader never opens it.
        "model-00003-of-00003.safetensors": {
            f"mtp.layers.0.mlp.experts.{i}.gate_proj.weight_scale_inv": side_bytes // side_experts
            for i in range(side_experts)
        },
    }
    sizes = {shard: _write_fake_shard(tmp_path / shard, tensors) for shard, tensors in shards.items()}
    weight_map = {name: shard for shard, tensors in shards.items() for name in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return sizes


def _scattered_checkpoint(tmp_path, **kw) -> dict[str, int]:
    return _fake_checkpoint(tmp_path, expert_tensor_bytes=64 << 10, experts=96, side_bytes=16 << 20, **kw)


def _main_stack_experts():
    """The predicate a reader uses to keep the served model's experts (qwen4_exp's anchor)."""
    import re

    return re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.").match


def _unknown_banks_config():
    """A config that leaves ``bank_bytes_estimate()`` unknown, forcing the expert-bytes fallback."""
    from types import SimpleNamespace

    return SimpleNamespace(
        num_moe_layers=None, expert_quant="none", moe_weight_format=None,
        num_experts=None, hidden_size=None, moe_intermediate_size=None,
    )


def _serve_expert_matcher(monkeypatch, matcher):
    """Stand in for the model's own expert key predicate (resolving it needs a registered spec)."""
    import freetoken.moe.expert_pieces as ep

    monkeypatch.setattr(ep, "expert_key_matcher", lambda path, config, kind: matcher)


def test_expert_storage_sizes_experts_not_the_checkpoint(tmp_path):
    from freetoken.models.weight import expert_storage

    sizes = _scattered_checkpoint(tmp_path)
    generic = expert_storage(str(tmp_path))
    assert generic.expert_bytes == 96 * (64 << 10) + (16 << 20)
    # a reader that only knows ".experts." in the name would buffer the speculative head's shard
    assert generic.expert_shard_bytes == sizes["model-00003-of-00003.safetensors"]

    served = expert_storage(str(tmp_path), _main_stack_experts())
    assert served.scattered
    assert served.expert_bytes == 96 * (64 << 10)
    # the reader buffers whole shards, but only the shards holding tensors it will place
    assert served.expert_shard_bytes == sizes["model-00001-of-00003.safetensors"]
    assert served.expert_shard_bytes < max(sizes.values())


def test_expert_key_matcher_is_best_effort():
    from types import SimpleNamespace

    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import expert_key_matcher

    assert expert_key_matcher("a/path", SimpleNamespace(architectures=["Qwen4Exp"]), QuantKind.NONE) is None
    # an unresolvable family must not raise out of a sizing heuristic
    assert expert_key_matcher("a/path", SimpleNamespace(architectures=["NoSuchArch"]), QuantKind.NVFP4) is None


def test_parallel_reader_gate_is_the_direct_read_seam(monkeypatch):
    """The gate asks the host_banks seam whether an unbuffered read exists, not the platform for
    its POSIX spelling. Windows has neither ``os.O_DIRECT`` nor ``os.preadv`` and still reads
    scattered experts in parallel, which the old ``hasattr`` gate silently denied it."""
    import importlib

    import freetoken.moe.expert_banks as eb
    from freetoken.moe import host_banks

    for name in ("O_DIRECT", "preadv"):
        monkeypatch.delattr(os, name, raising=False)
    importlib.reload(eb)
    try:
        assert eb._PARALLEL_READER_SUPPORTED is host_banks.DIRECT_READ_SUPPORTED is True
    finally:
        importlib.reload(eb)


def test_mem_available_bytes_answers_where_proc_meminfo_is_absent():
    """The auto expert-load guard read /proc/meminfo and gave up elsewhere, so the OOM veto was
    silently off on Windows; GlobalMemoryStatusEx answers there. The value is re-read through a
    hand-built MEMORYSTATUSEX because a mistyped struct layout shows up as garbage, not an error."""
    from freetoken.moe.host_banks import mem_available_bytes

    avail = mem_available_bytes()
    assert avail is not None and avail > 0
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes as wt

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        assert ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(status))
        assert 0 < avail < status.ullTotalPhys
        assert abs(status.ullAvailPhys - avail) < max(1 << 20, status.ullAvailPhys // 10)


def test_auto_expert_load_sizes_the_experts_the_reader_opens(tmp_path, monkeypatch):
    import freetoken.moe.expert_banks as eb
    from freetoken.models.weight import EXPERT_PREFETCH_SHARDS, expert_storage

    monkeypatch.setattr(eb, "_PARALLEL_READER_SUPPORTED", True)
    _serve_expert_matcher(monkeypatch, _main_stack_experts())
    sizes = _scattered_checkpoint(tmp_path)
    served = expert_storage(str(tmp_path), _main_stack_experts())
    need = served.expert_bytes + (EXPERT_PREFETCH_SHARDS + 1) * served.expert_shard_bytes
    # what the reader opens is smaller than the checkpoint: sizing by every shard plus the largest
    # file vetoed a box that fits the experts and the buffers it reads
    assert need < sum(sizes.values()) + max(sizes.values())

    monkeypatch.setattr(eb, "_mem_available_bytes", lambda: need + (1 << 20))
    assert eb._auto_pick_parallel(str(tmp_path), _unknown_banks_config(), None, False) is True

    monkeypatch.setattr(eb, "_mem_available_bytes", lambda: need - (1 << 20))
    assert eb._auto_pick_parallel(str(tmp_path), _unknown_banks_config(), None, False) is False


def test_auto_expert_load_stays_conservative_without_the_reader_predicate(tmp_path, monkeypatch):
    import freetoken.moe.expert_banks as eb
    from freetoken.models.weight import EXPERT_PREFETCH_SHARDS, expert_storage

    monkeypatch.setattr(eb, "_PARALLEL_READER_SUPPORTED", True)
    _serve_expert_matcher(monkeypatch, None)  # a family that reads its experts its own way
    _scattered_checkpoint(tmp_path)
    served = expert_storage(str(tmp_path), _main_stack_experts())
    served_need = served.expert_bytes + (EXPERT_PREFETCH_SHARDS + 1) * served.expert_shard_bytes
    # the side file's dropped experts are still counted, so the veto stands: sizing never guesses
    monkeypatch.setattr(eb, "_mem_available_bytes", lambda: served_need + (1 << 20))
    assert eb._auto_pick_parallel(str(tmp_path), _unknown_banks_config(), None, False) is False


def test_auto_expert_load_prefers_the_kernel_bank_size(tmp_path, monkeypatch):
    import freetoken.moe.expert_banks as eb
    from freetoken.models.weight import EXPERT_PREFETCH_SHARDS, expert_storage

    monkeypatch.setattr(eb, "_PARALLEL_READER_SUPPORTED", True)
    _serve_expert_matcher(monkeypatch, _main_stack_experts())
    _scattered_checkpoint(tmp_path)
    served = expert_storage(str(tmp_path), _main_stack_experts())
    # RAM exactly enough for the served expert bytes, but not for what the layout needs
    need = served.expert_bytes + (EXPERT_PREFETCH_SHARDS + 1) * served.expert_shard_bytes
    monkeypatch.setattr(eb, "_mem_available_bytes", lambda: need + (1 << 20))
    monkeypatch.setattr(eb, "bank_bytes_estimate", lambda config, method: None)
    assert eb._auto_pick_parallel(str(tmp_path), _unknown_banks_config(), None, False) is True
    monkeypatch.setattr(eb, "bank_bytes_estimate", lambda config, method: 8 * served.expert_bytes)
    assert eb._auto_pick_parallel(str(tmp_path), _unknown_banks_config(), None, False) is False


def test_auto_expert_load_keeps_serial_when_parallel_wins_nothing(tmp_path, monkeypatch):
    import freetoken.moe.expert_banks as eb

    monkeypatch.setattr(eb, "_mem_available_bytes", lambda: 1 << 40)
    monkeypatch.setattr(eb, "_PARALLEL_READER_SUPPORTED", True)
    _serve_expert_matcher(monkeypatch, _main_stack_experts())
    _scattered_checkpoint(tmp_path)
    assert eb._auto_pick_parallel(str(tmp_path), _unknown_banks_config(), None, True) is False  # dummy

    monkeypatch.setattr(eb, "_PARALLEL_READER_SUPPORTED", False)
    assert eb._auto_pick_parallel(str(tmp_path), _unknown_banks_config(), None, False) is False

    monkeypatch.setattr(eb, "_PARALLEL_READER_SUPPORTED", True)
    packed = tmp_path / "packed"
    packed.mkdir()
    # experts pre-packed into a few big tensors: serial already saturates the disk
    _fake_checkpoint(packed, expert_tensor_bytes=64 << 20, experts=3, side_bytes=1 << 20)
    assert eb._auto_pick_parallel(str(packed), _unknown_banks_config(), None, False) is False


def test_pack_thread_limit_splits_cores_between_ranks(monkeypatch):
    # every TP rank fills its own banks on this host, so the fill must not assume it owns the cores
    import os

    import freetoken.moe.expert_banks as eb
    from freetoken.distributed import DistributedInfo

    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    def _limit(ranks: int) -> int:
        monkeypatch.setattr(eb, "try_get_tp_info", lambda: DistributedInfo(rank=0, size=ranks))
        return eb._pack_thread_limit()

    assert _limit(1) == eb._PACK_THREADS  # pieces are small enough that a wider pool is pure barrier cost
    assert _limit(2) == eb._PACK_THREADS
    assert _limit(16) == 2
    assert _limit(64) == 1  # more ranks than cores still copies, it just stops splitting


def test_expert_bank_fill_runs_on_a_bounded_pool(monkeypatch):
    # the bound has to cover pack and nothing else: leaving it set would narrow the CPU executor and the vision tower
    import freetoken.moe.expert_banks as eb
    from freetoken.moe.expert_banks import build_expert_banks

    _init_tp()
    E, H, I = 4, 8, 16
    layer = _bf16_offload_layer(0, E, 2, H, I)
    method = layer.quant_method
    monkeypatch.setattr(eb, "_pack_thread_limit", lambda: 1)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # the pool the fill runs on is what's under test, not the page-lock

    def _pieces():
        yield 0, 0, E, {
            "gate_up": torch.full((E, 2 * I, H), 0.5, dtype=torch.bfloat16),
            "down": torch.full((E, H, I), 0.25, dtype=torch.bfloat16),
        }

    before = torch.get_num_threads()
    seen = {}
    real_pack = method.pack

    def _counting(pieces, out):
        seen["during"] = torch.get_num_threads()
        return real_pack(pieces, out)

    monkeypatch.setattr(method, "pack", _counting)
    banks = build_expert_banks(method, 1, _pieces(), device=torch.device("cpu"))

    assert seen["during"] == 1
    assert torch.get_num_threads() == before  # given back once the banks are filled
    assert banks.sources["gate_up"][0].shape == (E, 2 * I, H)
    assert torch.equal(banks.sources["down"][0], torch.full((E, H, I), 0.25, dtype=torch.bfloat16))


def test_expert_bank_fill_restores_threads_when_pack_raises(monkeypatch):
    import freetoken.moe.expert_banks as eb
    from freetoken.moe.expert_banks import build_expert_banks

    _init_tp()
    E, H, I = 2, 8, 16
    layer = _bf16_offload_layer(0, E, 2, H, I)
    method = layer.quant_method
    monkeypatch.setattr(eb, "_pack_thread_limit", lambda: 1)
    before = torch.get_num_threads()

    def _boom(pieces, out):
        raise RuntimeError("pack failed")

    monkeypatch.setattr(method, "pack", _boom)
    pieces = [(0, 0, E, {"gate_up": torch.zeros(E, 2 * I, H, dtype=torch.bfloat16),
                         "down": torch.zeros(E, H, I, dtype=torch.bfloat16)})]
    with pytest.raises(RuntimeError, match="pack failed"):
        build_expert_banks(method, 1, iter(pieces), device=torch.device("cpu"))

    assert torch.get_num_threads() == before


def _policy_cache():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=4, num_experts=8, cache_size=64, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.auto_promote = True
    return cache


def test_auto_promote_policy_turns_on_when_working_set_fits():
    cache = _policy_cache()
    # slots_avail = 64 - 2*8 = 48; ws = 10/layer * 4 = 40 <= 48 -> on, full fraction
    cache._promote_ema_touched = 10.0
    cache._update_promote_policy()
    assert cache._promote_on and cache.promote_auto_frac == 1.0


def test_auto_promote_policy_partial_fraction_when_ws_grows():
    cache = _policy_cache()
    cache._promote_ema_touched = 10.0
    cache._update_promote_policy()
    cache._promote_ema_touched = 15.0  # ws = 60 > 48 -> stay on, partial frac
    cache._update_promote_policy()
    assert cache._promote_on
    assert abs(cache.promote_auto_frac - 48 / 60) < 1e-9


def test_auto_promote_policy_arms_with_partial_fraction_up_to_2x_budget():
    cache = _policy_cache()
    cache._promote_ema_touched = 18.0  # ws = 72 = 1.5x slots_avail(48): arm, partial
    cache._update_promote_policy()
    assert cache._promote_on
    assert abs(cache.promote_auto_frac - 48 / 72) < 1e-9
    # beyond 2x the budget a fresh policy never arms (feedback cannot save it)
    cache2 = _policy_cache()
    cache2._promote_ema_touched = 25.0  # ws = 100 > 96
    cache2._update_promote_policy()
    assert not cache2._promote_on and cache2.promote_auto_frac == 0.0


def test_auto_promote_policy_demotes_after_thrashing_with_hysteresis():
    cache = _policy_cache()
    cache._promote_ema_touched = 10.0
    cache._update_promote_policy()
    cache._promote_harvests = 4  # past the cold-start warmup guard
    cache._promote_ema_hitrate = 0.01  # near-zero reuse
    cache._update_promote_policy()
    assert cache._promote_on and cache._promote_off_streak == 1
    cache._update_promote_policy()
    assert not cache._promote_on and cache.promote_auto_frac == 0.0
    # hysteresis: ws=40 <= 48 but > 0.8*48=38.4 -> stays off
    cache._promote_ema_hitrate = 0.5
    cache._update_promote_policy()
    assert not cache._promote_on
    cache._promote_ema_touched = 9.0  # ws = 36 <= 38.4 -> re-arms
    cache._update_promote_policy()
    assert cache._promote_on and cache.promote_auto_frac == 1.0


def test_auto_promote_policy_cold_start_does_not_demote():
    # The first chunks of a document are all-miss (hitrate ~0) by construction; the
    # harvest-count warmup guard must keep them from tripping the thrash demotion.
    cache = _policy_cache()
    cache._promote_ema_touched = 10.0
    cache._update_promote_policy()
    assert cache._promote_on
    cache._promote_ema_hitrate = 0.0
    cache._update_promote_policy()
    cache._update_promote_policy()
    cache._update_promote_policy()
    assert cache._promote_on and cache._promote_off_streak == 0


def test_ondemand_chunk_begin_is_noop_without_pinned_stats():
    cache = _policy_cache()  # cpu device -> _promote_pin is None
    cache.ondemand_chunk_begin()  # must not raise
    assert cache.promote_auto_frac == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="lru_ensure is a CUDA kernel")
def test_promote_touched_reuses_across_chunks_and_guards_buffer_slots():
    # GPU regression for the mask-driven fast promote (auto-promote policy):
    # 1) cross-chunk reuse: a second chunk routing mostly the same experts must
    #    keep their slots (only genuinely new experts are staged),
    # 2) multi-slice buffer guard: lru_ensure increments its step PER CALL and
    #    excludes only usage == (call step) from victims; a guard stamped once
    #    before the slice loop goes stale and later slices dump promotions into
    #    the double-buffer slots [0, 2E), where the plan re-classifies them as
    #    misses and the next chunk invalidates them (measured in production as
    #    2x chunk cadence: zero reuse AND double H2D traffic).
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.offload_kernels import promote_touched

    dev = torch.device("cuda")
    E = 8

    def host(role):
        fp8 = torch.float8_e4m3fn
        if role == "gate_up":
            t = torch.randint(0, 255, (E, 256, 128), dtype=torch.uint8)
        elif role == "gate_up_scale":
            t = (torch.rand(E, 256, 16) * 0.05 + 0.5).to(fp8)
        elif role == "gate_up_global":
            t = (torch.rand(E, 256) + 0.5).half()
        elif role == "down":
            t = torch.randint(0, 255, (E, 256, 64), dtype=torch.uint8)
        elif role == "down_scale":
            t = (torch.rand(E, 256, 8) * 0.05 + 0.5).to(fp8)
        else:
            t = (torch.rand(E, 256) + 0.5).half()
        return [t.pin_memory(), t.pin_memory()]

    banks = {r: host(r) for r in ("gate_up", "gate_up_scale", "gate_up_global",
                                  "down", "down_scale", "down_global")}

    # --- 1) cross-chunk reuse -------------------------------------------------
    cache = OffloadMoeCache(num_layers=2, num_experts=E, cache_size=40,
                            device=dev, quant_format="nvfp4")
    cache.set_bank_sources(banks)
    t1 = torch.zeros(E, dtype=torch.int32, device=dev)
    t1[[0, 1, 2]] = 1
    promote_touched(cache, 0, t1, sub_k=4)
    sf = cache.slot_for_id[0].clone()
    assert (sf[[0, 1, 2]] >= 2 * E).all()
    t2 = torch.zeros(E, dtype=torch.int32, device=dev)
    t2[[1, 2, 3]] = 1
    promote_touched(cache, 0, t2, sub_k=4)
    sf2 = cache.slot_for_id[0]
    assert (sf2[[1, 2]] == sf[[1, 2]]).all(), "reused rows must keep their slots"
    assert sf2[3] >= 2 * E, "new expert promoted"

    # --- 2) multi-slice stale-guard regression --------------------------------
    cache2 = OffloadMoeCache(num_layers=2, num_experts=E, cache_size=2 * E + 2,
                             device=dev, quant_format="nvfp4")
    cache2.set_bank_sources(banks)
    t3 = torch.zeros(E, dtype=torch.int32, device=dev)
    t3[[0, 1, 4, 5]] = 1
    promote_touched(cache2, 0, t3, sub_k=2)  # misses in slice 0 AND slice 2
    sf3 = cache2.slot_for_id[0]
    assigned = sf3 >= 0
    assert assigned.sum() == 2  # only 2 non-buffer slots exist; self-eviction is legal
    assert (sf3[assigned] >= 2 * E).all(), (
        f"promotions landed in double-buffer slots (stale guard): {sf3.tolist()}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernels need CUDA")
def test_slot_direct_prefill_gemm_matches_position_id_reference():
    # P1'' (slot-direct prefill): with every routed row LRU-resident (promote_touched),
    # the grouped prefill GEMM reading rows THROUGH the id->slot map must produce the
    # identical output to the reference reading position==expert-id banks. This is what
    # lets the on-demand path skip the per-chunk double-buffer D2D staging.
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.offload_kernels import promote_touched

    dev = torch.device("cuda")
    E, S, K, I, NL = 8, 40, 256, 128, 2
    fp8 = torch.float8_e4m3fn
    torch.manual_seed(5)

    def host(role):
        if role == "gate_up":
            t = torch.randint(0, 255, (E, 2 * I, K // 2), dtype=torch.uint8)
        elif role == "gate_up_scale":
            t = (torch.rand(E, 2 * I, K // 16) * 0.05 + 0.5).to(fp8)
        elif role == "gate_up_global":
            t = (torch.rand(E, 2 * I) + 0.5).half()
        elif role == "down":
            t = torch.randint(0, 255, (E, K, I // 2), dtype=torch.uint8)
        elif role == "down_scale":
            t = (torch.rand(E, K, I // 16) * 0.05 + 0.5).to(fp8)
        else:
            t = (torch.rand(E, K) + 0.5).half()
        return [t.pin_memory() for _ in range(NL)]

    banks = {r: host(r) for r in ("gate_up", "gate_up_scale", "gate_up_global",
                                  "down", "down_scale", "down_global")}
    cache = OffloadMoeCache(num_layers=NL, num_experts=E, cache_size=S,
                            device=dev, quant_format="nvfp4", prefill_overlap=True)
    cache.set_bank_sources(banks)

    T, TOPK = 6, 2
    x = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    w = torch.rand(T, TOPK, device=dev, dtype=torch.float32) + 0.05
    ids = torch.tensor([[0, 3], [1, 4], [2, 5], [0, 6], [7, 1], [3, 2]],
                       dtype=torch.int32, device=dev)
    touched = torch.zeros(E, dtype=torch.int32, device=dev)
    touched.scatter_(0, ids.reshape(-1).long(), 1)
    promote_touched(cache, 0, touched, sub_k=4)
    torch.cuda.synchronize()
    sf = cache.slot_for_id[0]
    assert (sf[touched.bool()] >= 0).all()

    out_sd = fused_experts_nvfp4(x, *cache.bank_views(), w, ids, E, "silu", False,
                                 slot_map=sf)
    order = ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")
    ref = fused_experts_nvfp4(x, *(banks[r][0].to(dev) for r in order), w, ids, E,
                              "silu", False)
    assert torch.equal(out_sd, ref), (
        f"slot-direct mismatch: max|diff|={(out_sd.float()-ref.float()).abs().max().item()}")

    # stats=True keeps the policy feedback alive on the slot-direct path:
    # a warm re-promote counts every touched row as a hit, no fresh misses.
    cache.auto_promote = True
    cache._promote_pin = torch.zeros(2, dtype=torch.int64, device=dev)
    cache.promote_touched(0, ids, stats=True)
    torch.cuda.synchronize()
    hits0, misses0 = cache._promote_acc.tolist()
    n_touched = int(torch.unique(ids).numel())
    assert (hits0, misses0) == (n_touched, 0), f"expected ({n_touched},0), got {(hits0, misses0)}"


def test_miss_route_mask_unions_the_k_split_plans(monkeypatch):
    """The fetch-overlap miss mask must flag exactly the routes whose slots the
    (possibly chunked) ensure just staged for fetch: first occurrence of each new
    expert in chunk order, across BOTH plan buffers; a warm cache masks nothing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")
    from freetoken.moe.offload_cache import OffloadMoeCache

    torch.manual_seed(5)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=64, cache_size=128, device=torch.device("cuda"),
        decode_fetch_overlap=True,  # the --no-decode-fetch-overlap knob's default
    )
    cache.ensure_dual_plans(2)                 # plan-1 buffers for the K-split ensure
    cache.reset()
    ids = torch.randint(0, 64, (8, 10), dtype=torch.int32, device="cuda")
    orig = ids.clone()                         # ensure rewrites ids -> slots in place

    cache.ensure_experts(0, ids[:5], plan=0)   # K-split chunk 0 -> plan 0
    cache.ensure_experts(0, ids[5:], plan=1)   # chunk 1 -> plan 1
    mask = cache.miss_route_mask(ids, (0, 1))

    # the mask is per-SLOT: EVERY occurrence of a just-staged expert flags True (the
    # complementary GEMV passes zero by route weight, so duplicates of a miss must ride
    # the miss pass -- their slot's bytes are still in flight during the hit pass).
    # an expert missed iff it was new at its first occurrence (cold cache: all of
    # them), and every occurrence inherits that verdict
    first_new = set()
    seen2 = set()
    for row in ids.tolist():
        for eid in row:
            if eid not in seen2:
                first_new.add(eid)
            seen2.add(eid)
    expected = [[eid in first_new for eid in row] for row in ids.tolist()]
    assert mask.tolist() == expected

    # warm: a second ensure of the same layer finds every expert resident (the slots
    # are stamped even without the payload copy), so nothing is staged and the mask
    # is empty -- the graph-replay steady state for a repeated route set.
    ids2 = orig.clone()
    cache.ensure_experts(0, ids2[:5], plan=0)
    cache.ensure_experts(0, ids2[5:], plan=1)
    assert not cache.miss_route_mask(ids2, (0, 1)).any()


def test_residency_split_separates_working_set_from_warm_cache():
    # The gauge counts valid slot-map entries (-1 = empty slot) and, among them, the slots
    # stamped within the last forward (num_layers step increments back from `step`). The
    # unbound method works on a duck-typed namespace so no CUDA cache is needed here.
    from types import SimpleNamespace

    from freetoken.moe.offload_cache import OffloadMoeCache

    fake = SimpleNamespace(
        id_of_slot=torch.tensor([-1, 0, 5, -1, 7, -1], dtype=torch.int32),
        usage=torch.tensor([0, 42, 12, 0, 41, 0], dtype=torch.int64),
        step=torch.tensor(44, dtype=torch.int64), num_layers=4)
    # Filled 3; of those, slots 1 and 4 were touched in steps 41..44 and slot 2 is warm.
    assert OffloadMoeCache.residency_split(fake) == (3, 2)

    empty = SimpleNamespace(id_of_slot=torch.full((8,), -1, dtype=torch.int32),
                            usage=torch.full((8,), 99, dtype=torch.int64),
                            step=torch.tensor(100, dtype=torch.int64), num_layers=4)
    assert OffloadMoeCache.residency_split(empty) == (0, 0)
