"""CUDA graph batch-size candidate selection (engine/graph.py::_determine_cuda_graph_bs).

The default list must cover every decode batch size natively at small bs: a size
missing from the capture list pads the batch up to the next captured graph, and
the padded dummy row's work is paid while its token is thrown away (conc-3 on the
old [1, 2, 4] list ran the bs=4 graph and lost ~25% per-user throughput).
"""

from freetoken.engine.graph import _determine_cuda_graph_bs

GB = 1 << 30


def test_small_max_covers_every_batch_size_natively():
    # max_running_requests=4 (this deployment): no padding gap at bs=3
    assert _determine_cuda_graph_bs(None, 4, 3 * GB) == [1, 2, 3, 4]
    assert _determine_cuda_graph_bs(None, 1, 3 * GB) == [1]
    assert _determine_cuda_graph_bs(None, 8, 3 * GB) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_large_max_is_dense_below_eight_then_strides_by_four_then_eight():
    got = _determine_cuda_graph_bs(None, 160, 3 * GB)
    assert got[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert got[8:] == list(range(12, 33, 4)) + list(range(40, 161, 8))
    assert 3 in got and 160 in got
    assert 20 in got  # bs20 must not pad to 24: dummy rows cost 20% of the step


def test_default_max_clamps_to_max_running_req():
    # a decode batch can never exceed max_running_req; captures above it are dead
    got = _determine_cuda_graph_bs(None, None, 3 * GB, max_running_req=20)
    assert got == [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20]
    # an explicit --cuda-graph-max-bs still wins over the clamp
    got = _determine_cuda_graph_bs(None, 160, 3 * GB, max_running_req=20)
    assert got[-1] == 160


def test_h200_default_max_uses_the_same_shape():
    got = _determine_cuda_graph_bs(None, None, 90 * GB)  # >80GB free -> max 256
    assert got[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert got[-1] == 256
    assert 9 not in got  # 9..15 pad to 16, as before


def test_explicit_list_and_disabled_cases_are_untouched():
    assert _determine_cuda_graph_bs([1, 4], None, 3 * GB) == [1, 4]
    assert _determine_cuda_graph_bs(None, 0, 3 * GB) == []
    # a small-machine default without an explicit max still caps by memory branch
    got = _determine_cuda_graph_bs(None, None, 3 * GB)
    assert got[:8] == [1, 2, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Dual-microbatch decode helpers (engine/graph.py): the host-side arithmetic
# that slices one captured graph into two skewed half-batches. A wrong bound
# here still replays a valid-looking graph -- it just reads the other half's
# rows, so the expectations pin the row window and slot wiring directly.


def _runner(**attrs):
    from freetoken.engine.graph import GraphRunner

    r = GraphRunner.__new__(GraphRunner)
    r._dual_stream = attrs.pop("dual_stream", object())
    r.attn_backend = attrs.pop("attn_backend", None)
    r.moe_offload_cache = attrs.pop("moe_offload_cache", None)
    r.dummy_req = attrs.pop("dummy_req", object())
    r._dual_cu = attrs.pop("dual_cu", None)
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


def _model(**attrs):
    from types import SimpleNamespace

    defaults = {"supports_dual_decode": True}
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def _backend(**attrs):
    from types import SimpleNamespace

    defaults = {"stage_dual_replay": True, "dual_profile": True}
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def test_dual_ok_rejects_small_or_odd_batches(monkeypatch):
    import freetoken.engine.graph as graph_mod

    monkeypatch.setattr(graph_mod, "dual_ar_available", lambda: True)
    r = _runner(attn_backend=_backend(), moe_offload_cache=object())
    m = _model()
    assert r._dual_ok(m, 3) is False  # below 4
    assert r._dual_ok(m, 5) is False  # odd


def test_dual_ok_requires_a_configured_stream(monkeypatch):
    import freetoken.engine.graph as graph_mod

    monkeypatch.setattr(graph_mod, "dual_ar_available", lambda: True)
    r = _runner(dual_stream=None, attn_backend=_backend(), moe_offload_cache=object())
    assert r._dual_ok(_model(), 4) is False


def test_dual_ok_gates_on_model_backend_and_ar(monkeypatch):
    import freetoken.engine.graph as graph_mod

    monkeypatch.setattr(graph_mod, "dual_ar_available", lambda: False)
    good = _runner(attn_backend=_backend(), moe_offload_cache=object())
    assert good._dual_ok(_model(), 4) is False  # second AR instance missing

    monkeypatch.setattr(graph_mod, "dual_ar_available", lambda: True)
    assert good._dual_ok(_model(supports_dual_decode=False), 4) is False
    assert good._dual_ok(_model(), 4) is True
    assert _runner(attn_backend=_backend(), moe_offload_cache=None)._dual_ok(_model(), 4) is False
    missing_stage = _backend()
    del missing_stage.stage_dual_replay  # the attribute itself must be absent
    assert _runner(
        attn_backend=missing_stage, moe_offload_cache=object()
    )._dual_ok(_model(), 4) is False


def test_dual_ok_uses_the_decode_backend_when_present(monkeypatch):
    import freetoken.engine.graph as graph_mod

    monkeypatch.setattr(graph_mod, "dual_ar_available", lambda: True)
    decode = _backend()
    r = _runner(attn_backend=_backend(decode_backend=decode), moe_offload_cache=object())
    assert r._dual_ok(_model(), 4) is True
    # a wrapper that advertises nothing but a capable decode backend still qualifies
    r2 = _runner(
        attn_backend=_backend(stage_dual_replay=None, dual_profile=None, decode_backend=decode),
        moe_offload_cache=object(),
    )
    assert r2._dual_ok(_model(), 4) is True


def test_build_dual_half_slices_the_row_window(monkeypatch):
    import torch
    from freetoken.engine.graph import GraphRunner

    bs = 6
    buffer = type(
        "B",
        (),
        {
            "input_ids": torch.arange(bs * 3, dtype=torch.int64).view(bs, 3),
            "out_loc": torch.arange(bs * 2, dtype=torch.int64).view(bs, 2),
            "positions": torch.arange(bs, dtype=torch.int32),
            "mrope_positions": torch.arange(3 * bs, dtype=torch.int32).view(3, bs),
            "table_idx": torch.arange(bs, dtype=torch.int64),
        },
    )()
    r = _runner(
        dual_cu=[torch.arange(4, dtype=torch.int32), torch.arange(5, dtype=torch.int32)],
    )

    monkeypatch.delenv("FREETOKEN_DUAL_SLOT0", raising=False)
    half = r._build_dual_half(buffer, 2, 6, 1)

    assert [half.padded_reqs[i] is r.dummy_req for i in range(4)] == [True] * 4
    assert torch.equal(half.input_ids, buffer.input_ids[2:6])
    assert torch.equal(half.out_loc, buffer.out_loc[2:6])
    assert torch.equal(half.positions, buffer.positions[2:6])
    assert torch.equal(half.mrope_positions, buffer.mrope_positions[:, 2:6])
    assert torch.equal(half.linear_table_idx, buffer.table_idx[2:6])
    assert torch.equal(half.fla_metadata.cu_seqlens, r._dual_cu[1][:5])
    assert torch.equal(half.fla_metadata.cache_indices, buffer.table_idx[2:6])
    assert half.ple_row_offset == 2
    assert half.dual_slot == 1


def test_build_dual_half_respects_the_slot0_probe_env(monkeypatch):
    import torch

    buffer = type(
        "B",
        (),
        {
            "input_ids": torch.arange(12, dtype=torch.int64).view(4, 3),
            "out_loc": torch.arange(8, dtype=torch.int64).view(4, 2),
            "positions": torch.zeros(4, dtype=torch.int32),
            "mrope_positions": None,
            "table_idx": torch.zeros(4, dtype=torch.int64),
        },
    )()
    r = _runner(dual_cu=[torch.arange(3, dtype=torch.int32)] * 2)

    monkeypatch.setenv("FREETOKEN_DUAL_SLOT0", "1")
    half = r._build_dual_half(buffer, 0, 2, 1)
    assert half.dual_slot == 0  # bisection probe routes BOTH halves to slot 0
    assert half.mrope_positions is None  # non-mrope models carry no mrope slice
