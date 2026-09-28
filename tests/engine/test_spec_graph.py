"""SpecGraphRunner staging and state bookkeeping (engine/spec_graph.py), CPU-only.

The two-row verify graph itself is captured/replayed CUDA-side and needs the disk
PLE backend plus a live engine, so the graph path is the e2e/scheduler gate's job.
What fails SILENTLY and is pure host logic is the staging table: the S_* scalar
layout, the per-step fanout derivations, the state save/restore around capture, and
the adopt-or-disable gate. A wrong index here still looks like a number -- the
graph replays with the wrong slot drafted, staged or restored -- so these are
pinned directly on the CPU tensors the methods touch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch

from freetoken.engine.spec_graph import (
    S_DRAFT,
    S_LIVE,
    S_N,
    S_POS,
    S_SCRATCH,
    S_SEQ,
    S_TABLE,
    SpecGraphRunner,
)


def _runner():
    """A runner with only the CPU-host / scalar tensors its host-only methods touch."""
    r = SpecGraphRunner.__new__(SpecGraphRunner)
    r.device = torch.device("cpu")
    r.h_scal = torch.zeros(S_N, dtype=torch.int64)
    r.d_scal = torch.zeros(S_N, dtype=torch.int64)
    r.d_positions = torch.zeros(2, dtype=torch.int32)
    r.d_mrope = torch.zeros(3, 2, dtype=torch.int32)
    r.d_seq = torch.zeros(1, dtype=torch.int32)
    r.d_live = torch.zeros(1, dtype=torch.int32)
    r.d_scratch = torch.zeros(1, dtype=torch.int64)
    r.d_table = torch.zeros(1, dtype=torch.int64)
    r.c_off = torch.tensor([0, 1], dtype=torch.int64)
    return r


def _req(**kw):
    return SimpleNamespace(
        cached_len=100, table_idx=3, linear_slot_idx=7, spec_slot_idx=5,
        device_len=102, **kw,
    )


def test_stage_layout_pins_each_field_to_its_slot():
    # The graph's _fanout reads these six scalars back OUT of the staged device row by
    # index; a swapped S_* constant would still copy a valid integer and the graph
    # would draft/stage the wrong request slot. Pin the index -> field mapping.
    r = _runner()
    r._stage(SimpleNamespace(spec_draft_id=42), _req())
    assert r.h_scal.tolist() == [100, 3, 7, 5, 42, 102]
    assert r.h_scal[S_POS].item() == 100
    assert r.h_scal[S_TABLE].item() == 3
    assert r.h_scal[S_LIVE].item() == 7
    assert r.h_scal[S_SCRATCH].item() == 5
    assert r.h_scal[S_DRAFT].item() == 42
    assert r.h_scal[S_SEQ].item() == 102
    assert S_N == 6


def test_fanout_derives_each_device_buffer_from_the_staged_scalars():
    r = _runner()
    r._stage(SimpleNamespace(spec_draft_id=42), _req())
    r._fanout()
    # positions are [cached_len, cached_len+1]; mrope mirrors them across its 3 dims
    assert r.d_positions.tolist() == [100, 101]
    assert r.d_mrope.tolist() == [[100, 101], [100, 101], [100, 101]]
    # the per-request slot scalars land in the device buffers the graph gathers from
    assert r.d_seq.tolist() == [102]
    assert r.d_live.tolist() == [7]
    assert r.d_scratch.tolist() == [5]
    assert r.d_table.tolist() == [3]


def test_save_restore_round_trips_and_clones():
    slots = 8
    pool = SimpleNamespace(
        conv_states=torch.randn(2, slots, 4),
        recurrent_states=torch.randn(2, slots, 3, 4),
        slot_states={"foo": torch.randn(1, slots, 5)},
    )
    r = _runner()
    r.engine = SimpleNamespace(linear_state_pool=pool)

    slot = 5
    saved = r.save_state(slot)
    # the saved snapshots are detached clones (mutating the pool must not corrupt them)
    assert [s.shape for s in saved] == [(2, 4), (2, 3, 4), (1, 5)]
    assert pool.conv_states[:, slot].data_ptr() != saved[0].data_ptr()

    # corrupt every slot for the captured state, then restore
    pool.conv_states[:, slot] += 100
    pool.recurrent_states[:, slot] += 100
    pool.slot_states["foo"][:, slot] += 100
    r.restore_state(slot, saved)
    assert torch.equal(pool.conv_states[:, slot], saved[0])
    assert torch.equal(pool.recurrent_states[:, slot], saved[1])
    assert torch.equal(pool.slot_states["foo"][:, slot], saved[2])


def test_invalidate_drops_the_graph_and_rewinds_the_warm_count():
    r = _runner()
    r.graph = object()
    r.warm = 3
    r.invalidate()
    assert r.graph is None and r.warm == 0


def test_capture_without_the_disk_ple_backend_disables_and_stays_eager():
    # The capture path gates on the disk PLE table BEFORE any CUDA work; a missing
    # table (or one without fill_spec_rows) must disable the runner, not raise.
    r = _runner()
    r.disabled = False
    ok = r.capture_after_step(SimpleNamespace(), SimpleNamespace(), None, None, {}, None)
    assert ok is False and r.disabled is True

    r = _runner()
    r.disabled = False
    ok = r.capture_after_step(
        SimpleNamespace(), SimpleNamespace(_ple_table=object()), None, None, {}, None
    )
    assert ok is False and r.disabled is True


def test_capture_replay_divergence_raises_and_disables():
    # A capture whose replay disagrees with the eager payload must surface loudly and
    # disable the runner; patch the statics so the diverged payload reaches the check
    # without a real graph. The adoption is all-or-nothing.
    r = _runner()
    table = SimpleNamespace(fill_spec_rows=lambda req: None)
    with mock.patch.object(SpecGraphRunner, "_alloc_static", lambda self, model: None), \
            mock.patch.object(SpecGraphRunner, "_stage", lambda self, b, req: None), \
            mock.patch.object(SpecGraphRunner, "restore_state", lambda *a: None):
        r._stream = mock.MagicMock()
        r._body = lambda engine, model: None  # leaves the payload at its staged zeros
        r.d_payload = torch.zeros(4, dtype=torch.int64)
        eager = {"y1": 41, "y2": 42, "draft": 77, "accept": True}
        import pytest
        # the divergence raises and the runner is disabled before the error is
        # re-wrapped for the caller (all-or-nothing adoption; no half-installed graph)
        with pytest.raises(RuntimeError, match="graph capture failed"):
            r.capture_after_step(SimpleNamespace(), SimpleNamespace(_ple_table=table),
                                 None, None, eager, None)
        assert r.disabled is True and r.graph is None