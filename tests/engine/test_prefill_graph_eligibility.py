"""Eligibility gates of the prefill chunk graph runner (phase 1 scope).

The gates decide which prefill batches may replay the captured T-bucket graph;
everything else must fall back to eager. CPU-only: no capture, no CUDA.
"""
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch
from freetoken.engine.prefill_graph import PrefillGraphRunner


def _runner():
    engine = SimpleNamespace(device=torch.device("cpu"))
    return PrefillGraphRunner(engine)


def _batch(*, extend_len=128, cached_len=256, can_decode=False, mm=False,
           track=True, slot=7, stash=False, spec_off=False):
    req = SimpleNamespace(
        can_decode=can_decode, extend_len=extend_len, cached_len=cached_len,
        mm_items=["img"] if mm else None, linear_slot_idx=slot,
        table_idx=3, device_len=cached_len + extend_len,
        spec_residual=(torch.zeros(4, 4) if stash else None), spec_off=spec_off,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.mm_gather_plan = None
    batch.fla_metadata = SimpleNamespace(track_dst=torch.zeros(1) if track else None)
    return batch


def _engine(*, pool=True, block_table=True, frac=1.0, cache=True, tokens=True):
    return SimpleNamespace(
        linear_state_pool=object() if pool else None,
        attn_backend=SimpleNamespace(_block_table=(lambda x: x) if block_table else None)
        if block_table else object(),
        moe_offload_cache=SimpleNamespace(promote_auto_frac=frac) if cache else None,
        spec_token_pool=object() if tokens else None,
    )


@pytest.fixture
def slot_direct(monkeypatch):
    import freetoken.layers.moe as m
    monkeypatch.setattr(m, "_PREFILL_SLOT_DIRECT", True)


def test_eligible_happy_path(slot_direct):
    r = _runner()
    assert r.eligible(_engine(), _batch())


def test_gates_reject_out_of_scope(slot_direct):
    r = _runner()
    eng = _engine()
    assert not r.eligible(eng, _batch(extend_len=127))       # T bucket only
    assert not r.eligible(eng, _batch(cached_len=0))         # first chunk (fresh state)
    assert not r.eligible(eng, _batch(can_decode=True))      # sampling chunk -> eager
    assert not r.eligible(eng, _batch(mm=True))              # image rows -> eager
    assert not r.eligible(eng, _batch(track=False))          # no GDN track -> phase 2
    assert not r.eligible(_engine(pool=False), _batch())
    assert not r.eligible(_engine(block_table=False), _batch())
    assert not r.eligible(_engine(frac=0.0), _batch())       # promote policy off
    assert not r.eligible(_engine(cache=False), _batch())
    assert not r.eligible(_engine(tokens=False), _batch())
    two = _batch()
    two.reqs.append(two.reqs[0])
    two.padded_reqs = list(two.reqs)
    assert not r.eligible(eng, two)                          # multi-req batch -> eager
    r.disabled = True
    assert not r.eligible(eng, _batch())


def test_a_live_mtp_stash_keeps_the_chunk_eager(slot_direct):
    """The replay returns before the engine's residual-stash gate, so graphing a continuation
    chunk would commit rows the head has no residual for -- a hole in the span, and the request
    is off-spec for its whole life. Once the request is off-spec there is nothing to protect."""
    r = _runner()
    eng = _engine()
    assert not r.eligible(eng, _batch(stash=True))
    assert r.eligible(eng, _batch(stash=True, spec_off=True))


def test_not_eligible_without_slot_direct(monkeypatch):
    import freetoken.layers.moe as m
    monkeypatch.setattr(m, "_PREFILL_SLOT_DIRECT", False)
    assert not _runner().eligible(_engine(), _batch())


def test_decode_and_spec_batches_never_eligible(slot_direct):
    r = _runner()
    d = _batch()
    d.phase = "decode"
    assert not r.eligible(_engine(), d)
    s = _batch()
    s.spec_mode = "verify"
    assert not r.eligible(_engine(), s)
