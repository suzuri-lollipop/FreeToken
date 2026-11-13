"""Thinking-budget wrap-up (engine/think_wrap.py) regression tests.

The mechanism forces the end-of-thinking tag when a request is mid-think and its
remaining max_tokens budget drops to the reserve, so the model answers instead of
being cut off mid-thought. These pin the state machine and the safety boundaries:
only inside the think block, once per block, never for ignore_eos, decode-only.
Reverting the trigger / the in-think gate / the once-per-block arm turns these red.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.think_wrap import ThinkWrap, ThinkWrapState


# ---------------------------------------------------------------- state machine

def test_single_token_tags_trigger_at_reserve():
    st = ThinkWrapState(start_ids=[100], end_ids=[101])
    assert st.next_force(remaining=5, reserve=10) is None  # not thinking yet
    st.observe(100)                                        # <think>
    assert st.in_think
    assert st.next_force(remaining=50, reserve=10) is None  # plenty of budget
    assert st.next_force(remaining=10, reserve=10) == 101   # hit reserve -> </think>
    # once per block: armed, no re-force even though still "in think" until observed
    assert st.next_force(remaining=9, reserve=10) is None
    st.observe(101)                                        # </think> lands
    assert not st.in_think
    assert st.next_force(remaining=8, reserve=10) is None   # thinking over


def test_no_force_when_budget_exhausted():
    st = ThinkWrapState(start_ids=[100], end_ids=[101])
    st.observe(100)
    assert st.next_force(remaining=0, reserve=10) is None
    assert st.next_force(remaining=-1, reserve=10) is None


def test_multi_token_end_tag_forced_in_order():
    st = ThinkWrapState(start_ids=[100], end_ids=[101, 102])
    st.observe(100)
    assert st.next_force(remaining=5, reserve=10) == 101
    assert st.next_force(remaining=4, reserve=10) == 102
    assert st.next_force(remaining=3, reserve=10) is None   # queue drained, armed
    st.observe(101)
    st.observe(102)
    assert not st.in_think


def test_rearms_on_a_new_think_block():
    st = ThinkWrapState(start_ids=[100], end_ids=[101])
    st.observe(100)
    assert st.next_force(remaining=5, reserve=10) == 101
    st.observe(101)
    st.observe(100)                                        # a second think block
    assert st.in_think
    assert st.next_force(remaining=5, reserve=10) == 101


def test_natural_end_before_reserve_never_forces():
    st = ThinkWrapState(start_ids=[100], end_ids=[101])
    st.observe(100)
    st.observe(101)                                        # model ended thinking on its own
    assert st.next_force(remaining=5, reserve=10) is None


# ---------------------------------------------------------------- controller

class _Tok:
    def encode(self, text, add_special_tokens=False):
        return {"<think>": [100], "</think>": [101]}.get(text, [999])


def _controller(monkeypatch, *, on="1", reserve="0.1", speculative="none"):
    monkeypatch.setenv("FREETOKEN_THINK_WRAP", on)
    monkeypatch.setenv("FREETOKEN_THINK_WRAP_RESERVE", reserve)
    cfg = SimpleNamespace(speculative=speculative)
    return ThinkWrap(_Tok(), cfg)


def _req(output_len=100, ignore_eos=False):
    return SimpleNamespace(
        output_len=output_len,
        max_device_len=1000 + output_len,
        device_len=1000,
        input_ids=torch.arange(1000, dtype=torch.int32),  # prompt, no think tag
        prompt_len=1000,
        sampling_params=SimpleNamespace(ignore_eos=ignore_eos),
    )


def test_controller_disabled_by_env(monkeypatch):
    assert _controller(monkeypatch, on="0").enabled is False


def test_controller_disabled_under_speculative(monkeypatch):
    assert _controller(monkeypatch, speculative="mtp").enabled is False


def test_reserve_is_fraction_of_max_tokens(monkeypatch):
    tw = _controller(monkeypatch, reserve="0.1")
    assert tw.reserve_for(_req(output_len=100)) == 10
    assert tw.reserve_for(_req(output_len=8)) == 1  # clamped to >= 1


def test_apply_forces_end_tag_mid_think_at_reserve(monkeypatch):
    tw = _controller(monkeypatch)
    req = _req(output_len=100)
    tw.observe(req, 100)                       # thinking
    req.device_len = 1090                      # remaining = 1100-1090 = 10 = reserve
    batch = SimpleNamespace(is_decode=True, reqs=[req])
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(batch, tokens)
    assert tokens[0].item() == 101             # forced </think>


def test_preopened_think_block_detected_from_prompt(monkeypatch):
    # Thinking templates pre-open the block: the prompt ends with the start tag (+newline),
    # so the model never emits it. in_think must be seeded from the prompt tail.
    tw = _controller(monkeypatch)
    req = _req(output_len=100)
    req.input_ids = torch.tensor([1, 2, 3, 100, 7], dtype=torch.int32)  # ...<think>\n
    req.prompt_len = 5
    req.device_len = 1090                       # remaining = 10 = reserve
    tw.observe(req, 555)                        # first generated token = thinking content
    assert req._think_wrap.in_think
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(SimpleNamespace(is_decode=True, reqs=[req]), tokens)
    assert tokens[0].item() == 101              # forced </think>


def test_prompt_without_think_tag_does_not_arm(monkeypatch):
    tw = _controller(monkeypatch)
    req = _req(output_len=100)                  # prompt = arange, no start tag
    req.device_len = 1090
    tw.observe(req, 555)                        # plain content, not thinking
    assert not req._think_wrap.in_think
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(SimpleNamespace(is_decode=True, reqs=[req]), tokens)
    assert tokens[0].item() == 555              # untouched


def test_apply_skips_ignore_eos(monkeypatch):
    tw = _controller(monkeypatch)
    req = _req(output_len=100, ignore_eos=True)
    tw.observe(req, 100)
    req.device_len = 1090
    batch = SimpleNamespace(is_decode=True, reqs=[req])
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(batch, tokens)
    assert tokens[0].item() == 555             # untouched: full budget wanted


def test_apply_skips_prefill_and_non_thinking(monkeypatch):
    tw = _controller(monkeypatch)
    req = _req(output_len=100)
    tw.observe(req, 100)
    req.device_len = 1090
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(SimpleNamespace(is_decode=False, reqs=[req]), tokens)  # prefill
    assert tokens[0].item() == 555
    req2 = _req(output_len=100)                # never thought
    req2.device_len = 1090
    tw.apply(SimpleNamespace(is_decode=True, reqs=[req2]), tokens)
    assert tokens[0].item() == 555