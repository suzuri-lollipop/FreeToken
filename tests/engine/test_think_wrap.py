"""Thinking-budget wrap-up (engine/think_wrap.py) regression tests.

The mechanism forces a wrap-up nudge message + the end-of-thinking tag when a
request is mid-think and its remaining max_tokens budget drops to the answer
reserve (llama.cpp's reasoning_budget_message prepended to the end tag), so the
model answers instead of being cut off mid-thought. These pin the state machine
and the safety boundaries: message-before-tag order, the trigger leaving the
full reserve for the answer, only inside the think block, once per block, never
for ignore_eos, decode-only, and the UTF-8 boundary wait. Reverting the message
injection / the trigger / the in-think gate / the once-per-block arm turns these
red.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.think_wrap import DEFAULT_WRAP_MESSAGE, ThinkWrap, ThinkWrapState


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


def test_message_forced_before_end_tag():
    # llama.cpp prepends reasoning_budget_message to the end tag: the nudge must
    # land first, in order, then the tag -- a bare-tag revert turns this red.
    st = ThinkWrapState(start_ids=[100], end_ids=[101, 102], message_ids=[200, 201])
    st.observe(100)
    assert st.next_force(remaining=50, reserve=10) is None  # plenty of budget
    assert st.next_force(remaining=14, reserve=10) == 200   # reserve + len(sequence)
    assert st.next_force(remaining=13, reserve=10) == 201
    assert st.next_force(remaining=12, reserve=10) == 101   # end tag after the message
    assert st.next_force(remaining=11, reserve=10) == 102
    assert st.next_force(remaining=10, reserve=10) is None  # drained, armed
    st.observe(101)
    st.observe(102)
    assert not st.in_think


def test_trigger_leaves_full_reserve_for_the_answer():
    # arm at reserve + len(sequence), not at reserve: firing late would let the
    # injection eat the answer budget it exists to protect
    st = ThinkWrapState(start_ids=[100], end_ids=[101], message_ids=[200, 201])
    st.observe(100)
    assert st.next_force(remaining=14, reserve=10) is None  # one above reserve + 3
    assert st.next_force(remaining=13, reserve=10) == 200


def test_no_force_when_sequence_leaves_no_answer_room():
    st = ThinkWrapState(start_ids=[100], end_ids=[101], message_ids=[200, 201])
    st.observe(100)
    assert st.next_force(remaining=3, reserve=10) is None   # sequence fits, answer does not
    assert st.next_force(remaining=2, reserve=10) is None
    assert st.next_force(remaining=4, reserve=10) == 200    # one answer token survives


def test_utf8_gate_delays_arming_not_draining():
    st = ThinkWrapState(start_ids=[100], end_ids=[101], message_ids=[200, 201])
    st.observe(100)
    assert st.next_force(remaining=13, reserve=10, utf8_ok=False) is None  # mid-character
    assert st.next_force(remaining=13, reserve=10, utf8_ok=True) == 200
    # once armed the sequence completes even if the boundary flag flips again
    assert st.next_force(remaining=12, reserve=10, utf8_ok=False) == 201
    assert st.next_force(remaining=11, reserve=10, utf8_ok=False) == 101


# ---------------------------------------------------------------- controller

_MSG = "\n\nWrap up now."  # stand-in nudge message with a known tokenization


class _Tok:
    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        return {"<think>": [100], "</think>": [101], _MSG: [200, 201]}.get(text, [999])

    def decode(self, ids):
        # id 777 stands in for a byte-fallback token leaving a partial character
        return "\ufffd" if ids and ids[-1] == 777 else "ok"


def _controller(monkeypatch, *, on="1", reserve="0.1", speculative="none", message=""):
    monkeypatch.setenv("FREETOKEN_THINK_WRAP", on)
    monkeypatch.setenv("FREETOKEN_THINK_WRAP_RESERVE", reserve)
    # empty message = the bare-tag path (llama.cpp's default); the message tests
    # pass _MSG or unset the var to exercise the default nudge
    monkeypatch.setenv("FREETOKEN_THINK_WRAP_MESSAGE", message)
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


def test_disabled_by_default(monkeypatch):
    # Off by default until per-family think-tag auto-detection lands: an unset
    # FREETOKEN_THINK_WRAP must not enable the wrap-up. Reverting the default to
    # "1" turns this red.
    monkeypatch.delenv("FREETOKEN_THINK_WRAP", raising=False)
    tw = ThinkWrap(_Tok(), SimpleNamespace(speculative="none"))
    assert tw.enabled is False


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


def test_apply_injects_message_then_end_tag(monkeypatch):
    tw = _controller(monkeypatch, message=_MSG)   # force sequence = [200, 201, 101]
    req = _req(output_len=100)                    # reserve = 10
    tw.observe(req, 100)                          # thinking
    req.device_len = 1087                         # remaining = 13 = reserve + 3
    batch = SimpleNamespace(is_decode=True, reqs=[req])
    for expect in (200, 201, 101):                # nudge first, end tag last
        tokens = torch.tensor([555], dtype=torch.int32)
        tw.apply(batch, tokens)
        assert tokens[0].item() == expect
        req.device_len += 1
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(batch, tokens)
    assert tokens[0].item() == 555                # once per block: no re-force


def test_apply_waits_for_utf8_boundary(monkeypatch):
    tw = _controller(monkeypatch, message=_MSG)
    req = _req(output_len=100)
    tw.observe(req, 100)                          # thinking
    req.input_ids = torch.tensor([1, 2, 777], dtype=torch.int32)  # tail = partial char
    req.device_len = 1087                         # remaining = 13 = reserve + 3
    batch = SimpleNamespace(is_decode=True, reqs=[req])
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(batch, tokens)
    assert tokens[0].item() == 555                # mid-character: injection delayed
    req.input_ids = torch.tensor([1, 2, 777, 88], dtype=torch.int32)  # character completed
    req.device_len = 1088                         # remaining = 12, still in the window
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(batch, tokens)
    assert tokens[0].item() == 200                # boundary complete: nudge starts


def test_default_message_is_the_nudge(monkeypatch):
    # The fix for the force-closed-with-stop report: the default launch injects a
    # wrap-up nudge, not a bare end tag. An emptied default turns this red.
    monkeypatch.setenv("FREETOKEN_THINK_WRAP", "1")
    monkeypatch.delenv("FREETOKEN_THINK_WRAP_MESSAGE", raising=False)
    tw = ThinkWrap(_Tok(), SimpleNamespace(speculative="none"))
    assert DEFAULT_WRAP_MESSAGE
    assert tw.message_ids == _Tok().encode(DEFAULT_WRAP_MESSAGE, add_special_tokens=False)
    assert tw.message_ids                         # non-empty on this tokenizer
    req = _req(output_len=100)                    # reserve = 10, sequence = [999, 101]
    tw.observe(req, 100)
    req.device_len = 1088                         # remaining = 12 = reserve + 2
    tokens = torch.tensor([555], dtype=torch.int32)
    tw.apply(SimpleNamespace(is_decode=True, reqs=[req]), tokens)
    assert tokens[0].item() == tw.message_ids[0]  # the nudge, not the end tag
