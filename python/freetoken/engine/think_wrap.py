"""Thinking-budget wrap-up: conclude reasoning before max_tokens runs out.

Opt-in: OFF by default (set FREETOKEN_THINK_WRAP=1 to enable). The built-in think
tags only match the standard think-tag families; per-family tag auto-detection is
not implemented yet, so other families would silently no-op.

Port of llama.cpp's reasoning-budget mechanism (common/reasoning-budget.cpp),
keyed to the request's remaining max_tokens budget instead of a separate
counter: while a request is inside a thinking block and its remaining output
budget drops to the answer reserve, the next tokens are forced to a wrap-up
NUDGE MESSAGE followed by the end-of-thinking tag -- llama.cpp's
`reasoning_budget_message` prepended to the end tag. The message reads as the
model's own decision to conclude ("I need to wrap up my thinking now..."), so
the model spends what is left on a final answer (or tool call) instead of
having its thinking block cut off mid-thought by a bare end tag. The injection
starts early enough that the full reserve survives it for the answer.

Safety boundaries:
  * only acts INSIDE the thinking block (start tag seen, end tag not yet), so
    post-thinking content and tool calls are never rewritten;
  * skipped for ignore_eos requests (they deliberately want the full budget)
    and while speculative decoding is armed (it owns token manipulation);
  * forces the sequence at most once per thinking block (re-arms on a new start);
  * never starts mid-character: like llama.cpp's WAITING_UTF8 state, arming
    waits until the generated tail ends on a complete UTF-8 boundary;
  * never arms when the sequence would not fit with answer room to spare.

Sampling is eager (outside the decode CUDA graph), so the forced token is
written into next_tokens_gpu before the token_pool/CPU fan-out and the model
input and the API stream stay consistent.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Batch, Req

logger = init_logger(__name__)

# Injected as the model's own concluding thought before the end tag. An empty
# FREETOKEN_THINK_WRAP_MESSAGE forces the bare tag (llama.cpp's default).
DEFAULT_WRAP_MESSAGE = "\n\nI need to wrap up my thinking now and provide the final answer."


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


class ThinkWrapState:
    """Per-request thinking tracker + wrap-up force queue."""

    __slots__ = ("start_ids", "end_ids", "force_ids", "in_think", "hist", "maxlen",
                 "force_queue", "force_pos", "armed")

    def __init__(self, start_ids: list[int], end_ids: list[int], message_ids: list[int] | None = None):
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.force_ids = list(message_ids or []) + list(end_ids)
        self.in_think = False
        self.hist: list[int] = []
        self.maxlen = max(len(start_ids), len(end_ids), 1)
        self.force_queue: list[int] = []
        self.force_pos = 0
        self.armed = False

    def _tail(self, ids: list[int]) -> bool:
        n = len(ids)
        return n > 0 and len(self.hist) >= n and self.hist[-n:] == ids

    def observe(self, tok: int) -> None:
        self.hist.append(tok)
        if len(self.hist) > self.maxlen:
            del self.hist[0]
        if self._tail(self.end_ids):
            self.in_think = False
            self.armed = False
            self.force_queue = []
            self.force_pos = 0
        elif self._tail(self.start_ids):
            self.in_think = True
            self.armed = False
            self.force_queue = []
            self.force_pos = 0

    def next_force(self, remaining: int, reserve: int, utf8_ok: bool = True) -> int | None:
        # drain the queued wrap-up tokens first (message + multi-token tags)
        if self.force_pos < len(self.force_queue):
            tok = self.force_queue[self.force_pos]
            self.force_pos += 1
            return tok
        # arm so the FULL reserve survives the injection for the answer; the
        # strict lower bound keeps at least one answer token even if the
        # sequence alone would fit. utf8_ok=False delays arming one step.
        n = len(self.force_ids)
        if self.in_think and not self.armed and utf8_ok and n < remaining <= reserve + n:
            self.armed = True
            self.force_queue = list(self.force_ids)
            self.force_pos = 1
            return self.force_queue[0]
        return None


class ThinkWrap:
    """Controller shared by the scheduler drain (observe) and engine sample (force)."""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.enabled = _env_on("FREETOKEN_THINK_WRAP", "0")
        try:
            self.frac = max(0.0, float(os.environ.get("FREETOKEN_THINK_WRAP_RESERVE", "0.1")))
        except ValueError:
            self.frac = 0.1
        self.start_ids: list[int] = []
        self.end_ids: list[int] = []
        self.message_ids: list[int] = []
        # Opt-in (off by default): the built-in tags only match the standard think-tag
        # families and per-family tag auto-detection is not implemented yet, so other
        # families would silently no-op. Enable with FREETOKEN_THINK_WRAP=1 (plus START/END
        # for non-default tags). Tag detection below is the real gate: a model that never
        # emits the start tag never triggers.
        if self.enabled and getattr(config, "speculative", "none") != "none":
            self.enabled = False  # spec decoding owns token manipulation
        if self.enabled:
            self.start_ids = self._encode(
                tokenizer, os.environ.get("FREETOKEN_THINK_WRAP_START", "<think>"))
            self.end_ids = self._encode(
                tokenizer, os.environ.get("FREETOKEN_THINK_WRAP_END", "</think>"))
            self.message_ids = self._encode(
                tokenizer, os.environ.get("FREETOKEN_THINK_WRAP_MESSAGE", DEFAULT_WRAP_MESSAGE))
            if not self.start_ids or not self.end_ids or self.start_ids == self.end_ids:
                self.enabled = False
        if self.enabled:
            logger.info_rank0(
                "thinking wrap-up enabled: injecting a wrap-up message + end-of-thinking tag "
                f"when the remaining max_tokens budget falls to {self.frac:.0%} of the budget"
            )

    @staticmethod
    def _encode(tokenizer, text: str) -> list[int]:
        try:
            return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]
        except Exception:  # noqa: BLE001 -- a missing tag just disables the wrap-up
            return []

    def _active(self, req: "Req") -> bool:
        return self.enabled and not req.sampling_params.ignore_eos

    def _prompt_opens_think(self, req: "Req") -> bool:
        # Thinking templates usually pre-open the block (the prompt ends with the start
        # tag, possibly + a newline), so the model never emits the start tag itself.
        n = len(self.start_ids)
        ids = req.input_ids[: req.prompt_len]
        if ids.numel() < n:
            return False
        tail = ids[-(n + 2):].tolist()
        for skip in range(3):  # tolerate up to two trailing tokens (e.g. a newline)
            end = len(tail) - skip
            if end >= n and tail[end - n:end] == self.start_ids:
                return True
        return False

    def observe(self, req: "Req", tok: int) -> None:
        if not self._active(req):
            return
        st = getattr(req, "_think_wrap", None)
        if st is None:
            st = ThinkWrapState(self.start_ids, self.end_ids, self.message_ids)
            if self._prompt_opens_think(req):
                st.in_think = True
            req._think_wrap = st
        st.observe(tok)

    def reserve_for(self, req: "Req") -> int:
        return max(1, int(req.output_len * self.frac))

    def _tail_utf8_complete(self, req: "Req") -> bool:
        # llama.cpp waits for UTF-8 completion before forcing: byte-fallback tokens
        # can leave a partial character at the tail, and injecting there would split
        # it. A trailing U+FFFD from decoding the tail marks an incomplete character
        # (8 tokens cover the widest 4-byte character plus context).
        try:
            tail = req.input_ids[-8:].tolist()
            return not self.tokenizer.decode(tail).endswith("\ufffd")
        except Exception:  # noqa: BLE001 -- an unreadable tail must not block the wrap-up
            return True

    def apply(self, batch: "Batch", next_tokens_gpu) -> None:
        if not self.enabled or not batch.is_decode:
            return
        for i, req in enumerate(batch.reqs):
            if not self._active(req):
                continue
            st = getattr(req, "_think_wrap", None)
            if st is None:
                continue
            remaining = req.max_device_len - req.device_len
            reserve = self.reserve_for(req)
            # the boundary check only matters on a step that could arm the force
            utf8_ok = True
            if st.in_think and not st.armed and remaining <= reserve + len(st.force_ids):
                utf8_ok = self._tail_utf8_complete(req)
            force = st.next_force(remaining, reserve, utf8_ok)
            if force is not None:
                next_tokens_gpu[i] = force
