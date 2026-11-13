"""Thinking-budget wrap-up: conclude reasoning before max_tokens runs out.

Port of llama.cpp's reasoning-budget idea, keyed to the request's remaining
max_tokens budget instead of a separate counter: while a request is inside a
thinking block and its remaining output budget drops to the reserve, the next
token(s) are forced to the end-of-thinking tag so the model spends what is left
on the final answer (or tool call) instead of being cut off mid-thought.

Safety boundaries:
  * only acts INSIDE the thinking block (start tag seen, end tag not yet), so
    post-thinking content and tool calls are never rewritten;
  * skipped for ignore_eos requests (they deliberately want the full budget)
    and while speculative decoding is armed (it owns token manipulation);
  * forces the end tag at most once per thinking block (re-arms on a new start).

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


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


class ThinkWrapState:
    """Per-request thinking tracker + end-tag force queue."""

    __slots__ = ("start_ids", "end_ids", "in_think", "hist", "maxlen",
                 "force_queue", "force_pos", "armed")

    def __init__(self, start_ids: list[int], end_ids: list[int]):
        self.start_ids = start_ids
        self.end_ids = end_ids
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

    def next_force(self, remaining: int, reserve: int) -> int | None:
        # drain the queued end-tag tokens first (multi-token tags)
        if self.force_pos < len(self.force_queue):
            tok = self.force_queue[self.force_pos]
            self.force_pos += 1
            return tok
        if self.in_think and not self.armed and 0 < remaining <= reserve:
            self.armed = True
            self.force_queue = list(self.end_ids)
            self.force_pos = 1
            return self.force_queue[0]
        return None


class ThinkWrap:
    """Controller shared by the scheduler drain (observe) and engine sample (force)."""

    def __init__(self, tokenizer, config):
        self.enabled = _env_on("FREETOKEN_THINK_WRAP")
        try:
            self.frac = max(0.0, float(os.environ.get("FREETOKEN_THINK_WRAP_RESERVE", "0.1")))
        except ValueError:
            self.frac = 0.1
        self.start_ids: list[int] = []
        self.end_ids: list[int] = []
        # No reasoning_parser gate here: the engine config does not carry it, and the
        # tag detection below is the real gate (a model that never emits the start tag
        # never triggers). Non-thinking deployments can opt out via FREETOKEN_THINK_WRAP=0.
        if self.enabled and getattr(config, "speculative", "none") != "none":
            self.enabled = False  # spec decoding owns token manipulation
        if self.enabled:
            self.start_ids = self._encode(
                tokenizer, os.environ.get("FREETOKEN_THINK_WRAP_START", "<think>"))
            self.end_ids = self._encode(
                tokenizer, os.environ.get("FREETOKEN_THINK_WRAP_END", "</think>"))
            if not self.start_ids or not self.end_ids or self.start_ids == self.end_ids:
                self.enabled = False
        if self.enabled:
            logger.info_rank0(
                "thinking wrap-up enabled: forcing the end-of-thinking tag when the "
                f"remaining max_tokens budget falls to {self.frac:.0%} of the budget"
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
            st = ThinkWrapState(self.start_ids, self.end_ids)
            if self._prompt_opens_think(req):
                st.in_think = True
            req._think_wrap = st
        st.observe(tok)

    def reserve_for(self, req: "Req") -> int:
        return max(1, int(req.output_len * self.frac))

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
            force = st.next_force(remaining, self.reserve_for(req))
            if force is not None:
                next_tokens_gpu[i] = force