"""Piped prefill worker (FREETOKEN_PIPED_PREFILL) - the Strata PIPED_CONCURRENT port.

Prefill chunks run on a dedicated thread and a dedicated default-priority CUDA stream
while the scheduler's main loop keeps issuing decode batches on the engine's
high-priority stream. Decode never waits for the prompt: the GPU schedules replay
kernels ahead of the queued chunk work, and the two host threads overlap the launch
cost. This is the structural fix phase 1 (mixed batches) could not reach: a mixed step
advances riding decodes once per CHUNK (~3s here); a piped step advances them once per
REPLAY (~25ms).

The handoff is single-slot (Strata parity): one chunk in flight, and the next chunk of
a chunked prompt can only be prepared after the current one drains (its cached_len
advances there). The thread touches NO scheduler host state:

  * main thread owns: scheduling, _prepare_batch (allocation, radix match/lock,
    metadata, staging copies on the scheduler stream), the drain, aborts, rebuilds.
  * pipe thread owns: the forward launch itself - input gather, engine.forward_batch,
    the sampled-token write to token_pool, and its own batch's Req device counters
    (complete_one, disjoint from the running decodes the main loop reads).

Ordering is three events:
  prepare_done (scheduler stream)  -> the chunk's staging copies/pool rows are visible
  gate         (engine stream)     -> the decode step issued alongside this handoff has
                                      LAUNCHED; the chunk's kernel storm starts behind
                                      it, never delaying the replay's launch
  chunk_done   (prefill stream)    -> everything the chunk wrote is visible to the
                                      drain and to later pool reuse
A failed launch marks the pipe dead: the main loop drains the slot as an aborted chunk
and runs every later chunk on the old inline path.
"""
from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx

# FREETOKEN_PIPED_TIMEOUT_S: hard ceiling on one piped chunk's in-flight time. A
# rendezvous bug would otherwise leave a spinning collective pinning a GPU at 100%
# forever, invisible to the stall watchdog (the spinning rank is burning CPU).
# Crossing it crashes the scheduler on purpose: the supervisor restarts the engine.
_TIMEOUT_S = float(os.environ.get("FREETOKEN_PIPED_TIMEOUT_S", "600"))
# cap for a forced drain waiting our own worker to publish (divergence detection)
_FORCED_WAIT_S = float(os.environ.get("FREETOKEN_PIPED_FORCED_WAIT_S", "30"))

if TYPE_CHECKING:
    from .scheduler import Scheduler, ForwardInput


_SERIAL_DEBUG = __import__("os").environ.get("FREETOKEN_PIPED_SERIAL", "0") == "1"


class PrefillPiper:
    def __init__(self, scheduler: "Scheduler"):
        self.s = scheduler
        self.engine = scheduler.engine
        self.device = scheduler.device
        self.dead = False
        self.inflight = False  # handed off, not yet drained (slot not consumed)
        self.slot: "tuple[ForwardInput, object, BaseException | None] | None" = None
        self._item: "ForwardInput | None" = None
        # set by the worker while the forward runs: the abort handler must see the
        # in-flight batch during the whole launch->completion window, not only the
        # queued (pre-start) and done (post-slot) states around it
        self.running: "ForwardInput | None" = None
        self.handoff_ts: float = 0.0
        self.timeout_s = _TIMEOUT_S
        self._cv = threading.Condition()
        self._prepare_done = torch.cuda.Event()
        self._gate: torch.cuda.Event | None = None
        self.chunk_done = torch.cuda.Event()  # after the pool write, covers the whole chunk
        self.thread = threading.Thread(target=self._run, daemon=True, name="prefill-pipe")
        self.thread.start()

    # ------------------------------------------------------------- main-thread API

    def can_take(self) -> bool:
        """A fresh chunk may be handed off: nothing in flight, nothing undrained."""
        return not self.dead and not self.inflight and self.slot is None

    def wait_slot(self, timeout: float = 0.05) -> bool:
        """Block until the worker publishes its slot (or dies). The main loop uses this
        instead of spinning when the pipe is the only work left."""
        with self._cv:
            return self._cv.wait_for(
                lambda: self.slot is not None or self.dead, timeout
            )

    def handoff(self, forward_input: "ForwardInput") -> None:
        """Called on the main thread AFTER _prepare_batch staged this stream's work and
        AFTER record_gate(), so the pipe can order itself behind both."""
        self._prepare_done.record(self.s.stream)
        with self._cv:
            self._item = forward_input
            self.inflight = True
            self.handoff_ts = time.perf_counter()
            self._cv.notify()
        if _SERIAL_DEBUG:
            # bisect knob: keep the thread + stream plumbing, serialize the GPU.
            # If corruption vanishes under this, the bug is a concurrent-state race;
            # if it stays, the bug is in the piped ordering itself.
            while self.slot is None:
                self.wait_slot()
            self.chunk_done.synchronize()

    def record_gate(self) -> None:
        """Called on the main thread inside the engine-stream context after a batch was
        issued there: the pipe's kernels wait behind this launch, never ahead of it."""
        if self._gate is None:
            self._gate = torch.cuda.Event()
        self._gate.record(self.engine.stream)

    def poll(self, force: bool = False, strict: bool = False):
        """Consume the slot once the chunk's stream point passed, returning
        (forward_input, forward_output, error). None while still running.

        force=True is rank0's broadcast drain decision under TP: the worker must have
        published (launches rendezvous chunk-by-chunk with the peer pipe, so publish
        is bounded), and the drain's own copy_done.synchronize absorbs any remaining
        GPU skew - but the ITERATION on which scheduler state moves is rank-identical.

        strict=True (tp>1) makes that flag the only gate for a successful chunk: a rank
        whose own GPU finished EARLIER must not drain an iteration ahead of the
        broadcast, or it joins the chunk's request to its running set one step before
        the peer does and the next decode batch is composed differently on each rank -
        mismatched NCCL sequences, and the rank that launched first spins forever. An
        errored slot still drains at once: it ends the rank either way."""
        if self.slot is None:
            if not force:
                return None
            # force=True came from rank0's broadcast, but THIS rank has nothing
            # published: a scheduling divergence. Give our worker a bounded time to
            # catch up (publish is bounded while the peer pipe is healthy); beyond
            # that the ranks are desynchronized and only a restart re-aligns them -
            # never wait forever, that is how a wedged collective pins a GPU at 100%.
            deadline = time.perf_counter() + _FORCED_WAIT_S
            while self.slot is None and not self.dead:
                if time.perf_counter() > deadline:
                    raise RuntimeError(
                        "piped drain divergence: rank0 signaled a ready chunk this "
                        "rank never received (scheduling desynchronized); restarting"
                    )
                self.wait_slot()
            if self.slot is None:
                return None
        out = self.slot
        if out[2] is None and not force and (strict or not self.chunk_done.query()):
            return None
        self.slot = None
        self.inflight = False
        self.handoff_ts = 0.0
        return out

    def stalled(self) -> bool:
        return (
            self.inflight
            and self.handoff_ts > 0.0
            and time.perf_counter() - self.handoff_ts > self.timeout_s
        )

    def inflight_batch(self):
        """The batch whose forward is (or was until drain) on the prefill stream - the
        abort handler's in-flight membership test needs it next to _last_data."""
        for holder in (self._item, self.running, None if self.slot is None else self.slot[0]):
            if holder is not None:
                return holder.batch
        return None

    # ------------------------------------------------------------- worker thread

    @torch.inference_mode()
    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        # the engine Context routes this thread's forward to its own batch slot, and the
        # distributed layer routes its collectives to the pipe's own comm (tp>1); at tp1
        # there are no collectives to route, so the register call is a no-op marker
        get_global_ctx().register_pipe_thread()
        from freetoken.distributed import register_pipe_comm_thread

        register_pipe_comm_thread()
        while True:
            with self._cv:
                while self._item is None and not self.dead:
                    self._cv.wait()
                if self.dead:
                    return
                item, self._item = self._item, None
                self.running = item
            fout, err = None, None
            try:
                with torch.cuda.stream(self.engine.prefill_stream):
                    self.engine.prefill_stream.wait_event(self._prepare_done)
                    if self._gate is not None:
                        self.engine.prefill_stream.wait_event(self._gate)
                    try:
                        fout = self.s._issue_piped_forward(item)
                    finally:
                        # record even on a raise: the discard path waits this event to
                        # free resources behind whatever kernels DID get launched
                        self.chunk_done.record(self.engine.prefill_stream)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the main loop
                self.dead = True
                err = exc
            self.running = None
            with self._cv:
                self.slot = (item, fout, err)
                self._cv.notify_all()

    def shutdown(self) -> None:
        with self._cv:
            self.dead = True
            self._cv.notify()
