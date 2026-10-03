"""Requests landing at arbitrary points of the scheduler loop.

Users do not arrive on a beat. Two prompts can already be waiting in the same
receive window, a lone one can drop in between two decode steps of somebody
else, or land while a long prompt is mid-chunk with the page pool nearly swept.
Each of those timings yields a different admission set, a different phase for
the next forward, and a different set of locked pages -- and the client outside
only ever sees the replies, never the branch that produced them.

Each test replays a seeded arrival trace through a real scheduler step:
``_process_one_msg`` admits, ``_schedule_next_batch`` picks the phase,
``PrefillManager`` admits and chunks, ``_process_last_data`` drains and frees.
Only the engine seam is faked -- ``_prepare_batch`` keeps the page allocation
the real one performs ahead of any engine work and drops the graph and
attention metadata, ``_forward`` replaces the kernel with the token advance its
bookkeeping describes. Non-overlap scheduling is what makes the timings real:
messages are always taken between iterations, so an arrival lands wherever the
trace says it does. Whatever the order, the trace must hold to:

* conservation -- every arrival closed out by exactly one terminal reply (a
  finished ``DetokenizeMsg``, the prompt-too-long ``ErrorReplyMsg``, or the
  abort ack), the accounting signal fired once, and the pool closed behind it:
  every page back, every page-table row released, ``check_integrity`` clean
  whenever the loop falls idle;
* one forward never costing more than ``max_extend_tokens``, nor more rows than
  the page table has;
* nobody already decoding waiting more than ``prefill_decode_interval`` steps in
  a row for their next token, however much fresh work landed while they waited.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import (
    AbortBackendMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

MAX_RUNNING = 4
WIDTH = 192          # page-table row width: longest prompt + output this soak writes
MAX_SEQ_LEN = WIDTH  # what the engine reports as its servable context
MAX_PROMPT = 40      # fits inside WIDTH next to MAX_OUTPUT tokens, always
MAX_OUTPUT = 6
TOKEN_SPACE = 64     # ids stay clear of 0, the pad / sampled-token sentinel

# Whatever the seed draws, the trace contains one of each: a prompt too long for
# the context, one too long for a single chunk, and one the client abandons while
# it is still in flight.
OVERSIZE_UID = 3
LONG_UID = 5
ABORT_UID = 7
SEEDS = [0, 7, 23, 133]


def _scheduler(num_pages: int, interval: int, budget: int) -> Scheduler:
    """A bare Scheduler over CPU-built managers; the caller fakes the engine seam."""
    page_table = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32)
    s = Scheduler.__new__(Scheduler)
    s.cache_manager = CacheManager(num_pages, 1, page_table, "radix")
    s.table_manager = TableManager(max_running_reqs=MAX_RUNNING, page_table=page_table)
    s.decode_manager = DecodeManager(page_size=1)
    s.prefill_manager = PrefillManager(s.cache_manager, s.table_manager, s.decode_manager)
    s.finished_reqs = set()
    s.eos_token_ids = set()  # 0 is the token the fake forward samples
    s.toolcall_anchor_id = None
    s.device = torch.device("cpu")
    s.config = SimpleNamespace(page_size=1, prefill_decode_interval=interval, speculative="none")
    s.status_reporter = SimpleNamespace(report_batch=lambda *_, **__: None)
    s.engine = SimpleNamespace(
        run_pending_host_fill=lambda: None, encoder_cache=None, max_seq_len=MAX_SEQ_LEN,
        linear_state_pool=None, clock_keeper=SimpleNamespace(notify=lambda: None),
    )
    s.sent: list = []
    s.send_result = s.sent.extend
    s._kv_usage_pages = s.cache_manager.page_usage
    s._mamba_slot_usage = lambda: None
    s._swa_token_usage = lambda: None
    s._gpu_mem_bytes = lambda: 0
    s._match_stop_str = lambda _req: None
    # The wall-clock interleave guards are unit-tested against a fake clock in
    # test_scheduler_fairness.py; the count-based guard is the one bounded here.
    s._account_prefill_debt = lambda _batch: None
    s._observe_chunk_time = lambda _batch: None
    s._prefill_debt_s = s._prefill_debt = s._chunk_target_s = 0.0
    s._chunk_ema_spt = None
    s._prefill_streak = 0
    s.prefill_budget = budget
    s._pending_abort_acks = set()
    s._abort_tombstones = {}
    s._last_data = None  # non-overlap: the loop drains whatever it has just issued
    s._free_req_resources = lambda req: Scheduler._free_req_resources(s, req)
    s._adaptive_prefill_budget = lambda: Scheduler._adaptive_prefill_budget(s)
    s._report_prompt_admissions = lambda batch: Scheduler._report_prompt_admissions(s, batch)
    s._restore_linear_states = lambda batch: Scheduler._restore_linear_states(s, batch)
    s._flush_abort_acks = lambda: Scheduler._flush_abort_acks(s)
    return s


class _Soak:
    """One seeded arrival trace over one scheduler, plus the promises it keeps."""

    def __init__(self, num_pages: int, interval: int, budget: int):
        assert interval > 0, "the interleave guard is what this soak bounds"
        self.s = _scheduler(num_pages, interval, budget)
        self.cm, self.tm = self.s.cache_manager, self.s.table_manager
        self.dm, self.pm = self.s.decode_manager, self.s.prefill_manager
        self.s._prepare_batch = self._prepare_batch
        self.s._forward = self._forward
        self.interval, self.budget = interval, budget
        self.arrived: list[int] = []
        self.terminal: dict[int, int] = {}
        self.admitted: dict[int, int] = {}
        self.tokens: dict[int, int] = {}
        self.gap: dict[int, int] = {}
        self.aborted: list[int] = []
        self.open: list[int] = []
        self.last_prompt: torch.Tensor | None = None
        self.mark = 0  # watermark into s.sent, so replies of a step are counted once
        self.steps = self.idle_steps = self.prefill_steps = self.decode_steps = 0
        self.chunked_steps = self.deferred_steps = self.max_burst = self.max_gap = 0
        self.max_tokens = self.max_rows = 0

    # -- the two engine seams -------------------------------------------------

    def _prepare_batch(self, batch):
        """The allocation half of the real _prepare_batch, with the per-step bounds."""
        tokens = sum(req.extend_len for req in batch.reqs)
        if batch.is_prefill:
            self.max_tokens = max(self.max_tokens, tokens)
            assert tokens <= self.budget, f"step {self.steps}: {tokens} tokens > budget {self.budget}"
        self.max_rows = max(self.max_rows, batch.size)
        assert batch.size <= MAX_RUNNING, f"step {self.steps}: {batch.size} rows > page table"
        assert len({r.uid for r in batch.reqs}) == batch.size, "one request got two rows"
        if batch.is_decode:
            for req in batch.reqs:
                req.decode_batch_idx += 1
        self.cm.allocate_paged(batch.reqs)
        return SimpleNamespace(batch=batch)

    def _forward(self, forward_input):
        """The token advance that engine.forward_batch bookkeeping describes."""
        batch = forward_input.batch
        for req in batch.reqs:
            req.complete_one()
        self.dm.filter_reqs(batch.reqs)  # Scheduler._forward joins the batch to running
        # Token 0 is never EOS and no stop string is set, so a request only ever
        # ends on its own output budget.
        return SimpleNamespace(
            next_tokens_gpu=None,
            next_tokens_cpu=torch.zeros(batch.size, dtype=torch.int32),
            copy_done_event=SimpleNamespace(synchronize=lambda: None),
            spec=None,
        )

    # -- one iteration of normal_loop ----------------------------------------

    def arrive(self, uid: int, input_ids: torch.Tensor, max_tokens: int) -> None:
        msg = UserMsg(uid=uid, input_ids=input_ids,
                      sampling_params=SamplingParams(max_tokens=max_tokens))
        Scheduler._process_one_msg(self.s, msg)
        self.arrived.append(uid)
        self.open.append(uid)

    def abort(self, uid: int) -> None:
        Scheduler._process_one_msg(self.s, AbortBackendMsg(uid=uid))
        self.aborted.append(uid)
        self.open.remove(uid)

    def step(self):
        forward_input = Scheduler._schedule_next_batch(self.s)
        mark = self.mark  # everything since the last step, arrivals and aborts included
        if forward_input is not None:
            self.s._restore_linear_states(forward_input.batch)
            fout = self.s._forward(forward_input)
            self.s.engine.run_pending_host_fill()
            Scheduler._process_last_data(self.s, (forward_input, fout))
        else:
            self.idle_steps += 1
        self.s._flush_abort_acks()  # normal_loop acks even on a step that ran nothing
        self.steps += 1
        self._account(self.s.sent[mark:], forward_input)
        self.mark = len(self.s.sent)
        if not self.pm.runnable and not self.dm.runnable:
            self.cm.check_integrity()  # the run_when_idle contract: nothing holds pages
        return forward_input

    def _account(self, new, forward_input) -> None:
        if forward_input is not None:
            batch = forward_input.batch
            if batch.is_prefill:
                self.prefill_steps += 1
                self.chunked_steps += any(isinstance(r, ChunkedReq) for r in batch.reqs)
            else:
                self.decode_steps += 1
                self.deferred_steps += bool(self.pm.pending_list)
        served: set[int] = set()
        for msg in new:
            if isinstance(msg, DetokenizeMsg):
                self.tokens[msg.uid] = self.tokens.get(msg.uid, 0) + 1
                served.add(msg.uid)
                if msg.finished:
                    self.terminal[msg.uid] = self.terminal.get(msg.uid, 0) + 1
            elif isinstance(msg, PromptAdmittedMsg):
                self.admitted[msg.uid] = self.admitted.get(msg.uid, 0) + 1
            elif isinstance(msg, ErrorReplyMsg):
                self.terminal[msg.uid] = self.terminal.get(msg.uid, 0) + 1
        running = {req.uid for req in self.dm.running_reqs}
        self.open = [uid for uid in self.open if uid not in self.terminal]
        self.gap = {uid: g for uid, g in self.gap.items() if uid in running}
        for uid in running:
            if uid in served:
                self.gap[uid] = 0  # this step's forward answered it, prefill or decode
            else:
                self.gap[uid] = self.gap.get(uid, 0) + 1
                self.max_gap = max(self.max_gap, self.gap[uid])
                assert self.gap[uid] <= self.interval, (
                    f"step {self.steps}: request {uid} waited {self.gap[uid]} steps for its "
                    f"next token (prefill_decode_interval={self.interval})"
                )

    # -- the arrival process --------------------------------------------------

    def _prompt(self, rng: random.Random, length: int | None = None, oversize: bool = False):
        if oversize:
            n = rng.randint(WIDTH, WIDTH + 8)  # no room left for a single output token
        elif length is None:
            n = rng.randint(MAX_PROMPT // 5, MAX_PROMPT)
        else:
            n = length
        ids: list[int] = []
        if not oversize and self.last_prompt is not None and rng.random() < 0.5:
            shared = rng.randint(1, min(n - 1, len(self.last_prompt)))
            ids = self.last_prompt[:shared].tolist()  # a prefix somebody else already paid for
        ids += [rng.randrange(1, TOKEN_SPACE) for _ in range(n - len(ids))]
        self.last_prompt = torch.tensor(ids, dtype=torch.int32)
        return self.last_prompt

    def run(self, rng: random.Random, n_requests: int, max_steps: int,
            abort_p: float = 0.0, burst_gap: int = 4) -> None:
        next_at, uid, landed, force_abort = 0, 0, 0, False
        while True:
            burst = 0
            while landed < n_requests and self.steps >= next_at:
                uid += 1
                landed += 1
                burst += 1
                forced = uid == ABORT_UID
                if uid == OVERSIZE_UID:
                    ids, out = self._prompt(rng, oversize=True), 1
                elif uid == LONG_UID or forced:  # long enough to need more than one chunk
                    ids, out = self._prompt(rng, length=MAX_PROMPT), MAX_OUTPUT
                else:
                    ids, out = self._prompt(rng), rng.randint(1, MAX_OUTPUT)
                self.arrive(uid, ids, out)
                force_abort = forced
                # the next arrival either queues up behind this one or waits a while
                next_at = self.steps if uid == 1 else self.steps + rng.randint(0, burst_gap)
            self.max_burst = max(self.max_burst, burst)
            forcing = force_abort and self.steps >= 2 and ABORT_UID in self.open
            if self.open and (forcing or (abort_p > 0 and rng.random() < abort_p)):
                # A client hanging up on a request still in flight. The forced one is
                # the long prompt, so every seed hits the race and not just most.
                force_abort = False
                self.abort(ABORT_UID if forcing else self.open[rng.randrange(len(self.open))])
            forward_input = self.step()
            if (forward_input is None and landed == n_requests
                    and not self.pm.pending_list and not self.dm.running_reqs):
                return
            if self.steps > max_steps:
                raise AssertionError(
                    f"trace stuck after {self.steps} steps: {len(self.pm.pending_list)} pending, "
                    f"{len(self.dm.running_reqs)} running, {len(self.terminal)}/{len(self.arrived)} "
                    f"closed out, {len(self.s._pending_abort_acks)} abort acks queued"
                )

    def finish(self) -> None:
        assert not self.pm.pending_list and not self.dm.running_reqs
        assert self.tm.available_size == MAX_RUNNING, "page-table rows did not come back"
        self.cm.check_integrity()
        missing = set(self.arrived) - set(self.terminal)
        assert not missing, f"{sorted(missing)} never got a terminal reply"
        assert all(n == 1 for n in self.terminal.values()), self.terminal
        assert set(self.admitted) <= set(self.arrived)
        assert all(n == 1 for n in self.admitted.values()), self.admitted
        assert set(self.aborted) <= set(self.arrived)
        assert not self.s._pending_abort_acks and self.s._last_data is None


def _soak(seed: int, *, num_pages: int, interval: int = 8, budget: int = 24,
          n_requests: int = 18, abort_p: float = 0.0, max_steps: int = 600) -> _Soak:
    soak = _Soak(num_pages=num_pages, interval=interval, budget=budget)
    soak.run(random.Random(seed), n_requests=n_requests, abort_p=abort_p, max_steps=max_steps)
    soak.finish()
    return soak


@pytest.mark.parametrize("seed", SEEDS)
def test_every_arrival_is_closed_out_exactly_once(seed):
    """Random timings and sizes over a generous pool: replies are a bijection."""
    soak = _soak(seed, num_pages=1024)
    # the trace really did hold the interesting cases, whatever the seed drew
    assert soak.max_burst >= 2, "no two requests ever landed in one receive window"
    assert soak.chunked_steps, "no request ever had to be chunked"
    assert soak.max_tokens <= soak.budget and soak.max_rows <= MAX_RUNNING
    assert min(soak.tokens.values()) >= 1  # every served uid said something back


@pytest.mark.parametrize("seed", SEEDS)
def test_arrivals_never_stall_a_running_decode_past_the_interval(seed):
    """A prompt landing between two decode steps cannot push anyone's ITL out."""
    soak = _soak(seed, num_pages=256, interval=2, budget=16)
    assert soak.decode_steps and soak.prefill_steps > soak.decode_steps
    assert soak.deferred_steps, "prefill never queued up behind a running decode"
    # The guard is what pins this to the bound: run the same trace with the
    # interleave disabled in _schedule_next_batch and the gap reaches 5 to 7.
    assert soak.max_gap == soak.interval, "arrivals never contended with a running decode"


@pytest.mark.parametrize("seed", SEEDS)
def test_arrivals_against_a_nearly_swept_pool_still_drain(seed):
    """Prompts arriving while the pool is swept: defer, evict, backfill, drain."""
    soak = _soak(seed, num_pages=MAX_PROMPT + MAX_OUTPUT + 24, budget=16)
    assert soak.deferred_steps, "the pool was never tight enough to defer an admission"
    assert len(soak.arrived) == len(soak.terminal)


@pytest.mark.parametrize("seed", SEEDS)
def test_arrivals_abandoned_mid_flight_leave_nothing_behind(seed):
    """A client giving up at a random step is still accounted for exactly once."""
    soak = _soak(seed, num_pages=256, abort_p=0.4)
    assert soak.aborted, "the trace never abandoned anything"
    assert all(uid in soak.terminal for uid in soak.aborted)


