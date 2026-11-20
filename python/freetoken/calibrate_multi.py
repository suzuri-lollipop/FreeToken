"""Tune the concurrent-serving knobs (FREETOKEN_PIPED_PREFILL / MIXED / ADMIT_*) for THIS PC.

The single-stream calibrator (`calibrate_lookup`) makes one request fast. This one makes
MANY requests fast together: it maximizes the aggregate tok/s of the profile's FULL
concurrent batch (--max-running-requests), and keeps a long prefill from freezing the
running decodes. The mechanism is machine-dependent in exactly the way lookup's width
is -- chunk cost, PCIe share between prefill and decode expert streaming, and the stream
priorities all decide whether piping pays here -- so the tool measures the real server
on this PC instead of trusting the defaults measured on one box.

Arms are probed at the server's PRODUCTION sampling by default (the request omits
temperature, so sampling_defaults=model and the high-temperature guard apply exactly as
they do for real traffic); --temperature 0.0 switches to deterministic greedy, whose
lower variance suits a re-run that must diff against an older report.

Every arm answers the SAME three workloads:

  - two regime probes at the profile's FULL --max-running-requests, run BEFORE the
    staggered workload: a decode-only probe (one short request per running slot,
    launched together, fixed answers, no aggressor) and a short-interleave probe (the
    same slots with ~1024-token prompts admitted 5 s apart -- the launcher bench's gap
    -- with 1024-token answers). Their MEAN aggregate tok/s is the selection score:
    the batch the profile actually serves, steady decode and mixed short-chat traffic
    alike;
  - a staggered workload for the stall guard: `victims` short requests (a 128-token
    prompt, a fixed 256-token answer) launched 2 s apart, then ONE long aggressor (a
    16384-token prompt) 6 s in, while the victims are mid-decode. It measures the worst
    inter-token gap of the victims AFTER the aggressor is admitted (the stall a human
    feels) and the aggressor's TTFT. Victims stay capped at 3: the aggressor stall is
    per-stream, and the full-batch cost is the regime probes' job;
  - the arms: `none` (the inline prefill/decode interleave), `mixed` (decode rows ride
    the chunk's forward, FREETOKEN_MIXED_PREFILL_TOKENS=1024), `piped` (chunks run on a
    worker thread and a side stream while decode replays keep the high-priority engine
    stream), and `piped-sf` (piping plus short-first admission).

A candidate wins only when it adds at least TPS_MIN_GAIN mean regime tok/s over the
baseline AND does not worsen the worst staggered stall by more than MULTI_STALL_TOL.
Each regime also keeps a floor of the baseline's tok/s (DECODE_TPS_LOSS_MAX /
INTERLEAVE_TPS_LOSS_MAX), so an arm cannot buy a mean win by regressing one regime.
The staggered wall's own tok/s is reported but never selects: it is aggressor-dominated
and capped at 3 victims, so it does not measure the batch the profile serves.

Why the probes select instead of the staggered wall: piped prefill structurally disables
the decode fetch overlap (single-writer staging) and, under TP, the shm one-shot
all-reduce -- on a fast-link rig that cost lands on EVERY decode step -- and even where
decode is neutral, sharing the link with short chunks costs the interleaved regime
(reported: a 200+ tok/s production rig dropped below 200 after piped won a stall-first
calibration whose staggered wall had shown +17.7% tok/s). The spec and MoE dimensions
carry the same score and guards.

A second dimension compares speculative backends (--spec, default auto = mtp + dflash;
dflash needs a draft checkpoint). Each spec arm runs with prefill off and wins on the
same mean regime tok/s (>= SPEC_MIN_GAIN) without worsening the worst stall by more than
SPEC_STALL_TOL. The winner is then stacked on the prefill winner in a combo arm: under
piped the engine refuses speculation (spec verify shares the PLE/plan staging with the
pipe's chunk -- Strata lifts this with STRATA_PIPED_MTP, this port has not yet), so the
combo verifies that empirically and the report records why the two cannot stack.

A third dimension compares MoE serving strategies (--moe, default auto = offload +
hybrid). The profile's own --moe-strategy is stripped so every arm starts from the same
auto baseline; each strategy arm runs with prefill and speculation off and wins on the
same contract as spec (>= MOE_MIN_GAIN tok/s, worst stall within MOE_STALL_TOL). hybrid
computes part of each step's expert misses on the CPU instead of fetching them over
PCIe, so it pays on boxes whose CPU MoE bandwidth beats the link -- exactly what
`ft bench bw --model <ckpt> --tp-size N` measures; without a matching profile row the
hybrid arm still runs but keeps a fetch cap of 1, and the report says so. The winner's
flags go into the report's chosen.moe.argv and the generated script, and a combo arm
verifies it stacks on the prefill/spec winners before it is applied.

Scoring uses the server's own measurement when available: --enable-timings-report plus
timings_per_token gives per-chunk engine-side predicted_ms (the inter-token gap without
HTTP/client jitter), and continuous_usage_stats gives the exact token count even when a
chunk carries several tokens. FREETOKEN_LATENCY_OUT is collected per arm as a diagnostic.
Client arrival times remain the fallback and are reported alongside.

Arm validity is read from the arm's own server log (the "piped prefill enabled" line,
the [chunktime] mix= markers, the "piping disabled" fallback): an arm whose mechanism
never activated is EXCLUDED, not silently scored as if it did.

The output is the same pair as the lookup tool: a JSON report and an executable script,
with the winner's full knob set (explicit zeros included) in `env`, written atomically.
Feed it an existing launch config to keep the profile's argv:

    python -m freetoken.calibrate_multi --config freetoken-<model>.json

The server must not already be running on the chosen port; this tool owns it for the
duration. Expect ~25 min full (one engine start per arm, two regime probes plus the
staggered workload per rep), ~8 min --quick.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import shlex

from freetoken.calibrate_lookup import (
    DEFAULT_PORT,
    REQUEST_TIMEOUT,
    ROOT,
    Server,
    ServerError,
    WARMUP_TIMEOUT,
    _arg_value,
    _atomic_write,
    _on_sigterm,
    _port_free,
    _quote_env,
    _sanitize,
    ensure_distributed_timeout,
    hardware,
    spec_from_args,
)

MULTI_KEYS = (
    "FREETOKEN_PIPED_PREFILL",
    "FREETOKEN_MIXED_PREFILL_TOKENS",
    "FREETOKEN_ADMIT_SHORT_FIRST",
    "FREETOKEN_ADMIT_MAX_WAIT_S",
)
ARMS: list[tuple[str, dict[str, str]]] = [
    ("none", {}),
    ("mixed", {"FREETOKEN_MIXED_PREFILL_TOKENS": "1024",
               "FREETOKEN_ADMIT_SHORT_FIRST": "1024"}),
    ("piped", {"FREETOKEN_PIPED_PREFILL": "1"}),
    ("piped-sf", {"FREETOKEN_PIPED_PREFILL": "1",
                  "FREETOKEN_ADMIT_SHORT_FIRST": "1024"}),
]
# Selection score: the mean aggregate tok/s of the two full-concurrency regime probes
# (regime_tps). A prefill arm must add at least this much over the baseline -- the same
# 3% bar as the spec/MoE dimensions, above the probes' rep-median noise.
TPS_MIN_GAIN = 0.03
# ...and must not worsen the worst staggered stall by more than this (the spec/MoE bar):
# throughput is the objective, but an arm that freezes the running streams is not a win.
MULTI_STALL_TOL = 0.1
# decode-only probe floor: a winner keeps at least this share of the baseline's
# concurrent decode tok/s. 5% covers probe noise; a real fast-path loss (piped's
# fetch-overlap / shm-AR disable) is larger and must reject the arm, not pass it.
DECODE_TPS_LOSS_MAX = 0.05
# short-interleave probe floor: same contract for the staggered short-prompt regime
# (the launcher bench's shape). Even where pure decode is neutral, sharing the link
# with concurrent chunks costs here; measured -6% on the 2x gen3 dev rig.
INTERLEAVE_TPS_LOSS_MAX = 0.05
TIE_S = 0.15
STALL_EVENT_S = 0.5  # the launcher bench's stall threshold; keep the two in step

# Speculative-decoding dimension. Only backends this engine actually registers are
# offered (spec.PROPOSERS: mtp, dflash); "lookup" is not a live backend in this fork.
SPEC_ALGOS = ("mtp", "dflash")
SPEC_MIN_GAIN = 0.03   # a spec arm must add >=3% aggregate tok/s over the no-spec arm
SPEC_STALL_TOL = 0.1   # ...while worsening the worst stall by at most this many seconds
SPEC_COMBO_TPS_TOL = 0.02  # a prefill+spec combo keeps the spec win within this fraction
# every flag that arms a speculative backend, both spellings, stripped so the tool owns
# the dimension (the profile's own --speculative-* would otherwise leak into every arm)
SPEC_FLAGS = (
    "--speculative", "--speculative-algorithm", "--spec-algo",
    "--speculative-num-steps", "--spec-num-steps",
    "--speculative-draft-model-path", "--spec-draft-model",
    "--speculative-token-map", "--spec-token-map",
)

# MoE-strategy dimension. Only the offload family is offered: 'fused' needs every
# expert resident in VRAM (a wrong guess is a weight-load OOM, not a slower arm) and
# 'cpu' computes EVERYTHING on the CPU (a different serving regime, not a miss split).
MOE_STRATEGIES = ("offload", "hybrid")
MOE_MIN_GAIN = 0.03    # same contract as spec: >=3% aggregate tok/s over the base arm
MOE_STALL_TOL = 0.1    # ...without worsening the worst stall by more than this
MOE_COMBO_TPS_TOL = 0.02  # stacking on the other winners keeps the win within this
# both spellings of the strategy flag, stripped so the tool owns the dimension
MOE_FLAGS = ("--moe-strategy", "--moe-backend")


def arms_for(spec: dict, smoke: bool = False) -> list[tuple[str, dict[str, str], str]]:
    """The arms to MEASURE on this config: all four on any TP. (Piping under TP runs
    the pipe's collectives on a dedicated torch NCCL group; the earlier TP1-only
    refusal was chasing a PLE native-store race, not a comm deadlock.)"""
    out = []
    for label, env in ARMS:
        if smoke and label not in ("none", "piped"):
            continue
        out.append((label, env, ""))
    return out


def winner_env(label: str) -> dict[str, str]:
    """The full knob set for one arm: explicit zeros, so applying a winner also clears
    the loser's values out of a launch config that carried them."""
    env = {k: "0" for k in MULTI_KEYS}
    env["FREETOKEN_ADMIT_MAX_WAIT_S"] = "30"
    env.update(dict(ARMS)[label])
    return env


def argv_with_port(argv, port: int) -> list[str]:
    """Rewrite --port (both forms) and touch nothing else. Unlike the lookup tool, a
    multi arm must run under the profile's real --speculative config: that config is
    exactly what the arm-validity check is allowed to reject, so stripping it here
    would measure a server the profile never starts."""
    out, items = [], [str(x) for x in argv]
    i = 0
    while i < len(items):
        t = items[i]
        if t == "--port":
            i += 2
            continue
        if t.startswith("--port="):
            i += 1
            continue
        out.append(t)
        i += 1
    return out + ["--port", str(int(port))]


def strip_spec_args(argv) -> list[str]:
    """Drop every speculative-backend flag (both spellings, value or = form) so the
    calibration owns the spec dimension instead of inheriting the profile's."""
    out: list[str] = []
    items = [str(x) for x in argv]
    i = 0
    while i < len(items):
        t = items[i]
        if t in SPEC_FLAGS:
            i += 2
            continue
        if any(t.startswith(f + "=") for f in SPEC_FLAGS):
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def spec_argv(algo: str, draft_path: str | None = None,
              num_steps: int | None = None) -> list[str]:
    """The server flags that arm one speculative backend (mtp drafts from the target;
    dflash needs its own draft checkpoint)."""
    if algo == "none":
        return []
    out = ["--speculative-algorithm", algo]
    if algo == "dflash":
        if not draft_path:
            raise ValueError("dflash needs --spec-draft-path (a DFlash 2 draft checkpoint)")
        out += ["--speculative-draft-model-path", str(draft_path)]
    if num_steps:
        out += ["--speculative-num-steps", str(int(num_steps))]
    return out


def spec_backends_for(requested: str, draft_path: str | None) -> list[tuple[str, str]]:
    """Resolve --spec into (algo, skip_reason) pairs. 'auto' offers every registered
    backend; dflash is skipped without a draft checkpoint rather than failing the run."""
    if requested.strip().lower() in ("none", "off", ""):
        return []
    if requested.strip().lower() == "auto":
        algos = list(SPEC_ALGOS)
    else:
        algos = [a.strip().lower() for a in requested.split(",") if a.strip()]
        for a in algos:
            if a not in SPEC_ALGOS:
                raise ValueError(f"--spec: unknown backend {a!r} (have: {', '.join(SPEC_ALGOS)})")
    out = []
    for a in algos:
        if a == "dflash" and not draft_path:
            out.append((a, "no DFlash 2 draft checkpoint (--spec-draft-path)"))
        else:
            out.append((a, ""))
    return out


def strip_moe_args(argv) -> list[str]:
    """Drop the strategy flag (both spellings, value or = form) so the calibration
    owns the MoE dimension instead of inheriting the profile's."""
    out: list[str] = []
    items = [str(x) for x in argv]
    i = 0
    while i < len(items):
        t = items[i]
        if t in MOE_FLAGS:
            i += 2
            continue
        if any(t.startswith(f + "=") for f in MOE_FLAGS):
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def moe_argv(strategy: str) -> list[str]:
    """The server flags that arm one MoE strategy. Explicit even for offload: the
    winner must survive a later 'auto' resolution change, and an explicit offload
    clears a stale hybrid out of the launch argv."""
    return ["--moe-strategy", strategy]


def moe_strategies_for(requested: str) -> list[tuple[str, str]]:
    """Resolve --moe into (strategy, skip_reason) pairs. 'auto' offers the offload
    family; a dense model or an unsupported expert format fails its arm at server
    start and is excluded with the reason (the spec dimension's contract)."""
    if requested.strip().lower() in ("none", "off", ""):
        return []
    if requested.strip().lower() == "auto":
        strategies = list(MOE_STRATEGIES)
    else:
        strategies = [s.strip().lower() for s in requested.split(",") if s.strip()]
        for s in strategies:
            if s not in MOE_STRATEGIES:
                raise ValueError(f"--moe: unknown strategy {s!r} (have: {', '.join(MOE_STRATEGIES)})")
    return [(s, "") for s in strategies]


def argv_with_timings(argv) -> list[str]:
    """Add --enable-timings-report once (the server-side gate for per-chunk timings)."""
    items = [str(x) for x in argv]
    if any(t == "--enable-timings-report" or t.startswith("--enable-timings-report=")
           for t in items):
        return items
    return items + ["--enable-timings-report"]


def _syllables(n: int, seed: int = 7) -> str:
    vocab = "ab ba na ka ma ra sa ta ha ya".split()
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(vocab[x % len(vocab)])
    return " ".join(out)  # one syllable tokenizes to ~1 token (measured 1:0.998)


def workload(cfg: dict) -> list[dict]:
    """The staggered items: N victims (short prompt, fixed answer) then the aggressor.

    Victims carry a distinguishable first token (the numbered instruction) so a greedy
    answer is stable, and every request pins its output with ignore_eos: an arm that
    stops sooner is not credited for doing less work."""
    items = []
    for i in range(cfg["victims"]):
        items.append({
            "role": "victim",
            "prompt": f"Repeat after me exactly: item {i}. Then count "
                      + _syllables(cfg["short"]),
            "max_tokens": cfg["out"],
            "at": 2.0 * i,
        })
    items.append({
        "role": "aggressor",
        "prompt": "Summarize this text: " + _syllables(cfg["long"]),
        "max_tokens": max(32, cfg["out"] // 2),
        "at": cfg["gap"],
    })
    return items


def request_body(model: str, prompt: str, max_tokens: int,
                 temperature: float | None) -> dict:
    """The probe request. temperature=None OMITS the field so the server applies its
    production defaults (sampling_defaults=model: the checkpoint's generation_config,
    plus the high-temperature guard) - calibrating the regime traffic actually runs.
    Pass 0.0 for the deterministic greedy variant."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        # server-side measurement (#887/#868): per-chunk timings need the server's
        # --enable-timings-report too; continuous usage gives the exact token count
        # even when one chunk carries several tokens.
        "timings_per_token": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    if temperature is not None:
        body["temperature"] = temperature
        body["top_k"] = 1
        body["top_p"] = 1.0
    return body


def _stream_chat(server: Server, model: str, prompt: str, max_tokens: int,
                 t_start: float, delay: float, timeout: float = REQUEST_TIMEOUT,
                 temperature: float | None = None) -> dict:
    """One streamed request; returns per-token arrival times (perf_counter)."""
    body = request_body(model, prompt, max_tokens, temperature)
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        server.base + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.perf_counter() < t_start + delay:
        time.sleep(0.005)
    t0 = time.perf_counter()
    with opener.open(req, timeout=timeout) as r:
        arrivals, ttft, usage_tokens = [], None, 0
        engine_gaps: list[float | None] = []
        last_pred_ms: float | None = None
        timings_final: dict | None = None
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage_tokens = int((obj["usage"] or {}).get("completion_tokens") or 0)
            t = obj.get("timings")
            if t:
                timings_final = t
            pred_ms = t.get("predicted_ms") if t else None
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content")
                if piece:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        # engine-side gap since the previous delivered chunk: the
                        # cumulative predicted_ms delta carries no HTTP/client jitter
                        if pred_ms is not None and last_pred_ms is not None:
                            engine_gaps.append(max(0.0, (pred_ms - last_pred_ms) / 1000.0))
                        else:
                            engine_gaps.append(None)
                    arrivals.append(now)
                    if pred_ms is not None:
                        last_pred_ms = pred_ms
    return {"ttft": ttft or 0.0, "arrivals": arrivals, "tokens": len(arrivals),
            "usage_tokens": usage_tokens, "engine_gaps": engine_gaps,
            "timings": timings_final, "end": time.perf_counter()}


def run_workload(server: Server, model: str, items: list[dict],
                 temperature: float | None = None) -> dict:
    """Launch the staggered workload and score it. The stall window opens when the
    aggressor is admitted: gaps BEFORE that are the victims' own (short) prefill."""
    t_start = time.perf_counter()
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def one(name: str, item: dict) -> None:
        r = _stream_chat(server, model, item["prompt"], item["max_tokens"],
                         t_start, item["at"], temperature=temperature)
        with lock:
            results[name] = r

    threads = [threading.Thread(target=one, args=(f"{i}", it)) for i, it in enumerate(items)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # the stall window opens when the aggressor is ADMITTED: the inline prefill that
    # follows is exactly what the victims are made to wait through. Opening at the
    # aggressor's first token would score only the post-prefill tail and hide the stall.
    open_at = t_start + items[-1]["at"]
    agg = results[f"{len(items) - 1}"]
    victim = {f"{i}": results[f"{i}"] for i in range(len(items) - 1)}
    scored = score_window(victim, open_at)
    wall = max(r["end"] for r in results.values()) - t_start
    # exact server token count when continuous usage reported it, else the chunk count
    tokens = sum(r["usage_tokens"] or r["tokens"] for r in results.values())
    used_engine = any(g is not None for r in victim.values()
                      for g in (r.get("engine_gaps") or []))
    scored |= {
        "ttft_long": agg["ttft"],
        "ttft_short": statistics.mean([r["ttft"] for r in victim.values()]) if victim else 0.0,
        "out_tps": tokens / max(wall, 1e-6), "tokens": tokens, "wall": wall,
        "measurement": "server-timings" if used_engine else "client-arrivals",
    }
    return scored


def decode_workload(cfg: dict) -> list[dict]:
    """The decode-only probe: one short request per running slot (the profile's full
    --max-running-requests), all launched together, fixed answers, no aggressor. After
    the brief prefill ramp every step is a concurrent decode at the served batch -- the
    regime long-answer traffic lives in, which the staggered workload's
    aggressor-dominated wall does not score."""
    n = cfg.get("concurrency") or cfg["victims"] + 1
    return [{"prompt": f"Repeat after me exactly: decode probe {i}. Then count "
                       + _syllables(24),
             "max_tokens": cfg["out"], "at": 0.0}
            for i in range(n)]


def interleave_workload(cfg: dict) -> list[dict]:
    """The short-interleave probe: one ~1024-token prompt per running slot, admitted
    5 s apart (the launcher bench's gap) with 1024-token answers -- staggered chat
    traffic into a decoding batch, no 16k aggressor. The spacing matches the bench:
    a tighter 1 s spacing piles the prefills up and flatters piped, which is not the
    regime the profile serves. The decode-only probe covers the full-batch steady
    state; this one scores what sharing the machine with short chunks costs the
    running decodes at the bench's arrival cadence."""
    # per-request seed: identical prompts would radix-hit and turn the probe into a
    # cache test (the launcher bench varies its seed for the same reason)
    n = cfg.get("concurrency") or cfg["victims"] + 1
    return [{"prompt": "Summarize this text: " + _syllables(1010, seed=100 + i),
             "max_tokens": 1024, "at": 5.0 * i}
            for i in range(n)]


def run_concurrent_probe(server: Server, model: str, items: list[dict],
                         prefix: str, temperature: float | None = None) -> dict:
    """Launch one regime probe; score aggregate tok/s (the selection score's input)
    plus the worst inter-token gap (a smoothness diagnostic, not a selection input).
    prefix names the report keys."""
    t_start = time.perf_counter()
    results: list[dict | None] = [None] * len(items)

    def one(i: int, item: dict) -> None:
        results[i] = _stream_chat(server, model, item["prompt"], item["max_tokens"],
                                  t_start, item["at"], temperature=temperature)

    threads = [threading.Thread(target=one, args=(i, it)) for i, it in enumerate(items)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = max(r["end"] for r in results) - t_start
    tokens = sum(r["usage_tokens"] or r["tokens"] for r in results)
    worst = 0.0
    for r in results:
        a = r["arrivals"]
        worst = max(worst, max((y - x for x, y in zip(a, a[1:])), default=0.0))
    return {f"{prefix}_tps": tokens / max(wall, 1e-6), f"{prefix}_tokens": tokens,
            f"{prefix}_wall": wall, f"{prefix}_itl_worst": worst}


def score_window(victims: dict[str, dict], open_at: float) -> dict:
    """Pure: the victims' inter-token gaps after the aggressor was admitted. The window
    boundary uses the client clock (when the aggressor was sent); each gap's MAGNITUDE
    uses the engine-side predicted_ms delta when present (no HTTP/client jitter), else
    the client arrival delta. stall_worst_client always reports the client-side number."""
    worst = worst_client = count = 0
    secs = 0.0
    for r in victims.values():
        arrivals = r["arrivals"]
        egaps = r.get("engine_gaps") or []
        for i, (a, b) in enumerate(zip(arrivals, arrivals[1:])):
            if b < open_at:  # a gap ENDING before the window never involved the aggressor
                continue
            client_gap = b - a
            gap = egaps[i] if i < len(egaps) and egaps[i] is not None else client_gap
            if gap > STALL_EVENT_S:
                count += 1
                secs += gap
            worst = max(worst, gap)
            worst_client = max(worst_client, client_gap)
    return {"stall_worst": worst, "stall_count": count, "stall_secs": secs,
            "stall_worst_client": worst_client}


@dataclass
class MultiScore:
    label: str
    env: dict[str, str]
    stall_worst: float
    stall_count: int
    stall_secs: float
    ttft_long: float
    ttft_short: float
    out_tps: float
    tokens: int
    wall: float
    reps: list[dict] = field(default_factory=list)
    valid: bool = True
    note: str = ""
    stall_worst_client: float = 0.0
    measurement: str = "client-arrivals"
    spec: str | None = None
    moe: str | None = None
    latency: dict = field(default_factory=dict)
    decode_tps: float = 0.0
    decode_itl_worst: float = 0.0
    interleave_tps: float = 0.0
    interleave_itl_worst: float = 0.0

    def line(self) -> str:
        return (f"{self.label:13s} stall={self.stall_worst:5.2f}s x{self.stall_count:3d} "
                f"ttft={self.ttft_long:6.2f}s short={self.ttft_short:5.2f}s "
                f"tps={self.out_tps:6.1f} dec={self.decode_tps:6.1f} "
                f"ilv={self.interleave_tps:6.1f}"
                + (f"  [{self.note}]" if self.note else ""))


def regime_tps(s: MultiScore) -> float:
    """The selection score: mean aggregate tok/s of the two full-concurrency regime
    probes. The staggered wall (out_tps) is aggressor-dominated and capped at 3
    victims, so it measures the stall, not the batch the profile serves."""
    return (s.decode_tps + s.interleave_tps) / 2.0


def arm_active(label: str, log_text: str) -> tuple[bool, str]:
    """Did the arm's mechanism actually engage? Read the arm's own server log. A combo
    label '<prefill>+spec-<algo>' or '<prefill>+moe-<strategy>' is judged on its prefill
    part; a pure spec arm is valid once the server ran (its drafts show in the timings
    block, not a log marker); a pure MoE arm must show the strategy it asked for."""
    prefill = label.split("+spec-", 1)[0].split("+moe-", 1)[0]
    if prefill in ("piped", "piped-sf"):
        if "piped prefill enabled" in log_text:
            return True, ""
        if "piping disabled" in log_text:
            why = next((ln.split("piping disabled")[0].strip()[-80:]
                        for ln in log_text.splitlines() if "piping disabled" in ln), "")
            return False, f"piping disabled: {why}"
        return False, "no 'piped prefill enabled' line"
    if prefill == "mixed":
        if " mix=" in log_text:
            return True, ""
        return False, "no [chunktime] mix= line (mixed steps never ran)"
    if label.startswith("moe-"):
        strategy = label[len("moe-"):]
        if f"moe_strategy='{strategy}'" not in log_text:
            return False, f"no resolved moe_strategy='{strategy}' line"
        if strategy == "hybrid" and "fixed fetch cap of 1" in log_text:
            # the arm ran, but without a matching `ft bench bw` row the fetch split
            # fell back -- score it honestly and say how to unlock the matched split
            return True, ("hybrid ran with fetch cap 1: run `ft bench bw --model "
                          "<checkpoint> --tp-size N` for the bandwidth-matched split")
    return True, ""


def _latency_summary(path: Path) -> dict:
    """The arm's FREETOKEN_LATENCY_OUT JSONL: window count, mean phase share, and the
    spec-verify tally. Diagnostic only (the probe is host-side, never a device sync);
    an absent/empty file means it never ran. Every TP rank appends, pid separates them."""
    try:
        recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    except (OSError, json.JSONDecodeError):
        return {}
    if not recs:
        return {}
    phases: dict[str, list] = {}
    verifies = accepted = 0
    for rec in recs:
        for name, ph in (rec.get("phases") or {}).items():
            phases.setdefault(name, []).append(ph.get("pct", 0.0))
        sp = rec.get("spec") or {}
        verifies += sp.get("verifies", 0)
        accepted += sp.get("accepted", 0)
    out: dict = {"windows": len(recs),
                 "phase_pct_mean": {k: round(statistics.mean(v), 1) for k, v in phases.items()}}
    if verifies:
        out["spec_verifies"] = verifies
        out["spec_accepted"] = accepted
    return out


def evaluate_multi(spec: dict, label: str, env: dict[str, str], cfg: dict,
                   reps: int, log_dir: Path, port: int,
                   temperature: float | None = None,
                   spec_args: list[str] | None = None, spec_algo: str | None = None,
                   moe_args: list[str] | None = None, moe_strategy: str | None = None,
                   fatal: bool = True) -> MultiScore:
    # the tool owns the spec and MoE dimensions: strip the profile's flags, then arm
    # this arm's backend/strategy; --enable-timings-report gates the per-chunk timings (#887)
    base_argv = argv_with_timings(argv_with_port(strip_moe_args(strip_spec_args(spec["argv"])), port))
    base_argv = base_argv + list(spec_args or []) + list(moe_args or [])
    last_error: Exception | None = None
    latency_path = log_dir / f"{_sanitize(label)}-latency.jsonl"
    for attempt in range(3):
        arm_port = port + 2 * attempt
        if not _port_free(arm_port) or not _port_free(arm_port + 1):
            continue
        items = workload(cfg)
        ditems = decode_workload(cfg)
        iitems = interleave_workload(cfg)
        per: list[dict] = []
        dper: list[dict] = []
        iper: list[dict] = []
        log_name = f"{_sanitize(label)}.log" if attempt == 0 else f"{_sanitize(label)}-try{attempt + 1}.log"
        latency_path = log_dir / (Path(log_name).stem + "-latency.jsonl")
        run_spec = {**spec, "port": arm_port, "argv": argv_with_port(base_argv, arm_port)}
        full_env = {**env, "FREETOKEN_CHUNK_TIMING": "1", "FREETOKEN_LATENCY": "1",
                    "FREETOKEN_LATENCY_OUT": str(latency_path)}
        try:
            with Server(run_spec, full_env, log_dir / log_name) as server:
                model = spec.get("model_name") or "default"
                # warm-up doubles as the liveness probe (see calibrate_lookup)
                _stream_chat(server, model, "Repeat after me exactly: warm up.", 16,
                             time.perf_counter(), 0.0, timeout=WARMUP_TIMEOUT,
                             temperature=temperature)
                print(f"    {label}: running the regime probes + staggered workload "
                      f"({len(ditems)}-slot probes, {len(items)} staggered requests "
                      f"x {max(1, reps)} reps)", flush=True)
                for _ in range(max(1, reps)):
                    # regime probes first: they score the clean-cache steady decode and
                    # the short-interleave regime, before the aggressor fills the cache
                    dper.append(run_concurrent_probe(server, model, ditems, "decode",
                                                     temperature))
                    iper.append(run_concurrent_probe(server, model, iitems, "interleave",
                                                     temperature))
                    per.append(run_workload(server, model, items, temperature))
            log_text = (log_dir / log_name).read_text(errors="replace")
            break
        except (ServerError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            time.sleep(3)
    else:
        if not fatal:
            # a spec backend the checkpoint cannot serve (no MTP layers, bad draft) is
            # an excluded arm with a reason, not a failed calibration
            return MultiScore(label=label, env=dict(env), stall_worst=0.0, stall_count=0,
                              stall_secs=0.0, ttft_long=0.0, ttft_short=0.0, out_tps=0.0,
                              tokens=0, wall=0.0, valid=False, spec=spec_algo,
                              moe=moe_strategy,
                              note=f"server failed: {last_error}")
        raise ServerError(f"{label}: three attempts failed (last: {last_error})") from last_error

    # measurement is a per-rep label, not a number: median only the numeric keys
    med = {k: statistics.median([p[k] for p in per]) for k, v in per[0].items()
           if isinstance(v, (int, float))}
    med["measurement"] = per[0]["measurement"]
    valid, note = arm_active(label, log_text)
    return MultiScore(
        label=label, env=dict(env),
        stall_worst=float(med["stall_worst"]), stall_count=int(med["stall_count"]),
        stall_secs=float(med["stall_secs"]), ttft_long=float(med["ttft_long"]),
        ttft_short=float(med["ttft_short"]), out_tps=float(med["out_tps"]),
        tokens=int(med["tokens"]), wall=float(med["wall"]), reps=per,
        valid=valid, note=note,
        stall_worst_client=float(med.get("stall_worst_client", 0.0)),
        measurement=str(med.get("measurement", "client-arrivals")),
        spec=spec_algo, moe=moe_strategy, latency=_latency_summary(latency_path),
        decode_tps=float(statistics.median([d["decode_tps"] for d in dper])),
        decode_itl_worst=float(statistics.median([d["decode_itl_worst"] for d in dper])),
        interleave_tps=float(statistics.median([d["interleave_tps"] for d in iper])),
        interleave_itl_worst=float(statistics.median([d["interleave_itl_worst"] for d in iper])),
    )


def _regime_guard(cands: list[MultiScore], base: MultiScore) -> tuple[list[MultiScore], str]:
    """Drop arms whose decode-only or short-interleave probe regresses beyond its
    tolerance. Returns (kept, note); when none survive, the note names the fastest
    regressing arm and the probe it failed. An unmeasured probe (0.0 -- an excluded
    arm, an old report) cannot judge and passes."""
    checks = (("decode-only", base.decode_tps, "decode_tps", DECODE_TPS_LOSS_MAX),
              ("short-interleave", base.interleave_tps, "interleave_tps",
               INTERLEAVE_TPS_LOSS_MAX))
    kept: list[MultiScore] = []
    dropped: list[tuple[MultiScore, str]] = []
    for s in cands:
        note = ""
        for name, bval, attr, tol in checks:
            v = getattr(s, attr)
            if bval > 0 and 0 < v < bval * (1 - tol):
                note = (f"{100 * (1 - v / bval):.0f}% {name} tok/s "
                        f"({bval:.1f} -> {v:.1f}; max {100 * tol:.0f}%)")
                break
        if note:
            dropped.append((s, note))
        else:
            kept.append(s)
    if kept or not cands:
        return kept, ""
    worst, note = max(dropped, key=lambda sn: sn[0].out_tps)
    name = worst.spec or worst.moe or worst.label
    return kept, f"{name} costs {note}"


def choose_multi(scores: list[MultiScore]) -> tuple[str, str]:
    """The selection contract (see module docstring): the highest mean regime tok/s at
    the profile's full concurrency wins, provided the gain clears TPS_MIN_GAIN and the
    worst staggered stall does not worsen beyond MULTI_STALL_TOL. Pure so the unit
    tests pin it."""
    by = {s.label: s for s in scores}
    base = by["none"]
    cands = [s for s in scores if s.label != "none" and s.valid and regime_tps(s) > 0]
    if not cands:
        return "none", "no concurrency arm activated on this machine"
    cands, dnote = _regime_guard(cands, base)
    if not cands:
        return "none", dnote
    best = max(cands, key=lambda s: (regime_tps(s), -s.stall_worst, -s.ttft_short))
    base_tps = regime_tps(base)
    gain = regime_tps(best) / base_tps - 1 if base_tps > 0 else 0.0
    if gain < TPS_MIN_GAIN:
        return "none", (f"{best.label} adds only {100 * gain:.1f}% concurrent tok/s "
                        f"(need {100 * TPS_MIN_GAIN:.0f}%)")
    if best.stall_worst > base.stall_worst + MULTI_STALL_TOL:
        return "none", (f"{best.label} worsens the worst stall {base.stall_worst:.2f}s -> "
                        f"{best.stall_worst:.2f}s (max +{MULTI_STALL_TOL}s)")
    return best.label, (f"concurrent tok/s {base_tps:.1f} -> {regime_tps(best):.1f} "
                        f"(+{100 * gain:.1f}%), stall {base.stall_worst:.2f}s -> "
                        f"{best.stall_worst:.2f}s")


def choose_spec(spec_scores: list[MultiScore], base: MultiScore) -> tuple[str | None, str]:
    """The spec dimension's contract: a backend wins when it adds >= SPEC_MIN_GAIN
    mean regime tok/s over the no-spec arm AND does not worsen the worst stall by more
    than SPEC_STALL_TOL (spec speeds decode; it must not freeze the running streams).
    Pure so the unit tests pin it. Returns (algo, note)."""
    cands = [s for s in spec_scores if s.valid and s.spec and regime_tps(s) > 0]
    if not cands:
        return None, "no speculative arm activated on this machine"
    cands, dnote = _regime_guard(cands, base)
    if not cands:
        return None, dnote
    best = max(cands, key=regime_tps)
    base_tps = regime_tps(base)
    gain = regime_tps(best) / base_tps - 1 if base_tps > 0 else 0.0
    if gain < SPEC_MIN_GAIN:
        return None, (f"{best.spec} adds only {100 * gain:.1f}% concurrent tok/s "
                      f"(need {100 * SPEC_MIN_GAIN:.0f}%)")
    if best.stall_worst > base.stall_worst + SPEC_STALL_TOL:
        return None, (f"{best.spec} worsens the worst stall {base.stall_worst:.2f}s -> "
                      f"{best.stall_worst:.2f}s (max +{SPEC_STALL_TOL}s)")
    return best.spec, (f"concurrent tok/s {base_tps:.1f} -> {regime_tps(best):.1f} "
                       f"(+{100 * gain:.1f}%), "
                       f"stall {base.stall_worst:.2f}s -> {best.stall_worst:.2f}s")


def choose_moe(moe_scores: list[MultiScore], base: MultiScore) -> tuple[str | None, str]:
    """The MoE dimension's contract: a strategy wins when it adds >= MOE_MIN_GAIN
    mean regime tok/s over the base arm AND does not worsen the worst stall by more
    than MOE_STALL_TOL (hybrid trades PCIe misses for CPU compute; it must not freeze
    the running streams). Pure so the unit tests pin it. Returns (strategy, note)."""
    cands = [s for s in moe_scores if s.valid and s.moe and regime_tps(s) > 0]
    if not cands:
        return None, "no MoE strategy arm activated on this machine"
    cands, dnote = _regime_guard(cands, base)
    if not cands:
        return None, dnote
    best = max(cands, key=regime_tps)
    base_tps = regime_tps(base)
    gain = regime_tps(best) / base_tps - 1 if base_tps > 0 else 0.0
    if gain < MOE_MIN_GAIN:
        return None, (f"{best.moe} adds only {100 * gain:.1f}% concurrent tok/s "
                      f"(need {100 * MOE_MIN_GAIN:.0f}%)")
    if best.stall_worst > base.stall_worst + MOE_STALL_TOL:
        return None, (f"{best.moe} worsens the worst stall {base.stall_worst:.2f}s -> "
                      f"{best.stall_worst:.2f}s (max +{MOE_STALL_TOL}s)")
    return best.moe, (f"concurrent tok/s {base_tps:.1f} -> {regime_tps(best):.1f} "
                      f"(+{100 * gain:.1f}%), "
                      f"stall {base.stall_worst:.2f}s -> {best.stall_worst:.2f}s")


def render_script_multi(spec: dict, chosen: dict | None) -> str:
    """The runnable pair to the lookup tool's script: same argv, winner env exported,
    plus the winning speculative backend's and MoE strategy's flags when applied."""
    env = {str(k): str(v) for k, v in (spec.get("env") or {}).items() if k not in MULTI_KEYS}
    if chosen:
        env.update({str(k): str(v) for k, v in (chosen.get("env") or {}).items()})
    cwd = spec.get("cwd") or str(ROOT)
    python = spec.get("python") or sys.executable
    argv = argv_with_port(strip_moe_args(strip_spec_args(spec["argv"])),
                          int(spec.get("port") or DEFAULT_PORT))
    if chosen and (chosen.get("spec") or {}).get("applied"):
        argv = argv + list(chosen["spec"].get("argv") or [])
    if chosen and (chosen.get("moe") or {}).get("applied"):
        argv = argv + list(chosen["moe"].get("argv") or [])
    lines = ["#!/bin/sh",
             f"# generated by freetoken.calibrate_multi.py on {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"cd {shlex.quote(str(cwd))}"]
    for key in sorted(env):
        lines.append(f"export {key}={_quote_env(env[key])}")
    lines.append("exec " + " ".join(shlex.quote(x) for x in [python, *argv]))
    return "\n".join(lines) + "\n"


def victims_from_spec(spec: dict, default: int = 3) -> int:
    raw = _arg_value(list(spec.get("argv") or []), "--max-running-requests")
    try:
        mrr = int(raw) if raw else default + 1
    except ValueError:
        mrr = default + 1
    return max(1, min(default, mrr - 1))


def concurrency_from_spec(spec: dict, default: int = 4) -> int:
    """The profile's full --max-running-requests. The regime probes score the batch
    the profile actually serves: the staggered workload caps its victims at 3, but an
    8-parallel profile decodes at bs8, and the fast-path cost scales with the batch."""
    raw = _arg_value(list(spec.get("argv") or []), "--max-running-requests")
    try:
        mrr = int(raw) if raw else default
    except ValueError:
        mrr = default
    return max(1, mrr)


def calibrate_multi(spec: dict, a: argparse.Namespace) -> dict:
    out_dir = Path(a.out_dir).expanduser().resolve()
    log_dir = out_dir / "logs"
    cfg = {"short": 128, "long": 2048 if a.smoke else a.long,
           "out": 32 if a.smoke else a.out_tokens, "gap": 6.0,
           "victims": victims_from_spec(spec),
           "concurrency": concurrency_from_spec(spec)}
    reps = 1 if (a.quick or a.smoke) else max(1, a.reps)
    arms = arms_for(spec, a.smoke)
    base_port = int(spec.get("port") or DEFAULT_PORT)
    # the tool owns both dimensions: strip the profile's spec flags (remembering its
    # draft path for dflash) and its stale multi-knob env values
    draft_path = (getattr(a, "spec_draft_path", "")
                  or _arg_value(list(spec.get("argv") or []), "--speculative-draft-model-path")
                  or _arg_value(list(spec.get("argv") or []), "--spec-draft-model"))
    num_steps = getattr(a, "spec_num_steps", 0) or None
    spec = {**spec,
            "argv": strip_moe_args(strip_spec_args(ensure_distributed_timeout(spec["argv"]))),
            "env": {k: v for k, v in (spec.get("env") or {}).items() if k not in MULTI_KEYS}}

    scores: list[MultiScore] = []
    next_port = [base_port]

    def take_port() -> int:
        while True:
            port = next_port[0]
            next_port[0] += 2
            if _port_free(port) and _port_free(port + 1):
                return port

    def measure(label, env, spec_args=None, spec_algo=None, moe_args=None,
                moe_strategy=None, fatal=True) -> MultiScore:
        print(f"  measuring {label} ...", flush=True)
        s = evaluate_multi(spec, label, env, cfg, reps, log_dir, take_port(),
                           a.temperature, spec_args=spec_args, spec_algo=spec_algo,
                           moe_args=moe_args, moe_strategy=moe_strategy, fatal=fatal)
        print("    " + s.line(), flush=True)
        scores.append(s)
        return s

    # --- prefill dimension (speculation off) ---
    for label, env, skip in arms:
        if skip:
            print(f"  skipping {label}: {skip}", flush=True)
            scores.append(MultiScore(label=label, env=env, stall_worst=0.0,
                                     stall_count=0, stall_secs=0.0, ttft_long=0.0,
                                     ttft_short=0.0, out_tps=0.0, tokens=0, wall=0.0,
                                     valid=False, note=skip))
            continue
        measure(label, env)

    label, note = choose_multi(scores)
    by = {s.label: s for s in scores}
    base = by["none"]

    # --- speculative dimension (prefill off): mtp / dflash compared on the same workload ---
    backends = spec_backends_for(getattr(a, "spec", "auto"), draft_path)
    if a.smoke:
        backends = [b for b in backends if b[0] == "mtp"] or backends[:1]
    spec_scores: list[MultiScore] = []
    for algo, skip in backends:
        slabel = f"spec-{algo}"
        if skip:
            print(f"  skipping {slabel}: {skip}", flush=True)
            s = MultiScore(label=slabel, env=winner_env("none"), stall_worst=0.0,
                           stall_count=0, stall_secs=0.0, ttft_long=0.0, ttft_short=0.0,
                           out_tps=0.0, tokens=0, wall=0.0, valid=False, note=skip, spec=algo)
            scores.append(s)
            spec_scores.append(s)
            continue
        spec_scores.append(measure(slabel, winner_env("none"),
                                   spec_args=spec_argv(algo, draft_path, num_steps),
                                   spec_algo=algo, fatal=False))

    salgo, snote = choose_spec(spec_scores, base)
    spec_chosen = {"algorithm": salgo, "argv": [], "applied": False, "compatible": False,
                   "regime_tps": 0.0, "gain_tps_pct": 0.0, "stall_worst": 0.0, "note": snote}
    if salgo:
        sargs = spec_argv(salgo, draft_path, num_steps)
        sarm = next(s for s in spec_scores if s.spec == salgo)
        spec_chosen.update(argv=sargs, regime_tps=regime_tps(sarm),
                           stall_worst=sarm.stall_worst,
                           gain_tps_pct=round(100 * (regime_tps(sarm) / regime_tps(base) - 1), 1)
                           if regime_tps(base) > 0 else 0.0)
        if label == "none":
            # no prefill winner: the spec arm itself is the launch
            spec_chosen.update(applied=True, compatible=True)
        else:
            # try to stack the spec winner on the prefill winner. Under piped the engine
            # refuses speculation (spec verify shares the PLE/plan staging with the pipe's
            # chunk), so this arm verifies that empirically and reports the reason.
            combo = measure(f"{label}+spec-{salgo}", winner_env(label), spec_args=sargs,
                            spec_algo=salgo, fatal=False)
            holds = (combo.valid
                     and regime_tps(combo) >= regime_tps(sarm) * (1 - SPEC_COMBO_TPS_TOL)
                     and combo.stall_worst <= by[label].stall_worst + TIE_S)
            spec_chosen.update(
                applied=holds, compatible=combo.valid,
                note=(snote if holds else
                      f"{salgo} wins alone but not beside {label}: "
                      f"{combo.note or 'combo regressed'}"))
            if holds:
                spec_chosen.update(regime_tps=regime_tps(combo), stall_worst=combo.stall_worst)

    # --- MoE dimension (prefill off, speculation off): offload / hybrid on the same workload ---
    strategies = moe_strategies_for(getattr(a, "moe", "auto"))
    if a.smoke:
        strategies = [s for s in strategies if s[0] == "hybrid"] or strategies[:1]
    moe_scores: list[MultiScore] = []
    for strategy, skip in strategies:
        mlabel = f"moe-{strategy}"
        if skip:
            print(f"  skipping {mlabel}: {skip}", flush=True)
            s = MultiScore(label=mlabel, env=winner_env("none"), stall_worst=0.0,
                           stall_count=0, stall_secs=0.0, ttft_long=0.0, ttft_short=0.0,
                           out_tps=0.0, tokens=0, wall=0.0, valid=False, note=skip,
                           moe=strategy)
            scores.append(s)
            moe_scores.append(s)
            continue
        moe_scores.append(measure(mlabel, winner_env("none"),
                                  moe_args=moe_argv(strategy), moe_strategy=strategy,
                                  fatal=False))

    mstrategy, mnote = choose_moe(moe_scores, base)
    moe_chosen = {"strategy": mstrategy, "argv": [], "applied": False, "compatible": False,
                  "regime_tps": 0.0, "gain_tps_pct": 0.0, "stall_worst": 0.0, "note": mnote}
    if mstrategy:
        margs = moe_argv(mstrategy)
        marm = next(s for s in moe_scores if s.moe == mstrategy)
        moe_chosen.update(argv=margs, regime_tps=regime_tps(marm),
                          stall_worst=marm.stall_worst,
                          gain_tps_pct=round(100 * (regime_tps(marm) / regime_tps(base) - 1), 1)
                          if regime_tps(base) > 0 else 0.0)
        if label == "none" and not spec_chosen["applied"]:
            # no other winner: the strategy arm itself is the launch
            moe_chosen.update(applied=True, compatible=True)
        else:
            # verify the strategy stacks on the current winner stack (prefill env +
            # the spec flags when they were applied). The combo must not regress the
            # stack's own throughput/stall -- the gain itself was proven standalone.
            ref_tps = spec_chosen["regime_tps"] if spec_chosen["applied"] else (
                regime_tps(by[label]) if label != "none" else regime_tps(base))
            ref_stall = spec_chosen["stall_worst"] if spec_chosen["applied"] else (
                by[label].stall_worst if label != "none" else base.stall_worst)
            parts = ([label] if label != "none" else []) + (
                [f"spec-{salgo}"] if spec_chosen["applied"] else []) + [f"moe-{mstrategy}"]
            combo = measure("+".join(parts), winner_env(label),
                            spec_args=spec_chosen["argv"] if spec_chosen["applied"] else None,
                            spec_algo=salgo if spec_chosen["applied"] else None,
                            moe_args=margs, moe_strategy=mstrategy, fatal=False)
            holds = (combo.valid
                     and regime_tps(combo) >= ref_tps * (1 - MOE_COMBO_TPS_TOL)
                     and combo.stall_worst <= ref_stall + TIE_S)
            moe_chosen.update(
                applied=holds, compatible=combo.valid,
                note=(mnote if holds else
                      f"{mstrategy} wins alone but regresses the {'+'.join(parts[:-1])} stack: "
                      f"{combo.note or 'combo regressed'}"))
            if holds:
                moe_chosen.update(regime_tps=regime_tps(combo), stall_worst=combo.stall_worst)

    chosen = {
        "label": label, "env": winner_env(label), "note": note,
        "stall_worst": by[label].stall_worst, "out_tps": by[label].out_tps,
        "regime_tps": regime_tps(by[label]), "base_regime_tps": regime_tps(base),
        "decode_tps": by[label].decode_tps, "base_decode_tps": base.decode_tps,
        "interleave_tps": by[label].interleave_tps,
        "base_interleave_tps": base.interleave_tps,
        "gain_stall_s": round(base.stall_worst - by[label].stall_worst, 2),
        "gain_tps_pct": round(100 * (regime_tps(by[label]) / regime_tps(base) - 1), 1)
        if regime_tps(base) > 0 else 0.0,
        "spec": spec_chosen,
        "moe": moe_chosen,
    }
    return {
        "schema_version": 1, "backend": "freetoken-multi",
        "hardware": hardware(), "workload": {**cfg, "reps": reps},
        "arms": [{k: v for k, v in s.__dict__.items() if k != "reps"}
                 | {"env": s.env, "regime_tps": regime_tps(s)}
                 for s in scores],
        "chosen": chosen, "multi_enabled": label != "none",
        "spec_enabled": bool(spec_chosen["applied"]),
        "moe_enabled": bool(moe_chosen["applied"]),
    }


def parse_live_servers(pgrep_out: str) -> list[str]:
    """pgrep lines that are RUNNING FreeToken servers (the calibrator itself matches
    "-m freetoken" too, so key on the server's --model-path and skip calibrate)."""
    out = []
    for line in (pgrep_out or "").splitlines():
        if "--model-path" in line and "calibrate" not in line:
            out.append(line.strip())
    return out


def find_live_servers() -> list[str]:
    import subprocess

    # '--' ends pgrep's options: the pattern starts with a dash, and without it
    # pgrep exits 2 with empty stdout -- the guard silently never fires.
    r = subprocess.run(["pgrep", "-af", "--", "-m freetoken"], capture_output=True, text=True)
    return parse_live_servers(r.stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Tune concurrent serving (piped/mixed/short-first prefill, spec, MoE) "
                    "for the highest aggregate tok/s at the profile's concurrency; "
                    "write a run config and start script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", help="existing launch JSON (python/argv/cwd/env/port/model_name)")
    ap.add_argument("--model-path", help="model snapshot or directory (without --config)")
    ap.add_argument("--served-model-name", default="")
    ap.add_argument("--python", default="", help="server interpreter (default: this interpreter)")
    ap.add_argument("--server-args", default="", help="extra server flags for --model-path mode")
    ap.add_argument("--port", type=int, default=0, help=f"server port (default {DEFAULT_PORT})")
    ap.add_argument("--out-dir", default=".", help="where the JSON and .sh are written")
    ap.add_argument("--tag", default="", help="file tag (default: from the served model name)")
    ap.add_argument("--quick", action="store_true", help="one rep per arm")
    ap.add_argument("--smoke", action="store_true",
                    help="two arms and a short aggressor: checks the plumbing, not a calibration")
    ap.add_argument("--reps", type=int, default=2, help="measured repeats per arm")
    ap.add_argument("--long", type=int, default=16384, help="aggressor prompt tokens")
    ap.add_argument("--out-tokens", type=int, default=256, help="victim answer length")
    ap.add_argument("--temperature", type=float, default=None,
                    help="probe sampling temperature; default OMITs it so the server's "
                         "production defaults (generation_config + the high-temp guard) "
                         "apply. 0.0 = deterministic greedy arms")
    ap.add_argument("--spec", default="auto",
                    help="speculative backends to compare: 'auto' (mtp + dflash), 'none', "
                         "or a comma list of mtp,dflash. dflash is skipped without a draft "
                         "checkpoint. Each is measured beside the prefill winner when the "
                         "engine allows it")
    ap.add_argument("--spec-draft-path", default="",
                    help="DFlash 2 draft checkpoint for the dflash arm (else taken from the "
                         "profile's --speculative-draft-model-path)")
    ap.add_argument("--spec-num-steps", type=int, default=0,
                    help="draft tokens per step for the spec arms (0 = each method's default)")
    ap.add_argument("--moe", default="auto",
                    help="MoE strategies to compare: 'auto' (offload + hybrid), 'none', or a "
                         "comma list of offload,hybrid. hybrid needs a CPU MoE executor the "
                         "checkpoint supports; without a matching `ft bench bw --tp-size N` "
                         "row it runs with a fetch cap of 1 (the report says so). The winner's "
                         "--moe-strategy goes into the report and the start script")
    a = ap.parse_args(argv)
    import signal as _signal
    if hasattr(_signal, "SIGTERM"):
        _signal.signal(_signal.SIGTERM, _on_sigterm)

    # A second server on this box is not just a port collision: the expert banks are
    # pinned in HOST RAM (63 GB on this model), so a second load OOM-kills whichever
    # process the kernel picks -- measured as a calibration dying mid-arm beside an
    # orphaned probe. Refuse up front with the list of what to stop.
    live = find_live_servers()
    if live and not a.smoke:
        print("refusing to calibrate: a FreeToken server is already running (it holds "
              "the expert banks in host RAM and the VRAM):\n  " + "\n  ".join(live) +
              "\nstop it first (launcher, or kill the pid above)", file=sys.stderr)
        return 1

    spec = spec_from_args(a)
    tag = _sanitize(a.tag or spec.get("model_name") or "model")
    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if a.smoke else ("quick" if a.quick else "full")
    print(f"calibrating concurrent prefill for {spec.get('model_name')} on port "
          f"{spec['port']} ({mode}); this restarts the model per arm", flush=True)
    started = time.time()
    try:
        result = calibrate_multi(spec, a)
    except KeyboardInterrupt:
        print("\ninterrupted; the server was stopped", file=sys.stderr)
        return 130
    except ServerError as e:
        print(f"calibration failed: {e}", file=sys.stderr)
        return 1
    result["elapsed_s"] = round(time.time() - started, 1)
    result["tag"] = tag

    chosen = result["chosen"]
    # pass the winner when ANY dimension won: prefill=none + a spec or MoE win still
    # exports the explicit-zero knob set and the winner's flags
    has_winner = result["multi_enabled"] or result["spec_enabled"] or result["moe_enabled"]
    script_text = render_script_multi(spec, chosen if has_winner else None)
    json_path = out_dir / f"freetoken-multi-{tag}.json"
    script_path = out_dir / f"run-freetoken-multi-{tag}.sh"
    _atomic_write(json_path, json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    _atomic_write(script_path, script_text)
    script_path.chmod(0o755)

    print(f"\nchosen prefill: {chosen['label']} -- {chosen['note']}", flush=True)
    sp = chosen.get("spec") or {}
    if sp.get("algorithm"):
        state = "applied" if sp.get("applied") else "NOT applied"
        print(f"chosen spec: {sp['algorithm']} ({state}) -- {sp.get('note', '')}", flush=True)
    mo = chosen.get("moe") or {}
    if mo.get("strategy"):
        state = "applied" if mo.get("applied") else "NOT applied"
        print(f"chosen moe: {mo['strategy']} ({state}) -- {mo.get('note', '')}", flush=True)
    if not result["multi_enabled"]:
        print("the concurrency knobs stay off on this machine; the script starts without them",
              flush=True)
    print(f"config: {json_path}")
    print(f"script: {script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
