#!/usr/bin/env python3
"""Token generation speed of a FreeToken server: TTFT, per-stream decode tok/s, aggregate tok/s.

Targets a server that is already running, so the default is measuring the machine in front of
you; no model path, no GPU needed on the client side:

    python benchmarks/bench_token_speed.py                       # http://127.0.0.1:1919
    python benchmarks/bench_token_speed.py --server http://192.168.1.20:1919 --api-key $KEY

Pass --model instead to have the bench spawn its own server, measure it, and shut it down:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python benchmarks/bench_token_speed.py \
        --model /path/to/model --decode 256 --requests 8 --json speed.jsonl

Concurrency. --concurrency C sends requests in waves of C, released together by a barrier so
thread start-up does not smear the window, and --concurrency-sweep 1,2,4,8 prints how the same
workload scales. Two numbers come out of a concurrent run and they answer different questions:
"decode per stream" is what one client experiences, "aggregate output" is what the box produces.

Speaks the OpenAI wire protocol, so every number is what a client sees -- scheduler, tokenizer,
detokenizer and the HTTP/SSE hop included -- not a bare engine forward.

Method. Each request is streamed with stream_options.include_usage, so token arrivals are
timestamped as SSE events and the closing chunk reports exact token counts. For one stream:

    decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

anchored on the first and last token, which stays correct when the detokenizer merges a few
tokens into one event (multibyte characters). ignore_eos pins the step count at --decode
whatever the model wanted to say. TTFT spans send -> first token, so it covers queueing,
tokenization and prefill; it is not a prefill-only measurement.

Server sharing. The bench cannot reserve a server, so it polls /v1/stats while measuring and
reports the highest concurrent-request count and the engine's own peak rates it saw: active
above your own concurrency plus one (the poll itself is an in-flight request) means someone
else was driving the same engine and the numbers are correspondingly pessimistic.

Sampling. Fields left at their default are filled from the checkpoint recommendation the server
reports on /v1/stats (model.sampling, read from generation_config.json), falling back to
temperature 1.0 / top_p 0.95 / top_k 64; an explicit --temperature / --top-p / --top-k wins.
Sampled rather than greedy keeps MoE routing realistic.

Prompt. --prompt-words repeats a neutral sentence to about that many words and prepends a random
nonce; the nonce leads because the radix prefix cache matches from the prompt head, so a unique
head makes every measured request a cold prefill. --reuse-prompt drops the nonce and measures the
warm prefix-hit path; --prompt-file takes the body from a file. Filler prose is out of
distribution, so expert routing is representative of generic text, not of a reasoning workload --
for that comparison use bench_decode_moe.py.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Sampling for fields the caller leaves alone and the server does not recommend: sampled rather
# than greedy keeps MoE routing realistic, and matches the fallback bench_decode_moe.py applies.
FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

FILLER_SENTENCE = (
    "the quick brown fox jumps over the lazy dog while the compiler waits for a signal "
    "and the scheduler rewrites its plan for the morning"
)
WORDS_PER_FILLER = len(FILLER_SENTENCE.split())

# How often the sharing-detector thread reads /v1/stats while a level is measured.
STATS_POLL_S = 0.5
# Threads reach the wave barrier within milliseconds; the timeout only stops a hang.
WAVE_BARRIER_TIMEOUT_S = 60.0

# THINKING_ON/OFF_KWARGS in freetoken/tokenizer/effort.py: the spellings the shipped templates read.
THINKING_KWARGS = {
    "on": {"enable_thinking": True, "thinking_mode": "enabled"},
    "off": {"enable_thinking": False, "thinking_mode": "disabled"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=os.environ.get("FREETOKEN_SERVER", "http://127.0.0.1:1919"),
                   help="origin of the server to measure (default $FREETOKEN_SERVER or %(default)s)")
    p.add_argument("--model", default=None,
                   help="checkpoint dir or HF id: spawn ft serve, measure it, shut it down")
    p.add_argument("--api-key", default=None, help="send this as Authorization: Bearer")
    p.add_argument("--serve-arg", action="append", default=[], metavar="ARG",
                   help="extra flag for the spawned ft serve, repeatable; needs the = form "
                        "(--serve-arg=--tp-size=2) because the value starts with dashes")
    p.add_argument("--server-timeout", type=float, default=1800,
                   help="seconds to wait for a spawned server to become ready")
    work = p.add_argument_group("workload")
    work.add_argument("--endpoint", choices=("chat", "completions"), default="chat",
                      help="chat renders the checkpoint template, completions sends the raw prompt")
    work.add_argument("--prompt-words", type=int, default=64, help="words of filler to prefill")
    work.add_argument("--prompt-file", default=None, help="prompt body from this file")
    work.add_argument("--reuse-prompt", action="store_true",
                      help="identical prompt per request: measures the prefix-hit path")
    work.add_argument("--decode", type=int, default=128, help="tokens to generate per request")
    work.add_argument("--requests", type=int, default=8, help="requests to measure, rounded down to full waves")
    work.add_argument("--warmup", type=int, default=1, help="requests to run and discard first")
    work.add_argument("--concurrency", type=int, default=1, help="requests in flight per wave")
    work.add_argument("--concurrency-sweep", default=None, metavar="C,C,...",
                      help="measure each of these concurrencies and print a scaling table")
    work.add_argument("--temperature", type=float, default=FALLBACK_SAMPLING["temperature"])
    work.add_argument("--top-p", type=float, default=FALLBACK_SAMPLING["top_p"])
    work.add_argument("--top-k", type=int, default=FALLBACK_SAMPLING["top_k"])
    work.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                      help="force chat-template thinking on reasoning checkpoints")
    p.add_argument("--seed", type=int, default=0, help="seed for the filler / nonce sequence")
    p.add_argument("--json", dest="json_out", default=None, help="append the result row(s) here")
    args = p.parse_args(argv)
    args.sweep = [args.concurrency]
    if args.concurrency_sweep:
        try:
            args.sweep = [int(c) for c in args.concurrency_sweep.split(",") if c.strip()]
        except ValueError:
            p.error("--concurrency-sweep wants a comma list of integers")
        if not args.sweep or any(c < 1 for c in args.sweep):
            p.error("--concurrency-sweep wants integers >= 1")
    if args.requests < 1:
        p.error("--requests must be >= 1")
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    if args.decode < 2:
        p.error("--decode must be >= 2 (one token leaves no decode window)")
    return args


def apply_server_sampling(args: argparse.Namespace, stats: dict | None) -> str:
    """Fill sampling fields still at the script default with the server's own recommendation.

    /v1/stats model.sampling comes from the checkpoint's generation_config.json, so the routing
    workload matches what the box is normally asked to do; a flag the caller passed wins."""
    rec = ((stats or {}).get("model") or {}).get("sampling") or {}
    filled = {}
    for field, default in FALLBACK_SAMPLING.items():
        if getattr(args, field) == default and field in rec:
            setattr(args, field, rec[field])
            filled[field] = rec[field]
    if not filled:
        return "script defaults"
    return f"{filled} <- server /v1/stats model.sampling"


def get_json(url: str, api_key: str | None = None, timeout: float = 30) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return json.loads(resp.read())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def die_with_log(msg: str, log_path: str) -> None:
    tail = "".join(Path(log_path).read_text(errors="replace").splitlines(keepends=True)[-30:])
    sys.exit(f"[bench] {msg}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with code {proc.returncode} during startup", log_path)
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):  # not bound yet / reset / partial response
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("status") == "ok" and health.get("maintenance", "serving") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def check_target(origin: str, api_key: str | None) -> None:
    """Fail early, and say why, instead of failing inside the first streamed request."""
    try:
        health = get_json(f"{origin}/health", api_key, timeout=10)
    except urllib.error.HTTPError:  # an OpenAI-compatible server without /health: measure anyway
        return
    except OSError as e:
        sys.exit(f"[bench] cannot reach {origin} ({e}); start it with ft serve or pass --server")
    if health.get("status") == "loading":
        sys.exit(f"[bench] server is still loading ({health.get('phase')} "
                 f"{health.get('progress')}); retry once it reports ok")
    if health.get("status") == "error":
        sys.exit(f"[bench] server reported an error: {health.get('message')}")
    if health.get("maintenance", "serving") != "serving":
        sys.exit(f"[bench] server is not serving: maintenance={health.get('maintenance')}")


def pump_output(src, log_f) -> None:
    """Mirror the spawned server to our terminal while keeping the log file complete."""
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the whole session (frontend + workers), escalate; runs in finally.

    Best-effort by design: killpg runs even when the frontend already exited, because a crashed
    frontend leaves live non-daemon workers in the group holding the GPU."""
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # whole group already gone
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue


def serve_cmd(args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--max-running-requests", str(max(args.sweep)),
    ]
    # Values start with dashes, so argparse only accepts them in the --serve-arg=--flag=value form.
    cmd += args.serve_arg
    return cmd


class StatsPoller(threading.Thread):
    """Reads /v1/stats while a level runs: catches traffic that is not ours and the engine's peak.

    The API exposes no per-request server-side timing, so a shared box otherwise shows up only as
    unexplained slowness; the peak active count is what distinguishes the two."""

    def __init__(self, origin: str, api_key: str | None):
        super().__init__(daemon=True)
        self.origin = origin
        self.api_key = api_key
        self._stop = threading.Event()
        self.max_active = 0
        self.max_decode_tps = 0.0
        self.max_prefill_tps = 0.0
        self.vram_bytes = 0
        self.samples = 0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                doc = get_json(f"{self.origin}/v1/stats", self.api_key, timeout=5)
            except (OSError, ValueError):
                self._stop.wait(STATS_POLL_S)
                continue
            reqs = doc.get("requests") or {}
            tp = doc.get("throughput") or {}
            self.max_active = max(self.max_active, int(reqs.get("active") or 0))
            self.max_decode_tps = max(self.max_decode_tps, float(tp.get("decode_tps") or 0.0))
            self.max_prefill_tps = max(self.max_prefill_tps, float(tp.get("prefill_tps") or 0.0))
            self.vram_bytes = int(doc.get("vram_bytes") or self.vram_bytes)
            self.samples += 1
            self._stop.wait(STATS_POLL_S)

    def stop(self) -> None:
        self._stop.set()


def build_body(model_id: str, prompt: str, args: argparse.Namespace) -> tuple[str, dict]:
    body = {
        "model": model_id,
        "max_tokens": args.decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.endpoint == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
        if args.thinking != "auto":
            body["chat_template_kwargs"] = dict(THINKING_KWARGS[args.thinking])
        return "/v1/chat/completions", body
    body["prompt"] = prompt
    return "/v1/completions", body


def stream_request(origin: str, model_id: str, prompt: str, args: argparse.Namespace) -> dict:
    """One streamed request: per-token arrival stamps, the text, and the exact usage block."""
    path, body = build_body(model_id, prompt, args)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(f"{origin}{path}", data=json.dumps(body).encode(), headers=headers)
    stamps: list[float] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        sys.exit(f"[bench] request failed: HTTP {e.code}: {e.read()[:500]!r}")
    # Bytes, not a text-mode reader: the server sends ensure_ascii=False JSON with no charset on
    # text/event-stream, and a text reader would decode that as latin-1.
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue  # blank separators between events
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta")
                if delta is not None:
                    text = delta.get("reasoning_content") or delta.get("content")
                else:
                    text = choice.get("text")
                if text:
                    stamps.append(now)
    end = time.perf_counter()
    if usage is None:
        sys.exit("[bench] stream ended without a usage chunk; is this a FreeToken server?")
    if len(stamps) < 2:
        sys.exit(f"[bench] need >=2 token events to measure decode, got {len(stamps)}")
    return {"t0": t0, "end": end, "stamps": stamps, "usage": usage}


def filler_body(words: int) -> str:
    return " ".join([FILLER_SENTENCE] * max(1, -(-words // WORDS_PER_FILLER)))


def make_prompt(body: str, nonce: str) -> str:
    # The nonce leads because the radix cache matches from the prompt head: a unique head makes
    # the whole prompt a cold prefill.
    return f"bench {nonce} {body}" if nonce else body


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def request_metrics(r: dict) -> dict:
    stamps, usage = r["stamps"], r["usage"]
    completion, prompt = usage["completion_tokens"], usage["prompt_tokens"]
    steps = max(1, completion - 1)
    window = stamps[-1] - stamps[0]
    gaps = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_ms": (stamps[0] - r["t0"]) * 1e3,
        "decode_tok_s": steps / window if window > 0 else float("nan"),
        "e2e_ms": (r["end"] - r["t0"]) * 1e3,
        "itl_p50_ms": pct(gaps, 0.50),
        "itl_p95_ms": pct(gaps, 0.95),
    }


def run_level(origin: str, model_id: str, concurrency: int, count: int, body: str,
              rng: random.Random, args: argparse.Namespace,
              poll: bool = True) -> tuple[list[dict], float, StatsPoller | None]:
    """--count requests as full waves of --concurrency; returns metrics, wall time, and the poller.

    Wave wall time is measured around the barrier release, so the aggregate rate charges the run
    for queueing and for the slowest stream in every wave."""
    prompts = [make_prompt(body, "" if args.reuse_prompt else f"{rng.getrandbits(48):012x}")
               for _ in range(concurrency * count)]
    poller = StatsPoller(origin, args.api_key) if poll else None
    if poller:
        poller.start()
    rows: list[dict] = []
    wall = 0.0
    try:
        for i in range(count):
            wave = prompts[i * concurrency:(i + 1) * concurrency]
            barrier = threading.Barrier(len(wave))

            def fire(prompt: str) -> dict:
                barrier.wait(timeout=WAVE_BARRIER_TIMEOUT_S)
                return stream_request(origin, model_id, prompt, args)

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                results = list(pool.map(fire, wave))
            wall += time.perf_counter() - t0
            rows.extend(request_metrics(r) for r in results)
    finally:
        if poller:
            poller.stop()
    return rows, wall, poller


def level_row(concurrency: int, rows: list[dict], wall: float, poller: StatsPoller | None) -> dict:
    decode = [r["decode_tok_s"] for r in rows]
    ttft = [r["ttft_ms"] for r in rows]
    prompt_total = sum(r["prompt_tokens"] for r in rows)
    completion_total = sum(r["completion_tokens"] for r in rows)
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "wall_s": wall,
        "ttft_ms_mean": sum(ttft) / len(ttft),
        "ttft_ms_p50": pct(ttft, 0.50),
        "ttft_ms_p95": pct(ttft, 0.95),
        "ttft_ms_max": max(ttft),
        "prefill_tok_s": prompt_total / wall if wall > 0 else float("nan"),
        "decode_tok_s_mean": sum(decode) / len(decode),
        "decode_tok_s_min": min(decode),
        "decode_tok_s_p50": pct(decode, 0.50),
        "output_tok_s": completion_total / wall if wall > 0 else float("nan"),
        "requests_per_s": len(rows) / wall if wall > 0 else float("nan"),
        "itl_ms_p50": pct([r["itl_p50_ms"] for r in rows], 0.50),
        "itl_ms_p95": pct([r["itl_p95_ms"] for r in rows], 0.95),
        "active_max": poller.max_active if poller else None,
        "stats_samples": poller.samples if poller else 0,
        "engine_decode_tok_s_peak": poller.max_decode_tps if poller else None,
        "engine_prefill_tok_s_peak": poller.max_prefill_tps if poller else None,
        "vram_gib": (poller.vram_bytes if poller else 0) / 2**30,
    }


def print_level(r: dict, per_request: list[dict] | None) -> None:
    print(f"\n==== concurrency {r['concurrency']}: {r['requests']} requests ====", flush=True)
    if per_request:
        for i, q in enumerate(per_request):
            print(f"  req {i:2d}          : ttft {q['ttft_ms']:7.1f} ms | decode {q['decode_tok_s']:7.2f} tok/s "
                  f"| {q['completion_tokens']} tok in {q['e2e_ms'] / 1e3:.2f} s", flush=True)
    print(f"  TTFT             : mean {r['ttft_ms_mean']:7.1f} / p50 {r['ttft_ms_p50']:7.1f} "
          f"/ p95 {r['ttft_ms_p95']:7.1f} / max {r['ttft_ms_max']:7.1f} ms "
          f"({r['prompt_tokens'] // r['requests']} prompt tok, {r['prefill_tok_s']:.0f} tok/s incl. queue)")
    print(f"  decode per stream: mean {r['decode_tok_s_mean']:6.2f} / p50 {r['decode_tok_s_p50']:6.2f} "
          f"/ min {r['decode_tok_s_min']:6.2f} tok/s (ITL p50 {r['itl_ms_p50']:.2f} / p95 {r['itl_ms_p95']:.2f} ms)")
    print(f"  aggregate output : {r['output_tok_s']:6.2f} tok/s over {r['wall_s']:.2f} s wall "
          f"({r['completion_tokens']} completion tok, {r['requests_per_s']:.2f} req/s)")
    if r["active_max"] is not None:
        # The server counts the /v1/stats call itself, so a clean run reads C+1 at the boundary.
        shared = " (foreign traffic on this server)" if r["active_max"] > r["concurrency"] + 1 else ""
        print(f"  engine self-view : decode peak {r['engine_decode_tok_s_peak']:.1f} / "
              f"prefill peak {r['engine_prefill_tok_s_peak']:.1f} tok/s, "
              f"active max {r['active_max']} (incl. this poll), vram {r['vram_gib']:.2f} GiB"
              f" ({r['stats_samples']} /v1/stats samples){shared}")


def print_sweep(levels: list[dict]) -> None:
    print("\n==== concurrency sweep ====", flush=True)
    print(f"{'conc':>5} {'agg tok/s':>11} {'decode mean':>12} {'decode min':>11} "
          f"{'TTFT p50':>10} {'TTFT p95':>10} {'ITL p50':>9} {'active':>7}")
    for r in levels:
        print(f"{r['concurrency']:>5} {r['output_tok_s']:>11.2f} {r['decode_tok_s_mean']:>12.2f} "
              f"{r['decode_tok_s_min']:>11.2f} {r['ttft_ms_p50']:>10.1f} {r['ttft_ms_p95']:>10.1f} "
              f"{r['itl_ms_p50']:>9.2f} {r['active_max'] if r['active_max'] is not None else '-':>7}")


def run_bench(origin: str, args: argparse.Namespace, body: str, rng: random.Random) -> list[dict]:
    model_id = get_json(f"{origin}/v1/models", args.api_key)["data"][0]["id"]
    try:
        stats0 = get_json(f"{origin}/v1/stats", args.api_key, timeout=10)
    except (OSError, ValueError):  # an OpenAI-compatible server without the control endpoints
        stats0 = None
    sampling_src = apply_server_sampling(args, stats0)
    print(f"[bench] target {origin} model_id={model_id} endpoint={args.endpoint} "
          f"decode={args.decode} prompt~{args.prompt_words}w cache="
          f"{'warm' if args.reuse_prompt else 'cold'}\n"
          f"[bench] sampling temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"({sampling_src})", flush=True)
    if args.warmup:
        warm_waves = max(1, -(-args.warmup // args.sweep[0]))
        print(f"[bench] warming up ({warm_waves * args.sweep[0]} discarded requests)", flush=True)
        run_level(origin, model_id, args.sweep[0], warm_waves, body, rng, args, poll=False)
    levels: list[dict] = []
    for concurrency in args.sweep:
        waves = max(1, args.requests // concurrency)
        if waves * concurrency != args.requests:
            print(f"[bench] --requests {args.requests} is not a multiple of concurrency {concurrency}: "
                  f"measuring {waves * concurrency} in {waves} full waves", flush=True)
        rows, wall, poller = run_level(origin, model_id, concurrency, waves, body, rng, args)
        row = level_row(concurrency, rows, wall, poller)
        row.update({
            "model": args.model or model_id,
            "origin": origin,
            "endpoint": args.endpoint,
            "prompt_words": args.prompt_words,
            "prompt_cache": "warm" if args.reuse_prompt else "cold",
            "decode_tokens": args.decode,
            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
            "sampling_src": sampling_src,
        })
        levels.append(row)
        print_level(row, rows if (concurrency == 1 and len(args.sweep) == 1) else None)
    if len(levels) > 1:
        print_sweep(levels)
    return levels


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    body = Path(args.prompt_file).read_text() if args.prompt_file else filler_body(args.prompt_words)
    rng = random.Random(args.seed)

    if args.model:
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        fd, log_path = tempfile.mkstemp(prefix="bench-speed-", suffix=".log")
        cmd = serve_cmd(args, port)
        print(f"[bench] spawning: {' '.join(cmd)}\n[bench] server log: {log_path}", flush=True)
        with os.fdopen(fd, "wb") as log_f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
            pump.start()
            try:
                wait_ready(origin, proc, log_path, args.server_timeout)
                levels = run_bench(origin, args, body, rng)
            finally:
                stop_server(proc)
                pump.join(timeout=10)
        for row in levels:
            row["server_log"] = log_path
    else:
        origin = args.server.rstrip("/")
        check_target(origin, args.api_key)
        levels = run_bench(origin, args, body, rng)

    if args.json_out:
        with open(args.json_out, "a") as f:
            for row in levels:
                f.write(json.dumps(row) + "\n")
        print(f"\n[bench] wrote {len(levels)} row(s) to {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
