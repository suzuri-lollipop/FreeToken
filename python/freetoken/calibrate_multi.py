"""Tune the concurrent-prefill knobs (FREETOKEN_PIPED_PREFILL / MIXED / ADMIT_*) for THIS PC.

The single-stream calibrator (`calibrate_lookup`) makes one request fast. This one makes
MANY requests fast together: while a long prompt is being prefilled, do the running
decodes keep streaming? The mechanism is machine-dependent in exactly the way lookup's
width is -- chunk cost, PCIe share between prefill and decode expert streaming, and the
stream priorities all decide whether piping pays here -- so the tool measures the real
server on this PC instead of trusting the defaults measured on one box.

Arms are probed at the server's PRODUCTION sampling by default (the request omits
temperature, so sampling_defaults=model and the high-temperature guard apply exactly as
they do for real traffic); --temperature 0.0 switches to deterministic greedy, whose
lower variance suits a re-run that must diff against an older report.

Every arm answers the SAME staggered workload:

  - `victims` short greedy requests (a 128-token prompt, a fixed 256-token answer)
    launched 2 s apart, then ONE long aggressor (a 16384-token prompt) 6 s in, while
    the victims are mid-decode;
  - what is scored: the worst inter-token gap of the victims AFTER the aggressor is
    admitted (the stall a human feels), the aggressor's TTFT, and aggregate tok/s;
  - the arms: `none` (the inline prefill/decode interleave), `mixed` (decode rows ride
    the chunk's forward, FREETOKEN_MIXED_PREFILL_TOKENS=1024), `piped` (chunks run on a
    worker thread and a side stream while decode replays keep the high-priority engine
    stream), and `piped-sf` (piping plus short-first admission).

A candidate wins only when it cuts the worst stall by at least STALL_GAIN_S (or lands
below the human-perceptible STALL_FLOOR while the baseline is above it) AND gives up at
most TPS_LOSS_MAX of aggregate throughput. The throughput guard is deliberately loose:
concurrency trades prompt-side throughput for the running streams' gaps -- that is the
feature -- so only a catastrophic loss (a link too slow to share) rejects an arm.

Arm validity is read from the arm's own server log (the "piped prefill enabled" line,
the [chunktime] mix= markers, the "piping disabled" fallback): an arm whose mechanism
never activated is EXCLUDED, not silently scored as if it did.

The output is the same pair as the lookup tool: a JSON report and an executable script,
with the winner's full knob set (explicit zeros included) in `env`, written atomically.
Feed it an existing launch config to keep the profile's argv:

    python -m freetoken.calibrate_multi --config freetoken-<model>.json

The server must not already be running on the chosen port; this tool owns it for the
duration. Expect ~15 min full (4 arms x one engine start), ~5 min --quick.
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
STALL_GAIN_S = 0.5
STALL_FLOOR_S = 0.5
TPS_LOSS_MAX = 0.25
TIE_S = 0.15
STALL_EVENT_S = 0.5  # the launcher bench's stall threshold; keep the two in step


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
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content")
                if piece:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    arrivals.append(now)
    return {"ttft": ttft or 0.0, "arrivals": arrivals, "tokens": len(arrivals),
            "usage_tokens": usage_tokens, "end": time.perf_counter()}


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
    tokens = sum(r["tokens"] for r in results.values())
    scored |= {
        "ttft_long": agg["ttft"],
        "ttft_short": statistics.mean([r["ttft"] for r in victim.values()]) if victim else 0.0,
        "out_tps": tokens / max(wall, 1e-6), "tokens": tokens, "wall": wall,
    }
    return scored


def score_window(victims: dict[str, dict], open_at: float) -> dict:
    """Pure: the victims' inter-token gaps after the aggressor was admitted."""
    worst = count = 0
    secs = 0.0
    for r in victims.values():
        for a, b in zip(r["arrivals"], r["arrivals"][1:]):
            if b < open_at:  # a gap ENDING before the window never involved the aggressor
                continue
            gap = b - a
            if gap > STALL_EVENT_S:
                count += 1
                secs += gap
            worst = max(worst, gap)
    return {"stall_worst": worst, "stall_count": count, "stall_secs": secs}


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

    def line(self) -> str:
        return (f"{self.label:9s} stall={self.stall_worst:5.2f}s x{self.stall_count:3d} "
                f"ttft={self.ttft_long:6.2f}s short={self.ttft_short:5.2f}s "
                f"tps={self.out_tps:6.1f}"
                + (f"  [{self.note}]" if self.note else ""))


def arm_active(label: str, log_text: str) -> tuple[bool, str]:
    """Did the arm's mechanism actually engage? Read the arm's own server log."""
    if label in ("piped", "piped-sf"):
        if "piped prefill enabled" in log_text:
            return True, ""
        if "piping disabled" in log_text:
            why = next((ln.split("piping disabled")[0].strip()[-80:]
                        for ln in log_text.splitlines() if "piping disabled" in ln), "")
            return False, f"piping disabled: {why}"
        return False, "no 'piped prefill enabled' line"
    if label == "mixed":
        if " mix=" in log_text:
            return True, ""
        return False, "no [chunktime] mix= line (mixed steps never ran)"
    return True, ""


def evaluate_multi(spec: dict, label: str, env: dict[str, str], cfg: dict,
                   reps: int, log_dir: Path, port: int,
                   temperature: float | None = None) -> MultiScore:
    run_spec = {**spec, "port": port,
                "argv": argv_with_port(list(spec["argv"]), port)}
    full_env = {**env, "FREETOKEN_CHUNK_TIMING": "1"}
    last_error: Exception | None = None
    for attempt in range(3):
        arm_port = port + 2 * attempt
        if not _port_free(arm_port) or not _port_free(arm_port + 1):
            continue
        items = workload(cfg)
        per: list[dict] = []
        log_name = f"{_sanitize(label)}.log" if attempt == 0 else f"{_sanitize(label)}-try{attempt + 1}.log"
        try:
            with Server(run_spec, full_env, log_dir / log_name) as server:
                model = spec.get("model_name") or "default"
                # warm-up doubles as the liveness probe (see calibrate_lookup)
                _stream_chat(server, model, "Repeat after me exactly: warm up.", 16,
                             time.perf_counter(), 0.0, timeout=WARMUP_TIMEOUT,
                             temperature=temperature)
                print(f"    {label}: running the staggered workload "
                      f"({len(items)} requests x {max(1, reps)} reps)", flush=True)
                for _ in range(max(1, reps)):
                    per.append(run_workload(server, model, items, temperature))
            log_text = (log_dir / log_name).read_text(errors="replace")
            break
        except (ServerError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            time.sleep(3)
    else:
        raise ServerError(f"{label}: three attempts failed (last: {last_error})") from last_error

    med = {k: statistics.median([p[k] for p in per]) for k in per[0]}
    valid, note = arm_active(label, log_text)
    return MultiScore(
        label=label, env=dict(env),
        stall_worst=float(med["stall_worst"]), stall_count=int(med["stall_count"]),
        stall_secs=float(med["stall_secs"]), ttft_long=float(med["ttft_long"]),
        ttft_short=float(med["ttft_short"]), out_tps=float(med["out_tps"]),
        tokens=int(med["tokens"]), wall=float(med["wall"]), reps=per,
        valid=valid, note=note,
    )


def choose_multi(scores: list[MultiScore]) -> tuple[str, str]:
    """The selection contract (see module docstring). Pure so the unit tests pin it."""
    by = {s.label: s for s in scores}
    base = by["none"]
    cands = [s for s in scores if s.label != "none" and s.valid]
    if not cands:
        return "none", "no concurrency arm activated on this machine"
    best = min(cands, key=lambda s: (round(s.stall_worst / TIE_S), -s.out_tps, s.ttft_short))
    gained = base.stall_worst - best.stall_worst
    floor_win = best.stall_worst <= STALL_FLOOR_S and base.stall_worst > STALL_FLOOR_S
    if not (gained >= STALL_GAIN_S or floor_win):
        return "none", (f"{best.label} cuts the worst stall only {gained:.2f}s "
                        f"(need {STALL_GAIN_S}s or sub-{STALL_FLOOR_S}s floor)")
    if best.out_tps < base.out_tps * (1 - TPS_LOSS_MAX):
        return "none", (f"{best.label} costs {100 * (1 - best.out_tps / base.out_tps):.0f}% "
                        f"aggregate throughput (max {100 * TPS_LOSS_MAX:.0f}%)")
    return best.label, (f"worst stall {base.stall_worst:.2f}s -> {best.stall_worst:.2f}s, "
                        f"tps {base.out_tps:.1f} -> {best.out_tps:.1f}")


def render_script_multi(spec: dict, chosen: dict | None) -> str:
    """The runnable pair to the lookup tool's script: same argv, winner env exported."""
    env = {str(k): str(v) for k, v in (spec.get("env") or {}).items() if k not in MULTI_KEYS}
    if chosen:
        env.update({str(k): str(v) for k, v in (chosen.get("env") or {}).items()})
    cwd = spec.get("cwd") or str(ROOT)
    python = spec.get("python") or sys.executable
    argv = argv_with_port(list(spec["argv"]), int(spec.get("port") or DEFAULT_PORT))
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


def calibrate_multi(spec: dict, a: argparse.Namespace) -> dict:
    out_dir = Path(a.out_dir).expanduser().resolve()
    log_dir = out_dir / "logs"
    cfg = {"short": 128, "long": 2048 if a.smoke else a.long,
           "out": 32 if a.smoke else a.out_tokens, "gap": 6.0,
           "victims": victims_from_spec(spec)}
    reps = 1 if (a.quick or a.smoke) else max(1, a.reps)
    arms = arms_for(spec, a.smoke)
    base_port = int(spec.get("port") or DEFAULT_PORT)
    spec["argv"] = ensure_distributed_timeout(spec["argv"])
    # strip stale multi-knob values from the launch env: the arms own them
    spec = {**spec, "env": {k: v for k, v in (spec.get("env") or {}).items()
                            if k not in MULTI_KEYS}}

    scores: list[MultiScore] = []
    next_port = [base_port]

    def take_port() -> int:
        while True:
            port = next_port[0]
            next_port[0] += 2
            if _port_free(port) and _port_free(port + 1):
                return port

    for label, env, skip in arms:
        if skip:
            print(f"  skipping {label}: {skip}", flush=True)
            scores.append(MultiScore(label=label, env=env, stall_worst=0.0,
                                     stall_count=0, stall_secs=0.0, ttft_long=0.0,
                                     ttft_short=0.0, out_tps=0.0, tokens=0, wall=0.0,
                                     valid=False, note=skip))
            continue
        print(f"  measuring {label} ...", flush=True)
        score = evaluate_multi(spec, label, env, cfg, reps, log_dir, take_port(),
                               a.temperature)
        print("    " + score.line(), flush=True)
        scores.append(score)

    label, note = choose_multi(scores)
    by = {s.label: s for s in scores}
    base = by["none"]
    chosen = {
        "label": label, "env": winner_env(label),
        "note": note,
        "stall_worst": by[label].stall_worst, "out_tps": by[label].out_tps,
        "gain_stall_s": round(base.stall_worst - by[label].stall_worst, 2),
        "gain_tps_pct": round(100 * (by[label].out_tps / base.out_tps - 1), 1)
        if base.out_tps > 0 else 0.0,
    }
    return {
        "schema_version": 1, "backend": "freetoken-multi",
        "hardware": hardware(), "workload": {**cfg, "reps": reps},
        "arms": [{k: v for k, v in s.__dict__.items() if k != "reps"} | {
            "env": s.env} for s in scores],
        "chosen": chosen, "multi_enabled": label != "none",
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

    r = subprocess.run(["pgrep", "-af", "-m freetoken"], capture_output=True, text=True)
    return parse_live_servers(r.stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Tune concurrent prefill (piped/mixed/short-first) on this PC; "
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
    script_text = render_script_multi(spec, chosen if result["multi_enabled"] else None)
    json_path = out_dir / f"freetoken-multi-{tag}.json"
    script_path = out_dir / f"run-freetoken-multi-{tag}.sh"
    _atomic_write(json_path, json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    _atomic_write(script_path, script_text)
    script_path.chmod(0o755)

    print(f"\nchosen: {chosen['label']} -- {chosen['note']}", flush=True)
    if not result["multi_enabled"]:
        print("the concurrency knobs stay off on this machine; the script starts without them",
              flush=True)
    print(f"config: {json_path}")
    print(f"script: {script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
