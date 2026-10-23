"""Tune --speculative lookup for THIS PC and write its run config and start script.

The drafter's best width is a property of the machine, not of the model: a verify is `1 + K`
extend rows, and what those rows cost relative to one plain decode step decides whether K=2 or
K=6 pays.  `engine/lookup.py` has defaults measured on one box (2x RTX PRO 4000 Blackwell), and
they are a starting point, not a law.  This tool measures the real server here:

  - the server is started once per candidate with `FREETOKEN_LOOKUP_*` in its environment (the
    values are read when a request's LookupState is built, so changing them needs a restart);
  - every candidate answers the same four prompts after a warm-up that fills the prefix cache:
    two code edits that copy most of their input, a prose rewrite that copies a long spec, and
    a control with almost no repetition (the control catches a width that only helps copy);
  - candidates are visited in coordinate order (width, minimum match, minimum context,
    acceptance floor), each value measured `--reps` times per prompt, and the score is all
    completion tokens over all wall time.  A candidate is kept only when it beats both the
    default and the no-spec baseline by MIN_GAIN; on a PC where lookup loses, the tool says so
    and writes a script without it.  The parallel sweep then chooses `--max-running-requests`,
    and lookup stays enabled only if it also beats plain batch decoding at that ceiling.

The output is the Strata-style pair: a JSON launch config with the measured `env` and the
`--speculative lookup` argument, and an executable ``run-freetoken-lookup-<tag>.sh`` that exports
those variables and execs the same server.  Both are written atomically.  Requests use a fixed
`max_tokens` with `ignore_eos`, so an arm that stops sooner is not credited for doing less work.

The input can be an existing config JSON (``--config``) so a profile's full argv is preserved:

    {
      "python": "/path/.venv/bin/python",
      "argv": ["-m", "freetoken", "--model-path", "...", "--served-model-name", "...",
               "--port", "1920", "--tp-size", "2"],
      "cwd": "/path/to/fork",
      "env": {},
      "port": 1920,
      "model_name": "RadixArk/Qwen3.8-Flash-Next"
    }

or the same arguments on the command line:

    python -m freetoken.calibrate_lookup --model-path <snapshot> \
      --served-model-name RadixArk/Qwen3.8-Flash-Next --port 1920 \
      --server-args "--tp-size 2 --max-running-requests 4 --memory-ratio 0.88 \
        --num-tokens 131072 --moe-strategy offload --moe-cache-auto --moe-cpu-layers 0 \
        --kv-cache-dtype fp8 --prefill-decode-interval 4"

The server must not already be running on the chosen port; this tool owns it for the duration.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
MIN_GAIN = 0.03
# A lookup launch that is within this much of plain batch decoding at the chosen parallel
# ceiling is still worth keeping: it costs ~nothing there and wins single-request bursts.
MAX_PARALLEL_LOSS = 0.02
DEFAULT_PORT = 1920
DEFAULT_MAX_TOKENS = 192
DEFAULT_REPS = 2
STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 120
WARMUP_TIMEOUT = 30.0
STOP_TIMEOUT = 60.0
_LIVE_PROCS: list[subprocess.Popen] = []


def _stop_live() -> None:
    """A signal or an unhandled error must not leave the calibration's server holding the VRAM."""
    live = [p for p in _LIVE_PROCS if p.poll() is None]
    for proc in live:
        try:
            if hasattr(os, "killpg"):
                pgid = os.getpgid(proc.pid)
                if pgid != os.getpgid(0):
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()
            else:
                proc.terminate()
        except (OSError, ProcessLookupError):
            pass
    for proc in live:
        try:
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass


atexit.register(_stop_live)

KS = (1, 2, 3, 4, 5, 6)
MATCHES = (3, 4, 5)
CONTEXTS = (0, 128, 256, 512)
ACCEPTANCE_FLOORS = (0.6, 0.7, 0.8)
QUICK_KS = (1, 4, 6)
QUICK_MATCHES = (3, 5)
QUICK_CONTEXTS = (128, 256)
QUICK_ACCEPTANCE = (0.7, 0.8)


# ------------------------------------------------------------------------------------------------ prompts
CODE_SOURCE = '''\
class SuffixDrafter:
    """Propose the tokens that followed the most recent occurrence of the current suffix."""

    WAYS = 4

    def __init__(self, min_match=3, max_match=32, capacity=1 << 17):
        self.min_match = min_match
        self.max_match = max_match
        self.capacity = capacity
        self.table = {}
        self.hist = []
        self.last_match = 0

    def append(self, tokens):
        for token in tokens:
            self.hist.append(int(token))
            end = len(self.hist) - 1
            if end < 2:
                continue
            key = (self.hist[end - 2] << 36) | (self.hist[end - 1] << 18) | self.hist[end]
            ends = self.table.get(key)
            if ends is None:
                if len(self.table) >= self.capacity:
                    continue
                self.table[key] = [end]
            else:
                ends.append(end)
                if len(ends) > self.WAYS:
                    del ends[0]

    def propose(self, max_draft):
        n = len(self.hist)
        if n < self.min_match:
            return []
        key = (self.hist[n - 3] << 36) | (self.hist[n - 2] << 18) | self.hist[n - 1]
        ends = self.table.get(key) or []
        best = []
        best_len = 0
        for end in reversed(ends):
            length = 3
            while (length < self.max_match and end - length >= 0
                   and self.hist[end - length] == self.hist[n - 1 - length]):
                length += 1
            if length > best_len:
                best_len = length
                best = self.hist[end + 1:end + 1 + max_draft]
        self.last_match = best_len if best_len >= self.min_match else 0
        return best

    def reset(self):
        self.table.clear()
        self.hist.clear()
        self.last_match = 0
'''

CODE_PROMPT = (
    "次の Python ファイルを、クラス名 `SuffixDrafter` を `SuffixIndex` に変え、"
    "それ以外は一切変えずに、完全なファイルとして出力してください。"
    "説明や Markdown のフェンスは付けず、コードだけを出力してください。\n\n" + CODE_SOURCE
)

PROSE = '''\
社内エージェント基盤の運用手順です。すべての担当者はこの順序を守ってください。

1. すべての推論リクエストは 1919 番ポートの FreeToken サーバーに送る。
2. コード編集では対象ファイル全体を出力し、差分だけにしない。
3. 変更点はタイムアウトを 30 秒から 60 秒にすることだけとする。
4. 失敗時は 3 回まで再試行し、それでも失敗したら人間にエスカレーションする。
5. ログにはリクエスト ID とトークン数を必ず残す。
6. モデルの出力が途中で切れた場合は、同じプロンプトで続きを要求する。
7. 顧客データを含むプロンプトは外部検索に送らない。
8. 1 日の終わりに使用トークン数を集計し、翌朝の朝会で共有する。
9. モデルの更新は検証環境で 24 時間動かしてから本番に反映する。
10. 障害時はまず影響範囲を確認し、次にロールバック可否を判断する。
11. シークレットは環境変数から読み、リポジトリには絶対にコミットしない。
12. ベンチマークは同じプロンプト、同じ温度、同じ最大トークン数で比較する。
13. 変更を加えたら、必ず起動スクリプトと設定 JSON の両方を更新する。
14. 設定 JSON は人が読める形で保存し、差分レビューを必須にする。
15. 起動スクリプトは実行権限を付け、絶対パスで参照する。
16. モデルの重みは共有キャッシュに置き、各環境で重複させない。
17. GPU のメモリ使用量を起動前後で記録し、異常な増加を検知する。
18. 推論サーバーはヘルスチェックが通ってからトラフィックを受ける。
19. 応答が 60 秒を超えた場合はタイムアウトとして扱い、再試行する。
20. すべての設定変更は変更履歴に残し、理由と測定値を添える。
'''

PROSE_PROMPT = (
    "以下の運用仕様のタイムアウトを 30 秒から 60 秒に変え、それ以外は一切変えずに、"
    "仕様全文をそのまま出力してください。説明は不要です。\n\n" + PROSE
)

CONTROL_PROMPT = "GPU の冷却ファンについて、五七五の俳句を 5 句作ってください。"

CODE_PROMPT_2 = (
    "次の Python ファイルを、クラス名 `SuffixDrafter` を `SuffixIndex`、"
    "メソッド `append` を `push` に変え、それ以外は一切変えずに、完全なファイルとして"
    "出力してください。説明や Markdown のフェンスは付けず、コードだけを出力してください。\n\n"
    + CODE_SOURCE
)


def prompts() -> list[dict]:
    return [
        {"name": "code-edit", "prompt": CODE_PROMPT},
        {"name": "prose-edit", "prompt": PROSE_PROMPT},
        {"name": "control", "prompt": CONTROL_PROMPT},
        {"name": "code-edit-2", "prompt": CODE_PROMPT_2},
    ]


def prompts_for(n: int) -> list[dict]:
    """`n` distinct prompts for a batch of `n`: the four bases, then variants with a unique tail."""
    base = prompts()
    out: list[dict] = []
    for i in range(max(1, n)):
        item = base[i % len(base)]
        if i < len(base):
            out.append(item)
        else:
            out.append({"name": f"{item['name']}-{i + 1}",
                        "prompt": item["prompt"] + f"\n\n(回答 {i + 1})"})
    return out


# ------------------------------------------------------------------------------------------------ helpers
def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _quote_env(value: Any) -> str:
    return shlex.quote(str(value))


def _server_args_without_spec(argv: list[str]) -> list[str]:
    """Drop --speculative (both spellings) and --port from a launch argv."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        item = str(argv[i])
        if item == "--speculative":
            i += 2
            continue
        if item.startswith("--speculative="):
            i += 1
            continue
        if item == "--port":
            i += 2
            continue
        if item.startswith("--port="):
            i += 1
            continue
        out.append(item)
        i += 1
    return out


def normalize_argv(argv: Iterable[str], port: int) -> list[str]:
    """A server argv with no speculative flag, exactly one --port, and the port value last."""
    out = _server_args_without_spec([str(x) for x in argv])
    return out + ["--port", str(int(port))]


def set_parallel(argv: Iterable[str], value: int) -> list[str]:
    """One --max-running-requests, at the end, with the calibrated value."""
    out: list[str] = []
    items = [str(x) for x in argv]
    i = 0
    while i < len(items):
        item = items[i]
        if item == "--max-running-requests":
            i += 2
            continue
        if item.startswith("--max-running-requests="):
            i += 1
            continue
        out.append(item)
        i += 1
    return out + ["--max-running-requests", str(int(value))]


def ensure_distributed_timeout(argv: Iterable[str], seconds: int = 180) -> list[str]:
    """A calibration run loads the model a dozen times; 60 s is too tight for a slow rank.

    The value is only added when the caller did not choose one, and it lands in the generated
    script too: a launch that needs 180 s at startup on this PC should not fail at 60 s later.
    """
    items = [str(x) for x in argv]
    if any(item == "--distributed-timeout" or item.startswith("--distributed-timeout=")
           for item in items):
        return items
    return items + ["--distributed-timeout", str(int(seconds))]


def current_parallel(argv: Iterable[str], default: int = 4) -> int:
    items = [str(x) for x in argv]
    try:
        return int(_arg_value(items, "--max-running-requests") or default)
    except ValueError:
        return default


def arm_argv(spec: dict, speculative: bool) -> list[str]:
    """The argv a measured arm runs: the base launch, plus lookup only for a lookup candidate."""
    return list(spec["argv"]) + (["--speculative", "lookup"] if speculative else [])


def render_script(spec: dict, chosen: dict | None) -> str:
    """The executable start script for a calibrated (or disabled) launch spec."""
    argv = normalize_argv(spec["argv"], spec.get("port", DEFAULT_PORT))
    env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
    if chosen and chosen.get("enabled_at_parallel", True):
        env.update({str(k): str(v) for k, v in chosen["env"].items()})
        argv = argv + ["--speculative", "lookup"]
    cwd = spec.get("cwd") or str(ROOT)
    python = spec.get("python") or sys.executable
    lines = [
        "#!/bin/sh",
        f"# generated by freetoken.calibrate_lookup.py on {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"cd {shlex.quote(str(cwd))}",
    ]
    for key in sorted(env):
        lines.append(f"export {key}={_quote_env(env[key])}")
    lines.append("exec " + " ".join(shlex.quote(x) for x in [python, *argv]))
    return "\n".join(lines) + "\n"


def _sanitize(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "model"


def hardware() -> dict:
    """GPU/CPU/RAM identity without importing torch (the calibration server needs the VRAM)."""
    gpus: list[str] = []
    for info in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
        try:
            for line in info.read_text(errors="replace").splitlines():
                if line.startswith("Model:"):
                    gpus.append(line.split(":", 1)[1].strip())
        except OSError:
            pass
    if not gpus:
        try:
            out = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            ).stdout
            gpus = sorted({ln.split(": ", 1)[1] for ln in out.splitlines()
                           if "VGA" in ln and "NVIDIA" in ln})
        except (OSError, subprocess.SubprocessError):
            pass
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram_gb = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except (OSError, ValueError):
        pass
    return {"gpu": gpus, "cpu": cpu, "ram_gb": ram_gb}


def spec_from_args(a: argparse.Namespace) -> dict:
    if a.config:
        spec = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    else:
        if not a.model_path:
            raise SystemExit("--model-path is required without --config")
        extra = shlex.split(a.server_args or "")
        name = (a.served_model_name or _arg_value(extra, "--served-model-name")
                or Path(a.model_path).name)
        spec = {
            "python": a.python or sys.executable,
            "argv": [
                "-m", "freetoken",
                "--model-path", str(Path(a.model_path).expanduser().resolve()),
                "--served-model-name", name,
            ] + extra,
            "cwd": str(ROOT),
            "env": {},
            "model_name": name,
        }
    if "argv" not in spec or not spec["argv"]:
        raise SystemExit("the launch spec needs a non-empty 'argv'")
    spec.setdefault("python", sys.executable)
    spec.setdefault("cwd", str(ROOT))
    spec.setdefault("env", {})
    # A previous calibration's variables must not leak into the baseline arm; every candidate
    # sets the three it sweeps explicitly.
    spec["env"] = {k: v for k, v in spec["env"].items()
                   if not str(k).startswith("FREETOKEN_LOOKUP_")}
    spec.setdefault(
        "model_name",
        a.served_model_name
        or _arg_value(spec["argv"], "--served-model-name")
        or Path(_arg_value(spec["argv"], "--model-path") or spec["argv"][0]).name,
    )
    spec["port"] = int(a.port or spec.get("port") or DEFAULT_PORT)
    spec["argv"] = normalize_argv(spec["argv"], spec["port"])
    # Make `python -m freetoken` work from a checkout even when the venv has no editable install.
    pp = spec["env"].get("PYTHONPATH")
    root_py = str(ROOT / "python")
    spec["env"]["PYTHONPATH"] = root_py + (os.pathsep + pp if pp else "")
    return spec


# ------------------------------------------------------------------------------------------------ server
class ServerError(RuntimeError):
    pass


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class Server:
    def __init__(self, spec: dict, env: dict[str, str], log_path: Path):
        self.spec = spec
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._log = None
        self.base = f"http://127.0.0.1:{int(spec['port'])}"

    def _tail(self, n: int = 40) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    def _wait_health(self) -> None:
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(
                    f"server exited with {self.proc.returncode} during startup\n{self._tail()}"
                )
            tail = self._tail(80)
            # A TP rank can die after the frontend has bound its port, so /health keeps saying
            # "ok" while the scheduler is gone. The Gloo timeout is the reliable marker; fail now
            # instead of waiting the full startup budget and then timing out on a request.
            if "Timed out waiting" in tail and "gloo" in tail:
                raise ServerError(f"scheduler died during startup\n{tail}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    doc = json.load(r)
                if doc.get("status") == "ok":
                    return
            except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
                pass
            time.sleep(2)
        raise ServerError(
            f"server did not become healthy in {STARTUP_TIMEOUT:.0f}s\n{self._tail()}"
        )

    def __enter__(self) -> "Server":
        port = int(self.spec["port"])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise ServerError(
                    f"port {port} is already serving; stop that server or pass --port"
                )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        env = {**os.environ, **self.spec.get("env", {}), **self.env}
        cmd = [self.spec.get("python") or sys.executable, *self.spec["argv"]]
        self._log.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        for k in sorted(self.env):
            self._log.write(f"# {k}={self.env[k]}\n")
        self._log.flush()
        self.proc = subprocess.Popen(
            cmd, cwd=self.spec.get("cwd") or str(ROOT), env=env,
            stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        _LIVE_PROCS.append(self.proc)
        try:
            self._wait_health()
        except Exception:
            self._stop()
            raise
        return self

    def post(self, path: str, body: dict, timeout: float = 900.0) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            return json.load(r)

    def get(self, path: str) -> dict:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(self.base + path, timeout=30) as r:
            return json.load(r)

    def _stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            # The server runs its TP ranks in the same new session, and terminating only the
            # launcher leaves a rank alive: the next start then loses the Gloo handshake with a
            # 60-second timeout in the scheduler's receive loop. Signal the whole group.
            self._signal_group(signal.SIGTERM)
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        if proc is not None and proc in _LIVE_PROCS:
            _LIVE_PROCS.remove(proc)
        self.proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        # The next server must not race the previous one's port or its freed VRAM.
        deadline = time.time() + 30
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", int(self.spec["port"]))) != 0:
                    break
            time.sleep(0.5)

    def _signal_group(self, sig: int) -> None:
        proc = self.proc
        if proc is None:
            return
        if hasattr(os, "killpg"):
            try:
                pgid = os.getpgid(proc.pid)
                if pgid != os.getpgid(0):          # never signal the calibrator's own group
                    os.killpg(pgid, sig)
                    return
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    def __exit__(self, *exc) -> None:
        self._stop()


# ------------------------------------------------------------------------------------------------ measurement
@dataclass
class Score:
    label: str
    env: dict[str, str]
    per_prompt: dict[str, float]
    tok_s: float
    tokens: int
    seconds: float
    acceptance: float | None = None
    drafted_ratio: float | None = None
    concurrency: int = 1

    def line(self) -> str:
        parts = " ".join(f"{k}={v:.1f}" for k, v in self.per_prompt.items())
        acc = "-" if self.acceptance is None else f"{self.acceptance:.3f}"
        par = f" x{self.concurrency}" if self.concurrency > 1 else ""
        return f"{self.label:14s}{par:3s} {self.tok_s:6.1f} tok/s  acc={acc}  ({parts})"


def _measure_once(server: Server, model: str, prompt: str, max_tokens: int,
                  timeout: float = REQUEST_TIMEOUT) -> tuple[int, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        # Fix the decode budget. A free-form answer (the control prompt) can hit EOS a few
        # tokens apart from run to run, and then a "faster" arm is just one that stopped
        # sooner. Every arm now emits exactly max_tokens, so tok/s compares decode work.
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    doc = server.post("/v1/chat/completions", body, timeout=timeout)
    elapsed = max(1e-6, time.perf_counter() - t0)
    usage = doc.get("usage") or {}
    tokens = int(usage.get("completion_tokens") or 0)
    return tokens, elapsed


def _measure_batch(server: Server, model: str, items: list[dict],
                   max_tokens: int) -> tuple[int, float, dict[str, float]]:
    """Send all items at once and time the batch: the caller warmed every prompt already."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = [pool.submit(_measure_once, server, model, item["prompt"], max_tokens)
                   for item in items]
        results = [f.result() for f in futures]
    seconds = max(1e-6, time.perf_counter() - t0)
    tokens = sum(t for t, _ in results)
    per_prompt = {item["name"]: t / s for item, (t, s) in zip(items, results) if s > 0}
    return tokens, seconds, per_prompt


def measured_argv(spec: dict, speculative: bool, port: int) -> list[str]:
    """The argv for one measured arm: fresh port, no stale flag, lookup only when asked."""
    argv = normalize_argv(spec["argv"], port)
    if speculative:
        argv += ["--speculative", "lookup"]
    return argv


def evaluate(spec: dict, label: str, env: dict[str, str], max_tokens: int,
             reps: int, log_dir: Path, speculative: bool = False,
             concurrency: int = 1, port: int | None = None) -> Score:
    model = spec.get("model_name") or "default"
    # Every arm answers the same four prompts, so the token count is the same at every
    # concurrency: N=2 sends two batches of two, N=1 sends four one at a time.
    items = prompts_for(max(concurrency, len(prompts())))
    base_port = int(port if port is not None else spec.get("port") or DEFAULT_PORT)
    last_error: Exception | None = None
    for attempt in range(3):
        arm_port = base_port + 2 * attempt
        if port is not None:
            # normalize_argv strips the speculative flag for the final launch script, so put it
            # back for a measured arm after the port rewrite.
            run_spec = {**spec, "port": arm_port,
                        "argv": measured_argv(spec, speculative, arm_port)}
        else:
            run_spec = {**spec, "argv": arm_argv(spec, speculative)}
        rates: dict[str, list[float]] = {item["name"]: [] for item in items}
        totals: list[int] = []
        walls: list[float] = []
        spec_stats: dict = {}
        log_name = f"{_sanitize(label)}.log" if attempt == 0 else f"{_sanitize(label)}-try{attempt + 1}.log"
        try:
            with Server(run_spec, env, log_dir / log_name) as server:
                for item in items:
                    # The warm-up doubles as a liveness probe: if a TP rank died after /health
                    # went green, this fails in 30 s instead of hanging a full request budget.
                    _measure_once(server, model, item["prompt"], 16, timeout=WARMUP_TIMEOUT)
                for _ in range(max(1, reps)):
                    total = 0
                    wall = 0.0
                    for i in range(0, len(items), concurrency):
                        chunk = items[i:i + concurrency]
                        tokens, seconds, per_prompt = _measure_batch(server, model, chunk, max_tokens)
                        total += tokens
                        wall += seconds
                        for name, rate in per_prompt.items():
                            rates[name].append(rate)
                    totals.append(total)
                    walls.append(wall)
                spec_stats = server.get("/v1/stats").get("spec") or {}
            break
        except (ServerError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            if attempt < 2:
                print(f"    {label} failed ({type(e).__name__}: {e}); retrying on port "
                      f"{arm_port + 2}", file=sys.stderr, flush=True)
                time.sleep(3)
    else:
        raise ServerError(f"{label}: three attempts failed (last: {last_error})") from last_error

    tokens = statistics.median(totals)
    seconds = statistics.median(walls)
    return Score(
        label=label, env=dict(env),
        per_prompt={name: statistics.median(v) for name, v in rates.items() if v},
        tok_s=statistics.median([t / s for t, s in zip(totals, walls) if s > 0]),
        tokens=tokens, seconds=seconds,
        acceptance=spec_stats.get("acceptance_rate"),
        drafted_ratio=spec_stats.get("drafted_ratio"),
        concurrency=concurrency,
    )

def parse_parallels(text: str) -> list[int]:
    values: set[int] = set()
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as e:
            raise SystemExit(f"--parallel: {part!r} is not an integer") from e
        if value < 1:
            raise SystemExit("--parallel values must be at least 1")
        values.add(value)
    return sorted(values)


def _candidate_env(k: int, match: int, context: int, floor: float) -> dict[str, str]:
    return {
        "FREETOKEN_LOOKUP_MAX_DRAFT": str(k),
        "FREETOKEN_LOOKUP_MIN_MATCH": str(match),
        "FREETOKEN_LOOKUP_MIN_CONTEXT": str(context),
        "FREETOKEN_LOOKUP_MIN_ACCEPTANCE": str(floor),
    }


def choose(scores: dict[tuple, Score], baseline: Score, default_key: tuple,
           min_gain: float = MIN_GAIN) -> tuple | None:
    """Best measured key, the default when the gain is inside noise, or None when lookup loses."""
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k].tok_s)
    if scores[best].tok_s <= baseline.tok_s * (1.0 + min_gain):
        return None
    if best != default_key and default_key in scores:
        if scores[best].tok_s <= scores[default_key].tok_s * (1.0 + min_gain):
            return default_key
    return best


def calibrate(spec: dict, a: argparse.Namespace) -> dict:
    out_dir = Path(a.out_dir).expanduser().resolve()
    log_dir = out_dir / "logs"
    if a.smoke:
        ks, matches, contexts, floors = (4, 5), (3,), (256,), (0.8,)
    elif a.quick:
        ks, matches, contexts, floors = QUICK_KS, QUICK_MATCHES, QUICK_CONTEXTS, QUICK_ACCEPTANCE
    else:
        ks, matches, contexts, floors = KS, MATCHES, CONTEXTS, ACCEPTANCE_FLOORS
    reps = 1 if (a.quick or a.smoke) else max(1, a.reps)
    default_key = (4, 3, 256, 0.8)
    # The width/match/context search runs with the gate off (0), so an arm is not punished for
    # a policy value that is itself being chosen afterwards. The floor sweep then measures the
    # production gate at the winning shape, and the default is measured again at 0.8 for a fair
    # "did it beat the default" comparison.
    raw_floor = 0.0
    cache: dict[tuple, Score] = {}
    # The engine's Gloo store listens on port+1. Reusing one port across a dozen restarts
    # eventually leaves a rank waiting on a stale store (a 60 s timeout inside the receive
    # loop), so every arm gets a fresh pair and the step of two keeps port+1 out of the way.
    next_port = [int(spec.get("port") or DEFAULT_PORT)]

    def take_port() -> int:
        while True:
            port = next_port[0]
            next_port[0] += 2
            if _port_free(port) and _port_free(port + 1):
                return port

    spec["argv"] = ensure_distributed_timeout(spec["argv"])

    def get(k: int, match: int, context: int, floor: float) -> Score:
        key = (k, match, context, floor)
        if key in cache:
            return cache[key]
        label = f"k={k} m={match} c={context} a={floor:g}"
        print(f"  measuring {label} ...", flush=True)
        score = evaluate(spec, label, _candidate_env(*key), a.max_tokens, reps, log_dir,
                         speculative=True, port=take_port())
        cache[key] = score
        print("    " + score.line(), flush=True)
        return score

    print("baseline (no speculation, first):", flush=True)
    baseline_start = evaluate(spec, "baseline-start", {}, a.max_tokens, reps, log_dir,
                              port=take_port())
    print("  " + baseline_start.line(), flush=True)

    print(f"width: {list(ks)} (acceptance gate off)", flush=True)
    for k in ks:
        get(k, 3, 256, raw_floor)
    best_k = max({k: cache[(k, 3, 256, raw_floor)] for k in ks},
                 key=lambda k: cache[(k, 3, 256, raw_floor)].tok_s)

    print(f"minimum match: {list(matches)} at k={best_k}", flush=True)
    for match in matches:
        get(best_k, match, 256, raw_floor)
    best_m = max({m: cache[(best_k, m, 256, raw_floor)] for m in matches},
                 key=lambda m: cache[(best_k, m, 256, raw_floor)].tok_s)

    print(f"minimum context: {list(contexts)} at k={best_k}, match={best_m}", flush=True)
    for context in contexts:
        get(best_k, best_m, context, raw_floor)
    best_c = max({c: cache[(best_k, best_m, c, raw_floor)] for c in contexts},
                 key=lambda c: cache[(best_k, best_m, c, raw_floor)].tok_s)
    print(f"  best shape: k={best_k} match={best_m} context={best_c}", flush=True)

    print(f"acceptance floor: {sorted(set(floors) | {0.8})} at that shape", flush=True)
    for floor in sorted(set(floors) | {0.8}):
        get(best_k, best_m, best_c, floor)
    # The default arm is a final candidate of its own (the winning shape may differ from it).
    get(*default_key)

    # The first arm after a model load is the coldest: the expert tier is still filling and
    # graph capture has just happened.  Measure plain decoding again after the sweeps and keep
    # the faster of the two as the baseline, so lookup has to beat the machine at its best.
    print("baseline (no speculation, last):", flush=True)
    baseline_end = evaluate(spec, "baseline-end", {}, a.max_tokens, reps, log_dir,
                            port=take_port())
    print("  " + baseline_end.line(), flush=True)
    baseline = max((baseline_start, baseline_end), key=lambda x: x.tok_s)

    final_candidates = {key: score for key, score in cache.items() if key[3] > 0}
    chosen_key = choose(final_candidates, baseline, default_key)
    chosen = final_candidates[chosen_key] if chosen_key is not None else None
    chosen_env = _candidate_env(*chosen_key) if chosen_key is not None else {}

    base_argv = list(spec["argv"])
    default_parallel = current_parallel(base_argv)
    requested = parse_parallels(a.parallel) if a.parallel else [1, 2, 4]
    parallels = sorted(set(requested) | {1})
    if a.parallel is None:
        parallels = sorted(set(parallels) | {default_parallel})
        if a.smoke:
            parallels = sorted({1, min(2, default_parallel)})
    print(f"parallel requests: {parallels} (default {default_parallel})", flush=True)
    # A single batch is noisy on this rig (88-103 tok/s across identical runs): the parallel
    # decision is a percentage comparison, so median three batches instead of trusting one.
    parallel_reps = reps if a.smoke else max(5, reps)
    parallel_scores: dict[int, Score] = {}
    for n in parallels:
        if n == 1:
            parallel_scores[n] = chosen if chosen is not None else baseline
            continue
        # The server's --max-running-requests is the ceiling for this batch; it has to match
        # the number of requests sent, or a batch above the old ceiling would only queue.
        spec["argv"] = set_parallel(base_argv, n)
        print(f"  measuring parallel={n} ...", flush=True)
        try:
            score = evaluate(spec, f"parallel={n}", chosen_env if chosen is not None else {},
                             a.max_tokens, parallel_reps, log_dir,
                             speculative=chosen is not None, concurrency=n,
                             port=take_port())
        except ServerError as e:
            print(f"    parallel={n} failed, skipped: {e}", file=sys.stderr)
            continue
        parallel_scores[n] = score
        print("    " + score.line(), flush=True)
    best_n = max(parallel_scores, key=lambda n: parallel_scores[n].tok_s)
    if best_n != default_parallel and default_parallel in parallel_scores:
        if parallel_scores[best_n].tok_s <= parallel_scores[default_parallel].tok_s * (1.0 + MIN_GAIN):
            best_n = default_parallel
    spec["argv"] = set_parallel(base_argv, best_n)
    if best_n == 1:
        baseline_at = baseline
    else:
        print(f"baseline at parallel={best_n}:", flush=True)
        baseline_at = evaluate(spec, f"baseline-parallel={best_n}", {}, a.max_tokens,
                               parallel_reps, log_dir, concurrency=best_n, port=take_port())
        print("  " + baseline_at.line(), flush=True)
    final = parallel_scores[best_n]
    final_gain = final.tok_s / baseline_at.tok_s - 1.0 if baseline_at.tok_s else 0.0
    single = parallel_scores[1]
    parallel_gain = final.tok_s / single.tok_s - 1.0 if single.tok_s else 0.0
    # The launch spends nearly all of its time at the parallel ceiling. If lookup does not beat
    # plain batch decoding there, keep the parallel gain and leave lookup out of the script; the
    # single-stream win is still recorded in the JSON for a caller that wants it.
    lookup_enabled = chosen is not None and lookup_worth_keeping(final_gain)
    single_stream_gain = (
        chosen.tok_s / baseline.tok_s - 1.0 if chosen is not None and baseline.tok_s else 0.0
    )
    if chosen is not None and not lookup_enabled:
        print(f"  lookup loses {final_gain * 100:.1f}% at parallel={best_n}; "
              f"the script will run without it (single-stream gain was "
              f"{single_stream_gain * 100:.1f}%)", flush=True)
    elif chosen is not None and final_gain < 0:
        print(f"  lookup is {-final_gain * 100:.1f}% slower at parallel={best_n}; "
              f"keeping it for its {single_stream_gain * 100:.1f}% single-stream win",
              flush=True)

    result = {
        "schema": SCHEMA,
        "mode": "smoke" if a.smoke else ("quick" if a.quick else "full"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": hardware(),
        "model": {"name": spec.get("model_name"), "path": _arg_value(spec["argv"], "--model-path")},
        "baseline": _score_doc(baseline),
        "baseline_arms": {"start": _score_doc(baseline_start), "end": _score_doc(baseline_end)},
        "chosen": {
            "speculative": "lookup" if chosen is not None else "none",
            "env": chosen_env,
            "tok_s": chosen.tok_s if chosen else baseline.tok_s,
            "gain": (chosen.tok_s / baseline.tok_s - 1.0) if chosen and baseline.tok_s else 0.0,
            "single_stream_gain": (chosen.tok_s / baseline.tok_s - 1.0)
            if chosen and baseline.tok_s else 0.0,
            "acceptance": chosen.acceptance if chosen else None,
            "drafted_ratio": chosen.drafted_ratio if chosen else None,
            "key": {"max_draft": chosen_key[0], "min_match": chosen_key[1],
                    "min_context": chosen_key[2], "min_acceptance": chosen_key[3]}
            if chosen_key else None,
            "parallel": best_n,
            "enabled_at_parallel": lookup_enabled,
            "tok_s_at_parallel": final.tok_s,
            "gain_at_parallel": final_gain,
        },
        "lookup_enabled": lookup_enabled,
        "parallel": {
            "default": default_parallel,
            "chosen": best_n,
            "sweep": {str(n): _score_doc(score) for n, score in sorted(parallel_scores.items())},
            "baseline_at_chosen": _score_doc(baseline_at),
            "gain_vs_single": parallel_gain,
            "lookup_gain_at_chosen": final_gain,
        },
        "user_params": {
            "max_running_requests": default_parallel,
            "kv_cache_dtype": _arg_value(spec["argv"], "--kv-cache-dtype"),
            "num_tokens": _arg_value(spec["argv"], "--num-tokens"),
            "memory_ratio": _arg_value(spec["argv"], "--memory-ratio"),
        },
        "measurements": {
            f"k={key[0]} match={key[1]} context={key[2]} a={key[3]:g}": _score_doc(score)
            for key, score in sorted(cache.items())
        },
        "launch": {
            "python": spec.get("python"),
            "argv": normalize_argv(spec["argv"], spec["port"]),
            "cwd": spec.get("cwd"),
            "env": dict(spec.get("env") or {}),
            "port": spec["port"],
            "model_name": spec.get("model_name"),
        },
    }
    return result


def _arg_value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def lookup_worth_keeping(gain_at_parallel: float) -> bool:
    """Whether lookup should stay in a launch that may carry parallel traffic.

    It only ever drafts one-request batches, so the parallel comparison is about its fixed
    cost, not its draft policy. A small loss is accepted because the same server still gets
    the single-stream win; a large one means starting without it.
    """
    return gain_at_parallel >= -MAX_PARALLEL_LOSS


def _score_doc(score: Score) -> dict:
    return {
        "tok_s": round(score.tok_s, 3),
        "tokens": score.tokens,
        "seconds": round(score.seconds, 3),
        "concurrency": score.concurrency,
        "per_prompt": {k: round(v, 3) for k, v in score.per_prompt.items()},
        "acceptance": score.acceptance,
        "drafted_ratio": score.drafted_ratio,
    }


def _on_sigterm(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Tune --speculative lookup on this PC; write a run config and start script.",
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
    ap.add_argument("--quick", action="store_true", help="fewer candidates, one rep per prompt")
    ap.add_argument("--smoke", action="store_true",
                    help="two widths and short answers: checks the plumbing, not a calibration")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS, help="measured repeats per prompt")
    ap.add_argument("--parallel", default=None,
                    help="concurrent request counts to measure (default 1,2,4; pass 1 for a "
                         "single-stream launch)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    a = ap.parse_args(argv)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_sigterm)
    spec = spec_from_args(a)
    tag = _sanitize(a.tag or spec.get("model_name") or "model")
    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if a.smoke else ("quick" if a.quick else "full")
    print(f"calibrating {spec.get('model_name')} on port {spec['port']} "
          f"({mode}); this restarts the model per candidate", flush=True)
    started = time.time()
    try:
        result = calibrate(spec, a)
    except KeyboardInterrupt:
        print("\ninterrupted; the server was stopped", file=sys.stderr)
        return 130
    except ServerError as e:
        print(f"calibration failed: {e}", file=sys.stderr)
        return 1
    result["elapsed_s"] = round(time.time() - started, 1)
    result["tag"] = tag

    chosen = result["chosen"] if result["chosen"]["speculative"] == "lookup" else None
    script_text = render_script(spec, chosen)
    json_path = out_dir / f"freetoken-lookup-{tag}.json"
    script_path = out_dir / f"run-freetoken-lookup-{tag}.sh"
    _atomic_write(json_path, json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    _atomic_write(script_path, script_text)
    script_path.chmod(0o755)

    ch = result["chosen"]
    if ch["speculative"] == "lookup":
        print(f"\nchosen: {ch['key']} -> {ch['tok_s']:.1f} tok/s "
              f"({ch['gain'] * 100:+.1f}% over baseline)", flush=True)
    else:
        print(f"\nlookup did not beat the baseline by {MIN_GAIN * 100:.0f}%; "
              f"the script starts without it", flush=True)
    print(f"config: {json_path}")
    print(f"script: {script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
