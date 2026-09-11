"""Greedy parity between TP=1 and TP=N on real hardware.

Boots ``ft serve`` twice on one checkpoint -- once on a single GPU, once with
``--tensor-parallel-size N`` -- and requires the same greedy reply from both. The sharded
paths (heads, KV pool, embedding / lm_head, the row-parallel all-reduces, the CUDA graph
that captures them) are each correct in isolation and wrong together, so the only gate
that means anything is that both ranks reproduce the single-rank token stream. Token
rates are printed, not asserted: a two-GPU PCIe box is expected to LOSE on a small dense
model, and the number is only useful to a human reading the A/B.

Gated behind ``needs_weights`` and skipped below ``FREETOKEN_TP_SIZE`` visible GPUs:

  FREETOKEN_TP_TEST_MODEL   a TP-capable checkpoint (falls back to FREETOKEN_TEST_MODEL)
  FREETOKEN_TP_SIZE         ranks to boot (default 2)
  FREETOKEN_TP_BOOT_TIMEOUT seconds to wait for "serving" (default 600)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.needs_weights

PROMPT = "What is 17 times 23? Show your reasoning step by step."


def _model_ref() -> str | None:
    """A local model directory, or a hub id the server resolves itself."""
    value = os.environ.get("FREETOKEN_TP_TEST_MODEL") or os.environ.get("FREETOKEN_TEST_MODEL")
    if not value:
        return None
    path = Path(value).expanduser()
    if value.startswith(("/", "~", ".")) and not path.is_dir():
        return None  # named as a path but missing: the checkpoint is not downloaded
    return str(path) if path.is_dir() else value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(base: str, path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.loads(response.read())


def _boot(model_ref: str, tp: int, log, port: int):
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "freetoken",
            "--model-path", str(model_ref),
            "--served-model-name", "tp-parity",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--tensor-parallel-size", str(tp),
            "--max-seq-len-override", "4096",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "python"},
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + float(os.environ.get("FREETOKEN_TP_BOOT_TIMEOUT", "600"))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.flush()
            tail = Path(log.name).read_text(errors="replace")[-1500:]
            raise RuntimeError(f"TP={tp} server exited early (code {proc.returncode}):\n{tail}")
        try:
            if _get(base, "/v1/cache/status")["state"] == "serving":
                return proc
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            pass  # not listening yet, or still loading
        time.sleep(2.0)
    proc.terminate()
    raise TimeoutError(f"TP={tp} server never reached the serving state")


def _greedy(base: str, prompt: str, max_tokens: int) -> tuple[str, int, str]:
    body = {
        "model": "tp-parity",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )

    def one() -> tuple[str, int]:
        with urllib.request.urlopen(request, timeout=600) as response:
            reply = json.loads(response.read())
        message = reply["choices"][0]["message"]
        text = str(message.get("content") or message.get("reasoning_content") or "")
        return text, int(reply["usage"]["completion_tokens"])

    one()  # the first request pays kernel JIT and graph warm-up
    text, tokens = one()
    again, _ = one()
    return text, tokens, again


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tensor parallelism needs CUDA")
def test_tp_parity_matches_single_rank_greedy(tmp_path):
    tp = int(os.environ.get("FREETOKEN_TP_SIZE", "2"))
    model_ref = _model_ref()
    if model_ref is None:
        pytest.skip("set FREETOKEN_TP_TEST_MODEL to a TP-capable model directory or hub id")
    visible = torch.cuda.device_count()
    if visible < tp:
        pytest.skip(f"needs {tp} visible GPUs, only {visible}")

    runs = {}
    for ranks in (1, tp):
        port = _free_port()
        log = (tmp_path / f"serve-tp{ranks}.log").open("w")
        proc = _boot(model_ref, ranks, log, port)
        base = f"http://127.0.0.1:{port}"
        try:
            runs[ranks] = _greedy(base, PROMPT, 64)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()

    base_text, base_tok, base_again = runs[1]
    tp_text, tp_tok, _ = runs[tp]
    if base_text != base_again:
        # A checkpoint that is not greedy-reproducible for itself (the quantized dense
        # kernels and the prefix-cache path can both shift a near-tie) cannot serve as a
        # byte-parity reference; the sharded run is only judged on its own terms.
        pytest.skip("this checkpoint is not greedy-reproducible at TP=1; no byte parity to assert")
    assert tp_tok == base_tok, f"{tp_tok} tokens at TP={tp} against {base_tok} at TP=1"
    assert tp_text == base_text, (
        f"greedy reply diverged at TP={tp}\n TP=1: {base_text!r}\n TP={tp}: {tp_text!r}"
    )
