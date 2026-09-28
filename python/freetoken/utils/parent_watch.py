"""Die with the parent process.

The serve workers (scheduler ranks, tokenizer, detokenizer) are spawned
``daemon=False`` and block on their zmq recv when idle. Every orderly stop
path tears them down explicitly, but an UNCLEAN parent death (SIGKILL, an OOM
kill that picks the frontend, a segfault) skips all of them: nothing signals
the workers, and a process blocked in ``recv()`` never notices on its own, so
the whole engine tree (weights, KV pools, CUDA contexts, hundreds of threads)
outlives the server forever.

Two mechanisms, belt and braces: ``PR_SET_PDEATHSIG`` asks the kernel for a
signal the moment the parent dies; a ppid poll closes the race where the
parent dies between ``fork`` and the ``prctl`` (and covers hosts without
``prctl``). Stdlib only, importable without torch.
"""

from __future__ import annotations

import os
import signal
import threading
import time

_PR_SET_PDEATHSIG = 1
_POLL_S = 1.0


def install_parent_watchdog(poll_s: float = _POLL_S) -> None:
    """Arrange for SIGTERM to this process when its parent dies. Must be called
    from the worker itself, before it blocks in its service loop."""
    parent = os.getppid()
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() != parent:
            # the parent died in the fork->prctl window; the kernel signal is
            # not coming, send it ourselves
            os.kill(os.getpid(), signal.SIGTERM)
            return
    except Exception:  # noqa: BLE001 -- no libc / no prctl: the poll still works
        pass

    def _watch() -> None:
        while True:
            time.sleep(poll_s)
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=_watch, daemon=True, name="parent-watchdog").start()
