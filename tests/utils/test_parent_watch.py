"""An engine worker whose frontend dies uncleanly (SIGKILL, OOM kill, segfault) must
not outlive it: the watchdog turns the parent's death into the worker's own SIGTERM.
Two-tier subprocess: the middle process spawns the watched worker, then dies with
os._exit (no teardown, like a SIGKILLed frontend); the worker must follow."""

import os
import subprocess
import sys
import time

_WORKER = (
    "import os, time\n"
    "from freetoken.utils.parent_watch import install_parent_watchdog\n"
    "install_parent_watchdog(poll_s=0.1)\n"
    "print(os.getpid(), flush=True)\n"
    "time.sleep(30)\n"
)

_MIDDLE = (
    "import os, subprocess, sys, time\n"
    "w = subprocess.Popen([sys.executable, '-c', os.environ['WORKER_SRC']])\n"
    "print(w.pid, flush=True)\n"
    "time.sleep(1.0)\n"
    "print('alive' if w.poll() is None else 'dead', flush=True)\n"
    "os._exit(0)\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_worker_dies_with_its_parent_but_not_before_it():
    env = {**os.environ, "WORKER_SRC": _WORKER}
    mid = subprocess.run(
        [sys.executable, "-c", _MIDDLE],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    lines = mid.stdout.split()
    assert lines[:1] and lines[0].isdigit(), mid.stderr
    wpid = int(lines[0])
    # the watchdog must not fire while the parent is merely slow
    assert lines[1] == "alive"

    deadline = time.monotonic() + 15
    while _alive(wpid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(wpid), f"worker {wpid} outlived its parent"
