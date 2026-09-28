"""A client that disconnects mid-stream cancels the /bench/run SSE generator; the
`ft bench bw` child must die with it (it runs on a GPU whose serve was stopped for
exclusivity), and a bench that finishes on its own must not be killed."""

import asyncio
import os
import sys

from freetoken.daemon.app import _bench_run_stream


def test_client_disconnect_kills_the_bench_child():
    async def go():
        holder = {}

        async def spawn(*args, **kwargs):
            proc = await asyncio.create_subprocess_exec(*args, **kwargs)
            holder["proc"] = proc
            return proc

        argv = [
            sys.executable,
            "-c",
            "import time; print('FTBENCH 1 2 warm', flush=True); time.sleep(30)",
        ]
        gen = _bench_run_stream(argv, dict(os.environ), spawn=spawn)
        first = await asyncio.wait_for(gen.__anext__(), 30)
        assert "event: progress" in first
        await gen.aclose()  # starlette does this when the SSE client goes away
        proc = holder["proc"]
        await asyncio.wait_for(proc.wait(), 10)
        return proc.returncode

    assert asyncio.run(go()) == -9  # SIGKILL from the generator's finally


def test_completed_bench_is_not_killed():
    async def go():
        holder = {}

        async def spawn(*args, **kwargs):
            proc = await asyncio.create_subprocess_exec(*args, **kwargs)
            holder["proc"] = proc
            return proc

        argv = [sys.executable, "-c", "print('FTBENCH 1 1 done', flush=True)"]
        events = [ev async for ev in _bench_run_stream(argv, dict(os.environ), spawn=spawn)]
        return events, holder["proc"].returncode

    events, rc = asyncio.run(go())
    assert rc == 0  # exited on its own: no kill
    assert events and events[-1].startswith("event: ")
