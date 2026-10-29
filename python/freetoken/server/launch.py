from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from freetoken.distributed import DistributedInfo
from freetoken.utils import init_logger
from freetoken.utils.parent_watch import install_parent_watchdog

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import BackendHandle


def _report_startup_error(ack_queue: mp.Queue, exc: BaseException) -> None:
    """Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits,
    so the supervisor reports the real cause (e.g. a config ValueError) instead of the generic
    "backend worker … exited during load". Best-effort and flushed (close + join_thread), since
    the process is about to terminate; a failure to report must never mask the original error."""
    try:
        ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        ack_queue.close()
        ack_queue.join_thread()
    except Exception:  # noqa: BLE001 -- reporting is a nicety; never shadow the real exception
        pass


def _abandon_partial_spawn(processes: list) -> None:
    """Tear down the children of a spawn that failed halfway through.

    A rank 0 left alive holds the fixed rendezvous port (``distributed_addr``) every later
    attempt has to bind, and the workers are non-daemon, so a half-spawned generation would
    both poison the restart loop and keep the serve from ever exiting. Best-effort: terminate,
    wait, kill, reap."""
    for p in processes:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:  # noqa: BLE001 -- never started / already gone
            continue
    for p in processes:
        try:
            p.join(timeout=10.0)
            if p.is_alive():
                p.kill()
                p.join(timeout=2.0)
        except Exception:  # noqa: BLE001 -- a handle that cannot be reaped is not ours to wait on
            continue


def _detach_process_group() -> None:
    """Shell mode only: move this worker out of the terminal's foreground process group.

    The shell binds ^C to "cancel this turn", but a terminal delivers SIGINT to the whole
    foreground group — which, with the engine running in this same process, includes the
    workers. They would take the same ^C and exit (``_run_scheduler`` below stops gracefully on
    KeyboardInterrupt), leaving the shell chatting with a dead engine. Nothing depends on the
    signal reaching them: the parent tears the workers down explicitly on every stop path
    (uvicorn's lifespan, plus the SIGTERM/SIGHUP handler and the reap backstop in api_server).

    ``ft serve`` keeps the default — uvicorn owns ^C there, and the group-wide delivery is part
    of how it stops."""
    try:
        os.setpgrp()
    except OSError:  # no job control (already a group leader / unusual environment)
        pass


def _run_tokenize_worker(detach: bool, **kwargs) -> None:
    """Module-level so it survives the spawn pickle; exists only to detach the group first."""
    if detach:
        _detach_process_group()
    install_parent_watchdog()
    from freetoken.tokenizer import tokenize_worker

    tokenize_worker(**kwargs)


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    if args.shell_mode:
        _detach_process_group()
    install_parent_watchdog()

    # published (not bound) here: the engine binds it after the allocator setup
    from freetoken.gpu_select import set_assigned_gpu

    # resolved UUIDs when we have them, the raw --gpu entries when NVML could not resolve them, else one CUDA ordinal per rank
    targets = args.gpu_assigned or args.gpu or tuple(str(r) for r in range(args.tp_info.size))
    set_assigned_gpu(targets[args.tp_info.rank])

    import torch
    from freetoken.scheduler import Scheduler

    if args.tp_info.is_primary():
        from freetoken.utils.progress import set_progress_sink

        set_progress_sink(
            lambda desc, done, total: ack_queue.put(("progress", desc, done, total))
        )

    with torch.inference_mode():
        try:
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()
        except Exception as exc:  # noqa: BLE001 -- surface the reason, then let it propagate
            # A startup failure (bad config, OOM, corrupt weights) would otherwise reach the
            # parent only as a dead process -> a generic "exited during load". Push the real
            # reason first so the supervisor (and the desktop failure modal) can surface it;
            # the traceback still prints and the process still exits non-zero.
            _report_startup_error(ack_queue, exc)
            raise

        # Report the real per-unit cache VRAM costs (KV/expert/mamba), the device-wide free
        # VRAM, the per-pool rebuild floors and this rank's whole-device footprint before the
        # ready ack, so the supervisor has them (and the desktop's slider bounds) by the time
        # the gate flips. EVERY rank reports its own payload keyed by rank: under TP each rank
        # owns a different device with its own weights and pool shards, so the all-GPU meminfo
        # view needs one per rank. Optional + best-effort: a failure here must never keep the
        # model from serving, and older consumers ignore ("meta", …).
        try:
            from freetoken.kvcache.cache_status import compute_cache_status_meta

            meta = compute_cache_status_meta(scheduler.engine)
            # the parent must not touch CUDA to learn this
            meta["gpus"] = scheduler.gpus
            meta["rank"] = args.tp_info.rank
            meta["tp_size"] = args.tp_info.size
            ack_queue.put(("meta", meta))
        except Exception:  # noqa: BLE001 -- metadata is a nicety; readiness is not
            pass
        # All ranks join this barrier so the ready ack cannot overtake a rank's still-in-flight
        # meta: the supervisor stops draining ack_queue at ready and would never read it.
        scheduler.sync_all_ranks()
        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")
            # The supervisor stops draining ack_queue once ready, so uninstall the sink:
            # runtime cache rebuilds re-run the graph capture (which emits progress) and
            # would otherwise push onto a queue nobody reads for the server's lifetime.
            set_progress_sink(None)

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(
        sys.argv[1:] if argv is None else argv,
        run_shell,
        prog=prog,
    )
    logger = init_logger(__name__, "initializer")

    if server_args.gpu:
        # resolve here so a typo is one clear error before any worker spawns
        from freetoken.gpu_select import resolve_gpu_uuids

        try:
            server_args = replace(server_args, gpu_assigned=resolve_gpu_uuids(server_args.gpu))
        except ValueError as exc:
            raise SystemExit(f"{prog or 'ft serve'}: error: {exc}") from exc
        logger.info(
            f"--gpu {','.join(server_args.gpu)} -> "
            f"{', '.join(server_args.gpu_assigned) if server_args.gpu_assigned else 'resolved at CUDA init (no NVML)'}"
        )

    # One clean error here beats N workers each dying on the same unsupported TP config
    # after they have built a CUDA context.
    from freetoken.engine.config import tp_preflight_error
    from freetoken.gpu_select import physical_gpu_count

    if (tp_error := tp_preflight_error(server_args)) is not None:
        raise SystemExit(f"{prog or 'ft serve'}: error: {tp_error}")
    # Each rank binds its own card by --gpu entry or CUDA ordinal; a rank with no card
    # would only notice deep inside the engine, after every worker is up.
    gpus = physical_gpu_count()
    if gpus is not None and server_args.tp_info.size > gpus:
        raise SystemExit(
            f"{prog or 'ft serve'}: error: --tensor-parallel-size "
            f"{server_args.tp_info.size} needs {server_args.tp_info.size} GPUs, "
            f"this machine sees {gpus}"
        )

    def start_subprocess() -> "BackendHandle":
        import multiprocessing as mp

        from .supervisor import BackendHandle

        mp.set_start_method("spawn", force=True)
        detach = server_args.shell_mode  # see _detach_process_group

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []

        def _start(name: str, target, args=(), kwargs=None) -> None:
            p = mp.Process(
                target=target,
                args=args,
                kwargs=kwargs or {},
                daemon=False,
                name=name,
            )
            # Tracked before start(): a spawn that dies halfway must still hand the ranks that
            # ARE up to _abandon_partial_spawn instead of losing track of them.
            processes.append(p)
            p.start()

        try:
            for i in range(world_size):
                new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
                _start(f"freetoken-TP{i}-scheduler", _run_scheduler, args=(new_args, ack_queue))

            num_tokenizers = server_args.num_tokenizer
            _start(
                "freetoken-detokenizer-0",
                _run_tokenize_worker,
                kwargs={
                    "detach": detach,
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_detokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "mm": server_args.mm,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": num_tokenizers,
                    "ack_queue": ack_queue,
                },
            )
            for i in range(num_tokenizers):
                _start(
                    f"freetoken-tokenizer-{i}",
                    _run_tokenize_worker,
                    kwargs={
                        "detach": detach,
                        "tokenizer_path": server_args.model_path,
                        "addr": server_args.zmq_tokenizer_addr,
                        "backend_addr": server_args.zmq_backend_addr,
                        "frontend_addr": server_args.zmq_frontend_addr,
                        "local_bs": 1,
                        "mm": server_args.mm,
                        "create": server_args.tokenizer_create_addr,
                        "tokenizer_id": i,
                        "ack_queue": ack_queue,
                    },
                )
        except Exception:  # noqa: BLE001 -- unwind the half-spawn, then let the real error travel
            _abandon_partial_spawn(processes)
            raise

        # Expected ready acks: 1 primary scheduler + num_tokenizers + 1 detokenizer.
        return BackendHandle(
            ack_queue=ack_queue,
            processes=processes,
            expected_acks=num_tokenizers + 2,
        )

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
