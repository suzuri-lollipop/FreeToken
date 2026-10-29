"""Read-only control-plane endpoints consumed by the desktop app: /health (lifecycle),
/v1/stats (runtime metrics, Task 6), /v1/requests (request log ring, Task 5), /v1/meminfo
(VRAM breakdown) + /meminfo (its HTML dashboard).

All handlers read a shared FrontendManager snapshot via ``get_state``; nothing here touches
the scheduler or blocks. Registered on the app alongside the OpenAI/Anthropic/Responses routes.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import FastAPI


def build_health(state: Any, version: str) -> dict:
    """Full-lifecycle health doc: loading -> ok -> error.

    ``loaded`` answers the question the status code cannot: is there an engine behind this API
    right now. The HTTP status stays 200 while the server itself answers (strata's contract), so
    a caller that only checks the status must read this flag. ``generation`` counts engine
    (re)starts inside one serve process, ``restarting`` says what the recovery is doing."""
    instance_id = getattr(state, "instance_id", None)
    generation = int(getattr(state, "engine_generation", 0) or 0)
    auto_load = bool(getattr(state, "auto_load", False))
    fatal = getattr(state, "fatal_error", None)
    restarting = getattr(state, "restart", None)

    def _recovery(doc: dict) -> dict:
        doc["loaded"] = False
        doc["generation"] = generation
        doc["auto_load"] = auto_load
        if restarting:
            doc["restarting"] = dict(restarting)
        return doc

    if fatal:
        return _recovery({"status": "error", "message": fatal, "instance_id": instance_id})

    mstate = getattr(state, "maintenance_state", "serving")
    config = getattr(state, "config", None)
    model = getattr(config, "served_model_name", None)

    if mstate == "loading":
        lp = getattr(state, "load_progress", None)
        return _recovery({
            "status": "loading",
            "phase": lp.phase if lp is not None else "other",
            "progress": {
                "done_bytes": lp.done_bytes if lp is not None else 0,
                "total_bytes": lp.total_bytes if lp is not None else 0,
            },
            "model": model,
            "instance_id": instance_id,
        })

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    # "rebuilding" still has a live engine (it is applying a cache resize); "stopping" does not
    # take new work, so it is not loaded for a caller deciding whether to send one.
    return {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
        "loaded": mstate in ("serving", "rebuilding"),
        "generation": generation,
        "auto_load": auto_load,
    }


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
) -> None:
    @app.get("/health")
    async def health():
        return build_health(get_state(), app.version)

    from . import request_ring

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = request_ring.requests_since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    from .stats import build_stats

    @app.get("/v1/stats")
    async def stats():
        doc = build_stats(
            get_state(), request_ring.requests_p95_ms(), request_ring.requests_ttft_mean_ms()
        )
        # Surface the model's recommended sampling (from its generation_config.json / GGUF
        # metadata) so clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc["model"]["sampling"] = get_model_sampling() or {}
        return doc

    from fastapi.responses import HTMLResponse

    from .meminfo import MEMINFO_HTML, build_meminfo

    @app.get("/v1/meminfo")
    async def meminfo():
        return build_meminfo(get_state())

    @app.get("/meminfo")
    async def meminfo_page():
        # Self-contained dashboard (vanilla JS polling /v1/meminfo); no assets, no build.
        return HTMLResponse(MEMINFO_HTML)
