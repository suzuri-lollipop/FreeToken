"""Expert banks for the offload MoE cache: load, pack and pin the routed experts.

The expert kernel (``QuantMethod.kernel``) owns the bank layout and the pack step; the
checkpoint side delivers pieces (``moe.expert_pieces``) and this module fills the pinned host
banks from them (``build_expert_banks``). GGUF experts keep their packed ggml rows
(``models.gguf.experts``).
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field

import torch

from freetoken.distributed import try_get_tp_info

import hashlib
import tempfile
import time
from pathlib import Path
from freetoken.layers.quantization import QuantKind
from freetoken.utils import init_logger

from .host_banks import DIRECT_READ_SUPPORTED, HostBank, mem_available_bytes
from .offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS, placeholder_bank

logger = init_logger(__name__)

# Whether the chunked, page-cache-bypassing shard reader can run here. This used to ask whether
# os.O_DIRECT and os.preadv exist, which Windows lacks -- pinning every Windows box to the serial
# read even though host_banks opens the same unbuffered positioned read through
# FILE_FLAG_NO_BUFFERING + ReadFile. Ask the seam for the capability, not the platform for the
# symbol. Kept as a module attribute so tests can force either build.
_PARALLEL_READER_SUPPORTED = DIRECT_READ_SUPPORTED

# most intra-op threads one rank uses while it fills banks; measured as fast as the full pool on
# the big stacked pieces and faster on the small per-expert ones
_PACK_THREADS = 4


def _pack_thread_limit() -> int:
    """Intra-op threads to run the bank fill on, for this host and this many ranks on it.

    ``pack`` copies one expert's pieces into its bank rows, a few hundred KiB per call for a
    per-expert checkpoint -- too small to pay for an OpenMP region, and every rank on the box is
    running one. Two TP ranks at the default 24-thread width on 32 cores starved each other's
    barriers and filled Qwen3.8-Flash-Next NVFP4 at 58 MB/s each (15 min); bounded, the same pair
    loads in 27 s. Dividing the cores also keeps a host with fewer cores than ranks busy."""
    info = try_get_tp_info()
    ranks = max(1, info.size) if info is not None else 1
    return max(1, min(_PACK_THREADS, (os.cpu_count() or 1) // ranks))


@contextlib.contextmanager
def _pack_threads():
    """Narrow torch's intra-op pool for the bank fill, then give the caller's width back."""
    want = _pack_thread_limit()
    had = torch.get_num_threads()
    if want != had:
        torch.set_num_threads(want)
    try:
        yield
    finally:
        torch.set_num_threads(had)


@dataclass(frozen=True)
class ExpertBanks:
    """Loaded expert banks, normalized for ``OffloadMoeCache`` wiring."""

    quant_format: str  # _BANK_SCHEMAS key
    # Pinned host banks, keyed by the format's schema: one [num_experts, ...]
    # tensor per layer (independent allocations -> per-layer host attributes).
    sources: dict[str, list[torch.Tensor]]
    # marlin/b12x per-expert global scales ([L*E]); None for formats without them
    gate_up_alpha: torch.Tensor | None = field(default=None)
    down_alpha: torch.Tensor | None = field(default=None)
    # per-layer HostResidency values actually applied by the loader; None -> all pinned (also the degrade signal when a request was not honored)
    layer_residency: list[str] | None = field(default=None)
    # disk-tier banks: {base_name: [HostBank | None per layer]} for the layers whose
    # backing is the FTW shard file itself (mmap, fault-on-touch); None unless a
    # disk-tier load produced them. The cache uses them for readahead prefetch.
    file_banks: dict[str, list] | None = field(default=None)
    # expert-profile hot copies: {layer_id: {canonical_role: (compact tensor, slot
    # map)}} -- the profile's pinned rows of otherwise file-mapped layers; the CPU
    # executor's per-(layer, expert) table prefers them over the mmap
    hot: dict[int, dict[str, tuple[torch.Tensor, dict[int, int]]]] | None = field(default=None)
    # True iff the ``layer_sink`` passed to the loader was actually engaged (each layer
    # streamed straight to its sink instead of staying materialized here) -- set by
    # convert.py's per-format streaming gate; ``sources`` may hold released tensors.
    streamed: bool = False
    # the expert (kind, kernel) the banks were packed for; None for the legacy providers
    kind: QuantKind | None = None
    kernel: str | None = None
    layout: dict | None = None
    # per-layer quant tag (e.g. the gate_up and down GGUF types), for layers whose banks match in dtype and row shape but not in quant; None = one quant
    layer_tags: list | None = None
    # per bank, per layer: bytes past an expert matrix's last row that the kernel reads and that must read as zeros (ggml MMQ); None = no such reads
    row_tail_bytes: dict[str, list[int]] | None = None
    # per-layer quant_format of experts in several formats (``quant_format``/``kind``/``kernel`` then name the most common one); None = every layer is ``quant_format``
    layer_formats: list[str] | None = None
    # the residency plan holding DEFERRED layers' banks, for host_banks.probe_pin; None when nothing was deferred
    deferred_plan: object | None = None


def _dummy_fill(role: str, tensor: torch.Tensor) -> None:
    """Random but finite bank contents for --use-dummy-weight."""
    if role.endswith("_scale"):
        if tensor.dtype is torch.uint8:
            tensor.fill_(127)  # e8m0 exponent code for 1.0
        else:
            tensor.fill_(1.0)
    elif role.endswith("_global"):
        tensor.fill_(0.01)
    elif tensor.dtype in (torch.uint8, torch.int32):
        tensor.view(torch.uint8).random_(0, 256)
    elif tensor.dtype is torch.float8_e4m3fn:
        tensor.view(torch.uint8).random_(0, 16)  # small codes, no NaN / inf
    else:
        tensor.normal_()


def build_expert_banks(
    method,
    num_layers: int,
    pieces,
    *,
    device: torch.device,
    layer_sink=None,
    dummy: bool = False,
    resident: bool = False,
    layers: list[int] | None = None,
) -> ExpertBanks:
    """Fill host banks in the kernel's layout from a stream of expert pieces.

    ``pieces`` yields ``(layer_id, e0, e1, {role: tensor[e1 - e0, ...]})`` in any order;
    each batch is packed in place into rows ``e0:e1`` of that layer's banks. A layer is
    complete once its ``num_experts`` rows have arrived: with ``layer_sink=None`` its banks
    are pinned in the background, otherwise the sink receives them (converter). ``dummy``
    skips the pieces and fills the banks with finite random contents.
    ``resident`` packs into banks on ``device`` instead of host banks (the resident 'fused' experts).
    ``layers`` fills only those bank layers (the ones ``method`` serves, for experts of several
    formats): pieces of other layers are skipped and the other layers' sources are None.
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, pin_banks
    from freetoken.moe.legacy_format import legacy_format_for

    kernel = method.kernel
    layout = method.layout()
    E = method.cfg.num_experts
    ids = list(range(num_layers)) if layers is None else sorted(layers)
    filled = set(ids)
    specs = {role: ((E, *spec.shape), spec.dtype) for role, spec in layout.items() if not spec.resident}
    if resident:
        assert layer_sink is None
        hb = None
        banks = {role: {l: torch.empty(shape, dtype=dtype, device=device) for l in ids} for role, (shape, dtype) in specs.items()}
    else:
        hb = {role: {l: HostBank(shape, dtype) for l in ids} for role, (shape, dtype) in specs.items()}
        banks = {role: {l: b.tensor for l, b in per_layer.items()} for role, per_layer in hb.items()}
    alphas = {
        role: torch.empty(num_layers * E, dtype=spec.dtype, device=device)
        for role, spec in layout.items() if spec.resident
    }

    def _sources() -> dict[str, list]:
        return {role: [per_layer.get(l) for l in range(num_layers)] for role, per_layer in banks.items()}

    if dummy:
        for role, per_layer in banks.items():
            for tensor in per_layer.values():
                _dummy_fill(role, tensor)
        for alpha in alphas.values():
            alpha.fill_(1.0)
        if hb is not None and torch.cuda.is_available():
            pin_banks(hb)
        return ExpertBanks(
            legacy_format_for(method.kind, kernel.name), _sources(),
            gate_up_alpha=alphas.get("gate_up_alpha"), down_alpha=alphas.get("down_alpha"),
            kind=method.kind, kernel=kernel.name, layout=layout,
        )

    def _fill(sink) -> None:
        tracker = LayerCompletionTracker(E, hb, sink) if sink is not None else None
        # a reader that skips a layer or mislabels a piece must fail here, not serve uninitialized rows
        written = torch.zeros(num_layers, E, dtype=torch.int32)
        for layer_id, e0, e1, piece in pieces:
            if not (0 <= layer_id < num_layers and 0 <= e0 < e1 <= E):
                raise ValueError(f"expert piece out of range: layer {layer_id}, experts {e0}:{e1} of {num_layers} x {E}")
            if layer_id not in filled:
                continue
            # refuse before writing: a duplicate row would also complete the layer early and hand the sink a half-filled bank
            if written[layer_id, e0:e1].any():
                raise ValueError(f"expert rows written more than once: layer {layer_id}, experts {e0}:{e1}")
            written[layer_id, e0:e1] = 1
            out = {role: banks[role][layer_id][e0:e1] for role in specs}
            got = method.pack(piece, out)
            for role, values in got.items():
                alphas[role][layer_id * E + e0 : layer_id * E + e1] = values.to(alphas[role].dtype)
            if tracker is not None:
                for _ in range(e1 - e0):
                    tracker.note(layer_id)
        missing = (written[ids] == 0).nonzero().tolist()
        if missing:
            missing = [[ids[l], e] for l, e in missing]
            raise ValueError(f"expert banks were not filled: {len(missing)} (layer, expert) rows missing (first {missing[:4]})")

    with _pack_threads():
        if layer_sink is not None:
            _fill(layer_sink)
        elif torch.cuda.is_available() and not resident:
            with PinPipeline() as pins:
                _fill(pins)
        else:
            _fill(None)

    return ExpertBanks(
        legacy_format_for(method.kind, kernel.name), _sources(),
        gate_up_alpha=alphas.get("gate_up_alpha"), down_alpha=alphas.get("down_alpha"),
        streamed=layer_sink is not None, kind=method.kind, kernel=kernel.name, layout=layout,
    )


def method_groups(layer_methods) -> list[tuple]:
    """``[(method, [bank layers])]`` of the layers' expert methods, one entry per (kind, kernel), most layers first."""
    groups: dict = {}
    for layer, method in enumerate(layer_methods):
        groups.setdefault((method.kind, method.kernel.name), (method, []))[1].append(layer)
    return sorted(groups.values(), key=lambda g: (-len(g[1]), g[1][0]))


def is_mixed(layer_methods) -> bool:
    return layer_methods is not None and len(method_groups(layer_methods)) > 1


def merge_expert_banks(parts: list[ExpertBanks], layer_methods, num_experts: int) -> ExpertBanks:
    """One ``ExpertBanks`` over the per-format ``parts`` (each filled only on its layers): the banks are the
    union of the roles, a layer without a role gets a zero-byte placeholder (see ``OffloadMoeCache.layer_layouts``)."""
    from freetoken.moe.legacy_format import legacy_format_for

    num_layers = len(layer_methods)
    sources: dict[str, list] = {}
    for part in parts:
        if part.gate_up_alpha is not None or part.down_alpha is not None:
            raise ValueError(f"experts of several formats cannot use the {part.kernel} kernel, which keeps GPU-resident alphas")
        for role, per_layer in part.sources.items():
            merged = sources.setdefault(role, [None] * num_layers)
            for layer, tensor in enumerate(per_layer):
                if tensor is not None:
                    assert merged[layer] is None, (role, layer)
                    merged[layer] = tensor
    for per_layer in sources.values():
        for layer, tensor in enumerate(per_layer):
            if tensor is None:
                per_layer[layer] = placeholder_bank(num_experts)
    formats = [legacy_format_for(m.kind, m.kernel.name) for m in layer_methods]
    primary = parts[0]
    return ExpertBanks(
        primary.quant_format, sources, streamed=all(p.streamed for p in parts),
        kind=primary.kind, kernel=primary.kernel, layer_tags=formats, layer_formats=formats,
    )


def _mixed_expert_banks(model_path, model_config, layer_methods, device, dummy, parallel, workers, chunk, layer_sink=None) -> ExpertBanks:
    from freetoken.moe.expert_pieces import iter_expert_pieces

    num_layers = model_config.num_moe_layers
    assert len(layer_methods) == num_layers, (len(layer_methods), num_layers)
    groups = method_groups(layer_methods)
    # resolve every format's reader before any bank exists, so a missing parallel reader falls back to serial with nothing allocated
    readers = [
        None if dummy else iter_expert_pieces(
            model_path, model_config, method.kind, layers=layers, parallel=parallel, workers=workers, chunk=chunk
        )
        for method, layers in groups
    ]
    parts = []
    for (method, layers), pieces in zip(groups, readers):
        parts.append(build_expert_banks(
            method, num_layers, pieces, device=device, layer_sink=layer_sink, dummy=dummy, layers=layers,
        ))
        logger.info_rank0(f"expert banks: {method.kind} / {method.kernel.name} on {len(layers)} of {num_layers} MoE layers")
    return merge_expert_banks(parts, layer_methods, model_config.num_experts)


_PARALLEL_CHUNK = 8 << 20  # default O_DIRECT chunk for the parallel reader


def _tp_build_fence(before: bool, model_path: str) -> None:
    """Serialize the per-rank bank builds through a marker-file chain: rank r waits
    for r-1's fresh marker before its own build and writes one after it.

    Two ranks materializing ~80 GiB of expert banks each peak the host's RAM at
    roughly twice the single-rank footprint -- on a box whose pin budget is sized
    for the whole model once, the OOM killer takes both ranks at the same
    milestone. A barrier would time out at the group's 60 s default (a build runs
    minutes), so the chain is files: mtime must be newer than the waiter's start
    (a stale marker from a crashed boot is ignored), the last rank unlinks the
    chain, and a rank whose predecessor died raises after the deadline instead of
    waiting forever."""
    info = try_get_tp_info()
    if info is None or info.size < 2:
        return
    if os.environ.get("FREETOKEN_TP_BANK_STAGGER", "1").strip().lower() in ("0", "false", "no", "off"):
        return  # concurrent builds fit once the per-shard writeback sync keeps dirty pages bounded
    # The fence key must be SHARED across ranks. A rank-specific disk-tier sidecar
    # (...-ftw-banks-tp2r0 vs -tp2r1) would give each rank a different key, so the chain
    # deadlocks (each rank waits on a marker the other never writes). Strip the
    # -tp{size}r{rank} suffix so both ranks fence on the common base path.
    suffix = f"-tp{info.size}r{info.rank}"
    shared = model_path[: -len(suffix)] if model_path.endswith(suffix) else model_path
    key = hashlib.sha1(shared.encode()).hexdigest()[:12]
    marker = lambda r: Path(tempfile.gettempdir()) / f"freetoken-bank-build-{key}-{r}.done"  # noqa: E731

    if before:
        if info.rank == 0:
            return
        path = marker(info.rank - 1)
        floor = time.time() - 15.0
        deadline = time.time() + 1800.0
        while True:
            try:
                if path.stat().st_mtime >= floor:
                    return
            except FileNotFoundError:
                pass
            if time.time() > deadline:
                raise RuntimeError(
                    f"timed out waiting for TP rank {info.rank - 1} to finish its expert bank build ({path})"
                )
            time.sleep(2.0)

    path = marker(info.rank)
    path.write_text("")
    if info.rank == info.size - 1:
        # Do NOT unlink here: earlier ranks poll for this marker every 2 s and the
        # write->unlink window is microseconds, so they would almost always miss it
        # and hang until the deadline. Rank 0 cleans the chain after observing it.
        return
    # The last rank still has ~a build of work; the first collectives after the load
    # (the auto cache-size all_reduce) time out at the group's 60 s default, so every
    # rank waits here for the chain's tail and the ranks leave the build together.
    tail = marker(info.size - 1)
    floor = time.time() - 15.0
    deadline = time.time() + 1800.0
    while True:
        try:
            if tail.stat().st_mtime >= floor:
                break
        except FileNotFoundError:
            pass
        if time.time() > deadline:
            raise RuntimeError(
                f"timed out waiting for TP rank {info.size - 1} to finish its expert bank build ({tail})"
            )
        time.sleep(2.0)
    # Safe to clean now: the last rank has written its marker and returned, and no
    # other rank polls the chain anymore. Stale markers from a crashed boot are
    # ignored by the mtime freshness floor, so leftovers are harmless either way.
    if info.rank == 0:
        for r in range(info.size):
            try:
                marker(r).unlink()
            except FileNotFoundError:
                pass


def _method_expert_banks(model_path, model_config, method, device, dummy, parallel, workers, chunk, layer_sink=None, resident=False) -> ExpertBanks:
    from freetoken.moe.expert_pieces import iter_expert_pieces

    num_layers = model_config.num_moe_layers
    if dummy:
        return build_expert_banks(method, num_layers, None, device=device, dummy=True, resident=resident)
    _tp_build_fence(before=True, model_path=model_path)
    pieces = iter_expert_pieces(
        model_path, model_config, method.kind, parallel=parallel, workers=workers, chunk=chunk
    )
    banks = build_expert_banks(method, num_layers, pieces, device=device, layer_sink=layer_sink, resident=resident)
    _tp_build_fence(before=False, model_path=model_path)
    return banks


def _mem_available_bytes() -> int | None:
    """Free RAM for the OOM guard, read the platform's way (see
    :func:`freetoken.moe.host_banks.mem_available_bytes`). Windows has no /proc/meminfo, which
    used to leave this guard silently disabled there: auto picked parallel on boxes with no room
    for the reader's in-flight buffers."""
    return mem_available_bytes()


def _host_ram_fits_parallel(model_path: str) -> bool:
    """Best-effort: can free host RAM hold the expert banks plus the parallel reader's one
    extra (non-reclaimable) whole-shard buffer? Unknown (non-local path / no /proc) -> True,
    i.e. keep the fast path. Banks ~= checkpoint size (experts dominate); transient ~= the
    largest shard. Uses MemAvailable (counts reclaimable cache) -- the OOM-relevant figure."""
    try:  # the loader's index-first shard list also covers subdirectory layouts (a root glob does not)
        from freetoken.models.loader import iter_weight_files

        sizes = [os.path.getsize(p) for p in iter_weight_files(model_path)]
    except Exception:
        return True
    if not sizes:
        return True
    avail = _mem_available_bytes()
    if avail is None:
        return True
    if avail <= sum(sizes) + max(sizes):
        gib = 1 << 30
        logger.warning_rank0(
            f"expert banks: {sum(sizes) / gib:.1f} GiB of checkpoint shards > "
            f"{avail / gib:.1f} GiB available RAM -> serial build (avoids parallel-reader OOM; "
            "override with --expert-load parallel)"
        )
        return False
    return True


def _auto_pick_parallel(model_path: str, model_config, method, dummy: bool) -> bool:
    """``--expert-load auto``: read scattered experts with the chunked unbuffered reader
    (``O_DIRECT`` / ``FILE_FLAG_NO_BUFFERING``), drop to the serial build when host RAM can't hold
    the result.

    The parallel reader keeps whole shards in flight as ANONYMOUS (non-reclaimable) buffers on top
    of the bank-sized resident set, so a memory-tight box OOMs there where the serial path
    (reclaimable file mmap) survives. Both figures are sized by what the reader will actually open:
    a checkpoint can hold its experts next to a far bigger side file the loader never reads
    (Qwen3.8-Flash-Next's 50 GiB PLE/MTP file, whose own experts belong to the dropped speculative
    head), and sizing by the whole checkpoint then refuses parallel even on boxes with more RAM than
    the checkpoint has, pinning every one of those models to the slow serial read.
    """
    if not _PARALLEL_READER_SUPPORTED or dummy:
        return False
    from freetoken.models.weight import EXPERT_PREFETCH_SHARDS, expert_storage
    from freetoken.moe.expert_pieces import expert_key_matcher

    kind = method.kind if method is not None else None
    storage = expert_storage(model_path, expert_key_matcher(model_path, model_config, kind))
    if storage is None or not storage.scattered:
        return False
    try:  # the kernel layout is exact; fall back to the checkpoint's expert bytes
        banks_bytes = bank_bytes_estimate(model_config, method) or storage.expert_bytes
    except Exception:
        banks_bytes = storage.expert_bytes
    transient_bytes = (EXPERT_PREFETCH_SHARDS + 1) * storage.expert_shard_bytes
    avail = _mem_available_bytes()
    if avail is None:
        return True
    return avail > banks_bytes + transient_bytes


def ftw_bank_bytes(model_path: str) -> int | None:
    """Total expert-bank bytes of an FTW checkpoint, from its metadata (no bank IO).
    ``None`` when the checkpoint is not FTW -- callers that size things pre-load (auto split residency) then leave the load unchanged."""
    import json

    meta = os.path.join(model_path, "freetoken_weight.json")
    if not os.path.isfile(meta):
        return None
    with open(meta, encoding="utf-8") as f:
        tensors = json.load(f).get("tensors", [])
    return sum(t["nbytes"] for t in tensors if t.get("kind") == "experts_bank")


def _method_expert_bytes(method) -> int:
    return method.cfg.num_experts * sum(
        math.prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()
        for spec in method.layout().values() if not spec.resident
    )


def bank_bytes_estimate(model_config, method=None, layer_methods=None) -> int | None:
    """Estimated total expert-bank bytes of a raw checkpoint before loading it.

    With a bound expert ``method`` the kernel's layout gives the exact host bytes (``layer_methods``,
    one per MoE layer, for experts of several formats); otherwise the format-tag table sizes the
    format. ``None`` for unknown formats or missing dims (callers then skip the pre-load sizing)."""
    layers = getattr(model_config, "num_moe_layers", None)
    if layer_methods is not None:
        return sum(_method_expert_bytes(m) for m in layer_methods)
    if method is not None and layers:
        return layers * _method_expert_bytes(method)
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (
        getattr(model_config, "moe_weight_format", None) or "bf16"
    )
    per_expert = _BANK_BYTES_PER_EXPERT.get(fmt)
    layers = getattr(model_config, "num_moe_layers", None)
    experts = getattr(model_config, "num_experts", None)
    hidden = getattr(model_config, "hidden_size", None)
    inter = getattr(model_config, "moe_intermediate_size", None)
    if per_expert is None or not all((layers, experts, hidden, inter)):
        return None
    return layers * experts * per_expert(hidden, inter)


def load_expert_banks(
    model_path: str,
    model_config,
    *,
    method=None,
    device: torch.device,
    dummy: bool = False,
    parallel: bool | None = None,
    workers: int = 8,
    chunk: int = _PARALLEL_CHUNK,
    layer_sink=None,
    layer_residency: list[str] | None = None,
    resident: bool = False,
    banks_path: str | None = None,
    hot_experts: dict[int, list[int]] | None = None,
    layer_methods: list | None = None,
) -> ExpertBanks:
    """Load (or fabricate, with ``dummy=True``) the expert banks. Two paths, both returning
    the same normalized ``ExpertBanks`` and both pinning after fill:

    * **Fast path (FTW)**: if ``model_path`` (or ``banks_path``, the disk tier's repacked
      sidecar) is a converted FTW checkpoint, read its repacked banks directly (contiguous
      chunked O_DIRECT; DISK-labeled layers mmap their shard instead). No auto-conversion.
    * **Slow path** (the original checkpoint): auto-pick **parallel** (the common parallel chunked
      O_DIRECT reader) when experts are stored as many small tensors -- the serial read is
      slow there -- else the **serial baseline** (packed experts: serial already saturates,
      parallel only adds read amplification). parallel unavailable for a quant falls back to serial.

    ``parallel`` overrides the slow-path auto-pick: ``None`` = auto (production), ``True`` /
    ``False`` = force parallel / serial (used by the loader benchmark and the converter).

    ``layer_sink`` (the converter only): forwarded to whichever provider is picked; a
    provider only engages it (and reports ``ExpertBanks.streamed=True``) for its own
    streamable formats, so callers must check ``streamed`` rather than assume it fired.

    ``method`` (the bound expert quant method of the model's offload layers) selects
    the generic path: the family's pieces packed by the method's kernel. Without it only an
    FTW checkpoint loads.

    ``layer_residency``: per-layer ``HostResidency`` labels applied at settle time -- explicitly on the FTW fast path, ambiently (``requested_residency``) in the slow-path providers.
    Applied labels are echoed on ``ExpertBanks.layer_residency``; a loader that settles some other way leaves it ``None`` (CPU-layer decode still works on pinned banks, it just saves no pin quota).

    ``resident`` (with ``method``): the banks land on ``device`` for the resident 'fused' experts; no host bank outlives the load.

    ``banks_path``: where the repacked banks live when they are not inside ``model_path``
    itself -- the disk tier's sidecar FTW. Dense weights still come from ``model_path``.

    ``layer_methods`` (one bound method per MoE layer) serves experts of several formats: each
    format fills its own layers and the banks merge over the union of roles (``merge_expert_banks``).
    """
    from freetoken.checkpoint.ftw import is_ftw_checkpoint, load_ftw_banks, load_ftw_banks_to_device

    mixed = is_mixed(layer_methods)
    if mixed and (resident or layer_residency is not None):
        raise ValueError("experts of several formats are served from the offload slot cache only (no resident experts, no --moe-cpu-layers)")
    if resident:
        assert method is not None and layer_sink is None and layer_residency is None
    ftw_banks_path = banks_path or model_path
    # FTW banks are repacked for the shard split they were converted with, so an uneven
    # (bandwidth-weighted) split cannot replay them. At TP>1 the disk-tier sidecar is
    # built per rank (rank-local expert slices), so the fast path replays it correctly;
    # only an uneven split still has to rebuild from the source checkpoint.
    uneven = method is not None and getattr(method.cfg, "uneven_tp_shard", False)
    if ftw_banks_path and is_ftw_checkpoint(ftw_banks_path) and not dummy:
        if resident:
            banks = load_ftw_banks_to_device(ftw_banks_path, num_layers=model_config.num_moe_layers, device=device, workers=workers, chunk=chunk)
        elif uneven:
            banks = None
            logger.warning_rank0(
                "expert banks: FTW fast path skipped (uneven bandwidth-weighted expert "
                "shard); rebuilding from the source checkpoint"
            )
        else:
            # the same build fence as the slow path: two ranks reading the sidecar
            # and settling (cudaHostRegister) concurrently drop the load from
            # ~6 GB/s to well under 1 (they share one NVMe queue and the driver's
            # serialized pin path), so rank r+1 waits for rank r's marker.
            _tp_build_fence(before=True, model_path=ftw_banks_path)
            try:
                banks = load_ftw_banks(
                    ftw_banks_path, num_layers=model_config.num_moe_layers, workers=workers, chunk=chunk,
                    layer_residency=layer_residency, hot_experts=hot_experts,
                )
            finally:
                _tp_build_fence(before=False, model_path=ftw_banks_path)
        if banks is not None:
            logger.info_rank0(f"expert banks: FTW fast path (FTW checkpoint {ftw_banks_path})")
            return banks

    if parallel and not _PARALLEL_READER_SUPPORTED:
        logger.warning_rank0(
            "expert banks: no unbuffered read seam on this platform (neither O_DIRECT nor "
            "FILE_FLAG_NO_BUFFERING) -> serial build"
        )
        parallel = False

    auto = parallel is None
    if auto:
        # --expert-load serial/parallel bypasses the RAM check by forcing ``parallel``;
        # the pick sizes itself from the experts the reader opens, not the whole checkpoint
        parallel = _auto_pick_parallel(model_path, model_config, method, dummy)
    logger.info_rank0(f"expert banks: slow path ({'parallel' if parallel else 'serial'} build)")
    # parallel's reader resolves hub ids + handles single-file/no-index checkpoints, so it won't
    # OSError on those (which would leak the banks it pre-allocated, since host banks live for
    # the process). Only NotImplementedError (quant has no parallel reader; raised before any
    # allocation) falls back to serial.
    from freetoken.moe.host_banks import requested_residency

    def _build(par: bool) -> ExpertBanks:
        if mixed:
            return _mixed_expert_banks(model_path, model_config, layer_methods, device, dummy, par, workers, chunk, layer_sink)
        if method is not None and method.kind is QuantKind.GGUF:
            from freetoken.models.gguf.experts import load_gguf_expert_banks

            # each layer keeps its own ggml types, so the banks come from the file, not from one method's layout
            return load_gguf_expert_banks(model_path, model_config.num_moe_layers, first_layer=model_config.first_k_dense_replace, dummy=dummy, device=device if resident else None)
        if method is None:
            raise ValueError("the experts of a raw checkpoint are packed by the offload layers' MoE quant method; pass method=")
        return _method_expert_banks(model_path, model_config, method, device, dummy, par, workers, chunk, layer_sink, resident)

    with requested_residency(layer_residency) as residency_plan:
        try:
            banks = _build(parallel)
        except NotImplementedError as exc:
            if not parallel:
                raise
            logger.warning_rank0(f"parallel reader unavailable ({exc}); falling back to serial build")
            banks = _build(False)
    return _echo_residency(banks, layer_residency, residency_plan)


def pin_probe_enabled(config, mixed_formats: bool) -> bool:
    """Whether ``--moe-cpu-layers auto`` finds the pin limit by pinning until refused.

    Only where pinning is known to be capped; FREETOKEN_PIN_BUDGET_GB fixes the limit instead (then a failed pin is an error)."""
    from freetoken.checkpoint.ftw import is_ftw_checkpoint
    from freetoken.moe.host_banks import pin_capped

    return (
        (config.moe_cpu_layers or "").strip() == "auto"
        and pin_capped()
        and not os.environ.get("FREETOKEN_PIN_BUDGET_GB")
        and config.moe_strategy in ("offload", "hybrid")
        and not mixed_formats
        and not config.use_dummy_weight
        and not (config.model_path and is_ftw_checkpoint(config.model_path))  # FTW settles its banks itself
    )


def probe_banks(banks: ExpertBanks, num_layers: int) -> tuple[ExpertBanks, frozenset[int]]:
    """Settle DEFERRED banks with host_banks.probe_pin; returns the banks and the OS-locked layers."""
    import dataclasses
    import time

    from freetoken.moe.host_banks import HostResidency, probe_pin

    plan = banks.deferred_plan
    if plan is None:
        return banks, frozenset()  # the loader settled the banks itself (pinned)
    t0 = time.perf_counter()
    labels = probe_pin(plan, num_layers)
    locked = frozenset(i for i, r in enumerate(labels) if r != HostResidency.PINNED.value)
    pinned_bytes = sum(b.nbytes for l, bs in plan.deferred.items() if l not in locked for b in bs)
    logger.info_rank0(
        f"--moe-cpu-layers auto (pin probe): pinned {num_layers - len(locked)} MoE layers ({pinned_bytes / 2**30:.2f} GiB) "
        f"in {time.perf_counter() - t0:.1f}s, locked {len(locked)} {sorted(locked)}"
    )
    return dataclasses.replace(banks, layer_residency=labels, deferred_plan=None), locked


def _echo_residency(banks: ExpertBanks, requested, plan) -> ExpertBanks:
    """Stamp an honored residency request onto the ExpertBanks; keep None (and warn) when no settle point consulted the plan."""
    if requested is None or banks.layer_residency is not None:
        return banks
    if plan is not None and plan.applied:
        import dataclasses

        if plan.deferred:
            return dataclasses.replace(banks, layer_residency=list(requested), deferred_plan=plan)
        labels = [plan.actual.get(i, r) for i, r in enumerate(requested)]
        downgraded = [i for i, r in enumerate(requested) if labels[i] != r]
        if downgraded:
            # an auto split can downgrade most of the model, so name the count when
            # the list would run to the whole model
            shown = downgraded if len(downgraded) <= 12 else f"{len(downgraded)} layers"
            msg = (
                f"host-locked banks: {shown} settled pageable instead of OS-locked; their bytes "
                f"stay resident only as long as the OS does not reclaim them"
            )
            from freetoken.moe.host_banks import os_lock_refusal

            # the refusal above already names the ceiling behind it, and no retry of any other
            # bank changes that answer -- so the echo reports it at info, not twice as a warning
            if os_lock_refusal():
                logger.info_rank0(msg)
            else:
                logger.warning_rank0(f"{msg} (the page-lock quota refused them)")
        return dataclasses.replace(banks, layer_residency=labels)
    from freetoken.moe.host_banks import HostResidency

    if any(r != HostResidency.PINNED.value for r in requested):
        logger.warning_rank0(
            "--moe-cpu-layers: this checkpoint's bank loader settles banks without "
            "per-layer residency (pre-pins everything); CPU-layer decode still works "
            "but saves no pinned quota"
        )
    return banks


def attach_resident_banks(layers, banks: ExpertBanks) -> None:
    """Give each resident MoE layer its rows of the device ``banks``; bank layer ``i`` goes to the layer whose ``layer_id`` is ``i``."""
    ids = sorted(layer.layer_id for layer in layers if layer.layer_id is not None)
    num_layers = len(next(iter(banks.sources.values())))
    if ids != list(range(num_layers)) or len(ids) != len(layers):
        raise ValueError(f"resident MoE layer ids {ids} do not cover the {num_layers} expert bank layers")
    for layer in layers:
        i, e = layer.layer_id, layer.num_experts
        layout = layer.quant_method.layout()
        rows = {role: (e, *spec.shape) for role, spec in layout.items() if not spec.resident}
        if set(banks.sources) != set(rows):
            raise ValueError(f"expert banks hold {sorted(banks.sources)}, the {layer.quant_method.kernel.name} kernel reads {sorted(rows)}")
        for role, shape in rows.items():
            bank = banks.sources[role][i]
            if tuple(bank.shape) != shape or bank.dtype != layout[role].dtype:
                raise ValueError(
                    f"bank {role!r} of layer {i} is {tuple(bank.shape)} {bank.dtype} but the {layer.quant_method.kernel.name} kernel "
                    f"wants {shape} {layout[role].dtype}; the banks were packed for another kernel or model"
                )
        alphas = []
        # apply() reads view.alphas as (gate_up, down)
        for role in (role for role in ("gate_up_alpha", "down_alpha") if role in layout):
            a = getattr(banks, role)
            if a is None or a.numel() != num_layers * e or a.dtype != layout[role].dtype:
                raise ValueError(f"the {layer.quant_method.kernel.name} kernel needs {num_layers * e} {layout[role].dtype} {role} values, the expert banks carry {None if a is None else (a.numel(), a.dtype)}")
            alphas.append(a)
        layer._expert_banks = {role: per_layer[i] for role, per_layer in banks.sources.items()}
        layer._expert_alphas = tuple(a[i * e : (i + 1) * e] for a in alphas) or None
