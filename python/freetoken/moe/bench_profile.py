"""Torch-free reader for the ``ft bench bw`` hardware profile (``benchbw/<gpu-uuid>.json``).

The profile holds one row per MoE layer kind (``layer_kind_key``): the CPU MoE and PCIe gather
bandwidths benched for that kind on this GPU and host. The engine reads it to set each hybrid
layer's fetch split, without importing the (torch-heavy) benchmark. ``benchbw.py`` writes it.
"""

from __future__ import annotations

import json
import os

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Bump when the row key or the measurements change: rows of another version are not comparable.
PROFILE_VERSION = 6


def _cache_dir() -> str:
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "freetoken")


def default_profile_path(gpu_uuid: str | None = None) -> str:
    """``$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json``, or the legacy ``benchbw.json`` without a uuid.

    One file per GPU: bandwidth differs between slots.
    """
    if gpu_uuid:
        return os.path.join(_cache_dir(), "benchbw", f"{gpu_uuid}.json")
    return os.path.join(_cache_dir(), "benchbw.json")


def latest_profile_path() -> str | None:
    """Newest ``benchbw/*.json``, else the legacy ``benchbw.json``, else None."""
    per_gpu = os.path.join(_cache_dir(), "benchbw")
    newest: tuple[float, str] | None = None
    try:
        for name in os.listdir(per_gpu):
            if not name.endswith(".json"):
                continue
            path = os.path.join(per_gpu, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if newest is None or mtime > newest[0]:
                newest = (mtime, path)
    except OSError:
        pass
    if newest is not None:
        return newest[1]
    legacy = default_profile_path()
    return legacy if os.path.isfile(legacy) else None


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _usable_profile(
    gpu_name: str | None, path: str | None, gpu_uuid: str | None = None
) -> dict | None:
    """The cached profile, or ``None`` when there is no file / it was benched on another GPU
    (bandwidths are hardware-specific, so a mismatch is ignored rather than trusted).

    Lookup: explicit ``path`` (else ``FREETOKEN_BENCHBW_PATH``) -> ``benchbw/<gpu_uuid>.json`` -> legacy ``benchbw.json``.
    """
    explicit = path or os.environ.get("FREETOKEN_BENCHBW_PATH")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [default_profile_path(gpu_uuid)] if gpu_uuid else []
        candidates.append(default_profile_path())
    prof = None
    for src in candidates:
        prof = _load(src)
        if isinstance(prof, dict):
            break
        if os.path.exists(src):
            # unreadable profile for this card: stay on the safe default, do not borrow the legacy file
            return None
    if not isinstance(prof, dict):
        return None
    if prof.get("version") != PROFILE_VERSION:
        logger.warning(
            f"benchbw profile {src} has version {prof.get('version')}, this engine reads "
            f"{PROFILE_VERSION}; ignoring it, rerun `ft bench bw`"
        )
        return None
    prof_gpu = (prof.get("gpu") or {}).get("name")
    if gpu_name and prof_gpu and prof_gpu != gpu_name:
        logger.warning(
            f"benchbw profile {src} was measured on {prof_gpu!r}, not this GPU "
            f"({gpu_name!r}); ignoring it"
        )
        return None
    return prof


def layer_kind_key(fmt: str, options: dict[str, int], hidden: int, inter: int, top_k: int, cpu_threads: int) -> str:
    """Row key of one MoE layer kind, e.g. ``gguf[down_type=8,gate_up_type=8]:2816x704:top8:t12``.

    ``cpu_threads`` is the CPU executor's worker count: the per-step pool overhead makes the CPU bandwidth depend on it and on ``top_k``."""
    opts = ",".join(f"{k}={v}" for k, v in sorted(options.items()))
    fmt = f"{fmt}[{opts}]" if opts else fmt
    return f"{fmt}:{hidden}x{inter}:top{top_k}:t{cpu_threads}"


def fetch_fraction(row: dict | None, *, cpu_share: float = 1.0, hide: float = 1.0, pcie_gbs: float | None = None) -> float | None:
    """The hybrid fetch split of a row: of a decode step's expert misses, the fraction to fetch over PCIe so the fetch and the CPU compute finish together.

    From the overlapped pair (both sides measured while running concurrently): pcie_ov / (pcie_ov + cpu_ov);
    without it, pcie / cpu. Clamped to 1.

    ``pcie_gbs`` replaces the row's PCIe term with a live measurement of THIS rank's link: the profile is a
    machine-wide sample and cannot express two cards in different slots. ``cpu_share`` scales the row's CPU
    term down for the per-rank slice of one machine-wide RAM bandwidth, and ``hide`` weights it by
    ``cpu_hideability()`` (see ``_CPU_HIDEABILITY``); both default to the plain balance."""
    if not isinstance(row, dict):
        return None
    if pcie_gbs and pcie_gbs > 0:
        cpu = row.get("cpu_moe_gbs") or row.get("cpu_moe_overlap_gbs")
        if cpu:
            c = cpu / cpu_share
            return _bound_hideability(pcie_gbs / (pcie_gbs + c * hide), pcie_gbs / (pcie_gbs + c))
    cpu_ov, pcie_ov = row.get("cpu_moe_overlap_gbs"), row.get("pcie_gather_overlap_gbs")
    if cpu_ov and pcie_ov:
        c = cpu_ov / cpu_share
        return _bound_hideability(pcie_ov / (pcie_ov + c * hide), pcie_ov / (pcie_ov + c))
    cpu, pcie = row.get("cpu_moe_gbs"), row.get("pcie_gather_gbs")
    if cpu and pcie:
        c = cpu / cpu_share
        return _bound_hideability(pcie / (c * hide), pcie / c)
    return None


# A byte fetched over PCIe costs more than a byte computed on the CPU, even at equal measured
# bandwidth: the gather runs as an SM kernel on the decode critical path (and under TP the peer
# rank idles at its all-reduce until the slower link arrives), while the CPU overflow overlaps
# the GPU's dense GEMMs. Weighting the CPU term by this factor moves the balance point off
# pcie:cpu onto pcie:(cpu x hide). Measured on a 2-GPU PCIe rig (gen5 x16 beside gen4 x4,
# nvfp4 512-expert MoE, 4-way decode): the unweighted split lost ~6% conc=8 throughput, and
# the plateau was flat across rank fractions 0.25-0.45 / 0.10-0.20.
_CPU_HIDEABILITY = 2.2

# The hideability weighting is calibrated on ONE rig, so bound how far it may move the split
# from what this machine's own measured bandwidths say. Relative, not absolute -- an absolute
# floor would sit right on a slow link's legitimate operating point (a gen4 x4 rank resolves
# to ~0.14, and a 0.10 floor starts clipping it once the probe reads 6 GB/s instead of 7).
# 0.4 is deliberately slack at today's factor: a weight of H can shift the split by at most
# 1/H, so 2.2 reaches 0.4545 and never trips this. It is a ceiling on a FUTURE recalibration
# -- nobody may raise _CPU_HIDEABILITY past 2.5 and silently drag an unvalidated machine's
# split with it.
_HIDEABILITY_MAX_SHIFT = 0.4


def cpu_hideability(tp_size: int = 1) -> float:
    """The CPU-term weight of ``fetch_fraction``: ``_CPU_HIDEABILITY`` under TP, 1.0 for one rank.

    The factor is calibrated on a 2-rank rig; a single rank was never measured, so it keeps the
    plain bandwidth balance rather than inheriting an extrapolated correction."""
    return _CPU_HIDEABILITY if max(1, int(tp_size or 1)) > 1 else 1.0


def cpu_share(tp_size: int = 1) -> float:
    """How the machine-wide CPU MoE bandwidth splits across ``tp_size`` rank executors.

    The per-rank slice does not shrink linearly: RAM-bound pools on disjoint core slices each keep
    most of the stream rate (measured on 2 ranks over one dual-channel controller: ~62% each, not
    50%), hence tp**0.75."""
    return max(1, int(tp_size or 1)) ** 0.75


def _bound_hideability(weighted: float, unweighted: float) -> float:
    """Clamp the hideability-weighted split to at most ``_HIDEABILITY_MAX_SHIFT`` of the
    plain bandwidth balance. ``weighted <= unweighted`` always holds (the factor is >= 1),
    so the floor is the only side that can bind."""
    return min(1.0, max(weighted, _HIDEABILITY_MAX_SHIFT * min(1.0, unweighted)))


def load_hybrid_fetch_fraction(
    quant_format: str,
    gpu_name: str | None = None,
    path: str | None = None,
    gpu_uuid: str | None = None,
    tp_size: int = 1,
    pcie_gbs: float | None = None,
) -> float | None:
    """Benched hybrid fetch fraction for ``quant_format``, or ``None``.

    The hybrid backend's bandwidth-matched fetch split: of a decode step's expert misses,
    fetch this fraction over PCIe and compute the rest on the CPU, so both finish together.
    ``pcie_gbs`` -- a live measurement of THIS rank's own link -- replaces the profile's
    PCIe term when given: the cached one is a machine-wide sample that cannot express two
    cards sitting in different slots, and its contended-overlap variant understates the
    decode-time gather rate several-fold. The CPU term always comes from the profile.
    Without ``pcie_gbs`` the *overlapped* pair is preferred (CPU MoE and PCIe gather
    measured while running concurrently -- the real contention regime): fetched/misses =
    pcie_ov / (pcie_ov + cpu_ov). Older profiles without it fall back to the standalone
    bandwidths under a full-DRAM-contention assumption (cpu keeps cpu - pcie under DMA),
    which reduces to pcie/cpu. Per-dtype entry first, then any per-model entry with this
    format. ``None`` = no usable profile; clamped to [0, 1].

    Under tensor parallelism each rank keeps its OWN PCIe link (the profile is keyed by
    that rank's GPU uuid) but the CPU MoE pool is one machine-wide RAM bandwidth shared by
    all ``tp_size`` rank executors. The per-rank CPU slice does not shrink linearly with
    tp_size: RAM-bound pools on disjoint core slices each keep most of the stream rate
    (measured on 2 ranks over one dual-channel controller: ~62% each, not 50%), so the
    shared term scales as tp^0.75. Scaling the CPU term keeps the split bandwidth-matched
    per rank (a rank fetches a larger share over its private link because its CPU slice is
    smaller).

    The CPU term is then weighted by ``_CPU_HIDEABILITY``: equal measured bandwidths do not
    mean equal cost, because the PCIe gather holds SMs on the critical path while the CPU
    overflow hides under the GPU's dense work. Without it the balanced split over-fetches.
    The weighting is applied only at ``tp_size > 1`` -- it is calibrated on a 2-rank rig and
    a single rank was never measured, so one rank keeps the plain bandwidth balance rather
    than inheriting an extrapolated correction. The result is clamped off both degenerate
    ends, where the backend pays one mode's per-layer overhead for the other's work.
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof = _usable_profile(gpu_name, path, gpu_uuid)
    if prof is None:
        return None
    ranks = max(1, int(tp_size or 1))
    cpu_share = ranks ** 0.75
    hide = _CPU_HIDEABILITY if ranks > 1 else 1.0
    entries = [(prof.get("dtype_kernels") or {}).get(fmt)] + [
        (wl.get("kernels") or {}).get(fmt)
        for wl in (prof.get("workloads") or {}).values()
        if isinstance(wl, dict)
    ]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if pcie_gbs and pcie_gbs > 0:
            cpu = entry.get("cpu_moe_gbs") or entry.get("cpu_moe_overlap_gbs")
            if cpu:
                # standalone CPU rate vs a standalone link measurement: same conditions
                c = cpu / cpu_share
                return _bound_hideability(
                    pcie_gbs / (pcie_gbs + c * hide), pcie_gbs / (pcie_gbs + c)
                )
        cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
        if cpu_ov and pcie_ov:
            c = cpu_ov / cpu_share
            return _bound_hideability(pcie_ov / (pcie_ov + c * hide), pcie_ov / (pcie_ov + c))
        cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
        if cpu and pcie:
            c = cpu / cpu_share
            return _bound_hideability(pcie / (c * hide), pcie / c)
    return None


def locked_fetch_fraction(row: dict | None) -> float | None:
    """The fetch split of a locked layer of this row's kind: p / (2p + c) from the same pair as ``fetch_fraction``.

    A locked fetch is a CPU memcpy into staging, then the PCIe gather; taking the memcpy rate as the CPU MoE rate c, the two steps in series run at p*c / (p + c), which against the CPU compute gives p / (2p + c)."""
    if not isinstance(row, dict):
        return None
    for cpu_key, pcie_key in (("cpu_moe_overlap_gbs", "pcie_gather_overlap_gbs"), ("cpu_moe_gbs", "pcie_gather_gbs")):
        cpu, pcie = row.get(cpu_key), row.get(pcie_key)
        if cpu and pcie:
            return pcie / (2 * pcie + cpu)
    return None


def load_layer_kinds(gpu_name: str | None = None, path: str | None = None, gpu_uuid: str | None = None) -> dict | None:
    """The rows of the usable profile for this GPU (key -> row), or None without one."""
    prof = _usable_profile(gpu_name, path, gpu_uuid)
    if prof is None:
        return None
    rows = prof.get("layer_kinds")
    return rows if isinstance(rows, dict) else {}
