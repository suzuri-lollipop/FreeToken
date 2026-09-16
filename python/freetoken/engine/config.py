from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.layers.quantization.scheme import FP8_BLOCK, MX_GROUP, NVFP4_GROUP
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig
    from freetoken.models.register import ModelSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    # Flat residency (--moe-flat-residency): with an offload slot cache big enough to hold
    # every expert (num_moe_layers * num_experts slots), drop the LRU entirely -- each
    # expert keeps a permanent GPU slot, so prefill/decode stop streaming expert weights
    # over PCIe after the one load at startup. Off by default; needs --moe-strategy offload.
    moe_flat_residency: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # --kv-cache-dtype: "auto" keeps the model dtype; "fp8_e4m3" stores the KV pool in
    # e4m3 (half the bytes, one static descale pair). See kvcache/kv_quant.py.
    kv_cache_dtype: str = "auto"
    # --kv-cache-quant-scale: the static scale both K and V divide by on the way in.
    # 1.0 fits normalized attention states; raise it (2, 4, ...) when activations have a
    # wider range than e4m3 can hold, lower it to spend the range you do not need.
    kv_cache_quant_scale: float = 1.0
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        return replace(model_config, quant=quant)

    @cached_property
    def kv_quant(self):
        """``KVQuant`` for a quantized KV pool, None when the pool keeps ``dtype``.

        Cached: the pool and every attention read must agree on one scale pair for the
        lifetime of the process (the scales reach kernels as scalars baked into captured
        graphs), so this is resolved once instead of per call.
        """
        from freetoken.kvcache.kv_quant import resolve_kv_quant

        return resolve_kv_quant(self, self.kv_cache_dtype, self.kv_cache_quant_scale)

    @property
    def kv_dtype(self) -> torch.dtype:
        """The dtype the KV pool's buffer is allocated in."""
        return self.dtype if self.kv_quant is None else self.kv_quant.dtype

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def tp_size(self) -> int:
        return self.tp_info.size

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"


# The scale-block width each expert format keeps along the sharded intermediate axis: a rank's
# slice has to stay a whole number of these, or the loaded scales no longer line up.
_EXPERT_SCALE_GROUP = {
    "none": 1,
    "fp8_block": FP8_BLOCK,
    "nvfp4": NVFP4_GROUP,
    "mxfp4": MX_GROUP,
    "mxfp8": MX_GROUP,
}


def _shard_ok(total: int, tp_size: int, *, allow_replicate: bool) -> bool:
    """Mirror of ``utils.misc.div_even``: whether ``total`` units split over ``tp_size`` ranks."""
    if total % tp_size == 0:
        return True
    return allow_replicate and 0 < total < tp_size and tp_size % total == 0


def tp_shard_error(model_config, tp_size: int) -> str | None:
    """Why this model geometry cannot be sharded across ``tp_size`` ranks, or None.

    Pure so the check is CPU-testable and runs before a CUDA context exists, rather than
    tripping the sharders' asserts once every rank is half way through building the model.
    """
    if tp_size < 2:
        return None
    # is_moe is the engine's own key; a family may leave moe_enabled unset and be routed by
    # its model_type (qwen3_moe), so checking only the flag would miss it.
    moe = getattr(model_config, "is_moe", False) or getattr(model_config, "moe_enabled", False)
    problems: list[str] = []
    if not _shard_ok(model_config.num_qo_heads, tp_size, allow_replicate=False):
        problems.append(f"{model_config.num_qo_heads} query heads are not divisible by {tp_size}")

    groups = getattr(model_config, "attention_groups", ()) or ()
    kv_heads = [g.num_kv_heads for g in groups if getattr(g, "num_kv_heads", None) is not None]
    if not kv_heads:
        kv_heads = [model_config.num_kv_heads]
    for kv in kv_heads:
        if not _shard_ok(kv, tp_size, allow_replicate=True):
            problems.append(
                f"{kv} KV heads neither split across nor replicate over {tp_size} ranks"
            )
    for group in groups:
        for attr, what in (("num_key_heads", "key"), ("num_value_heads", "value")):
            heads = getattr(group, attr, None)
            if heads is not None and not _shard_ok(heads, tp_size, allow_replicate=True):
                problems.append(
                    f"linear group {group.name!r} {what} heads ({heads}) neither split "
                    f"across nor replicate over {tp_size} ranks"
                )

    if moe:
        expert_quant = str(getattr(model_config, "expert_quant", "none"))
        block = _EXPERT_SCALE_GROUP.get(expert_quant, 1)
        inter = model_config.moe_intermediate_size
        if inter and inter % (tp_size * block):
            problems.append(
                f"expert intermediate size {inter} cannot be split into {tp_size} ranks "
                f"on the {block}-column scale group of {expert_quant} experts"
            )
        shared = getattr(model_config, "shared_expert_intermediate_size", 0)
        if shared and shared % tp_size:
            problems.append(f"shared-expert intermediate size {shared} is not divisible by {tp_size}")
    elif model_config.intermediate_size % tp_size:
        problems.append(
            f"MLP intermediate size {model_config.intermediate_size} is not divisible by {tp_size}"
        )
    # The tower hands each of these widths to div_even, and its reader cuts the same axis.
    vision = getattr(model_config, "vision_config", None)
    if vision is not None:
        merged = vision.hidden_size * vision.spatial_merge_size**2
        for what, total in (
            ("heads", vision.num_heads),
            ("hidden size", vision.hidden_size),
            ("MLP intermediate size", vision.intermediate_size),
            ("merged width", merged),
        ):
            if total % tp_size:
                problems.append(f"vision {what} ({total}) does not divide into {tp_size} ranks")
    return "; ".join(problems) or None


def tp_preflight_error(config: EngineConfig) -> str | None:
    """Why this config cannot serve at ``config.tp_info.size``, or None when it can.

    The rank-spawning entry points call this so a family or geometry that has no TP
    sharder is reported once, before any worker burns a CUDA context and a weight load.
    """
    tp_size = getattr(config, "tp_size", 1)  # duck-typed test configs omit it
    if tp_size < 2:
        return None
    model_config = config.model_config
    # FTW stores the tensors after the reader fused them at full width, and its replay is
    # model-agnostic, so no rank-local slice exists to hand the sharded buffers.
    from freetoken.checkpoint.ftw import is_ftw_checkpoint

    if is_ftw_checkpoint(config.model_path):
        return (
            "an FTW checkpoint replays its stored full-width tensors, so it serves at "
            "--tensor-parallel-size 1; point --model at the HF directory instead"
        )
    if not config.model_spec.tp_supported:
        return (
            f"{model_config.model_type} does not shard its checkpoint for tensor parallelism "
            "yet; run with --tensor-parallel-size 1"
        )
    # A tower the family's reader still emits at full width would meet rank-sharded buffers at the load.
    unsharded = [e.kind for e in config.active_encoders if not e.tp_sharded]
    if unsharded:
        kinds = ", ".join(unsharded)
        return (
            f"the {kinds} encoder's weights are not tensor-parallel sharded yet; run with "
            "--text-model-only (or --mm-disable to name the encoders to drop)"
        )
    geometry = tp_shard_error(model_config, tp_size)
    if geometry:
        return f"--tensor-parallel-size {tp_size}: {geometry}"
    # Auto resolves a MoE model to the offload family, so treat it as the same request here.
    # The offload banks are TP-sharded for nvfp4 only: the layout declares the rank's slice
    # and pack cuts the pieces, while the bf16 / mxfp4 / block-fp8 stacks keep their readers at
    # full width. The CPU executor has no TP path, so cpu and hybrid stay refused.
    from freetoken.moe import is_offload_moe_strategy

    strategy = config.moe_strategy
    moe = getattr(model_config, "is_moe", False) or getattr(model_config, "moe_enabled", False)
    expert_quant = str(getattr(model_config, "expert_quant", "none"))
    if moe and (is_offload_moe_strategy(strategy) or strategy == "auto"):
        if expert_quant != "nvfp4":
            return (
                f"{expert_quant} experts are not TP-sharded in the offload banks yet; run a bf16 "
                "MoE with --moe-strategy fused (experts resident on every rank), or a single rank"
            )
        if strategy in ("cpu", "hybrid"):
            return (
                f"--moe-strategy {strategy} computes experts on the CPU, whose executor has no "
                f"tensor-parallel path yet; use --moe-strategy offload with --tensor-parallel-size "
                f"{tp_size}"
            )
    return None
