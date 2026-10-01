from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import dual_ar_available, get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, mrope: bool = False
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, bs, dtype=torch.int32, device=device) if mrope else None
            ),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _slice] = batch.mrope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
    max_running_req: int = 0,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160
        # A decode batch can never exceed max_running_req (the table manager caps
        # it), so default captures above that are dead weight: ~37MB and ~0.4s
        # per size, taken from the same headroom the largest prefill chunk's
        # activations need. An explicit --cuda-graph-max-bs still wins.
        if max_running_req > 0:
            cuda_graph_max_bs = min(cuda_graph_max_bs, max_running_req)

    if cuda_graph_max_bs < 1:
        return []

    # Dense coverage at small bs: the per-user padding cost is largest when few
    # requests share a step padded up to the next captured size -- with the old
    # [1, 2, 4] list a bs=3 decode ran the bs=4 graph (dummy row's fetch/GEMM
    # work, only 3 tokens delivered: measured ~the same step time as bs=4, so
    # conc-3 per-user throughput came out BELOW conc-4). Capture is ~0.4s and
    # ~37MB per size here, so 1..8 dense is cheap. 9..11 pad to 12; from 12 the
    # stride is 4 up to 32 (above bs8 the dummy rows' padding cost outweighs the
    # capture time: bs20 padded to 24 spends 20% of its GDN/attn/GEMM rows for
    # nothing) and 8 above that, bounding startup at large maxes.
    dense_stop = min(cuda_graph_max_bs, 8)
    mid_stop = min(cuda_graph_max_bs, 32)
    candidates = (
        list(range(1, dense_stop + 1))
        + list(range(12, mid_stop + 1, 4))
        + list(range(40, cuda_graph_max_bs + 1, 8))
    )
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        max_running_req: int = 0,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
            max_running_req=max_running_req,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        # Dual-microbatch decode (FREETOKEN_DUAL_STREAM_DECODE=1): the bs>=4 even-size
        # decode graphs run the batch as two half-batches on skewed streams so one
        # half's PCIe miss fetch overlaps the other half's SM work. Opt-in; every
        # eligibility check below falls back to the single-stream graph.
        self._dual_stream = (
            torch.cuda.Stream(device=device)
            if os.getenv("FREETOKEN_DUAL_STREAM_DECODE", "0") == "1"
            else None
        )
        self._dual_bs: set = set()
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _dual_backend(self):
        return getattr(self.attn_backend, "decode_backend", self.attn_backend)

    def _dual_ok(self, model, bs: int) -> bool:
        if self._dual_stream is None or bs < 4 or bs % 2:
            return False
        backend = self._dual_backend()
        return bool(
            getattr(model, "supports_dual_decode", False)
            and self.moe_offload_cache is not None
            and hasattr(backend, "stage_dual_replay")
            and hasattr(backend, "dual_profile")
            and dual_ar_available()
        )

    def _build_dual_half(self, buffer, lo: int, hi: int, slot: int) -> Batch:
        """A half-batch VIEW over the static capture buffers (rows [lo, hi))."""
        from freetoken.attention.linear import FLAMetadata

        n = hi - lo
        half = Batch(reqs=[self.dummy_req] * n, phase="decode")
        half.padded_reqs = half.reqs
        _s = slice(lo, hi)
        half.input_ids = buffer.input_ids[_s]
        half.out_loc = buffer.out_loc[_s]
        half.positions = buffer.positions[_s]
        if buffer.mrope_positions is not None:
            half.mrope_positions = buffer.mrope_positions[:, _s]
        half.linear_table_idx = buffer.table_idx[_s]
        half.fla_metadata = FLAMetadata(
            cu_seqlens=self._dual_cu[slot][: n + 1],
            cache_indices=buffer.table_idx[_s],
        )
        half.ple_row_offset = lo
        # FREETOKEN_DUAL_SLOT0=1: both halves use the slot-0 instances (AR, MoE plan,
        # w8a16 ws) while keeping their row slices -- a bisection probe that separates
        # second-instance bugs from row-slicing bugs (only valid with DUAL_SERIAL=1).
        half.dual_slot = (
            0 if os.getenv("FREETOKEN_DUAL_SLOT0", "0") == "1" else slot
        )
        return half

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs, vocab_size, self.device, mrope=self.mrope
        )
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        if self._dual_stream is not None and getattr(model, "supports_dual_decode", False):
            self._dual_cu = [
                torch.arange(
                    (max(self.graph_bs_list) // 2) + 1, dtype=torch.int32, device=self.device
                )
                for _ in range(2)
            ]
            n_layers = len(model.model.layers.op_list)
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.ensure_dual_plans(n_layers)
            logger.info_rank0(
                f"Dual-microbatch decode capture armed for even bs >= 4 "
                f"(half = bs/2, {n_layers} layers; per-size eligibility still "
                f"needs the second shm AR instance)"
            )
        elif self.moe_offload_cache is not None and getattr(
            self.moe_offload_cache, "decode_fetch_overlap", False
        ):
            # The fetch-overlap decode's K-split ensure stages into BOTH plan buffers;
            # materialize them (and the per-layer dual events it never records on)
            # before the warm run, never during capture.
            self.moe_offload_cache.ensure_dual_plans(len(model.model.layers.op_list))
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            dual = self._dual_ok(model, bs)
            if dual:
                halfA = self._build_dual_half(self.buffer, 0, bs // 2, 0)
                halfB = self._build_dual_half(self.buffer, bs // 2, bs, 1)
                backend = self._dual_backend()
                with backend.dual_profile(0):
                    backend.prepare_for_capture(halfA)
                with backend.dual_profile(1):
                    backend.prepare_for_capture(halfB)
                model._dual_halves = (halfA, halfB)
                model._dual_stream = self._dual_stream
            try:
                with get_global_ctx().forward_batch(batch):
                    self.buffer.logits[:bs] = model.forward()
                    # Keep the offload cache warmed for capture. Resetting here forces
                    # CUDA graph capture to replay cold-cache expert copies.
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        self.buffer.logits[:bs] = model.forward()
                    self._reset_moe_offload_cache()
            finally:
                if dual:
                    model._dual_halves = None
                    self._dual_bs.add(bs)
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        from freetoken.moe import _debug_stats

        _dbg = _debug_stats.probe()
        if _dbg is not None:
            import time as _time

            _t0 = _time.perf_counter()
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        if batch.padded_size in self._dual_bs:
            self._dual_backend().stage_dual_replay(batch)
        else:
            self.attn_backend.prepare_for_replay(batch)
        if _dbg is not None:
            _t1 = _time.perf_counter()
        g.replay()
        if _dbg is not None:
            _t2 = _time.perf_counter()
            _dbg.host_phase("rp.stage", _t1 - _t0)
            _dbg.host_phase("rp.launch", _t2 - _t1)
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self._dual_bs = set()
        gc.collect()
