from __future__ import annotations

import ctypes
import functools
import gc
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import dual_ar_available, get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.attention.linear import LinearAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs * rows_per_req] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    rows_per_req: int = 1

    @classmethod
    def init(cls, bs: int, device: torch.device, dummy_loc: int, mrope: bool = False, rows_per_req: int = 1) -> GraphCaptureBuffer:
        n = bs * rows_per_req
        return GraphCaptureBuffer(
            input_ids=torch.zeros(n, dtype=torch.int32, device=device),
            out_loc=torch.full((n,), dummy_loc, dtype=torch.int32, device=device),
            positions=torch.zeros(n, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, n, dtype=torch.int32, device=device) if mrope else None
            ),
            rows_per_req=rows_per_req,
        )

    def set_batch(self, batch: Batch) -> None:
        bs = batch.padded_size
        _rows = slice(bs * self.rows_per_req)
        batch.input_ids = self.input_ids[_rows]
        batch.out_loc = self.out_loc[_rows]
        batch.positions = self.positions[_rows]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _rows]

    def copy_from(self, batch: Batch) -> None:
        _rows = slice(batch.padded_size * self.rows_per_req)
        self.input_ids[_rows] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_rows] = batch.out_loc
        self.positions[_rows] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _rows] = batch.mrope_positions


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


@functools.cache
def _cu_graph_upload():
    if sys.platform != "linux" or torch.version.hip is not None:
        return None
    fn = ctypes.CDLL("libcuda.so.1").cuGraphUpload
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _upload_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> None:
    """Upload the graph at capture time.

    Otherwise the first replay pushes every node behind the graph's own stream WAIT, and a large graph then blocks cudaGraphLaunch until the WAIT is released. PLE wait-sync releases it only after the launch returns, so that deadlocks.
    """
    upload = _cu_graph_upload()
    if upload is None:
        return
    rc = upload(graph.raw_cuda_graph_exec(), stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"cuGraphUpload failed with CUresult {rc}")
    stream.synchronize()


class GraphRunner:
    """CUDA graphs of ``forward`` over padded batches of ``rows_per_req`` rows per request, one graph per batch size; ``replay`` returns the real requests' rows of each output."""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        forward: Callable[[], Any],
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        max_running_req: int = 0,
        *,
        rows_per_req: int = 1,
        phase: Literal["decode", "verify"] = "decode",
        ctx_override: Dict[str, Any] | None = None,
        batch_hook: Callable[[Batch], None] | None = None,
        pool: Any = None,
        out_dtype: torch.dtype | None = None,
        linear_backend: LinearAttnBackend | None = None,
        # dual-microbatch decode reads the model (supports_dual_decode, layer count,
        # _dual_halves); the verify/draft runners leave it None and never arm dual.
        model: BaseLLMModel | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
            max_running_req=max_running_req,
        )
        # the padded rows replay the dummy request, so it must carry rows_per_req rows
        assert rows_per_req == 1 or dummy_req.extend_len == rows_per_req, (rows_per_req, dummy_req.extend_len)
        self.forward = forward
        self.attn_backend = attn_backend
        self.linear_backend = linear_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        self.rows_per_req = rows_per_req
        self.phase = phase
        self.ctx_override = ctx_override or {}
        self.batch_hook = batch_hook
        self.graph_pool = pool
        self.out_dtype = out_dtype
        self._out: List[torch.Tensor] | None = None
        self._out_is_tuple = False
        self._out_rows: Dict[int, tuple[int, ...]] = {}
        self.model = model
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
        self._capture_graphs(max_seq_len)

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

    def _store_output(self, bs: int, out: Any) -> None:
        self._out_is_tuple = isinstance(out, tuple)
        outs = out if self._out_is_tuple else (out,)
        if self._out is None:
            # the largest batch is captured first, so its outputs size the buffers
            assert self.out_dtype is None or len(outs) == 1, "out_dtype applies to a single output"
            self._out = [torch.empty_like(o, dtype=self.out_dtype or o.dtype) for o in outs]
        for buf, o in zip(self._out, outs, strict=True):
            buf[: o.shape[0]] = o
        self._out_rows[bs] = tuple(o.shape[0] for o in outs)

    def _capture_graphs(self, max_seq_len: int):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list, rows_per_req=self.rows_per_req)
        if self.linear_backend is not None:
            self.linear_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list, rows_per_req=self.rows_per_req)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # capture runs the forward eagerly and stores K/V, so aim it at the dummy page as replay does for padded rows; slot 0 is a real page
        dummy_loc = int(get_global_ctx().page_table[self.dummy_req.table_idx, 0])
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, self.device, dummy_loc, mrope=self.mrope, rows_per_req=self.rows_per_req)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        _dual_model = self.model if self.phase == "decode" else None
        if self._dual_stream is not None and _dual_model is not None and getattr(_dual_model, "supports_dual_decode", False):
            self._dual_cu = [
                torch.arange(
                    (max(self.graph_bs_list) // 2) + 1, dtype=torch.int32, device=self.device
                )
                for _ in range(2)
            ]
            n_layers = len(_dual_model.model.layers.op_list)
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.ensure_dual_plans(n_layers)
            logger.info_rank0(
                f"Dual-microbatch decode capture armed for even bs >= 4 "
                f"(half = bs/2, {n_layers} layers; per-size eligibility still "
                f"needs the second shm AR instance)"
            )
        elif self.moe_offload_cache is not None and _dual_model is not None and getattr(
            self.moe_offload_cache, "decode_fetch_overlap", False
        ):
            # The fetch-overlap decode's K-split ensure stages into BOTH plan buffers;
            # materialize them (and the per-layer dual events it never records on)
            # before the warm run, never during capture.
            self.moe_offload_cache.ensure_dual_plans(len(_dual_model.model.layers.op_list))
        pool = self.graph_pool
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase=self.phase, rows_per_req=self.rows_per_req)
            batch.padded_reqs = batch.reqs
            if self.batch_hook is not None:
                self.batch_hook(batch)
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by prepare_for_replay).
            if self.linear_backend is not None:
                self.linear_backend.prepare_for_capture(batch)
            dual = self._dual_ok(self.model, bs)
            if dual:
                halfA = self._build_dual_half(self.buffer, 0, bs // 2, 0)
                halfB = self._build_dual_half(self.buffer, bs // 2, bs, 1)
                backend = self._dual_backend()
                with backend.dual_profile(0):
                    backend.prepare_for_capture(halfA)
                with backend.dual_profile(1):
                    backend.prepare_for_capture(halfB)
                self.model._dual_halves = (halfA, halfB)
                self.model._dual_stream = self._dual_stream
            try:
                with get_global_ctx().forward_batch(batch, **self.ctx_override):
                    self._store_output(bs, self.forward())
                    # Keep the offload cache warmed for capture. Resetting here forces
                    # CUDA graph capture to replay cold-cache expert copies.
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        self._store_output(bs, self.forward())
                    # Upload at capture time: otherwise the first replay pushes every node
                    # behind the graph's own stream WAIT and a large graph blocks launch.
                    _upload_graph(graph, self.stream)
                    self._reset_moe_offload_cache()
            finally:
                if dual:
                    self.model._dual_halves = None
                    self._dual_bs.add(bs)
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
        self.graph_pool = pool

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return (
            batch.phase == self.phase
            and batch.rows_per_req == self.rows_per_req
            and batch.size <= self.max_graph_bs
        )

    def replay(self, batch: Batch) -> Any:
        assert self.can_use_cuda_graph(batch) and self._out is not None
        from freetoken.moe import _debug_stats

        _dbg = _debug_stats.probe()
        if _dbg is not None:
            import time as _time

            _t0 = _time.perf_counter()
        self.buffer.copy_from(batch)
        padded = batch.padded_size
        g = self.graph_map[padded]
        if padded in self._dual_bs:
            self._dual_backend().stage_dual_replay(batch)
        else:
            self.attn_backend.prepare_for_replay(batch)
        # writes the real GDN slots into the persistent capture buffer the dual halves
        # (and the single-stream graph) read at their baked addresses
        if self.linear_backend is not None:
            self.linear_backend.prepare_for_replay(batch)
        if _dbg is not None:
            _t1 = _time.perf_counter()
        g.replay()
        if _dbg is not None:
            _t2 = _time.perf_counter()
            _dbg.host_phase("rp.stage", _t1 - _t0)
            _dbg.host_phase("rp.launch", _t2 - _t1)
        outs = tuple(
            buf[: rows // padded * batch.size] for buf, rows in zip(self._out, self._out_rows[padded], strict=True)
        )
        return outs if self._out_is_tuple else outs[0]

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
        # GraphCaptureBuffer tensors (input/out_loc/positions/...) and the output buffers (_out).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self._dual_bs = set()
        self._out = None
        gc.collect()
