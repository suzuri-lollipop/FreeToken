from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator

logger = init_logger(__name__)


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


@dataclass
class P2POneShotDistributedImpl(DistributedImpl):
    """Small-message fast path over the P2P one-shot kernel; everything else
    (prefill-sized reductions, other dtypes, non-contiguous) delegates to the
    NCCL impl it wraps. The bf16 pair sum is bit-identical to ncclSum's fp32
    accumulation over two ranks, so both paths return the same bytes."""

    inner: DistributedImpl
    ar: object  # P2POneShotAllReducer
    max_elems: int

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.numel() <= self.max_elems
        ):
            return self.ar.all_reduce_(x)
        return self.inner.all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.all_gather(x)


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from freetoken.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))

    # Small-message fast path for decode-sized reductions (~26-37us NCCL LL ->
    # single-digit us): GPU-P2P one-shot where the PCIe/NVLink data plane works,
    # else a host-shared-memory rendezvous via front-end stream memops. Both
    # builders are collective, probe-gated and never raise; any decline leaves
    # the NCCL impl on top. Kill switches: FREETOKEN_P2P_AR / FREETOKEN_SHM_AR.
    inner = DistributedCommunicator.plugins[-1]
    fast = None
    from .p2p_ar import p2p_ar_enabled

    if p2p_ar_enabled():
        from .p2p_ar import P2POneShotAllReducer

        ar = P2POneShotAllReducer.try_build(tp_info, tp_cpu_group)
        if ar is not None:
            fast = (ar, ar.max_elems, "gpu-p2p")
    if fast is None:
        from .shm_ar import shm_ar_enabled

        if shm_ar_enabled():
            from .shm_ar import ShmOneShotAllReducer

            ar = ShmOneShotAllReducer.try_build(tp_info, tp_cpu_group)
            if ar is not None:
                fast = (ar, ar.max_elems, "host-shm")
    if fast is not None:
        ar, max_elems, kind = fast
        DistributedCommunicator.plugins.append(
            P2POneShotDistributedImpl(inner, ar, max_elems)
        )
        logger.info_rank0(
            f"One-shot {kind} all-reduce enabled for contiguous bf16 <= "
            f"{ar.max_bytes // 1024} KiB (NCCL above)"
        )


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
