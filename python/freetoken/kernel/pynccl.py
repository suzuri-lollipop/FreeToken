from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any, Literal

from freetoken.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


def _nccl_ldflags() -> list[str]:
    """Link flags for the NCCL that torch actually loads.

    ``-lnccl`` wants an unversioned ``libnccl.so`` on the linker path. The nvidia-nccl
    wheel ships only ``libnccl.so.2``, and a CUDA toolkit's copy is whatever symlink the
    installer left behind -- on a box whose toolkit symlink points at a deleted venv the
    link fails outright. So resolve the wheel torch depends on and link its soname by
    path; the rpath keeps a standalone dlopen of the built module working too. Falls
    back to ``-lnccl`` where there is no wheel (a system NCCL, or RCCL on ROCm).
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("nvidia")
    for root in spec.submodule_search_locations if spec is not None else ():
        for lib in sorted(Path(root).glob("nccl/lib/libnccl.so*")):
            return [str(lib), f"-Wl,-rpath,{lib.parent}"]
    return ["-lnccl"]


@functools.cache
def _load_nccl_module() -> Module:
    # TODO(ROCm): NCCL -> RCCL migration for multi-GPU tensor parallelism on AMD.
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=_nccl_ldflags())


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("freetoken.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def configure_pynccl_window() -> bool:
    """Whether the symmetric-memory window is registered; sets the NCCL knob before comm init.

    NCCL reads this at ``ncclCommInitRank``, so it must run before the wrapper is built. An
    explicit ``NCCL_WIN_ENABLE`` in the environment always wins.
    """
    enabled = ENV.PYNCCL_SYMMETRIC_WINDOW.value
    os.environ.setdefault("NCCL_WIN_ENABLE", "1" if enabled else "0")
    return os.environ["NCCL_WIN_ENABLE"] == "1"


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    configure_pynccl_window()
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
