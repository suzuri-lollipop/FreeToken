from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def num_splits_for(element_size: int) -> int:
    """Split factor for a row of ``element_size`` bytes; also used by the AOT
    shape table (kernel/aot_models.py), which must reproduce it exactly."""
    if element_size % 2048 == 0:
        return 4
    if element_size % 1024 == 0:
        return 2
    return 1


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    if not weights.is_cuda:
        # The JIT launcher is CUDA-only: handing it host pointers corrupts the heap
        # silently (the abort surfaces later in an unrelated free). CPU callers get the
        # equivalent gather here, mirroring index.cu / masked_index_kernel semantics.
        idx = indices.long()
        if vocab_range is None:
            gathered = weights[idx]
        else:
            start, length = vocab_range
            local = idx - start
            valid = (local >= 0) & (local < length)
            gathered = weights[local.clamp(0, weights.shape[0] - 1)]
            gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)
        return output.copy_(gathered)

    element_size = weights.shape[1] * weights.element_size()
    module = _jit_index_module(element_size, num_splits=num_splits_for(element_size))
    module.launch(weights, indices, output, vocab_range)
    return output
