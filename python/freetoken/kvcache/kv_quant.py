"""Quantized KV-cache storage: dtype parsing, the scale pair, and pool sizing.

The pool allocates its KV buffer in the quantized dtype and every reader scales back
with the static per-tensor pair held here, so no scale buffer rides along with the
cache: the byte cost per token is exactly the storage dtype's itemsize, and the scales
reach the kernels as scalars (CUDA-graph safe, since they never change).

A static scale is a calibration-free approximation of the activation range: values
beyond ``scale * 448`` (the e4m3 max) clamp at the grid edge. Scale 1.0 suits
normalised attention states; checkpoints with an amax calibration record them per K/V
once that exists -- the pair is already split for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The dtype each supported --kv-cache-dtype name stores the pool buffer in.
KV_QUANT_DTYPES = {
    "fp8_e4m3": torch.float8_e4m3fn,
}

# Spellings accepted for the same storage, so the flag reads like the other engines'
# (--kv-cache-dtype fp8) without a second dtype behind it.
KV_QUANT_DTYPE_ALIASES = {
    "fp8": "fp8_e4m3",
    "fp8e4m3": "fp8_e4m3",
    "e4m3": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
}

# Largest finite magnitude of torch.float8_e4m3fn; quantization clamps to it so an
# outlier lands on the grid edge instead of the NaN code torch's cast produces.
E4M3_MAX = 448.0


@dataclass(frozen=True)
class KVQuant:
    """Storage dtype + static descale pair for a quantized KV cache.

    ``compute_dtype`` records the activation dtype the cache serves (the dtype ``store_kv``
    receives and the query arrives in). Backends that plan their kernels before the first
    forward -- FlashInfer's ``plan()`` wants the query and the cache dtype separately --
    read it here instead of guessing from the pool's storage dtype.
    """

    dtype: torch.dtype
    k_scale: float = 1.0
    v_scale: float = 1.0
    compute_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.dtype not in KV_QUANT_DTYPES.values():
            raise ValueError(f"unsupported kv-cache quant dtype: {self.dtype}")
        for name, scale in (("k_scale", self.k_scale), ("v_scale", self.v_scale)):
            if not scale > 0.0:
                raise ValueError(f"kv-cache quant {name} must be positive, got {scale}")

    @property
    def itemsize(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()


def parse_kv_cache_dtype(name: str | None) -> torch.dtype | None:
    """Storage dtype for a ``--kv-cache-dtype`` name, or None for "auto"/"none"."""
    if name is None or name in ("", "auto", "none"):
        return None
    name = KV_QUANT_DTYPE_ALIASES.get(name, name)
    try:
        return KV_QUANT_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown kv-cache dtype {name!r}; choices: auto, "
            + ", ".join(sorted(KV_QUANT_DTYPES))
            + " (aliases: "
            + ", ".join(sorted(KV_QUANT_DTYPE_ALIASES))
            + ")"
        ) from None


def has_native_e4m3() -> bool:
    """True when an fp8 KV buffer can be stored and read natively (sm_89+).

    Pre-sm_89 there is no fp8 storage path at all, not just no arithmetic: the
    emulation in ``kernel/triton/e4m3_compat.py`` keeps quantized values in a bf16
    buffer, which would make a "quantized" cache cost more than plain bf16. The
    delegation to that module's host-side gate is what keeps the pool dtype and the
    kernels' compile-time branch from disagreeing when
    ``FREETOKEN_FORCE_E4M3_EMU`` forces the emulated path on a capable GPU.
    """
    from freetoken.kernel.triton.e4m3_compat import e4m3_native
    from freetoken.utils import is_arch_supported

    return is_arch_supported(8, 9) and e4m3_native()


def resolve_kv_quant(config, requested: str, scale: float) -> KVQuant | None:
    """The pool's quantization from engine knobs, or None to keep the model dtype.

    Raises rather than silently running at full precision: a user asking for a fp8
    cache is asking for the capacity, and a no-op would report it as granted.
    """
    dtype = parse_kv_cache_dtype(requested)
    if dtype is None:
        return None
    if not has_native_e4m3():
        raise RuntimeError(
            f"--kv-cache-dtype {requested} needs native fp8e4nv: an sm_89 or newer CUDA "
            "GPU, and FREETOKEN_FORCE_E4M3_EMU unset (the emulated path has no fp8 storage)"
        )
    model_dtype = config.dtype
    quant = KVQuant(
        dtype=dtype,
        k_scale=float(scale),
        v_scale=float(scale),
        compute_dtype=model_dtype,
    )
    if quant.itemsize >= torch.tensor([], dtype=model_dtype).element_size():
        raise RuntimeError(
            f"--kv-cache-dtype {requested} ({quant.itemsize} B/elem) stores no smaller "
            f"than the model dtype {model_dtype}; the flag only pays off below it"
        )
    return quant
