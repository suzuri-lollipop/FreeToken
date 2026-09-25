"""W8A16 (fp8-e4m3 weight, bf16 activation) decode GEMM for the dense projections.

Decode (M <= 8) dense GEMVs are weight-read bound: an fp8 weight copy halves the
HBM bytes versus bf16. Unlike the W8A8 ``_scaled_mm`` path this never quantizes
the activation (no per-call amax/quant kernels, no activation precision loss), so
it also wins on the small shapes where W8A8's ~25 us/call overhead ate the GEMM
saving (docs/qwen38_flash_next_optimizations.md rejected fp8 for row/o_proj on
exactly that overhead -- measured here: the existing fp8 W8A8 in_proj call runs
~38 us vs ~11-15 us for this kernel).

Small-N shapes (the hyper-connection down projection is [336, 10240]) need
split-K to fill the GPU. The split-K path writes per-chunk partials into a
cached fp32 workspace (plain stores, no atomic contention) and a tiny epilogue
reduces them, applies the per-channel scale and the bias, casts to bf16 and
re-zeroes the workspace for the next call. SPLIT_K == 1 fuses everything into
the main kernel.

Weights are quantized per output channel at load-finalize time (see
layers/quantization/linear/unquantized.py); the original bf16 tensor is dropped,
so prefill (M > max_batch) dequantizes on the fly -- one cheap elementwise pass
per call, amortized over a whole chunk's tokens.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

_BLOCK_M = 16  # tl.dot's floor; decode M <= 8 pads
_WS: dict[tuple, torch.Tensor] = {}


@triton.jit
def _w8a16_main_kernel(
    x_ptr, w_ptr, s_ptr, b_ptr, y_ptr, ws_ptr,
    M, N, K,
    stride_xm, stride_wn,
    BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FUSED: tl.constexpr,          # SPLIT_K == 1: scale+bias+cast in this kernel
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    m_mask = offs_m < M
    # tile-aligned k chunks: unaligned k_per made every iteration partial-masked
    # (measured 21.6us vs 15.6us on the [336,10240] hc-down shape)
    k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per = tl.cdiv(k_tiles, SPLIT_K)
    k_lo = pid_k * tiles_per * BLOCK_K
    k_hi = tl.minimum(k_lo + tiles_per * BLOCK_K, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k_lo + offs_k)[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k_lo + offs_k)[None, :]
    for k0 in range(k_lo, k_hi, BLOCK_K):
        k_mask = (k0 + offs_k) < k_hi
        a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wb = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        if e4m3_native_cx():
            wb16 = wb.to(tl.bfloat16)
        else:
            wb16 = e4m3_u8_to_f32(wb.to(tl.uint8)).to(tl.bfloat16)
        acc += tl.dot(a, tl.trans(wb16))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    if FUSED:
        scale = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc * scale[None, :]
        if BIAS:
            acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])
    else:
        # per-chunk partial plane, no atomics: ws[pid_k, m, n]
        ws_ptrs = ws_ptr + pid_k * (BLOCK_M * N) + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _w8a16_epilogue_kernel(
    ws_ptr, s_ptr, b_ptr, y_ptr, M, N,
    SPLIT_K: tl.constexpr,
    PLANE: tl.constexpr,          # ws plane stride (BLOCK_M * N), shared with the main kernel
    BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    offs_n = offs % N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(ws_ptr + k * PLANE + offs, mask=mask, other=0.0)
    scale = tl.load(s_ptr + offs_n, mask=mask, other=0.0)
    out = acc * scale
    if BIAS:
        out += tl.load(b_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, out.to(y_ptr.dtype.element_ty), mask=mask)


def _workspace(key, shape, device) -> torch.Tensor:
    ws = _WS.get(key)
    if ws is None or ws.numel() < shape[0] * shape[1]:
        ws = torch.zeros(shape, dtype=torch.float32, device=device)
        _WS[key] = ws
    return ws


def _config(M: int, N: int, K: int) -> tuple[int, int, int, int, int]:
    """(block_n, block_k, split_k, num_warps, num_stages) -- swept COLD (rotating
    weight set, L2 never re-serves a weight) on this rig's RTX PRO 4000 (sm120)
    over the production decode shapes; see _scratch/agent4/tune2.py + tune3.py.
    Split-K only pays for the skinny-N/deep-K hyper-connection down projection
    ([336, 10240]: 11 n-programs cannot fill the GPU alone); the wide shapes win
    with 64-wide n-tiles fused single-launch."""
    block_n = 32 if N < 512 else 64
    block_k = 64 if (K < 1024 or N < 512) else (256 if K >= 3072 else 128)
    n_progs = triton.cdiv(N, block_n)
    split_k = 1
    if n_progs < 96 and K >= 4096:
        k_tiles = triton.cdiv(K, block_k)
        split_k = max(2, min(k_tiles, triton.cdiv(176, n_progs)))
    warps = 8 if block_k >= 256 else 4
    return block_n, block_k, split_k, warps, 3


def w8a16_linear_decode(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """y = x @ (w_fp8 * scale[:, None]).T + bias for tiny-M decode batches."""
    M, K = x.shape
    N = w_fp8.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    block_n, block_k, split_k, warps, stages = _config(M, N, K)
    w_view = e4m3_kernel_view(w_fp8)
    if split_k == 1:
        _w8a16_main_kernel[(triton.cdiv(N, block_n), 1)](
            x, w_view, scale, bias if bias is not None else x, y, x,
            M, N, K, x.stride(0), w_fp8.stride(0),
            BIAS=bias is not None, SPLIT_K=1, FUSED=True,
            BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=warps, num_stages=stages,
        )
        return y
    ws = _workspace((x.device.index, N), (split_k * _BLOCK_M, N), x.device)
    _w8a16_main_kernel[(triton.cdiv(N, block_n), split_k)](
        x, w_view, scale, bias if bias is not None else x, y, ws,
        M, N, K, x.stride(0), w_fp8.stride(0),
        BIAS=bias is not None, SPLIT_K=split_k, FUSED=False,
        BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=warps, num_stages=stages,
    )
    block = 1024
    _w8a16_epilogue_kernel[(triton.cdiv(M * N, block),)](
        ws, scale, bias if bias is not None else scale, y, M, N,
        SPLIT_K=split_k, PLANE=_BLOCK_M * N, BIAS=bias is not None, BLOCK=block,
        num_warps=4,
    )
    return y


def quantize_weight_w8a16(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel e4m3 quantization: (w_fp8 [N,K], scale [N] fp32)."""
    amax = w.float().abs().amax(dim=1).clamp(min=1e-12)
    scale = amax / 448.0
    w_fp8 = (w.float() / scale[:, None]).to(torch.float8_e4m3fn)
    return w_fp8, scale


def dequantize_w8a16(w_fp8: torch.Tensor, scale: torch.Tensor, dtype) -> torch.Tensor:
    return (w_fp8.float() * scale[:, None]).to(dtype)
