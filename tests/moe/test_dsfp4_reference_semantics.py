"""ds_fp4 (DeepSeek W4A8) experts against an INDEPENDENT torch transcription over
``fp4_gemm(act_quant(x, block), W)`` at both fp8 activation blocks (128 on V4, 32 on V4.1):

    x_q  = fp8_roundtrip(x, block)                       # act_quant(..., inplace)
    gate = bf16(x_q @ deq(W1)^T), up = bf16(x_q @ deq(W3)^T)   # fp4_gemm outputs bf16
    h    = bf16(silu(min(gate, L)) * clamp(up, -L, L))
    y_r  = bf16(w_r * (fp8_roundtrip(h, block) @ deq(W2)^T))   # routing weight in the down epilogue
    y    = sum_r y_r

The routing weight scales the down output (the placement main serves V4 with); the reference
``Expert.forward`` scales the intermediate before its fp8 quant instead. The GEMV (decode), the
grouped GEMM (prefill) and the CPU executor all follow this transcription.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, TOP_K, LIMIT = 16, 512, 256, 4, 7.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _banks(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def u8(*shape, low=0, high=256):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=g).to(device)

    return (u8(E, 2 * I, H // 2), u8(E, 2 * I, H // 32, low=120, high=130), u8(E, H, I // 2), u8(E, H, I // 32, low=120, high=130))


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[E, N, K//2] e2m1 pairs (even channel low nibble) + [E, N, K//32] e8m0 -> [E, N, K] fp32."""
    lut = E2M1.to(packed.device)
    lo, hi = lut[(packed & 0xF).long()], lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # [E, N, K]
    s = torch.exp2(scale.view(torch.uint8).float() - 127.0).repeat_interleave(32, dim=-1)
    return vals * s


def _fp8_roundtrip(x: torch.Tensor, block: int) -> torch.Tensor:
    g = x.float().unflatten(-1, (-1, block))
    s = torch.exp2(torch.ceil(torch.log2(g.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    q = (g / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


# fp32 accumulation noise, relative: the GEMV walks K sequentially, the grouped GEMM in tl.dot
# trees and this reference through cuBLAS, so a route output is only reproducible to ~1e-5.
_FP32_NOISE = 2.0 ** -16


def _tie_allowance(raw: torch.Tensor) -> torch.Tensor:
    """One bf16 step where ``raw`` sits within :data:`_FP32_NOISE` of the midpoint of its two bf16
    neighbours (0 elsewhere): there the rounding is decided by the accumulation order."""
    mag = raw.abs()
    b16 = mag.to(torch.bfloat16)
    b32 = b16.float()
    up = torch.nextafter(b16, torch.full_like(b16, float("inf"))).float()
    dn = torch.nextafter(b16, torch.zeros_like(b16)).float()
    mid = torch.where(mag >= b32, 0.5 * (b32 + up), 0.5 * (b32 + dn))
    return torch.where((mag - mid).abs() <= _FP32_NOISE * mag, 0.5 * (up - dn), 0.0)


def reference(x, slots, weights, banks, block, ties=False):
    """The MoE over dequantized banks: per-route bf16 expert outputs accumulated into an fp32 ``y``,
    returned in fp32 (the kernels round that sum to bf16). With ``ties`` also return the slack each
    element gains from :func:`_tie_allowance`.
    """
    gu_p, gu_s, dn_p, dn_s = banks
    W13 = _dequant(gu_p, gu_s)  # [E, 2I, H]
    W2 = _dequant(dn_p, dn_s)  # [E, H, I]
    xq = _fp8_roundtrip(x, block).float()
    T = x.shape[0]
    y = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    allow = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for t in range(T):
        for r in range(TOP_K):
            e = int(slots[t, r])
            gu = (xq[t] @ W13[e].T).to(torch.bfloat16).float()
            gate, up = gu[:I].clamp(max=LIMIT), gu[I:].clamp(-LIMIT, LIMIT)
            h = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            hq = _fp8_roundtrip(h.view(1, -1), block).float().view(-1)
            raw = weights[t, r].float() * (hq @ W2[e].T)
            y[t] += raw.to(torch.bfloat16).float()
            if ties:
                allow[t] += _tie_allowance(raw)
    return (y, allow) if ties else y


def assert_matches(got, x, slots, weights, banks, block):
    """The one freedom every implementation has over :func:`reference`: a route's bf16 output can
    flip one step where its fp32 value lands on a midpoint, and the routes cancel -- 1.8% of the
    elements (measured here) sum hundreds of magnitude down to single digits, where that step is a
    large relative error on ``y`` (observed: one flip, 0.5 on a 4.5 output, from an fp32 value 8e-6
    off the midpoint)."""
    want, allow = reference(x, slots, weights, banks, block, ties=True)
    bad = (got - want).abs() > 2e-2 + allow + 2e-2 * want.abs()
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {bad.numel()} off, worst abs {float((got - want).abs().max()):.4g} "
        f"at {bad.nonzero()[:4].tolist()}"
    )


def _inputs(T, device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(T, H, dtype=torch.bfloat16, generator=g) * 0.5).to(device)
    slots = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(T)]).to(device=device, dtype=torch.int32).contiguous()
    w = torch.rand(T, TOP_K, generator=g)
    w = (1.5 * w / w.sum(-1, keepdim=True)).float().to(device).contiguous()  # route_scale-style weights > 1 too
    return x, slots, w


@pytest.mark.parametrize("block", [32, 128])
def test_gemv_path_matches_the_transcription(block):
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    banks = _banks("cuda")
    x, slots, w = _inputs(6, "cuda")
    got = routed_experts_fp4(x, slots, w, *banks, LIMIT, act_block=block).float()
    assert_matches(got, x, slots, w, banks, block)


@pytest.mark.parametrize("block", [32, 128])
def test_cpu_executor_matches_the_transcription(block):
    """The ``ds_fp4`` CPU executor (cpu / hybrid decode) implements the same contract as the GPU path."""
    from types import SimpleNamespace

    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    gu_p, gu_s, dn_p, dn_s = _banks("cuda")
    pinned = {}
    for name, t in (("gate_up_packed", gu_p), ("gate_up_scale", gu_s), ("down_packed", dn_p), ("down_scale", dn_s)):
        buf = alloc_pinned_tensor(*t.shape, dtype=torch.uint8)
        buf.copy_(t.cpu())
        pinned[name] = buf
    cache = SimpleNamespace(quant_format="ds_fp4", bank_sources={k: [v] for k, v in pinned.items()}, num_layers=1, num_experts=E)
    dev = torch.device("cuda", 0)
    x, slots, w = _inputs(2, dev)
    ex = CpuMoeExecutor(cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False, num_threads=4, max_tokens=2, device=dev, swiglu_limit=LIMIT, act_block=block)
    got = ex.decode(0, x, w, slots.clone()).clone().float()
    torch.cuda.synchronize()
    del ex
    assert_matches(got, x, slots, w, (gu_p, gu_s, dn_p, dn_s), block)


def test_grouped_prefill_path_matches_the_transcription():
    from freetoken.moe import fused_ds_fp4

    banks = _banks("cuda")
    x, slots, w = _inputs(64, "cuda")
    got = fused_ds_fp4.routed_experts_fp4_prefill(x, slots, w, *banks, LIMIT, E, act_block=32).float()
    assert_matches(got, x, slots, w, banks, 32)


@pytest.mark.parametrize("block", [32, 128])
def test_inactive_routes_contribute_zero_without_reading_their_slots(block):
    """Hybrid decode hands the GPU kernel slot -1 (weight 0) for the routes the CPU computes. The kernel
    must produce exactly zero for them without touching any slot: every slot the active routes do not
    use is filled with 0xFF (an e8m0 scale of NaN), and the result still equals the reference over the
    active routes and the CPU executor's output for the same split."""
    from types import SimpleNamespace

    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    gu_p, gu_s, dn_p, dn_s = (t.clone() for t in _banks("cuda"))
    x, slots, w = _inputs(2, "cuda")
    # routes: token 0 keeps routes 0..1 on the GPU, token 1 keeps route 3; the rest go to the CPU
    on_gpu = torch.zeros_like(slots, dtype=torch.bool)
    on_gpu[0, :2] = True
    on_gpu[1, 3] = True
    gpu_slots = torch.where(on_gpu, slots, -1)
    gpu_w = torch.where(on_gpu, w, 0.0)
    used = set(slots[on_gpu].tolist())
    poison = [e for e in range(E) if e not in used]
    for t in (gu_p, gu_s, dn_p, dn_s):
        t[poison] = 0xFF
    got = routed_experts_fp4(x, gpu_slots, gpu_w, gu_p, gu_s, dn_p, dn_s, LIMIT, act_block=block)
    assert torch.isfinite(got).all()
    clean = _banks("cuda")
    # the poisoned-bank result equals the clean-bank result with the same holes (neither reads them),
    # and each share matches the reference over its own routes (a zero weight drops a route)
    got_clean = routed_experts_fp4(x, gpu_slots, gpu_w, *clean, LIMIT, act_block=block)
    assert torch.equal(got, got_clean)
    assert_matches(got.float(), x, slots, gpu_w, clean, block)
    # the CPU executor computes the complementary split from the same raw ids (ids < 0 skipped)
    pinned = {}
    for name, t in (("gate_up_packed", clean[0]), ("gate_up_scale", clean[1]), ("down_packed", clean[2]), ("down_scale", clean[3])):
        buf = alloc_pinned_tensor(*t.shape, dtype=torch.uint8)
        buf.copy_(t.cpu())
        pinned[name] = buf
    cache = SimpleNamespace(quant_format="ds_fp4", bank_sources={k: [v] for k, v in pinned.items()}, num_layers=1, num_experts=E)
    ex = CpuMoeExecutor(cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False, num_threads=4, max_tokens=2, device=torch.device("cuda", 0), swiglu_limit=LIMIT, act_block=block)
    cpu_ids = torch.where(on_gpu, -1, slots)
    cpu_out = ex.decode(0, x, w, cpu_ids.clone()).clone()
    torch.cuda.synchronize()
    del ex
    assert_matches(cpu_out.float(), x, slots, torch.where(on_gpu, 0.0, w), clean, block)
