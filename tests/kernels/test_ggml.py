"""Packed GGUF ops against gguf-py's CPU dequantization and PyTorch math."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel import ggml
from freetoken.layers.quantization.ggml_types import pad_rows
from freetoken.third_party.gguf import GGMLQuantizationType as QType
from freetoken.third_party.gguf import quants

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _q4_weights(shape, seed=690):
    values = np.random.default_rng(seed).normal(0, 0.1, shape).astype(np.float32)
    packed = quants.quantize(values, QType.Q4_0)
    reference = torch.from_numpy(quants.dequantize(packed, QType.Q4_0))
    # ggml's last MMQ tile may read an extra partial row.
    storage = torch.zeros(*packed.shape[:-2], packed.shape[-2] + 1, packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[..., :-1, :].copy_(torch.from_numpy(packed))
    return storage[..., :-1, :], reference


def _nvfp4_weights(shape, seed=765):
    """Random NVFP4 blocks over every fp4 code and every UE4M3 scale code below NaN (0x7F), padded for MMQ."""
    rng = np.random.default_rng(seed)
    rows, k = shape
    blocks = k // 64
    d = rng.integers(0, 0x7F, (rows, blocks, 4), dtype=np.uint8)
    qs = rng.integers(0, 256, (rows, blocks, 32), dtype=np.uint8)
    packed = np.concatenate([d, qs], axis=-1).reshape(rows, blocks * 36)
    reference = torch.from_numpy(quants.dequantize(packed, QType.NVFP4))
    storage = torch.zeros(rows + pad_rows(int(QType.NVFP4), k), packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[:rows].copy_(torch.from_numpy(packed))
    return storage, reference


def _iq1_m_weights(shape, exact_sums, seed=1729):
    """Random IQ1_M blocks over every grid index, shift bit and 3-bit scale, padded for MMQ.

    Super-scales are random fp16 in [0.005, 0.02], every bit of which the loader decodes. With ``exact_sums`` they are
    m / 1024 (m in 5..20) instead: every product with ``_activation`` is then a multiple of 2^-19 and the sums are exact
    in fp32, so the kernel and the reference round a bf16 / fp16 result alike.
    """
    rng = np.random.default_rng(seed)
    *lead, rows, k = shape
    blocks = (*lead, rows, k // 256)
    qs = rng.integers(0, 256, (*blocks, 32), dtype=np.uint8)
    qh = rng.integers(0, 256, (*blocks, 16), dtype=np.uint8)
    d = rng.integers(5, 21, blocks) / 1024 if exact_sums else rng.uniform(0.005, 0.02, blocks)
    d = d.astype(np.float16).view(np.uint16)
    # the super-scale's four nibbles sit in the top nibble of the four uint16 scale words
    scales = rng.integers(0, 1 << 12, (*blocks, 4), dtype=np.uint16) | (((d[..., None] >> (4 * np.arange(4, dtype=np.uint16))) & 0xF) << 12)
    packed = np.concatenate([qs, qh, scales.astype("<u2").view(np.uint8)], axis=-1).reshape(*lead, rows, -1)
    reference = torch.from_numpy(quants.dequantize(packed, QType.IQ1_M))
    storage = torch.zeros(*lead, rows + 1, packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[..., :-1, :].copy_(torch.from_numpy(packed))
    return storage[..., :-1, :], reference


def _activation(shape, dtype, seed=19):
    rng = np.random.default_rng(seed)
    half = rng.integers(-127, 128, (*shape[:-1], shape[-1] // 32, 16))
    half[..., 0] = 127
    # Exactly representable q8_1 inputs with zero block sums isolate weight math.
    values = np.concatenate([half, -half], axis=-1).reshape(shape).astype(np.float32) / 64
    return torch.from_numpy(values).to("cuda", dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_dequantize_and_rows(dtype):
    w, ref = _q4_weights((64, 256))
    actual = ggml.dequantize(w, int(QType.Q4_0), 256, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = ggml.get_rows(w, int(QType.Q4_0), 256, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_matrix_matches_cpu(dtype, batch):
    w, ref = _q4_weights((64, 256))
    x = _activation((batch, 256), dtype)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x)
    expected = F.linear(x.cpu().float(), ref).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 4, 17])
@pytest.mark.parametrize("per_route_x", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
def test_q4_moe_routes_match_cpu(dtype, batch, per_route_x, inactive):
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((batch, 2, 256) if per_route_x else (batch, 256), dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    if inactive:  # hybrid decode hands the GPU the CPU's routes as id -1: zero rows, no expert read
        ids[::2, 0] = -1
    actual = ggml.mul_mat_id(w, int(QType.Q4_0), x, ids, inactive=inactive)
    xs = x.cpu().float() if per_route_x else x.cpu().float().unsqueeze(1).expand(-1, 2, -1)
    expected = torch.einsum("tjk, tjok -> tjo", xs, ref[ids.cpu().long().clamp_min(0)])
    expected[ids.cpu() < 0] = 0
    torch.testing.assert_close(actual.cpu(), expected.to(dtype), atol=2e-4, rtol=1e-3)


# several sort blocks and two expert-scan passes; one sort block and five passes; a 32k-token prefill chunk; more experts than the 48 KiB shared memory default holds (an offloaded MoE's slot pool), in one sort block and in several
@pytest.mark.parametrize("n_experts, top_k, batch", [(300, 8, 1500), (1100, 2, 600), (4, 2, 32768), (12300, 8, 16), (12300, 8, 600)])
@pytest.mark.parametrize("per_route_x", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
def test_q4_moe_skewed_routes_match_cpu(n_experts, top_k, batch, per_route_x, inactive):
    w, ref = _q4_weights((n_experts, 64, 256))
    x = _activation((batch, top_k, 256) if per_route_x else (batch, 256), torch.float32)
    gen = torch.Generator().manual_seed(866)
    # Zipf-like expert popularity, a third of the experts never routed to
    popularity = (1 / torch.arange(1, n_experts + 1, dtype=torch.float64) ** 1.2)[torch.randperm(n_experts, generator=gen)]
    popularity[torch.randperm(n_experts, generator=gen)[: n_experts // 3]] = 0
    ids = torch.multinomial(popularity.expand(batch, -1), top_k, generator=gen).to(torch.int32)
    if inactive:
        ids[::3, 0] = -1
    actual = ggml.mul_mat_id(w, int(QType.Q4_0), x, ids.cuda(), inactive=inactive).cpu()
    xs = x.cpu().float() if per_route_x else x.cpu().float().unsqueeze(1).expand(-1, top_k, -1)
    expected = torch.zeros(batch, top_k, 64)
    for e in ids.unique().tolist():
        if e >= 0:
            route = ids == e
            expected[route] = xs[route] @ ref[e].T
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17, 300])
def test_iq1_m_matrix_and_moe_routes_match_cpu(dtype, batch):
    # llama.cpp has no IQ1_M MMQ: above the vector kernel's batch this runs FreeToken's IQ1_M tile loader
    exact_sums = dtype != torch.float32
    w, ref = _iq1_m_weights((64, 512), exact_sums)
    x = _activation((batch, 512), dtype)
    actual = ggml.mul_mat(w, int(QType.IQ1_M), x)
    torch.testing.assert_close(actual.cpu(), F.linear(x.cpu().float(), ref).to(dtype), atol=2e-4, rtol=1e-3)
    w, ref = _iq1_m_weights((4, 64, 512), exact_sums)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    actual = ggml.mul_mat_id(w, int(QType.IQ1_M), x, ids)
    expected = torch.einsum("tk, tjok -> tjo", x.cpu().float(), ref[ids.cpu().long()]).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_shared_q8_matches_internal_quantization(dtype):
    w, _ = _q4_weights((64, 256))
    x = _activation((1, 256), dtype)
    expected = ggml.mul_mat(w, int(QType.Q4_0), x)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x, xq=ggml.quantize_q8_1(x))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_q4_nondefault_stream():
    w, ref = _q4_weights((64, 256))
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = _activation((1, 256), torch.float32)
        actual = ggml.mul_mat(w, int(QType.Q4_0), x)
        copied = actual.clone()
    stream.synchronize()
    torch.testing.assert_close(copied.cpu(), F.linear(x.cpu(), ref), atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nvfp4_dequantize_and_rows(dtype):
    # 320 values: five NVFP4 blocks, not a multiple of the 256-value super-block the k-quant get_rows path needs
    w, ref = _nvfp4_weights((64, 320))
    actual = ggml.dequantize(w[:64], int(QType.NVFP4), 320, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = ggml.get_rows(w, int(QType.NVFP4), 320, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)



# ---------------------------------------------------------------- Vulkan backend
# The vendored ggml-vulkan module exposes the same op surface; route the ggml API at it
# and reuse the CPU references above. Skipped where glslc / a Vulkan device is absent.


@pytest.fixture
def vk_ggml(monkeypatch):
    from freetoken.kernel import ggml, ggml_vk

    try:
        module = ggml_vk.load_module()
    except Exception as exc:
        pytest.skip(f"Vulkan ggml module unavailable: {exc}")
    monkeypatch.setattr(ggml, "_module", lambda: module)
    return ggml


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vk_q4_dequantize_and_rows(vk_ggml, dtype):
    w, ref = _q4_weights((64, 256))
    actual = vk_ggml.dequantize(w, int(QType.Q4_0), 256, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = vk_ggml.get_rows(w, int(QType.Q4_0), 256, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 17])
def test_vk_q4_matrix_matches_cpu(vk_ggml, dtype, batch):
    w, ref = _q4_weights((64, 256))
    x = _activation((batch, 256), dtype)
    actual = vk_ggml.mul_mat(w, int(QType.Q4_0), x)
    expected = F.linear(x.cpu().float(), ref).to(dtype)
    # the Vulkan pipeline quantizes activations to q8_1 with its own fp16 scale rounding,
    # so even the CUDA-exact inputs carry quantization noise here (unlike the CUDA module)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_route_x", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
def test_vk_q4_moe_routes_match_cpu(vk_ggml, dtype, per_route_x, inactive):
    batch = 5
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((batch, 2, 256) if per_route_x else (batch, 256), dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    if inactive:
        ids[::2, 0] = -1
    actual = vk_ggml.mul_mat_id(w, int(QType.Q4_0), x, ids, inactive=inactive)
    xs = x.cpu().float() if per_route_x else x.cpu().float().unsqueeze(1).expand(-1, 2, -1)
    expected = torch.einsum("tjk, tjok -> tjo", xs, ref[ids.cpu().long().clamp_min(0)])
    expected[ids.cpu() < 0] = 0
    torch.testing.assert_close(actual.cpu(), expected.to(dtype), atol=2e-4, rtol=1e-3)


def test_vk_nvfp4_moe_scale_matches_cpu(vk_ggml):
    w, ref = _nvfp4_weights((4, 256))
    w = w.unsqueeze(1)  # [E, N=1, row_bytes]
    scale = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32, device="cuda")
    x = _activation((3, 256), torch.float32)
    ids = torch.tensor([[0, 1], [2, 3], [3, 0]], dtype=torch.int32, device="cuda")
    actual = vk_ggml.mul_mat_id(w, int(QType.NVFP4), x, ids, scale=scale)
    expected = torch.einsum("tk,tjk->tj", x.cpu(), ref[ids.cpu().long()])
    expected = expected * scale.cpu()[ids.cpu().long()]
    torch.testing.assert_close(actual.cpu().squeeze(-1), expected, atol=2e-4, rtol=1e-3)


def test_vk_swiglu_glu_matches_cpu(vk_ggml):
    w, ref = _q4_weights((4, 128, 256))  # gate rows then up rows
    x = _activation((3, 256), torch.float32)
    ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int32, device="cuda")
    actual = vk_ggml.mul_mat_id_glu(w, int(QType.Q4_0), x, ids, "swiglu")
    h = torch.einsum("tk,tjok->tjo", x.cpu(), ref[ids.cpu().long()])
    gate, up = h.chunk(2, dim=-1)
    expected = F.silu(gate) * up
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


def test_vk_weighted_sum_matches_cpu(vk_ggml):
    y = torch.randn(3, 2, 64, dtype=torch.bfloat16, device="cuda")
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32, device="cuda")
    actual = vk_ggml.moe_weighted_sum(y, weights)
    expected = (y.cpu().float() * weights.cpu().unsqueeze(-1)).sum(dim=1)
    # bf16 output rounding dominates at these magnitudes
    torch.testing.assert_close(actual.cpu().float(), expected, atol=8e-3, rtol=4e-3)


def test_vk_resident_bank_mode_matches_cpu(vk_ggml):
    from freetoken.kernel import ggml_vk

    m = ggml_vk._module()
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((5, 256), torch.float32)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(5)], dtype=torch.int32, device="cuda")
    m.set_bank_mode(True)
    try:
        actual = vk_ggml.mul_mat_id(w, int(QType.Q4_0), x, ids)
        # a second call reads the resident device copy, not a fresh upload
        again = vk_ggml.mul_mat_id(w, int(QType.Q4_0), x, ids)
    finally:
        m.set_bank_mode(False)
    expected = torch.einsum("tk,tjok->tjo", x.cpu(), ref[ids.cpu().long()])
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(again.cpu(), expected, atol=2e-4, rtol=1e-3)


def test_vk_moe_ffn_situ_matches_cpu(vk_ggml):
    E, I, K, H = 4, 32, 64, 64
    wgu, ref_gu = _q4_weights((E, 2 * I, K))
    wdn, ref_dn = _q4_weights((E, H, I))
    x = _activation((3, K), torch.float32)
    ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int32, device="cuda")
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32, device="cuda")
    scales = torch.tensor([[0.5, 0.75, 1.25, 1.0]] * E, dtype=torch.float32, device="cuda")
    actual = vk_ggml.moe_ffn_situ(wgu, int(QType.Q4_0), wdn, int(QType.Q4_0), x, ids, weights,
                                  global_scales=scales, alpha=4.0, limit=25.0)
    expected = torch.zeros(3, H)
    for ti in range(3):
        for j in range(2):
            e = ids[ti, j].item()
            gu = x[ti].cpu() @ ref_gu[e].T
            gate, up = gu.chunk(2)
            gate = gate * scales[e, 0].item()
            up = up * scales[e, 1].item()
            situ = (4.0 * torch.tanh(gate / 4.0)) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))
            y = (situ @ ref_dn[e].T) * scales[e, 2].item()
            expected[ti] += weights[ti, j].item() * y
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


def test_vk_moe_ffn_situ_large_prefill_graph(vk_ggml):
    # a prefill routes to ~all experts: the fused graph exceeds the default node cap
    E, I, K, H = 128, 64, 128, 128
    wgu, ref_gu = _q4_weights((E, 2 * I, K))
    wdn, ref_dn = _q4_weights((E, H, I))
    T = 96
    x = _activation((T, K), torch.float32)
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(0, E, (T, 8), generator=gen).to(torch.int32).cuda()
    weights = torch.softmax(torch.randn(T, 8, generator=gen), dim=-1).cuda()
    actual = vk_ggml.moe_ffn_situ(wgu, int(QType.Q4_0), wdn, int(QType.Q4_0), x, ids, weights, alpha=4.0, limit=25.0)
    expected = torch.zeros(T, H)
    xs = x.cpu()
    for ti in range(T):
        for j in range(8):
            e = ids[ti, j].item()
            gu = xs[ti] @ ref_gu[e].T
            g, u = gu.chunk(2)
            situ = (4.0 * torch.tanh(g / 4.0)) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))
            expected[ti] += weights[ti, j].item() * (situ @ ref_dn[e].T)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------- ik2-vk (maya-style)
def _ik2_ref(wgu, ref_gu, wdn, ref_dn, x, ids, weights, sc, gcap=4.0, ucap=25.0):
    T, k = ids.shape
    H = ref_dn.shape[1]
    expected = torch.zeros(T, H)
    xs = x.cpu().float()
    for ti in range(T):
        for j in range(k):
            e = ids[ti, j].item()
            if e < 0:
                continue
            gu = xs[ti] @ ref_gu[e].T
            g, u = gu.chunk(2)
            g = g * sc[e, 0].item()
            u = u * sc[e, 1].item()
            situ = (gcap * torch.tanh(g / gcap)) * torch.sigmoid(g) * (ucap * torch.tanh(u / ucap))
            # the shaders store h rounded to FP16 (maya ik2_exact's convention)
            situ = situ.half().float()
            expected[ti] += weights[ti, j].item() * ((situ @ ref_dn[e].T) * sc[e, 2].item())
    return expected


def _q6_weights(shape, seed=41):
    # random packed Q6_K super-blocks (gguf-py has no Q6_K encoder): ql[128] qh[64]
    # scales int8[16] d fp16, 210 bytes of 256 values
    rng = np.random.default_rng(seed)
    rows, k = shape
    n_super = k // 256
    q = rng.integers(0, 64, (rows, n_super, 256), dtype=np.uint8)
    packed = np.zeros((rows, n_super, 210), dtype=np.uint8)
    for n in range(2):
        for qd in range(4):
            vals = q[:, :, n * 128 + qd * 32 : n * 128 + qd * 32 + 32]
            lo = (vals & 0xF).astype(np.uint8)
            hi = (vals >> 4).astype(np.uint8)
            if qd < 2:
                packed[:, :, n * 64 + qd * 32 : n * 64 + qd * 32 + 32] |= lo
            else:
                packed[:, :, n * 64 + (qd - 2) * 32 : n * 64 + (qd - 2) * 32 + 32] |= lo << 4
            packed[:, :, 128 + n * 32 : 128 + n * 32 + 32] |= hi << (2 * qd)
    sc = rng.integers(-32, 32, (rows, n_super, 16), dtype=np.int8)
    packed[:, :, 192:208] = sc.view(np.uint8)
    d = np.full((rows, n_super), 0.01, dtype=np.float16)
    packed[:, :, 208:210] = d.view(np.uint8).reshape(rows, n_super, 2)
    packed = packed.reshape(rows, n_super * 210)
    return torch.from_numpy(packed.copy()), torch.from_numpy(quants.dequantize(packed, QType.Q6_K))


@pytest.mark.parametrize("ty", ["q8_0", "nvfp4", "q6_k"])
def test_ik2_vk_moe_matches_cpu(ty):
    from freetoken.kernel import ik2_vk

    E, T = 4, 5
    I, K, H = (256, 256, 256) if ty == "q6_k" else (64, 128, 128)
    if ty == "q8_0":
        rng = np.random.default_rng(11)
        def qw(shape):
            vals = rng.normal(0, 0.1, shape).astype(np.float32)
            p = quants.quantize(vals, QType.Q8_0)
            return torch.from_numpy(p), torch.from_numpy(quants.dequantize(p, QType.Q8_0))
        wgu, ref_gu = qw((E, 2 * I, K))
        wdn, ref_dn = qw((E, H, I))
        qtype = QType.Q8_0
    elif ty == "q6_k":
        sg, rg = _q6_weights((E * 2 * I, K))
        sd, rd = _q6_weights((E * H, I))
        wgu, ref_gu = sg.view(E, 2 * I, -1), rg.view(E, 2 * I, K)
        wdn, ref_dn = sd.view(E, H, -1), rd.view(E, H, I)
        qtype = QType.Q6_K
    else:
        sg, rg = _nvfp4_weights((E * 2 * I, K))
        sd, rd = _nvfp4_weights((E * H, I))
        wgu, ref_gu = sg[: E * 2 * I].cpu().view(E, 2 * I, -1), rg.view(E, 2 * I, K)
        wdn, ref_dn = sd[: E * H].cpu().view(E, H, -1), rd.view(E, H, I)
    x = _activation((T, K), torch.float32)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(T)], dtype=torch.int32, device="cuda")
    ids[::2, 0] = -1
    weights = torch.softmax(torch.randn(T, 2), dim=-1).cuda()
    sc = torch.tensor([[0.5, 0.75, 1.25, 1.0]] * E, dtype=torch.float32, device="cuda")
    qt = int(QType.NVFP4 if ty == "nvfp4" else qtype)
    actual = ik2_vk.moe(wgu, qt, wdn, qt, x, ids, weights, sc)
    expected = _ik2_ref(wgu, ref_gu, wdn, ref_dn, x, ids, weights, sc)
    # fixed-order FMA chains vs torch's matmul reduction, plus the FP16 h rounding:
    # a tie flipped by the reduction order moves h by one FP16 ulp (the synthetic
    # Q6_K blocks carry large weights, so that ulp is visible at the output)
    atol, rtol = (0.15, 5e-3) if ty == "q6_k" else (2e-3, 2e-3)
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)


def test_ik2_vk_moe_multi_chunk_prefill():
    from freetoken.kernel import ik2_vk

    # more routed experts than one chunk: the dequant must follow the sparse expert ids
    E, I, K, H, T = 128, 64, 128, 128, 30
    rng = np.random.default_rng(17)
    def qw(shape):
        vals = rng.normal(0, 0.1, shape).astype(np.float32)
        p = quants.quantize(vals, QType.Q8_0)
        return torch.from_numpy(p), torch.from_numpy(quants.dequantize(p, QType.Q8_0))
    wgu, ref_gu = qw((E, 2 * I, K))
    wdn, ref_dn = qw((E, H, I))
    x = _activation((T, K), torch.bfloat16)
    ids = torch.randint(0, E, (T, 8), generator=torch.Generator().manual_seed(5)).to(torch.int32).cuda()
    weights = torch.softmax(torch.randn(T, 8), dim=-1).cuda()
    sc = torch.ones(E, 4, device="cuda")
    actual = ik2_vk.moe(wgu, int(QType.Q8_0), wdn, int(QType.Q8_0), x, ids, weights, sc)
    expected = _ik2_ref(wgu, ref_gu, wdn, ref_dn, x, ids, weights, sc)
    # bf16 activations + fixed-order chains + FP16 h rounding
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0.3, rtol=5e-3)
