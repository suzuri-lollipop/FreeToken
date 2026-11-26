"""Op-level parity for the ERNIE 5.5 specifics, against independent references
transcribed from baidu/vLLM-ERNIE (mHC eps-outside RMS, multimax, split-feature
routing, SituGLU, VHA fold) and a brute-force softmax oracle for the sink kernel.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_mhc_ernie_rms_factor_matches_reference():
    """The per-layer mixer uses 1/(sqrt(mean)+eps); a large eps separates it from
    the shared rsqrt(mean+eps) -- the exact bug the vLLM-ERNIE test guards."""
    from freetoken.layers.mhc import mhc_fused_post_pre, mhc_pre

    torch.manual_seed(0)
    T, n, H = 8, 4, 512
    res = torch.randn(T, n, H, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(2 * n + n * n, n * H, dtype=torch.float32, device="cuda") * 1e-2
    scale = torch.ones(3, dtype=torch.float32, device="cuda")
    base = torch.randn(2 * n + n * n, dtype=torch.float32, device="cuda") * 0.1
    rms_eps, eps, iters, alpha = 0.5, 1e-6, 20, 2.0

    # vLLM-ERNIE tests/models/ernie5_5/test_mhc.py::_mhc_pre_ref
    x = res.view(T, -1).float()
    mixes = (x @ fn.t()) / (x.square().mean(dim=-1, keepdim=True).sqrt() + rms_eps)
    pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n])
    post_ref = torch.sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n]) * alpha
    comb = mixes[:, 2 * n :] * scale[2] + base[2 * n :]
    comb = torch.softmax(comb.view(-1, n, n), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    li_ref = ((pre + eps).unsqueeze(-1) * res.float()).sum(dim=1).to(torch.bfloat16)

    post, comb_t, li = mhc_pre(res, fn, scale, base, rms_eps, eps, alpha, iters, rms_eps_outside=True)
    torch.testing.assert_close(post.squeeze(-1), post_ref, atol=5e-3, rtol=0)
    torch.testing.assert_close(comb_t, comb, atol=5e-3, rtol=0)
    torch.testing.assert_close(li, li_ref, atol=5e-2, rtol=1e-2)

    _, post_f, comb_f, li_f = mhc_fused_post_pre(
        None, res, None, None, fn, scale, base, rms_eps, eps, alpha, iters, rms_eps_outside=True
    )
    torch.testing.assert_close(post_f.squeeze(-1), post_ref, atol=5e-3, rtol=0)
    torch.testing.assert_close(comb_f, comb, atol=5e-3, rtol=0)
    torch.testing.assert_close(li_f, li_ref, atol=5e-2, rtol=1e-2)

    # the two RMS variants must actually differ at large eps (guards a silent
    # fallback to the shared factor)
    p_out, _, _ = mhc_pre(res, fn, scale, base, rms_eps, eps, alpha, iters, rms_eps_outside=True)
    p_in, _, _ = mhc_pre(res, fn, scale, base, rms_eps, eps, alpha, iters, rms_eps_outside=False)
    assert (p_out - p_in).abs().max() > 5e-2


def test_multimax_bitwise_and_fallback():
    from freetoken.kernel.triton.ernie55 import multimax_logits

    torch.manual_seed(1)
    x = torch.randn(5, 1000, device="cuda", dtype=torch.float32) * 8
    r = torch.tensor([15.625, 15.9375, 8.75, 0.3671875], device="cuda")
    t = torch.tensor([-0.1337890625, 0.279296875, -0.09765625, -0.0301513671875], device="cuda")
    ref = (
        x + t[0] * (r[0] - x).relu() + t[1] * (x - r[1]).relu()
        + t[2] * (r[2] - x).relu().square() + t[3] * (x - r[3]).relu().square()
    )
    y = x.clone()
    multimax_logits(y, r, t)
    assert torch.equal(y, ref)
    xc, rc, tc = x.cpu(), r.cpu(), t.cpu()
    refc = (
        xc + tc[0] * (rc[0] - xc).relu() + tc[1] * (xc - rc[1]).relu()
        + tc[2] * (rc[2] - xc).relu().square() + tc[3] * (xc - rc[3]).relu().square()
    )
    assert torch.equal(multimax_logits(xc, rc, tc), refc)


@pytest.mark.parametrize("num_tokens", [1, 7, 256])
def test_split_sigmoid_sum_bitwise(num_tokens):
    from freetoken.kernel.triton.ernie55 import split_sigmoid_sum

    torch.manual_seed(0)
    logits = torch.randn(num_tokens, 1024, device="cuda", dtype=torch.float32) * 4
    edges = torch.tensor(
        [float("inf"), -float("inf"), 0.0, -0.0, 30.0, -88.0, -95.0, -104.0], device="cuda"
    )
    logits[0, :8] = edges
    logits[-1, 512:520] = edges.flip(0)
    tiny = torch.tensor([-88.0, -87.5, -88.5, -float("inf")], device="cuda")
    logits[:, 16:20] = tiny
    logits[:, 528:532] = tiny.roll(1)
    ref = logits[:, :512].sigmoid() + logits[:, 512:].sigmoid()
    assert torch.equal(split_sigmoid_sum(logits, 512), ref)


@pytest.mark.parametrize("ties", [False, True])
def test_routing_matches_torch_reference(ties):
    from freetoken.models.ernie5_5_moe.moe import ernie55_routing

    torch.manual_seed(2)
    scores = torch.rand(64, 512, device="cuda", dtype=torch.float32) * 2
    bias = torch.randn(512, device="cuda") * 0.1
    scale = torch.rand(512, device="cuda") + 2.5
    if ties:
        scores = (scores * 4).round() / 4
        bias = (bias * 8).round() / 8
    weights, ids = ernie55_routing(scores, 10, True, bias, scale)
    ref_ids = torch.topk(scores + bias, 10, dim=-1, sorted=False).indices
    ref_w = scores.gather(1, ref_ids)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True).clamp_min(1e-20)
    ref_w = ref_w * scale[ref_ids]
    assert torch.equal(ids, ref_ids.to(torch.int32))
    assert torch.equal(weights, ref_w.float())


def test_situ_matches_torch_reference():
    from freetoken.layers.activation import situ_and_mul

    torch.manual_seed(3)
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16) * 6
    beta, linear_beta = 4.0, 25.0
    gate, up = x[..., :128].float(), x[..., 128:].float()
    ref = (beta * torch.tanh(gate / beta) * torch.sigmoid(gate)) * (
        linear_beta * torch.tanh(up / linear_beta)
    )
    torch.testing.assert_close(situ_and_mul(x, alpha=beta, limit=linear_beta).float(), ref, atol=2e-2, rtol=2e-2)


def test_vha_fold_is_identity_plus_low_rank():
    from freetoken.models.ernie5_5_moe.attention import _VHAPostMix

    torch.manual_seed(4)
    vha = _VHAPostMix(64)
    vha.vha_U = torch.randn(64, 16, dtype=torch.bfloat16, device="cuda") * 0.05
    vha.vha_V = torch.randn(64, 16, dtype=torch.bfloat16, device="cuda") * 0.05
    vha.fold()
    o = torch.randn(7, 64, 512, dtype=torch.bfloat16, device="cuda")
    ref = o + torch.matmul(vha.vha_V, torch.matmul(vha.vha_U.t(), o))
    torch.testing.assert_close(vha.forward(o), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("splits", [0, 4])
def test_sparse_mla_sink_matches_softmax_oracle(splits):
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    torch.manual_seed(5)
    b, m, h, dv, dr = 2, 5, 64, 512, 64
    rows = 300
    pool = torch.randn(rows, dv + dr, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(b, m, h, dv + dr, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, rows, (b, m, 40), device="cuda", dtype=torch.int32)
    idx[:, :, 35:] = -1
    cnt = torch.full((b, m), 35, device="cuda", dtype=torch.int32)
    sink = torch.randn(h, device="cuda", dtype=torch.float32)
    scale = 0.0625
    o = glm_dsa_sparse_attn(q, pool, idx, scale, counts=cnt, d_v=dv, force_splits=splits, sink=sink)
    ref = torch.zeros(b, m, h, dv, device="cuda", dtype=torch.float32)
    for bi in range(b):
        for mi in range(m):
            n = int(cnt[bi, mi])
            kv = pool[idx[bi, mi, :n].long()].float()
            s = (q[bi, mi].float() @ kv.t()) * scale
            p = torch.softmax(torch.cat([s, sink[:, None]], dim=-1), dim=-1)
            ref[bi, mi] = p[:, :n] @ kv[:, :dv]
    torch.testing.assert_close(o.float(), ref, atol=1e-2, rtol=1e-2)
