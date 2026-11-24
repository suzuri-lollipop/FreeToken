"""Inkling-2-Mini against an fp32 transcription of the looped decoder (llama.cpp's
inkling2_loop graph + the transformers inkling2 per-layer math), on a miniature config
with the same structure: 8 physical layers, block 2, core [2, 6) run twice, SWA/full
alternation, dynamic K/V convs on the full layers, cross-pass controllers, SITU MoE.

The attention stub implements the rel-bias + per-query-sink semantics eagerly; the
triton kernels that serve them in production are diffed in tests/kernels.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

H, NH, NK, HD = 64, 4, 2, 16
D_REL, EXTENT, WINDOW = 4, 8, 8
QK_CLIP, REL_CLIP = math.sqrt(2 * HD), 2.0
LOG_FLOOR, LOG_ALPHA = 4, 0.1
KERNEL, RANK, BLOCK = 4, 8, 2
PHYS, CORE_START, CORE_END, PASSES = 8, 2, 6, 2
E, TOPK, INTER = 4, 2, 32
VOCAB, UNPADDED, EPS = 32, 30, 1e-6
GATE_CAP, UP_CAP, ROUTE_SCALE, SOFTCAP = 4.0, 25.0, 8.0, 30.0
SWA_PHYS = [0, 2, 4, 6]
_RDTYPE = torch.bfloat16  # the reference rounds where the serving dtype does; fp32 mode checks the math sharply
VISITS = list(range(CORE_START)) + list(range(CORE_START, CORE_END)) * PASSES + list(range(CORE_END, PHYS))
KV_DIM = NK * HD
CONV_CH = 2 * KV_DIM + 2 * H


def _hf_dict() -> dict:
    return dict(
        architectures=["Inkling2LoopForConditionalGeneration"],
        model_type="inkling2_loop",
        image_token_id=29,
        text_config=SimpleNamespace(
            model_type="inkling2_loop_text",
            hidden_size=H, num_hidden_layers=PHYS, vocab_size=VOCAB, unpadded_vocab_size=UNPADDED,
            num_attention_heads=NH, num_key_value_heads=NK, head_dim=HD,
            swa_num_attention_heads=NH, swa_num_key_value_heads=NK, swa_head_dim=HD,
            d_rel=D_REL, rel_extent=EXTENT, sliding_window_size=WINDOW,
            qk_l2_clip_threshold=QK_CLIP, rel_l2_clip_threshold=REL_CLIP,
            log_scaling_n_floor=LOG_FLOOR, log_scaling_alpha=LOG_ALPHA,
            sconv_kernel_size=KERNEL, rms_norm_eps=EPS,
            local_layer_ids=SWA_PHYS, n_layers_per_attn_res_block=BLOCK,
            n_routed_experts=E, num_experts_per_tok=TOPK, n_shared_experts=0,
            intermediate_size=INTER, mlp_gate_softcap=GATE_CAP, mlp_up_softcap=UP_CAP,
            route_scale=ROUTE_SCALE, use_gate_bias=True, final_logit_softcapping=SOFTCAP,
            loop_config=dict(core_start=CORE_START, core_end=CORE_END, num_passes=PASSES, gate_rank=RANK, is_varloop=False),
        ),
    )


def _rms(x, w):
    xf = x.float()
    return w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS))


def _urms(x, eps=1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def _clip(x, thr, scale=None):
    if scale is not None:
        x = x * scale
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (thr / (norm + 1e-6)).clamp(max=1.0)


def _situ(g, u):
    return (GATE_CAP * torch.tanh(g / GATE_CAP)) * torch.sigmoid(g) * (UP_CAP * torch.tanh(u / UP_CAP))


class _RefConv:
    """Per-request lag history, independent of the model's packed conv."""

    def __init__(self, channels, dynamic_heads=0):
        self.hist = {}
        self.channels = channels
        self.dh = dynamic_heads

    def __call__(self, x, w_lag, req, dyn=None):
        hist = self.hist.setdefault(req, torch.zeros(self.channels, KERNEL - 1, device=x.device))
        outs = []
        for t in range(x.shape[0]):
            xt = x[t]
            window = torch.cat([xt.unsqueeze(-1), hist.flip(-1)], dim=-1)  # [C, K] lag order
            taps = w_lag
            if dyn is not None:
                hd = self.channels // self.dh
                taps = w_lag.view(self.dh, hd, KERNEL) + dyn[t].view(self.dh, 1, KERNEL)
                taps = taps.expand(self.dh, hd, KERNEL).reshape(self.channels, KERNEL)
            outs.append((window * taps).sum(-1))
            hist = torch.cat([hist[:, 1:], xt.unsqueeze(-1)], dim=-1)
        self.hist[req] = hist
        out = torch.stack(outs, dim=0)
        scale = 1.0 / math.sqrt((2 if dyn is not None else 1) * KERNEL + 1)
        return ((out + x) * scale).to(_RDTYPE)


class _EagerBackend:
    """Rel-bias + per-query-sink attention with a per-visit KV store (global positions)."""

    def __init__(self):
        self.kv = {}

    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        T = q.shape[0]
        k = k.view(T, NK, HD)
        v = v.view(T, NK, HD)
        cu = batch.fla_metadata.cu_seqlens.tolist()
        pos = batch.get_attn_positions().tolist()
        store = self.kv.setdefault(layer_id, {})
        outs = torch.zeros_like(q)
        for i in range(len(cu) - 1):
            lo, hi = cu[i], cu[i + 1]
            entry = store.setdefault(i, {"k": [], "v": [], "pos": []})
            entry["k"].append(k[lo:hi])
            entry["v"].append(v[lo:hi])
            entry["pos"].extend(pos[lo:hi])
            kk = torch.cat(entry["k"], dim=0).repeat_interleave(NH // NK, dim=1)
            vv = torch.cat(entry["v"], dim=0).repeat_interleave(NH // NK, dim=1)
            kp = torch.tensor(entry["pos"], device=q.device)
            for j in range(lo, hi):
                qp = pos[j]
                logits = torch.einsum("hd,lhd->hl", q[j], kk).float() * HD**-0.5
                dist = (qp - kp).clamp(0, attn_spec.rel_table.shape[-1] - 1)
                logits = logits + attn_spec.rel_table[j][:, dist]
                mask = kp <= qp
                if attn_spec.sliding_window is not None:
                    mask = mask & (kp + attn_spec.sliding_window > qp)
                logits = logits.masked_fill(~mask, float("-inf"))
                sink = attn_spec.sinks[j]
                m = torch.maximum(logits.max(dim=-1).values, sink)
                p = torch.exp(logits - m[:, None])
                denom = p.sum(dim=-1) + torch.exp(sink - m)
                outs[j] = torch.einsum("hl,lhd->hd", p / denom[:, None], vv.float()).to(outs.dtype)
        return outs


def _reference(sd, ids, req_lens, backend=None):
    """fp32 logits of the whole looped forward; returns (logits, conv histories) for decode continuation."""
    device = ids.device
    x = _rms(sd["model.embed_tokens.weight"][ids], sd["model.embed_norm.weight"]).to(_RDTYPE)
    offsets = [0]
    for n in req_lens:
        offsets.append(offsets[-1] + n)
    pos = torch.cat([torch.arange(n, device=device) for n in req_lens])

    convs = {}

    def conv(key, channels, dh=0):
        if key not in convs:
            convs[key] = _RefConv(channels, dh)
        return convs[key]

    def mix(blocks, cur, query):
        states = torch.stack([b.float() for b in blocks] + [cur.float()], dim=1)
        states = _urms(states)
        scores = torch.einsum("tsh,qh->tsq", states, query) * H**-0.5
        return torch.einsum("tsq,tsh->tqh", torch.softmax(scores, dim=1), states)

    def ctrl(cur, msg, c):
        cur, msg = cur.to(_RDTYPE), msg.to(_RDTYPE)
        gate = torch.cat([_urms(cur), _urms(msg)], dim=-1).to(_RDTYPE)
        h = gate @ sd[f"{c}.down_weight"].T
        gain = torch.tanh(h @ sd[f"{c}.gain_weight"].T)
        forget = torch.tanh(h @ sd[f"{c}.forget_weight"].T)
        return ((1.0 - forget) * cur + gain * msg).to(_RDTYPE)

    def layer(ip, x, blocks, messages, visit):
        p = f"model.layers.{ip}"
        a = f"{p}.self_attn"
        swa = ip in SWA_PHYS
        inpSA = x
        if ip > 0:
            mixed = mix(blocks, x, sd[f"{p}.attn_res_query"])
            inputs = [mixed[:, 0]] * 3 if swa else [mixed[:, j] for j in range(3)]
        else:
            inputs = [x, x, x]
        if messages is not None:
            inputs = [ctrl(inputs[j], messages[:, j], f"{p}.controllers.{j}") for j in range(3)]
        if swa:
            normed = [_rms(i, sd[f"{a}.input_layernorm.weight"]).to(_RDTYPE) for i in inputs] if messages is not None \
                else [_rms(inputs[0], sd[f"{a}.input_layernorm.weight"]).to(_RDTYPE)] * 3
        else:
            normed = [_rms(inputs[j], sd[f"{a}.{qkv}_input_norm.weight"]).to(_RDTYPE) for j, qkv in enumerate("qkv")]
        xq, xk, xv = normed
        q = xq @ sd[f"{a}.q_proj.weight"].T
        k = xk @ sd[f"{a}.k_proj.weight"].T
        v = xv @ sd[f"{a}.v_proj.weight"].T
        r = xq @ sd[f"{a}.r_proj.weight"].T
        gate = xq @ sd[f"{a}.gate_proj.weight"].T
        sinks = xq @ sd[f"{a}.sink_proj.weight"].T
        T = x.shape[0]
        for name, raw, xin in (("k", k, xk), ("v", v, xv)):
            w = sd[f"{a}.{name}_conv.weight"]
            dyn = None
            if not swa:
                dyn = (xin @ sd[f"{a}.{name}_conv.conv_proj.weight"].T).view(T, NK, KERNEL)
            c = conv((visit, name), KV_DIM, 0 if swa else NK)
            for i, n in enumerate(req_lens):
                seg = raw[offsets[i] : offsets[i + 1]]
                dseg = dyn[offsets[i] : offsets[i + 1]] if dyn is not None else None
                raw[offsets[i] : offsets[i + 1]] = c(seg, w, i, dseg)
        q = _clip(q.view(T, NH, HD), QK_CLIP, sd[f"{a}.q_clip_scale"]).view(T, NH * HD)
        k = _clip(k.view(T, NK, HD), QK_CLIP).view(T, KV_DIM)
        v = _urms(v.view(T, NK, HD), EPS).to(_RDTYPE).view(T, KV_DIM)
        r = _clip(r.view(T, NH, D_REL), REL_CLIP, sd[f"{a}.rel_clip_scale"])
        if not swa:
            tau = 1.0 + LOG_ALPHA * torch.log(((pos + 1).float() / LOG_FLOOR).clamp(min=1.0))
            q = (q.view(T, NH, HD) * tau[:, None, None]).to(_RDTYPE).view(T, NH * HD)
            r = (r * tau[:, None, None]).to(_RDTYPE)
        extent = WINDOW if swa else EXTENT
        table = F.pad(r @ sd[f"{a}.rel_bias"], (0, 1))
        spec = SimpleNamespace(sliding_window=WINDOW if swa else None, rel_table=table, sinks=sinks)
        out = backend.forward(q.view(T, NH, HD), k, v, visit, None, spec)
        out = (out.reshape(T, NH * HD) * torch.sigmoid(gate)).to(_RDTYPE)
        out = out @ sd[f"{a}.o_proj.weight"].T
        # attention-output conv (single segment per request)
        c = conv((visit, "attn_post"), H)
        for i, n in enumerate(req_lens):
            seg = out[offsets[i] : offsets[i + 1]]
            out[offsets[i] : offsets[i + 1]] = c(seg, sd[f"{a}.attn_post_conv.weight"], i)
        if ip % BLOCK == 0:
            blocks.append(inpSA)
            residual = out.float()
        else:
            residual = inpSA.float() + out.float()
        ffn_in = mix(blocks, residual, sd[f"{p}.ffn_res_query"])[:, 0]
        if messages is not None:
            ffn_in = ctrl(ffn_in, messages[:, 3], f"{p}.controllers.3")
        h = _rms(ffn_in.to(_RDTYPE), sd[f"{p}.mlp_norm.weight"]).to(_RDTYPE)
        logits = h.float() @ sd[f"{p}.mlp.gate.weight"].T
        scores = torch.sigmoid(logits)
        idx = torch.topk(scores + sd[f"{p}.mlp.gate_bias"], TOPK, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE
        moe = torch.zeros_like(h)
        for e in range(E):
            hit = (idx == e).any(dim=-1)
            if not hit.any():
                continue
            xe = h[hit]
            gu = (xe.to(_RDTYPE) @ sd[f"{p}.mlp.experts.gate_up_proj"][e].T).float()
            gate_e, up_e = gu.chunk(2, dim=-1)
            ye = (_situ(gate_e, up_e).to(_RDTYPE) @ sd[f"{p}.mlp.experts.down_proj"][e].T).float()
            we = (w * (idx == e)).sum(-1)[hit]
            moe[hit] += ye * we[:, None]
        c = conv((visit, "ffn_post"), H)
        for i, n in enumerate(req_lens):
            seg = moe[offsets[i] : offsets[i + 1]]
            moe[offsets[i] : offsets[i + 1]] = c(seg, sd[f"{p}.ffn_post_conv.weight"], i)
        return (residual + moe).to(_RDTYPE)

    blocks = []
    for il in range(CORE_START):
        x = layer(il, x, blocks, None, il)
    prelude, prelude_blocks = x, list(blocks)
    core = CORE_END - CORE_START
    for p in range(PASSES):
        if p > 0:
            bank = blocks[len(prelude_blocks) + 1 :]
            bank_out = x
            x = prelude
            blocks = list(prelude_blocks)
        for ip in range(CORE_START, CORE_END):
            il = ip if p == 0 else PHYS + (p - 1) * core + (ip - CORE_START)
            messages = mix(bank, bank_out, sd[f"model.layers.{ip}.loop_msg_query"]) if p > 0 else None
            x = layer(ip, x, blocks, messages, il)
    for il in range(CORE_END, PHYS):
        x = layer(il, x, blocks, None, il)
    x = mix(blocks, x, sd["model.output_res_query"])[:, 0]
    x = _rms(x, sd["model.norm.weight"]).to(_RDTYPE)
    logits = (x @ sd["lm_head.weight"].T).float()
    logits = SOFTCAP * torch.tanh(logits / SOFTCAP)
    return logits[:, :UNPADDED], convs


def _fake_pool(slots=4):
    state = torch.zeros(len(VISITS), slots, CONV_CH, KERNEL - 1, dtype=torch.float32)

    class Pool:
        def has_slot_state(self, name):
            return name == "inkling2_conv"

        def slot_state(self, name, layer_id=None):
            return state[layer_id]

    return Pool(), state


def _build(monkeypatch, ctx):
    from freetoken.models.inkling2.config import parse_config
    from freetoken.models.inkling2.model import Inkling2LoopForConditionalGeneration

    for mod in ("attention", "conv", "model"):
        monkeypatch.setattr(f"freetoken.models.inkling2.{mod}.get_global_ctx", lambda: ctx)
    monkeypatch.setattr("freetoken.layers.embedding.get_global_ctx", lambda: ctx)
    monkeypatch.setattr("freetoken.layers.moe.get_global_ctx", lambda: ctx)
    config = parse_config(SimpleNamespace(**_hf_dict()))
    model = Inkling2LoopForConditionalGeneration(config)
    torch.manual_seed(0)
    sd = {k: (torch.randn(v.shape) * 0.1).to(v.dtype) for k, v in model.state_dict().items()}
    for name in sd:
        if name.endswith("norm.weight") and "input_norm" not in name:
            sd[name] = sd[name].abs() + 0.5
    # resident expert banks stay out of the state dict (the engine attaches them after load)
    for p in range(PHYS):
        experts = model.model.layers.op_list[p].mlp.experts
        gate_up = (torch.randn(E, 2 * INTER, H) * 0.1).to(_RDTYPE)
        down = (torch.randn(E, H, INTER) * 0.1).to(_RDTYPE)
        experts._expert_banks = {"gate_up": gate_up, "down": down}
        experts._expert_alphas = None
        sd[f"model.layers.{p}.mlp.experts.gate_up_proj"] = gate_up
        sd[f"model.layers.{p}.mlp.experts.down_proj"] = down
    model.load_state_dict({k: v for k, v in sd.items() if ".experts." not in k})
    return model, sd


def _batch(ids, req_lens, pool, decode=False, prefix=0):
    device = ids.device
    cu = torch.tensor([0] + list(torch.tensor(req_lens).cumsum(0)), device=device, dtype=torch.int64)
    fla = SimpleNamespace(
        cache_indices=torch.ones(len(req_lens), device=device, dtype=torch.int32),
        cu_seqlens=cu,
        has_initial_state=torch.tensor([prefix > 0] * len(req_lens), device=device),
        track_dst=None,
        track_boundary_row=None,
    )
    positions = torch.cat([torch.arange(prefix + n, device=device) for n in req_lens]) if not decode \
        else torch.full((len(req_lens),), prefix, device=device, dtype=torch.long)
    return SimpleNamespace(
        input_ids=ids, size=len(req_lens), is_prefill=not decode, is_decode=decode,
        padded_reqs=[SimpleNamespace(extend_len=n, table_idx=1, cached_len=prefix) for n in req_lens],
        fla_metadata=fla, mm_embeds=None, mm_rows=None, features=None,
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: cu[1 : bs + 1] - 1),
        get_attn_positions=lambda: positions,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE / norm kernels")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_matches_the_reference(monkeypatch, dtype):
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    global _RDTYPE
    prev_dtype, prev_device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    _RDTYPE = dtype
    try:
        pool, _ = _fake_pool()
        backend = _EagerBackend()
        ctx = SimpleNamespace(batch=None, attn_backend=backend, linear_state_pool=pool)
        model, sd = _build(monkeypatch, ctx)
        ids = torch.randint(0, UNPADDED, (11,), device="cuda")
        ctx.batch = _batch(ids, [11], pool)
        got = model.forward().float()
        torch.set_default_dtype(torch.float32)
        ref_backend = _EagerBackend()
        want, _ = _reference(sd, ids, [11], backend=_RefBackendAdapter(ref_backend, ctx))
        want = want[-1:]  # the LM head keeps each request's last prefill row
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_device)
    diff = (got - want).abs().mean(-1).max()
    tol = 5e-3 if dtype is torch.float32 else 0.1 * want.std()
    assert diff < tol, (diff.item(), want.std().item())


class _RefBackendAdapter:
    """The reference calls the backend with (q, k, v, visit, None, spec); reuse the eager store."""

    def __init__(self, backend, ctx):
        self.backend = backend
        self.ctx = ctx

    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        return self.backend.forward(q, k, v, layer_id, self.ctx.batch, attn_spec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE / norm kernels")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_continues_the_conv_and_kv_state(monkeypatch, dtype):
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    global _RDTYPE
    prev_dtype, prev_device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    _RDTYPE = dtype
    try:
        pool, _ = _fake_pool()
        backend = _EagerBackend()
        ctx = SimpleNamespace(batch=None, attn_backend=backend, linear_state_pool=pool)
        model, sd = _build(monkeypatch, ctx)
        ids = torch.randint(0, UNPADDED, (7,), device="cuda")
        ctx.batch = _batch(ids, [7], pool)
        model.forward()
        ref_backend = _EagerBackend()
        _, convs = _reference(sd, ids, [7], backend=_RefBackendAdapter(ref_backend, ctx))
        # one decode step
        nxt = torch.randint(0, UNPADDED, (1,), device="cuda")
        ctx.batch = _batch(nxt, [1], pool, decode=True, prefix=7)
        got = model.forward().float()
        torch.set_default_dtype(torch.float32)
        ref_ctx_batch = _batch(nxt, [1], pool, decode=True, prefix=7)
        adapter = _RefBackendAdapter(ref_backend, SimpleNamespace(batch=ref_ctx_batch))
        # continue the reference conv histories with the decode token
        want = _ref_decode_step(sd, nxt, convs, adapter)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_device)
    diff = (got - want).abs().mean(-1).max()
    tol = 5e-3 if dtype is torch.float32 else 0.1 * want.std()
    assert diff < tol, (diff.item(), want.std().item())


def _ref_decode_step(sd, ids, convs, backend):
    """One-token continuation of _reference using the prefill's conv histories."""
    device = ids.device
    x = _rms(sd["model.embed_tokens.weight"][ids], sd["model.embed_norm.weight"]).to(_RDTYPE)
    pos = torch.tensor([7], device=device)
    req_lens = [1]
    offsets = [0, 1]

    def mix(blocks, cur, query):
        states = _urms(torch.stack([b.float() for b in blocks] + [cur.float()], dim=1))
        scores = torch.einsum("tsh,qh->tsq", states, query) * H**-0.5
        return torch.einsum("tsq,tsh->tqh", torch.softmax(scores, dim=1), states)

    def ctrl(cur, msg, c):
        cur, msg = cur.to(_RDTYPE), msg.to(_RDTYPE)
        gate = torch.cat([_urms(cur), _urms(msg)], dim=-1).to(_RDTYPE)
        h = gate @ sd[f"{c}.down_weight"].T
        return ((1.0 - torch.tanh(h @ sd[f"{c}.forget_weight"].T)) * cur + torch.tanh(h @ sd[f"{c}.gain_weight"].T) * msg).to(_RDTYPE)

    def layer(ip, x, blocks, messages, visit):
        p, a = f"model.layers.{ip}", f"model.layers.{ip}.self_attn"
        swa = ip in SWA_PHYS
        inpSA = x
        if ip > 0:
            mixed = mix(blocks, x, sd[f"{p}.attn_res_query"])
            inputs = [mixed[:, 0]] * 3 if swa else [mixed[:, j] for j in range(3)]
        else:
            inputs = [x, x, x]
        if messages is not None:
            inputs = [ctrl(inputs[j], messages[:, j], f"{p}.controllers.{j}") for j in range(3)]
        if swa:
            n = _rms(inputs[0], sd[f"{a}.input_layernorm.weight"]).to(_RDTYPE)
            normed = [n] * 3 if messages is None else [_rms(i, sd[f"{a}.input_layernorm.weight"]).to(_RDTYPE) for i in inputs]
        else:
            normed = [_rms(inputs[j], sd[f"{a}.{qkv}_input_norm.weight"]).to(_RDTYPE) for j, qkv in enumerate("qkv")]
        xq, xk, xv = normed
        q = xq @ sd[f"{a}.q_proj.weight"].T
        k = xk @ sd[f"{a}.k_proj.weight"].T
        v = xv @ sd[f"{a}.v_proj.weight"].T
        r = xq @ sd[f"{a}.r_proj.weight"].T
        gate = xq @ sd[f"{a}.gate_proj.weight"].T
        sinks = xq @ sd[f"{a}.sink_proj.weight"].T
        for name, raw, xin in (("k", k, xk), ("v", v, xv)):
            dyn = None if swa else (xin @ sd[f"{a}.{name}_conv.conv_proj.weight"].T).view(1, NK, KERNEL)
            raw[:] = convs[(visit, name)](raw, sd[f"{a}.{name}_conv.weight"], 0, dyn)
        q = _clip(q.view(1, NH, HD), QK_CLIP, sd[f"{a}.q_clip_scale"]).view(1, NH * HD)
        k = _clip(k.view(1, NK, HD), QK_CLIP).view(1, KV_DIM)
        v = _urms(v.view(1, NK, HD), EPS).to(_RDTYPE).view(1, KV_DIM)
        r = _clip(r.view(1, NH, D_REL), REL_CLIP, sd[f"{a}.rel_clip_scale"])
        if not swa:
            tau = 1.0 + LOG_ALPHA * torch.log(((pos + 1).float() / LOG_FLOOR).clamp(min=1.0))
            q = (q.view(1, NH, HD) * tau[:, None, None]).to(_RDTYPE).view(1, NH * HD)
            r = (r * tau[:, None, None]).to(_RDTYPE)
        table = F.pad(r @ sd[f"{a}.rel_bias"], (0, 1))
        spec = SimpleNamespace(sliding_window=WINDOW if swa else None, rel_table=table, sinks=sinks)
        out = backend.forward(q.view(1, NH, HD), k, v, visit, None, spec).reshape(1, NH * HD)
        out = (out * torch.sigmoid(gate)).to(_RDTYPE) @ sd[f"{a}.o_proj.weight"].T
        out[:] = convs[(visit, "attn_post")](out, sd[f"{a}.attn_post_conv.weight"], 0)
        if ip % BLOCK == 0:
            blocks.append(inpSA)
            residual = out.float()
        else:
            residual = inpSA.float() + out.float()
        ffn_in = mix(blocks, residual, sd[f"{p}.ffn_res_query"])[:, 0]
        if messages is not None:
            ffn_in = ctrl(ffn_in, messages[:, 3], f"{p}.controllers.3")
        h = _rms(ffn_in.to(_RDTYPE), sd[f"{p}.mlp_norm.weight"]).to(_RDTYPE)
        scores = torch.sigmoid(h.float() @ sd[f"{p}.mlp.gate.weight"].T)
        idx = torch.topk(scores + sd[f"{p}.mlp.gate_bias"], TOPK, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE
        moe = torch.zeros_like(h)
        for e in range(E):
            hit = (idx == e).any(dim=-1)
            if not hit.any():
                continue
            gu = (h[hit].to(_RDTYPE) @ sd[f"{p}.mlp.experts.gate_up_proj"][e].T).float()
            ge, ue = gu.chunk(2, dim=-1)
            ye = (_situ(ge, ue).to(_RDTYPE) @ sd[f"{p}.mlp.experts.down_proj"][e].T).float()
            moe[hit] += ye * (w * (idx == e)).sum(-1)[hit][:, None]
        moe[:] = convs[(visit, "ffn_post")](moe, sd[f"{p}.ffn_post_conv.weight"], 0)
        return (residual + moe).to(_RDTYPE)

    blocks = []
    for il in range(CORE_START):
        x = layer(il, x, blocks, None, il)
    prelude, prelude_blocks = x, list(blocks)
    core = CORE_END - CORE_START
    for p in range(PASSES):
        if p > 0:
            bank = blocks[len(prelude_blocks) + 1 :]
            bank_out = x
            x = prelude
            blocks = list(prelude_blocks)
        for ip in range(CORE_START, CORE_END):
            il = ip if p == 0 else PHYS + (p - 1) * core + (ip - CORE_START)
            messages = mix(bank, bank_out, sd[f"model.layers.{ip}.loop_msg_query"]) if p > 0 else None
            x = layer(ip, x, blocks, messages, il)
    for il in range(CORE_END, PHYS):
        x = layer(il, x, blocks, None, il)
    x = _rms(mix(blocks, x, sd["model.output_res_query"])[:, 0], sd["model.norm.weight"]).to(_RDTYPE)
    logits = SOFTCAP * torch.tanh((x @ sd["lm_head.weight"].T).float() / SOFTCAP)
    return logits[:, :UNPADDED]


# ---------------------------------------------------------------- DSpark drafter (mtp.safetensors)
# The reference below transcribes maya-ik2's drafter kernels (src/ik2/ik2_draft.cu, v1.1.73)
# independently of the serving module: BF16 gemm inputs with F32 sums and F32 outputs,
# F32 norms / rope / attention / GRU gates, the silu and rnn-input BF16 roundings, and the
# FP16 folded Markov bias. A change to the draft math (mix order, rope half, gate order,
# window bounds) turns these red.

def _draft_sd(C=64, R=16, V=32, NH=4, NK=2, F=32, n_layers=3, taps=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    bf = lambda *s: (torch.randn(*s, generator=g) * 0.1).to(torch.bfloat16)
    sd = {
        "fuse_weight": bf(n_layers + 1, taps),
        "norm.weight": bf(C),
        "mask_embed": bf(C),
        "markov_head.dflare_norm.weight": bf(C),
        "markov_head.markov_w1.weight": bf(V, R),
        "markov_head.rnn_proj.weight": bf(3 * R, 2 * R + 2 * C),
        "markov_head.rnn_proj.bias": bf(3 * R),
        "markov_head.hidden_out.weight": bf(C, R),
        "markov_head.hidden_out.bias": bf(C),
    }
    for i in range(n_layers):
        b, a = f"layers.{i}.", f"layers.{i}.self_attn."
        sd.update({
            b + "input_layernorm.weight": bf(C),
            b + "post_attention_layernorm.weight": bf(C),
            b + "dflare_norm.weight": bf(C),
            a + "q_norm.weight": bf(128),
            a + "k_norm.weight": bf(128),
            a + "k_norm_ctx.weight": bf(128),
            a + "q_proj.weight": bf(NH * 128, C),
            a + "k_proj.weight": bf(NK * 128, C),
            a + "v_proj.weight": bf(NK * 128, C),
            a + "o_proj.weight": bf(C, NH * 128),
            a + "k_proj_ctx.weight": bf(NK * 128, C),
            a + "v_proj_ctx.weight": bf(NK * 128, C),
            b + "mlp.gate_proj.weight": bf(F, C),
            b + "mlp.up_proj.weight": bf(F, C),
            b + "mlp.down_proj.weight": bf(C, F),
        })
    return sd


def _ref_dspark(sd, taps, anchor, anchor_h, P, head_logits, n_vocab, k=7, full_cap=8192):
    """Maya's drafter in fp32: returns (draft tokens, block hidden, base logits)."""
    import math as _m

    C = sd["norm.weight"].shape[0]
    n_layers = sum(1 for key in sd if key.endswith(".input_layernorm.weight") and key.startswith("layers."))
    n_taps = sd["fuse_weight"].shape[1]
    R = sd["markov_head.markov_w1.weight"].shape[1]
    NH = sd["layers.0.self_attn.q_proj.weight"].shape[0] // 128
    NK = sd["layers.0.self_attn.k_proj.weight"].shape[0] // 128
    F = sd["layers.0.mlp.gate_proj.weight"].shape[0]
    local = (lambda i: i != 1) if n_layers == 3 else (lambda i: i in (0, 1, 3, 4))
    alpha = torch.softmax(sd["fuse_weight"].float(), dim=-1)
    bf = lambda t: t.to(torch.bfloat16).float()  # Maya's to_bf16: round, then F32 sums
    gemm = lambda x, w: bf(x) @ w.float().T

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()

    def rope(x, pos):
        i = torch.arange(64, dtype=torch.float32)
        ang = pos.float().unsqueeze(-1) * (1e6 ** (-2.0 * i / 128))
        sn, cs = ang.sin(), ang.cos()
        while sn.dim() < x.dim():
            sn, cs = sn.unsqueeze(-2), cs.unsqueeze(-2)
        a, b = x[..., :64], x[..., 64:]
        return torch.cat([a * cs - b * sn, b * cs + a * sn], dim=-1)

    m = taps.shape[0]
    ctx = []
    for li in range(n_layers):
        b, a = f"layers.{li}.", f"layers.{li}.self_attn."
        mixed = rms(torch.einsum("mtc,t->mc", taps.float(), alpha[li]), sd[b + "dflare_norm.weight"])
        K = gemm(mixed, sd[a + "k_proj_ctx.weight"]).view(m, NK, 128)
        V = gemm(mixed, sd[a + "v_proj_ctx.weight"]).view(m, NK, 128)
        K = rope(rms(K, sd[a + "k_norm_ctx.weight"]), torch.arange(m))
        ctx.append((K.reshape(m, NK * 128), V.reshape(m, NK * 128)))
    f_rnn = rms(torch.einsum("tc,t->c", taps[-1].float(), alpha[n_layers]), sd["markov_head.dflare_norm.weight"])

    res = torch.stack([anchor_h.float()] + [sd["mask_embed"].float()] * 7)
    pos = torch.arange(P, P + 8)
    x = res
    for li in range(n_layers):
        b, a = f"layers.{li}.", f"layers.{li}.self_attn."
        if li > 0:
            res = x + res
        h = rms(res, sd[b + "input_layernorm.weight"])
        q = rope(rms(gemm(h, sd[a + "q_proj.weight"]).view(8, NH, 128), sd[a + "q_norm.weight"]), pos)
        kk = rope(rms(gemm(h, sd[a + "k_proj.weight"]).view(8, NK, 128), sd[a + "k_norm.weight"]), pos)
        vv = gemm(h, sd[a + "v_proj.weight"]).view(8, NK, 128)
        win = 1023 if local(li) else full_cap
        lo = max(0, P - win)
        keys = torch.cat([ctx[li][0][lo:], kk.reshape(8, NK * 128)]).view(-1, NK, 128).repeat_interleave(NH // NK, 1)
        vals = torch.cat([ctx[li][1][lo:], vv.reshape(8, NK * 128)]).view(-1, NK, 128).repeat_interleave(NH // NK, 1)
        scores = torch.einsum("qhd,khd->hqk", q, keys) / _m.sqrt(128)
        o = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vals).reshape(8, NH * 128)
        x = gemm(o, sd[a + "o_proj.weight"])
        res = x + res
        h = rms(res, sd[b + "post_attention_layernorm.weight"])
        gu = gemm(h, torch.cat([sd[b + "mlp.gate_proj.weight"], sd[b + "mlp.up_proj.weight"]]))
        g, u = gu[:, :F], gu[:, F:]
        act = bf(g / (1.0 + torch.exp(-g)) * u)
        x = gemm(act, sd[b + "mlp.down_proj.weight"])
    res = x + res
    hidden = rms(res, sd["norm.weight"])
    base = head_logits(hidden[1 : k + 1]).float()

    hout_w = sd["markov_head.hidden_out.weight"]
    hout_b = sd["markov_head.hidden_out.bias"]
    fw = head_logits(hout_w.t().to(torch.bfloat16)).float().t().to(torch.float16)
    fb = head_logits(hout_b.unsqueeze(0).to(torch.bfloat16)).float()[0].to(torch.float16)
    state = torch.zeros(R)
    toks = [anchor]
    for t in range(k):
        rnn_in = bf(torch.cat([state, sd["markov_head.markov_w1.weight"][toks[t]].float(), hidden[t + 1], f_rnn]))
        proj = gemm(rnn_in.unsqueeze(0), sd["markov_head.rnn_proj.weight"])[0] + sd["markov_head.rnn_proj.bias"].float()
        gr, cr, orw = proj[:R], proj[R : 2 * R], proj[2 * R :]
        gate = torch.sigmoid(gr)
        state = gate * state + (1.0 - gate) * torch.tanh(cr)
        latent = torch.tanh(orw)
        bias = fw.float() @ latent + fb.float()  # Maya: FP16 weights, F32 latent, F32 sums
        scores = base[t] + bias
        toks.append(int(scores[:n_vocab].argmax()))
    return torch.tensor(toks[1:]), hidden, base


def _head_fn(head_w):
    return lambda x: (x.to(torch.bfloat16).float() @ head_w.float().T)


def _draft_case(sd, n_vocab, m=24, seed=3):
    g = torch.Generator().manual_seed(seed)
    C = sd["norm.weight"].shape[0]
    taps = torch.randn(m, sd["fuse_weight"].shape[1], C, generator=g)
    anchor = int(torch.randint(0, n_vocab, (1,), generator=g))
    anchor_h = torch.randn(C, generator=g)
    head_w = (torch.randn(sd["markov_head.markov_w1.weight"].shape[0], C, generator=g) * 0.1).to(torch.bfloat16)
    return taps, anchor, anchor_h, m, head_w


@pytest.mark.parametrize("n_layers,taps", [(3, 4), (5, 8)])
def test_dspark_draft_matches_maya_reference(tmp_path, n_layers, taps):
    import safetensors.torch

    from freetoken.models.inkling2.draft import DSparkDraft

    sd = _draft_sd(n_layers=n_layers, taps=taps)
    path = tmp_path / "mtp.safetensors"
    safetensors.torch.save_file(sd, str(path))
    draft = DSparkDraft.load(str(path), torch.device("cpu"), n_vocab=30)
    t, anchor, anchor_h, P, head_w = _draft_case(sd, 30)
    head = _head_fn(head_w)
    ref_toks, ref_hidden, ref_base = _ref_dspark(sd, t, anchor, anchor_h, P, head, 30)
    # the proposer's rings: windowed layers keep the last 1023 rows, the full layer everything
    ctx_k, ctx_v = [], []
    for li in range(n_layers):
        mixed = draft.mix_taps(t, li)
        K, V = draft.context_rows(mixed, li, torch.arange(P))
        ctx_k.append(K)
        ctx_v.append(V)
    f_rnn = draft.f_rnn(t[-1])
    fold_w, fold_b = draft.fold(head)
    toks = draft.propose(anchor, anchor_h, P, ctx_k, ctx_v, f_rnn, head, fold_w, fold_b)
    hidden = draft.block_hidden(anchor_h, P, ctx_k, ctx_v)
    assert torch.equal(toks, ref_toks), f"draft tokens diverged: {toks.tolist()} vs {ref_toks.tolist()}"
    # the module's bf16 gemm rounds its outputs once more than Maya's F32 pipeline
    assert torch.allclose(hidden, ref_hidden, rtol=2e-2, atol=2e-2)
    base = head(hidden[1:8]).float()
    assert torch.allclose(base, ref_base, rtol=2e-2, atol=2e-2)


def test_dspark_fold_chunks_inside_the_w8a16_decode_window(tmp_path):
    # one R-row head pass dequantizes the whole W8A16 head (a full-vocab fp32
    # transient, OOM on a 24 GiB rank); the fold stays within the kernel's M <= 32
    import safetensors.torch

    from freetoken.models.inkling2.draft import DSparkDraft

    sd = _draft_sd(R=64)
    path = tmp_path / "mtp.safetensors"
    safetensors.torch.save_file(sd, str(path))
    draft = DSparkDraft.load(str(path), torch.device("cpu"), n_vocab=30)
    head_w = (torch.randn(32, 64, generator=torch.Generator().manual_seed(1)) * 0.1).to(torch.bfloat16)
    seen = []

    def head(x):
        seen.append(x.shape[0])
        return x.to(torch.bfloat16).float() @ head_w.float().T

    fold_w, fold_b = draft.fold(head)
    assert seen and max(seen) <= 32
    assert fold_w.shape == (32, 64) and fold_b.shape == (32,)


@pytest.mark.needs_weights
@pytest.mark.skipif(not os.environ.get("FREETOKEN_IK2_DRAFT"), reason="set FREETOKEN_IK2_DRAFT to the official mtp.safetensors (or its folder)")
def test_dspark_official_drafter_matches_reference():
    from freetoken.models.inkling2.draft import DSparkDraft

    path = os.environ["FREETOKEN_IK2_DRAFT"]
    draft = DSparkDraft.load(path, torch.device("cpu"), n_vocab=200058)
    assert draft.C == 2304 and len(draft.layers) == 3 and draft.R == 256
    import safetensors.torch

    sd = safetensors.torch.load_file(path if path.endswith(".safetensors") else os.path.join(path, "mtp.safetensors"))
    t, anchor, anchor_h, P, head_w = _draft_case(sd, 200058, m=24)
    head = _head_fn(head_w)
    ref_toks, ref_hidden, _ = _ref_dspark(sd, t, anchor, anchor_h, P, head, 200058)
    ctx_k, ctx_v = [], []
    for li in range(3):
        K, V = draft.context_rows(draft.mix_taps(t, li), li, torch.arange(P))
        ctx_k.append(K)
        ctx_v.append(V)
    fold_w, fold_b = draft.fold(head)
    toks = draft.propose(anchor, anchor_h, P, ctx_k, ctx_v, draft.f_rnn(t[-1]), head, fold_w, fold_b)
    assert torch.equal(toks, ref_toks), f"official drafter tokens diverged: {toks.tolist()} vs {ref_toks.tolist()}"


def test_dspark_check_source_rejects_non_drafter(tmp_path):
    import safetensors.torch

    from freetoken.models.inkling2.draft import DSparkDraft

    with pytest.raises(ValueError, match="mtp.safetensors"):
        DSparkDraft.check_source(str(tmp_path))
    p = tmp_path / "mtp.safetensors"
    safetensors.torch.save_file({"norm.weight": torch.zeros(4)}, str(p))
    with pytest.raises(ValueError, match="not an Inkling-2 DSpark drafter"):
        DSparkDraft.check_source(str(p))


# ---------------------------------------------------------------- conv verify (speculative)

def test_conv_verify_stash_and_commit(monkeypatch):
    """The verify convs read the live history, leave it untouched, and the commit rolls it
    to the accepted rows -- against the per-request conv oracle, static and dynamic taps."""
    from freetoken.distributed import info
    from freetoken.models.inkling2.conv import CONV_STATE, ConvMeta, Inkling2Conv, Inkling2ConvVerify, short_conv_reference
    from freetoken.models.inkling2.config import parse_config

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    config = parse_config(SimpleNamespace(**_hf_dict()))
    rows, bs = 3, 2
    spec = next(s for s in config.slot_states if s.name == CONV_STATE)
    handler = Inkling2ConvVerify(spec, bs, rows, torch.device("cpu"), torch.float32)
    state = torch.zeros(len(VISITS), bs + 1, CONV_CH, KERNEL - 1, dtype=torch.float32)
    torch.manual_seed(5)
    state.normal_(0, 0.3)
    live_before = state.clone()

    class Pool:
        def slot_state(self, name, layer_id=None):
            return state[layer_id]

        def verify_handler(self, name):
            return handler

    ctx = SimpleNamespace(linear_state_pool=Pool())
    monkeypatch.setattr("freetoken.models.inkling2.conv.get_global_ctx", lambda: ctx)
    slots = torch.tensor([1, 2], dtype=torch.int64)
    meta = ConvMeta(slots=slots, cu_seqlens=torch.arange(bs + 1) * rows, seq_lens=(rows,) * bs,
                    fresh=None, is_decode=False, is_verify=True, rows_per_req=rows)
    torch.manual_seed(6)
    for visit in (0, 3):
        for channels, heads, off in ((KV_DIM, NK, 0), (H, 0, 2 * KV_DIM)):
            conv = Inkling2Conv(config, channels, num_heads=heads)
            conv.weight = (torch.randn(channels, KERNEL) * 0.2).float()
            if heads:
                conv.conv_proj.weight = (torch.randn(heads * KERNEL, H) * 0.1).to(_RDTYPE)
            x = (torch.randn(bs * rows, channels) * 0.5).to(_RDTYPE)
            dyn_input = (torch.randn(bs * rows, H) * 0.5).to(_RDTYPE) if heads else None
            slab = state[visit, :, off : off + channels, :]
            out = conv.forward(x, meta, slab, dyn_input, verify_key=(visit, off))
            dyn = (dyn_input @ conv.conv_proj.weight.T).float() if heads else None
            ref_out, _ = short_conv_reference(x, (rows,) * bs, state[visit, slots, off : off + channels, :],
                                              conv.weight, KERNEL, dyn, heads)
            assert torch.allclose(out, ref_out, rtol=1e-5, atol=1e-5), f"visit {visit} heads {heads}"
            assert torch.equal(state, live_before), "the verify forward wrote the live conv state"
            # commit n accepted rows per request: the oracle's state after n rows
            counts = torch.tensor([1, 3], dtype=torch.int64)
            handler.commit(Pool(), slots, counts)
            for i, n in enumerate(counts.tolist()):
                s0 = int(slots[i])
                _, ref_state = short_conv_reference(x[i * rows : i * rows + n], (n,),
                                                    live_before[visit, s0 : s0 + 1, off : off + channels, :],
                                                    conv.weight, KERNEL,
                                                    dyn[i * rows : i * rows + n] if heads else None, heads)
                assert torch.allclose(state[visit, slots[i], off : off + channels, :], ref_state, rtol=1e-6, atol=1e-6)
            state.copy_(live_before)
            handler.history[visit].zero_()


def test_dspark_ring_write_and_window_read(tmp_path):
    """The proposer's rings: positions key by pos % cap (a long prompt wraps the local rings),
    and a proposal reads exactly the window rows Maya's k_lo bounds select."""
    import safetensors.torch

    from freetoken.models.inkling2.draft import RING, WINDOW, DSparkDraft
    from freetoken.spec.dspark.proposer import CTX_FULL, CTX_LOCAL, FRNN, DSparkProposer

    sd = _draft_sd()
    path = tmp_path / "mtp.safetensors"
    safetensors.torch.save_file(sd, str(path))
    draft = DSparkDraft.load(str(path), torch.device("cpu"), n_vocab=30)
    prop = DSparkProposer.__new__(DSparkProposer)
    prop.draft = draft
    prop.config = SimpleNamespace(max_seq_len=16384)
    cap = 8192
    prop.ctx = SimpleNamespace(slot_states={
        CTX_LOCAL: torch.zeros(2, 3, 2, RING, 256, dtype=torch.float32),
        CTX_FULL: torch.zeros(1, 3, 2, cap, 256, dtype=torch.float16),
        FRNN: torch.zeros(3, 64, dtype=torch.float32),
    })
    torch.manual_seed(11)
    m = 3000  # past the local rings' 2048 rows: the early positions are overwritten
    taps = torch.randn(m, 4, 64)
    prop._write_context(1, taps, 0, m)
    # the write touched slot 1 only: slot 0's rings stay zero (the slot axis, not the K/V axis)
    assert not prop.ctx.slot_states[CTX_LOCAL][:, 0].any()
    assert not prop.ctx.slot_states[CTX_FULL][:, 0].any()
    prop.ctx.slot_states[FRNN][1] = draft.f_rnn(taps[-1])
    assert torch.equal(prop.ctx.slot_states[FRNN][1], draft.f_rnn(taps[-1]))
    ks, vs = prop._read_context(1, m)
    for li in range(3):
        win = WINDOW - 1 if draft.layers[li].local else cap
        pos = torch.arange(max(0, m - win), m)
        K, V = draft.context_rows(draft.mix_taps(taps[pos], li), li, pos)
        ring, _ = prop._rings(li)
        # the gathered window rows must be the fresh context of exactly those positions
        assert torch.allclose(ks[li], K.to(ring.dtype).float(), rtol=1e-3, atol=1e-3)
        assert torch.allclose(vs[li], V.to(ring.dtype).float(), rtol=1e-3, atol=1e-3)
    # a short history reads from position 0, not a negative window
    ks0, _ = prop._read_context(2, 5)
    assert ks0[0].shape[0] == 5
