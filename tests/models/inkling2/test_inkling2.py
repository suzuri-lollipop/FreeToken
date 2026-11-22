"""Inkling-2-Mini against an fp32 transcription of the looped decoder (llama.cpp's
inkling2_loop graph + the transformers inkling2 per-layer math), on a miniature config
with the same structure: 8 physical layers, block 2, core [2, 6) run twice, SWA/full
alternation, dynamic K/V convs on the full layers, cross-pass controllers, SITU MoE.

The attention stub implements the rel-bias + per-query-sink semantics eagerly; the
triton kernels that serve them in production are diffed in tests/kernels.
"""

from __future__ import annotations

import math
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
