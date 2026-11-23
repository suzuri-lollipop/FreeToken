"""Inkling-2-Mini GGUF (mainline inkling2_loop layout): config parity with the HF path,
the reader's conv/rel-bias/query transforms, and the gate|up deinterleave of the expert banks.
The synthetic files use the miniature geometry of test_inkling2.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.inkling2.config import parse_config
from freetoken.models.inkling2.gguf import (
    iter_gguf_weights,
    parse_gguf_config,
    to_gguf,
)

from .test_inkling2 import (
    QK_CLIP, REL_CLIP,
    BLOCK, CORE_END, CORE_START, D_REL, E, EXTENT, H, HD, INTER, KERNEL, NH, NK, PHYS,
    RANK, SWA_PHYS, UNPADDED, VOCAB, WINDOW, _hf_dict,
)

KV_DIM = NK * HD
GGML_F32, GGML_F16, GGML_BF16, GGML_Q8_0 = 0, 1, 30, 8


def _metadata() -> dict:
    a = "inkling2_loop"
    return {
        "general.architecture": a,
        f"{a}.embedding_length": H,
        f"{a}.block_count": PHYS,
        f"{a}.context_length": 4096,
        f"{a}.attention.head_count": NH,
        f"{a}.attention.head_count_kv": NK,
        f"{a}.attention.key_length": HD,
        f"{a}.attention.relative_length": D_REL,
        f"{a}.attention.relative_extent": EXTENT,
        f"{a}.attention.sliding_window": WINDOW,
        f"{a}.attention.sliding_window_pattern": [p in SWA_PHYS for p in range(PHYS)],
        f"{a}.attention.qk_clip": float(torch.tensor(22.627416997969522, dtype=torch.float32)),
        f"{a}.attention.relative_clip": 2.0,
        f"{a}.attention.log_scale_floor": 4,
        f"{a}.attention.log_scale_alpha": float(torch.tensor(0.1, dtype=torch.float32)),
        f"{a}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{a}.shortconv.l_cache": KERNEL,
        f"{a}.attn_res.block_size": BLOCK,
        f"{a}.expert_count": E,
        f"{a}.expert_used_count": 2,
        f"{a}.expert_feed_forward_length": INTER,
        f"{a}.expert_weights_scale": 8.0,
        f"{a}.activation.situ_beta": 4.0,
        f"{a}.activation.situ_linear_beta": 25.0,
        f"{a}.final_logit_softcapping": 30.0,
        f"{a}.vocab_size": VOCAB,
        f"{a}.vocab_size_unpadded": UNPADDED,
        f"{a}.loop.start": CORE_START,
        f"{a}.loop.end": CORE_END,
        f"{a}.loop.count": 2,
        f"{a}.loop.gate_rank": RANK,
        f"{a}.loop.varying": False,
    }


def test_gguf_config_matches_the_hf_config():
    from dataclasses import fields

    md = _metadata()
    md["inkling2_loop.attention.qk_clip"] = float(torch.tensor(QK_CLIP, dtype=torch.float32))
    md["inkling2_loop.attention.relative_clip"] = float(torch.tensor(REL_CLIP, dtype=torch.float32))
    shim = GgufConfigShim(
        architectures=["Inkling2LoopGGUFForCausalLM"], model_path="", model_type="inkling2_loop",
        metadata=md, vocab_size=VOCAB, tie_word_embeddings=False,
        hf_fields={"image_token_id": 29},
    )
    got = parse_gguf_config(shim)
    want = parse_config(SimpleNamespace(**_hf_dict()))
    for name in ("num_layers", "num_qo_heads", "num_kv_heads", "head_dim", "hidden_size", "vocab_size",
                 "num_experts", "num_experts_per_tok", "moe_intermediate_size", "attention_groups", "slot_states"):
        assert getattr(got, name) == getattr(want, name), name
    for f in fields(want.inkling2_args):
        g, w = getattr(got.inkling2_args, f.name), getattr(want.inkling2_args, f.name)
        if isinstance(w, float):
            assert g == pytest.approx(w, abs=1e-6), f.name
        else:
            assert g == w, f.name
    assert got.expert_quant == "gguf"
    assert got.moe_layer_count == PHYS


# ----------------------------------------------------------------------------------------------
# minimal GGUF v3 writer (float / bf16 / q8_0 tensors, 32-byte alignment)
# ----------------------------------------------------------------------------------------------

def _write_gguf(path, md: dict, tensors: list[tuple[str, torch.Tensor, int, list[int]]]):
    """tensors: (name, torch-order data, ggml_type, ggml ne list)."""
    out = bytearray()

    def u32(v): out.extend(struct.pack("<I", v))
    def u64(v): out.extend(struct.pack("<Q", v))
    def s(v):
        b = v.encode("utf-8"); u64(len(b)); out.extend(b)

    def value(v):
        if isinstance(v, bool): u32(7); out.extend(struct.pack("<B", int(v)))
        elif isinstance(v, int): u32(4); u32(v)
        elif isinstance(v, float): u32(6); out.extend(struct.pack("<f", v))
        elif isinstance(v, str): u32(8); s(v)
        elif isinstance(v, list):
            u32(9)
            it = 8 if isinstance(v[0], str) else (7 if isinstance(v[0], bool) else (4 if isinstance(v[0], int) else 6))
            u32(it); u64(len(v))
            for x in v:
                if it == 8: s(x)
                elif it == 7: out.extend(struct.pack("<B", int(x)))
                elif it == 4: u32(x)
                else: out.extend(struct.pack("<f", x))
        else: raise TypeError(v)

    out.extend(b"GGUF"); u32(3); u64(len(tensors)); u64(len(md))
    for k, v in md.items():
        s(k); value(v)
    blobs = []
    offset = 0
    for name, data, gtype, ne in tensors:
        s(name); u32(len(ne))
        for d in ne: u64(d)
        u32(gtype); u64(offset)
        raw = data.numpy().tobytes() if gtype not in (GGML_Q8_0, GGML_BF16, GGML_F16) else (
            data if gtype == GGML_Q8_0 else data.view(torch.uint16).numpy().tobytes()
        )
        pad = (-len(raw)) % 32
        blobs.append(raw + b"\0" * pad)
        offset += len(raw) + pad
    pad = (-len(out)) % 32
    out.extend(b"\0" * pad)
    for b in blobs:
        out.extend(b)
    path.write_bytes(bytes(out))


def _q8_0(w: torch.Tensor) -> tuple[bytes, list[int]]:
    """torch [out, in] -> packed q8_0 rows + ggml ne [in, out]."""
    out_dim, in_dim = w.shape
    assert in_dim % 32 == 0
    rows = bytearray()
    for r in range(out_dim):
        for b in range(in_dim // 32):
            block = w[r, b * 32 : (b + 1) * 32].float()
            amax = block.abs().max().item()
            scale = amax / 127.0 if amax > 0 else 1.0
            qs = torch.clamp(torch.round(block / scale), -128, 127).to(torch.int8)
            rows.extend(struct.pack("<e", scale))
            rows.extend(qs.numpy().tobytes())
    return bytes(rows), [in_dim, out_dim]


def _bf16(name, w):
    return (name, w.to(torch.bfloat16).contiguous(), GGML_BF16, list(reversed(w.shape)))


def _f32(name, w):
    return (name, w.float().contiguous(), GGML_F32, list(reversed(w.shape)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ggml linear segments load on the device")
def test_gguf_reader_transforms(tmp_path):
    torch.manual_seed(2)
    md = _metadata()
    md["tokenizer.ggml.tokens"] = ["<|endoftext|>"] * UNPADDED + ["<|unused_200054|>"] + ["x"] * (VOCAB - UNPADDED - 1)
    tensors = [
        _bf16("token_embd.weight", torch.randn(VOCAB, H)),
        _bf16("output.weight", torch.randn(VOCAB, H)),
        _f32("token_embd_norm.weight", torch.rand(H) + 0.5),
        _f32("output_norm.weight", torch.rand(H) + 0.5),
        _f32("output_res_query.weight", torch.randn(H)),
    ]
    stored = {}
    for p in range(PHYS):
        blk = f"blk.{p}"
        swa = p in SWA_PHYS
        for leaf, rows in (("attn_q", NH * HD), ("attn_k", KV_DIM), ("attn_v", KV_DIM), ("attn_rel", NH * D_REL), ("attn_gate", NH * HD), ("attn_output", H)):
            w = torch.randn(rows, H) * 0.1
            stored[f"{blk}.{leaf}"] = w
            if p == 0 and leaf == "attn_q":
                raw, ne = _q8_0(w)
                tensors.append((f"{blk}.{leaf}.weight", raw, GGML_Q8_0, ne))
            else:
                tensors.append(_bf16(f"{blk}.{leaf}.weight", w))
        tensors.append(_bf16(f"{blk}.attn_sink_proj.weight", torch.randn(NH, H)))
        tensors.append(_f32(f"{blk}.attn_q_clip.weight", torch.randn(NH, HD)))
        tensors.append(_f32(f"{blk}.attn_rel_clip.weight", torch.randn(NH, D_REL)))
        # the converter stores the bias table transposed: torch [extent, d_rel]
        rel = torch.randn(EXTENT if not swa else WINDOW, D_REL)
        stored[f"{blk}.rel"] = rel
        tensors.append(_f32(f"{blk}.attn_rel_bias.weight", rel))
        for cname, channels in (("attn_k_conv", KV_DIM), ("attn_v_conv", KV_DIM), ("attn_post_conv", H), ("ffn_post_conv", H)):
            # conv1d order [C, K] in torch; ggml ne [K, C]
            w = torch.randn(channels, KERNEL)
            stored[f"{blk}.{cname}"] = w
            tensors.append(_f32(f"{blk}.{cname}.weight", w))
        if swa:
            tensors.append(_f32(f"{blk}.attn_norm.weight", torch.rand(H) + 0.5))
        else:
            for qkv in ("q", "k", "v"):
                tensors.append(_f32(f"{blk}.attn_{qkv}_inp_norm.weight", torch.rand(H) + 0.5))
            tensors.append(_bf16(f"{blk}.attn_k_conv_proj.weight", torch.randn(NK * KERNEL, H)))
            tensors.append(_bf16(f"{blk}.attn_v_conv_proj.weight", torch.randn(NK * KERNEL, H)))
        tensors.append(_f32(f"{blk}.ffn_norm.weight", torch.rand(H) + 0.5))
        tensors.append(_f32(f"{blk}.ffn_gate_inp.weight", torch.randn(E, H)))
        tensors.append((f"{blk}.exp_probs_b.bias", torch.randn(E).float().contiguous(), GGML_F32, [E]))
        if p > 0:
            q = torch.randn(3, H) if not swa else torch.randn(H)
            stored[f"{blk}.attn_res_query"] = q
            tensors.append(_f32(f"{blk}.attn_res_query.weight", q))
        tensors.append(_f32(f"{blk}.ffn_res_query.weight", torch.randn(H)))
        if CORE_START <= p < CORE_END:
            tensors.append(_f32(f"{blk}.loop_msg_query.weight", torch.randn(4, H)))
            for kind in ("q", "k", "v", "ffn"):
                tensors.append(_f32(f"{blk}.loop_{kind}_ctrl_down.weight", torch.randn(RANK, 2 * H)))
                tensors.append(_f32(f"{blk}.loop_{kind}_ctrl_gain.weight", torch.randn(H, RANK)))
                tensors.append(_f32(f"{blk}.loop_{kind}_ctrl_forget.weight", torch.randn(H, RANK)))
        tensors.append(_bf16(f"{blk}.ffn_gate_up_exps.weight", torch.randn(E, 2 * INTER, H)))
        tensors.append(_bf16(f"{blk}.ffn_down_exps.weight", torch.randn(E, H, INTER)))
    path = tmp_path / "inkling2_loop-test.gguf"
    _write_gguf(path, md, tensors)

    got = dict(iter_gguf_weights(str(path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True, include_vision=False))
    # conv weights: flipped into lag order, fp32
    assert torch.equal(got["model.layers.0.self_attn.k_conv.weight"], stored["blk.0.attn_k_conv"].flip(-1))
    assert torch.equal(got["model.layers.0.ffn_post_conv.weight"], stored["blk.0.ffn_post_conv"].flip(-1))
    # rel bias: transposed back to [d_rel, extent]
    assert torch.equal(got["model.layers.0.self_attn.rel_bias"], stored["blk.0.rel"].transpose(0, 1))
    # queries: SWA [1, H], full [3, H]
    assert got["model.layers.2.attn_res_query"].shape == (1, H)
    assert got["model.layers.5.attn_res_query"].shape == (3, H)
    assert got["model.output_res_query"].shape == (1, H)
    # router fp32 + bias
    assert got["model.layers.0.mlp.gate.weight"].dtype is torch.float32
    assert got["model.layers.0.mlp.gate_bias"].shape == (E,)
    # q8_0 linear lands as packed ggml segments; bf16 ones as a plain weight
    assert "model.layers.0.self_attn.q_proj.weight_0" in got or "model.layers.0.self_attn.q_proj.weight" in got
    assert got["model.layers.0.self_attn.sink_proj.weight"].dtype is torch.bfloat16
    # loop controllers and message queries
    assert got["model.layers.2.controllers.0.down_weight"].shape == (RANK, 2 * H)
    assert got["model.layers.2.loop_msg_query"].shape == (4, H)
    # experts stay out of the dense pass
    assert not any(".experts." in k for k in got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="resident expert banks")
def test_gguf_expert_banks_read_gate_up_halves_as_is(tmp_path):
    torch.manual_seed(3)
    md = _metadata()
    gate_up = torch.randn(E, 2 * INTER, H).to(torch.bfloat16)
    down = torch.randn(E, H, INTER).to(torch.bfloat16)
    tensors = [
        ("blk.0.ffn_gate_up_exps.weight", gate_up.contiguous(), GGML_BF16, [H, 2 * INTER, E]),
        ("blk.0.ffn_down_exps.weight", down.contiguous(), GGML_BF16, [INTER, H, E]),
    ]
    path = tmp_path / "inkling2_loop-exps.gguf"
    _write_gguf(path, md, tensors)
    from freetoken.models.gguf.experts import load_gguf_expert_banks

    banks = load_gguf_expert_banks(str(path), 1, device=torch.device("cuda"))
    gu = banks.sources["gate_up"][0]
    # the mainline converter already deinterleaved the checkpoint rows: the bank is the file as-is
    assert torch.equal(gu, gate_up.cuda())
    assert torch.equal(banks.sources["down"][0], down.cuda())


def test_to_gguf_maps_the_module_tree():
    assert to_gguf("model.embed_tokens", {}) == ("token_embd",)
    assert to_gguf("lm_head", {}) == ("output",)
    assert to_gguf("model.layers.7.self_attn.sink_proj", {}) == ("blk.7.attn_sink_proj",)
    assert to_gguf("model.layers.7.mlp.experts.0.gate_proj", {}) == ("blk.7.ffn_gate_up_exps",)
    assert to_gguf("model.layers.7.mlp.experts.0.up_proj", {}) == ("blk.7.ffn_gate_up_exps",)
    assert to_gguf("model.layers.7.mlp.experts.0.down_proj", {}) == ("blk.7.ffn_down_exps",)
    assert to_gguf("model.layers.7.self_attn.k_conv.weight", {}) == ()
    assert to_gguf("model.norm", {}) == ()


def test_mmproj_patch_embedding_permutes_back_to_linear_rows(tmp_path):
    from freetoken.models.gguf.mmproj import Mmproj
    from freetoken.models.inkling2.gguf import iter_mmproj_weights

    # the converter writes the linear weight [out, (kh,kw,c)] as [out, c, kh, kw]
    out, kh, kw, c = 8, 4, 4, 3
    linear_rows = torch.arange(out * kh * kw * c, dtype=torch.float32).reshape(out, kh * kw * c)
    stored = linear_rows.reshape(out, kh, kw, c).permute(0, 3, 1, 2).contiguous()
    tensors = [
        ("v.patch_embd.weight", stored, GGML_F32, [kw, kh, c, out]),
        ("v.class_embd", torch.zeros(out), GGML_F32, [out]),
        ("v.class_position_embd.weight", torch.zeros(out), GGML_F32, [out]),
        ("v.position_embd.weight", torch.zeros(16, out), GGML_F32, [out, 16]),
        ("v.pre_ln.weight", torch.ones(out), GGML_F32, [out]),
        ("v.pre_ln.bias", torch.zeros(out), GGML_F32, [out]),
        ("mm.1.weight", torch.zeros(8, out * 4), GGML_F32, [out * 4, 8]),
        ("mm.2.weight", torch.zeros(8, 8), GGML_F32, [8, 8]),
    ]
    for part in ("weight", "bias"):
        for src in ("attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down", "ln1", "ln2"):
            rows = 8 if src in ("attn_q", "attn_k", "attn_v", "attn_out", "ln1", "ln2") else 8
            tensors.append((f"v.blk.0.{src}.{part}", torch.zeros(rows, 8), GGML_F32, [8, rows]))
    for ls in ("ls1", "ls2"):
        tensors.append((f"v.blk.0.{ls}.weight", torch.ones(out), GGML_F32, [out]))
    path = tmp_path / "mmproj-test.gguf"
    _write_gguf(path, {"general.architecture": "clip", "general.type": "mmproj"}, tensors)
    got = dict(iter_mmproj_weights(Mmproj(path=str(path), source="test", projector="inkling2v", weight_type="F32")))
    assert torch.equal(got["visual.backbone.embeddings.patch_embedding.weight"], linear_rows)
