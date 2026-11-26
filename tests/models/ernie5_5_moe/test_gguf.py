"""Strata-convention GGUF (HF-name-preserving, no llama.cpp converter transforms).

The dense reader must yield the SAME model buffers as the HF path: the GGUF keeps
the checkpoint's HF tensor names and the HF config as a JSON metadata blob, so
there are no converter transforms to undo. This builds a tiny dequantized HF
checkpoint, repacks it into a Strata-style GGUF (Strata's own gguf_writer, which
handles the types gguf-py cannot), and checks config geometry + dense round-trip.
"""

import json
import os
import sys

import pytest
import torch

from .common import make_checkpoint, make_dequantized_checkpoint

# Strata's own GGUF writer is the reference (no llama.cpp / gguf-py): it lives in
# the Strata checkout beside this fork. Skip when it is not present.
_FORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_STRATA_TOOLS = os.path.join(os.path.dirname(os.path.dirname(_FORK_ROOT)), "strata", "terminal", "tools")
pytestmark = pytest.mark.skipif(not os.path.isdir(_STRATA_TOOLS), reason="Strata gguf_writer not present")


def _write_strata_gguf(hf_folder: str, gguf_path: str) -> None:
    sys.path.insert(0, _STRATA_TOOLS)
    try:
        import gguf_writer
        from gguf_writer import GGUFWriter, _Tensor
    finally:
        sys.path.pop(0)
    from safetensors.torch import load_file

    # Strata's writer ships F32/F16/Q2_0; the raw MXFP4 packed rows + e8m0 scales
    # are I8 (24), added here so the HF-format expert bytes travel unchanged.
    gguf_writer.TYPE_IDS.setdefault("I8", 24)

    cfg = json.load(open(os.path.join(hf_folder, "config.json")))
    tensors = {}
    for f in os.listdir(hf_folder):
        if f.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(hf_folder, f)))

    w = GGUFWriter()
    w.add("general.architecture", "ernie5_5_moe")
    w.add("ernie5_5_moe.hf_config", json.dumps(cfg))
    # _vocab_size falls back to the tokenizer list when there is no llama.cpp
    # token_embd.weight; the HF-name GGUF carries the vocab this way.
    w.add("tokenizer.ggml.tokens", [f"<t{i}>" for i in range(int(cfg["vocab_size"]))])
    for name, t in tensors.items():
        # float_tensor reverses the ggml dims back to torch order: store the GGUF
        # shape reversed, with the tensor's own C-order bytes.
        if t.dtype == torch.int8:
            a = t.numpy()
            w.tensors.append(_Tensor(name, list(reversed(a.shape)), "I8", a.tobytes()))
        elif t.dtype == torch.float8_e8m0fnu:
            a = t.view(torch.uint8).numpy()   # preserve the e8m0 scale byte
            w.tensors.append(_Tensor(name, list(reversed(a.shape)), "I8", a.tobytes()))
        else:
            a = t.float().numpy()
            w.add_f32(name, a.ravel().reshape(list(reversed(a.shape))))
    w.write(gguf_path)


def test_gguf_config_matches_hf(tmp_path):
    from freetoken.models.ernie5_5_moe import parse_config, parse_gguf_config
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.utils import cached_load_hf_config

    hf = make_dequantized_checkpoint(str(tmp_path / "hf"))
    gguf = str(tmp_path / "ernie.gguf")
    _write_strata_gguf(hf, gguf)

    hf_cfg = parse_config(cached_load_hf_config(hf))
    gg_cfg = parse_gguf_config(build_gguf_shim(gguf))
    assert gg_cfg.model_type == "ernie5_5_moe"
    assert gg_cfg.num_layers == hf_cfg.num_layers
    assert gg_cfg.hidden_size == hf_cfg.hidden_size
    assert gg_cfg.num_experts == hf_cfg.num_experts
    assert gg_cfg.dsv4_args.window_size == hf_cfg.dsv4_args.window_size
    assert tuple(gg_cfg.dsv4_args.layer_compress_ratios) == tuple(hf_cfg.dsv4_args.layer_compress_ratios)


def test_gguf_dense_weights_round_trip(tmp_path):
    from freetoken.models.ernie5_5_moe import iter_gguf_weights, iter_weights

    hf = make_dequantized_checkpoint(str(tmp_path / "hf"))
    gguf = str(tmp_path / "ernie.gguf")
    _write_strata_gguf(hf, gguf)

    dev = torch.device("cpu")
    hf_w = dict(iter_weights(hf, dev, include_moe_experts=False))
    gg_w = dict(iter_gguf_weights(gguf, dev, include_moe_experts=False))
    assert set(gg_w) == set(hf_w), (set(hf_w) - set(gg_w), set(gg_w) - set(hf_w))
    for name in hf_w:
        a, b = hf_w[name].float(), gg_w[name].float()
        assert a.shape == b.shape, name
        assert torch.equal(a, b), f"{name} diverged (max {(a - b).abs().max()})"


def test_gguf_expert_pieces_round_trip(tmp_path):
    """The offload expert pieces (BF16 dequantized artifact) read from the HF-name
    GGUF must match the HF path piece-for-piece: same bank layer / expert range /
    role tensors, so the offload banks fill identically."""
    from freetoken.distributed.info import tp_view
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.ernie5_5_moe import iter_expert_pieces, parse_config
    from freetoken.utils import cached_load_hf_config

    hf = make_dequantized_checkpoint(str(tmp_path / "hf"))
    gguf = str(tmp_path / "ernie.gguf")
    _write_strata_gguf(hf, gguf)
    cfg = parse_config(cached_load_hf_config(hf))

    with tp_view(0, 1):
        hf_pieces = list(iter_expert_pieces(hf, cfg, QuantKind.NONE))
        gg_pieces = list(iter_expert_pieces(gguf, cfg, QuantKind.NONE))
    assert hf_pieces and len(hf_pieces) == len(gg_pieces)
    for (bl_h, e0_h, e1_h, ph), (bl_g, e0_g, e1_g, pg) in zip(hf_pieces, gg_pieces):
        assert (bl_h, e0_h, e1_h) == (bl_g, e0_g, e1_g)
        assert set(ph) == set(pg)
        for role in ph:
            a, b = ph[role].float(), pg[role].float()
            assert a.shape == b.shape, (bl_h, e0_h, role)
            assert torch.equal(a, b), f"expert piece {(bl_h, e0_h, role)} diverged"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguf_forward_matches_hf(tmp_path):
    """Full serving path on the Strata-convention GGUF: dense + resident BF16
    experts load through the GGUF readers and prefill/decode reproduces the HF
    dequantized model's logits (the weights round-trip bit-exact, so only BF16
    accumulation order may differ)."""
    from .harness import TinyEngine

    hf = make_dequantized_checkpoint(str(tmp_path / "hf"))
    gguf = str(tmp_path / "ernie.gguf")
    _write_strata_gguf(hf, gguf)
    tokens = [5, 17, 3, 42, 8, 61, 2, 19] * 4

    hf_eng = TinyEngine(hf, max_seq_len=1024, max_running_req=2)
    gg_eng = TinyEngine(gguf, max_seq_len=1024, max_running_req=2, gguf=True)

    hf_req = hf_eng.new_request(0, tokens, max_new=8)
    gg_req = gg_eng.new_request(0, tokens, max_new=8)
    hf_logits = hf_eng.prefill([hf_req])
    gg_logits = gg_eng.prefill([gg_req])
    assert torch.isfinite(gg_logits).all()
    assert torch.allclose(hf_logits, gg_logits, atol=2e-2, rtol=2e-2), (hf_logits - gg_logits).abs().max()

    hf_eng.finish_prefill([hf_req])
    gg_eng.finish_prefill([gg_req])
    prev_h = hf_logits[0].argmax().item()
    prev_g = gg_logits[0].argmax().item()
    for _ in range(4):
        lh = hf_eng.decode([hf_req], [prev_h])
        lg = gg_eng.decode([gg_req], [prev_g])
        assert torch.allclose(lh, lg, atol=2e-2, rtol=2e-2), (lh - lg).abs().max()
        prev_h, prev_g = lh[0].argmax().item(), lg[0].argmax().item()


def test_gguf_mxfp4_expert_pieces_round_trip(tmp_path):
    """The compressed path: MXFP4 packed rows + e8m0 scales travel as raw I8 under
    HF names, and the offload pieces match the HF quantized checkpoint byte-for-byte
    (the ds_fp4 kernel reads the same layout it reads from safetensors)."""
    from freetoken.distributed.info import tp_view
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.ernie5_5_moe import iter_expert_pieces, parse_config
    from freetoken.utils import cached_load_hf_config

    hf = make_checkpoint(str(tmp_path / "hf"), quantized=True, seed=5)
    gguf = str(tmp_path / "ernie.gguf")
    _write_strata_gguf(hf, gguf)
    cfg = parse_config(cached_load_hf_config(hf))

    with tp_view(0, 1):
        hf_pieces = list(iter_expert_pieces(hf, cfg, QuantKind.MXFP4))
        gg_pieces = list(iter_expert_pieces(gguf, cfg, QuantKind.MXFP4))
    assert hf_pieces and len(hf_pieces) == len(gg_pieces)
    for (bl_h, e0_h, e1_h, ph), (bl_g, e0_g, e1_g, pg) in zip(hf_pieces, gg_pieces):
        assert (bl_h, e0_h, e1_h) == (bl_g, e0_g, e1_g)
        assert set(ph) == set(pg)
        for role in ph:
            assert ph[role].dtype == pg[role].dtype, (bl_h, e0_h, role)
            assert torch.equal(ph[role], pg[role]), f"MXFP4 piece {(bl_h, e0_h, role)} diverged"
