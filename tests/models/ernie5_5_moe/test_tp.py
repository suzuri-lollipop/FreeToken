"""TP>=2 expert-bank sharding (the MXFP4 storage step): the standard OCP MXFP4
kernel's layout/pack must bank this rank's slice of the expert intermediate
(gate|up rows, down packed columns, e8m0 scale groups), TP1 byte-identical.

The kernel's TP apply (per-rank down partials + all-reduce) and the dense /
attention TP are separate steps; this guards the bank slicing only, so a wrong
slice axis or scale grain fails here rather than silently banking wrong rows.
"""

import pytest
import torch

from .common import make_checkpoint, make_dequantized_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _full_pieces(tmp_path):
    from freetoken.distributed.info import tp_view
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.ernie5_5_moe import iter_expert_pieces, parse_config
    from freetoken.utils import cached_load_hf_config

    hf = make_checkpoint(str(tmp_path / "hf"), quantized=True, seed=7)
    cfg = parse_config(cached_load_hf_config(hf))
    with tp_view(0, 1):
        pieces = list(iter_expert_pieces(hf, cfg, QuantKind.MXFP4))
    return pieces[0][3]  # one expert's {gate,up,down,*_scale}


@pytest.mark.parametrize("rank", [0, 1])
def test_mxfp4_pack_banks_local_intermediate_slice(tmp_path, rank):
    from freetoken.layers.quantization.moe.base import MoEConfig, fused_piece
    from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

    pieces = _full_pieces(tmp_path)
    e, inter, h_half = pieces["gate"].shape          # [E, I, H/2]
    hidden = h_half * 2
    cfg = MoEConfig(num_experts=e, hidden=hidden, intermediate=inter, top_k=10,
                    tp_rank=rank, tp_size=2, activation="situ", alpha=4.0, beta=0.0,
                    limit=25.0, act_block=128)
    kernel = TritonMxfp4MoEKernel()
    layout = kernel.layout(cfg)
    out = {role: torch.empty((e, *spec.shape), dtype=spec.dtype) for role, spec in layout.items()}
    kernel.pack(pieces, cfg, out)

    lo, hi = cfg.local_intermediate_range
    assert hi - lo == inter // 2
    # gate|up: the fused [E, 2I, H/2] rows [lo:hi] and [I+lo:I+hi]
    gu_full = fused_piece(pieces, "gate_up").view(torch.uint8)
    gu_ref = torch.cat([gu_full[:, lo:hi], gu_full[:, inter + lo:inter + hi]], dim=1)
    assert torch.equal(out["gate_up"], gu_ref)
    gus_full = fused_piece(pieces, "gate_up_scale").view(torch.float8_e8m0fnu)
    gus_ref = torch.cat([gus_full[:, lo:hi], gus_full[:, inter + lo:inter + hi]], dim=1)
    assert torch.equal(out["gate_up_scale"].view(torch.float8_e8m0fnu), gus_ref)
    # down: packed columns [lo/2:hi/2]; e8m0 scale columns [lo/32:hi/32]
    down_full = pieces["down"].view(torch.uint8)
    assert torch.equal(out["down"], down_full[:, :, lo // 2:hi // 2])
    downs_full = pieces["down_scale"].view(torch.float8_e8m0fnu)
    assert torch.equal(out["down_scale"].view(torch.float8_e8m0fnu), downs_full[:, :, lo // 32:hi // 32])


def _pack_banks(pieces, rank, size):
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

    e, inter, h_half = pieces["gate"].shape
    cfg = MoEConfig(num_experts=e, hidden=h_half * 2, intermediate=inter, top_k=10,
                    tp_rank=rank, tp_size=size, activation="situ", alpha=4.0, beta=0.0,
                    limit=25.0, act_block=128)
    kernel = TritonMxfp4MoEKernel()
    layout = kernel.layout(cfg)
    out = {role: torch.empty((e, *spec.shape), dtype=spec.dtype) for role, spec in layout.items()}
    kernel.pack(pieces, cfg, out)
    return out


def test_mxfp4_tp2_down_partials_sum_to_full(tmp_path):
    """The expert TP math: each rank's down GEMM sums only its intermediate slice,
    so rank0_partial + rank1_partial reproduces the TP1 full expert output (the MoE
    layer's all-reduce is exactly this sum). The FP8 activation round-trip blocks
    align with the even split, so the partials match to bf16 accumulation."""
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill

    pieces = _full_pieces(tmp_path)
    e, inter, h_half = pieces["gate"].shape
    hidden = h_half * 2
    dev = torch.device("cuda")
    full = _pack_banks(pieces, 0, 1)
    r0 = _pack_banks(pieces, 0, 2)
    r1 = _pack_banks(pieces, 1, 2)

    # the grouped-GEMM prefill path (>= _GROUPED_MIN_ROUTES routes) has no decode
    # K//2 % 128 floor, so the tiny geometry's TP2 local slice runs here
    g = torch.Generator(device="cpu").manual_seed(11)
    x = (torch.randn(768, hidden, generator=g) * 0.3).to(dev, torch.bfloat16)
    slots = torch.zeros(768, 1, dtype=torch.int32, device=dev)
    tw = torch.ones(768, 1, dtype=torch.float32, device=dev)

    def run(b):
        return routed_experts_fp4_prefill(
            x, slots, tw,
            b["gate_up"].to(dev), b["gate_up_scale"].to(dev),
            b["down"].to(dev), b["down_scale"].to(dev),
            25.0, num_rows=e, act_block=128, situ_alpha=4.0,
        )

    full_out = run(full)
    part_sum = run(r0) + run(r1)
    assert torch.isfinite(part_sum).all()
    assert torch.allclose(full_out, part_sum, atol=2e-2, rtol=2e-2), (full_out - part_sum).abs().max()


@pytest.mark.parametrize("rank", [0, 1])
def test_dense_mlp_reader_shards_at_tp2(tmp_path, rank):
    """The dense reader emits this rank's MLP slice (gate/up rows, down columns) at TP2;
    attention / latent projections stay replicated. The model's parallel Linear buffers
    are local, so the reader slice must match exactly."""
    from freetoken.distributed.info import tp_view
    from freetoken.models.ernie5_5_moe import iter_weights

    hf = make_dequantized_checkpoint(str(tmp_path / "hf"))
    dev = torch.device("cpu")
    with tp_view(0, 1):
        full = dict(iter_weights(hf, dev, include_moe_experts=False))
    with tp_view(rank, 2):
        sharded = dict(iter_weights(hf, dev, include_moe_experts=False))

    row_leaves = (".ffn.gate_proj.weight", ".ffn.up_proj.weight",
                  ".ffn.shared_experts.gate_proj.weight", ".ffn.shared_experts.up_proj.weight")
    col_leaves = (".ffn.down_proj.weight", ".ffn.shared_experts.down_proj.weight")
    vocab_leaves = ("model.embed.weight", "model.head.weight")
    saw_row = saw_col = saw_vocab = False
    for name, ft in full.items():
        st = sharded[name]
        if name.endswith(row_leaves):
            local = ft.shape[0] // 2
            assert torch.equal(st, ft[rank * local:(rank + 1) * local]), name
            saw_row = True
        elif name.endswith(col_leaves):
            local = ft.shape[1] // 2
            assert torch.equal(st, ft[:, rank * local:(rank + 1) * local]), name
            saw_col = True
        elif name in vocab_leaves:
            local = -(-ft.shape[0] // 2)   # vocab shard rounds up
            assert torch.equal(st, ft[rank * local:(rank + 1) * local]), name
            saw_vocab = True
        else:
            assert torch.equal(st, ft), f"{name} should stay replicated"
    assert saw_row and saw_col and saw_vocab


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_model_builds_and_loads_sharded_mlp(tmp_path, rank):
    """At TP2 the model's MLP buffers are local (col/row parallel) and the sharded
    dense reader fills them exactly; attention / latent projections stay full.
    This is the dense/expert TP integration at model-build time (the full forward
    additionally needs the cross-rank all-reduce on a 2-rank rig)."""
    from dataclasses import replace

    from freetoken.distributed.info import tp_view
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict
    from freetoken.layers import set_rope_device
    from freetoken.layers.embedding import VocabParallelEmbedding, set_w8a16_embed_default
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.models import create_model
    from freetoken.models.ernie5_5_moe import iter_weights, parse_config
    from freetoken.utils.hf import cached_load_hf_config
    from freetoken.utils.torch_utils import torch_dtype

    hf = make_dequantized_checkpoint(str(tmp_path / "hf"))
    dev = torch.device("cuda")
    set_rope_device(dev)
    mc = parse_config(cached_load_hf_config(hf))
    mc = replace(mc, moe_strategy="fused", decode_target="gpu", quant=NoQuantConfig())
    with tp_view(rank, 2):
        ambient = VocabParallelEmbedding._W8A16_EMBED_DEFAULT
        set_w8a16_embed_default(False)
        try:
            with torch.device("meta"), torch_dtype(torch.bfloat16):
                model = create_model(mc)
        finally:
            set_w8a16_embed_default(ambient)
        # dense layer-0 MLP gate buffer is this rank's half; attention stays full
        ffn0 = model.model.layers.op_list[0].ffn
        assert ffn0.gate_proj.local_output_size == ffn0.gate_proj.full_output_size // 2
        assert ffn0.down_proj.local_input_size == ffn0.down_proj.full_input_size // 2
        state = _materialize_loaded_weight_state_dict(
            model.state_dict(), iter_weights(hf, dev, include_moe_experts=False), device=dev,
        )
        model.load_state_dict(state)
        for name, p in model.state_dict().items():
            assert torch.isfinite(p).all(), name


def _ernie_tp_forward(rank, world, port, hf, want, tokens):
    """One rank of a real gloo group: build the TP model, load this rank's shards,
    prefill, and diff the TP1 reference logits. The collectives are the production
    TorchDistributedImpl, so a misplaced all-reduce / wrong shard shows up as a
    number that disagrees with the single-process forward."""
    from datetime import timedelta

    import torch.distributed as dist

    from freetoken.distributed import info
    from .harness import TinyEngine

    torch.cuda.set_device(rank)
    # NCCL: the forward is CUDA (the lm_head all-gather / embedding all-reduce need a
    # CUDA-capable collective; gloo is CPU-only and garbers device tensors)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        info._TP_INFO = info.DistributedInfo(rank=rank, size=world)
        eng = TinyEngine(hf, max_seq_len=1024, max_running_req=2, quantized=True)
        req = eng.new_request(0, tokens, max_new=8)
        got = eng.prefill([req]).cpu()
        want_dev = want.to(got.device)
        assert torch.isfinite(got).all(), f"rank{rank} got NaN/inf"
        # The vocab all-reduce and the expert/MLP partial-sum reductions accumulate in
        # bf16, and the mHC Sinkhorn (20 iters, alpha_res 5.3) amplifies that rounding;
        # the logits agree to a few percent relative (expert partials alone match 3e-5).
        denom = want_dev.abs().clamp_min(1.0)
        rel = ((got - want_dev).abs() / denom).max().item()
        assert rel < 0.05, f"rank{rank} rel_err={rel} maxabs={(got - want_dev).abs().max().item()} want_max={want_dev.abs().max().item()}"
    finally:
        dist.destroy_process_group()


def test_tp2_forward_matches_tp1(tmp_path):
    """End-to-end TP=2 on the 2-GPU rig: replicated attention/KV + sharded MLPs and
    MXFP4 experts, production all-reduce, reproduces the TP1 logits."""
    import socket

    import torch.multiprocessing as mp

    from .harness import TinyEngine

    # intermediate 512: the TP2 local slice (256) satisfies the fp8 GEMM / decode GEMV
    # block floors (the tiny 256 geometry's TP2 slice of 128 is below them)
    hf = make_checkpoint(str(tmp_path / "hf"), quantized=True, seed=3,
                         geo_overrides={"moe_intermediate_size": 512, "intermediate_size": 512})
    tokens = [5, 17, 3, 42, 8, 61, 2, 19] * 4
    ref = TinyEngine(hf, max_seq_len=1024, max_running_req=2, quantized=True)
    want = ref.prefill([ref.new_request(0, tokens, max_new=8)]).cpu()
    del ref
    torch.cuda.empty_cache()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_ernie_tp_forward, args=(2, port, hf, want, tokens), nprocs=2, join=True)


def test_tp2_config_gates_pass(tmp_path):
    """Config-time TP gates (CPU, no CUDA context): the geometry shards/replicates
    over 2 ranks (query heads split, the single MQA KV head replicates, the expert
    intermediate splits on its scale group) and the backend is TP-capable."""
    from freetoken.attention import attention_backend_info
    from freetoken.engine.config import tp_shard_error
    from freetoken.models.ernie5_5_moe import parse_config
    from freetoken.models.register import get_model_spec
    from freetoken.utils import cached_load_hf_config

    hf = make_checkpoint(str(tmp_path / "hf"), quantized=True, seed=1)
    mc = parse_config(cached_load_hf_config(hf))
    assert tp_shard_error(mc, 2) is None, tp_shard_error(mc, 2)
    assert attention_backend_info("ernie55_sparse").supports_tp
    assert get_model_spec("Ernie5_5_MoeForCausalLM").tp_supported
    assert get_model_spec("Ernie5_5_MoeGGUFForCausalLM").tp_supported


def test_mxfp4_pack_tp1_is_full_width(tmp_path):
    from freetoken.layers.quantization.moe.base import MoEConfig, fused_piece
    from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

    pieces = _full_pieces(tmp_path)
    e, inter, h_half = pieces["gate"].shape
    cfg = MoEConfig(num_experts=e, hidden=h_half * 2, intermediate=inter, top_k=10,
                    tp_rank=0, tp_size=1, activation="situ", alpha=4.0, beta=0.0,
                    limit=25.0, act_block=128)
    kernel = TritonMxfp4MoEKernel()
    layout = kernel.layout(cfg)
    out = {role: torch.empty((e, *spec.shape), dtype=spec.dtype) for role, spec in layout.items()}
    kernel.pack(pieces, cfg, out)
    assert torch.equal(out["gate_up"], fused_piece(pieces, "gate_up").view(torch.uint8))
    assert torch.equal(out["down"], pieces["down"].view(torch.uint8))
