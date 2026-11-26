"""The bundled DSpark drafter (model.mtp.*): load + a proposal on the real GPU.

The drafter is the weights-and-math port of the inkling2 DSpark head; the proposer owns
the per-request context state. These tests pin the load geometry and that a proposal runs
end to end on device and stays in the vocab (the target verifies every drafted token, so
the contract here is a well-formed proposal, not a specific answer).
"""

import json
import subprocess
import sys

import pytest
import safetensors.torch
import torch

from freetoken.models.ernie5_5_moe import parse_config
from freetoken.models.ernie5_5_moe.draft import Ernie55DSparkDraft
from freetoken.utils.hf import cached_load_hf_config

from .common import make_checkpoint


class _DictReader:
    def __init__(self, sd):
        self._sd = sd

    def get(self, key):
        return self._sd[key]


def _load(tmp_path, **geo):
    hf = make_checkpoint(str(tmp_path / "hf"), quantized=False, seed=3,
                         geo_overrides={"moe_intermediate_size": 512, "intermediate_size": 512, **geo})
    sd = safetensors.torch.load_file(hf + "/model-00001-of-00001.safetensors")
    args = parse_config(cached_load_hf_config(hf)).dsv4_args
    return Ernie55DSparkDraft.load(_DictReader(sd), args, n_vocab=args.vocab_size), args


def test_draft_load_geometry(tmp_path):
    draft, args = _load(tmp_path)
    assert draft.C == args.dim
    assert draft.BLOCK == args.dspark_block_size + 1
    assert len(draft.layers) == args.dspark_num_layers
    assert draft.N_TAPS == len(args.dspark_target_layer_ids) + 1
    assert draft.R == draft.w1.shape[1]
    assert draft.rnn_w.shape == (3 * draft.R, 2 * draft.R + 2 * draft.C)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_draft_propose_in_vocab(tmp_path):
    draft, args = _load(tmp_path)
    dev = torch.device("cuda")
    draft.to(dev)
    g = torch.Generator(device="cpu").manual_seed(7)
    taps = (torch.randn(1, draft.N_TAPS, draft.C, generator=g) * 0.1).to(dev)
    ctx_k, ctx_v = [], []
    for i in range(len(draft.layers)):
        mixed = draft.mix_taps(taps, i)
        k, v = draft.context_rows(mixed, i, torch.tensor([0], device=dev))
        ctx_k.append(k)
        ctx_v.append(v)
    f_rnn = draft.f_rnn(taps[0])
    anchor_h = (torch.randn(draft.C, generator=g) * 0.1).to(dev)
    head_w = (torch.randn(args.vocab_size, draft.C, generator=g) * 0.02).to(dev, torch.bfloat16)

    def head_logits(x):
        return torch.nn.functional.linear(x.to(torch.bfloat16), head_w).float()

    fold_w, fold_b = draft.fold(head_logits)
    toks = draft.propose(5, anchor_h, 1, ctx_k, ctx_v, f_rnn, head_logits, fold_w, fold_b)
    assert toks.shape == (draft.BLOCK - 1,)
    assert torch.isfinite(toks).all()
    assert ((toks >= 0) & (toks < args.vocab_size)).all()
    # deterministic: the same inputs propose the same chain
    toks2 = draft.propose(5, anchor_h, 1, ctx_k, ctx_v, f_rnn, head_logits, fold_w, fold_b)
    assert torch.equal(toks, toks2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_draft_matches_inkling2_dspark_math(tmp_path):
    """The ERNIE port is the inkling2 DSpark math: with identical weights and geometry
    (block 8, head_dim 128, theta 1e6) the two drafters propose the same chain."""
    from freetoken.models.inkling2.draft import DraftLayer as IK2Layer, DSparkDraft as IK2Draft

    draft, args = _load(tmp_path)
    dev = torch.device("cuda")
    draft.to(dev)
    ik2_layers = [IK2Layer(**{k: v for k, v in vars(l).items()}) for l in draft.layers]
    ik2 = IK2Draft(draft.C, draft.n_vocab, ik2_layers, alpha=draft.alpha, norm=draft.norm,
                   mask=draft.mask, mk_dflare=draft.mk_dflare, w1=draft.w1, rnn_w=draft.rnn_w,
                   rnn_b=draft.rnn_b, hout_w=draft.hout_w, hout_b=draft.hout_b).to(dev)
    g = torch.Generator(device="cpu").manual_seed(11)
    taps = (torch.randn(1, draft.N_TAPS, draft.C, generator=g) * 0.1).to(dev)
    ctx_k, ctx_v = [], []
    for i in range(len(draft.layers)):
        mixed = draft.mix_taps(taps, i)
        k, v = draft.context_rows(mixed, i, torch.tensor([0], device=dev))
        ctx_k.append(k)
        ctx_v.append(v)
    f_rnn = draft.f_rnn(taps[0])
    anchor_h = (torch.randn(draft.C, generator=g) * 0.1).to(dev)
    head_w = (torch.randn(args.vocab_size, draft.C, generator=g) * 0.02).to(dev, torch.bfloat16)

    def head_logits(x):
        return torch.nn.functional.linear(x.to(torch.bfloat16), head_w).float()

    fw, fb = draft.fold(head_logits)
    mine = draft.propose(5, anchor_h, 1, ctx_k, ctx_v, f_rnn, head_logits, fw, fb)
    theirs = ik2.propose(5, anchor_h, 1, ctx_k, ctx_v, f_rnn, head_logits, fw, fb)
    assert torch.equal(mine, theirs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_proposer_attach_and_propose(tmp_path):
    """The proposer loads model.mtp.* through the family reader, folds the Markov bias
    through the target's real head, and proposes a block of in-vocab drafts on device."""
    from freetoken.spec.dspark.ernie55 import Ernie55DSparkProposer
    from freetoken.spec.dspark.proposer import CTX_FULL, CTX_LOCAL, FRNN
    from freetoken.spec.proposer import DraftContext
    from .harness import TinyEngine

    hf = make_checkpoint(str(tmp_path / "hf"), quantized=True, seed=3,
                         geo_overrides={"moe_intermediate_size": 512, "intermediate_size": 512})
    eng = TinyEngine(hf, max_seq_len=1024, max_running_req=2, quantized=True)
    mc = parse_config(cached_load_hf_config(hf))

    class _TP:
        size = 1
        rank = 0

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.model_config = mc
    cfg.model_path = hf
    cfg.speculative_num_steps = mc.dsv4_args.dspark_block_size
    cfg.max_running_req = 2
    cfg.max_seq_len = 1024
    cfg.tp_info = _TP()
    cfg.speculative_draft_model_path = None
    cfg.use_dummy_weight = False
    cfg.speculative_token_map = None

    prop = Ernie55DSparkProposer(cfg, eng.model)
    dev = torch.device("cuda")
    slot_states = {}
    for spec in Ernie55DSparkProposer.slot_states(cfg):
        n = len(spec.layer_ids)
        shape = (n, 3) + tuple(spec.shape) if n else (3,) + tuple(spec.shape)
        slot_states[spec.name] = torch.zeros(shape, dtype=spec.dtype, device=dev)
    ctx = DraftContext(page_table=None, num_pages=0, device=dev,
                       stream=torch.cuda.current_stream(), dummy_req=None, slot_states=slot_states)
    prop.attach(ctx)
    assert prop.draft is not None
    assert prop.fold_w.shape == (mc.dsv4_args.vocab_size, prop.draft.R)

    class _Req:
        cached_len = 1
        linear_slot_idx = 0
        table_idx = 0

    pending = torch.tensor([5], device=dev)
    proposal = prop.propose([_Req()], torch.tensor([0], device=dev), pending)
    toks = proposal.draft_token_ids
    assert toks.shape == (1, mc.dsv4_args.dspark_block_size)
    assert ((toks >= 0) & (toks < mc.dsv4_args.vocab_size)).all()


# Served end to end in a subprocess: a second LLM in this process would trip the
# "CUDA already initialized" guard, and the plain / spec runs must not share state.
_SERVE_SCRIPT = """
import json, sys
import torch
from transformers import AutoConfig, PretrainedConfig
class _Cfg(PretrainedConfig):
    model_type = "ernie5_5_moe"
    def __init__(self, **kw):
        self.max_position_embeddings = kw.get("max_position_embeddings", 1024)
        kw.setdefault("max_position_embeddings", 1024)
        super().__init__(**kw)
AutoConfig.register("ernie5_5_moe", _Cfg)
from freetoken.llm import LLM
from freetoken.core import SamplingParams
hf, mode, k, gbs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kw = {"speculative_algorithm": "dspark", "speculative_num_steps": k} if mode == "spec" else {}
llm = LLM(model_path=hf, dtype=torch.bfloat16, max_running_req=2, cuda_graph_max_bs=gbs, **kw)
got = llm.generate([[5, 17, 3, 42]], [SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)])[0]["token_ids"]
print("TOKENS:" + json.dumps(got))
"""


def _serve(hf: str, mode: str, k: int, graphs: bool) -> list[int]:
    r = subprocess.run(
        [sys.executable, "-c", _SERVE_SCRIPT, hf, mode, str(k), "2" if graphs else "0"],
        capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, f"{mode} serve failed:\n{r.stderr[-3000:]}"
    line = next(l for l in r.stdout.splitlines() if l.startswith("TOKENS:"))
    return json.loads(line[len("TOKENS:"):])


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("k,graphs", [(7, False), (7, True), (1, False)])
def test_spec_decode_reproduces_greedy(tmp_path, k, graphs):
    """Served DSpark verify commits exactly the plain greedy chain, token for token.

    A verify row and a decode step run different GEMM / attention shapes, so their
    logits can differ by an ulp; fan-in scaled weights keep activations O(1) so that
    ulp does not grow through the mHC residual into a non-tie argmax flip (the
    verify-contract lever). This pins the two numerics contracts that make verify
    reproduce decode: the block-fp8 dense projections ride the W8A16 decode GEMV
    per row (not the W8A8 GEMM, which quantizes the activation), and the DSA
    trailing window starts at position 0 for sequences shorter than the window.
    """
    hf = make_checkpoint(str(tmp_path / "hf"), quantized=True, seed=3, fan_in=True,
                         geo_overrides={"moe_intermediate_size": 512, "intermediate_size": 512})
    vocab = {f"t{i}": i for i in range(64)}
    tokenizer = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
                 "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
                 "decoder": None, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}}
    with open(hf + "/tokenizer.json", "w") as f:
        json.dump(tokenizer, f)
    with open(hf + "/tokenizer_config.json", "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1024,
                   "bos_token": "t1", "eos_token": "t2", "unk_token": "t0", "pad_token": "t0"}, f)
    assert _serve(hf, "spec", k, graphs) == _serve(hf, "plain", k, graphs)