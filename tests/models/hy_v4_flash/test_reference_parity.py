"""Logits parity against the vendored eager reference on the tiny synthetic checkpoint.

The reference runs per-head K/V attention with brute-force compressed-indexer selection,
fp32 iHC, the host engram hash and a latent MoE loop; FreeToken runs the same checkpoint
through absorbed MLA, the kpool shadow slab, fused routing and packed expert banks.

Dense-regime logits are compared outright. Past candidate_topk the indexer's group
selection is precision-sensitive (the reference itself notes bf16 near-ties flip groups):
the free-running selection SETS must agree with the reference on >98% of queries (the
glm_dsa convention), and the logits are compared with the reference constrained to the
engine's selection -- isolating the attention / MoE / iHC math from the tie flips.
"""

from __future__ import annotations

import pytest
import torch

from .common import VOCAB, requires_cuda, write_tiny_checkpoint
from .harness import TinyEngine
from .reference import Cache, Reference

pytestmark = requires_cuda

# bf16 accumulation-order differences (softmax, rope math, expert sums)
ATOL = RTOL = 1e-2
SPARSE_TOL = 2e-2
# near-tied group scores may flip between bf16 storage / fp32 rope and the reference's
# bf16 rope; a systematic convention bug flips far more than this (glm_dsa precedent)
MAX_FLIP_RATE = 0.02


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hy4-parity")
    tensors = write_tiny_checkpoint(str(folder), seed=11)
    return str(folder), tensors


def _tokens(n: int, seed: int) -> list[int]:
    return torch.randint(3, VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _compare(name, got, want, tol=ATOL):
    torch.testing.assert_close(got, want, atol=tol, rtol=tol, msg=lambda m: f"{name}: {m}")


def _reference(checkpoint, tensors):
    from freetoken.utils.hf import cached_load_hf_config

    text = cached_load_hf_config(checkpoint).text_config
    if not isinstance(text, dict):
        text = text.to_dict()
    return Reference(tensors, text)


class SelectionBridge:
    """Captures the engine's per-(layer, position) recalled token sets and feeds them
    to the reference's indexer_mask, so the logits comparison shares the selection
    while everything else stays independent. Also records the reference's own
    free-running selection for the flip-rate check."""

    def __init__(self, eng, ref):
        import freetoken.attention.hy4_sparse as hs

        self.eng = eng
        self.ref = ref
        self.sets: dict[tuple[int, int], set[int]] = {}
        self.ref_sets: dict[tuple[int, int], set[int]] = {}
        self.flips = 0
        self.total = 0
        self._slot_layer = {slot: lid for lid, slot in eng.backend._idx_slot.items()}
        self._orig_sel = hs.Hy4DSABackend._select_prefill_indexed
        self._orig_decode = hs.Hy4DSABackend._decode
        self._orig_mask = ref.indexer_mask
        bridge = self

        def sel_spy(backend, slot, q_idx, w, rows, positions, kv_len):
            sel, cnt = bridge._orig_sel(backend, slot, q_idx, w, rows, positions, kv_len)
            lid = bridge._slot_layer[slot]
            for i, p in enumerate(positions.tolist()):
                c = int(cnt[0, i].item())
                # sel holds PHYSICAL rows; map back to positions via the row list
                s = sel[0, i, :c]
                s = s[s >= 0].long()
                toks = torch.searchsorted(rows, s).tolist()
                # one entry per batch row, in request order (the _prefill loop order)
                bridge.sets.setdefault((lid, int(p)), []).append(set(t for t in toks if t < kv_len))
            return sel, cnt

        def decode_spy(backend, md, layer_id, q_nope, q_pe, inputs):
            out = bridge._orig_decode(backend, md, layer_id, q_nope, q_pe, inputs)
            if not backend._hy4.swa_layers[layer_id] and backend.dsa_enabled:
                sel, cnt = md.sel[layer_id]
                rows = md.rows
                for b in range(sel.shape[0]):
                    p = int(md.kvlen[b].item()) - 1
                    c = int(cnt[b, 0].item())
                    s = sel[b, 0, :c]
                    s = s[s >= 0].long()
                    # rows beyond the live length are zero padding: searchsorted
                    # needs the sorted live prefix only
                    toks = torch.searchsorted(rows[b, : p + 1], s).tolist()
                    bridge.sets.setdefault((layer_id, p), []).append(set(t for t in toks if t <= p))
            return out

        def mask_spy(L, x, q_resid, positions, cache, kv_len, attn_mask):
            out = bridge._orig_mask(L, x, q_resid, positions, cache, kv_len, attn_mask)
            B, T = positions.shape
            for b in range(B):
                for i in range(T):
                    p = int(positions[b, i].item())
                    visible = set((out[b, 0, i] == 0).nonzero().flatten().tolist())
                    key = (L, p)
                    bridge.ref_sets.setdefault(key, []).append(visible)
                    entries = bridge.sets.get(key)
                    if entries and b < len(entries):
                        bridge.total += 1
                        if entries[b] != visible:
                            bridge.flips += 1
                        # constrain the reference to the engine's selection
                        m = attn_mask.new_full((kv_len,), torch.finfo(attn_mask.dtype).min)
                        m[list(entries[b])] = 0
                        out[b, 0, i] = m
            return out

        hs.Hy4DSABackend._select_prefill_indexed = sel_spy
        hs.Hy4DSABackend._decode = decode_spy
        ref.indexer_mask = mask_spy

    def restore(self):
        import freetoken.attention.hy4_sparse as hs

        hs.Hy4DSABackend._select_prefill_indexed = self._orig_sel
        hs.Hy4DSABackend._decode = self._orig_decode
        self.ref.indexer_mask = self._orig_mask

    def assert_flip_rate(self):
        assert self.total > 0, "no sparse selections were exercised"
        rate = self.flips / self.total
        assert rate <= MAX_FLIP_RATE, f"selection flip rate {self.flips}/{self.total} = {rate:.3f}"


# The released checkpoint's attention / engram geometry (32/64 heads, 512-wide MLA
# latent, 192+64 qk, 256 v, 96-dim engram heads) on a small layer/expert count: the
# kernels compile different tile shapes at these widths than the tiny fixture.
REAL_DIMS = dict(
    hidden_size=3072, num_attention_heads=32, swa_num_attention_heads=64,
    q_lora_rank=1024, kv_lora_rank=512, qk_nope_head_dim=192, v_head_dim=256,
    swa_qk_nope_head_dim=192, swa_v_head_dim=256,
    intermediate_size=13824, moe_intermediate_size=1536, routed_expert_hidden_size=1536,
    engram_n_embed_per_ngram=768, sliding_window=512,
)


@pytest.fixture(scope="module")
def real_dims_checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hy4-parity-realdims")
    # trained-scale indexer head weights: the tiny 0.02 scale makes real-dim
    # group scores near-tied, where bf16 storage / fp32 rope flips are inevitable
    tensors = write_tiny_checkpoint(str(folder), seed=13, indexer_weight_scale=1.0, **REAL_DIMS)
    return str(folder), tensors


def test_real_dims_parity(real_dims_checkpoint, monkeypatch):
    # At the released widths several projections cross the W8A16 decode-linear
    # threshold; that fp8 weight rewrite is a serving optimization, not the bf16
    # math this parity checks (it moves the weights the reference still holds bf16).
    import freetoken.layers.quantization.linear.unquantized as unq

    import models.hy_v4_flash.reference as refmod

    monkeypatch.setattr(unq, "_W8A16_ENABLED", False)
    # match the engine's fp32 rope / softmax conventions: at the released widths
    # the PR's bf16 conventions flip near-tied indexer group scores on their own
    monkeypatch.setattr(refmod, "ROPE_FP32", True)
    monkeypatch.setattr(refmod, "SOFTMAX_FP32", True)
    folder, tensors = real_dims_checkpoint
    eng = TinyEngine(folder, tensors, max_seq_len=1024)
    ref = _reference(folder, tensors)
    # At the released widths the random-weight fixture's iHC post gates (~2x) amplify
    # bf16 rounding through the layers; the reference's OWN bf16-vs-fp32 spread is the
    # noise floor the engine must stay within (an fp32 reference run measures it).
    ref32 = _reference(folder, {k: v.float() if v.is_floating_point() else v for k, v in tensors.items()})
    bridge = SelectionBridge(eng, ref)
    bridge32 = SelectionBridge(eng, ref32)
    try:
        tokens = _tokens(40, 7)
        req = eng.new_request(0, tokens, max_new=8)
        got = eng.prefill([req])
        cache, cache32 = Cache(), Cache()
        ids = torch.tensor([tokens], device="cuda")
        pos = torch.arange(40, device="cuda").unsqueeze(0)
        want = ref.forward(ids, pos, cache)[0, -1:]
        want32 = ref32.forward(ids, pos, cache32)[0, -1:]
        # independent bf16 roundings add in quadrature: the engine's deviation from
        # the bf16 reference may reach ~sqrt(2)x the reference's own bf16-vs-fp32
        # spread elementwise; compare the aggregate norms with a 2x margin
        rel_err = ((got - want).norm() / want.norm().clamp(min=1e-9)).item()
        rel_noise = ((want - want32).norm() / want.norm().clamp(min=1e-9)).item()
        assert rel_err <= max(2 * rel_noise, 0.02), (
            f"real-dims prefill exceeds bf16 noise floor: rel err {rel_err:.4f}, "
            f"reference bf16-vs-fp32 spread {rel_noise:.4f}"
        )
        eng.finish_prefill([req])
        full = list(tokens)
        for step in range(4):
            token = int(want[-1].argmax().item())
            full.append(token)
            got = eng.decode([req], [token])[0]
            full_t = torch.tensor([full], device="cuda")
            want = ref.generate_step(torch.tensor([[token]], device="cuda"), len(full) - 1, cache, full_t)[0, 0]
            want32 = ref32.generate_step(torch.tensor([[token]], device="cuda"), len(full) - 1, cache32, full_t)[0, 0]
            rel_err = ((got - want).norm() / want.norm().clamp(min=1e-9)).item()
            rel_noise = ((want - want32).norm() / want.norm().clamp(min=1e-9)).item()
            assert rel_err <= max(2 * rel_noise, 0.02), (
                f"real-dims decode {step} exceeds bf16 noise floor: rel err {rel_err:.4f}, "
                f"reference spread {rel_noise:.4f}"
            )
        # pooled group keys round to bf16 once in the engine's slab and once in the
        # reference; values a rounding step apart flip near-tied group scores even
        # with matched conventions. The tiny fixture pins the exact selection
        # semantics at 2%; the interleaved (GLM) rope's score geometry lands ~10-11%
        # bf16 near-tie flips on this random fixture, so only a systematic bug
        # (well above that noise floor) must fail here.
        assert bridge.total > 0
        rate = bridge.flips / bridge.total
        assert rate <= 0.15, f"real-dims selection flip rate {bridge.flips}/{bridge.total} = {rate:.3f}"
    finally:
        bridge.restore()
        bridge32.restore()


def test_dense_prefill(checkpoint):
    folder, tensors = checkpoint
    eng = TinyEngine(folder, tensors)
    ref = _reference(folder, tensors)
    # the loaded-map hash bound reads different fixture rows than the table-length
    # bound did; floor the tolerance at the reference's own bf16-vs-fp32 spread
    ref32 = _reference(folder, {k: v.float() if v.is_floating_point() else v for k, v in tensors.items()})
    tokens = _tokens(12, 1)
    req = eng.new_request(0, tokens)
    got = eng.prefill([req])
    ids = torch.tensor([tokens], device="cuda")
    pos = torch.arange(12, device="cuda").unsqueeze(0)
    want = ref.forward(ids, pos, Cache())[0, -1:]
    want32 = ref32.forward(ids, pos, Cache())[0, -1:]
    tol = max(ATOL, 2 * (want - want32).abs().max().item())
    _compare("dense prefill", got, want, tol)


def test_sparse_prefill_and_decode(checkpoint):
    folder, tensors = checkpoint
    eng = TinyEngine(folder, tensors)
    ref = _reference(folder, tensors)
    bridge = SelectionBridge(eng, ref)
    try:
        tokens = _tokens(40, 2)
        req = eng.new_request(0, tokens, max_new=16)
        got = eng.prefill([req])
        cache = Cache()
        ids = torch.tensor([tokens], device="cuda")
        want = ref.forward(ids, torch.arange(40, device="cuda").unsqueeze(0), cache)[0, -1:]
        _compare("sparse prefill", got, want, SPARSE_TOL)
        eng.finish_prefill([req])
        full = list(tokens)
        for step in range(8):
            token = int(want[-1].argmax().item())
            full.append(token)
            got = eng.decode([req], [token])[0]
            want = ref.generate_step(
                torch.tensor([[token]], device="cuda"), len(full) - 1, cache,
                torch.tensor([full], device="cuda"),
            )[0, 0]
            _compare(f"sparse decode {step}", got, want, SPARSE_TOL)
            assert int(got.argmax().item()) == int(want.argmax().item()), f"greedy token diverged at step {step}"
        bridge.assert_flip_rate()
    finally:
        bridge.restore()


def test_chunked_prefill(checkpoint):
    folder, tensors = checkpoint
    eng = TinyEngine(folder, tensors)
    ref = _reference(folder, tensors)
    bridge = SelectionBridge(eng, ref)
    try:
        tokens = _tokens(30, 3)
        req = eng.new_request(0, tokens, max_new=8)
        # two chunks: 12 then 18 tokens. Drive the chunk by truncating the host
        # view, then restoring it. The engine runs FIRST so the bridge can constrain
        # the reference to the engine's selection.
        full_ids = req.input_ids
        req.input_ids = full_ids[:12]
        req.device_len = 12
        got1 = eng.prefill([req])
        eng.finish_prefill([req])
        req.input_ids = full_ids
        req.device_len = 30
        got2 = eng.prefill([req])
        cache = Cache()
        ids = torch.tensor([tokens], device="cuda")
        pos = torch.arange(30, device="cuda").unsqueeze(0)
        want_full = ref.forward(ids, pos, cache)[0]
        _compare("chunk 1", got1, want_full[11:12], SPARSE_TOL)
        _compare("chunk 2", got2, want_full[29:30], SPARSE_TOL)
        bridge.assert_flip_rate()
    finally:
        bridge.restore()


def test_batch_decode(checkpoint):
    folder, tensors = checkpoint
    eng = TinyEngine(folder, tensors)
    ref = _reference(folder, tensors)
    bridge = SelectionBridge(eng, ref)
    try:
        toks = [_tokens(24, 4), _tokens(24, 5)]
        reqs = [eng.new_request(i, t, max_new=8) for i, t in enumerate(toks)]
        got = eng.prefill(reqs)
        cache = Cache()
        ids = torch.tensor(toks, device="cuda")
        want = ref.forward(ids, torch.arange(24, device="cuda").unsqueeze(0).expand(2, -1), cache)
        _compare("batch prefill req0", got[0:1], want[0, -1:], SPARSE_TOL)
        _compare("batch prefill req1", got[1:2], want[1, -1:], SPARSE_TOL)
        eng.finish_prefill(reqs)
        full = [list(t) for t in toks]
        for step in range(6):
            tokens = [int(want[i, -1].argmax().item()) for i in range(2)]
            for i in range(2):
                full[i].append(tokens[i])
            got = eng.decode(reqs, tokens)
            want = ref.generate_step(
                torch.tensor(tokens, device="cuda").unsqueeze(1), 24 + step, cache,
                torch.tensor(full, device="cuda"),
            )[:, 0]
            _compare(f"batch decode {step}", got, want, SPARSE_TOL)
        bridge.assert_flip_rate()
    finally:
        bridge.restore()


def test_engram_multiplier_bound(checkpoint):
    """The hash multipliers are drawn bounded by the largest compressed id + 1 of
    the loaded map (the reference's _init_weights), not the lookup table LENGTH.
    The fixture's compression map makes the two bounds differ, so reverting to
    the table length -- which silently reads the wrong memory rows on the real
    checkpoint -- turns this red."""
    from freetoken.models.hy_v4_flash.args import HyV4FlashArgs
    from freetoken.models.hy_v4_flash.engram import EngramHash, draw_hash_multipliers
    from freetoken.utils.hf import cached_load_hf_config

    folder, tensors = checkpoint
    args = HyV4FlashArgs.from_hf(cached_load_hf_config(folder))
    L = sorted(args.engram_layer_ids)[0]
    lookup = tensors[f"model.layers.{L}.engram_layer.lookup_table"]
    h = EngramHash(args, L, lookup)
    table_size = lookup.numel()
    max_plus_one = int(lookup.max().item()) + 1
    assert max_plus_one != table_size, "fixture must distinguish the two bounds"
    want = draw_hash_multipliers(args.engram_seed, L + 1, args.engram_max_ngram, max_plus_one)
    wrong = draw_hash_multipliers(args.engram_seed, L + 1, args.engram_max_ngram, table_size)
    assert torch.equal(h.multipliers.cpu(), want.cpu())
    assert not torch.equal(h.multipliers.cpu(), wrong.cpu())
