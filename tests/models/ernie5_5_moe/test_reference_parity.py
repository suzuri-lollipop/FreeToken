"""ERNIE 5.5 full-model parity against the independent torch reference
(``reference.py``, transcribed from baidu/vLLM-ERNIE), on a tiny synthetic
bf16 checkpoint: prefill logits, decode == prefill continuation (the KV cache /
compressor carry / indexer selection must reproduce the full-sequence math),
and a ragged two-request batch.
"""

import pytest
import torch

from .common import make_checkpoint
from .harness import TinyEngine
from .reference import Ernie55Reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# bf16 chains (absorbed-MLA bmm vs explicit up-projection, fused vs eager
# reductions) round differently; the oracle is the same math in fp32/bf16.
ATOL, RTOL = 0.06, 0.02


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return make_checkpoint(str(tmp_path_factory.mktemp("ernie55_ckpt")), quantized=False, seed=3)


@pytest.fixture(scope="module")
def engine(checkpoint):
    # the module's requests never free their window pages; budget the whole run
    return TinyEngine(checkpoint, max_seq_len=3072, max_running_req=2)


@pytest.fixture(scope="module")
def ref(checkpoint):
    return Ernie55Reference(checkpoint)


def test_prefill_parity(engine, ref):
    tokens = [5, 17, 3, 42, 8, 61, 2, 19] * 20  # 160 tokens: crosses one 128-block
    req = engine.new_request(0, tokens, max_new=8)
    logits = engine.prefill([req])
    expected = ref.forward(tokens)
    torch.testing.assert_close(logits[-1], expected[-1], atol=ATOL, rtol=RTOL)


def test_decode_matches_prefill_continuation(engine, ref):
    prompt = [7, 23, 1, 55, 12, 38] * 25  # 150 tokens
    continuation = [4, 19, 60, 2, 33, 11, 27, 50]
    req = engine.new_request(0, prompt, max_new=16)
    engine.prefill([req])
    engine.finish_prefill([req])
    full = ref.forward(prompt + continuation)
    for i, tok in enumerate(continuation):
        logits = engine.decode([req], [tok])
        torch.testing.assert_close(
            logits[0], full[len(prompt) + i], atol=ATOL, rtol=RTOL,
            msg=lambda m: f"decode step {i}: {m}",
        )


def test_dsa_short_window_rows(engine):
    """kvlen < window: the DSA trailing window is positions 0..kvlen-1. The old
    ``kvlen - window`` offset went negative and clamped every slot to row 0
    (attending position 0 on repeat); short sequences are the spec-verify anchor
    regime. kvlen >= window keeps the trailing offset unchanged."""
    backend = engine.backend
    win = backend.dsa_window
    rows = torch.arange(1000, 1000 + win, dtype=torch.int32, device="cuda").view(1, -1)
    for kvlen in (1, 5, win - 1):
        got = backend._dsa_window_rows(rows, torch.tensor([kvlen], device="cuda"))
        assert got[0, :kvlen].tolist() == rows[0, :kvlen].tolist()
        assert (got[0, kvlen:] == -1).all()
    long_rows = torch.arange(2000, 2000 + 300, dtype=torch.int32, device="cuda").view(1, -1)
    got = backend._dsa_window_rows(long_rows, torch.tensor([300], device="cuda"))
    assert got[0].tolist() == long_rows[0, 300 - win:300].tolist()


def test_int8_kv_decode_close_to_bf16(tmp_path):
    """int8 KV cache (the DSA latent slab) reproduces bf16-KV decode logits within
    quantization noise: the per-row ckv/kpe scales descale on read, so a teacher-forced
    chain stays close (a wrong store/read split or scale would diverge outright)."""
    from freetoken.kvcache.format import NONE, resolve_kv_format

    hf = make_checkpoint(str(tmp_path / "hf"), quantized=False, seed=3)
    prompt = [5, 17, 3, 42]
    cont = [35, 3, 14, 3, 30, 4]

    def run(kv_format):
        eng = TinyEngine(hf, max_seq_len=1024, max_running_req=2, kv_format=kv_format)
        req = eng.new_request(0, prompt, max_new=16)
        eng.prefill([req])
        eng.finish_prefill([req])
        return [eng.decode([req], [t])[0].clone() for t in cont]

    bf16 = run(NONE)
    int8 = run(resolve_kv_format("int8"))
    for i, (a, b) in enumerate(zip(bf16, int8)):
        torch.testing.assert_close(
            b, a, atol=0.15, rtol=0.05,
            msg=lambda m: f"int8 KV decode step {i}: {m}",
        )


def test_ragged_batch_parity(engine, ref):
    ta = [3, 14, 15, 9, 26, 53] * 12
    tb = [2, 37, 8, 28, 1, 42] * 22
    ra = engine.new_request(0, ta, max_new=4)
    rb = engine.new_request(1, tb, max_new=4)
    logits = engine.prefill([ra, rb])
    ea, eb = ref.forward(ta), ref.forward(tb)
    torch.testing.assert_close(logits[0], ea[-1], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(logits[1], eb[-1], atol=ATOL, rtol=RTOL)


def test_radix_prefix_hit_parity(engine, ref):
    """A page-aligned prefix hit resumes the HCA compressor carry BY VALUE from
    the donor's state ring and reuses its DSA latent/index rows; the extend must
    equal the cold full-sequence math."""
    prompt = [8, 24, 5, 61, 16, 43] * 43  # 258 tokens -> 2 full pages + tail
    donor = engine.new_request(0, prompt, max_new=4)
    engine.prefill([donor])
    engine.finish_prefill([donor])
    # share the first 256 tokens (2 pages), diverge afterwards
    tokens = prompt[:256] + [12, 34, 56, 7, 29, 48, 3, 60, 15, 31]
    req = engine.new_request_on_prefix(1, donor, tokens, max_new=4)
    assert req.cached_len == 256
    logits = engine.prefill([req])
    expected = ref.forward(tokens)
    torch.testing.assert_close(logits[-1], expected[-1], atol=ATOL, rtol=RTOL)
