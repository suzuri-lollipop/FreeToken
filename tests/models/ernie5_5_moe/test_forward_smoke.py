"""Serving-path smoke: the block-fp8 + MXFP4 (W4A8) checkpoint loads through the
real quant dialect (Fp8BlockConfig with the family's scale_fmt override, MXFP4
expert banks on the offload cache, SituGLU experts) and prefills / decodes.
Guards the silent failures the quant path can hide: a wrong scale dtype/name
mapping, an expert activation the ds_fp4 kernel does not implement, a bank
layer-id offset.
"""

import pytest
import torch

from .common import make_checkpoint
from .harness import TinyEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def qengine(tmp_path_factory):
    ck = make_checkpoint(str(tmp_path_factory.mktemp("ernie55_q")), quantized=True, seed=3)
    return TinyEngine(ck, max_seq_len=1024, max_running_req=2, quantized=True)


def test_quantized_serving_runs(qengine):
    tokens = [5, 17, 3, 42, 8, 61, 2, 19] * 20
    req = qengine.new_request(0, tokens, max_new=8)
    logits = qengine.prefill([req])
    assert logits.shape == (1, 64)
    assert torch.isfinite(logits).all()
    qengine.finish_prefill([req])
    prev = logits[0].argmax().item()
    for _ in range(4):
        logits = qengine.decode([req], [prev])
        assert torch.isfinite(logits).all()
        prev = logits[0].argmax().item()


def test_quantized_matches_dequant_oracle(tmp_path_factory):
    """The fp8/MXFP4 kernels reproduce the model on the EXACT dequantized weights
    (up to W4A8 activation quant + accumulation order) -- a broken scale mapping
    or expert activation diverges far past this bound."""
    from .common import make_dequantized_checkpoint
    from .reference import Ernie55Reference

    folder = str(tmp_path_factory.mktemp("ernie55_q2"))
    qck = make_checkpoint(folder + "/q", quantized=True, seed=7)
    dck = make_dequantized_checkpoint(folder + "/d", seed=7)
    qeng = TinyEngine(qck, max_seq_len=512, max_running_req=1, quantized=True)
    ref = Ernie55Reference(dck)
    tokens = [9, 21, 4, 57, 13, 30] * 24
    req = qeng.new_request(0, tokens, max_new=4)
    got = qeng.prefill([req])[0]
    want = ref.forward(tokens)[-1]
    torch.testing.assert_close(got, want, atol=0.35, rtol=0.04)
