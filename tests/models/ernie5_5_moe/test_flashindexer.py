"""FlashIndexer head routing: the optional top-8-of-64 gate loads from the
checkpoint's flashindexer/ sidecar and actually changes the indexer selection
(a gate that is built but never loaded/applied would leave the logits identical
to the all-heads run -- the silent failure vLLM-ERNIE's loader assert guards).
"""

import json
import os

import pytest
import torch

from .common import make_checkpoint, write_flashindexer_sidecar
from .harness import TinyEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _enable(folder):
    p = os.path.join(folder, "config.json")
    cfg = json.load(open(p))
    cfg["enable_flashindexer"] = True
    json.dump(cfg, open(p, "w"))
    write_flashindexer_sidecar(folder, cfg)


def test_flashindexer_gate_changes_selection(tmp_path):
    base = str(tmp_path / "base")
    make_checkpoint(base, quantized=False, seed=5)
    tokens = [4, 19, 2, 57, 11, 38] * 30  # 180 tokens: DSA selection is active
    eng_all = TinyEngine(base, max_seq_len=512, max_running_req=1)
    req = eng_all.new_request(0, tokens, max_new=4)
    logits_all = eng_all.prefill([req])[0].clone()

    gated = str(tmp_path / "gated")
    make_checkpoint(gated, quantized=False, seed=5)
    _enable(gated)
    eng_gate = TinyEngine(gated, max_seq_len=512, max_running_req=1)
    req2 = eng_gate.new_request(0, tokens, max_new=4)
    logits_gate = eng_gate.prefill([req2])[0]
    assert torch.isfinite(logits_gate).all()
    # top-8 head routing reweights the indexer scores -> a different selection
    assert not torch.allclose(logits_all, logits_gate, atol=1e-3)


def test_flashindexer_missing_sidecar_raises(tmp_path):
    folder = str(tmp_path / "nosidecar")
    make_checkpoint(folder, quantized=False, seed=5)
    p = os.path.join(folder, "config.json")
    cfg = json.load(open(p))
    cfg["enable_flashindexer"] = True
    json.dump(cfg, open(p, "w"))
    with pytest.raises(FileNotFoundError, match="flashindexer"):
        TinyEngine(folder, max_seq_len=512, max_running_req=1)
