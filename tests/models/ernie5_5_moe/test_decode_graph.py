"""Decode CUDA-graph capture / replay parity: the captured graph (DSV4 snapshot
+ DSA static row/kvlen buffers + compressor carry ring + indexer, all in-graph)
must reproduce the independent full-sequence reference at every replayed step.
Guards the overlap-safety staging: a replay that reads the LIVE page table or a
stale kvlen diverges silently from the eager math.
"""

import pytest
import torch

from .common import make_checkpoint
from .harness import TinyEngine
from .reference import Ernie55Reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ATOL, RTOL = 0.06, 0.02


def test_decode_graph_replay_matches_reference(tmp_path):
    ck = make_checkpoint(str(tmp_path / "ck"), quantized=False, seed=9)
    eng = TinyEngine(ck, max_seq_len=1024, max_running_req=1)
    ref = Ernie55Reference(ck)
    prompt = [6, 18, 2, 44, 11, 59] * 25  # 150 tokens
    continuation = [3, 27, 51, 8, 36, 14, 22, 47]
    req = eng.new_request(0, prompt, max_new=16)
    eng.prefill([req])
    eng.finish_prefill([req])
    full = ref.forward(prompt + continuation)

    backend = eng.backend
    backend.init_capture_graph(max_seq_len=1024, bs_list=[1])

    from freetoken.core import Batch

    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.input_ids = torch.zeros(1, dtype=torch.int32, device=eng.device)
    batch.positions = torch.zeros(1, dtype=torch.long, device=eng.device)
    batch.active_table_idx = torch.zeros(1, dtype=torch.long, device=eng.device)
    batch.out_loc = torch.zeros(1, dtype=torch.int32, device=eng.device)

    def stage(token):
        req.append_host(torch.tensor([token], dtype=torch.int32))
        req.complete_one()
        batch.input_ids.fill_(token)
        batch.positions.fill_(req.device_len - 1)
        batch.out_loc.copy_(eng.page_table[0, req.device_len - 1])
        backend.prepare_metadata(batch)

    # capture on the first continuation token's batch shape; the capture itself
    # runs on DUMMY rows (its output is discarded) -- every real step restages
    # the live rows into the SAME captured buffers and replays.
    stage(continuation[0])
    backend.prepare_for_capture(batch)
    with eng.ctx.forward_batch(batch), eng.model.forward_host_ctx(batch, False):
        eng.model.forward()  # warmup outside the graph
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = eng.model.forward()
    backend.prepare_for_replay(batch)
    graph.replay()
    torch.testing.assert_close(
        logits[0].float(), full[len(prompt)], atol=ATOL, rtol=RTOL,
        msg=lambda m: f"replay step 0: {m}",
    )
    for i, tok in enumerate(continuation[1:], start=1):
        stage(tok)
        backend.prepare_for_replay(batch)
        graph.replay()
        torch.testing.assert_close(
            logits[0].float(), full[len(prompt) + i], atol=ATOL, rtol=RTOL,
            msg=lambda m: f"replay step {i}: {m}",
        )
