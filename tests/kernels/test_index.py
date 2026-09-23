"""CPU fallback of the embedding gather in kernel/index.py.

The JIT launcher is CUDA-only; before the fallback existed, calling indexing()
with host tensors handed host pointers to a CUDA kernel and corrupted the heap
silently -- the abort surfaced later in an unrelated free. The subprocess test
pins the crash-free guarantee without risking this test session.
"""

import subprocess
import sys

import torch

from freetoken.kernel import indexing
from freetoken.kernel.index import num_splits_for


def test_cpu_gather_matches_reference():
    w = torch.randn(64, 32, dtype=torch.bfloat16)
    idx = torch.tensor([3, 0, 63, 7], dtype=torch.int32)
    got = indexing(w, idx)
    assert got.shape == (4, 32) and got.dtype == w.dtype
    assert torch.equal(got, w[idx.long()])
    out = torch.empty(4, 32, dtype=torch.bfloat16)
    assert indexing(w, idx, output=out) is out
    assert torch.equal(out, w[idx.long()])


def test_cpu_vocab_range_zeros_out_of_range_rows():
    w = torch.randn(16, 8, dtype=torch.bfloat16)  # this rank's vocab slice
    idx = torch.tensor([9, 3, 20, 16], dtype=torch.int32)  # global ids
    got = indexing(w, idx, vocab_range=(8, 12))
    assert torch.equal(got[0], w[1])  # 9 - 8 in range
    assert torch.equal(got[3], w[8])  # 16 - 8 = 8 < 12
    assert not got[1].any()  # 3 - 8 < 0 -> zeroed row
    assert not got[2].any()  # 20 - 8 = 12 >= length -> zeroed row


def test_repeated_cpu_indexing_does_not_corrupt_the_heap():
    # Before the fallback this aborted inside an unrelated free (the corruption is
    # planted by the launch itself), so run it in a subprocess.
    code = (
        "import torch; from freetoken.kernel import indexing;"
        "w = torch.randn(32, 16); i = torch.tensor([1, 2]);"
        "indexing(w, i); indexing(w, i);"
        "x = torch.nn.functional.linear(torch.randn(4, 16), torch.randn(8, 16));"
        "assert x.shape == (4, 8); print('ok')"
    )
    r = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr[-800:]
    assert "ok" in r.stdout


def test_num_splits_for_element_size():
    assert num_splits_for(2048) == 4
    assert num_splits_for(4096) == 4
    assert num_splits_for(1024) == 2
    assert num_splits_for(512) == 1
