"""Sampler parameter preparation (engine/sample.py), CPU-only.

prepare() converts per-request SamplingParams into device tensors for the
fused sampling kernels. Every wrong value here still looks like a number: a
temperature that leaks 0.0 into a sampling row, a top_k that reads -1 as "keep
every token", or a greedy mask that misses a row would keep generating forever
while looking perfectly healthy. Expectations come from the sampling contract
itself (which columns of a mixed batch must be argmax, which must be clamped).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.sample import Sampler
from freetoken.kvcache.naive_cache import NaiveCacheHandle

VOCAB = 64


def make_req(uid: int, params: SamplingParams) -> Req:
    return Req(
        input_ids=torch.tensor([1, 2, 3]),
        table_idx=0,
        cached_len=1,
        output_len=8,
        uid=uid,
        sampling_params=params,
        cache_handle=NaiveCacheHandle(),
    )


def make_batch(*reqs: Req) -> Batch:
    return Batch(list(reqs), phase="decode")


def prepare(*reqs: Req):
    return Sampler(torch.device("cpu"), VOCAB).prepare(make_batch(*reqs))


def test_all_greedy_batch_skips_sampling_tensors():
    args = prepare(make_req(1, SamplingParams(temperature=0.0)), make_req(2, SamplingParams(top_k=1)))
    assert args.temperatures is None
    assert args.top_k is None
    assert args.top_p is None
    assert args.greedy_mask is None


def test_top_k_one_counts_as_greedy_even_with_temperature():
    args = prepare(make_req(1, SamplingParams(temperature=1.0, top_k=1)))
    assert args.temperatures is None


def test_non_greedy_row_gets_raw_parameters():
    args = prepare(make_req(1, SamplingParams(temperature=0.5, top_k=5, top_p=0.9)))
    assert args.temperatures.tolist() == pytest.approx([0.5])
    assert args.temperatures.dtype == torch.float32
    assert args.top_k.tolist() == [5]
    assert args.top_k.dtype == torch.int32
    assert args.top_p.tolist() == pytest.approx([0.9])
    assert args.greedy_mask is None


def test_mixed_batch_neutralizes_the_greedy_row():
    greedy = SamplingParams(temperature=0.0, top_k=50, top_p=0.2)
    sampled = SamplingParams(temperature=0.7, top_k=5, top_p=0.9)
    args = prepare(make_req(1, greedy), make_req(2, sampled))

    # the greedy row must sample from a neutral distribution, not approximate argmax
    assert args.temperatures.tolist() == pytest.approx([1.0, 0.7])
    assert args.top_k.tolist() == [VOCAB, 5]
    assert args.top_p.tolist() == pytest.approx([1.0, 0.9])
    assert args.greedy_mask.tolist() == [True, False]
    assert args.greedy_mask.dtype == torch.bool


def test_invalid_top_k_reads_as_keep_all_tokens():
    args = prepare(make_req(1, SamplingParams(temperature=0.5, top_k=0)))
    assert args.top_k is None  # every row keeps the full vocab -> tensor is dropped


def test_out_of_range_top_p_clamps_into_range():
    low = prepare(make_req(1, SamplingParams(temperature=0.5, top_p=0.0)))
    assert low.top_p.tolist() == pytest.approx([1e-6])

    high = prepare(make_req(2, SamplingParams(temperature=0.5, top_p=1.5)))
    # clamped to exactly 1.0, which is uniform, so the tensor is dropped entirely
    assert high.top_p is None

    mixed = prepare(
        make_req(1, SamplingParams(temperature=0.5, top_p=0.5)),
        make_req(2, SamplingParams(temperature=0.5, top_p=1.5)),
    )
    assert mixed.top_p.tolist() == pytest.approx([0.5, 1.0])


def test_uniform_top_p_rows_drop_the_tensor():
    args = prepare(
        make_req(1, SamplingParams(temperature=0.5, top_p=1.0)),
        make_req(2, SamplingParams(temperature=0.7, top_p=1.0)),
    )
    assert args.top_p is None
    assert args.temperatures.tolist() == pytest.approx([0.5, 0.7])


def test_tiny_temperature_clamps_to_sampling_minimum():
    args = prepare(make_req(1, SamplingParams(temperature=1e-9, top_p=0.9)))
    assert args.temperatures.tolist() == pytest.approx([1e-6])


def test_sample_all_greedy_batch_is_argmax():
    sampler = Sampler(torch.device("cpu"), VOCAB)
    logits = torch.tensor([[0.1, 0.3, 0.6, 0.2], [0.9, 0.05, 0.05, 0.0]])

    args = prepare(
        make_req(1, SamplingParams(temperature=0.0)),
        make_req(2, SamplingParams(temperature=0.0)),
    )
    tokens = sampler.sample(logits, args)
    assert tokens.tolist() == [2, 0]


def test_sample_mixed_batch_forces_argmax_on_greedy_rows(monkeypatch):
    import freetoken.engine.sample as sample_mod

    sampler = Sampler(torch.device("cpu"), VOCAB)
    logits = torch.tensor(
        [[0.2, 0.8, 0.0, 0.0],
         [0.1, 0.1, 0.1, 0.7]],
        dtype=torch.float16,
    )
    expected_greedy = torch.argmax(logits, dim=-1).to(torch.int64)

    # the probability path is swapped for a sentinel: only the mask substitution
    # under test, never a real GPU sampling kernel on CPU tensors
    monkeypatch.setattr(sample_mod, "sample_impl", lambda *a: torch.tensor([0, 3]))

    greedy = SamplingParams(temperature=0.0)
    sampled = SamplingParams(temperature=0.8, top_p=0.9)
    args = prepare(make_req(1, greedy), make_req(2, sampled))
    tokens = sampler.sample(logits, args)
    assert tokens.tolist() == [expected_greedy[0].item(), 3]