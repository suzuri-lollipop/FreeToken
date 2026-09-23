"""CUDA graph batch-size candidate selection (engine/graph.py::_determine_cuda_graph_bs).

The default list must cover every decode batch size natively at small bs: a size
missing from the capture list pads the batch up to the next captured graph, and
the padded dummy row's work is paid while its token is thrown away (conc-3 on the
old [1, 2, 4] list ran the bs=4 graph and lost ~25% per-user throughput).
"""

from freetoken.engine.graph import _determine_cuda_graph_bs

GB = 1 << 30


def test_small_max_covers_every_batch_size_natively():
    # max_running_requests=4 (this deployment): no padding gap at bs=3
    assert _determine_cuda_graph_bs(None, 4, 3 * GB) == [1, 2, 3, 4]
    assert _determine_cuda_graph_bs(None, 1, 3 * GB) == [1]
    assert _determine_cuda_graph_bs(None, 8, 3 * GB) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_large_max_is_dense_below_eight_then_strides_by_eight():
    got = _determine_cuda_graph_bs(None, 160, 3 * GB)
    assert got[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert got[8:] == list(range(16, 161, 8))
    assert 3 in got and 160 in got


def test_h200_default_max_uses_the_same_shape():
    got = _determine_cuda_graph_bs(None, None, 90 * GB)  # >80GB free -> max 256
    assert got[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert got[-1] == 256
    assert 9 not in got  # 9..15 pad to 16, as before


def test_explicit_list_and_disabled_cases_are_untouched():
    assert _determine_cuda_graph_bs([1, 4], None, 3 * GB) == [1, 4]
    assert _determine_cuda_graph_bs(None, 0, 3 * GB) == []
    # a small-machine default without an explicit max still caps by memory branch
    got = _determine_cuda_graph_bs(None, None, 3 * GB)
    assert got[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
