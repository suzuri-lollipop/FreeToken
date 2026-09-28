"""Numeric helpers used across the engine (utils/misc.py).

These are checked against plain arithmetic, not against each other: a mistyped
div_even would silently change head counts and pool geometries, which is exactly
the class of bug a unit test must still catch.
"""

from __future__ import annotations

import pytest

from freetoken.utils.misc import align_ceil, align_down, div_ceil, div_even, mem_GB


def test_div_even_exact_division():
    assert div_even(16, 4) == 4
    assert div_even(7, 1) == 7


def test_div_even_rejects_inexact_division():
    with pytest.raises(AssertionError, match="must be divisible"):
        div_even(7, 3)


def test_div_even_replication_head_map():
    # b > a with b % a == 0: every KV head is replicated -> 1 logical head
    assert div_even(4, 8, allow_replicate=True) == 1


def test_div_even_replication_requires_divisibility():
    with pytest.raises(AssertionError):
        div_even(4, 6, allow_replicate=True)


def test_div_even_replicate_flag_off_keeps_the_assertion():
    with pytest.raises(AssertionError):
        div_even(4, 8)


def test_div_ceil_rounds_up():
    assert div_ceil(7, 3) == 3
    assert div_ceil(6, 3) == 2
    assert div_ceil(0, 3) == 0


def test_align_ceil_and_down():
    assert align_ceil(5, 8) == 8
    assert align_ceil(8, 8) == 8
    assert align_ceil(0, 8) == 0
    assert align_down(5, 8) == 0
    assert align_down(16, 8) == 16
    assert align_down(0, 8) == 0


def test_mem_GB_formats_bytes():
    assert mem_GB(2 * 1024**3) == "2.00 GiB"
    assert mem_GB(0) == "0.00 GiB"