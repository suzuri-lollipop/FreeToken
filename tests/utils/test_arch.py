"""Architecture gating (utils/arch.py).

The gate functions decide which backends/kernels the engine even considers; a
flipped comparison or a wrong major number silently disables a fast path on the
exact hardware it exists for (or enables a datacenter-only kernel on consumer
Blackwell). The torch capability probe is patched so the comparisons themselves
run anywhere.
"""

from __future__ import annotations

import pytest

import freetoken.utils.arch as arch


@pytest.fixture
def fake_capability(monkeypatch):
    def set_cap(cap):
        monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: cap)

    return set_cap


def test_unsupported_when_cuda_capability_is_unknown(fake_capability):
    fake_capability(None)
    assert arch.is_arch_supported(9, 0) is False
    assert not arch.is_sm90_supported()
    assert not arch.is_sm100_supported()
    assert not arch.is_sm90_family()
    assert not arch.is_sm100_family()


def test_supported_is_open_ended(fake_capability):
    fake_capability((12, 1))
    assert arch.is_arch_supported(9, 0)
    assert arch.is_arch_supported(12, 0)
    # an older-than-requested capability fails even when newer archs pass
    assert not arch.is_arch_supported(13, 0)


def test_supported_minor_bound(fake_capability):
    fake_capability((9, 1))
    assert not arch.is_arch_supported(9, 2)
    assert arch.is_arch_supported(9, 0)


def test_family_checks_are_exact_majors(fake_capability):
    fake_capability((10, 3))
    assert arch.is_sm100_family()
    assert not arch.is_sm90_family()

    # consumer Blackwell (sm_120) must NOT satisfy the datacenter family check
    fake_capability((12, 0))
    assert not arch.is_sm100_family()
    assert not arch.is_sm90_family()


def test_family_checks_use_major_only(fake_capability):
    fake_capability((9, 5))
    assert arch.is_sm90_family()