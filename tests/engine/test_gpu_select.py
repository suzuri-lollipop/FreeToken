"""--gpu spec parsing and resolution (gpu_select.py), CPU-only.

Every wrong answer here is silent and expensive: resolving "2" against a
CUDA_VISIBLE_DEVICES quota in the wrong namespace binds a worker to the wrong
card, and a typo'd --gpu that half-matches a UUID prefix burns a different GPU
than the one the operator named. The NVML probe is faked at the module boundary
so the parse/resolve arithmetic itself runs anywhere.
"""

from __future__ import annotations

import argparse

import pytest

import freetoken.gpu_select as gs


UUID_A = "GPU-11111111-2222-3333-4444-555555555555"
UUID_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture(autouse=True)
def _clean_assignment(monkeypatch):
    monkeypatch.setattr(gs, "_assigned_physical", None)
    monkeypatch.setattr(gs, "_assigned_visible", None)


def test_is_gpu_uuid_and_index():
    assert gs.is_gpu_uuid(UUID_A)
    assert gs.is_gpu_uuid("gpu-abc")  # case-insensitive, prefix only
    assert not gs.is_gpu_uuid("2")
    assert gs.is_gpu_index("17")
    assert not gs.is_gpu_index("-1")
    # str.isdigit() would accept these Unicode digits; index entries must be ASCII
    assert not gs.is_gpu_index("\u0662")  # ARABIC-INDIC DIGIT TWO
    assert not gs.is_gpu_index("2\u00b2")


def test_canonical_uppercases_uuid_prefix():
    assert gs._canonical("gpu-abc123") == "GPU-abc123"
    assert gs._canonical(UUID_A) == UUID_A
    assert gs._canonical("3") == "3"


def test_canonical_rejects_neither_uuid_nor_index():
    with pytest.raises(ValueError, match="neither a GPU UUID"):
        gs._canonical("gxp-1")
    with pytest.raises(ValueError, match="neither a GPU UUID"):
        gs._canonical("-3")


def test_parse_gpu_spec_splits_and_requires_homogeneous_entries():
    assert gs.parse_gpu_spec(" 0 , 1 ") == ("0", "1")
    assert gs.parse_gpu_spec(UUID_A) == (UUID_A,)
    with pytest.raises(ValueError, match="--gpu needs at least one GPU"):
        gs.parse_gpu_spec("")
    with pytest.raises(ValueError, match="--gpu needs at least one GPU"):
        gs.parse_gpu_spec("  , ")
    with pytest.raises(ValueError, match="all UUIDs or all indices"):
        gs.parse_gpu_spec(f"0,{UUID_A}")


def test_gpu_arg_and_single_gpu_arg_are_argparse_types():
    assert gs.gpu_arg("0") == ("0",)
    assert gs.single_gpu_arg("1") == "1"
    with pytest.raises(argparse.ArgumentTypeError):
        gs.gpu_arg("")
    with pytest.raises(argparse.ArgumentTypeError, match="exactly one GPU"):
        gs.single_gpu_arg("0,1")


def test_match_uuid_requires_a_unique_prefix():
    uuids = [UUID_A, UUID_B]
    assert gs._match_uuid("gpu-11111111", uuids, "here") == UUID_A
    with pytest.raises(ValueError, match="not found or not a unique prefix"):
        gs._match_uuid("gpu-", uuids, "here")  # matches both
    with pytest.raises(ValueError, match="not found or not a unique prefix"):
        gs._match_uuid("gpu-zzzz", uuids, "here")


def _patch_nvml(monkeypatch, uuids):
    monkeypatch.setattr(gs, "_nvml_uuids", lambda: uuids)


def test_resolve_indices_in_nvml_order(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gs.resolve_gpu_uuids(["1", "0"]) == (UUID_B, UUID_A)


def test_resolve_uuid_prefixes(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gs.resolve_gpu_uuids(["gpu-aaaaaaaa"]) == (UUID_B,)


def test_resolve_rejects_out_of_range_index_without_preset(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="only 2 GPU"):
        gs.resolve_gpu_uuids(["2"])


def test_resolve_returns_none_when_nvml_is_unavailable(monkeypatch):
    _patch_nvml(monkeypatch, None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gs.resolve_gpu_uuids(["0"]) is None


def test_resolve_rejects_duplicate_entries(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="same GPU appears twice"):
        gs.resolve_gpu_uuids(["0", "0"])


def test_resolve_ignores_empty_invisible_entries(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,")
    assert gs.resolve_gpu_uuids(["0"]) == (UUID_B,)


def test_resolve_index_counts_within_the_preset_quota(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert gs.resolve_gpu_uuids(["1"]) == (UUID_B,)
    with pytest.raises(ValueError, match="only 2 GPU\\(s\\) are visible"):
        gs.resolve_gpu_uuids(["2"])


def test_resolve_uuid_spec_against_index_preset_fails(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="give --gpu as an index"):
        gs.resolve_gpu_uuids(["gpu-11111111"])


def test_resolve_uuid_against_uuid_preset_within_and_outside(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{UUID_A}")
    assert gs.resolve_gpu_uuids(["GPU-11111111"]) == (UUID_A,)
    # a short spec that the preset entry itself prefixes resolves in both directions
    assert gs.resolve_gpu_uuids(["GPU-1111"]) == (UUID_A,)
    with pytest.raises(ValueError, match="not one of the GPUs visible"):
        gs.resolve_gpu_uuids(["GPU-aaaaaaaa"])


def test_preset_entry_rejects_mig_entries(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-xxx")
    with pytest.raises(ValueError, match="cannot resolve CUDA_VISIBLE_DEVICES entry"):
        gs.resolve_gpu_uuids(["0"])


def test_set_assigned_gpu_publish_once_semantics():
    gs.set_assigned_gpu("2")
    assert (gs._assigned_physical, gs._assigned_visible) == (None, 2)
    gs.set_assigned_gpu("2")  # agreeing second call is fine
    with pytest.raises(RuntimeError, match="called twice"):
        gs.set_assigned_gpu("3")


def test_set_assigned_gpu_uuid_lands_in_physical(monkeypatch):
    monkeypatch.setattr(gs, "_visible_of_physical", lambda uuid: None)
    gs.set_assigned_gpu(UUID_A)
    assert (gs._assigned_physical, gs._assigned_visible) == (UUID_A, None)
    assert gs.assigned_visible_gpu() is None  # not yet bound


def test_assign_gpu_resolves_then_publishes(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    gs.assign_gpu(None)  # flag not given -> no-op
    assert (gs._assigned_physical, gs._assigned_visible) == (None, None)

    gs.assign_gpu("1")
    assert (gs._assigned_physical, gs._assigned_visible) == (UUID_B, None)


def test_format_gpu_uuid():
    assert gs.format_gpu_uuid(None) is None
    assert gs.format_gpu_uuid("abc-123") == "GPU-abc-123"


def test_physical_gpu_count(monkeypatch):
    _patch_nvml(monkeypatch, [UUID_A, UUID_B])
    assert gs.physical_gpu_count() == 2
    _patch_nvml(monkeypatch, None)
    assert gs.physical_gpu_count() is None