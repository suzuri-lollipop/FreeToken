"""HostBank's file backing (the MoE disk tier): fault-on-touch views over a shard file.

CPU-only: covers the mmap window math (aligned + head-padded), the fault/refault cycle
through ``release()``, and the DISK settle label. pin() needs CUDA and is covered by the
GPU-marked offload tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.moe.host_banks import HostBank, HostResidency, _settle

BLK = 4096


def _write_pattern(path, nbytes: int) -> bytes:
    data = np.arange(nbytes, dtype=np.uint8).tobytes()
    with open(path, "wb") as f:
        f.write(data)
    return data


def test_file_backed_roundtrip_aligned(tmp_path):
    path = str(tmp_path / "shard.bin")
    data = _write_pattern(path, 3 * BLK)
    bank = HostBank((3 * BLK,), torch.uint8, backing="file", file=path, offset=0)
    assert bank._fd is not None
    assert bank.addr % BLK == 0
    assert torch.equal(bank.tensor, torch.frombuffer(bytearray(data), dtype=torch.uint8))


def test_file_backed_roundtrip_head_padded(tmp_path):
    """A window starting mid-page (flat-region style): the view skips the head bytes."""
    path = str(tmp_path / "shard.bin")
    data = _write_pattern(path, 3 * BLK)
    head = 1000  # odd on purpose: 1-byte elements tolerate any head
    nbytes = 2 * BLK + 17
    bank = HostBank((nbytes,), torch.uint8, backing="file", file=path, offset=head)
    assert bank.addr % BLK == head
    assert torch.equal(bank.tensor, torch.frombuffer(bytearray(data[head:head + nbytes]), dtype=torch.uint8))


def test_file_backed_roundtrip_multibyte_dtype(tmp_path):
    path = str(tmp_path / "shard.bin")
    t = torch.arange(2048, dtype=torch.float16)
    with open(path, "wb") as f:
        f.write(t.numpy().tobytes())
    bank = HostBank((2048,), torch.float16, backing="file", file=path)
    assert torch.equal(bank.tensor, t)


def test_file_backed_release_refaults_from_file(tmp_path):
    """MADV_DONTNEED on the private mapping drops the pages; the bytes come back from
    the file on the next touch (the disk tier's release-to-file semantics)."""
    path = str(tmp_path / "shard.bin")
    data = _write_pattern(path, 2 * BLK)
    bank = HostBank((2 * BLK,), torch.uint8, backing="file", file=path)
    bank.tensor[0:8] = torch.full((8,), 255, dtype=torch.uint8)  # private COW page
    bank.release()
    # the private copy is gone and the file bytes are back
    assert torch.equal(bank.tensor, torch.frombuffer(bytearray(data), dtype=torch.uint8))


def test_file_backed_willneed_and_noop_elsewhere(tmp_path):
    path = str(tmp_path / "shard.bin")
    data = _write_pattern(path, BLK)
    bank = HostBank((BLK,), torch.uint8, backing="file", file=path)
    bank.willneed()  # posix_fadvise WILLNEED: never raises, pages may arrive
    anon = HostBank((BLK,), torch.uint8)
    anon.willneed()  # no fd: no-op, not an error


def test_settle_disk_is_a_noop(tmp_path):
    path = str(tmp_path / "shard.bin")
    _write_pattern(path, BLK)
    bank = HostBank((BLK,), torch.uint8, backing="file", file=path)
    _settle(bank, HostResidency.DISK.value)
    assert bank.residency is HostResidency.PAGEABLE
    assert not bank._pinned and not bank._locked


def test_residency_enum_has_disk():
    assert HostResidency.DISK.value == "disk"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
