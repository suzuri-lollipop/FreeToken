"""FTW replay: dropping entries by name before their bytes are read, and the vision-tower presence check."""

import torch

from freetoken.checkpoint.ftw import (
    FTWReader, FTWWriter, ftw_tensor_names, iter_ftw_weights, layer_bank_entry_name,
    load_ftw_banks,
)
from freetoken.models.weight import ftw_lacks_vision, load_weight


def _write_ftw(out_dir, names):
    writer = FTWWriter(str(out_dir))
    tensors = {name: torch.full((4, 8), float(i), dtype=torch.bfloat16) for i, name in enumerate(names)}
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({})
    return tensors


def test_keep_drops_entries_before_their_bytes_are_read(tmp_path, monkeypatch):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight", "model.c.weight"])
    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)
    got = dict(iter_ftw_weights(str(tmp_path), keep=lambda name: not name.startswith("visual.")))
    assert list(got) == ["model.a.weight", "model.c.weight"]
    assert read == ["model.a.weight", "model.c.weight"]
    for name, tensor in got.items():
        assert torch.equal(tensor, tensors[name])


def test_no_keep_replays_every_entry(tmp_path):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight"])
    got = dict(iter_ftw_weights(str(tmp_path)))
    assert list(got) == list(tensors)
    assert torch.equal(got["visual.b.weight"], tensors["visual.b.weight"])


def test_load_weight_text_only_skips_the_tower(tmp_path):
    names = ["model.a.weight", "visual.b.weight", "vision_tower.c.weight"]
    _write_ftw(tmp_path, names)
    cpu = torch.device("cpu")
    assert [n for n, _ in load_weight(str(tmp_path), cpu, include_vision=False)] == ["model.a.weight"]
    assert [n for n, _ in load_weight(str(tmp_path), cpu)] == names


def test_unbuffered_read_matches_the_mmap_fallback(tmp_path):
    """The two FTW read paths have to land identical bytes: the unbuffered one (``os.O_DIRECT`` on
    POSIX, ``FILE_FLAG_NO_BUFFERING`` through the ``moe/host_banks`` seam on Windows) is the
    default and mmap is the fallback. Entries are sized to cover the three job shapes
    ``read_into`` builds: one chunk inside one shard, several chunks, and one chunk spanning
    shards -- plus a padded tail, which is what a short read used to trip over."""
    writer = FTWWriter(str(tmp_path), shard_limit=2 << 20)
    tensors = {
        "model.tail.weight": torch.randn(1000, dtype=torch.bfloat16),  # 2000 B -> one padded chunk
        "model.span.weight": torch.randn(4 << 20, dtype=torch.bfloat16),  # 8 MiB over 4 shards
        "model.chunked.weight": torch.randn(8 << 20, dtype=torch.bfloat16),  # 16 MiB, 4 chunks
    }
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({})

    reader = FTWReader(str(tmp_path))
    assert reader._direct, "the reader must try the unbuffered path on every platform"
    reader.close()

    got = dict(iter_ftw_weights(str(tmp_path), chunk=4 << 20))
    for name, tensor in tensors.items():
        assert torch.equal(got[name], tensor), name

    fallback = FTWReader(str(tmp_path))
    fallback._direct = False
    entry = fallback.tensors["model.chunked.weight"]
    dest = bytearray(-(-entry["nbytes"] // 4096) * 4096)
    fallback.read_into(memoryview(dest), entry)
    fallback.close()
    assert bytes(dest[: entry["nbytes"]]) == got["model.chunked.weight"].view(torch.int16).numpy().tobytes()


def test_ftw_lacks_vision(tmp_path):
    _write_ftw(tmp_path / "text", ["model.a.weight"])
    _write_ftw(tmp_path / "vl", ["model.a.weight", "visual.b.weight"])
    assert ftw_lacks_vision(str(tmp_path / "text"))
    assert not ftw_lacks_vision(str(tmp_path / "vl"))
    assert not ftw_lacks_vision(str(tmp_path))
    assert ftw_tensor_names(str(tmp_path / "vl"), "weight") == ["model.a.weight", "visual.b.weight"]


def _write_bank_ftw(out_dir, L, E, *, flat=False):
    """A minimal per-layer (or flat) bf16 experts_bank FTW: two banks x L layers."""
    writer = FTWWriter(str(out_dir))
    ref = {}
    shapes = {"gate_up": (E, 4, 8), "down": (E, 8, 2)}
    for base, shape in shapes.items():
        if flat:
            t = torch.randn(L * E, *shape[1:], dtype=torch.bfloat16)
            ref[(base, None)] = t
            writer.add_tensor(base, t, kind="experts_bank")
        else:
            for layer in range(L):
                t = torch.randn(shape, dtype=torch.bfloat16)
                ref[(base, layer)] = t
                writer.add_tensor(layer_bank_entry_name(base, layer), t, kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": L})
    return ref


def test_load_ftw_banks_disk_tier(tmp_path, monkeypatch):
    """DISK-labeled per-layer entries map from the shard file instead of being read:
    their entries never go through read_into, the bytes still match a read load, and
    the file-backed banks come back on ExpertBanks.file_banks for prefetch."""
    # the PINNED layers in this test pin after read; CPU-only here
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    L = 3
    ref = _write_bank_ftw(tmp_path, L, E=4)

    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)

    residency = ["disk", "pinned", "disk"]
    banks = load_ftw_banks(str(tmp_path), num_layers=L, layer_residency=residency)

    assert banks.layer_residency == residency
    # only the pinned middle layer's entries were read; the disk layers fault from the file
    assert sorted(read) == sorted(
        layer_bank_entry_name(b, 1) for b in ("gate_up", "down")
    )
    for base in ("gate_up", "down"):
        for layer in range(L):
            assert torch.equal(banks.sources[base][layer], ref[(base, layer)]), (base, layer)
    # file_banks: per role, HostBank exactly on the disk layers
    assert sorted(banks.file_banks) == ["down", "gate_up"]
    for banks_list in banks.file_banks.values():
        assert [b is not None for b in banks_list] == [True, False, True]
        banks_list[0].willneed()  # readahead hook is callable
        assert banks_list[0]._fd is not None
    # a disk bank survives release (pages drop, bytes re-fault from the file)
    bank = banks.file_banks["gate_up"][2]
    bank.release()
    assert torch.equal(banks.sources["gate_up"][2], ref[("gate_up", 2)])


def test_load_ftw_banks_disk_flat_falls_back_to_read(tmp_path, monkeypatch, caplog):
    """A flat bank region cannot be mmapped per layer (it can span shards); those
    layers are read like PAGEABLE ones and no file_banks are reported."""
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    L = 2
    ref = _write_bank_ftw(tmp_path, L, E=4, flat=True)

    import logging

    with caplog.at_level(logging.WARNING, logger="freetoken.checkpoint.ftw"):
        banks = load_ftw_banks(str(tmp_path), num_layers=L, layer_residency=["disk", "pinned"])
    assert banks.file_banks is None
    assert any("flat FTW region" in r.message for r in caplog.records)
    for base in ("gate_up", "down"):
        for layer in range(L):
            assert torch.equal(banks.sources[base][layer], ref[(base, None)][layer * 4:(layer + 1) * 4])
    assert banks.layer_residency == ["disk", "pinned"]


def test_file_backed_bank_content_matches_plain_load(tmp_path, monkeypatch):
    """The disk tier must serve the same bytes the pinned path reads: a full-disk load
    and a plain pinned load of the same FTW agree entry by entry."""
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    L = 2
    ref = _write_bank_ftw(tmp_path, L, E=4)
    plain = load_ftw_banks(str(tmp_path), num_layers=L)
    disk = load_ftw_banks(
        str(tmp_path), num_layers=L, layer_residency=["disk"] * L
    )
    for base in ("gate_up", "down"):
        for layer in range(L):
            assert torch.equal(plain.sources[base][layer], disk.sources[base][layer])
            assert torch.equal(disk.sources[base][layer], ref[(base, layer)])
    # alphas (none here) aside, the disk load touched no read path for the banks
    # (covered by the spy test above); here just the shape contract:
    assert disk.sources["gate_up"][0].shape == (4, 4, 8)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


def test_load_ftw_banks_hot_experts(tmp_path, monkeypatch):
    """hot_experts on a DISK layer gather the listed experts' rows into compact
    pinned banks (ExpertBanks.hot) whose bytes match the file, while the layer's
    cold rows still come from the mmap and the entry is never fully read."""
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    L = 3
    ref = _write_bank_ftw(tmp_path, L, E=4)

    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        if entry.get("name"):
            read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)

    residency = ["disk", "pinned", "disk"]
    hot = {0: [2, 0], 2: [3]}
    # the real boot loads banks under torch.inference_mode (thread-local: the read
    # pool is outside it); exercise that exact regime
    with torch.inference_mode():
        banks = load_ftw_banks(
            str(tmp_path), num_layers=L, layer_residency=residency, hot_experts=hot
        )

    # the pinned layer still reads whole; disk layers only read hot windows
    assert layer_bank_entry_name("gate_up", 1) in read
    assert layer_bank_entry_name("down", 1) in read
    for name in read:
        if "#L" in str(name):
            continue
    # no disk-layer entry was read whole
    assert layer_bank_entry_name("gate_up", 0) not in read
    assert layer_bank_entry_name("gate_up", 2) not in read
    assert layer_bank_entry_name("down", 0) not in read
    assert layer_bank_entry_name("down", 2) not in read

    # hot compacts: one per role per hot layer, rows in hot-list order
    assert sorted(banks.hot) == [0, 2]
    for layer, roles in banks.hot.items():
        assert sorted(roles) == ["down", "gate_up"]
        for role, (compact, slot) in roles.items():
            want = hot[layer]
            assert compact.shape[0] == len(want)
            assert slot == {e: i for i, e in enumerate(want)}
            for i, e in enumerate(want):
                assert torch.equal(compact[i], ref[(role, layer)][e]), (role, layer, e)

    # cold rows of the disk layers still fault from the file and match
    for base in ("gate_up", "down"):
        for layer in (0, 2):
            assert torch.equal(banks.sources[base][layer], ref[(base, layer)]), (base, layer)
    # file banks unchanged by the hot gather
    assert sorted(banks.file_banks) == ["down", "gate_up"]
