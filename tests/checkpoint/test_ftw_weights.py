"""FTW replay: dropping entries by name before their bytes are read, and the vision-tower presence check."""

import torch

from freetoken.checkpoint.ftw import FTWReader, FTWWriter, ftw_tensor_names, iter_ftw_weights
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
