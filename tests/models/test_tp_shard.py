"""The shared TP slicer (models/tp_shard.py) on synthetic tensors.

Every family's reader funnels through this, so its rules are pinned here directly instead of
only through a model's reader: which leaf splits on which axis, how a block scale narrows by
its own ratio, the q | k | v runs inside one GDN tensor, the rounded-up vocab shard, and the
leaves that must stay replicated because a divergent copy would silently change routing or
block selection on one rank.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.distributed import DistributedInfo, info
from freetoken.models.tp_shard import LEAF_AXIS, TpShard, module_leaf, tp_shard_for

H = 128  # hidden_size
QH, KVH, HD = 8, 4, 64  # attention heads (q_proj is 2x wide for its per-head gate) and head dim
KH, VH, GD = 2, 4, 32  # GDN key / value heads and head dim
I = 256  # MLP width (dense or shared expert)
VOCAB = 11  # deliberately not divisible by any rank count


def _config(**overrides):
    group = SimpleNamespace(
        num_key_heads=KH, num_value_heads=VH, key_head_dim=GD, value_head_dim=GD
    )
    cfg = {
        "head_dim": HD,
        "num_qo_heads": QH,
        "num_kv_heads": KVH,
        "hidden_size": H,
        "vocab_size": VOCAB,
        "intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "moe_enabled": False,
        "linear_attention_group": lambda: group,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _shard(rank: int, world: int, **config) -> TpShard:
    return tp_shard_for(_config(**config), rank, world)


def _index(rows: int, cols: int = H) -> torch.Tensor:
    """A tensor whose every element is its own row's value, so a slice shows where it came from."""
    return torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)


def test_row_and_column_leaves_split_on_their_own_axis():
    q, o = _index(QH * HD * 2), _index(H, QH * HD)
    r0, r1 = _shard(0, 2), _shard(1, 2)
    assert torch.equal(r0.tensor("q_proj", q), q[: QH * HD])
    assert torch.equal(r1.tensor("q_proj", q), q[QH * HD :])
    assert torch.equal(r0.tensor("o_proj", o), o[:, : QH * HD // 2])
    assert torch.equal(r1.tensor("o_proj", o), o[:, QH * HD // 2 :])


def test_replicated_leaves_are_left_alone():
    router, index_qk, key = _index(64), _index(5 * 64), _index(KVH * HD * 2, I)
    rank = _shard(1, 2)
    for leaf, tensor in (("gate", router), ("index_qk_proj", index_qk), ("key_proj", key)):
        assert torch.equal(rank.tensor(leaf, tensor), tensor)


def test_block_scale_narrows_by_its_own_ratio():
    """A 128-block scale table loses exactly the blocks its weight loses."""
    weight = _index(QH * HD * 2)  # 1024 rows x 128 cols
    scale = torch.arange((weight.shape[0] // 128) * (weight.shape[1] // 128), dtype=torch.float32)
    scale = scale.reshape(weight.shape[0] // 128, weight.shape[1] // 128)
    r0, r1 = _shard(0, 2), _shard(1, 2)
    assert r0.tensor("q_proj", scale).shape == (4, 1)
    assert torch.equal(r0.tensor("q_proj", scale), scale[:4])
    assert torch.equal(r1.tensor("q_proj", scale), scale[4:])


def test_per_tensor_scales_and_vectors_pass_through():
    rank = _shard(1, 2)
    scalar = torch.tensor(0.5)
    assert torch.equal(rank.tensor("q_proj", scalar), scalar)
    per_row = torch.rand(QH * HD * 2, 1)  # a [rows, 1] column must not be cut on axis 1
    assert torch.equal(rank.tensor("o_proj", per_row), per_row)


def test_gdn_in_proj_keeps_one_head_from_every_run():
    key, value = KH * GD, VH * GD
    rows = 2 * key + value

    def rank_rows(t: torch.Tensor, r: int) -> torch.Tensor:
        # the run layout under test, spelled out: this rank's slice of q, then k, then v
        return torch.cat(
            [
                t[r * key // 2 : (r + 1) * key // 2],
                t[key + r * key // 2 : key + (r + 1) * key // 2],
                t[2 * key + r * value // 2 : 2 * key + (r + 1) * value // 2],
            ]
        )

    r0, r1 = _shard(0, 2), _shard(1, 2)
    qkv = _index(rows)
    assert torch.equal(r0.tensor("in_proj_qkv", qkv), rank_rows(qkv, 0))
    assert torch.equal(r1.tensor("in_proj_qkv", qkv), rank_rows(qkv, 1))
    # conv1d carries the same runs on dim 0, with its trailing kernel dims untouched
    conv = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 1, 4)
    cut = r1.tensor("linear_attn.conv1d", conv)
    assert cut.shape == (rows // 2, 1, 4)
    assert torch.equal(cut, rank_rows(conv, 1))


def test_the_ple_conv1d_stays_replicated_under_the_gdn_rule():
    """qwen4's PLE conv is depthwise over hc*hidden; the row count can equal the GDN conv width."""
    key, value = KH * GD, VH * GD
    conv = _index(2 * key + value)
    assert torch.equal(_shard(1, 2).tensor(module_leaf("model.layers.1.ple.conv1d.weight"), conv), conv)


def test_vocab_shard_is_rounded_up_and_zero_padded():
    emb = _index(VOCAB)
    r0, r1 = _shard(0, 2), _shard(1, 2)
    assert r0.tensor("embed_tokens", emb).shape == (6, H)
    assert torch.equal(r0.tensor("embed_tokens", emb), emb[:6])
    assert torch.equal(r1.tensor("embed_tokens", emb)[:5], emb[6:11])
    assert torch.equal(r1.tensor("lm_head", emb)[5:], torch.zeros(1, H))


def test_a_run_too_short_to_split_is_left_whole():
    """Dropping the remainder silently is worse than the strict-load error this causes."""
    rows = _index(6)
    assert torch.equal(_shard(1, 4).tensor("gate_proj", rows), rows)


def test_gdn_heads_must_divide():
    from freetoken.models.qwen3_5_moe.gdn import gdn_local_dims

    assert gdn_local_dims(KH, VH, GD, GD, 2) == (1, 2, GD, 2 * GD, 4 * GD)
    with pytest.raises(ValueError, match="value heads to divide"):
        gdn_local_dims(3, VH, GD, GD, 2)


def test_module_leaf_names_the_module_of_a_key():
    assert module_leaf("model.embed_tokens.weight") == "embed_tokens"
    assert module_leaf("model.layers.0.linear_attn.conv1d.weight") == "linear_attn.conv1d"
    assert module_leaf("model.layers.1.ple.conv1d.weight") == "ple.conv1d"
    assert module_leaf("model.layers.0.linear_attn.A_log") == "A_log"
    assert module_leaf("model.layers.1.self_attn.o_proj.weight") == "o_proj"
    assert module_leaf("model.layers.0.mlp.gate.weight") == "gate"
    assert module_leaf("model.layers.0.mlp.down_proj.weight_scale_inv") == "down_proj"
    assert module_leaf("model.layers.0.ple.ple_embedding.ngram_embedding") == "ngram_embedding"
    # leaves outside LEAF_AXIS are replicated, so the slicer must not touch them
    for name in ("model.layers.0.mlp.gate.weight", "model.layers.1.self_attn.indexer.index_qk_proj.weight",
                 "model.layers.1.attn_hyper_connection.hc_norm.weight", "model.layers.0.ple.key_proj.weight",
                 "model.layers.1.ple.conv1d.weight"):
        assert module_leaf(name) not in LEAF_AXIS, name


def test_part_cuts_every_role_and_leaves_a_replicated_leaf_alone():
    rank = _shard(1, 2)
    weight = _index(QH * HD * 2)
    part = rank.part(
        "q_proj",
        {
            "weight": weight,
            "weight_global": weight[:, 0].clone(),  # one value per output row
            "input_scale": torch.tensor(1.0),
        },
    )
    assert torch.equal(part["weight"], weight[QH * HD :])
    assert part["weight_global"].shape == (QH * HD,)
    assert torch.equal(part["input_scale"], torch.tensor(1.0))
    untouched = {"weight": _index(64)}
    assert rank.part("gate", untouched) is untouched


def test_tp_shard_is_inert_on_a_single_rank(monkeypatch):
    from freetoken.models.tp_shard import tp_shard

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    assert tp_shard(_config()) is None
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(1, 2))
    assert tp_shard(_config()).world == 2



def test_a_cut_shard_owns_its_bytes():
    """The shard must not keep the unsharded checkpoint tensor resident.

    The engine adopts loaded tensors by assignment, so a narrow() view would pin the whole
    source storage for the process lifetime: on a 2-rank run every weight stranded its
    other half (~2 GiB of dead VRAM per rank on Qwen3.8-Flash-Next), which the MoE slot
    cache could otherwise spend on expert residency.
    """
    src = _index(QH * HD * 2)
    out = _shard(0, 2).tensor("q_proj", src)
    assert torch.equal(out, src[: QH * HD])
    assert out.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
    assert out.untyped_storage().size() == out.numel() * out.element_size()


def test_an_uncut_leaf_and_a_padded_vocab_shard_stay_exact():
    """No copy for a leaf this rank keeps whole, and the rounded-up vocab shard still owns
    its (padded) bytes."""
    replicated = _index(64)
    assert _shard(0, 2).tensor("gate", replicated) is replicated
    vocab = _index(VOCAB, H)
    for rank in (0, 1, 2):
        piece = _shard(rank, 3).tensor("embed_tokens", vocab)
        assert piece.untyped_storage().size() == piece.numel() * piece.element_size()
