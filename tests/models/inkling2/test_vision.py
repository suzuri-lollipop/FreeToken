"""Inkling-2 vision tower against an fp32 transcription of the transformers
Inkling2VisionModel: patch embedding, bilinear position-table resample (cross-checked
with F.interpolate), class token, axial rope, LayerScale blocks, gelu merger.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.inkling2.config import VisionConfig
from freetoken.models.inkling2.vision import Inkling2VisionModel

VC = VisionConfig(
    hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4,
    patch_size=4, merge_size=2, posemb_grid_size=4, projection_hidden_size=32,
    projector_bias=False, text_hidden_size=64, layer_norm_eps=1e-5, rope_theta=10000.0,
)
ROWS, COLS = 3, 3  # image tokens; native patch grid 6x6 resampled from the 4x4 table
HD = VC.hidden_size // VC.num_heads


def _reference(sd, pixel_values):
    merge, patch, side = VC.merge_size, VC.patch_size, VC.posemb_grid_size
    gh, gw = ROWS * merge, COLS * merge
    n_tok = ROWS * COLS
    n_patch = n_tok * merge * merge
    # native patches in merge-block order
    patches = pixel_values.reshape(-1, merge, patch, merge, patch, 3).transpose(2, 3)
    h = patches.reshape(n_patch, VC.patch_dim) @ sd["backbone.embeddings.patch_embedding.weight"].T
    # position table resample, cross-checked with F.interpolate in raster order
    table = sd["backbone.embeddings.position_embedding.weight"].reshape(1, side, side, VC.hidden_size).permute(0, 3, 1, 2)
    interp = F.interpolate(table, size=(gh, gw), mode="bilinear", align_corners=False)
    interp = interp.permute(0, 2, 3, 1).reshape(gh * gw, VC.hidden_size)
    # raster -> merge-block order
    order = torch.arange(gh * gw).reshape(gh // merge, merge, gw // merge, merge).permute(0, 2, 1, 3).reshape(-1)
    h = h + interp[order]
    cls = sd["backbone.embeddings.class_embedding"] + sd["backbone.embeddings.cls_position_embedding"]
    h = torch.cat([cls.unsqueeze(0), h], dim=0)
    h = F.layer_norm(h, (VC.hidden_size,), sd["backbone.ln_pre.weight"], sd["backbone.ln_pre.bias"], VC.layer_norm_eps)
    # axial rope: patches 1-indexed (row, col) block-major, cls at (0, 0)
    hpos, wpos = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    shape = (gh // merge, merge, gw // merge, merge)
    hpos = hpos.reshape(shape).transpose(1, 2).flatten() + 1
    wpos = wpos.reshape(shape).transpose(1, 2).flatten() + 1
    pos = torch.zeros(n_patch + 1, 2)
    pos[1:, 0], pos[1:, 1] = hpos, wpos
    spatial = HD // 2
    inv_freq = 1.0 / (VC.rope_theta ** (torch.arange(0, spatial, 2).float() / spatial))
    freqs = pos.unsqueeze(-1) * inv_freq
    angles = torch.cat([freqs[:, 1], freqs[:, 0]], dim=-1).repeat_interleave(2, dim=-1)
    cos, sin = angles.cos(), angles.sin()

    def rope(x):
        xr = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = xr.unbind(-1)
        rot = torch.stack((-x2, x1), -1).flatten(-2)
        return x * cos.unsqueeze(-2) + rot * sin.unsqueeze(-2)

    for i in range(VC.num_layers):
        p = f"backbone.layers.{i}"
        ln = F.layer_norm(h, (VC.hidden_size,), sd[f"{p}.ln_1.weight"], sd[f"{p}.ln_1.bias"], VC.layer_norm_eps)
        q = (ln @ sd[f"{p}.self_attn.q_proj.weight"].T + sd[f"{p}.self_attn.q_proj.bias"]).view(n_patch + 1, VC.num_heads, HD)
        k = (ln @ sd[f"{p}.self_attn.k_proj.weight"].T + sd[f"{p}.self_attn.k_proj.bias"]).view(n_patch + 1, VC.num_heads, HD)
        v = (ln @ sd[f"{p}.self_attn.v_proj.weight"].T + sd[f"{p}.self_attn.v_proj.bias"]).view(n_patch + 1, VC.num_heads, HD)
        q, k = rope(q), rope(k)
        att = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), scale=HD**-0.5
        )
        att = att.squeeze(0).transpose(0, 1).reshape(n_patch + 1, VC.hidden_size)
        att = att @ sd[f"{p}.self_attn.out_proj.weight"].T + sd[f"{p}.self_attn.out_proj.bias"]
        h = h + sd[f"{p}.ls_1_gamma"] * att
        ln = F.layer_norm(h, (VC.hidden_size,), sd[f"{p}.ln_2.weight"], sd[f"{p}.ln_2.bias"], VC.layer_norm_eps)
        mlp = F.gelu(ln @ sd[f"{p}.mlp.c_fc.weight"].T + sd[f"{p}.mlp.c_fc.bias"], approximate="none")
        mlp = mlp @ sd[f"{p}.mlp.c_proj.weight"].T + sd[f"{p}.mlp.c_proj.bias"]
        h = h + sd[f"{p}.ls_2_gamma"] * mlp
    merged = h[1:].reshape(n_tok, merge * merge * VC.hidden_size)
    out = F.gelu(merged @ sd["proj_in.weight"].T, approximate="none")
    return out @ sd["proj_out.weight"].T


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vision tower kernels")
def test_vision_tower_matches_the_reference(monkeypatch):
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    prev_dtype, prev_device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    try:
        tower = Inkling2VisionModel(VC)
        torch.manual_seed(0)
        sd = {k: torch.randn(v.shape) * 0.1 for k, v in tower.state_dict().items()}
        for name in sd:
            if name.endswith(("ln_1.weight", "ln_2.weight", "ln_pre.weight")):
                sd[name] = sd[name].abs() + 0.5
            if name.endswith("gamma"):
                sd[name] = sd[name].abs() * 0.1
        tower.load_state_dict(dict(sd))
        pixel_values = torch.randn(ROWS * COLS, VC.token_patch_size, VC.token_patch_size, 3)
        got = tower.forward(pixel_values, [1, ROWS, COLS])
        want = _reference(sd, pixel_values)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_device)
    assert (got - want).abs().max() < 2e-3, ((got - want).abs().max().item(), want.std().item())


# ------------------------------------------------------------------ tensor-parallel tower
def _tp_forward(rank: int, world: int, port: int):
    """One rank of a real gloo group: slice the full-width weights, encode, diff the reference.

    The collectives are the production ones (TorchDistributedImpl over the default group), so
    a misplaced all-reduce, a double-counted row-parallel bias or a wrong head slice shows up
    as a number that disagrees with the single-process transcription.
    """
    from datetime import timedelta

    import torch.distributed as dist
    from freetoken.distributed import info
    from freetoken.models.inkling2.vision import vision_module_leaf, vision_tp_shard_for

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_default_dtype(torch.float32)
        info._TP_INFO = info.DistributedInfo(rank=0, size=1)
        torch.manual_seed(0)
        full = Inkling2VisionModel(VC)
        sd = {k: torch.randn(v.shape) * 0.1 for k, v in full.state_dict().items()}
        for name in sd:
            if name.endswith(("ln_1.weight", "ln_2.weight", "ln_pre.weight")):
                sd[name] = sd[name].abs() + 0.5
            if name.endswith("gamma"):
                sd[name] = sd[name].abs() * 0.1
        pixel_values = torch.randn(ROWS * COLS, VC.token_patch_size, VC.token_patch_size, 3)
        want = _reference(sd, pixel_values)

        info._TP_INFO = info.DistributedInfo(rank=rank, size=world)
        tower = Inkling2VisionModel(VC)
        shard = vision_tp_shard_for(VC, rank, world)
        cut = {k: shard.tensor(vision_module_leaf("visual." + k), v) for k, v in sd.items()}
        tower.load_state_dict(cut)
        got = tower.forward(pixel_values, [1, ROWS, COLS])
        assert (got - want).abs().max() < 2e-3, ((got - want).abs().max().item(), want.std().item())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_tp_tower_matches_the_reference(world):
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_tp_forward, args=(world, port), nprocs=world, join=True)


def test_processor_patches_and_grid():
    from types import SimpleNamespace

    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.inkling2 import Inkling2MMProcessor

    hf = SimpleNamespace(vision_config=SimpleNamespace(native_patch_size=20, adapter_merge_size=2), image_token_id=200054)
    proc = Inkling2MMProcessor(hf, "", MultimodalConfig())
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.full((48, 80, 3), 128, dtype=np.uint8))
    items = proc.process([img])
    item = items[0]
    # ceil(48/40)=2 rows, ceil(80/40)=2 cols -> 4 tokens of 40x40x3
    assert item.grid_thw == [1, 2, 2]
    assert item.feature.shape == (4, 40, 40, 3)
    # the pad fill is -1 on the 0-255 scale; rescale+normalize maps it to (-1/255 - 0.5) / 0.5
    assert item.feature[2, 8:, 0, 0].min().item() == pytest.approx((-1.0 / 255 - 0.5) / 0.5, abs=1e-2)
    repl = proc.prompt_replacement(item)
    assert repl.full == [200054] * 4
