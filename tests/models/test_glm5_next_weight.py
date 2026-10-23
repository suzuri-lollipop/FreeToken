"""glm5_next (GLM-5.3-Flash) NVFP4 expert ingestion, both checkpoint dialects.

Synthetic multimodal-wrapper checkpoints (``model.language_model.*``) guard
``nvfp4_expert_spec``'s dialect selection and the shared ``iter_nvfp4_expert_pieces``
reader (Strata's tools/nvfp4_verify.py role for this engine): bit-exact packed / scale
pieces, the global-scale conversion (ModelOpt cast, compressed-tensors reciprocal), the
dense-prefix and trailing-MTP skips, and the fp16 range guard in nvfp4_banks.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.glm5_next.weight import nvfp4_expert_spec
from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces

H, I = 32, 16  # hidden, moe intermediate (both % 16 for the block scales)
LAYERS, DENSE, E = 3, 1, 2  # layers, leading dense layers, experts per layer
MTP = LAYERS  # the trailing MTP layer's index; its experts must be skipped

_PROJ_GLOBAL = {"gate_proj": 1e-3, "up_proj": 2e-3, "down_proj": 5e-4}

_MODELOPT_QUANT = {"quant_algo": "NVFP4", "quant_method": "modelopt"}
_CT_QUANT = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}

_ROLES = {"gate", "gate_scale", "gate_global", "up", "up_scale", "up_global",
          "down", "down_scale", "down_global"}


def _raw(dialect: str, *, seed: int = 0, global_scale: float | None = None) -> dict[str, torch.Tensor]:
    """Every expert tensor of a tiny GLM-5.3-layout checkpoint, including the
    dense-prefix (layer 0) and trailing-MTP (layer ``MTP``) rows the reader must skip."""
    g = torch.Generator().manual_seed(seed)
    raw: dict[str, torch.Tensor] = {}
    for layer in (*range(LAYERS), MTP):
        for e in range(E):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                rows, k = (H, I) if proj == "down_proj" else (I, H)
                base = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
                packed = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, generator=g)
                scale = (torch.rand((rows, k // 16), generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
                glob = _PROJ_GLOBAL[proj] if global_scale is None else global_scale
                if dialect == "modelopt":
                    raw[f"{base}.weight"] = packed
                    raw[f"{base}.weight_scale"] = scale
                    raw[f"{base}.weight_scale_2"] = torch.full((), glob, dtype=torch.float32)
                else:
                    raw[f"{base}.weight_packed"] = packed
                    raw[f"{base}.weight_scale"] = scale
                    # the CT dialect stores the quant-side global; ingest reciprocals it
                    raw[f"{base}.weight_global_scale"] = torch.full((), 1.0 / glob, dtype=torch.float32)
    return raw


def _folder(tmp_path, raw) -> str:
    save_file(raw, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def _config() -> SimpleNamespace:
    return SimpleNamespace(num_layers=LAYERS, first_k_dense_replace=DENSE, num_experts=E)


def _select(monkeypatch, quant: dict) -> None:
    """Serve ``quant`` as the checkpoint's quantization_config, as RawConfigShim would."""
    import freetoken.models.glm5_next.weight as weight_mod

    monkeypatch.setattr(
        weight_mod, "cached_load_hf_config",
        lambda path: SimpleNamespace(quantization_config=quant),
    )


def _pieces_by_expert(pieces) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    out = {}
    for bank_layer, e0, e1, piece in pieces:
        assert e1 == e0 + 1
        out[(bank_layer, e0)] = piece
    return out


def test_modelopt_dialect_pieces_are_bit_exact(tmp_path, monkeypatch):
    _select(monkeypatch, _MODELOPT_QUANT)
    raw = _raw("modelopt")
    folder = _folder(tmp_path, raw)

    spec = nvfp4_expert_spec(folder, _config())
    assert spec.desc == "GLM-5.3 NVFP4 experts" and spec.global_reciprocal is False
    by_expert = _pieces_by_expert(iter_nvfp4_expert_pieces(folder, _config(), spec, primary=False))

    # The dense prefix (layer 0) and the trailing MTP layer feed no bank; the two MoE
    # layers land on bank layers 0 and 1 with all E experts each.
    assert set(by_expert) == {(b, e) for b in range(LAYERS - DENSE) for e in range(E)}
    piece = by_expert[(0, 1)]  # checkpoint layer 1, expert 1
    assert set(piece) == _ROLES
    base = f"model.language_model.layers.1.mlp.experts.1"
    for proj, role in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        assert torch.equal(piece[role][0], raw[f"{base}.{proj}.weight"])
        assert torch.equal(piece[f"{role}_scale"][0], raw[f"{base}.{proj}.weight_scale"])
        expected = raw[f"{base}.{proj}.weight_scale_2"].reshape(-1).to(torch.float16)
        assert piece[f"{role}_global"].dtype is torch.float16
        assert torch.equal(piece[f"{role}_global"].reshape(-1), expected)
    # per-projection globals stay separate (no cross-wiring, no folding into weights)
    assert piece["gate_global"].item() != piece["up_global"].item() != piece["down_global"].item()


def test_ct_dialect_reciprocal_is_bit_exact(tmp_path, monkeypatch):
    _select(monkeypatch, _CT_QUANT)
    raw = _raw("ct")
    folder = _folder(tmp_path, raw)

    spec = nvfp4_expert_spec(folder, _config())
    assert spec.desc.endswith("(compressed-tensors)") and spec.global_reciprocal is True
    by_expert = _pieces_by_expert(iter_nvfp4_expert_pieces(folder, _config(), spec, primary=False))
    assert set(by_expert) == {(b, e) for b in range(LAYERS - DENSE) for e in range(E)}

    piece = by_expert[(1, 0)]  # checkpoint layer 2, expert 0
    base = f"model.language_model.layers.2.mlp.experts.0"
    for proj, role in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        assert torch.equal(piece[role][0], raw[f"{base}.{proj}.weight_packed"])
        assert torch.equal(piece[f"{role}_scale"][0], raw[f"{base}.{proj}.weight_scale"])
        expected = (1.0 / raw[f"{base}.{proj}.weight_global_scale"].float()).reshape(-1).to(torch.float16)
        assert torch.equal(piece[f"{role}_global"].reshape(-1), expected)


def test_missing_expert_tensor_is_refused(tmp_path, monkeypatch):
    _select(monkeypatch, _MODELOPT_QUANT)
    raw = _raw("modelopt")
    del raw["model.language_model.layers.2.mlp.experts.1.down_proj.weight_scale_2"]
    folder = _folder(tmp_path, raw)
    spec = nvfp4_expert_spec(folder, _config())
    expected = (LAYERS - DENSE) * E * 9
    with pytest.raises(ValueError, match=f"expected {expected}"):
        list(iter_nvfp4_expert_pieces(folder, _config(), spec, primary=False))


@pytest.mark.parametrize(
    "dialect, global_scale",
    [
        ("modelopt", 1e30),   # fp32 -> fp16 overflow
        ("modelopt", 1e-9),   # flushes to zero in fp16
        ("ct", 1e-30),        # reciprocal overflows fp16
    ],
)
def test_global_scale_out_of_fp16_range_is_refused(tmp_path, monkeypatch, dialect, global_scale):
    _select(monkeypatch, _MODELOPT_QUANT if dialect == "modelopt" else _CT_QUANT)
    folder = _folder(tmp_path, _raw(dialect, global_scale=global_scale))
    spec = nvfp4_expert_spec(folder, _config())
    with pytest.raises(ValueError, match="fp16"):
        list(iter_nvfp4_expert_pieces(folder, _config(), spec, primary=False))


def test_real_world_subnormal_global_scale_is_accepted(tmp_path, monkeypatch):
    """nvidia's ModelOpt exports carry weight_scale_2 down to ~2.2e-5 (fp16 subnormal);
    the guard must not refuse those -- degraded precision, but the checkpoints are real."""
    _select(monkeypatch, _MODELOPT_QUANT)
    raw = _raw("modelopt", global_scale=2.17e-5)
    folder = _folder(tmp_path, raw)
    spec = nvfp4_expert_spec(folder, _config())
    by_expert = _pieces_by_expert(iter_nvfp4_expert_pieces(folder, _config(), spec, primary=False))
    got = by_expert[(0, 0)]["gate_global"].reshape(-1)[0]
    assert got != 0 and got.dtype is torch.float16
    assert abs(float(got) - 2.17e-5) / 2.17e-5 < 0.01  # subnormal keeps ~1% precision
