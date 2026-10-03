"""HF config/tokenizer plumbing (utils/hf.py) with stubbed backends, no network.

The functions under test decide what a checkpoint carries without instantiating
HF modeling code: stop-token unions, tool-call anchors, sampling defaults and
the raw-config fallback shim. Wrong answers here do not raise - a model just
runs to max_tokens, misses its tool anchor, or ignores its recommended
sampling - so the tests pin the documented contracts with fake tokenizers and
configs in place of the Hub.
"""

from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import pytest

import freetoken.utils.hf as hf


class StubTokenizer:
    def __init__(self, eos: int | None = 1, encoded: dict[str, list[int]] | None = None):
        self.eos_token_id = eos
        self._encoded = encoded or {}

    def encode(self, text, add_special_tokens=False):
        return self._encoded[text]


def _patch_generation_config(monkeypatch, gc):
    monkeypatch.setattr(hf, "GenerationConfig", SimpleNamespace(from_pretrained=lambda path: gc))


def test_load_eos_token_ids_unions_tokenizer_and_generation_config(monkeypatch):
    _patch_generation_config(monkeypatch, SimpleNamespace(eos_token_id=[1, 151645, 2]))
    assert hf.load_eos_token_ids("model", StubTokenizer(eos=1)) == {1, 151645, 2}


def test_load_eos_token_ids_accepts_single_int_and_falls_back_to_tokenizer(monkeypatch):
    _patch_generation_config(monkeypatch, SimpleNamespace(eos_token_id=151645))
    assert hf.load_eos_token_ids("model", StubTokenizer(eos=1)) == {1, 151645}

    _patch_generation_config(monkeypatch, SimpleNamespace(eos_token_id=None))
    assert hf.load_eos_token_ids("model", StubTokenizer(eos=None)) == frozenset()


def test_load_eos_when_generation_config_is_missing(monkeypatch):
    def boom(path):
        raise OSError("no config")

    monkeypatch.setattr(hf, "GenerationConfig", SimpleNamespace(from_pretrained=boom))
    assert hf.load_eos_token_ids("model", StubTokenizer(eos=1)) == {1}


def test_toolcall_anchor_requires_exactly_one_token(monkeypatch):
    tok = StubTokenizer(encoded={"<|tool|>": [12345], "<|multi|>": [1, 2]})
    assert hf.load_toolcall_anchor_id(tok, "<|tool|>") == 12345
    assert hf.load_toolcall_anchor_id(tok, "<|multi|>") is None
    assert hf.load_toolcall_anchor_id(tok, "") is None
    assert hf.load_toolcall_anchor_id(tok, None) is None


def test_generation_sampling_recommends_greedy_when_configured(monkeypatch):
    _patch_generation_config(monkeypatch, SimpleNamespace(do_sample=False, temperature=1.0))
    assert hf.load_generation_sampling("model") == {"temperature": 0.0}


def test_generation_sampling_picks_present_keys(monkeypatch):
    _patch_generation_config(
        monkeypatch, SimpleNamespace(do_sample=True, temperature=1.0, top_k=20, top_p=None)
    )
    assert hf.load_generation_sampling("model") == {"temperature": 1.0, "top_k": 20}


def test_generation_sampling_empty_when_no_config(monkeypatch):
    def boom(path):
        raise OSError("no config")

    monkeypatch.setattr(hf, "GenerationConfig", SimpleNamespace(from_pretrained=boom))
    assert hf.load_generation_sampling("model") == {}


def test_raw_config_shim_attribute_access():
    shim = hf.RawConfigShim({"hidden_size": 42, "text_config": {"hidden_size": 7}})
    assert shim.hidden_size == 42
    assert isinstance(shim.text_config, hf.RawConfigShim)
    assert shim.text_config.hidden_size == 7
    assert shim._name_or_path == ""  # must not raise like other underscore names


def test_raw_config_shim_raises_for_unknown_attributes():
    shim = hf.RawConfigShim({})
    with pytest.raises(AttributeError):
        shim.missing
    with pytest.raises(AttributeError):
        shim._private


def test_raw_config_shim_to_dict_is_a_deep_copy():
    shim = hf.RawConfigShim({"nested": {"x": 1}})
    doc = shim.to_dict()
    doc["nested"]["x"] = 99
    assert shim.nested["x"] == 1


def test_sidecar_quantization_config(tmp_path):
    assert hf.sidecar_quantization_config(str(tmp_path)) is None

    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"format": "old-export", "quantization": {"bits": 4}})
    )
    assert hf.sidecar_quantization_config(str(tmp_path)) == {
        "quant_method": "modelopt",
        "bits": 4,
    }


def test_optional_hf_file_local_branch(tmp_path):
    assert hf.optional_hf_file(str(tmp_path), "nope.json") is None
    (tmp_path / "yes.json").write_text("{}")
    assert hf.optional_hf_file(str(tmp_path), "yes.json") == str(tmp_path / "yes.json")


def _write_config(tmp_path, doc):
    (tmp_path / "config.json").write_text(json.dumps(doc))
    return str(tmp_path)


def _raiser(exc):
    def raise_it(*args, **kwargs):
        raise exc

    return raise_it


def test_cached_load_falls_back_to_raw_json_on_unknown_model_type(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"model_type": "future-model", "hidden_size": 7, "text_config": {"n": 1}})
    monkeypatch.setattr(
        hf, "AutoConfig",
        SimpleNamespace(from_pretrained=_raiser(ValueError("model type 'future-model' is unavailable"))),
    )
    config = hf.cached_load_hf_config(path)
    assert isinstance(config, hf.RawConfigShim)
    assert config.model_type == "future-model"
    assert config.hidden_size == 7
    assert config.text_config.n == 1  # nested *_config sub-dicts get wrapped too


def test_cached_load_reraises_other_config_valueerrors(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"model_type": "x"})
    monkeypatch.setattr(
        hf, "AutoConfig", SimpleNamespace(from_pretrained=_raiser(ValueError("bad path")))
    )
    with pytest.raises(ValueError, match="bad path"):
        hf.cached_load_hf_config(path)


def test_sidecar_quantization_config_merges_when_config_json_has_none(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"model_type": "x"})
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    monkeypatch.setattr(
        hf, "AutoConfig",
        SimpleNamespace(from_pretrained=_raiser(ValueError("unknown model type 'x'"))),
    )
    config = hf.cached_load_hf_config(path)
    # *_config keys get the attribute-view wrapper; the raw dict round-trips via to_dict
    assert config.quantization_config.to_dict() == {"quant_method": "modelopt", "bits": 4}


def test_config_json_quantization_wins_over_the_sidecar(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"model_type": "x", "quantization_config": {"quant_method": "fp8"}})
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    monkeypatch.setattr(
        hf, "AutoConfig",
        SimpleNamespace(from_pretrained=_raiser(ValueError("unknown model type 'x'"))),
    )
    config = hf.cached_load_hf_config(path)
    assert config.quantization_config.to_dict() == {"quant_method": "fp8"}


def test_weight_allow_patterns_reads_the_shard_index(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {"a": "m.safetensors", "b": "n.safetensors", "c": "m.safetensors"}}))
    monkeypatch.setattr(hf, "hf_hub_download", lambda repo, filename, **kw: str(index))
    assert hf._weight_allow_patterns("repo") == ["m.safetensors", "n.safetensors"]


def test_weight_allow_patterns_falls_back_to_glob(tmp_path, monkeypatch):
    def boom(repo, filename, **kw):
        raise OSError("no index")

    monkeypatch.setattr(hf, "hf_hub_download", boom)
    assert hf._weight_allow_patterns("repo") == ["*.safetensors"]


def test_download_hf_weight_passes_local_dirs_through(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "snapshot_download", _raiser(AssertionError("must not hit the hub")))
    assert hf.download_hf_weight(str(tmp_path)) == str(tmp_path)


def test_download_hf_weight_failure_becomes_value_error(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise OSError("no such repo")

    monkeypatch.setattr(hf, "snapshot_download", boom)
    monkeypatch.setattr(hf, "_weight_allow_patterns", lambda repo: ["*.safetensors"])
    with pytest.raises(ValueError, match="neither a local directory nor a valid model ID"):
        hf.download_hf_weight("org/model")


def test_load_tokenizer_fills_an_empty_chat_template_from_the_hub(tmp_path, monkeypatch):
    stub = StubTokenizer()
    monkeypatch.setattr(hf, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda p: stub))
    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "TEMPLATE"}))
    monkeypatch.setattr(hf, "hf_hub_download", lambda repo_id, filename: str(tmp_path / "chat_template.json"))

    tok = hf.load_tokenizer("org/model")
    assert tok is stub
    assert tok.chat_template == "TEMPLATE"


def test_config_cache_follows_rewritten_content_at_the_same_path(tmp_path, monkeypatch):
    """The config cache keys on content, not just path: pytest's ``failed`` tmp retention
    reuses a deleted passed test's numbered dir for a later test, and a path-keyed cache
    served the second test the first one's config (wrong quant scheme -> wrong rejection)."""
    class _Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def to_dict(self):
            return dict(self.__dict__)

        @classmethod
        def from_pretrained(cls, path, **kw):
            with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
                return cls(**json.load(fh))

    monkeypatch.setattr(hf, "AutoConfig", _Cfg)

    def write(hidden: int, pad: str):
        (tmp_path / "config.json").write_text(json.dumps({
            "architectures": ["RewriteProbe"],
            "hidden_size": hidden,
            "pad": pad,
        }))

    write(1, "a")
    assert hf.cached_load_hf_config(str(tmp_path)).hidden_size == 1
    write(2, "bb")  # different size AND mtime: either alone must bust the cache
    assert hf.cached_load_hf_config(str(tmp_path)).hidden_size == 2


def _checkpoint(path, model_type: str, dtype: str, rope_parameters: dict) -> str:
    path.mkdir()
    text_config = {"hidden_size": 64, "max_position_embeddings": 4096, "rope_parameters": rope_parameters}
    (path / "config.json").write_text(json.dumps({"model_type": model_type, "dtype": dtype, "text_config": text_config}))
    return str(path)


@pytest.mark.parametrize("model_type", ["qwen3_5", "a_model_type_transformers_does_not_know"])
def test_hf_overrides_load_as_if_the_checkpoint_config_said_so(tmp_path, model_type):
    # --hf-overrides, against the same config written into config.json; vLLM's merge:
    # a nested config section updates key by key, any other value is replaced whole
    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}
    checkpoint = _checkpoint(tmp_path / "checkpoint", model_type, "bfloat16", {"rope_type": "default", "rope_theta": 1e6})
    edited = _checkpoint(tmp_path / "edited", model_type, "float16", yarn)
    overrides = {"dtype": "float16", "text_config": {"rope_parameters": yarn}}
    requested = copy.deepcopy(overrides)

    overridden = hf.cached_load_hf_config(checkpoint, overrides)
    expected = hf.cached_load_hf_config(edited)
    assert overridden.dtype == expected.dtype
    assert overridden.text_config.to_dict() == expected.text_config.to_dict()
    # transformers fills rope defaults into the rope_parameters it is handed
    assert overrides == requested
    # a fresh copy each time: the override never reaches the cached checkpoint config
    assert hf.cached_load_hf_config(checkpoint).text_config.rope_parameters["rope_type"] == "default"
