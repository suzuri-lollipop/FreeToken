"""The pure parts of the lookup calibration (no server, no GPU)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from freetoken.calibrate_lookup import (
    ROOT,
    Score,
    choose,
    normalize_argv,
    render_script,
    spec_from_args,
)


def _score(label: str, tok_s: float) -> Score:
    return Score(label=label, env={}, per_prompt={"code-edit": tok_s}, tok_s=tok_s,
                 tokens=100, seconds=100 / tok_s)


def test_normalize_argv_drops_speculation_and_keeps_one_port():
    argv = normalize_argv(
        ["-m", "freetoken", "--speculative", "mtp", "--port", "1919",
         "--model-path", "/m", "--port=1918"],
        1920,
    )
    assert argv == ["-m", "freetoken", "--model-path", "/m", "--port", "1920"]


def test_render_script_exports_the_measured_values():
    spec = {"python": "/py", "argv": ["-m", "freetoken", "--port", "1919"],
            "cwd": "/work", "env": {"PYTHONPATH": "/work/python"}, "port": 1920}
    chosen = {"env": {"FREETOKEN_LOOKUP_MAX_DRAFT": "5"}}
    text = render_script(spec, chosen)
    assert "cd /work" in text
    assert "export FREETOKEN_LOOKUP_MAX_DRAFT=5" in text
    assert "export PYTHONPATH=/work/python" in text
    assert "exec /py -m freetoken --port 1920 --speculative lookup" in text


def test_render_script_without_lookup_has_no_speculative_or_lookup_env():
    spec = {"python": "/py", "argv": ["-m", "freetoken"], "cwd": "/work",
            "env": {}, "port": 1920}
    text = render_script(spec, None)
    assert "--speculative" not in text
    assert "FREETOKEN_LOOKUP_" not in text
    assert text.rstrip().endswith("exec /py -m freetoken --port 1920")


def test_choose_keeps_a_clear_winner_and_rejects_noise_and_losses():
    default = (4, 3, 256)
    baseline = _score("baseline", 60.0)
    default_score = _score("default", 80.0)
    # a clear win over both the default and the baseline
    assert choose({default: default_score, (5, 3, 256): _score("k5", 90.0)},
                  baseline, default) == (5, 3, 256)
    # better than baseline but within 3% of the default: keep the default
    assert choose({default: default_score, (5, 3, 256): _score("k5", 82.0)},
                  baseline, default) == default
    # worse than plain decoding: disable
    assert choose({default: _score("default", 55.0)}, baseline, default) is None


def test_spec_from_args_loads_a_config_and_strips_old_lookup_env(tmp_path):
    cfg = {
        "python": "/py",
        "argv": ["-m", "freetoken", "--model-path", "/m", "--port", "1919",
                 "--speculative", "lookup"],
        "env": {"PYTHONPATH": "/old", "FREETOKEN_LOOKUP_MAX_DRAFT": "9", "KEEP": "1"},
        "port": 1919,
        "model_name": "unit/model",
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    a = SimpleNamespace(config=str(path), model_path=None, served_model_name="",
                        python="", server_args="", port=1920)
    spec = spec_from_args(a)
    assert spec["port"] == 1920
    assert spec["argv"][-2:] == ["--port", "1920"]
    assert "--speculative" not in spec["argv"]
    assert spec["env"]["KEEP"] == "1"
    assert "FREETOKEN_LOOKUP_MAX_DRAFT" not in spec["env"]
    assert spec["env"]["PYTHONPATH"].startswith(str(ROOT / "python"))
    assert spec["env"]["PYTHONPATH"].endswith(":/old")
    assert spec["model_name"] == "unit/model"


def test_arm_argv_adds_lookup_only_to_a_candidate():
    from freetoken.calibrate_lookup import arm_argv

    spec = {"argv": ["-m", "freetoken", "--port", "1920"]}
    assert arm_argv(spec, False) == ["-m", "freetoken", "--port", "1920"]
    assert arm_argv(spec, True)[-2:] == ["--speculative", "lookup"]


def test_ft_exposes_the_calibrate_command():
    from freetoken.cli import COMMANDS

    assert COMMANDS["calibrate"] == "_run_calibrate"


def test_set_parallel_replaces_the_ceiling():
    from freetoken.calibrate_lookup import current_parallel, set_parallel

    argv = ["-m", "freetoken", "--max-running-requests", "4", "--model-path", "/m"]
    out = set_parallel(argv, 2)
    assert out[-2:] == ["--max-running-requests", "2"]
    assert out.count("--max-running-requests") == 1
    assert current_parallel(out) == 2


def test_parse_parallels_and_batch_prompts():
    from freetoken.calibrate_lookup import parse_parallels, prompts_for

    assert parse_parallels("4,1,2,2") == [1, 2, 4]
    four = prompts_for(4)
    assert len(four) == 4
    assert len({p["name"] for p in four}) == 4
    assert len({p["prompt"] for p in four}) == 4
    assert len(prompts_for(6)) == 6


def test_candidate_env_includes_the_acceptance_floor():
    from freetoken.calibrate_lookup import _candidate_env

    env = _candidate_env(5, 3, 256, 0.7)
    assert env["FREETOKEN_LOOKUP_MAX_DRAFT"] == "5"
    assert env["FREETOKEN_LOOKUP_MIN_MATCH"] == "3"
    assert env["FREETOKEN_LOOKUP_MIN_CONTEXT"] == "256"
    assert env["FREETOKEN_LOOKUP_MIN_ACCEPTANCE"] == "0.7"


def test_measured_arm_keeps_the_speculative_flag_after_the_port_rewrite():
    from freetoken.calibrate_lookup import measured_argv

    spec = {"argv": ["--model-path", "m", "--speculative", "lookup", "--port", "1919"]}
    argv = measured_argv(spec, True, 1922)
    assert argv.count("--speculative") == 1
    assert argv[-2:] == ["--speculative", "lookup"]
    assert argv[argv.index("--port") + 1] == "1922"
    assert measured_argv(spec, False, 1922).count("--speculative") == 0


def test_distributed_timeout_is_added_once_and_respects_the_caller():
    from freetoken.calibrate_lookup import ensure_distributed_timeout

    argv = ensure_distributed_timeout(["--model-path", "m"])
    assert argv[-2:] == ["--distributed-timeout", "180"]
    assert ensure_distributed_timeout(argv) == argv
    assert ensure_distributed_timeout(["--distributed-timeout=90"]) == ["--distributed-timeout=90"]


def test_render_script_omits_lookup_when_it_lost_at_the_parallel_ceiling():
    from freetoken.calibrate_lookup import render_script

    spec = {"argv": ["--model-path", "m"], "port": 1920, "python": "/usr/bin/python3"}
    chosen = {
        "env": {"FREETOKEN_LOOKUP_MAX_DRAFT": "6"},
        "enabled_at_parallel": False,
    }
    text = render_script(spec, chosen)
    assert "--speculative" not in text
    assert "FREETOKEN_LOOKUP_MAX_DRAFT" not in text


def test_parallel_loss_tolerance_keeps_a_near_neutral_lookup():
    from freetoken.calibrate_lookup import lookup_worth_keeping

    assert lookup_worth_keeping(0.05)
    assert lookup_worth_keeping(-0.01)  # within the 2% tolerance: single-stream win remains
    assert not lookup_worth_keeping(-0.117)
