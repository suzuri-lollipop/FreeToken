"""The pure parts of the multi (concurrent-prefill) calibration: no server, no GPU.

These pin the CONTRACT the live arms are judged by -- the winner env shape, the
selection thresholds, the log-based arm-validity markers, and the port rewrite that
must NOT strip --speculative (a multi arm measures the profile's real config).
"""
from __future__ import annotations

import pytest

from freetoken.calibrate_multi import (
    ARMS,
    MULTI_KEYS,
    MultiScore,
    arms_for,
    parse_live_servers,
    argv_with_port,
    arm_active,
    choose_multi,
    render_script_multi,
    score_window,
    victims_from_spec,
    winner_env,
    workload,
)


def _score(label, stall, tps, base_stall=None, valid=True, ttft_short=1.0):
    return MultiScore(label=label, env={}, stall_worst=stall, stall_count=1,
                      stall_secs=stall, ttft_long=5.0, ttft_short=ttft_short,
                      out_tps=tps, tokens=1000, wall=10.0, valid=valid)


def test_winner_env_is_the_full_knob_set_with_explicit_zeros():
    for label, _env in ARMS:
        env = winner_env(label)
        assert set(env) == set(MULTI_KEYS), label
    assert winner_env("none")["FREETOKEN_PIPED_PREFILL"] == "0"
    assert winner_env("mixed")["FREETOKEN_MIXED_PREFILL_TOKENS"] == "1024"
    assert winner_env("mixed")["FREETOKEN_PIPED_PREFILL"] == "0"
    assert winner_env("piped-sf")["FREETOKEN_PIPED_PREFILL"] == "1"
    assert winner_env("piped-sf")["FREETOKEN_ADMIT_SHORT_FIRST"] == "1024"
    assert winner_env("piped")["FREETOKEN_ADMIT_SHORT_FIRST"] == "0"


def test_arm_validity_reads_the_arms_own_log_markers():
    assert arm_active("piped", "... piped prefill enabled ...") == (True, "")
    ok, why = arm_active("piped", "FREETOKEN_PIPED_PREFILL: speculative decoding "
                                  "armed; piping disabled (eager spec verify...)")
    assert not ok and "speculative" in why
    assert arm_active("mixed", "[chunktime] #3 T=1024 window=2801ms mix=2")[0]
    assert not arm_active("mixed", "[chunktime] #3 T=8192 window=8264ms")[0]
    assert arm_active("none", "whatever")[0]


def test_choose_multi_requires_a_stall_win_and_a_survivable_tps_cost():
    scores = [_score("none", 3.8, 51.0), _score("mixed", 2.1, 23.0),
              _score("piped", 0.1, 45.0), _score("piped-sf", 0.1, 44.0)]
    label, note = choose_multi(scores)
    assert label == "piped"  # tie bin -> higher tps; mixed rejected by the tps guard
    assert "3.80s -> 0.10s" in note

    label, note = choose_multi([_score("none", 0.4, 51.0), _score("piped", 0.1, 50.0)])
    assert label == "none" and "only 0.30s" in note  # gain under STALL_GAIN_S

    label, note = choose_multi([_score("none", 3.8, 51.0), _score("piped", 0.1, 35.0)])
    assert label == "none" and "throughput" in note  # catastrophic tps loss

    label, note = choose_multi([_score("none", 3.0, 51.0),
                                _score("piped", 0.2, 50.0, valid=False)])
    assert label == "none" and "activated" in note  # invalid arms never win


def test_choose_multi_prefers_the_lower_short_ttft_arm_inside_a_tie():
    scores = [_score("none", 3.8, 51.0), _score("piped", 0.10, 45.0, ttft_short=9.0),
              _score("piped-sf", 0.12, 45.0, ttft_short=4.0)]
    label, _ = choose_multi(scores)
    assert label == "piped-sf"


def test_argv_with_port_keeps_speculative_and_rewrites_both_port_forms():
    argv = argv_with_port(["-m", "freetoken", "--speculative", "mtp", "--port", "1919",
                           "--port=1918", "--tp-size", "2"], 1921)
    assert argv == ["-m", "freetoken", "--speculative", "mtp", "--tp-size", "2",
                    "--port", "1921"]


def test_render_script_exports_the_winner_and_strips_stale_knobs():
    spec = {"python": "/py", "argv": ["-m", "freetoken", "--port", "1919"], "cwd": "/w",
            "env": {"FREETOKEN_PIPED_PREFILL": "1", "PYTHONPATH": "/p"}, "port": 1919}
    text = render_script_multi(spec, {"env": winner_env("piped-sf")})
    assert "cd /w" in text
    assert "export PYTHONPATH=/p" in text
    assert text.count("FREETOKEN_PIPED_PREFILL=1") == 1
    assert "export FREETOKEN_ADMIT_SHORT_FIRST=1024" in text
    assert "exec /py -m freetoken --port 1919" in text
    # the none winner: every knob explicitly off
    off = render_script_multi(spec, {"env": winner_env("none")})
    assert "export FREETOKEN_PIPED_PREFILL=0" in off


def test_victims_follow_max_running_requests():
    assert victims_from_spec({"argv": ["-m", "freetoken"]}) == 3
    assert victims_from_spec({"argv": ["--max-running-requests", "2"]}) == 1
    assert victims_from_spec({"argv": ["--max-running-requests=9"]}) == 3


def test_workload_staggers_victims_and_one_aggressor():
    items = workload({"short": 128, "long": 16384, "out": 256, "gap": 6.0, "victims": 3})
    assert [it["role"] for it in items] == ["victim"] * 3 + ["aggressor"]
    assert [it["at"] for it in items] == [0.0, 2.0, 4.0, 6.0]
    assert items[-1]["max_tokens"] == 128
    assert len(items[-1]["prompt"].split()) >= 16384


def test_cli_dispatches_the_mode_without_breaking_the_bare_default(monkeypatch):
    from freetoken import cli

    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    import freetoken.calibrate_multi as multi
    import freetoken.calibrate_lookup as lookup
    monkeypatch.setattr(lookup, "main", fake_main)
    monkeypatch.setattr(multi, "main", fake_main)
    assert cli._run_calibrate(["--config", "x.json"]) == 0  # bare = single
    assert seen["argv"] == ["--config", "x.json"]
    assert cli._run_calibrate(["multi", "--config", "x.json"]) == 0
    assert seen["argv"] == ["--config", "x.json"]


def test_score_window_opens_at_the_aggressor_send_time():
    # a victim that stalls 6 s while the aggressor prefills, then streams: the stall
    # happened BEFORE the aggressor's first token and must be counted (open_at = send)
    victims = {
        "0": {"arrivals": [0.0, 1.0, 7.0, 7.1, 7.2]},  # 6 s gap opened at t=6
        "1": {"arrivals": [6.5, 6.6]},                 # too short to gap after open
    }
    out = score_window(victims, open_at=6.0)
    assert out["stall_worst"] == pytest.approx(6.0)
    assert out["stall_count"] == 1 and out["stall_secs"] == pytest.approx(6.0)
    # a gap that ENDED before the window (the victims' own prefill ramp) never counts
    assert score_window(victims, open_at=7.25)["stall_count"] == 0
    # a stall that straddles the window opening is counted: that is the aggressor's freeze
    assert score_window(victims, open_at=6.9)["stall_worst"] == pytest.approx(6.0)


def test_arms_for_measures_every_arm_on_any_tp():
    tp2 = {"argv": ["-m", "freetoken", "--tp-size", "2"]}
    arms = arms_for(tp2)
    assert [label for label, _e, skip in arms] == ["none", "mixed", "piped", "piped-sf"]
    assert all(skip == "" for _l, _e, skip in arms)
    smoke = arms_for({"argv": []}, smoke=True)
    assert [label for label, _e, _s in smoke] == ["none", "piped"]


def test_parse_live_servers_spots_the_server_not_the_calibrator():
    lines = "\n".join([
        "100 /py/.venv/bin/python -m freetoken --model-path /m --tp-size 1",
        "200 /py/.venv/bin/python -m freetoken.calibrate_multi --config x.json",
        "300 bash -c pgrep -af -m freetoken",
    ])
    live = parse_live_servers(lines)
    assert len(live) == 1 and live[0].startswith("100 ")


def test_request_body_probes_the_production_sampling_by_default():
    from freetoken.calibrate_multi import request_body

    prod = request_body("m", "p", 64, None)
    # omitted temperature = the server's sampling_defaults=model + guard regime
    assert "temperature" not in prod and "top_k" not in prod and "top_p" not in prod
    assert prod["ignore_eos"] is True and prod["stream"] is True

    greedy = request_body("m", "p", 64, 0.0)
    assert greedy["temperature"] == 0.0 and greedy["top_k"] == 1 and greedy["top_p"] == 1.0
