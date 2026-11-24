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
    argv_with_timings,
    arm_active,
    choose_multi,
    choose_moe,
    choose_spec,
    concurrency_from_spec,
    decode_workload,
    interleave_workload,
    moe_argv,
    moe_strategies_for,
    render_script_multi,
    run_concurrent_probe,
    score_window,
    smoke_backends,
    spec_argv,
    spec_backends_for,
    strip_moe_args,
    strip_spec_args,
    target_model_type,
    tp_from_spec,
    victims_from_spec,
    winner_env,
    workload,
)


def _score(label, stall, tps, base_stall=None, valid=True, ttft_short=1.0, decode_tps=0.0,
           interleave_tps=0.0):
    return MultiScore(label=label, env={}, stall_worst=stall, stall_count=1,
                      stall_secs=stall, ttft_long=5.0, ttft_short=ttft_short,
                      out_tps=tps, tokens=1000, wall=10.0, valid=valid,
                      decode_tps=decode_tps, interleave_tps=interleave_tps)


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


def test_choose_multi_requires_a_concurrent_tps_gain_and_a_survivable_stall():
    # the selection score is the mean regime tok/s at the profile's full concurrency;
    # the staggered wall's stall is a guard, not the objective
    scores = [_score("none", 3.8, 51.0, decode_tps=200.0, interleave_tps=130.0),
              _score("mixed", 2.1, 23.0, decode_tps=205.0, interleave_tps=131.0),
              _score("piped", 0.1, 45.0, decode_tps=230.0, interleave_tps=150.0),
              _score("piped-sf", 0.1, 44.0, decode_tps=228.0, interleave_tps=148.0)]
    label, note = choose_multi(scores)
    assert label == "piped"  # highest mean regime tok/s (165 -> 190)
    assert "+15.2%" in note and "165.0 -> 190.0" in note

    # gain under TPS_MIN_GAIN -> none, however much the stall improves
    label, note = choose_multi([
        _score("none", 3.8, 51.0, decode_tps=200.0, interleave_tps=130.0),
        _score("piped", 0.1, 45.0, decode_tps=202.0, interleave_tps=131.0)])
    assert label == "none" and "concurrent tok/s" in note

    # a throughput win that freezes the running streams beyond MULTI_STALL_TOL -> none
    label, note = choose_multi([
        _score("none", 1.0, 51.0, decode_tps=200.0, interleave_tps=130.0),
        _score("piped", 1.5, 60.0, decode_tps=240.0, interleave_tps=156.0)])
    assert label == "none" and "stall" in note

    label, note = choose_multi([
        _score("none", 3.0, 51.0, decode_tps=200.0, interleave_tps=130.0),
        _score("piped", 0.2, 50.0, valid=False, decode_tps=240.0, interleave_tps=156.0)])
    assert label == "none" and "activated" in note  # invalid arms never win

    # no measured regime probe -> no score -> none (the probes ARE the objective)
    label, _ = choose_multi([_score("none", 3.8, 51.0), _score("piped", 0.1, 45.0)])
    assert label == "none"


def test_choose_multi_breaks_ties_by_stall_then_short_ttft():
    scores = [_score("none", 3.8, 51.0, decode_tps=200.0, interleave_tps=130.0),
              _score("piped", 0.10, 45.0, ttft_short=9.0,
                     decode_tps=230.0, interleave_tps=150.0),
              _score("piped-sf", 0.12, 45.0, ttft_short=4.0,
                     decode_tps=230.0, interleave_tps=150.0)]
    assert choose_multi(scores)[0] == "piped"  # equal tok/s -> the lower stall
    scores[1] = _score("piped", 0.12, 45.0, ttft_short=9.0,
                       decode_tps=230.0, interleave_tps=150.0)
    assert choose_multi(scores)[0] == "piped-sf"  # equal stall -> the lower short ttft


def test_choose_multi_rejects_a_decode_regression_the_staggered_wall_hid():
    # piped wins the staggered wall and cuts the stall, but the decode-only probe
    # shows the fast-path cost (200 -> 170): the per-regime floor rejects the arm
    scores = [_score("none", 3.8, 51.0, decode_tps=200.0, interleave_tps=130.0),
              _score("piped", 0.1, 60.0, decode_tps=170.0, interleave_tps=160.0)]
    label, note = choose_multi(scores)
    assert label == "none" and "decode-only" in note and "170.0" in note
    # within the noise band the throughput win stands
    scores[1] = _score("piped", 0.1, 60.0, decode_tps=195.0, interleave_tps=160.0)
    assert choose_multi(scores)[0] == "piped"
    # a decode-safe candidate still wins when another arm regresses
    scores = [_score("none", 3.8, 51.0, decode_tps=200.0, interleave_tps=130.0),
              _score("piped", 0.1, 60.0, decode_tps=170.0, interleave_tps=160.0),
              _score("mixed", 1.0, 48.0, decode_tps=202.0, interleave_tps=140.0)]
    assert choose_multi(scores)[0] == "mixed"


def test_choose_multi_rejects_an_interleave_regression_with_neutral_decode():
    # the dev-rig shape: pure decode is neutral under piped, but sharing the link with
    # short chunks costs the staggered chat regime (132.7 -> 124.5 measured)
    scores = [_score("none", 3.8, 51.0, decode_tps=169.0, interleave_tps=133.0),
              _score("piped", 0.1, 45.0, decode_tps=167.5, interleave_tps=124.5)]
    label, note = choose_multi(scores)
    assert label == "none" and "short-interleave" in note and "124.5" in note
    # both probes inside the band and a real gain: the win stands
    scores[1] = _score("piped", 0.1, 45.0, decode_tps=185.0, interleave_tps=130.0)
    assert choose_multi(scores)[0] == "piped"


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


def test_decode_workload_launches_everyone_together_without_an_aggressor():
    cfg = {"short": 128, "long": 16384, "out": 256, "gap": 6.0, "victims": 3,
           "concurrency": 4}
    items = decode_workload(cfg)
    assert len(items) == 4
    assert all(it["at"] == 0.0 for it in items)
    assert all(it["max_tokens"] == 256 for it in items)
    # short prompts: the probe is decode-dominated, not a second prefill benchmark
    assert all(len(it["prompt"].split()) < 64 for it in items)
    # the probe batch follows the profile concurrency, not the capped victim count
    assert len(decode_workload({**cfg, "concurrency": 8})) == 8
    # a cfg without concurrency (old reports) falls back to victims+1
    assert len(decode_workload({"victims": 3, "out": 256})) == 4


def test_interleave_workload_matches_the_bench_cadence_at_the_profile_concurrency():
    cfg = {"short": 128, "long": 16384, "out": 256, "gap": 6.0, "victims": 3,
           "concurrency": 4}
    items = interleave_workload(cfg)
    assert len(items) == 4
    # the launcher bench's 5 s gap: a tighter spacing piles prefills up and flatters
    # piped, which is not the regime the profile serves
    assert [it["at"] for it in items] == [0.0, 5.0, 10.0, 15.0]
    assert all(it["max_tokens"] == 1024 for it in items)
    # ~1024-token prompts: the launcher bench's input length, not the 16k aggressor
    assert all(900 <= len(it["prompt"].split()) <= 1100 for it in items)
    # distinct prompts: a radix prefix hit would turn the probe into a cache test
    assert len({it["prompt"] for it in items}) == len(items)
    assert len(interleave_workload({**cfg, "concurrency": 8})) == 8


def test_concurrency_from_spec_reads_the_full_max_running_requests():
    assert concurrency_from_spec({"argv": ["--max-running-requests", "8"]}) == 8
    assert concurrency_from_spec({"argv": ["--max-running-requests=8"]}) == 8
    assert concurrency_from_spec({"argv": []}) == 4
    assert concurrency_from_spec({"argv": ["--max-running-requests", "1"]}) == 1
    # the staggered workload's victims stay capped; the regime probes do not
    assert victims_from_spec({"argv": ["--max-running-requests", "8"]}) == 3


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


def test_find_live_servers_gives_pgrep_a_parseable_pattern(monkeypatch):
    # the pattern starts with a dash: without an end-of-options marker pgrep exits 2
    # with empty stdout, and the refuse-to-calibrate-beside-a-live-server guard
    # silently never fires (double expert-bank load -> OOM kill)
    import subprocess
    from freetoken.calibrate_multi import find_live_servers

    got = {}

    def fake_run(argv, **kw):
        got["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="100 /py -m freetoken --model-path /m\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert find_live_servers() == ["100 /py -m freetoken --model-path /m"]
    argv = got["argv"]
    pattern = argv[-1]
    assert pattern.startswith("-") and argv[argv.index(pattern) - 1] == "--"


def test_request_body_probes_the_production_sampling_by_default():
    from freetoken.calibrate_multi import request_body

    prod = request_body("m", "p", 64, None)
    # omitted temperature = the server's sampling_defaults=model + guard regime
    assert "temperature" not in prod and "top_k" not in prod and "top_p" not in prod
    assert prod["ignore_eos"] is True and prod["stream"] is True

    greedy = request_body("m", "p", 64, 0.0)
    assert greedy["temperature"] == 0.0 and greedy["top_k"] == 1 and greedy["top_p"] == 1.0


def test_request_body_asks_for_server_side_timings_and_usage():
    from freetoken.calibrate_multi import request_body

    body = request_body("m", "p", 64, None)
    assert body["timings_per_token"] is True
    assert body["stream_options"]["include_usage"] is True
    assert body["stream_options"]["continuous_usage_stats"] is True


def test_strip_spec_args_removes_every_backend_flag_both_spellings():
    argv = ["-m", "freetoken", "--speculative-algorithm", "mtp", "--spec-num-steps=3",
            "--speculative-draft-model-path", "/d", "--spec-token-map=/t",
            "--speculative", "lookup", "--tp-size", "2", "--port", "1919"]
    assert strip_spec_args(argv) == ["-m", "freetoken", "--tp-size", "2", "--port", "1919"]
    # idempotent and leaves a spec-free argv untouched
    assert strip_spec_args(["-m", "freetoken", "--tp-size", "2"]) == ["-m", "freetoken", "--tp-size", "2"]


def test_spec_argv_arms_each_backend_and_dflash_needs_a_draft():
    assert spec_argv("none") == []
    assert spec_argv("mtp") == ["--speculative-algorithm", "mtp"]
    assert spec_argv("mtp", num_steps=4) == ["--speculative-algorithm", "mtp",
                                             "--speculative-num-steps", "4"]
    assert spec_argv("dflash", "/draft") == ["--speculative-algorithm", "dflash",
                                             "--speculative-draft-model-path", "/draft"]
    with pytest.raises(ValueError):
        spec_argv("dflash")
    # dspark reads the target's bundled mtp.safetensors; a draft path overrides it
    assert spec_argv("dspark") == ["--speculative-algorithm", "dspark"]
    assert spec_argv("dspark", "/mtp.safetensors", num_steps=5) == [
        "--speculative-algorithm", "dspark",
        "--speculative-draft-model-path", "/mtp.safetensors",
        "--speculative-num-steps", "5"]


def test_spec_backends_for_auto_skips_dflash_without_a_draft():
    assert spec_backends_for("none", None) == []
    auto = spec_backends_for("auto", None)
    assert [a for a, _ in auto] == ["mtp", "dflash"]
    assert auto[0][1] == "" and "draft" in auto[1][1]  # dflash skipped, mtp runnable
    assert spec_backends_for("auto", "/draft")[1][1] == ""  # with a draft dflash runs
    assert [a for a, _ in spec_backends_for("mtp", None)] == ["mtp"]
    with pytest.raises(ValueError):
        spec_backends_for("lookup", None)  # not a live backend in this fork


def test_spec_backends_for_auto_offers_dspark_on_inkling():
    # Inkling-2-Mini: dspark is the runnable draft, the Qwen backends are pre-skipped
    ink = spec_backends_for("auto", None, model_type="inkling2_loop")
    assert [a for a, _ in ink] == ["dspark", "mtp", "dflash"]
    assert ink[0][1] == ""
    assert "dspark" in ink[1][1] and "DFlash" in ink[2][1]
    assert spec_backends_for("dspark", None, model_type="inkling2_loop") == [("dspark", "")]
    # dspark is tp 1 only (the drafter's rings are rank-local)
    skip = spec_backends_for("dspark", None, model_type="inkling2_loop", tp=2)
    assert skip[0][0] == "dspark" and "tp" in skip[0][1]
    # other checkpoints: auto keeps mtp + dflash, an explicit dspark is pre-skipped
    assert [a for a, _ in spec_backends_for("auto", None, model_type="qwen4_exp")] == [
        "mtp", "dflash"]
    skip = spec_backends_for("dspark", None, model_type="qwen4_exp")
    assert "inkling2_loop" in skip[0][1]
    # unreadable model_type (GGUF target): an explicit dspark runs, the engine judges
    assert spec_backends_for("dspark", None) == [("dspark", "")]


def test_target_model_type_and_tp_read_the_profile_argv(tmp_path):
    mp = tmp_path / "model"
    mp.mkdir()
    (mp / "config.json").write_text('{"model_type": "inkling2_loop"}', "utf-8")
    argv = ["-m", "freetoken", "--model-path", str(mp), "--tp-size", "2"]
    assert target_model_type(argv) == "inkling2_loop"
    assert tp_from_spec(argv) == 2
    assert tp_from_spec(["--tensor-parallel-size=4"]) == 4
    assert tp_from_spec([]) == 1
    # a broken / missing config -> None (auto keeps mtp + dflash)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0")
    assert target_model_type(["--model", str(gguf)]) is None
    assert target_model_type([]) is None
    (mp / "config.json").write_text("{broken", "utf-8")
    assert target_model_type(argv) is None


def test_concurrent_probe_dead_request_is_a_server_error(monkeypatch):
    # a failed probe request (503 during a rebuild, a timeout) must invalidate the
    # arm through the retry/exclude path, not crash the tool on a None result
    from freetoken.calibrate_multi import ServerError

    def chat(server, model, prompt, max_tokens, t_start, delay, temperature=None):
        if prompt == "bad":
            raise OSError("503 Service Unavailable")
        return {"end": t_start + 1.0, "arrivals": [t_start + 0.5, t_start + 1.0],
                "tokens": 2, "usage_tokens": 2}
    monkeypatch.setattr("freetoken.calibrate_multi._stream_chat", chat)
    items = [{"prompt": "ok", "max_tokens": 4, "at": 0.0},
             {"prompt": "bad", "max_tokens": 4, "at": 0.0}]
    with pytest.raises(ServerError, match="decode probe failed"):
        run_concurrent_probe(object(), "m", items, "decode")


def test_smoke_backends_prefers_runnable_mtp_then_dspark():
    # a pre-skipped mtp (Inkling) must not shadow the runnable dspark arm
    assert smoke_backends([("mtp", ""), ("dflash", "no draft")]) == [("mtp", "")]
    assert smoke_backends([("dspark", ""), ("mtp", "skip"), ("dflash", "skip")]) == [
        ("dspark", "")]
    assert smoke_backends([("mtp", "skip")]) == [("mtp", "skip")]
    assert smoke_backends([]) == []


def test_target_model_type_reads_gguf_architecture(tmp_path):
    # a GGUF target carries its config in the file's KV section: auto must see
    # inkling2_loop there too, or the dspark arm would never be offered
    import struct
    p = tmp_path / "inkling.gguf"
    out = bytearray(b"GGUF")
    out.extend(struct.pack("<I", 3))      # version
    out.extend(struct.pack("<Q", 0))      # tensor_count
    out.extend(struct.pack("<Q", 1))      # metadata_kv_count
    key, val = b"general.architecture", b"inkling2_loop"
    out.extend(struct.pack("<Q", len(key)) + key)
    out.extend(struct.pack("<I", 8))      # GGUF_TYPE_STRING
    out.extend(struct.pack("<Q", len(val)) + val)
    p.write_bytes(bytes(out))
    assert target_model_type(["--model-path", str(p)]) == "inkling2_loop"
    assert spec_backends_for("auto", None,
                             model_type=target_model_type(["--model-path", str(p)]))[0] == ("dspark", "")


def test_argv_with_timings_adds_the_gate_once():
    assert argv_with_timings(["-m", "freetoken"]) == ["-m", "freetoken", "--enable-timings-report"]
    once = argv_with_timings(["-m", "freetoken", "--enable-timings-report"])
    assert once.count("--enable-timings-report") == 1


def _spec_score(algo, tps, stall, valid=True, decode_tps=0.0, interleave_tps=0.0):
    return MultiScore(label=f"spec-{algo}", env={}, stall_worst=stall, stall_count=0,
                      stall_secs=0.0, ttft_long=5.0, ttft_short=1.0, out_tps=tps,
                      tokens=1000, wall=10.0, valid=valid, spec=algo,
                      decode_tps=decode_tps, interleave_tps=interleave_tps)


def test_choose_spec_needs_a_tps_gain_without_a_stall_regression():
    base = _score("none", 1.0, 50.0, decode_tps=200.0, interleave_tps=130.0)
    # +20% concurrent tok/s, stall unchanged -> mtp wins
    algo, note = choose_spec([_spec_score("mtp", 60.0, 1.0,
                                          decode_tps=240.0, interleave_tps=156.0)], base)
    assert algo == "mtp" and "+20.0%" in note
    # only +1% -> below SPEC_MIN_GAIN
    algo, note = choose_spec([_spec_score("mtp", 50.5, 1.0,
                                          decode_tps=202.0, interleave_tps=131.3)], base)
    assert algo is None and "tok/s" in note
    # +20% but worsens the stall beyond SPEC_STALL_TOL -> rejected
    algo, note = choose_spec([_spec_score("mtp", 60.0, 1.5,
                                          decode_tps=240.0, interleave_tps=156.0)], base)
    assert algo is None and "stall" in note
    # invalid arms never win; the faster valid backend is chosen
    algo, _ = choose_spec([_spec_score("mtp", 70.0, 1.0, valid=False,
                                       decode_tps=280.0, interleave_tps=182.0),
                           _spec_score("dflash", 60.0, 1.0,
                                       decode_tps=240.0, interleave_tps=156.0)], base)
    assert algo == "dflash"
    assert choose_spec([], base)[0] is None


def test_choose_spec_carries_the_decode_guard():
    base = _score("none", 1.0, 50.0, decode_tps=200.0, interleave_tps=130.0)
    # a mean win bought with -10% pure decode: the per-regime floor rejects it
    algo, note = choose_spec([_spec_score("mtp", 60.0, 1.0,
                                          decode_tps=180.0, interleave_tps=170.0)], base)
    assert algo is None and "decode-only" in note
    algo, _ = choose_spec([_spec_score("mtp", 60.0, 1.0,
                                       decode_tps=210.0, interleave_tps=156.0)], base)
    assert algo == "mtp"
    # the interleave probe guards the same way
    base = _score("none", 1.0, 50.0, decode_tps=200.0, interleave_tps=133.0)
    algo, note = choose_spec([_spec_score("mtp", 60.0, 1.0,
                                          decode_tps=240.0, interleave_tps=120.0)], base)
    assert algo is None and "short-interleave" in note


def test_score_window_uses_engine_gaps_for_magnitude_client_for_the_boundary():
    # client arrivals show a 6 s gap, but the engine only stalled 2 s (the rest was
    # delivery jitter): the scored stall is the engine's 2 s, the client's 6 s is kept
    victims = {"0": {"arrivals": [0.0, 1.0, 7.0, 7.1],
                     "engine_gaps": [1.0, 2.0, 0.1]}}
    out = score_window(victims, open_at=6.0)
    assert out["stall_worst"] == pytest.approx(2.0)
    assert out["stall_worst_client"] == pytest.approx(6.0)
    assert out["stall_count"] == 1  # 2 s > STALL_EVENT_S
    # without engine gaps the client delta is scored (backwards compatible)
    plain = {"0": {"arrivals": [0.0, 1.0, 7.0, 7.1]}}
    assert score_window(plain, open_at=6.0)["stall_worst"] == pytest.approx(6.0)


def test_arm_active_judges_a_combo_on_its_prefill_part():
    # piped+spec is refused while spec is armed: the combo is invalid with the reason
    ok, why = arm_active("piped+spec-mtp", "FREETOKEN_PIPED_PREFILL: speculative decoding "
                                           "armed; piping disabled (spec verify...)")
    assert not ok and "speculative" in why
    assert arm_active("piped+spec-mtp", "... piped prefill enabled ...")[0]
    # a pure spec arm is valid once the server ran
    assert arm_active("spec-mtp", "whatever")[0]
    assert arm_active("mixed+spec-mtp", "[chunktime] #3 T=1024 mix=2")[0]


def test_render_script_applies_the_spec_flags_only_when_applied():
    spec = {"python": "/py", "argv": ["-m", "freetoken", "--speculative-algorithm", "mtp",
                                      "--port", "1919"], "cwd": "/w", "env": {}, "port": 1919}
    chosen = {"env": winner_env("none"),
              "spec": {"applied": True, "argv": spec_argv("mtp")}}
    text = render_script_multi(spec, chosen)
    # the profile's own spec flag is stripped, the winner's added once
    assert text.count("--speculative-algorithm mtp") == 1
    assert "exec /py -m freetoken --port 1919 --speculative-algorithm mtp" in text
    # not applied -> no spec flags at all
    off = render_script_multi(spec, {"env": winner_env("none"),
                                     "spec": {"applied": False, "argv": spec_argv("mtp")}})
    assert "--speculative-algorithm" not in off


def _moe_score(strategy, tps, stall, valid=True, note="", decode_tps=0.0,
               interleave_tps=0.0):
    return MultiScore(label=f"moe-{strategy}", env={}, stall_worst=stall, stall_count=0,
                      stall_secs=0.0, ttft_long=5.0, ttft_short=1.0, out_tps=tps,
                      tokens=1000, wall=10.0, valid=valid, moe=strategy, note=note,
                      decode_tps=decode_tps, interleave_tps=interleave_tps)


def test_strip_moe_args_removes_both_spellings_and_forms():
    argv = ["-m", "freetoken", "--moe-strategy", "hybrid", "--moe-backend=offload",
            "--tp-size", "2"]
    assert strip_moe_args(argv) == ["-m", "freetoken", "--tp-size", "2"]


def test_moe_strategies_for_auto_offers_the_offload_family():
    assert moe_strategies_for("auto") == [("offload", ""), ("hybrid", "")]
    assert moe_strategies_for("none") == []
    assert moe_strategies_for("hybrid") == [("hybrid", "")]
    with pytest.raises(ValueError):
        moe_strategies_for("fused")


def test_choose_moe_needs_a_tps_gain_without_a_stall_regression():
    base = _score("none", 1.0, 50.0, decode_tps=200.0, interleave_tps=130.0)
    # +20% concurrent tok/s, stall unchanged -> hybrid wins
    strategy, note = choose_moe([_moe_score("hybrid", 60.0, 1.0,
                                            decode_tps=240.0, interleave_tps=156.0)], base)
    assert strategy == "hybrid" and "+20.0%" in note
    # only +1% -> below MOE_MIN_GAIN
    strategy, note = choose_moe([_moe_score("hybrid", 50.5, 1.0,
                                            decode_tps=202.0, interleave_tps=131.3)], base)
    assert strategy is None and "tok/s" in note
    # +20% but worsens the stall beyond MOE_STALL_TOL -> rejected
    strategy, note = choose_moe([_moe_score("hybrid", 60.0, 1.5,
                                            decode_tps=240.0, interleave_tps=156.0)], base)
    assert strategy is None and "stall" in note
    # invalid arms never win; the faster valid strategy is chosen
    strategy, _ = choose_moe([_moe_score("hybrid", 70.0, 1.0, valid=False,
                                         decode_tps=280.0, interleave_tps=182.0),
                              _moe_score("offload", 55.0, 1.0,
                                         decode_tps=220.0, interleave_tps=143.0)], base)
    assert strategy == "offload"
    assert choose_moe([], base)[0] is None


def test_choose_moe_carries_the_decode_guard():
    base = _score("none", 1.0, 50.0, decode_tps=200.0, interleave_tps=130.0)
    # hybrid can win the mean on CPU compute while the pure-decode split regresses:
    # the per-regime floor rejects it
    strategy, note = choose_moe([_moe_score("hybrid", 60.0, 1.0,
                                            decode_tps=180.0, interleave_tps=170.0)], base)
    assert strategy is None and "decode-only" in note
    strategy, _ = choose_moe([_moe_score("hybrid", 60.0, 1.0,
                                         decode_tps=210.0, interleave_tps=156.0)], base)
    assert strategy == "hybrid"
    # the interleave probe guards the same way
    base = _score("none", 1.0, 50.0, decode_tps=200.0, interleave_tps=133.0)
    strategy, note = choose_moe([_moe_score("hybrid", 60.0, 1.0,
                                            decode_tps=240.0, interleave_tps=120.0)], base)
    assert strategy is None and "short-interleave" in note


def test_arm_active_checks_the_moe_strategy_marker():
    resolved = "Resolved config: moe_strategy='hybrid', attention_backend='qsa_sparse'"
    assert arm_active("moe-hybrid", resolved) == (True, "")
    ok, why = arm_active("moe-hybrid", "Resolved config: moe_strategy='offload'")
    assert not ok and "hybrid" in why
    # a cap-1 fallback still ran: valid, with the bench-bw hint in the note
    ok, note = arm_active("moe-hybrid", resolved + "\nusing a fixed fetch cap of 1 there")
    assert ok and "ft bench bw" in note
    assert arm_active("moe-offload", "Resolved config: moe_strategy='offload'")[0]
    # a combo is judged on its prefill part, not the strategy marker
    ok, why = arm_active("piped+moe-hybrid", "piping disabled (spec verify...)")
    assert not ok and "piping" in why
    assert arm_active("piped+moe-hybrid", "... piped prefill enabled ...")[0]


def test_render_script_applies_the_moe_flags_only_when_applied():
    spec = {"python": "/py", "argv": ["-m", "freetoken", "--moe-strategy", "offload",
                                      "--port", "1919"], "cwd": "/w", "env": {}, "port": 1919}
    chosen = {"env": winner_env("none"),
              "moe": {"applied": True, "argv": moe_argv("hybrid")}}
    text = render_script_multi(spec, chosen)
    # the profile's own strategy is stripped, the winner's added once
    assert text.count("--moe-strategy hybrid") == 1
    assert "--moe-strategy offload" not in text
    assert "exec /py -m freetoken --port 1919 --moe-strategy hybrid" in text
    # not applied -> no strategy flag at all (auto resolution stays in charge)
    off = render_script_multi(spec, {"env": winner_env("none"),
                                     "moe": {"applied": False, "argv": moe_argv("hybrid")}})
    assert "--moe-strategy" not in off
    # spec and moe winners stack in the script
    both = render_script_multi(spec, {"env": winner_env("none"),
                                      "spec": {"applied": True, "argv": spec_argv("mtp")},
                                      "moe": {"applied": True, "argv": moe_argv("hybrid")}})
    assert "--speculative-algorithm mtp --moe-strategy hybrid" in both
