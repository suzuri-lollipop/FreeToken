"""Start the server from a launch JSON, the same spec file calibrate_lookup.py reads.

The launcher (llm-launcher) runs this JSON directly from a profile; this module gives a
checkout a standalone path - `python -m freetoken.serve_spec freetoken-launch.json` -
so the JSON, not the shell command line, is the single source of truth for a start.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_spec(path: Path) -> dict:
    # utf-8-sig: Windows editors leave a BOM that plain json.loads chokes on
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict) or not spec.get("argv"):
        raise SystemExit(f"the launch spec needs a non-empty 'argv': {path}")
    return spec


def replace_arg(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    for i, tok in enumerate(out):
        if tok == flag and i + 1 < len(out):
            out[i + 1] = value
            return out
    return out + [flag, value]


def build(spec: dict, port: int, api_key: str) -> tuple[list[str], dict[str, str], str]:
    argv = [str(a) for a in spec["argv"]]
    if port:
        argv = replace_arg(argv, "--port", str(int(port)))
    if api_key:
        argv = replace_arg(argv, "--api-key", api_key)
    cmd = [str(spec.get("python") or sys.executable), *argv]
    env = {**os.environ, **{str(k): str(v) for k, v in (spec.get("env") or {}).items()}}
    # a checkout without an editable install still resolves `python -m freetoken` this way
    root_py = str(ROOT / "python")
    pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root_py + (os.pathsep + pp if pp else "")
    cwd = str(spec.get("cwd") or ROOT)
    return cmd, env, (cwd if Path(cwd).is_dir() else str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m freetoken.serve_spec",
                                 description="start the server from a launch JSON")
    ap.add_argument("config", help="launch JSON (python/argv/cwd/env/port/model_name)")
    ap.add_argument("--port", type=int, default=0, help="override the JSON --port")
    ap.add_argument("--api-key", default="", help="override the JSON --api-key")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not start")
    a = ap.parse_args(argv)

    cmd, env, cwd = build(load_spec(Path(a.config)), a.port, a.api_key)
    if a.dry_run:
        print(f"cwd: {cwd}")
        print("$ " + " ".join(shlex.quote(x) for x in cmd))
        return 0
    os.chdir(cwd)
    # exec keeps signals and the exit code between the user's shell and the server
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())
