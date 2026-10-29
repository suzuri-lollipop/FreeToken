#!/bin/sh
# FreeToken fork bootstrap: the first run creates .venv and installs the package; later runs just start the server.
# With no args the start comes from freetoken-launch.json (Strata-style launch JSON); args after the flags
# go to `python -m freetoken` instead and win over it (e.g. ./setup.sh --model deepseek-ai/DeepSeek-V4).
# The launcher UI (llm-launcher) runs the same steps itself via its install button, so this file is for terminal use.
cd "$(dirname "$0")" || exit 1

START=1
REINSTALL=0
EXTRAS="${FREETOKEN_EXTRAS-accel}"   # FREETOKEN_EXTRAS= installs core deps only
ARGS=""                              # pass-through serve args, newline-separated (never eval'd)
for a in "$@"; do
  case "$a" in
    --no-start)  START=0 ;;
    --reinstall) REINSTALL=1 ;;
    --extras=*)  EXTRAS="${a#--extras=}" ;;
    -h|--help)
      echo "usage: ./setup.sh [--no-start] [--reinstall] [--extras=accel|fi|sgl|] [server args, e.g. --model ID]"
      exit 0 ;;
    *) ARGS="$ARGS$a
" ;;
  esac
done

# the repo standard is uv (AGENTS.md); without uv fall back to the stock venv (python3 -m venv)
make_venv() {
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv || { echo "could not create .venv with uv"; exit 1; }
    return
  fi
  # python 3.10+ WITH pip inside the venv: Debian/Ubuntu split `venv` from `ensurepip` (python3-venv package)
  ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
  PY=""
  for c in python3 python; do
    if command -v $c >/dev/null 2>&1 && ok_py $c; then PY=$c; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed. On Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "On Fedora: sudo dnf install python3 python3-pip - On Arch: sudo pacman -S python python-pip"
    exit 1
  fi
  $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
}

[ "$REINSTALL" = 1 ] && rm -rf .venv
[ ! -x .venv/bin/python ] && make_venv

# idempotent: an editable install already pointing here survives every later run
if ! .venv/bin/python -c 'import freetoken' >/dev/null 2>&1; then
  # an install IS needed: a venv without pip (uv-made, or made on Debian without python3-venv)
  # cannot run pip without uv - replace it, but only now, so a working venv is never deleted
  if ! command -v uv >/dev/null 2>&1 && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    rm -rf .venv; make_venv
  fi
  if [ -n "$EXTRAS" ]; then SPEC=".[$EXTRAS]"; else SPEC="."; fi
  # PyPI torch 2.14.1 is already the cu130 build, so plain PyPI resolution is correct (no index pin needed)
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -e "$SPEC" || exit 1
  else
    .venv/bin/python -m pip install --progress-bar off -U pip >/dev/null || exit 1
    .venv/bin/python -m pip install --progress-bar off -e "$SPEC" || exit 1
  fi
  .venv/bin/python -c 'import freetoken' || { echo "install finished but 'import freetoken' still fails; see the log above"; exit 1; }
fi

# launch JSON template (the start's single source of truth); a user-edited file is never touched
if [ ! -f freetoken-launch.json ]; then
  cat > freetoken-launch.json <<EOF
{
 "python": "$PWD/.venv/bin/python",
 "argv": ["-m", "freetoken", "--model-path", "/path/to/model", "--port", "1919"],
 "cwd": "$PWD",
 "env": {},
 "port": 1919,
 "model_name": ""
}
EOF
  echo "wrote freetoken-launch.json: edit --model-path, then ./setup.sh starts from it (or point a launcher profile at it)"
fi

[ "$START" = 1 ] || exit 0
if [ -z "$ARGS" ] && [ -f freetoken-launch.json ]; then
  if grep -q "/path/to/model" freetoken-launch.json; then
    echo "edit --model-path in freetoken-launch.json first, or pass server args to ./setup.sh" >&2
    exit 1
  fi
  exec .venv/bin/python -m freetoken.serve_spec freetoken-launch.json
fi
if [ -n "$ARGS" ]; then
  # rebuild the positional parameters from ARGS (newline-split) instead of eval; set -f stops glob expansion
  OIFS=$IFS; set -f; IFS='
'
  set --
  for a in $ARGS; do set -- "$@" "$a"; done
  IFS=$OIFS; set +f
fi
exec .venv/bin/python -m freetoken "$@"
