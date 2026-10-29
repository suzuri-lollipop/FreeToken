#!/bin/sh
# FreeToken fork bootstrap: the first run creates .venv and installs the package; later runs just start the server.
# --model downloads the model into Freetoken-data next to this folder (like Strata-data / Maya-data), writes the
# launch JSON and run.sh, so every later run can start with ./run.sh alone.
# With no args and no --model the start comes from freetoken-launch.json (Strata-style launch JSON); args after the
# flags go to `python -m freetoken` instead and win over the JSON.
# The launcher UI (llm-launcher) runs the same steps itself via its install button, so this file is for terminal use.
cd "$(dirname "$0")" || exit 1

START=1
REINSTALL=0
EXTRAS="${FREETOKEN_EXTRAS-accel}"   # FREETOKEN_EXTRAS= installs core deps only
ARGS=""                              # pass-through serve args, newline-separated (never eval'd)
MODEL=""                             # --model: HF repo id or local folder to download / run
DATA_DIR="${FREETOKEN_DATA_DIR-$(cd .. && pwd)/Freetoken-data}"
while [ $# -gt 0 ]; do
  a="$1"; shift
  case "$a" in
    --no-start)   START=0 ;;
    --reinstall)  REINSTALL=1 ;;
    --extras=*)   EXTRAS="${a#--extras=}" ;;
    --model=*)    MODEL="${a#--model=}" ;;
    --model)      MODEL="${1-}"; [ $# -gt 0 ] && shift ;;
    --data-dir=*) DATA_DIR="${a#--data-dir=}" ;;
    --data-dir)   DATA_DIR="${1-}"; [ $# -gt 0 ] && shift ;;
    -h|--help)
      echo "usage: ./setup.sh [--no-start] [--reinstall] [--extras=accel|fi|sgl|] [--model ID|PATH] [--data-dir DIR]"
      echo "       [server args, e.g. --port 8080]"
      echo "  --model     download (resumable) from Hugging Face into <data-dir>/models and write run.sh"
      echo "  --data-dir  where models live (default: ../Freetoken-data)"
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

# first run in a terminal asks which model to run, like Strata's setup does; scripts and args keep working untouched
if [ -z "$MODEL" ] && [ -z "$ARGS" ] && [ "$START" = 1 ] && [ -t 0 ]; then
  if [ ! -f freetoken-launch.json ] || grep -q "/path/to/model" freetoken-launch.json; then
    printf 'Model to run (Hugging Face repo id or local folder, empty = only write the json template): '
    read -r MODEL
  fi
fi

# resolve --model to a local folder: a path is used as is, an 'org/model' id downloads resumably
MODEL_DIR=""
if [ -n "$MODEL" ]; then
  if [ -d "$MODEL" ]; then
    MODEL_DIR="$MODEL"
    MODEL_NAME="$(basename "$MODEL")"
  else
    case "$MODEL" in
      */*) ;;
      *) echo "--model '$MODEL' is neither a folder nor an 'org/model' Hugging Face id" >&2; exit 1 ;;
    esac
    MODEL_NAME="$(printf '%s' "$MODEL" | sed 's|.*/||')"
    MODEL_DIR="$DATA_DIR/models/$(printf '%s' "$MODEL" | sed 's|/|--|g')"
    mkdir -p "$MODEL_DIR" || exit 1
    echo "downloading $MODEL -> $MODEL_DIR (interrupt and rerun to resume)"
    # partial files stay on a failure: snapshot_download continues from the .cache metadata in the folder
    .venv/bin/python - "$MODEL" "$MODEL_DIR" <<'PY' || exit 1
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
  fi
fi

# the launch JSON (the start's single source of truth). An explicit --model rewrites it; the template
# is written only when absent and a user-edited file is never touched.
write_json() {
  cat > freetoken-launch.json <<EOF
{
 "python": "$PWD/.venv/bin/python",
 "argv": ["-m", "freetoken", "--model-path", "$1", "--port", "1919"],
 "cwd": "$PWD",
 "env": {},
 "port": 1919,
 "model_name": "$2"
}
EOF
}
if [ -n "$MODEL_DIR" ]; then
  write_json "$MODEL_DIR" "$MODEL_NAME"
  echo "wrote freetoken-launch.json for $MODEL_NAME"
elif [ ! -f freetoken-launch.json ]; then
  write_json "/path/to/model" ""
  echo "wrote freetoken-launch.json: edit --model-path, then ./setup.sh starts from it (or point a launcher profile at it)"
fi

# run.sh: the Strata run-<model>.sh equivalent, regenerated whenever the json holds a real model
if [ -f freetoken-launch.json ] && ! grep -q "/path/to/model" freetoken-launch.json; then
  cat > run.sh <<EOF
#!/bin/sh
# generated by setup.sh: starts the model recorded in freetoken-launch.json (server args after it win)
cd "$PWD"
exec "$PWD/.venv/bin/python" -m freetoken.serve_spec "$PWD/freetoken-launch.json" "\$@"
EOF
  chmod +x run.sh
fi

[ "$START" = 1 ] || exit 0
if [ -z "$ARGS" ] && [ -f freetoken-launch.json ] && ! grep -q "/path/to/model" freetoken-launch.json; then
  exec ./run.sh
fi
if [ -z "$ARGS" ]; then
  echo "no model set yet: run ./setup.sh --model <repo id>, edit --model-path in freetoken-launch.json, or pass server args" >&2
  exit 1
fi
# rebuild the positional parameters from ARGS (newline-split) instead of eval; set -f stops glob expansion
OIFS=$IFS; set -f; IFS='
'
set --
for a in $ARGS; do set -- "$@" "$a"; done
# a --model was consumed by this script: hand the engine the downloaded folder unless args already name one
if [ -n "$MODEL_DIR" ]; then
  case "$ARGS" in *--model*) ;; *) set -- --model-path "$MODEL_DIR" "$@" ;; esac
fi
IFS=$OIFS; set +f
exec .venv/bin/python -m freetoken "$@"
