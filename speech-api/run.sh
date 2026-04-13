#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
mkdir -p "$HF_HOME"
ORTCAP="$("$ROOT/.venv/bin/python" -c "import onnxruntime,os; print(os.path.join(os.path.dirname(onnxruntime.__file__),\"capi\"))")"
export LD_LIBRARY_PATH="${ORTCAP}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
exec "$ROOT/.venv/bin/uvicorn" app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-3000}" "$@"
