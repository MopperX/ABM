#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

if [[ "${1:-}" == "--background" ]]; then
  shift
  if [[ "$(uname -s)" == "Darwin" ]]; then
    STATE_ROOT="${BENCH_RESULTS_DIR:-$HOME/Library/Application Support/ai-benchmark}"
  else
    STATE_ROOT="${BENCH_RESULTS_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ai-benchmark}"
  fi
  mkdir -p "$STATE_ROOT"
  nohup "$PY" "$ROOT/scripts/web_ui.py" "$@" >"$STATE_ROOT/web-ui.log" 2>&1 < /dev/null &
  echo $! > "$STATE_ROOT/web-ui.pid"
  echo "AI Benchmark web UI started in the background (PID $!)."
  exit 0
fi

exec "$PY" "$ROOT/scripts/web_ui.py" "$@"
