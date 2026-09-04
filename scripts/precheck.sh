#!/usr/bin/env bash
set -euo pipefail

# The benchmark start command deliberately does not run this automatically.
# Invoke this script first to prepare dependencies, caches and selected models.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# >= 4)); then
  exec "$ROOT/scripts/preflight.sh" "$@"
fi

SELECTED="${1:-core}"
PROFILE="${2:-standard}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-$HOME/Library/Application Support/ai-benchmark}"
else
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ai-benchmark}"
fi
MACHINE="$(hostname -s 2>/dev/null || hostname)"
SCAN_DIR="$RESULTS_ROOT/scans/$MACHINE"
mkdir -p "$SCAN_DIR"
CONFIG="${BENCH_MODELS_CONFIG:-$ROOT/config/models.toml}"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
"$PY" "$ROOT/scripts/scan_machine.py" --models "$CONFIG" --output "$SCAN_DIR/latest.json" --eligible-config "$SCAN_DIR/eligible.models.tsv"
WITH_LIVECODEBENCH=0
[[ ",$SELECTED," == *,coding-agent,* ]] && WITH_LIVECODEBENCH=1
exec "$ROOT/scripts/preflight.sh" "$SELECTED" "$SCAN_DIR/eligible.models.tsv" "$WITH_LIVECODEBENCH" "$PROFILE"
