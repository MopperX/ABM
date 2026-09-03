#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
SELECTED="${1:-core}"
MACHINE_CONFIG="${2:?machine config required}"
WITH_LIVECODEBENCH="${3:-0}"
BENCH_PROFILE="${4:-standard}"
export BENCH_PROFILE
export BENCH_MACHINE_CONFIG="$MACHINE_CONFIG"

if [[ "$(uname -s)" == "Darwin" ]]; then
  BENCH_RESULTS_ROOT="${BENCH_RESULTS_DIR:-$HOME/Library/Application Support/ai-benchmark}"
else
  BENCH_RESULTS_ROOT="${BENCH_RESULTS_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ai-benchmark}"
fi
export BENCH_RESULTS_ROOT
if [[ -z "${BENCH_CACHE_DIR:-}" ]]; then
  ollama_models_dir="${OLLAMA_MODELS:-}"
  if [[ -z "$ollama_models_dir" ]] && command -v systemctl >/dev/null 2>&1; then
    ollama_models_dir="$(systemctl show ollama --property=Environment --value 2>/dev/null | sed -n 's/.*OLLAMA_MODELS=\([^ ]*\).*/\1/p' | head -n1)"
  fi
  export BENCH_CACHE_DIR="${ollama_models_dir:-$HOME/.ollama/models}/benchmark-cache"
else
  export BENCH_CACHE_DIR
fi
BENCH_MIN_FREE_GB="${BENCH_MIN_FREE_GB:-20}"

[[ "$BENCH_MIN_FREE_GB" =~ ^[0-9]+$ ]] || {
  echo "ERROR: BENCH_MIN_FREE_GB must be a non-negative whole number." >&2
  exit 2
}
available_kib="$(df -Pk "$BENCH_RESULTS_ROOT" | awk 'NR==2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] || {
  echo "ERROR: could not determine free disk space for $BENCH_RESULTS_ROOT." >&2
  exit 1
}
required_kib=$((BENCH_MIN_FREE_GB * 1024 * 1024))
if ((available_kib < required_kib)); then
  available_gb=$((available_kib / 1024 / 1024))
  echo "ERROR: only ${available_gb} GiB is free in $BENCH_RESULTS_ROOT; ${BENCH_MIN_FREE_GB} GiB is required." >&2
  echo "Set BENCH_MIN_FREE_GB to override the preflight threshold." >&2
  exit 1
fi

need_cmd() { command -v "$1" >/dev/null 2>&1; }

if ! need_cmd python3; then
  echo "ERROR: python3 is missing. Run ./bootstrap.sh first." >&2
  exit 1
fi
if ! need_cmd curl; then
  echo "ERROR: curl is missing. Run ./bootstrap.sh first." >&2
  exit 1
fi
if ! need_cmd ollama; then
  echo "ERROR: Ollama is missing. Run ./bootstrap.sh first." >&2
  exit 1
fi

"$REPO_ROOT/.venv/bin/python" - "$REPO_ROOT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from lib.benchlib import host_snapshot

for warning in host_snapshot()["readiness"]["warnings"]:
    print(f"WARNING: host readiness: {warning}", file=sys.stderr)
PY

# Ensure Ollama server is reachable before detaching the benchmark.
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama is not running; starting the service..."
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew services start ollama >/dev/null 2>&1 || true
  elif command -v systemctl >/dev/null 2>&1; then
    $SUDO systemctl start ollama >/dev/null 2>&1 || true
  fi
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    nohup ollama serve >/tmp/benchmark-ollama.log 2>&1 &
  fi
  for _ in {1..30}; do
    curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null || {
  echo "ERROR: Ollama API is unreachable." >&2
  exit 1
}

# Module dependency hooks are executed while the user is still attached, so sudo prompts cannot
# strand a headless run. Hooks will be added as modules are implemented.
IFS=',' read -r -a benches <<< "$SELECTED"
for bench in "${benches[@]}"; do
  hook="$REPO_ROOT/benchmarks/$bench/dependencies.sh"
  if [[ -x "$hook" ]]; then
    "$hook"
  fi
done

if [[ "$WITH_LIVECODEBENCH" == "1" ]]; then
  "$REPO_ROOT/benchmarks/coding-agent/livecodebench-dependencies.sh"
fi

# Pull only enabled models relevant to selected benchmarks.
python3 - "$MACHINE_CONFIG" "$SELECTED" "$REPO_ROOT" <<'PY'
import subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from lib.benchlib import parse_model_rows
path = Path(sys.argv[1])
selected = set(sys.argv[2].split(','))
models=[]
for r in parse_model_rows(path):
    if r['backend'].lower() != 'ollama':
        continue
    suites=set(x.strip() for x in r['suites'].split(',') if x.strip())
    web=r['web'].lower() in {'1','true','yes','on'}
    relevant=bool(selected & suites)
    if 'web' in selected and 'web' in suites and not web:
        relevant=False
    if relevant:
        models.append(r['model'])
for model in dict.fromkeys(models):
  print(f'Refreshing model tag: {model}', flush=True)
  subprocess.run(['ollama', 'pull', model], check=True)
PY
