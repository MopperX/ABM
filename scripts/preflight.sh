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
  BENCH_RESULTS_ROOT="${BENCH_RESULTS_DIR:-/Library/Application Support/ai-benchmark-v4}"
else
  BENCH_RESULTS_ROOT="${BENCH_RESULTS_DIR:-/var/lib/ai-benchmark-v4}"
fi
export BENCH_RESULTS_ROOT
export BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-$BENCH_RESULTS_ROOT/cache}"

need_cmd() { command -v "$1" >/dev/null 2>&1; }

if ! need_cmd python3; then
  echo "FOUT: python3 ontbreekt. Voer eerst ./bootstrap.sh uit." >&2
  exit 1
fi
if ! need_cmd curl; then
  echo "FOUT: curl ontbreekt. Voer eerst ./bootstrap.sh uit." >&2
  exit 1
fi
if ! need_cmd ollama; then
  echo "FOUT: Ollama ontbreekt. Voer eerst ./bootstrap.sh uit." >&2
  exit 1
fi

# Ensure Ollama server is reachable before detaching the benchmark.
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama draait niet; service wordt gestart..."
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew services start ollama >/dev/null 2>&1 || true
  elif command -v systemctl >/dev/null 2>&1; then
    $SUDO systemctl start ollama >/dev/null 2>&1 || true
  fi
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    nohup ollama serve >/tmp/ai-benchmark-v4-ollama.log 2>&1 &
  fi
  for _ in {1..30}; do
    curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null || {
  echo "FOUT: Ollama API is niet bereikbaar." >&2
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
python3 - "$MACHINE_CONFIG" "$SELECTED" <<'PY'
import csv, subprocess, sys
from pathlib import Path
path = Path(sys.argv[1])
selected = set(sys.argv[2].split(','))
with path.open(encoding='utf-8', newline='') as f:
    rows = (line for line in f if line.strip() and not line.lstrip().startswith('#'))
    reader = csv.DictReader(rows, delimiter='\t', fieldnames=['enabled','model','benchmarks','modes','web','notes'])
    models=[]
    for r in reader:
        if (r['enabled'] or '').strip().lower() not in {'1','true','yes','on'}:
            continue
        benches=set(x.strip() for x in (r['benchmarks'] or '').split(',') if x.strip())
        web=(r['web'] or '').strip().lower() in {'1','true','yes','on'}
        if (selected & benches) or ('web' in selected and web):
            models.append(r['model'].strip())
for model in dict.fromkeys(models):
    chk = subprocess.run(['ollama','show',model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if chk.returncode != 0:
        print(f'Model ontbreekt; downloaden: {model}', flush=True)
        subprocess.run(['ollama','pull',model], check=True)
    else:
        print(f'Model aanwezig: {model}')
PY
