#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

log() { printf '\n==> %s\n' "$*"; }

install_homebrew() {
  if command -v brew >/dev/null 2>&1; then return; fi
  log "Homebrew ontbreekt; installeren"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

if [[ "$OS" == "Linux" ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "FOUT: momenteel worden Linux-systemen met apt (Ubuntu/Debian/WSL) ondersteund." >&2
    exit 1
  fi
  log "Basisdependencies installeren via apt"
  $SUDO apt-get update
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git jq dialog python3 python3-venv python3-pip pciutils procps

  if ! command -v ollama >/dev/null 2>&1; then
    log "Ollama installeren"
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-/var/lib/ai-benchmark-v4}"
  log "Persistente resultaatmap aanmaken: $RESULTS_ROOT"
  $SUDO mkdir -p "$RESULTS_ROOT/runs" "$RESULTS_ROOT/logs" "$RESULTS_ROOT/cache"
  $SUDO chown -R "$(id -u):$(id -g)" "$RESULTS_ROOT"
elif [[ "$OS" == "Darwin" ]]; then
  install_homebrew
  log "Basisdependencies installeren via Homebrew"
  brew install git jq dialog python ollama
  brew services start ollama >/dev/null 2>&1 || true
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-/Library/Application Support/ai-benchmark-v4}"
  log "Persistente resultaatmap aanmaken: $RESULTS_ROOT"
  $SUDO mkdir -p "$RESULTS_ROOT/runs" "$RESULTS_ROOT/logs" "$RESULTS_ROOT/cache"
  $SUDO chown -R "$(id -u):$(id -g)" "$RESULTS_ROOT"
else
  echo "FOUT: ondersteund: Ubuntu/Debian, Ubuntu onder WSL, macOS." >&2
  exit 1
fi

log "Python virtual environment voorbereiden"
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip >/dev/null
if [[ -s "$ROOT/requirements.txt" ]]; then
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
fi

log "Scripts uitvoerbaar maken"
chmod +x "$ROOT/bootstrap.sh" "$ROOT/benchmark-v4" "$ROOT/scripts/"*.sh "$ROOT/scripts/"*.py
find "$ROOT/benchmarks" -name '*.sh' -exec chmod +x {} + 2>/dev/null || true

cat <<TXT

Bootstrap gereed.

Start interactief:
  ./benchmark-v4

Of direct:
  ./benchmark-v4 start core --profile standard --machine <config-naam>

Resultaten:
  $RESULTS_ROOT
TXT
