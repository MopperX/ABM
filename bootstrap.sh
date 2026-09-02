#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
OLLAMA_INSTALL_ROOT="${OLLAMA_INSTALL_ROOT:-/usr/local}"
UPDATE_HOST=0

case "${1:-}" in
  "") ;;
  --update) UPDATE_HOST=1 ;;
  -h|--help)
    echo "Usage: ./bootstrap.sh [--update]"
    echo "  --update  upgrade OS packages, Ollama, and compatible Python dependencies"
    exit 0
    ;;
  *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
esac

log() { printf '\n==> %s\n' "$*"; }

install_homebrew() {
  if command -v brew >/dev/null 2>&1; then return; fi
  log "Installing Homebrew"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  else
    echo "ERROR: Homebrew installation completed without a usable brew executable." >&2
    exit 1
  fi
}

install_ollama() {
  local archive checksum actual release_json tmp version machine
  machine="$(uname -m)"
  if [[ "$OS" == "Darwin" ]]; then
    archive="ollama-darwin.tgz"
  elif [[ "$machine" == "x86_64" || "$machine" == "amd64" ]]; then
    archive="ollama-linux-amd64.tar.zst"
  elif [[ "$machine" == "aarch64" || "$machine" == "arm64" ]]; then
    archive="ollama-linux-arm64.tar.zst"
  else
    echo "ERROR: Ollama is not configured for architecture: $machine" >&2
    exit 1
  fi

  release_json="$(curl -fsSL --retry 3 https://api.github.com/repos/ollama/ollama/releases/latest)"
  version="$(jq -r '.tag_name | ltrimstr("v")' <<<"$release_json")"
  checksum="$(jq -r --arg archive "$archive" '.assets[] | select(.name == $archive) | .digest | ltrimstr("sha256:")' <<<"$release_json")"
  [[ -n "$version" && "$version" != "null" && -n "$checksum" && "$checksum" != "null" ]] || {
    echo "ERROR: latest Ollama release is missing metadata for $archive." >&2
    exit 1
  }
  if command -v ollama >/dev/null 2>&1 && ollama --version 2>&1 | grep -Fq "$version"; then
    return
  fi

  tmp="$(mktemp -d)"
  log "Installing Ollama $version"
  curl -fL --retry 3 -o "$tmp/$archive" "https://github.com/ollama/ollama/releases/download/v${version}/$archive"
  if [[ "$OS" == "Darwin" ]]; then
    actual="$(shasum -a 256 "$tmp/$archive" | awk '{print $1}')"
  else
    actual="$(sha256sum "$tmp/$archive" | awk '{print $1}')"
  fi
  [[ "$actual" == "$checksum" ]] || {
    rm -rf "$tmp"
    echo "ERROR: checksum verification failed for $archive" >&2
    exit 1
  }
  $SUDO mkdir -p "$OLLAMA_INSTALL_ROOT"
  if [[ "$OS" == "Darwin" ]]; then
    $SUDO tar -xzf "$tmp/$archive" -C "$OLLAMA_INSTALL_ROOT"
  else
    $SUDO tar --zstd -xf "$tmp/$archive" -C "$OLLAMA_INSTALL_ROOT"
  fi
  rm -rf "$tmp"
  command -v ollama >/dev/null 2>&1 && ollama --version 2>&1 | grep -Fq "$version" || {
    echo "ERROR: Ollama $version was installed but is not available in PATH." >&2
    exit 1
  }
}

if [[ "$OS" == "Linux" ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: Linux systems with apt (Ubuntu/Debian/WSL) are currently supported." >&2
    exit 1
  fi
  log "Installing base dependencies with apt"
  $SUDO apt-get update
  if ((UPDATE_HOST)); then
    log "Upgrading installed apt packages"
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
  fi
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git jq dialog python3 python3-venv python3-pip pciutils procps zstd

  install_ollama
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ai-benchmark}"
  log "Creating persistent results directory: $RESULTS_ROOT"
  mkdir -p "$RESULTS_ROOT/runs" "$RESULTS_ROOT/logs" "$RESULTS_ROOT/cache"
elif [[ "$OS" == "Darwin" ]]; then
  install_homebrew
  if ((UPDATE_HOST)); then
    log "Updating Homebrew packages"
    brew update
    brew upgrade
  fi
  log "Installing base dependencies with Homebrew"
  brew install git jq dialog python
  install_ollama
  brew services start ollama >/dev/null 2>&1 || true
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-$HOME/Library/Application Support/ai-benchmark}"
  log "Creating persistent results directory: $RESULTS_ROOT"
  mkdir -p "$RESULTS_ROOT/runs" "$RESULTS_ROOT/logs" "$RESULTS_ROOT/cache"
else
  echo "ERROR: supported platforms are Ubuntu/Debian, Ubuntu on WSL, and macOS." >&2
  exit 1
fi

log "Preparing Python virtual environment"
BOOTSTRAP_VENV="$(mktemp -d)"
python3 -m venv "$BOOTSTRAP_VENV"
"$BOOTSTRAP_VENV/bin/pip" install --upgrade pip uv >/dev/null
"$BOOTSTRAP_VENV/bin/uv" venv --clear --managed-python --python 3 "$ROOT/.venv"
rm -rf "$BOOTSTRAP_VENV"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip >/dev/null
if [[ -s "$ROOT/requirements.txt" ]]; then
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
fi
if ((UPDATE_HOST)); then
  log "Updating Python dependencies"
  bash "$ROOT/scripts/update-python-dependencies.sh"
fi

log "Scripts uitvoerbaar maken"
chmod +x "$ROOT/bootstrap.sh" "$ROOT/benchmark" "$ROOT/scripts/"*.sh "$ROOT/scripts/"*.py
find "$ROOT/benchmarks" -name '*.sh' -exec chmod +x {} + 2>/dev/null || true

cat <<TXT

Bootstrap complete.

Start interactively:
  ./benchmark

Or directly:
  ./benchmark start core --profile standard --machine <config-name>

Results:
  $RESULTS_ROOT
TXT
