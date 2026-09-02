#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OS="$(uname -s)"
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

ensure_docker_linux() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "LiveCodeBench sandbox: Docker installeren"
    $SUDO apt-get update
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io login
  fi
  if ! $SUDO docker info >/dev/null 2>&1; then
    if command -v systemctl >/dev/null 2>&1; then $SUDO systemctl start docker >/dev/null 2>&1 || true; fi
    if ! $SUDO docker info >/dev/null 2>&1; then
      echo "Docker daemon starten"
      $SUDO sh -c 'nohup dockerd >/var/tmp/ai-benchmark-v4-dockerd.log 2>&1 &' || true
      for _ in $(seq 1 30); do $SUDO docker info >/dev/null 2>&1 && break; sleep 1; done
    fi
  fi
  $SUDO docker info >/dev/null 2>&1 || { echo "FOUT: Docker daemon kon niet worden gestart." >&2; exit 1; }
  if ! docker info >/dev/null 2>&1; then
    user="$(id -un)"
    if [[ "$user" != "root" ]]; then
      $SUDO usermod -aG docker "$user"
      sg docker -c 'docker info >/dev/null 2>&1' || {
        echo "FOUT: Docker is geïnstalleerd maar de docker-groep kan niet zonder nieuwe login worden geactiveerd." >&2
        exit 1
      }
    fi
  fi
}

ensure_docker_macos() {
  command -v brew >/dev/null 2>&1 || { echo "FOUT: Homebrew ontbreekt; voer ./bootstrap.sh uit." >&2; exit 1; }
  command -v docker >/dev/null 2>&1 || brew install docker
  command -v colima >/dev/null 2>&1 || brew install colima
  if ! docker info >/dev/null 2>&1; then
    echo "LiveCodeBench sandbox: Colima starten"
    colima start --cpu 2 --memory 4 --disk 20
  fi
  docker info >/dev/null 2>&1 || { echo "FOUT: Docker/Colima is niet bereikbaar." >&2; exit 1; }
}

if [[ "$OS" == "Linux" ]]; then ensure_docker_linux
elif [[ "$OS" == "Darwin" ]]; then ensure_docker_macos
else echo "FOUT: LiveCodeBench sandbox ondersteunt Linux/WSL en macOS." >&2; exit 1
fi

VENV="$ROOT/.venv-lcb"
if [[ ! -x "$VENV/bin/python" ]]; then python3 -m venv "$VENV"; fi
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
current="$($VENV/bin/python -c 'import importlib.metadata as m; print(m.version("evalscope"))' 2>/dev/null || true)"
if [[ "$current" != "1.11.1" ]]; then
  echo "LiveCodeBench evaluator installeren: EvalScope 1.11.1 + sandbox"
  "$VENV/bin/pip" install --upgrade 'evalscope[sandbox]==1.11.1'
fi

# Pull the exact sandbox base image while the user is still attached.
if docker info >/dev/null 2>&1; then docker pull python:3.11-slim >/dev/null
else sg docker -c 'docker pull python:3.11-slim >/dev/null'; fi

echo "LiveCodeBench sandbox gereed (EvalScope 1.11.1, python:3.11-slim)."
