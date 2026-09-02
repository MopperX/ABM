#!/usr/bin/env bash
set -euo pipefail
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
OS="$(uname -s)"
if [[ "$OS" == "Linux" ]]; then
  missing=()
  command -v php >/dev/null 2>&1 || missing+=(php-cli)
  command -v git >/dev/null 2>&1 || missing+=(git)
  if ((${#missing[@]})); then
    echo "Installing Coding & Agent dependencies: ${missing[*]}"
    $SUDO apt-get update
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
  fi
elif [[ "$OS" == "Darwin" ]]; then
  command -v brew >/dev/null 2>&1 || { echo "ERROR: Homebrew is missing; run ./bootstrap.sh." >&2; exit 1; }
  pkgs=()
  command -v php >/dev/null 2>&1 || pkgs+=(php)
  command -v git >/dev/null 2>&1 || pkgs+=(git)
  if ((${#pkgs[@]})); then
    echo "Installing Coding & Agent dependencies: ${pkgs[*]}"
    brew install "${pkgs[@]}"
  fi
fi
