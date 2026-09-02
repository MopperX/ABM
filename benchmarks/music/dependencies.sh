#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; PIP="$ROOT/.venv/bin/pip"
PROFILE="${BENCH_PROFILE:-standard}"
: "${BENCH_CACHE_DIR:?BENCH_CACHE_DIR missing}"
: "${BENCH_MACHINE_CONFIG:?BENCH_MACHINE_CONFIG missing}"
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  command -v brew >/dev/null 2>&1 || { echo "FOUT: Homebrew ontbreekt; voer eerst ./bootstrap.sh uit." >&2; exit 1; }
  brew list libsndfile >/dev/null 2>&1 || brew install libsndfile
else
  missing=()
  for p in libsndfile1 ffmpeg; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if ((${#missing[@]})); then
    $SUDO apt-get update
    $SUDO apt-get install -y "${missing[@]}"
  fi
fi

"$PIP" install -q \
  'torch==2.13.0' \
  'transformers>=4.57,<5' 'accelerate>=1,<2' \
  'huggingface_hub>=0.28,<2' 'safetensors>=0.4,<1' 'sentencepiece>=0.2,<1' \
  'soundfile>=0.12,<1' 'librosa>=0.11,<1' 'numpy>=1.26,<3' 'scipy>=1.11,<2'

export HF_HOME="$BENCH_CACHE_DIR/music/hf-home"
"$PY" "$ROOT/scripts/music_prepare.py" \
  --profile "$PROFILE" \
  --cache-root "$BENCH_CACHE_DIR" \
  --machine-config "$BENCH_MACHINE_CONFIG"
