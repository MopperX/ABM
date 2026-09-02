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
  brew list cmake >/dev/null 2>&1 || brew install cmake
  brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
  brew list libsndfile >/dev/null 2>&1 || brew install libsndfile
else
  missing=()
  for p in build-essential cmake ffmpeg libsndfile1 bzip2 git; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if ((${#missing[@]})); then
    $SUDO apt-get update
    $SUDO apt-get install -y "${missing[@]}"
  fi
fi

# Python-side TTS, diarization and dataset preparation. Whisper inference itself uses whisper.cpp.
"$PIP" install -q \
  'sherpa-onnx==1.13.7' \
  'soundfile>=0.12,<1' 'numpy>=1.26,<3' \
  'datasets>=3,<5' 'huggingface_hub>=0.28,<2' 'pyarrow>=16,<24'

"$PY" "$ROOT/scripts/speech_prepare.py" \
  --profile "$PROFILE" \
  --cache-root "$BENCH_CACHE_DIR" \
  --machine-config "$BENCH_MACHINE_CONFIG"
