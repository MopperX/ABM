#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
CORE_REQUIREMENTS="$ROOT/benchmarks/core/requirements.txt"
CORE_LOCK="$ROOT/benchmarks/core/requirements.lock"

[[ -x "$PY" && -x "$PIP" ]] || {
  echo "ERROR: benchmark Python environment is missing; run ./bootstrap.sh first." >&2
  exit 1
}

log() { printf '\n==> %s\n' "$*"; }

log "Updating Python packaging tools"
"$PIP" install --upgrade pip uv >/dev/null
PYTHON_VERSION="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

log "Updating shared suite dependencies within declared compatibility ranges"
"$PIP" install --upgrade --upgrade-strategy eager \
  'torch==2.13.0' 'torchvision==0.28.0' \
  'diffusers==0.40.0' 'transformers>=4.57,<5' 'accelerate>=1,<2' \
  'huggingface_hub>=0.28,<2' 'safetensors>=0.4,<1' 'Pillow>=10,<13' \
  'hpsv2==1.2.0' 'scipy>=1.11,<2' \
  'sentencepiece>=0.2,<1' 'soundfile>=0.12,<1' 'librosa>=0.11,<1' \
  'numpy>=1.26,<3' 'sherpa-onnx==1.13.7' 'pyarrow>=16,<24' \
  'pypdf>=5,<7'

log "Resolving Core dependencies for Python $PYTHON_VERSION"
"$PY" -m uv pip compile --upgrade --universal --python-version "$PYTHON_VERSION" --generate-hashes \
  --output-file "$CORE_LOCK" "$CORE_REQUIREMENTS"
"$PIP" install --require-hashes -r "$CORE_LOCK"

if [[ -x "$ROOT/.venv-lcb/bin/pip" ]]; then
  log "Updating LiveCodeBench dependencies within the EvalScope compatibility pin"
  "$ROOT/.venv-lcb/bin/pip" install --upgrade 'evalscope[sandbox]==1.11.1'
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-$HOME/Library/Application Support/ai-benchmark}"
else
  RESULTS_ROOT="${BENCH_RESULTS_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ai-benchmark}"
fi
while IFS= read -r web_pip; do
  log "Updating isolated SearXNG Python dependencies: $web_pip"
  "$web_pip" install --upgrade pyyaml msgspec typing-extensions pybind11
 done < <(find "$RESULTS_ROOT" -type f -path '*/searxng-venv/bin/pip' -perm -u+x 2>/dev/null)

log "Python dependency update complete"
