#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; PIP="$ROOT/.venv/bin/pip"
PROFILE="${BENCH_PROFILE:-standard}"
: "${BENCH_CACHE_DIR:?BENCH_CACHE_DIR missing}"
: "${BENCH_MACHINE_CONFIG:?BENCH_MACHINE_CONFIG missing}"

# Current stable cross-platform stack as of benchmark v4 build.
"$PIP" install -q \
  'torch==2.13.0' 'torchvision==0.28.0' \
  'diffusers==0.40.0' 'transformers>=4.57,<5' 'accelerate>=1,<2' \
  'huggingface_hub>=0.28,<2' 'safetensors>=0.4,<1' 'Pillow>=10,<13' \
  'hpsv2==1.2.0' 'scipy>=1.11,<2'

export HF_HOME="$BENCH_CACHE_DIR/image/hf-home"
export HPS_ROOT="$BENCH_CACHE_DIR/image/hpsv2"
"$PY" "$ROOT/scripts/image_prepare.py" \
  --profile "$PROFILE" \
  --cache-root "$BENCH_CACHE_DIR" \
  --machine-config "$BENCH_MACHINE_CONFIG"
