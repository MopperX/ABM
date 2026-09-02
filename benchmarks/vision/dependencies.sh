#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; PIP="$ROOT/.venv/bin/pip"
PROFILE="${BENCH_PROFILE:-standard}"
: "${BENCH_CACHE_DIR:?BENCH_CACHE_DIR missing}"
"$PIP" install -q 'Pillow>=10,<13' 'datasets>=3,<5' 'huggingface_hub>=0.28,<2'
export HF_HOME="$BENCH_CACHE_DIR/vision/hf-home"
export HF_DATASETS_CACHE="$BENCH_CACHE_DIR/vision/hf-datasets"
"$PY" "$ROOT/scripts/vision_prepare.py" --profile "$PROFILE" --cache-root "$BENCH_CACHE_DIR"
