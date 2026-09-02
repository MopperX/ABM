#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
[[ -x "$PY" ]] || { echo "ERROR: base virtual environment is missing; run ./bootstrap.sh." >&2; exit 1; }

if ! "$PY" -c 'import pypdf' >/dev/null 2>&1; then
  echo "RAG dependency installeren: pypdf"
  "$PIP" install 'pypdf>=5,<7'
fi

EMBED_MODEL="embeddinggemma"
if ! ollama show "$EMBED_MODEL" >/dev/null 2>&1; then
  echo "RAG embeddingmodel downloaden: $EMBED_MODEL"
  ollama pull "$EMBED_MODEL"
else
  echo "RAG embeddingmodel aanwezig: $EMBED_MODEL"
fi

CACHE="${BENCH_CACHE_DIR:?BENCH_CACHE_DIR is missing}/beir"
DATA="$CACHE/scifact"
ZIP="$CACHE/scifact.zip"
mkdir -p "$CACHE"
if [[ ! -f "$DATA/corpus.jsonl" || ! -f "$DATA/queries.jsonl" || ! -f "$DATA/qrels/test.tsv" ]]; then
  echo "BEIR SciFact dataset downloaden voor externe retrievalcontrole"
  curl -fL --retry 3 -o "$ZIP.tmp" 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
  mv "$ZIP.tmp" "$ZIP"
  "$PY" - "$ZIP" "$CACHE" <<'PY'
import sys,zipfile
from pathlib import Path
z=Path(sys.argv[1]); out=Path(sys.argv[2])
with zipfile.ZipFile(z) as f: f.extractall(out)
PY
  "$PY" - "$ZIP" "$CACHE/scifact.sha256" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]); Path(sys.argv[2]).write_text(hashlib.sha256(p.read_bytes()).hexdigest()+"\n")
PY
fi
