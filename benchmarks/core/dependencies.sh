#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
PIP="$REPO_ROOT/.venv/bin/pip"
RESULTS_ROOT="${BENCH_RESULTS_ROOT:?BENCH_RESULTS_ROOT is required}"
CACHE_ROOT="${BENCH_CACHE_DIR:-$RESULTS_ROOT/cache}/core"
mkdir -p "$CACHE_ROOT"

IFEVAL_REVISION="641dd8404c65862627fa38865775694ee8b5c572"
MMLU_PRO_REVISION="24ac2da5bb7c7b42ea1a984c6b535e35a73d30b3"

printf '\n==> Core externe benchmarkdependencies controleren\n'
if ! "$PY" -c 'import absl, langdetect, nltk, immutabledict, datasets' >/dev/null 2>&1; then
  "$PIP" install -r "$REPO_ROOT/benchmarks/core/requirements.txt"
else
  echo "Python dependencies aanwezig."
fi

printf '\n==> NLTK-data voor IFEval voorbereiden\n'
export NLTK_DATA="$CACHE_ROOT/nltk"
mkdir -p "$NLTK_DATA"
"$PY" - "$NLTK_DATA" <<'PY'
import sys, nltk
out=sys.argv[1]
nltk.data.path.insert(0, out)
for name, resource in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(name, download_dir=out, quiet=True, raise_on_error=True)
        except Exception:
            if name == "punkt":
                raise
PY

printf '\n==> IFEval referentie-implementatie voorbereiden\n'
IFEVAL_REPO="$CACHE_ROOT/ifeval/google-research"
if [[ ! -d "$IFEVAL_REPO/.git" ]]; then
  rm -rf "$IFEVAL_REPO"
  mkdir -p "$(dirname "$IFEVAL_REPO")"
  git clone --quiet --filter=blob:none --no-checkout https://github.com/google-research/google-research.git "$IFEVAL_REPO"
  git -C "$IFEVAL_REPO" sparse-checkout init --cone
  git -C "$IFEVAL_REPO" sparse-checkout set instruction_following_eval
fi
if [[ "$(git -C "$IFEVAL_REPO" rev-parse HEAD 2>/dev/null || true)" != "$IFEVAL_REVISION" ]]; then
  git -C "$IFEVAL_REPO" fetch --quiet origin "$IFEVAL_REVISION" --depth=1
  git -C "$IFEVAL_REPO" checkout --quiet --detach "$IFEVAL_REVISION"
  git -C "$IFEVAL_REPO" sparse-checkout reapply >/dev/null 2>&1 || true
fi

printf '\n==> TruthfulQA binary dataset voorbereiden\n'
TRUTH_REPO="$CACHE_ROOT/truthfulqa/TruthfulQA"
if [[ ! -d "$TRUTH_REPO/.git" ]]; then
  rm -rf "$TRUTH_REPO"
  mkdir -p "$(dirname "$TRUTH_REPO")"
  git clone --quiet --depth=1 https://github.com/sylinrl/TruthfulQA.git "$TRUTH_REPO"
fi
# Bewust geen git pull: de eerste opgehaalde commit blijft lokaal gefixeerd totdat de cache bewust wordt verwijderd.
TRUTH_COMMIT="$(git -C "$TRUTH_REPO" rev-parse HEAD)"

printf '\n==> MMLU-Pro dataset voorbereiden (vaste revision)\n'
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
mkdir -p "$CACHE_ROOT/mmlu-pro"
"$PY" - "$CACHE_ROOT/mmlu-pro" "$MMLU_PRO_REVISION" <<'PY'
import json, sys
from pathlib import Path
from datasets import load_dataset

out = Path(sys.argv[1]); revision = sys.argv[2]
test_path = out / "test.jsonl"
val_path = out / "validation.jsonl"
meta_path = out / "source.json"
if not (test_path.exists() and val_path.exists() and meta_path.exists()):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", revision=revision, cache_dir=str(out.parent / "huggingface"))
    for split, path in [("test", test_path), ("validation", val_path)]:
        with path.open("w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    meta_path.write_text(json.dumps({
        "dataset": "TIGER-Lab/MMLU-Pro",
        "revision": revision,
        "test_rows": len(ds["test"]),
        "validation_rows": len(ds["validation"]),
    }, indent=2) + "\n", encoding="utf-8")
PY

"$PY" - "$CACHE_ROOT" "$IFEVAL_REVISION" "$TRUTH_COMMIT" "$MMLU_PRO_REVISION" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
ifeval_rev, truth_rev, mmlu_rev=sys.argv[2:5]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

ifeval_data=root/'ifeval/google-research/instruction_following_eval/data/input_data.jsonl'
truth_data=root/'truthfulqa/TruthfulQA/TruthfulQA.csv'
mmlu_test=root/'mmlu-pro/test.jsonl'
meta={
  "ifeval": {"source":"google-research/google-research", "revision":ifeval_rev, "sha256":sha(ifeval_data)},
  "truthfulqa": {"source":"sylinrl/TruthfulQA", "revision":truth_rev, "sha256":sha(truth_data)},
  "mmlu_pro": {"source":"TIGER-Lab/MMLU-Pro", "revision":mmlu_rev, "sha256":sha(mmlu_test)},
}
(root/'sources.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
print(json.dumps(meta, indent=2))
PY
