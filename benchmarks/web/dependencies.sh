#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
: "${BENCH_CACHE_DIR:?BENCH_CACHE_DIR missing}"
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

CACHE="$BENCH_CACHE_DIR/web"
SRC="$CACHE/searxng-src"
VENV="$CACHE/searxng-venv"
SETTINGS="$CACHE/settings.yml"
PREPARED="$CACHE/prepared.json"
LOG="$CACHE/searxng.log"
REVISION="d226b78bc"
mkdir -p "$CACHE"

if [[ "$(uname -s)" == "Darwin" ]]; then
  command -v brew >/dev/null 2>&1 || { echo "ERROR: Homebrew is missing; run ./bootstrap.sh." >&2; exit 1; }
  for pkg in libxml2 libxslt openssl@3; do
    brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"
  done
else
  missing=()
  for pkg in python3-dev build-essential libxslt1-dev zlib1g-dev libffi-dev libssl-dev; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
  done
  if ((${#missing[@]})); then
    $SUDO apt-get update
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
  fi
fi

if [[ ! -d "$SRC/.git" ]]; then
  echo "Downloading SearXNG source code (one time)"
  rm -rf "$SRC"
  git clone --quiet https://github.com/searxng/searxng.git "$SRC"
fi
if ! git -C "$SRC" cat-file -e "$REVISION^{commit}" 2>/dev/null; then
  git -C "$SRC" fetch --quiet origin master
fi
git -C "$SRC" checkout --quiet --detach "$REVISION"
RESOLVED="$(git -C "$SRC" rev-parse HEAD)"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip setuptools wheel pyyaml msgspec typing-extensions pybind11
# Editable install is the upstream-supported development/runtime path and keeps the exact checked-out source auditable.
"$VENV/bin/pip" install -q --use-pep517 --no-build-isolation -e "$SRC"

# Pick a dedicated loopback port. Reuse the previous port only when it still answers as SearXNG;
# otherwise choose a genuinely free port so an unrelated local service is never overwritten.
PORT=""
if [[ -f "$PREPARED" ]]; then
  OLD_PORT="$($PY - "$PREPARED" <<'PY'
import json,sys
try:
    v=json.load(open(sys.argv[1])).get('port')
    print(v if isinstance(v,int) else '')
except Exception:
    print('')
PY
)"
  if [[ -n "$OLD_PORT" ]]; then
    if curl -fsS --max-time 5 --get "http://127.0.0.1:$OLD_PORT/search" --data-urlencode 'q=ollama' --data 'format=json' | "$PY" -c 'import json,sys; x=json.load(sys.stdin); assert isinstance(x.get("results"),list)' >/dev/null 2>&1; then
      PORT="$OLD_PORT"
    fi
  fi
fi
if [[ -z "$PORT" ]]; then
  PORT="$($PY - <<'PY'
import socket
for port in range(18888,18899):
    s=socket.socket()
    try:
        s.bind(('127.0.0.1',port)); print(port); break
    except OSError:
        pass
    finally:
        s.close()
else:
    raise SystemExit('ERROR: no free SearXNG benchmark port in range 18888-18898')
PY
)"
fi

cat > "$SETTINGS" <<EOF2
use_default_settings: true
general:
  debug: false
  instance_name: "Benchmark SearXNG"
search:
  safe_search: 0
  formats:
    - html
    - json
server:
  bind_address: "127.0.0.1"
  port: $PORT
  limiter: false
  image_proxy: false
  secret_key: "benchmark-local-only-$PORT"
valkey:
  url: false
EOF2

URL="http://127.0.0.1:$PORT"
valid_search() {
  curl -fsS --max-time 8 --get "$URL/search" --data-urlencode 'q=ollama' --data 'format=json' | "$PY" -c 'import json,sys; x=json.load(sys.stdin); assert isinstance(x.get("results"),list)' >/dev/null 2>&1
}
if ! valid_search; then
  if [[ -f "$CACHE/searxng.pid" ]]; then
    oldpid="$(cat "$CACHE/searxng.pid" 2>/dev/null || true)"
    [[ -n "$oldpid" ]] && kill "$oldpid" 2>/dev/null || true
  fi
  echo "Starting local SearXNG at $URL"
  nohup env SEARXNG_SETTINGS_PATH="$SETTINGS" "$VENV/bin/python" -m searx.webapp >"$LOG" 2>&1 < /dev/null &
  echo $! > "$CACHE/searxng.pid"
  for _ in {1..45}; do
    valid_search && break
    sleep 1
  done
fi
valid_search || { echo "ERROR: local SearXNG is unusable; see $LOG" >&2; exit 1; }

"$PY" - "$PREPARED" "$URL" "$PORT" "$RESOLVED" "$SRC" "$SETTINGS" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
out=Path(sys.argv[1])
data={
  'backend':'searxng','url':sys.argv[2],'port':int(sys.argv[3]),'revision':sys.argv[4],
  'source_path':sys.argv[5],'settings_path':sys.argv[6],
  'prepared_at':datetime.now(timezone.utc).isoformat(timespec='seconds')
}
out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps(data,indent=2))
PY
