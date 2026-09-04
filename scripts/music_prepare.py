#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.benchlib import parse_model_rows

MUSICBENCH_REPO = "amaai-lab/MusicBench"
MUSICBENCH_REV = "b141e962aacc19ffd51c15732738040377989203"
MUSICBENCH_FILE = "MusicBench_test_B.json"
CLAP_MODEL = "laion/clap-htsat-fused"
CLAP_REV = "365dea6ef167def6676140ed93bbc43f84dabb28"


def atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def music_config_for(machine_cfg: Path, repo: Path) -> Path:
    return machine_cfg


def parse_models(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in parse_model_rows(path):
        suites={x.strip() for x in row['suites'].split(',') if x.strip()}
        if 'music' not in suites or row['backend'].lower() != 'transformers-musicgen':
            continue
        out.append({
            'enabled':row['enabled'], 'backend':row['backend'], 'model':row['model'],
            'revision':row['revision'] or 'main', 'capabilities':row['capabilities'] or 'text',
            'guidance':row['guidance'] or '3.0', 'top_k':row['top_k'] or '250',
            'temperature':row['temperature'] or '1.0', 'notes':row['notes'],
        })
    return out

def even_subset(rows: list[Any], n: int) -> list[tuple[int, Any]]:
    if not rows or n <= 0:
        return []
    if n >= len(rows):
        return list(enumerate(rows))
    if n == 1:
        return [(0, rows[0])]
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    seen: list[int] = []
    for i in idx:
        if i not in seen:
            seen.append(i)
    return [(i, rows[i]) for i in seen]


def _hf_model_snapshot(repo_id: str, revision: str, cache_dir: Path) -> tuple[str, str]:
    from huggingface_hub import model_info, snapshot_download

    info = model_info(repo_id, revision=revision)
    allow = [
        "*.json", "*.model", "*.txt", "*.safetensors", "tokenizer.*", "spiece.model",
        "merges.txt", "vocab.json", "preprocessor_config.json", "special_tokens_map.json",
    ]
    path = snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        cache_dir=str(cache_dir),
        allow_patterns=allow,
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.ckpt"],
    )
    return path, info.sha


def prepare_models(models: list[dict[str, str]], cache: Path) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for m in models:
        backend = m["backend"] or "transformers-musicgen"
        if backend != "transformers-musicgen":
            prepared.append({**m, "status": "unsupported-backend", "local_path": None, "resolved_revision": None})
            continue
        repo_id = m["model"]
        rev = m["revision"] or "main"
        print(f"Music-model cachen: {repo_id}@{rev}", flush=True)
        path, sha = _hf_model_snapshot(repo_id, rev, cache / "hf-models")
        prepared.append({**m, "status": "ready", "local_path": path, "resolved_revision": sha})
    return prepared


def prepare_clap(cache: Path) -> dict[str, Any]:
    print(f"CLAP evaluator cachen: {CLAP_MODEL}@{CLAP_REV}", flush=True)
    path, sha = _hf_model_snapshot(CLAP_MODEL, CLAP_REV, cache / "hf-models")
    return {"model": CLAP_MODEL, "revision": CLAP_REV, "resolved_revision": sha, "local_path": path}


def _normalise_musicbench_rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for key in ("data", "test", "samples", "rows"):
            if isinstance(raw.get(key), list):
                return [x for x in raw[key] if isinstance(x, dict)]
        if all(isinstance(v, dict) for v in raw.values()):
            return list(raw.values())
    raise RuntimeError("Unknown MusicBench JSON format")


def _load_musicbench_rows(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON document or the JSONL format used by MusicBench test-B."""
    text = path.read_text(encoding="utf-8")
    try:
        return _normalise_musicbench_rows(json.loads(text))
    except json.JSONDecodeError as document_error:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as line_error:
                raise RuntimeError(
                    f"MusicBench data is neither valid JSON nor valid JSONL: {path} "
                    f"(line {line_number})"
                ) from line_error
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"MusicBench JSONL record is not an object: {path} (line {line_number})"
                )
            rows.append(row)
        if rows:
            return rows
        raise RuntimeError(f"MusicBench data contains no records: {path}") from document_error


def prepare_musicbench(profile: str, cache: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(
        repo_id=MUSICBENCH_REPO,
        repo_type="dataset",
        filename=MUSICBENCH_FILE,
        revision=MUSICBENCH_REV,
        cache_dir=str(cache / "hf-datasets"),
    )
    rows = _load_musicbench_rows(Path(f))
    n = {"quick": 5, "standard": 20, "full": min(400, len(rows))}[profile]
    selected = even_subset(rows, n)
    samples: list[dict[str, Any]] = []
    for j, (idx, row) in enumerate(selected):
        prompt = str(row.get("main_caption") or row.get("alt_caption") or "").strip()
        key = row.get("key")
        if isinstance(key, list):
            expected_key = " ".join(str(x) for x in key[:2]).strip() or None
        else:
            expected_key = str(key).strip() if key else None
        bpm = row.get("bpm")
        try:
            bpm = float(bpm) if bpm is not None else None
        except Exception:
            bpm = None
        samples.append({
            "id": f"musicbench-{j:04d}",
            "source_index": idx,
            "prompt": prompt,
            "bpm": bpm,
            "key": expected_key,
            "prompt_bpm": row.get("prompt_bpm"),
            "prompt_key": row.get("prompt_key"),
            "prompt_bt": row.get("prompt_bt"),
            "prompt_ch": row.get("prompt_ch"),
            "location": row.get("location"),
        })
    manifest = {
        "source": MUSICBENCH_REPO,
        "revision": MUSICBENCH_REV,
        "file": MUSICBENCH_FILE,
        "profile": profile,
        "source_count": len(rows),
        "count": len(samples),
        "samples": samples,
    }
    atomic(cache / "musicbench" / profile / "manifest.json", manifest)
    return manifest


def prepare_melody_fixture(cache: Path) -> dict[str, Any]:
    out = cache / "fixtures" / "benchmark-melody.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sr = 32000
    notes = [261.6256, 329.6276, 392.0, 523.2511, 392.0, 329.6276, 293.6648, 261.6256]
    note_s = 0.75
    parts = []
    for freq in notes:
        n = int(sr * note_s)
        t = np.arange(n, dtype=np.float32) / sr
        env = np.ones(n, dtype=np.float32)
        fade = min(int(sr * 0.04), n // 4)
        if fade:
            env[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            env[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
        sig = 0.20 * np.sin(2 * math.pi * freq * t) + 0.04 * np.sin(2 * math.pi * freq * 2 * t)
        parts.append((sig * env).astype(np.float32))
    audio = np.concatenate(parts)
    sf.write(out, audio, sr, subtype="PCM_16")
    meta = {"path": str(out), "sample_rate": sr, "duration_seconds": len(audio) / sr, "notes_hz": notes}
    atomic(cache / "fixtures" / "benchmark-melody.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["quick", "standard", "full"], required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--machine-config", required=True)
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cache = Path(a.cache_root) / "music"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = music_config_for(Path(a.machine_config), repo)
    models = parse_models(cfg)
    prepared = prepare_models(models, cache)
    clap = prepare_clap(cache)
    mb = prepare_musicbench(a.profile, cache)
    melody = prepare_melody_fixture(cache)
    atomic(
        cache / "prepared.json",
        {
            "profile": a.profile,
            "music_model_config": str(cfg),
            "models": prepared,
            "clap": clap,
            "musicbench_manifest": str(cache / "musicbench" / a.profile / "manifest.json"),
            "melody_fixture": melody,
        },
    )
    print(f"Music preflight complete: {len(prepared)} model(s), CLAP, {mb['count']} MusicBench prompts.")


if __name__ == "__main__":
    main()
