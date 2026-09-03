#!/usr/bin/env python3
"""Remove disposable benchmark assets after a successfully completed run."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ollama_models() -> list[str]:
    completed = subprocess.run(
        ["ollama", "list", "--format", "json"], text=True, capture_output=True, check=True
    )
    names = []
    for line in completed.stdout.splitlines():
        row = json.loads(line)
        name = row.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    policy = state.get("cleanup") or {}
    report: dict[str, Any] = {"requested": bool(policy.get("on_success")), "cache_removed": False, "models_removed": [], "errors": []}

    # This command is deliberately a no-op unless the runner has marked the run complete.
    if state.get("status") != "completed" or not policy.get("on_success"):
        report["skipped"] = "run is not completed or cleanup was not requested"
        atomic_json(run_dir / "summary" / "cleanup.json", report)
        return 0

    if policy.get("remove_cache"):
        cache_dir = policy.get("cache_dir")
        if cache_dir:
            cache = Path(cache_dir).expanduser().resolve()
            if cache.name != "benchmark-cache":
                report["errors"].append(f"Refusing to remove unexpected cache directory: {cache}")
            elif cache.exists():
                try:
                    shutil.rmtree(cache)
                    report["cache_removed"] = True
                    report["cache_dir"] = str(cache)
                except OSError as exc:
                    report["errors"].append(f"Could not remove cache {cache}: {exc}")

    keep = set(policy.get("keep_ollama_models") or [])
    if policy.get("remove_ollama_models"):
        try:
            for model in ollama_models():
                if model in keep:
                    continue
                completed = subprocess.run(["ollama", "rm", model], text=True, capture_output=True)
                if completed.returncode == 0:
                    report["models_removed"].append(model)
                else:
                    report["errors"].append(f"Could not remove {model}: {completed.stderr.strip() or completed.stdout.strip()}")
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            report["errors"].append(f"Could not list Ollama models: {exc}")

    report["kept_ollama_models"] = sorted(keep)
    report["status"] = "completed_with_errors" if report["errors"] else "completed"
    atomic_json(run_dir / "summary" / "cleanup.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
