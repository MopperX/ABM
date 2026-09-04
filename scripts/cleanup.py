#!/usr/bin/env python3
"""Remove disposable benchmark assets after a successfully completed run."""
from __future__ import annotations

import argparse
import csv
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


def selected_models(run_dir: Path, state: dict[str, Any]) -> list[dict[str, str]]:
    config = run_dir / "config" / "machine.models.tsv"
    if not config.exists():
        return []
    selected_suites = set(state.get("selected_benchmarks") or [])
    with config.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [
            row for row in rows
            if selected_suites.intersection(x.strip() for x in (row.get("suites") or "").split(",") if x.strip())
        ]


def removable_cache_dir(run_dir: Path, cache_dir: str | None) -> Path | None:
    """Accept only the standard cache location or the legacy results-root cache."""
    if not cache_dir:
        return None
    cache = Path(cache_dir).expanduser().resolve()
    legacy_cache = (run_dir.parents[1] / "cache").resolve()
    if cache.name == "benchmark-cache" or cache == legacy_cache:
        return cache
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interactive", action="store_true", help="ask about models selected by this run's machine scan")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    policy = state.get("cleanup") or {}
    report: dict[str, Any] = {"requested": bool(policy.get("on_success")), "interactive": args.interactive, "cache_removed": False, "models_removed": [], "errors": []}

    # This command is deliberately a no-op unless the runner has marked the run complete.
    if state.get("status") != "completed" or (not policy.get("on_success") and not args.interactive):
        report["skipped"] = "run is not completed or cleanup was not requested"
        atomic_json(run_dir / "summary" / "cleanup.json", report)
        return 0

    if args.interactive:
        run_models = selected_models(run_dir, state)
        selected_ollama = {row.get("model") for row in run_models if (row.get("backend") or "").lower() == "ollama"}
        if "rag" in set(state.get("selected_benchmarks") or []):
            selected_ollama.add("embeddinggemma")
        try:
            models = [model for model in ollama_models() if model in selected_ollama]
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            report["errors"].append(f"Could not list Ollama models: {exc}")
            report["status"] = "completed_with_errors"
            atomic_json(run_dir / "summary" / "cleanup.json", report)
            return 1

        keep: set[str] = set()
        print("Choose what to retain. Press Enter (or type y) to keep a model; type n to remove it.")
        for model in models:
            while True:
                answer = input(f"Keep '{model}'? [Y/n] ").strip().lower()
                if answer in {"", "y", "yes"}:
                    keep.add(model)
                    break
                if answer in {"n", "no"}:
                    break
                print("Please answer y or n.")
        policy = {"remove_ollama_models": True, "keep_ollama_models": sorted(keep)}
        # Interactive cleanup is the single post-run cleanup action: after choosing
        # Ollama models to retain, discard all reusable benchmark downloads too.
        policy["remove_cache"] = True
        report["requested"] = True

    if policy.get("remove_cache"):
        cache_dir = policy.get("cache_dir")
        if cache_dir:
            cache = removable_cache_dir(run_dir, cache_dir)
            if cache is None:
                report["errors"].append(f"Refusing to remove unexpected cache directory: {Path(cache_dir).expanduser().resolve()}")
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
