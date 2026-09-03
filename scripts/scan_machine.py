#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.benchlib import atomic_json, host_snapshot, parse_model_rows, utc_now

GIB = 1024**3


def available_vram_gib() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return sum(float(line.strip()) for line in output.splitlines() if line.strip()) / 1024
    except Exception:
        return None


def available_ram_gib(snapshot: dict) -> float | None:
    value = (snapshot.get("readiness") or {}).get("memory_available_bytes")
    if value is None:
        value = snapshot.get("memory")
    return (float(value) / GIB) if value is not None else None


def ollama_models_dir(configured: str | None) -> Path:
    if configured:
        return Path(configured).expanduser()
    if os.environ.get("OLLAMA_MODELS"):
        return Path(os.environ["OLLAMA_MODELS"]).expanduser()
    if shutil.which("systemctl"):
        try:
            environment = subprocess.check_output(
                ["systemctl", "show", "ollama", "--property=Environment", "--value"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            for value in shlex.split(environment):
                if value.startswith("OLLAMA_MODELS="):
                    return Path(value.split("=", 1)[1]).expanduser()
        except Exception:
            pass
    return Path("~/.ollama/models").expanduser()


def storage_usage(path: Path) -> dict[str, float | str]:
    storage_path = path.resolve()
    while not storage_path.exists():
        storage_path = storage_path.parent
    usage = shutil.disk_usage(storage_path)
    return {
        "path": str(path),
        "used_gib": usage.used / GIB,
        "free_gib": usage.free / GIB,
        "total_gib": usage.total / GIB,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess catalog models against current machine resources.")
    parser.add_argument("--models", required=True, help="Path to the global models.toml catalog")
    parser.add_argument("--output", required=True, help="Path for the local JSON scan report")
    parser.add_argument("--eligible-config", required=True, help="Path for the scan-generated eligible model TSV")
    parser.add_argument("--ollama-models-dir", help="Ollama model storage directory; defaults to OLLAMA_MODELS or the Ollama service setting")
    parser.add_argument("--report", action="store_true", help="Print allowed and excluded models")
    args = parser.parse_args()

    models_path = Path(args.models).resolve()
    catalog = tomllib.loads(models_path.read_text(encoding="utf-8"))
    snapshot = host_snapshot()
    ram_gib = available_ram_gib(snapshot)
    vram_gib = available_vram_gib()
    results_dir = Path(args.output).resolve().parent
    results_storage = storage_usage(results_dir)
    ollama_dir = ollama_models_dir(args.ollama_models_dir)
    ollama_storage = storage_usage(ollama_dir)

    findings = []
    eligible_rows = []
    for raw, row in zip(catalog.get("models", []), parse_model_rows(models_path, enabled_only=False)):
        required_ram = float(raw.get("minimum_ram_gib", 0))
        required_vram = float(raw.get("minimum_vram_gib", 0))
        required_disk = float(raw.get("minimum_free_disk_gib", 0))
        model_disk_gib = ollama_storage["free_gib"] if row["backend"].lower() == "ollama" else results_storage["free_gib"]
        model_disk_location = ollama_dir if row["backend"].lower() == "ollama" else results_dir
        ram_ok = ram_gib is not None and ram_gib >= required_ram
        vram_ok = vram_gib is not None and vram_gib >= required_vram
        disk_ok = model_disk_gib >= required_disk
        allowed = ram_ok and disk_ok and (vram_ok or bool(raw.get("allow_cpu_ram", False)))
        eligibility = "gpu-resident" if allowed and vram_ok else "cpu-ram" if allowed else "insufficient-resources"
        result = {"model": row["model"], "allowed": allowed, "eligibility": eligibility, "requirements": {"ram_gib": required_ram, "vram_gib": required_vram, "free_disk_gib": required_disk}, "disk_location": str(model_disk_location)}
        if allowed:
            eligible_rows.append(row)
            result["reason"] = "model meets the declared machine requirements"
        else:
            reasons = []
            if not ram_ok:
                reasons.append(f"requires {required_ram:g} GiB RAM; {ram_gib:.2f} GiB available" if ram_gib is not None else "available RAM could not be determined")
            if not disk_ok:
                reasons.append(f"requires {required_disk:g} GiB free disk at {model_disk_location}; {model_disk_gib:.2f} GiB available")
            if not vram_ok and not bool(raw.get("allow_cpu_ram", False)):
                reasons.append(f"requires {required_vram:g} GiB VRAM; {vram_gib:.2f} GiB available" if vram_gib is not None else "available VRAM could not be determined")
            result["reason"] = "; ".join(reasons)
        findings.append(result)

    report = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "models_config": str(models_path),
        "available_ram_gib": round(ram_gib, 2) if ram_gib is not None else None,
        "available_vram_gib": round(vram_gib, 2) if vram_gib is not None else None,
        "results_storage": {key: round(value, 2) if isinstance(value, float) else value for key, value in results_storage.items()},
        "ollama_models_storage": {key: round(value, 2) if isinstance(value, float) else value for key, value in ollama_storage.items()},
        "models": findings,
    }
    atomic_json(Path(args.output), report)
    with Path(args.eligible_config).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eligible_rows[0]) if eligible_rows else [], delimiter="\t")
        if eligible_rows:
            writer.writeheader()
            for row in eligible_rows:
                writer.writerow({key: value for key, value in row.items()})
    if args.report:
        print("Storage")
        print(f"Ollama models | {ollama_storage['path']} | used {ollama_storage['used_gib']:.2f} GiB | free {ollama_storage['free_gib']:.2f} GiB | total {ollama_storage['total_gib']:.2f} GiB")
        print(f"Results and cache | {results_storage['path']} | used {results_storage['used_gib']:.2f} GiB | free {results_storage['free_gib']:.2f} GiB | total {results_storage['total_gib']:.2f} GiB")
        print("Allowed models")
        allowed_disk_minimum_gib = sum(result["requirements"]["free_disk_gib"] for result in findings if result["allowed"])
        for result in findings:
            if result["allowed"]:
                requirements = result["requirements"]
                print(f"{result['model']} | RAM minimum {requirements['ram_gib']:g} GiB | disk minimum {requirements['free_disk_gib']:g} GiB")
        print(f"Allowed models total disk minimum | {allowed_disk_minimum_gib:g} GiB")
        print("Excluded models")
        for result in findings:
            if not result["allowed"]:
                print(f"{result['model']} | {result['reason']}")
    print(f"Wrote machine scan: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())