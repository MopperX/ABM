from __future__ import annotations

import csv
import json
import os
import platform
import re
import socket
import subprocess
import tempfile
import time
import threading
import tomllib
import hashlib
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib import request, error

BENCHMARKS = ["core", "coding-agent", "rag", "vision", "image", "speech", "music", "web"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_cache_root(run_dir: Path) -> Path:
    configured = os.environ.get("BENCH_CACHE_DIR")
    return Path(configured).expanduser() if configured else run_dir.parents[2] / "cache"


def parse_defaults(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


MODEL_CONFIG_FIELDS = [
    "enabled", "model", "backend", "suites", "modes", "web", "revision", "capabilities",
    "steps", "guidance", "offload", "top_k", "temperature", "language", "speaker", "notes",
]


def parse_model_rows(path: Path, *, enabled_only: bool = True) -> list[dict[str, str]]:
    if path.suffix == ".toml":
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        rows = document.get("models", [])
        if not isinstance(rows, list):
            raise ValueError(f"models must be an array of tables: {path}")
        rows_out: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"model entry must be a table: {path}")
            clean = {
                field: ",".join(str(value) for value in row[field]) if isinstance(row.get(field), list)
                else str(row.get(field, "")).strip()
                for field in MODEL_CONFIG_FIELDS
            }
            if isinstance(row.get("enabled"), bool):
                clean["enabled"] = "true" if row["enabled"] else "false"
            if enabled_only and clean["enabled"].lower() not in {"1", "true", "yes", "on"}:
                continue
            if clean["model"]:
                rows_out.append(clean)
        return rows_out

    rows_out: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t", fieldnames=MODEL_CONFIG_FIELDS)
        for row in reader:
            clean = {k: (row.get(k) or "").strip() for k in MODEL_CONFIG_FIELDS}
            if enabled_only and clean["enabled"].lower() not in {"1", "true", "yes", "on"}:
                continue
            if not clean["model"]:
                continue
            rows_out.append(clean)
    return rows_out


@dataclass(frozen=True)
class ModelConfig:
    model: str
    benchmarks: tuple[str, ...]
    modes: tuple[str, ...]
    web: bool
    notes: str = ""


def parse_machine_models(path: Path) -> list[ModelConfig]:
    """Return enabled Ollama models used by LLM-style benchmark suites.

    Image, speech and music rows live in the same machine TSV but are consumed by their
    modality-specific runners.
    """
    models: list[ModelConfig] = []
    for row in parse_model_rows(path):
        if row["backend"].lower() != "ollama":
            continue
        suites = tuple(x.strip() for x in row["suites"].split(",") if x.strip())
        modes = tuple(x.strip() for x in (row["modes"] or "standard").split(",") if x.strip()) or ("standard",)
        web = row["web"].lower() in {"1", "true", "yes", "on"}
        models.append(ModelConfig(row["model"], suites, modes, web, row["notes"]))
    return models


def host_snapshot() -> dict[str, Any]:
    def cmd(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, timeout=5).strip() or None
        except Exception:
            return None

    def read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip() or None
        except OSError:
            return None

    def meminfo_bytes(name: str) -> int | None:
        meminfo = read_text(Path("/proc/meminfo"))
        if meminfo is None:
            return None
        for line in meminfo.splitlines():
            key, _, value = line.partition(":")
            if key == name:
                try:
                    return int(value.strip().split()[0]) * 1024
                except (IndexError, ValueError):
                    return None
        return None

    system = platform.system()
    snapshot: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": system,
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "captured_at": utc_now(),
    }
    cpu_count = os.cpu_count() or 1
    try:
        load_average = os.getloadavg()
    except OSError:
        load_average = None
    readiness: dict[str, Any] = {"cpu_count": cpu_count, "load_average": load_average, "warnings": []}
    if load_average is not None and load_average[0] > cpu_count * 0.75:
        readiness["warnings"].append("one-minute CPU load exceeds 75% of logical CPUs")
    if system == "Linux":
        readiness["memory_available_bytes"] = meminfo_bytes("MemAvailable")
        if readiness["memory_available_bytes"] is not None and readiness["memory_available_bytes"] < 2 * 1024**3:
            readiness["warnings"].append("less than 2 GiB of memory is available")
    docker_containers = cmd(["docker", "ps", "-q"])
    readiness["active_docker_containers"] = docker_containers.splitlines() if docker_containers else []
    if readiness["active_docker_containers"]:
        readiness["warnings"].append("Docker has active containers")
    gpu_processes = cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    readiness["active_nvidia_compute_processes"] = gpu_processes.splitlines() if gpu_processes else []
    if readiness["active_nvidia_compute_processes"]:
        readiness["warnings"].append("NVIDIA GPU has active compute processes")
    snapshot["readiness"] = readiness
    snapshot["docker_version"] = cmd(["docker", "version", "--format", "{{.Server.Version}}"])
    if system == "Linux":
        snapshot["kernel"] = cmd(["uname", "-a"])
        snapshot["cpu"] = cmd(["bash", "-lc", "lscpu | awk -F: '/Model name/ {sub(/^[ \\t]+/,\"\",$2); print $2; exit}'"])
        snapshot["memory"] = cmd(["bash", "-lc", "free -b | awk '/Mem:/ {print $2}'"])
        snapshot["nvidia_smi"] = cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pci.link.gen.current,pci.link.width.current", "--format=csv,noheader"])
        kernel_release = read_text(Path("/proc/sys/kernel/osrelease")) or ""
        proc_version = read_text(Path("/proc/version")) or ""
        snapshot["wsl"] = "microsoft" in kernel_release.lower() or "microsoft" in proc_version.lower() or Path("/etc/wsl.conf").exists()
        snapshot["swap_total_bytes"] = meminfo_bytes("SwapTotal")
        snapshot["zram_devices"] = [path.name for path in sorted(Path("/sys/block").glob("zram*"))]
        governors = {value for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor") if (value := read_text(path))}
        snapshot["cpu_governors"] = sorted(governors)
        power_supplies = []
        for supply in sorted(Path("/sys/class/power_supply").glob("*")):
            supply_type = read_text(supply / "type")
            online = read_text(supply / "online")
            status = read_text(supply / "status")
            capacity = read_text(supply / "capacity")
            if supply_type or online or status or capacity:
                power_supplies.append({"name": supply.name, "type": supply_type, "online": online, "status": status, "capacity_percent": capacity})
        snapshot["power_supplies"] = power_supplies
    elif system == "Darwin":
        snapshot["cpu"] = cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or cmd(["sysctl", "-n", "hw.model"])
        snapshot["memory"] = cmd(["sysctl", "-n", "hw.memsize"])
        snapshot["macos"] = cmd(["sw_vers", "-productVersion"])
        snapshot["power_source"] = cmd(["pmset", "-g", "batt"])
    snapshot["ollama_version"] = cmd(["ollama", "--version"])
    return snapshot


def _json_request(url: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_model_identity(api: str, model: str) -> dict[str, Any]:
    """Capture the immutable and behavior-relevant identity of an installed model."""
    tags = _json_request(api.rstrip("/") + "/api/tags")
    candidates = [m for m in tags.get("models", []) if m.get("name") == model or m.get("model") == model]
    if not candidates:
        raise RuntimeError(f"Ollama model is not installed: {model}")
    tag = candidates[0]
    shown = _json_request(api.rstrip("/") + "/api/show", {"model": model, "verbose": False})
    template = shown.get("template") or ""
    parameters = shown.get("parameters") or ""
    info = shown.get("model_info") or {}
    model_context = next((v for k, v in info.items() if k.endswith(".context_length")), None)
    return {
        "model": model,
        "digest": tag.get("digest"),
        "size_bytes": tag.get("size"),
        "modified_at": tag.get("modified_at"),
        "details": tag.get("details") or shown.get("details"),
        "capabilities": shown.get("capabilities"),
        "model_context_length": model_context,
        "model_info": info,
        "template": template,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "parameters": parameters,
        "parameters_sha256": hashlib.sha256(parameters.encode("utf-8")).hexdigest(),
        "captured_at": utc_now(),
    }


def ollama_runtime_snapshot(api: str, model: str) -> dict[str, Any] | None:
    try:
        data = _json_request(api.rstrip("/") + "/api/ps")
        for item in data.get("models", []):
            if item.get("name") == model or item.get("model") == model:
                size = int(item.get("size") or 0)
                size_vram = int(item.get("size_vram") or 0)
                return {
                    "model": model,
                    "digest": item.get("digest"),
                    "size_bytes": size or None,
                    "size_vram_bytes": size_vram or None,
                    "vram_fraction": (size_vram / size) if size else None,
                    "allocated_context_length": item.get("context_length"),
                    "details": item.get("details"),
                    "captured_at": utc_now(),
                }
    except Exception:
        return None
    return None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {"level": 0.95, "low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def distribution_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = [float(v) for v in values if v is not None]
    if not rows:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None, "standard_deviation": None, "coefficient_of_variation": None}
    mean = statistics.mean(rows)
    sd = statistics.stdev(rows) if len(rows) > 1 else 0.0
    return {
        "count": len(rows), "mean": mean, "median": statistics.median(rows),
        "minimum": min(rows), "maximum": max(rows), "standard_deviation": sd,
        "coefficient_of_variation": (sd / mean) if mean else None,
    }


def call_performance_summary(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build the standard additive performance view from raw call records."""
    rows = list(calls)
    metrics = [row.get("metrics") or {} for row in rows]
    powers = [row.get("power") or {} for row in rows]
    task_wall = sum(float(m.get("wall_seconds") or 0) for m in metrics)
    task_energy = sum(float(p.get("estimated_gpu_energy_wh") or p.get("approx_energy_wh") or 0) for p in powers if p.get("available"))
    return {
        "generation_tokens_per_second": distribution_summary(m.get("generation_tokens_per_second") for m in metrics),
        "prompt_tokens_per_second": distribution_summary(m.get("prompt_tokens_per_second") for m in metrics),
        "wall_seconds": distribution_summary(m.get("wall_seconds") for m in metrics),
        "load_seconds": distribution_summary((float(m["load_duration_ns"]) / 1e9) if m.get("load_duration_ns") is not None else None for m in metrics),
        "estimated_gpu_energy_wh": distribution_summary(p.get("estimated_gpu_energy_wh") for p in powers if p.get("available")),
        "joules_per_output_token": distribution_summary(p.get("joules_per_output_token") for p in powers if p.get("available")),
        "task_totals": {"calls": len(rows), "wall_seconds": task_wall, "estimated_gpu_energy_wh": task_energy if any(p.get("available") for p in powers) else None},
    }


def mode_to_think(mode: str) -> tuple[bool, Any]:
    m = mode.strip().lower()
    if m == "standard":
        return False, None
    if m == "thinking":
        return True, True
    if m in {"nothinking", "no-thinking", "off"}:
        return True, False
    if m in {"low", "medium", "high", "max"}:
        return True, m
    raise ValueError(f"Unsupported reasoning mode: {mode}")




def _read_nvidia_sample() -> dict[str, float] | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu,clocks.sm,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=3
        )
        rows=[]
        for line in out.splitlines():
            try:
                values=[float(x.strip()) for x in line.split(",")]
                rows.append(values)
            except (ValueError, IndexError):
                pass
        if not rows:
            return None
        return {
            "power_w": sum(x[0] for x in rows),
            "temperature_c": max(x[1] for x in rows),
            "utilization_percent": statistics.mean(x[2] for x in rows),
            "sm_clock_mhz": statistics.mean(x[3] for x in rows),
            "memory_used_mb": sum(x[4] for x in rows),
            "memory_total_mb": sum(x[5] for x in rows),
        }
    except Exception:
        return None


class PowerSampler:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        first = _read_nvidia_sample()
        if first is None:
            return
        self.samples.append(first)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            value = _read_nvidia_sample()
            if value is not None:
                self.samples.append(value)

    def stop(self, elapsed_s: float) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"source": None, "available": False}
        powers = [x["power_w"] for x in self.samples]
        avg = statistics.mean(powers)
        return {
            "source": "nvidia-smi",
            "available": True,
            "samples": len(self.samples),
            "average_w": avg,
            "peak_w": max(powers),
            "estimated_gpu_energy_wh": avg * elapsed_s / 3600.0,
            "approx_energy_wh": avg * elapsed_s / 3600.0,
            "temperature_c": distribution_summary(x["temperature_c"] for x in self.samples),
            "utilization_percent": distribution_summary(x["utilization_percent"] for x in self.samples),
            "sm_clock_mhz": distribution_summary(x["sm_clock_mhz"] for x in self.samples),
            "memory_used_mb": distribution_summary(x["memory_used_mb"] for x in self.samples),
            "memory_total_mb": max(x["memory_total_mb"] for x in self.samples),
        }


_WARMED_CONFIGURATIONS: set[tuple[str, str, int]] = set()
_WARMUP_LOCK = threading.Lock()


def _warm_ollama(api: str, model: str, mode: str, context: int) -> dict[str, Any] | None:
    key = (model, mode, context)
    with _WARMUP_LOCK:
        if key in _WARMED_CONFIGURATIONS:
            return ollama_runtime_snapshot(api, model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": context, "num_predict": 1},
        }
        include_think, think_value = mode_to_think(mode)
        if include_think:
            payload["think"] = think_value
        _json_request(api.rstrip("/") + "/api/chat", payload, timeout=600)
        _WARMED_CONFIGURATIONS.add(key)
        return ollama_runtime_snapshot(api, model)


def ollama_chat(api: str, model: str, messages: list[dict[str, str]], mode: str, *, temperature: float, seed: int, context: int) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    runtime = _warm_ollama(api, model, mode, context)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": context},
    }
    include_think, think_value = mode_to_think(mode)
    if include_think:
        payload["think"] = think_value

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(api.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
    sampler = PowerSampler()
    sampler.start()
    started = time.monotonic()
    try:
        try:
            with request.urlopen(req, timeout=None) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    finally:
        elapsed = time.monotonic() - started
        power = sampler.stop(elapsed)
    energy_wh = power.get("estimated_gpu_energy_wh")
    output_tokens = int(response.get("eval_count") or 0)
    prompt_tokens = int(response.get("prompt_eval_count") or 0)
    if energy_wh is not None:
        energy_j = float(energy_wh) * 3600.0
        power["estimated_gpu_energy_j"] = energy_j
        power["joules_per_output_token"] = energy_j / output_tokens if output_tokens else None
        power["joules_per_prompt_token"] = energy_j / prompt_tokens if prompt_tokens else None
    if runtime is not None:
        response["_benchmark_runtime"] = runtime
    return payload, response, elapsed, power


def response_metrics(resp: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    eval_count = int(resp.get("eval_count") or 0)
    eval_duration_ns = int(resp.get("eval_duration") or 0)
    prompt_count = int(resp.get("prompt_eval_count") or 0)
    prompt_duration_ns = int(resp.get("prompt_eval_duration") or 0)
    return {
        "wall_seconds": elapsed_s,
        "total_duration_ns": resp.get("total_duration"),
        "load_duration_ns": resp.get("load_duration"),
        "prompt_eval_count": prompt_count,
        "prompt_eval_duration_ns": prompt_duration_ns,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "generation_tokens_per_second": (eval_count / (eval_duration_ns / 1e9)) if eval_count and eval_duration_ns else None,
        "prompt_tokens_per_second": (prompt_count / (prompt_duration_ns / 1e9)) if prompt_count and prompt_duration_ns else None,
        "answer_chars": len(((resp.get("message") or {}).get("content") or "")),
        "thinking_chars": len(((resp.get("message") or {}).get("thinking") or "")),
        "runtime": resp.get("_benchmark_runtime"),
    }


def evaluate_checks(answer: str, checks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for check in checks:
        patterns = check.get("patterns", [])
        matches = [bool(re.search(p, answer, flags=re.IGNORECASE | re.DOTALL)) for p in patterns]
        kind = check.get("kind", "any")
        if kind == "all":
            passed = all(matches)
        elif kind == "not":
            passed = not any(matches)
        else:
            passed = any(matches)
        results.append({"name": check.get("name"), "kind": kind, "severity": check.get("severity", "required"), "passed": passed, "patterns": patterns, "matches": matches})
    return results


def contract_metrics(checks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(checks)
    passed = sum(bool(c.get("passed")) for c in rows)
    total = len(rows)
    return {
        "checks_passed": passed,
        "checks_total": total,
        "contract_score": (passed / total) if total else None,
        "full_contract_pass": bool(rows) and passed == total,
        "critical_failure": any(c.get("severity") == "critical" and not c.get("passed") for c in rows),
    }
