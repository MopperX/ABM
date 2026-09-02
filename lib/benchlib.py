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


@dataclass(frozen=True)
class ModelConfig:
    model: str
    benchmarks: tuple[str, ...]
    modes: tuple[str, ...]
    web: bool
    notes: str = ""


def parse_machine_models(path: Path) -> list[ModelConfig]:
    models: list[ModelConfig] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t", fieldnames=["enabled", "model", "benchmarks", "modes", "web", "notes"])
        for row in reader:
            if (row.get("enabled") or "").strip().lower() not in {"1", "true", "yes", "on"}:
                continue
            model = (row.get("model") or "").strip()
            if not model:
                continue
            benches = tuple(x.strip() for x in (row.get("benchmarks") or "").split(",") if x.strip())
            modes = tuple(x.strip() for x in (row.get("modes") or "standard").split(",") if x.strip()) or ("standard",)
            web = (row.get("web") or "").strip().lower() in {"1", "true", "yes", "on"}
            models.append(ModelConfig(model, benches, modes, web, (row.get("notes") or "").strip()))
    return models


def host_snapshot() -> dict[str, Any]:
    def cmd(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, timeout=5).strip() or None
        except Exception:
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
    if system == "Linux":
        snapshot["kernel"] = cmd(["uname", "-a"])
        snapshot["cpu"] = cmd(["bash", "-lc", "lscpu | awk -F: '/Model name/ {sub(/^[ \\t]+/,\"\",$2); print $2; exit}'"])
        snapshot["memory"] = cmd(["bash", "-lc", "free -b | awk '/Mem:/ {print $2}'"])
        snapshot["nvidia_smi"] = cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pci.link.gen.current,pci.link.width.current", "--format=csv,noheader"])
        try:
            if "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower():
                snapshot["wsl"] = True
        except Exception:
            pass
    elif system == "Darwin":
        snapshot["cpu"] = cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or cmd(["sysctl", "-n", "hw.model"])
        snapshot["memory"] = cmd(["sysctl", "-n", "hw.memsize"])
        snapshot["macos"] = cmd(["sw_vers", "-productVersion"])
    snapshot["ollama_version"] = cmd(["ollama", "--version"])
    return snapshot


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




def _read_nvidia_power_w() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=3
        )
        vals=[]
        for line in out.splitlines():
            try:
                vals.append(float(line.strip()))
            except ValueError:
                pass
        return sum(vals) if vals else None
    except Exception:
        return None


class PowerSampler:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        first = _read_nvidia_power_w()
        if first is None:
            return
        self.samples.append(first)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            value = _read_nvidia_power_w()
            if value is not None:
                self.samples.append(value)

    def stop(self, elapsed_s: float) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"source": None, "available": False}
        avg = sum(self.samples) / len(self.samples)
        return {
            "source": "nvidia-smi:power.draw",
            "available": True,
            "samples": len(self.samples),
            "average_w": avg,
            "peak_w": max(self.samples),
            "approx_energy_wh": avg * elapsed_s / 3600.0,
        }


def ollama_chat(api: str, model: str, messages: list[dict[str, str]], mode: str, *, temperature: float, seed: int, context: int) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
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
        results.append({"name": check.get("name"), "kind": kind, "passed": passed, "patterns": patterns, "matches": matches})
    return results
