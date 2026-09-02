#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.benchlib import (  # noqa: E402
    BENCHMARKS,
    PowerSampler,
    atomic_json,
    host_snapshot,
    load_json,
    ollama_model_identity,
    parse_defaults,
    parse_machine_models,
    utc_now,
)
from scripts.core_runner import job_count as core_job_count, run_core  # noqa: E402
from scripts.coding_agent_runner import coding_job_count, run_coding_agent  # noqa: E402
from scripts.rag_runner import rag_answer_job_count, rag_fixed_job_count, prepare_rag_retrieval, run_rag_answers  # noqa: E402
from scripts.vision_runner import job_count as vision_job_count, run_vision  # noqa: E402
from scripts.image_runner import job_count as image_job_count, run_image  # noqa: E402
from scripts.speech_runner import job_count as speech_job_count, run_speech  # noqa: E402
from scripts.music_runner import job_count as music_job_count, run_music  # noqa: E402
from scripts.web_runner import job_count as web_job_count, run_web  # noqa: E402

STOP = False


def _signal_handler(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def load_state(run_dir: Path) -> dict[str, Any]:
    return load_json(state_path(run_dir))


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(state_path(run_dir), state)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--machine-config", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    machine_config = Path(args.machine_config).resolve()
    defaults = parse_defaults(REPO_ROOT / "config" / "defaults.conf")
    state = load_state(run_dir)
    state["pid"] = os.getpid()
    state["status"] = "running"
    state.setdefault("completed_jobs", {})
    state.setdefault("benchmarks", {})
    save_state(run_dir, state)
    (run_dir / "runner.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")

    metadata_path = run_dir / "raw" / "machine.json"
    if not metadata_path.exists():
        atomic_json(metadata_path, host_snapshot())

    idle_path = run_dir / "raw" / "hardware" / "idle-gpu.json"
    if not idle_path.exists():
        idle_sampler = PowerSampler(interval=0.5)
        idle_sampler.start()
        idle_started = time.monotonic()
        time.sleep(30)
        idle_elapsed = time.monotonic() - idle_started
        atomic_json(idle_path, {
            "duration_seconds": idle_elapsed,
            "measurement": idle_sampler.stop(idle_elapsed),
            "captured_at": utc_now(),
        })

    models = parse_machine_models(machine_config)
    selected = state["selected_benchmarks"]
    profile = state["profile"]
    include_livecodebench = bool(state.get("with_livecodebench", False))

    model_identity_path = run_dir / "raw" / "models" / "ollama.json"
    identities = {}
    api = defaults.get("OLLAMA_API", "http://127.0.0.1:11434")
    relevant_models = [
        mc for mc in models
        if any(suite in selected for suite in mc.benchmarks)
        and not (set(mc.benchmarks).intersection(selected) == {"web"} and not mc.web)
    ]
    for model_name in dict.fromkeys(mc.model for mc in relevant_models):
        identities[model_name] = ollama_model_identity(api, model_name)
    if model_identity_path.exists():
        original = load_json(model_identity_path)
        changed = [name for name, item in identities.items() if (original.get(name) or {}).get("digest") != item.get("digest")]
        if changed:
            raise RuntimeError("Ollama model digest changed since this run started: " + ", ".join(changed))
    else:
        atomic_json(model_identity_path, identities)

    # Planned jobs are model/mode/test invocations. Multiturn calls are reflected in live current state,
    # while completion percentage is based on logical test repeats so resume remains stable.
    logical_total = 0
    for bench in selected:
        if bench == "core":
            per_config_jobs = core_job_count(REPO_ROOT, run_dir, profile)
            for mc in models:
                if "core" in mc.benchmarks:
                    logical_total += len(mc.modes) * per_config_jobs
        elif bench == "coding-agent":
            per_config_jobs = coding_job_count(profile, include_livecodebench=include_livecodebench)
            for mc in models:
                if "coding-agent" in mc.benchmarks:
                    logical_total += len(mc.modes) * per_config_jobs
        elif bench == "rag":
            logical_total += rag_fixed_job_count()
            per_config_jobs = rag_answer_job_count(profile)
            for mc in models:
                if "rag" in mc.benchmarks:
                    logical_total += len(mc.modes) * per_config_jobs
        elif bench == "vision":
            per_config_jobs = vision_job_count(REPO_ROOT, run_dir, profile)
            for mc in models:
                if "vision" in mc.benchmarks:
                    logical_total += len(mc.modes) * per_config_jobs
        elif bench == "image":
            logical_total += image_job_count(run_dir, profile, machine_config)
        elif bench == "speech":
            logical_total += speech_job_count(run_dir, profile, machine_config)
        elif bench == "music":
            logical_total += music_job_count(run_dir, profile, machine_config)
        elif bench == "web":
            per_config_jobs = web_job_count(REPO_ROOT, profile)
            for mc in models:
                if mc.web and "web" in mc.benchmarks:
                    logical_total += len(mc.modes) * per_config_jobs
        else:
            logical_total += 1
    state["progress"] = {
        "completed": len(state["completed_jobs"]),
        "total": logical_total,
        "percent": round((len(state["completed_jobs"]) / logical_total * 100), 2) if logical_total else 0,
        "current": state.get("progress", {}).get("current"),
    }
    save_state(run_dir, state)

    def should_stop() -> bool:
        return STOP or (run_dir / "stop.requested").exists()

    def is_completed(key: str) -> bool:
        return key in state["completed_jobs"]

    def set_current(current: dict[str, Any]) -> None:
        state["progress"]["current"] = current
        save_state(run_dir, state)

    def mark_completed(key: str, result: dict[str, Any]) -> None:
        state["completed_jobs"][key] = {
            "completed_at": utc_now(),
            "pass": result.get("pass"),
            "checks_passed": result.get("checks_passed"),
            "checks_total": result.get("checks_total"),
        }
        done = len(state["completed_jobs"])
        total = state["progress"]["total"]
        state["progress"]["completed"] = done
        state["progress"]["percent"] = round((done / total * 100), 2) if total else 0
        save_state(run_dir, state)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "machine": state["machine"],
        "profile": profile,
        "selected_benchmarks": selected,
        "started_at": state.get("started_at"),
        "benchmarks": {},
    }

    try:
        for bench in selected:
            if should_stop():
                break
            state["benchmarks"].setdefault(bench, {})
            state["benchmarks"][bench]["status"] = "running"
            save_state(run_dir, state)

            if bench not in {"core", "coding-agent", "rag", "vision", "image", "speech", "music", "web"}:
                result = {"status": "not_implemented", "benchmark": bench}
                summary["benchmarks"][bench] = result
                state["benchmarks"][bench] = result
                state["completed_jobs"][f"module|{bench}"] = {"completed_at": utc_now(), "pass": None}
                state["progress"]["completed"] = len(state["completed_jobs"])
                total = state["progress"]["total"]
                state["progress"]["percent"] = round((len(state["completed_jobs"]) / total * 100), 2) if total else 0
                save_state(run_dir, state)
                continue


            if bench == "web":
                web_results=[]
                for mc in models:
                    if not mc.web or "web" not in mc.benchmarks:
                        continue
                    for mode in mc.modes:
                        if should_stop():
                            break
                        wr=run_web(
                            repo_root=REPO_ROOT,run_dir=run_dir,model=mc.model,mode=mode,profile=profile,
                            api=defaults.get("OLLAMA_API", "http://127.0.0.1:11434"),
                            temperature=float(defaults.get("TEMPERATURE", "0")),seed=int(defaults.get("SEED", "42")),
                            context=int(defaults.get("CONTEXT_LENGTH", "16384")),is_completed=is_completed,mark_completed=mark_completed,
                            should_stop=should_stop,set_current=set_current,
                        )
                        web_results.append(wr)
                        if wr.get("stopped"):
                            break
                web_status = "stopped" if should_stop() else ("no_enabled_models" if not web_results else "completed")
                summary["benchmarks"]["web"]={"status":web_status,"configurations":web_results}
                state["benchmarks"]["web"]["status"]=web_status
                save_state(run_dir,state)
                continue

            if bench == "image":
                image_result = run_image(
                    repo_root=REPO_ROOT, run_dir=run_dir, image_config=machine_config,
                    profile=profile, api=defaults.get("OLLAMA_API", "http://127.0.0.1:11434"),
                    seed=int(defaults.get("SEED", "42")), is_completed=is_completed, mark_completed=mark_completed,
                    should_stop=should_stop, set_current=set_current,
                )
                summary["benchmarks"]["image"] = {"status": "completed" if not should_stop() else "stopped", **image_result}
                state["benchmarks"]["image"]["status"] = summary["benchmarks"]["image"]["status"]
                save_state(run_dir, state)
                continue


            if bench == "speech":
                speech_result = run_speech(
                    repo_root=REPO_ROOT, run_dir=run_dir, speech_config=machine_config,
                    profile=profile, is_completed=is_completed, mark_completed=mark_completed,
                    should_stop=should_stop, set_current=set_current,
                )
                summary["benchmarks"]["speech"] = {"status": "completed" if not should_stop() else "stopped", **speech_result}
                state["benchmarks"]["speech"]["status"] = summary["benchmarks"]["speech"]["status"]
                save_state(run_dir, state)
                continue


            if bench == "music":
                music_result = run_music(
                    repo_root=REPO_ROOT, run_dir=run_dir, music_config=machine_config,
                    profile=profile, is_completed=is_completed, mark_completed=mark_completed,
                    should_stop=should_stop, set_current=set_current,
                )
                summary["benchmarks"]["music"] = {"status": "completed" if not should_stop() else "stopped", **music_result}
                state["benchmarks"]["music"]["status"] = summary["benchmarks"]["music"]["status"]
                save_state(run_dir, state)
                continue


            if bench == "vision":
                vision_results=[]
                for mc in models:
                    if "vision" not in mc.benchmarks:
                        continue
                    for mode in mc.modes:
                        if should_stop():
                            break
                        vr=run_vision(
                            repo_root=REPO_ROOT,run_dir=run_dir,model=mc.model,mode=mode,profile=profile,
                            api=defaults.get("OLLAMA_API", "http://127.0.0.1:11434"),
                            temperature=float(defaults.get("TEMPERATURE", "0")),seed=int(defaults.get("SEED", "42")),
                            context=int(defaults.get("CONTEXT_LENGTH", "16384")),is_completed=is_completed,mark_completed=mark_completed,
                            should_stop=should_stop,set_current=set_current,
                        )
                        vision_results.append(vr)
                        if vr.get("stopped"):
                            break
                summary["benchmarks"]["vision"]={"status":"completed" if not should_stop() else "stopped","configurations":vision_results}
                state["benchmarks"]["vision"]["status"]=summary["benchmarks"]["vision"]["status"]
                save_state(run_dir,state)
                continue

            if bench == "rag":
                retrieval=prepare_rag_retrieval(
                    repo_root=REPO_ROOT,run_dir=run_dir,profile=profile,
                    api=defaults.get("OLLAMA_API", "http://127.0.0.1:11434"),
                    is_completed=is_completed,mark_completed=mark_completed,set_current=set_current,should_stop=should_stop,
                )
                rag_results=[]
                if not retrieval.get("stopped"):
                    for mc in models:
                        if "rag" not in mc.benchmarks:
                            continue
                        for mode in mc.modes:
                            if should_stop(): break
                            rr=run_rag_answers(
                                repo_root=REPO_ROOT,run_dir=run_dir,model=mc.model,mode=mode,profile=profile,
                                api=defaults.get("OLLAMA_API", "http://127.0.0.1:11434"),
                                temperature=float(defaults.get("TEMPERATURE", "0")),seed=int(defaults.get("SEED", "42")),
                                context=int(defaults.get("CONTEXT_LENGTH", "16384")),is_completed=is_completed,mark_completed=mark_completed,
                                should_stop=should_stop,set_current=set_current,
                            )
                            rag_results.append(rr)
                            if rr.get("stopped"): break
                summary["benchmarks"]["rag"]={"status":"completed" if not should_stop() else "stopped","retrieval":retrieval,"configurations":rag_results}
                state["benchmarks"]["rag"]["status"]=summary["benchmarks"]["rag"]["status"]
                save_state(run_dir,state)
                continue

            if bench == "coding-agent":
                coding_results=[]
                for mc in models:
                    if "coding-agent" not in mc.benchmarks:
                        continue
                    for mode in mc.modes:
                        if should_stop():
                            break
                        config_result=run_coding_agent(
                            repo_root=REPO_ROOT, run_dir=run_dir, model=mc.model, mode=mode, profile=profile,
                            api=defaults.get("OLLAMA_API", "http://127.0.0.1:11434"),
                            temperature=float(defaults.get("TEMPERATURE", "0")),
                            seed=int(defaults.get("SEED", "42")),
                            context=int(defaults.get("CONTEXT_LENGTH", "16384")),
                            is_completed=is_completed, mark_completed=mark_completed, should_stop=should_stop, set_current=set_current,
                            include_livecodebench=include_livecodebench,
                        )
                        coding_results.append(config_result)
                        if config_result.get("stopped"):
                            break
                summary["benchmarks"]["coding-agent"]={"status":"completed" if not should_stop() else "stopped", "configurations":coding_results}
                state["benchmarks"]["coding-agent"]["status"]=summary["benchmarks"]["coding-agent"]["status"]
                save_state(run_dir,state)
                continue

            core_results = []
            for mc in models:
                if "core" not in mc.benchmarks:
                    continue
                for mode in mc.modes:
                    if should_stop():
                        break
                    config_result = run_core(
                        repo_root=REPO_ROOT,
                        run_dir=run_dir,
                        model=mc.model,
                        mode=mode,
                        profile=profile,
                        api=defaults.get("OLLAMA_API", "http://127.0.0.1:11434"),
                        temperature=float(defaults.get("TEMPERATURE", "0")),
                        seed=int(defaults.get("SEED", "42")),
                        context=int(defaults.get("CONTEXT_LENGTH", "16384")),
                        is_completed=is_completed,
                        mark_completed=mark_completed,
                        should_stop=should_stop,
                        set_current=set_current,
                    )
                    core_results.append(config_result)
                    if config_result.get("stopped"):
                        break
            summary["benchmarks"]["core"] = {"status": "completed" if not should_stop() else "stopped", "configurations": core_results}
            state["benchmarks"]["core"]["status"] = summary["benchmarks"]["core"]["status"]
            save_state(run_dir, state)

        if should_stop():
            state["status"] = "stopped"
            state["stopped_at"] = utc_now()
        else:
            state["status"] = "completed"
            state["completed_at"] = utc_now()
            state["progress"]["current"] = None
        summary["finished_at"] = utc_now()
        summary["status"] = state["status"]
        atomic_json(run_dir / "summary" / "summary.json", summary)
        save_state(run_dir, state)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = {"type": type(exc).__name__, "message": str(exc), "at": utc_now()}
        save_state(run_dir, state)
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            (run_dir / "runner.pid").unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
