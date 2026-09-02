from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable

from lib.benchlib import atomic_json, contract_metrics, distribution_summary, evaluate_checks, load_json, ollama_chat, response_metrics, utc_now, wilson_interval
from scripts.core_external import external_job_count, run_external

SYSTEM_PROMPT = (
    "You are a reliable technical assistant. Clearly distinguish facts, hypotheses, and unknowns. "
    "Do not invent logs, configurations, versions, measurements, people, or causes that do not "
    "follow from the supplied information. Answer in English."
)

PROFILE_REPEATS = {"quick": 1, "standard": 3, "full": 5}


def selected_tests(repo_root: Path, profile: str) -> list[dict[str, Any]]:
    tests = load_json(repo_root / "benchmarks" / "core" / "tests.json")["tests"]
    if profile == "quick":
        allowed = {"G1", "G2", "G3", "G4", "G5.1", "G5.2", "G5.3", "G6"}
        return [t for t in tests if t["id"] in allowed]
    return tests


def job_count(repo_root: Path, run_dir: Path, profile: str) -> int:
    # Logical checkpoint jobs: each practical repeat completes once; multi-turn API calls remain inside that job.
    practical = len(selected_tests(repo_root, profile)) * PROFILE_REPEATS[profile]
    return practical + external_job_count(run_dir, profile)


def run_core(
    *,
    repo_root: Path,
    run_dir: Path,
    model: str,
    mode: str,
    profile: str,
    api: str,
    temperature: float,
    seed: int,
    context: int,
    is_completed: Callable[[str], bool],
    mark_completed: Callable[[str, dict[str, Any]], None],
    should_stop: Callable[[], bool],
    set_current: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    tests = selected_tests(repo_root, profile)
    repeats = PROFILE_REPEATS[profile]
    config_slug = (model + "__" + mode).replace("/", "_").replace(":", "_")
    config_dir = run_dir / "raw" / "core" / config_slug
    config_dir.mkdir(parents=True, exist_ok=True)
    test_summaries: list[dict[str, Any]] = []

    for test in tests:
        repeat_summaries = []
        for repeat in range(1, repeats + 1):
            if should_stop():
                return {"stopped": True, "model": model, "mode": mode, "profile": profile, "practical": {"tests": test_summaries}}
            logical_job = f"core|{model}|{mode}|{test['id']}|{repeat}"
            summary_path = config_dir / "practical" / test["id"] / f"repeat-{repeat}" / "result.json"
            if is_completed(logical_job) and summary_path.exists():
                repeat_summaries.append(load_json(summary_path))
                continue

            test_dir = summary_path.parent
            test_dir.mkdir(parents=True, exist_ok=True)
            set_current({"benchmark": "core", "model": model, "mode": mode, "test": test["id"], "repeat": repeat, "repeats": repeats})
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            api_calls = []

            if test["type"] == "single":
                messages.append({"role": "user", "content": test["prompt"]})
                payload, response, elapsed, power = ollama_chat(api, model, messages, mode, temperature=temperature, seed=seed, context=context)
                api_calls.append({"request": payload, "response": response, "metrics": response_metrics(response, elapsed), "power": power})
                answer = ((response.get("message") or {}).get("content") or "")
            else:
                answer = ""
                for turn_no, turn in enumerate(test["turns"], start=1):
                    if should_stop():
                        return {"stopped": True, "model": model, "mode": mode, "profile": profile, "practical": {"tests": test_summaries}}
                    set_current({"benchmark": "core", "model": model, "mode": mode, "test": test["id"], "repeat": repeat, "repeats": repeats, "turn": turn_no, "turns": len(test["turns"])})
                    messages.append({"role": "user", "content": turn})
                    payload, response, elapsed, power = ollama_chat(api, model, messages, mode, temperature=temperature, seed=seed, context=context)
                    metrics = response_metrics(response, elapsed)
                    api_calls.append({"request": payload, "response": response, "metrics": metrics, "power": power})
                    answer = ((response.get("message") or {}).get("content") or "")
                    messages.append({"role": "assistant", "content": answer})

            checks = evaluate_checks(answer, test.get("checks", []))
            contract = contract_metrics(checks)
            all_metrics = [c["metrics"] for c in api_calls]
            tps = [m["generation_tokens_per_second"] for m in all_metrics if m.get("generation_tokens_per_second") is not None]
            result = {
                "test": test["id"], "title": test["title"], "repeat": repeat, "model": model, "mode": mode,
                "completed_at": utc_now(), **contract, "pass": contract["full_contract_pass"],
                "checks": checks, "final_answer": answer, "api_call_count": len(api_calls),
                "generation_tps_median": statistics.median(tps) if tps else None,
            }
            atomic_json(test_dir / "calls.json", api_calls)
            atomic_json(summary_path, result)
            mark_completed(logical_job, result)
            repeat_summaries.append(result)

        test_summaries.append({
            "test": test["id"], "title": test["title"], "repeats": repeat_summaries,
            "passes": sum(1 for r in repeat_summaries if r.get("pass")), "repeat_count": len(repeat_summaries),
        })

    all_repeats = [r for t in test_summaries for r in t["repeats"]]
    practical_summary = {
        "tests": test_summaries,
        "fully_passed_repeats": sum(1 for r in all_repeats if r.get("pass")),
        "total_repeats": len(all_repeats),
        "confidence_interval_95": wilson_interval(sum(1 for r in all_repeats if r.get("pass")), len(all_repeats)),
        "performance": distribution_summary(r["generation_tps_median"] for r in all_repeats if r.get("generation_tps_median") is not None),
    }

    if should_stop():
        return {"stopped": True, "model": model, "mode": mode, "profile": profile, "practical": practical_summary}

    external = run_external(
        run_dir=run_dir, model=model, mode=mode, profile=profile, api=api,
        temperature=temperature, seed=seed, context=context,
        is_completed=is_completed, mark_completed=mark_completed, should_stop=should_stop, set_current=set_current,
    )
    return {"stopped": bool(external.get("stopped")), "model": model, "mode": mode, "profile": profile,
            "practical": practical_summary, "external": external}
