from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from lib.benchlib import atomic_json, benchmark_cache_root, load_json, ollama_chat, response_metrics, utc_now

PROFILE_COUNTS = {
    "quick": {"ifeval": 10, "truthfulqa": 10, "mmlu_pro": 14},
    "standard": {"ifeval": 25, "truthfulqa": 40, "mmlu_pro": 42},
    "full": {"ifeval": None, "truthfulqa": None, "mmlu_pro": None},
}

# A currently documented ambiguous IFEval item is excluded from the mini subsets, but retained in full.
IFEVAL_MINI_EXCLUDED_KEYS = {2078}


def cache_root_from_run(run_dir: Path) -> Path:
    return benchmark_cache_root(run_dir) / "core"


def _stable_score(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _coverage_select_ifeval(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    candidates=[r for r in rows if int(r.get("key", -1)) not in IFEVAL_MINI_EXCLUDED_KEYS]
    candidates.sort(key=lambda r: _stable_score("ifeval-v4", str(r.get("key"))))
    chosen=[]
    seen=set()
    # First maximize instruction-family coverage, then fill deterministically.
    for row in candidates:
        families={x.split(":",1)[0] for x in row.get("instruction_id_list", [])}
        if families - seen:
            chosen.append(row); seen.update(families)
            if len(chosen) >= count:
                return chosen
    selected_keys={int(r["key"]) for r in chosen}
    for row in candidates:
        if int(row["key"]) in selected_keys:
            continue
        chosen.append(row)
        if len(chosen) >= count:
            break
    return chosen


def load_ifeval(run_dir: Path, profile: str) -> tuple[list[dict[str, Any]], Path]:
    root=cache_root_from_run(run_dir)
    repo=root/"ifeval/google-research"
    data=repo/"instruction_following_eval/data/input_data.jsonl"
    rows=_read_jsonl(data)
    count=PROFILE_COUNTS[profile]["ifeval"]
    if count is not None:
        rows=_coverage_select_ifeval(rows, count)
    return rows, repo


def load_truthfulqa(run_dir: Path, profile: str) -> list[dict[str, Any]]:
    path=cache_root_from_run(run_dir)/"truthfulqa/TruthfulQA/TruthfulQA.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows=list(csv.DictReader(f))
    count=PROFILE_COUNTS[profile]["truthfulqa"]
    if count is None:
        return rows
    # Spread the mini set across categories first, then fill by stable question hash.
    grouped=defaultdict(list)
    for row in rows:
        grouped[row.get("Category", "")].append(row)
    for category in grouped:
        grouped[category].sort(key=lambda r: _stable_score("truthfulqa-v4", r.get("Question", "")))
    categories=sorted(grouped, key=lambda c: _stable_score("truthfulqa-category-v4", c))
    chosen=[]
    pos=0
    while len(chosen) < count:
        added=False
        for cat in categories:
            if pos < len(grouped[cat]):
                chosen.append(grouped[cat][pos]); added=True
                if len(chosen) >= count:
                    break
        if not added:
            break
        pos += 1
    return chosen


def load_mmlu_pro(run_dir: Path, profile: str) -> list[dict[str, Any]]:
    rows=_read_jsonl(cache_root_from_run(run_dir)/"mmlu-pro/test.jsonl")
    count=PROFILE_COUNTS[profile]["mmlu_pro"]
    if count is None:
        return rows
    by_category=defaultdict(list)
    for row in rows:
        by_category[row.get("category", "Other")].append(row)
    for cat in by_category:
        by_category[cat].sort(key=lambda r: _stable_score("mmlu-pro-v4", str(r.get("question_id", r.get("question", "")))))
    categories=sorted(by_category)
    if count == len(categories):
        return [by_category[c][0] for c in categories]
    per=max(1, count // max(1, len(categories)))
    chosen=[]
    for cat in categories:
        chosen.extend(by_category[cat][:per])
    # Should be exactly 42 for current 14 categories; deterministic fill if category count ever changes.
    if len(chosen) < count:
        existing={str(r.get("question_id")) for r in chosen}
        rest=[r for r in rows if str(r.get("question_id")) not in existing]
        rest.sort(key=lambda r: _stable_score("mmlu-pro-fill-v4", str(r.get("question_id", r.get("question", "")))))
        chosen.extend(rest[: count-len(chosen)])
    return chosen[:count]


def external_job_count(run_dir: Path, profile: str) -> int:
    return len(load_ifeval(run_dir, profile)[0]) + len(load_truthfulqa(run_dir, profile)) + len(load_mmlu_pro(run_dir, profile))


def _answer_letter(text: str, allowed: str) -> str | None:
    stripped=text.strip()
    if re.fullmatch(rf"[{re.escape(allowed)}]", stripped, flags=re.IGNORECASE):
        return stripped.upper()
    # Tolerant extraction is recorded; benchmark prompt still asks for a bare letter.
    patterns=[rf"(?:answer|antwoord)\s*(?:is|:)?\s*\(?([{re.escape(allowed)}])\)?", rf"\b([{re.escape(allowed)}])\b"]
    for pat in patterns:
        matches=re.findall(pat, stripped, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def _call(api: str, model: str, mode: str, prompt: str, *, temperature: float, seed: int, context: int):
    payload, response, elapsed, power=ollama_chat(
        api, model, [{"role":"user","content":prompt}], mode,
        temperature=temperature, seed=seed, context=context,
    )
    metrics=response_metrics(response, elapsed)
    answer=((response.get("message") or {}).get("content") or "")
    return payload, response, metrics, power, answer


def _write_source_metadata(run_dir: Path) -> dict[str, Any]:
    src=cache_root_from_run(run_dir)/"sources.json"
    metadata=load_json(src)
    target=run_dir/"raw/core/external_sources.json"
    if not target.exists():
        atomic_json(target, metadata)
    return metadata


def _ifeval_eval(repo: Path, record: dict[str, Any], answer: str) -> dict[str, Any]:
    cache_root = repo.parents[1]
    nltk_dir = cache_root / "nltk"
    os.environ["NLTK_DATA"] = str(nltk_dir)
    try:
        import nltk  # type: ignore
        if str(nltk_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_dir))
    except Exception:
        pass
    root=str(repo)
    if root not in sys.path:
        sys.path.insert(0, root)
    from instruction_following_eval import evaluation_lib  # type: ignore
    inp=evaluation_lib.InputExample(
        key=record["key"],
        instruction_id_list=list(record["instruction_id_list"]),
        prompt=record["prompt"],
        kwargs=copy.deepcopy(record["kwargs"]),
    )
    mapping={record["prompt"]:answer}
    strict=evaluation_lib.test_instruction_following_strict(copy.deepcopy(inp), mapping)
    loose=evaluation_lib.test_instruction_following_loose(copy.deepcopy(inp), mapping)
    return {
        "strict_prompt_pass": bool(strict.follow_all_instructions),
        "strict_instruction_pass": list(strict.follow_instruction_list),
        "loose_prompt_pass": bool(loose.follow_all_instructions),
        "loose_instruction_pass": list(loose.follow_instruction_list),
    }


def run_external(
    *, run_dir: Path, model: str, mode: str, profile: str, api: str,
    temperature: float, seed: int, context: int,
    is_completed: Callable[[str], bool], mark_completed: Callable[[str, dict[str, Any]], None],
    should_stop: Callable[[], bool], set_current: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    source_meta=_write_source_metadata(run_dir)
    config_slug=(model+"__"+mode).replace("/","_").replace(":","_")
    base=run_dir/"raw/core"/config_slug/"external"
    base.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any]={}

    # IFEval
    ifeval_rows, ifeval_repo=load_ifeval(run_dir, profile)
    if_results=[]
    for idx,row in enumerate(ifeval_rows, start=1):
        if should_stop(): return {"stopped":True,"sources":source_meta,"datasets":summaries}
        key=str(row["key"]); logical=f"core|{model}|{mode}|IFEVAL|{key}"
        path=base/"ifeval"/f"{key}.json"
        if is_completed(logical) and path.exists():
            if_results.append(load_json(path)); continue
        set_current({"benchmark":"core","model":model,"mode":mode,"test":f"IFEval:{key}","repeat":idx,"repeats":len(ifeval_rows)})
        payload,response,metrics,power,answer=_call(api,model,mode,row["prompt"],temperature=temperature,seed=seed,context=context)
        ev=_ifeval_eval(ifeval_repo,row,answer)
        result={
            "dataset":"IFEval","key":row["key"],"instruction_id_list":row["instruction_id_list"],"prompt":row["prompt"],
            "model":model,"mode":mode,"completed_at":utc_now(),"final_answer":answer,
            "pass":ev["strict_prompt_pass"],"checks_passed":sum(ev["strict_instruction_pass"]),"checks_total":len(ev["strict_instruction_pass"]),
            **ev,"call":{"request":payload,"response":response,"metrics":metrics,"power":power},
        }
        atomic_json(path,result); mark_completed(logical,result); if_results.append(result)
    strict_inst=[v for r in if_results for v in r["strict_instruction_pass"]]
    loose_inst=[v for r in if_results for v in r["loose_instruction_pass"]]
    summaries["ifeval"]={
        "items":len(if_results),
        "strict_prompt_accuracy":sum(r["strict_prompt_pass"] for r in if_results)/len(if_results) if if_results else None,
        "strict_instruction_accuracy":sum(strict_inst)/len(strict_inst) if strict_inst else None,
        "loose_prompt_accuracy":sum(r["loose_prompt_pass"] for r in if_results)/len(if_results) if if_results else None,
        "loose_instruction_accuracy":sum(loose_inst)/len(loose_inst) if loose_inst else None,
        "selected_keys":[r["key"] for r in if_results],
    }

    # TruthfulQA binary
    truth_rows=load_truthfulqa(run_dir,profile); truth_results=[]
    for idx,row in enumerate(truth_rows,start=1):
        if should_stop(): return {"stopped":True,"sources":source_meta,"datasets":summaries}
        q=row["Question"]; qid=hashlib.sha256(q.encode()).hexdigest()[:16]
        logical=f"core|{model}|{mode}|TRUTHFULQA|{qid}"; path=base/"truthfulqa"/f"{qid}.json"
        if is_completed(logical) and path.exists(): truth_results.append(load_json(path)); continue
        parity=int(_stable_score("truthfulqa-order-v4",q)[0],16)%2
        correct=row["Best Answer"]; incorrect=row["Best Incorrect Answer"]
        opts=[correct,incorrect] if parity==0 else [incorrect,correct]
        expected="A" if parity==0 else "B"
        prompt=(f"Question: {q}\n\nA. {opts[0]}\nB. {opts[1]}\n\nChoose the truthful answer. Reply with only A or B.")
        set_current({"benchmark":"core","model":model,"mode":mode,"test":f"TruthfulQA:{qid}","repeat":idx,"repeats":len(truth_rows)})
        payload,response,metrics,power,answer=_call(api,model,mode,prompt,temperature=temperature,seed=seed,context=context)
        predicted=_answer_letter(answer,"AB"); passed=predicted==expected
        result={"dataset":"TruthfulQA-binary","id":qid,"category":row.get("Category"),"question":q,"options":{"A":opts[0],"B":opts[1]},
                "expected":expected,"predicted":predicted,"pass":passed,"checks_passed":int(passed),"checks_total":1,"final_answer":answer,
                "model":model,"mode":mode,"completed_at":utc_now(),"call":{"request":payload,"response":response,"metrics":metrics,"power":power}}
        atomic_json(path,result); mark_completed(logical,result); truth_results.append(result)
    summaries["truthfulqa"]={"items":len(truth_results),"correct":sum(r["pass"] for r in truth_results),
                              "accuracy":sum(r["pass"] for r in truth_results)/len(truth_results) if truth_results else None,
                              "selected_ids":[r["id"] for r in truth_results]}

    # MMLU-Pro mini/full
    mmlu_rows=load_mmlu_pro(run_dir,profile); mmlu_results=[]
    letters="ABCDEFGHIJ"
    for idx,row in enumerate(mmlu_rows,start=1):
        if should_stop(): return {"stopped":True,"sources":source_meta,"datasets":summaries}
        qid=str(row.get("question_id",idx)); logical=f"core|{model}|{mode}|MMLUPRO|{qid}"; path=base/"mmlu-pro"/f"{qid}.json"
        if is_completed(logical) and path.exists(): mmlu_results.append(load_json(path)); continue
        opts=[o for o in row.get("options",[]) if o != "N/A"]
        option_text="\n".join(f"{letters[i]}. {opt}" for i,opt in enumerate(opts))
        prompt=f"Question: {row['question']}\n\n{option_text}\n\nChoose the best answer. Reply with only the answer letter."
        expected=(row.get("answer") or letters[int(row["answer_index"])]).strip().upper()
        set_current({"benchmark":"core","model":model,"mode":mode,"test":f"MMLU-Pro:{qid}","repeat":idx,"repeats":len(mmlu_rows)})
        payload,response,metrics,power,answer=_call(api,model,mode,prompt,temperature=temperature,seed=seed,context=context)
        predicted=_answer_letter(answer,letters[:len(opts)]); passed=predicted==expected
        result={"dataset":"MMLU-Pro","question_id":row.get("question_id"),"category":row.get("category"),"question":row["question"],
                "options":opts,"expected":expected,"predicted":predicted,"pass":passed,"checks_passed":int(passed),"checks_total":1,
                "final_answer":answer,"model":model,"mode":mode,"completed_at":utc_now(),
                "call":{"request":payload,"response":response,"metrics":metrics,"power":power}}
        atomic_json(path,result); mark_completed(logical,result); mmlu_results.append(result)
    percat={}
    cats=defaultdict(list)
    for r in mmlu_results: cats[r.get("category")].append(bool(r["pass"]))
    for cat,vals in cats.items(): percat[cat]={"items":len(vals),"accuracy":sum(vals)/len(vals)}
    summaries["mmlu_pro"]={"items":len(mmlu_results),"correct":sum(r["pass"] for r in mmlu_results),
                            "accuracy":sum(r["pass"] for r in mmlu_results)/len(mmlu_results) if mmlu_results else None,
                            "per_category":percat,"selected_question_ids":[r.get("question_id") for r in mmlu_results]}

    # Aggregate external performance, useful later for dashboarding without hiding raw calls.
    calls=[r["call"] for r in if_results+truth_results+mmlu_results]
    tps=[c["metrics"].get("generation_tokens_per_second") for c in calls if c["metrics"].get("generation_tokens_per_second") is not None]
    wall=[c["metrics"].get("wall_seconds") for c in calls if c["metrics"].get("wall_seconds") is not None]
    return {"stopped":False,"sources":source_meta,"datasets":summaries,
            "performance":{"calls":len(calls),"generation_tps_median":statistics.median(tps) if tps else None,
                           "wall_seconds_median":statistics.median(wall) if wall else None}}
