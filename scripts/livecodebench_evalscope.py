from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from lib.benchlib import PowerSampler, atomic_json, load_json, utc_now

EVALSCOPE_VERSION = "1.11.1"
PROFILE_LIMITS = {"quick": 3, "standard": 10, "full": None}


def _openai_url(native_api: str) -> str:
    base = native_api.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    return base + "/v1"


def _reasoning_effort(mode: str) -> str | None:
    if mode == "nothinking":
        return "none"
    if mode == "thinking":
        return "medium"
    if mode in {"low", "medium", "high"}:
        return mode
    return None


def _next_attempt(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    nums=[]
    for p in base.glob("attempt-*"):
        try: nums.append(int(p.name.split("-",1)[1]))
        except Exception: pass
    d=base/f"attempt-{(max(nums) if nums else 0)+1}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _docker_prefix() -> list[str]:
    check=subprocess.run(["docker","info"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if check.returncode==0:
        return []
    if os.name == "posix" and subprocess.run(["sh","-lc","command -v sg >/dev/null 2>&1 && sg docker -c 'docker info >/dev/null 2>&1'"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        return ["sg","docker","-c"]
    raise RuntimeError("The Docker sandbox is not accessible to the benchmark user. Run the LiveCodeBench preflight again.")


def _sandbox_image_identity(prefix: list[str]) -> dict[str, Any]:
    command = ["docker", "image", "inspect", "python:3.11-slim", "--format", '{{json .}}']
    if prefix:
        command = [*prefix, shlex.join(command)]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if completed.returncode != 0:
        return {"reference": "python:3.11-slim", "image_id": None, "repo_digests": []}
    try:
        image = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"reference": "python:3.11-slim", "image_id": None, "repo_digests": []}
    return {
        "reference": "python:3.11-slim",
        "image_id": image.get("Id"),
        "repo_digests": image.get("RepoDigests") or [],
    }


def _parse_report(output: Path) -> tuple[Path, dict[str,Any]]:
    files=sorted(output.glob("reports/**/*.json"))
    if not files:
        raise RuntimeError(f"EvalScope did not produce a LiveCodeBench report in {output}")
    # One dataset/model is requested, so use the first JSON report and retain all raw files beside it.
    p=files[0]
    return p, load_json(p)


def run_livecodebench(
    *, repo_root:Path, run_dir:Path, model:str, mode:str, profile:str, api:str,
    temperature:float, seed:int,
    is_completed:Callable[[str],bool], mark_completed:Callable[[str,dict[str,Any]],None],
    should_stop:Callable[[],bool], set_current:Callable[[dict[str,Any]],None],
)->dict[str,Any]:
    logical=f"coding-agent|{model}|{mode}|LCB|1"
    slug=(model+"__"+mode).replace("/","_").replace(":","_")
    base=run_dir/"raw/coding-agent"/slug/"external/livecodebench"
    result_path=base/"result.json"
    if is_completed(logical) and result_path.exists():
        return load_json(result_path)
    if should_stop():
        return {"stopped":True,"benchmark":"livecodebench","model":model,"mode":mode}

    exe=repo_root/".venv-lcb/bin/evalscope"
    if not exe.exists():
        raise RuntimeError("EvalScope is missing; the LiveCodeBench dependency preflight was not run.")

    attempt=_next_attempt(base)
    output=attempt/"evalscope"
    output.mkdir(parents=True,exist_ok=True)
    set_current({"benchmark":"coding-agent","model":model,"mode":mode,"test":"LiveCodeBench","repeat":1,"repeats":1})

    generation={"temperature":temperature,"max_tokens":8192,"timeout":21600,"retries":1}
    effort=_reasoning_effort(mode)
    if effort is not None:
        generation["reasoning_effort"]=effort

    cmd=[
        str(exe),"eval",
        "--model",model,
        "--model-id",slug,
        "--api-url",_openai_url(api),
        "--api-key","EMPTY",
        "--eval-type","openai_api",
        "--datasets","live_code_bench",
        "--eval-batch-size","1",
        "--generation-config",json.dumps(generation,separators=(",",":")),
        "--sandbox",json.dumps({"enabled":True,"engine":"docker","default_config":{"network_enabled":False,"memory_limit":"1g","cpu_limit":1.0}},separators=(",",":")),
        "--seed",str(seed),
        "--work-dir",str(output),
        "--no-timestamp",
        "--enable-progress-tracker",
    ]
    limit=PROFILE_LIMITS[profile]
    if limit is not None:
        cmd += ["--limit",str(limit)]

    prefix=_docker_prefix()
    command_meta={
        "created_at":utc_now(),"evalscope_version":EVALSCOPE_VERSION,"model":model,"mode":mode,"profile":profile,
        "limit":limit,"api_url":_openai_url(api),"generation_config":generation,
        "sandbox":{"enabled":True,"engine":"docker","network_enabled":False,"memory_limit":"1g","cpu_limit":1.0,"remove_on_exit":True},
        "sandbox_base_image":_sandbox_image_identity(prefix),"command":cmd,
    }
    atomic_json(attempt/"command.json",command_meta)

    env=os.environ.copy()
    cache_root=Path(env.get("BENCH_CACHE_DIR",run_dir.parents[2]/"cache"))
    env.setdefault("HF_HOME",str(cache_root/"livecodebench/huggingface"))
    env.setdefault("MODELSCOPE_CACHE",str(cache_root/"livecodebench/modelscope"))
    env.setdefault("TOKENIZERS_PARALLELISM","false")
    Path(env["HF_HOME"]).mkdir(parents=True,exist_ok=True)
    Path(env["MODELSCOPE_CACHE"]).mkdir(parents=True,exist_ok=True)

    if prefix:
        actual=prefix+[shlex.join(cmd)]
    else:
        actual=cmd
    log_path=attempt/"evalscope.log"
    sampler=PowerSampler()
    sampler.start()
    started=time.monotonic()
    with log_path.open("w",encoding="utf-8") as log:
        proc=subprocess.Popen(actual,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,text=True)
        while proc.poll() is None:
            if should_stop():
                try: os.killpg(proc.pid,signal.SIGTERM)
                except Exception: proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try: os.killpg(proc.pid,signal.SIGKILL)
                    except Exception: proc.kill()
                    proc.wait()
                interrupted_elapsed=time.monotonic()-started
                interrupted_power=sampler.stop(interrupted_elapsed)
                atomic_json(attempt/"interrupted.json",{"stopped_at":utc_now(),"elapsed_seconds":interrupted_elapsed,"power":interrupted_power})
                return {"stopped":True,"benchmark":"livecodebench","model":model,"mode":mode,"attempt":attempt.name}
            time.sleep(1)
        rc=proc.returncode
    elapsed=time.monotonic()-started
    power=sampler.stop(elapsed)
    if rc != 0:
        raise RuntimeError(f"LiveCodeBench/EvalScope exited with code {rc}; see {log_path}")

    report_file,report=_parse_report(output)
    result={
        "type":"external-code","benchmark":"livecodebench","evaluator":"EvalScope","evaluator_version":EVALSCOPE_VERSION,
        "model":model,"mode":mode,"profile":profile,"completed_at":utc_now(),"pass":None,
        "limit":limit,"score":report.get("score"),"metrics":report.get("metrics"),"perf_metrics":report.get("perf_metrics"),
        "elapsed_seconds":elapsed,"power":power,"report_file":str(report_file.relative_to(base)),"attempt":attempt.name,
        "reasoning_effort":effort,"sandbox_network_enabled":False,
    }
    atomic_json(result_path,result)
    mark_completed(logical,result)
    return result
