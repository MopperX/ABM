from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lib.benchlib import atomic_json, ollama_chat, response_metrics, utc_now


@dataclass
class AgentRunResult:
    stopped: bool
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    steps: int
    tool_calls: int
    test_runs: int
    files_changed: list[str]
    diff: str
    diff_added_lines: int
    diff_removed_lines: int
    final_message: str
    raw_dir: str


def _safe_path(root: Path, rel: str) -> Path:
    rel = rel.strip().lstrip("/")
    p = (root / rel).resolve()
    rr = root.resolve()
    if p != rr and rr not in p.parents:
        raise ValueError(f"Path escapes workspace: {rel}")
    return p


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    # Do not leak the worker's API keys/tokens/credentials into model-edited test code.
    # PATH is retained so standard interpreters/tools remain available.
    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/tmp",
        "PYTHONUNBUFFERED": "1",
    }
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=clean_env,
    )


def init_workspace(template: Path, workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(template, workspace)
    _run(["git", "init", "-q"], workspace)
    _run(["git", "config", "user.email", "benchmark@example.invalid"], workspace)
    _run(["git", "config", "user.name", "AI Benchmark"], workspace)
    _run(["git", "add", "."], workspace)
    commit = _run(["git", "commit", "-qm", "fixture baseline"], workspace)
    if commit.returncode != 0:
        raise RuntimeError(f"Could not create fixture baseline: {commit.stderr}")


def git_diff(workspace: Path) -> str:
    cp = _run(["git", "diff", "--no-ext-diff", "--binary"], workspace)
    return cp.stdout


def changed_files(workspace: Path) -> list[str]:
    cp = _run(["git", "status", "--porcelain"], workspace)
    files=[]
    for line in cp.stdout.splitlines():
        if len(line) >= 4:
            files.append(line[3:])
    return files


def count_diff_lines(diff: str) -> tuple[int, int]:
    added=removed=0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def parse_action(text: str) -> dict[str, Any] | None:
    stripped=text.strip()
    candidates=[stripped]
    fenced=re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.I|re.S)
    candidates.extend(fenced)
    # Also try the first complete object-looking span.
    first=stripped.find("{"); last=stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first:last+1])
    for candidate in candidates:
        try:
            value=json.loads(candidate)
            if isinstance(value, dict) and isinstance(value.get("tool"), str):
                return value
        except Exception:
            pass
    return None


def run_fixture_tests(workspace: Path, timeout: int = 60) -> tuple[int, str, str]:
    workspace = workspace.resolve()
    test_py=workspace/".benchmark"/"test.py"
    test_sh=workspace/".benchmark"/"test.sh"
    if test_py.exists():
        cp=_run(["python3", str(test_py)], workspace, timeout=timeout)
        return cp.returncode, cp.stdout, cp.stderr
    if test_sh.exists():
        cp=_run(["bash", str(test_sh)], workspace, timeout=timeout)
        return cp.returncode, cp.stdout, cp.stderr
    raise RuntimeError("Fixture has no .benchmark/test.py or .benchmark/test.sh")


def execute_tool(workspace: Path, action: dict[str, Any]) -> tuple[str, bool]:
    tool=action.get("tool")
    args=action.get("args") or {}
    if not isinstance(args, dict):
        return "ERROR: args must be a JSON object", False

    try:
        if tool == "list_files":
            rel=str(args.get("path", "."))
            base=_safe_path(workspace, rel)
            if not base.exists():
                return f"ERROR: path does not exist: {rel}", False
            if ".benchmark" in base.parts:
                return "ERROR: .benchmark is hidden", False
            if base.is_file():
                return str(base.relative_to(workspace)), True
            rows=[]
            for p in sorted(base.rglob("*")):
                if ".git" in p.parts or ".benchmark" in p.parts or "__pycache__" in p.parts:
                    continue
                if p.is_file():
                    rows.append(str(p.relative_to(workspace)))
                if len(rows) >= 300:
                    rows.append("... truncated ...")
                    break
            return "\n".join(rows), True

        if tool == "read_file":
            rel=str(args.get("path", ""))
            p=_safe_path(workspace, rel)
            if ".benchmark" in p.parts:
                return "ERROR: .benchmark is hidden", False
            if not p.is_file():
                return f"ERROR: file not found: {rel}", False
            text=p.read_text(encoding="utf-8", errors="replace")
            if len(text) > 30000:
                text=text[:30000]+"\n... truncated ..."
            return text, True

        if tool == "search":
            pattern=str(args.get("pattern", ""))
            rel=str(args.get("path", "."))
            if not pattern:
                return "ERROR: pattern is required", False
            base=_safe_path(workspace, rel)
            if ".benchmark" in base.parts:
                return "ERROR: .benchmark is hidden", False
            rx=re.compile(pattern, re.I)
            hits=[]
            files=[base] if base.is_file() else list(base.rglob("*"))
            for p in files:
                if not p.is_file() or ".git" in p.parts or ".benchmark" in p.parts or "__pycache__" in p.parts:
                    continue
                try:
                    lines=p.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    continue
                for i,line in enumerate(lines,1):
                    if rx.search(line):
                        hits.append(f"{p.relative_to(workspace)}:{i}:{line}")
                        if len(hits) >= 200:
                            return "\n".join(hits)+"\n... truncated ...", True
            return "\n".join(hits) if hits else "NO MATCHES", True

        if tool == "write_file":
            rel=str(args.get("path", ""))
            content=args.get("content")
            if not rel or not isinstance(content, str):
                return "ERROR: write_file requires path and string content", False
            p=_safe_path(workspace, rel)
            if ".benchmark" in p.parts:
                return "ERROR: benchmark test files are read-only", False
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"WROTE {rel} ({len(content)} chars)", True

        if tool == "replace":
            rel=str(args.get("path", ""))
            old=args.get("old"); new=args.get("new")
            if not rel or not isinstance(old, str) or not isinstance(new, str):
                return "ERROR: replace requires path, old and new strings", False
            p=_safe_path(workspace, rel)
            if ".benchmark" in p.parts:
                return "ERROR: benchmark test files are read-only", False
            text=p.read_text(encoding="utf-8")
            if old not in text:
                return "ERROR: old text not found exactly", False
            if text.count(old) != 1:
                return f"ERROR: old text occurs {text.count(old)} times; make it unique", False
            p.write_text(text.replace(old,new,1), encoding="utf-8")
            return f"REPLACED in {rel}", True

        if tool == "run_tests":
            rc,out,err=run_fixture_tests(workspace)
            return f"EXIT={rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}", True

        if tool == "git_diff":
            return git_diff(workspace) or "NO DIFF", True

        if tool == "finish":
            return str(args.get("message", "finished")), True

        return f"ERROR: unsupported tool: {tool}", False
    except subprocess.TimeoutExpired:
        return "ERROR: tool command timed out", False
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}", False


def run_agent_task(
    *,
    workspace: Path,
    raw_dir: Path,
    issue: str,
    model: str,
    mode: str,
    api: str,
    temperature: float,
    seed: int,
    context: int,
    max_steps: int,
    should_stop: Callable[[], bool],
    set_current: Callable[[dict[str, Any]], None],
    current_base: dict[str, Any],
) -> AgentRunResult:
    raw_dir.mkdir(parents=True, exist_ok=True)
    system=(
        "You are a coding agent working in a local repository. Solve the issue by inspecting files, editing the repository, "
        "and running tests. You have only the tools described below. On every turn, output exactly one JSON object and no markdown.\n\n"
        "Tools:\n"
        '{"tool":"list_files","args":{"path":"."}}\n'
        '{"tool":"read_file","args":{"path":"relative/path"}}\n'
        '{"tool":"search","args":{"pattern":"regex","path":"."}}\n'
        '{"tool":"write_file","args":{"path":"relative/path","content":"full file contents"}}\n'
        '{"tool":"replace","args":{"path":"relative/path","old":"exact old text","new":"replacement"}}\n'
        '{"tool":"run_tests","args":{}}\n'
        '{"tool":"git_diff","args":{}}\n'
        '{"tool":"finish","args":{"message":"short summary"}}\n\n'
        "Do not modify .benchmark/. Read project documentation before inventing project-specific APIs. "
        "Run tests before finishing. Keep changes minimal and relevant."
    )
    messages=[{"role":"system","content":system},{"role":"user","content":issue}]
    steps=0; tool_calls=0; test_runs=0; final_message=""

    stopped=False
    for step in range(1,max_steps+1):
        if should_stop():
            stopped=True
            break
        steps=step
        current={**current_base,"turn":step,"turns":max_steps}
        set_current(current)
        payload,response,elapsed,power=ollama_chat(
            api,model,messages,mode,temperature=temperature,seed=seed,context=context
        )
        metrics=response_metrics(response,elapsed)
        message=response.get("message") or {}
        content=message.get("content") or ""
        call_record={
            "step":step,"at":utc_now(),"request":payload,"response":response,
            "metrics":metrics,"power":power,"assistant_content":content,
        }
        action=parse_action(content)
        if action is None:
            tool_result="ERROR: Output was not a valid single JSON tool object. Try again."
            call_record["parsed_action"]=None
            call_record["tool_result"]=tool_result
            atomic_json(raw_dir/f"step-{step:03d}.json",call_record)
            messages.append({"role":"assistant","content":content})
            messages.append({"role":"user","content":tool_result})
            continue

        tool_calls += 1
        tool=str(action.get("tool"))
        if tool == "run_tests":
            test_runs += 1
        tool_result,ok=execute_tool(workspace,action)
        call_record["parsed_action"]=action
        call_record["tool_ok"]=ok
        call_record["tool_result"]=tool_result
        atomic_json(raw_dir/f"step-{step:03d}.json",call_record)
        messages.append({"role":"assistant","content":content})
        messages.append({"role":"user","content":f"TOOL RESULT ({tool}):\n{tool_result}"})
        if tool == "finish":
            final_message=str((action.get("args") or {}).get("message", ""))
            break
        if should_stop():
            stopped=True
            break

    if stopped:
        diff=git_diff(workspace)
        files=changed_files(workspace)
        added,removed=count_diff_lines(diff)
        (raw_dir/"partial.diff").write_text(diff,encoding="utf-8")
        atomic_json(raw_dir/"partial.json",{
            "stopped":True,"steps":steps,"tool_calls":tool_calls,"test_runs":test_runs,
            "files_changed":files,"diff_added_lines":added,"diff_removed_lines":removed,
        })
        return AgentRunResult(stopped=True,passed=False,exit_code=-1,stdout="",stderr="",steps=steps,tool_calls=tool_calls,test_runs=test_runs,
            files_changed=files,diff=diff,diff_added_lines=added,diff_removed_lines=removed,final_message=final_message,raw_dir=str(raw_dir))

    rc,out,err=run_fixture_tests(workspace)
    diff=git_diff(workspace)
    files=changed_files(workspace)
    added,removed=count_diff_lines(diff)
    (raw_dir/"final-tests.stdout.txt").write_text(out,encoding="utf-8")
    (raw_dir/"final-tests.stderr.txt").write_text(err,encoding="utf-8")
    (raw_dir/"final.diff").write_text(diff,encoding="utf-8")
    atomic_json(raw_dir/"final.json",{
        "passed":rc==0,"exit_code":rc,"steps":steps,"tool_calls":tool_calls,"test_runs":test_runs,
        "files_changed":files,"diff_added_lines":added,"diff_removed_lines":removed,"final_message":final_message,
    })
    return AgentRunResult(
        stopped=False,passed=rc==0,exit_code=rc,stdout=out,stderr=err,steps=steps,tool_calls=tool_calls,test_runs=test_runs,
        files_changed=files,diff=diff,diff_added_lines=added,diff_removed_lines=removed,final_message=final_message,
        raw_dir=str(raw_dir),
    )
