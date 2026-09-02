from __future__ import annotations

import hashlib
import json
import re
import shutil
import statistics
from pathlib import Path
from typing import Any, Callable

from lib.benchlib import atomic_json, evaluate_checks, load_json, ollama_chat, response_metrics, utc_now
from scripts.agent_harness import init_workspace, run_agent_task
from scripts.livecodebench_evalscope import run_livecodebench

CHAT_PROFILES={
    "quick": {"tests":["C1","C2","C4","C6","C8","C9"],"repeats":1,"agents":["A1","A5"],"agent_repeats":1,"max_steps":22},
    "standard": {"tests":["C1","C2","C3","C4","C5","C6","C7","C8","C9"],"repeats":3,"agents":["A1","A2","A3","A4","A5"],"agent_repeats":1,"max_steps":40},
    "full": {"tests":["C1","C2","C3","C4","C5","C6","C7","C8","C9"],"repeats":3,"agents":["A1","A2","A3","A4","A5"],"agent_repeats":3,"max_steps":60},
}

SYSTEM=(
    "You are a senior PHP 8.3, Laravel, Livewire, SQL and modern browser JavaScript engineer. "
    "Follow the stated project conventions exactly. Do not invent APIs or facts. When code is requested, provide complete relevant code. "
    "Prefer secure, testable, minimal changes."
)

TASKS: dict[str,dict[str,Any]]={
"C1":{
 "title":"Laravel endpoint contract",
 "prompt":'''Implement a Laravel endpoint that updates a Project status. Requirements:\n- Route/controller receives an authenticated Project.\n- Validation must be in a dedicated Form Request.\n- status is required and only draft, active, archived are allowed.\n- authorization belongs in the Form Request.\n- return HTTP 200 JSON exactly with keys id and status.\n- invalid input must use Laravel's normal 422 validation response.\n- include concise Pest feature tests for success, unauthorized access and invalid status.\nShow the relevant PHP files.''',
 "checks":[
  {"name":"Form Request","patterns":[r"FormRequest",r"class\s+\w+Request"],"kind":"all"},
  {"name":"allowed statuses","patterns":[r"draft",r"active",r"archived"],"kind":"all"},
  {"name":"authorization","patterns":[r"function\s+authorize|authorize\s*\(",r"can\(|policy|user\(\)"],"kind":"all"},
  {"name":"validated data","patterns":[r"validated\(|safe\(\)"],"kind":"any"},
  {"name":"JSON id status","patterns":[r"['\"]id['\"]",r"['\"]status['\"]"],"kind":"all"},
  {"name":"Pest tests","patterns":[r"it\(|test\(",r"assertStatus\(200\)|assertOk\(\)",r"422|assertInvalid",r"403|assertForbidden"],"kind":"all"},
  {"name":"no manual 422","patterns":[r"response\(\)->json\([^)]{0,300},\s*422\s*\)",r"abort\(\s*422\s*\)"],"kind":"not"},
 ]},
"C2":{
 "title":"Secure transactional fix",
 "prompt":'''Review and replace this Laravel controller method with a safe implementation. Explain only the essential fixes, then show the corrected method.\n\n```php\npublic function store(Request $request)\n{\n    $user = User::find($request->user_id);\n    $user->roles()->attach($request->roles);\n    $user->update($request->all());\n    return response()->json($user, 201);\n}\n```\nRequirements: validate user_id and roles, reject invalid role IDs, make the user update plus role sync atomic, avoid mass assignment of arbitrary request data, and return 201.''',
 "checks":[
  {"name":"validation","patterns":[r"validate\(|validated\(|FormRequest",r"roles.*array",r"exists:.*roles|Rule::exists"],"kind":"all"},
  {"name":"transaction","patterns":[r"DB::transaction|beginTransaction"],"kind":"any"},
  {"name":"sync roles","patterns":[r"roles\(\)->sync"],"kind":"any"},
  {"name":"no request all","patterns":[r"request->all\(|\$request->all\("],"kind":"not"},
  {"name":"201","patterns":[r"201|HTTP_CREATED"],"kind":"any"},
 ]},
"C3":{
 "title":"Eloquent N+1 and aggregate",
 "prompt":'''A Laravel page lists 50,000 customers. Current code calls `$customer->orders()->count()` and `$customer->orders()->sum('total')` inside the Blade loop and renders all customers at once. Propose corrected Eloquent/query code. Requirements: no N+1 aggregate queries, server-side pagination, preserve order count and total order value per customer, and name useful database indexes.''',
 "checks":[
  {"name":"withCount","patterns":[r"withCount\s*\(.*orders"],"kind":"any"},
  {"name":"aggregate","patterns":[r"withSum\s*\(.*orders|selectSub|joinSub"],"kind":"any"},
  {"name":"pagination","patterns":[r"paginate\(|cursorPaginate\("],"kind":"any"},
  {"name":"indexes","patterns":[r"index",r"customer_id"],"kind":"all"},
  {"name":"no loop query recommendation","patterns":[r"foreach[\s\S]{0,200}orders\(\)->(?:count|sum)"],"kind":"not"},
 ]},
"C4":{
 "title":"Race-safe live search",
 "prompt":'''Write browser JavaScript for a search input that fetches `/api/search?q=...`. Requirements: 300 ms debounce, AbortController cancels the previous request, stale responses can never overwrite newer results, render API strings without unsafe innerHTML, show loading/error/empty states, and do not use jQuery.''',
 "checks":[
  {"name":"debounce","patterns":[r"300",r"setTimeout|debounce"],"kind":"all"},
  {"name":"AbortController","patterns":[r"AbortController",r"\.abort\("],"kind":"all"},
  {"name":"stale protection","patterns":[r"requestId|sequence|generation|controller\s*!==|signal\.aborted"],"kind":"any"},
  {"name":"safe text rendering","patterns":[r"textContent|createTextNode"],"kind":"any"},
  {"name":"no innerHTML","patterns":[r"innerHTML\s*="],"kind":"not"},
  {"name":"states","patterns":[r"loading",r"error",r"empty|no results"],"kind":"all"},
  {"name":"no jquery","patterns":[r"\$\(|jQuery"],"kind":"not"},
 ]},
"C5":{
 "title":"Code review",
 "prompt":'''Review this Laravel code. Identify concrete correctness, security, data-integrity or performance problems. Do not invent issues that are not supported by the snippet.\n\n```php\npublic function import(Request $request)\n{\n    foreach ($request->file('csv')->get() as $row) {\n        $project = Project::create($row);\n        foreach ($row['member_ids'] as $id) {\n            $project->members()->attach($id);\n        }\n    }\n    return back();\n}\n```\nAssume `$row` is untrusted parsed CSV input.''',
 "checks":[
  {"name":"mass assignment/untrusted fields","patterns":[r"mass assignment|Project::create\(\$row\)|untrusted.*fields|allowlist"],"kind":"any"},
  {"name":"validation","patterns":[r"validat"],"kind":"any"},
  {"name":"transaction/partial import","patterns":[r"transaction|partial import|atomic|rollback"],"kind":"any"},
  {"name":"duplicate pivot risk","patterns":[r"duplicate|syncWithoutDetaching|pivot|attach"],"kind":"any"},
  {"name":"CSV parsing caveat","patterns":[r"file\(.*csv.*\)->get|raw bytes|parse.*csv|CSV parser"],"kind":"any"},
 ]},
"C6":{
 "title":"Conversational debugging",
 "prompt":'''What is going wrong here and how would you fix it? The UI sometimes shows results for an older query after the user has already typed a newer query.\n\n```js\ninput.addEventListener('input', async () => {\n  const q = input.value;\n  const response = await fetch(`/api/search?q=${encodeURIComponent(q)}`);\n  const data = await response.json();\n  results.textContent = data.map(x => x.name).join(', ');\n});\n```\nKeep the explanation concise and show a robust corrected version.''',
 "checks":[
  {"name":"race root cause","patterns":[r"race condition|out of order|older.*response|stale response"],"kind":"any"},
  {"name":"cancellation","patterns":[r"AbortController",r"abort\("],"kind":"all"},
  {"name":"stale guard","patterns":[r"requestId|sequence|latest|controller\s*!==|signal\.aborted"],"kind":"any"},
 ]},
"C7":{
 "title":"Laravel Livewire state",
 "prompt":'''Create a Livewire component for editing a Project name. Requirements: public `$name`, populate it from the Project on mount, validate required|string|max:120, authorize update, save the Project, dispatch a `project-saved` event, and Blade must use `wire:model`, a project `<x-actions.button>`, and scoped `wire:loading.attr="loading"` + `wire:target="save"`.''',
 "checks":[
  {"name":"mount state","patterns":[r"function\s+mount",r"\$this->name"],"kind":"all"},
  {"name":"validation","patterns":[r"required",r"string",r"max:120"],"kind":"all"},
  {"name":"authorization","patterns":[r"authorize\(|Gate::|->can\("],"kind":"any"},
  {"name":"save","patterns":[r"function\s+save",r"->save\(|->update\("],"kind":"all"},
  {"name":"event","patterns":[r"dispatch\(['\"]project-saved"],"kind":"any"},
  {"name":"project button","patterns":[r"<x-actions\.button"],"kind":"any"},
  {"name":"loading target","patterns":[r"wire:loading\.attr=['\"]loading['\"]",r"wire:target=['\"]save['\"]"],"kind":"all"},
 ]},
"C8":{
 "title":"Livewire and Web Awesome dropdown",
 "prompt":'''Using the project's Blade wrappers, create a Livewire status dropdown. Project rules:\n- dropdown: `<x-actions.dropdown>`\n- trigger: `<x-actions.button slot-name="trigger" with-caret>`\n- items: `<x-actions.dropdown-item value="...">`\n- the parent emits `wa-select`; use Alpine `x-on:wa-select` to call `$wire.selectStatus($event.detail.item.value)`\nStatuses are all, active, paused and archived. Also show the Livewire `selectStatus(string $status)` method and reject unsupported values.''',
 "checks":[
  {"name":"wrappers","patterns":[r"<x-actions\.dropdown",r"<x-actions\.button",r"<x-actions\.dropdown-item"],"kind":"all"},
  {"name":"trigger convention","patterns":[r"slot-name=['\"]trigger['\"]",r"with-caret"],"kind":"all"},
  {"name":"event bridge","patterns":[r"x-on:wa-select",r"\$wire\.selectStatus\(\$event\.detail\.item\.value\)"],"kind":"all"},
  {"name":"statuses","patterns":[r"all",r"active",r"paused",r"archived"],"kind":"all"},
  {"name":"server validation","patterns":[r"in_array|Rule::in|ValidationException|abort\("],"kind":"any"},
  {"name":"no raw WA","patterns":[r"<wa-dropdown|<wa-button"],"kind":"not"},
 ]},
"C9":{
 "title":"Project component discovery",
 "prompt":'''You are given these project conventions:\n\nBUTTON: `<x-actions.button>` defaults are project-defined. For Laravel navigation use `route="name"`; it adds `wire:navigate`. For a dropdown trigger use `slot-name="trigger"`, not raw `slot`. Livewire actions use directives on the individual button.\nBUTTON GROUP: `<x-actions.button-group label="...">` requires a meaningful accessible label.\nCOPY BUTTON: `<x-actions.copy-button>` can copy literal `value` or use `from="element-id"`; `copy-label` is the accessible name.\n\nImplement a compact project action toolbar containing: Home navigation via route `home`; a grouped Save button with Livewire `save` and scoped loading feedback; and a copy button that copies text from element `project-url` with the accessible label `Copy project URL`. Use only project wrappers for these controls.''',
 "checks":[
  {"name":"home route","patterns":[r"<x-actions\.button[^>]*route=['\"]home['\"]"],"kind":"any"},
  {"name":"group label","patterns":[r"<x-actions\.button-group[^>]*label=['\"][^'\"]+['\"]"],"kind":"any"},
  {"name":"save livewire","patterns":[r"wire:click=['\"]save['\"]",r"wire:loading\.attr=['\"]loading['\"]",r"wire:target=['\"]save['\"]"],"kind":"all"},
  {"name":"copy from","patterns":[r"<x-actions\.copy-button[^>]*from=['\"]project-url['\"]",r"copy-label=['\"]Copy project URL['\"]"],"kind":"all"},
  {"name":"no raw WA","patterns":[r"<wa-button|<wa-button-group|<wa-copy-button"],"kind":"not"},
 ]},
}

AGENTS={
 "A1":{
  "title":"Find and fix median bug","fixture":"a1",
  "issue":"The median helper returns the wrong value for even-length inputs. Fix the implementation. Preserve the documented non-mutation behaviour and existing error behaviour. Run the tests before finishing.",
 },
 "A2":{
  "title":"Add repository feature","fixture":"a2",
  "issue":"Extend list_projects() with an optional status filter while preserving the existing query filter and alphabetical ordering. `status=None` means no status filter. Unknown statuses simply return no matching projects. Keep the API backwards compatible and run tests.",
 },
 "A3":{
  "title":"Debug and recover","fixture":"a3",
  "issue":"parse_duration() mishandles at least one documented unit. Diagnose the problem, make the smallest correct fix, and run the tests. Do not broaden the accepted syntax beyond the documented units.",
 },
 "A4":{
  "title":"Navigate larger repository","fixture":"a4",
  "issue":"A route-normalization regression is changing case-sensitive route segments. Find the authoritative contract in the repository, locate the implementation, fix only the relevant behaviour, and run tests.",
 },
 "A5":{
  "title":"Laravel Livewire Web Awesome repo task","fixture":"a5",
  "issue":"Add a project status filter to the existing Livewire ProjectList. Status must default to `all`, support all/active/paused/archived, and reject unsupported values in selectStatus(string $status). resetFilters() must reset both search and status. In the Blade view add a project-wrapper dropdown whose trigger follows project conventions, whose items have stable values, and whose wa-select event calls the Livewire method. Put the status controls and reset action in an accessible project button group. Preserve scoped loading feedback. Read the repository documentation before implementing and run tests.",
 },
}


def _slug(model:str,mode:str)->str:
    return (model+"__"+mode).replace("/","_").replace(":","_")


def coding_job_count(profile:str, include_livecodebench:bool=False)->int:
    cfg=CHAT_PROFILES[profile]
    return len(cfg["tests"])*int(cfg["repeats"]) + len(cfg["agents"])*int(cfg["agent_repeats"]) + (1 if include_livecodebench else 0)


def _call(api:str,model:str,mode:str,prompt:str,*,temperature:float,seed:int,context:int):
    payload,response,elapsed,power=ollama_chat(api,model,[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],mode,temperature=temperature,seed=seed,context=context)
    metrics=response_metrics(response,elapsed)
    answer=((response.get("message") or {}).get("content") or "")
    return payload,response,metrics,power,answer


def _chat_pass(checks:list[dict[str,Any]])->bool:
    return bool(checks) and all(bool(c.get("passed")) for c in checks)


def run_coding_agent(
    *,repo_root:Path,run_dir:Path,model:str,mode:str,profile:str,api:str,temperature:float,seed:int,context:int,
    is_completed:Callable[[str],bool],mark_completed:Callable[[str,dict[str,Any]],None],should_stop:Callable[[],bool],set_current:Callable[[dict[str,Any]],None],
    include_livecodebench:bool=False,
)->dict[str,Any]:
    cfg=CHAT_PROFILES[profile]; slug=_slug(model,mode)
    base=run_dir/"raw/coding-agent"/slug; base.mkdir(parents=True,exist_ok=True)
    chat_results=[]; agent_results=[]

    # Code-chat tests
    for test_id in cfg["tests"]:
        task=TASKS[test_id]
        for rep in range(1,int(cfg["repeats"])+1):
            if should_stop():
                return {"stopped":True,"model":model,"mode":mode,"profile":profile,"chat":_summarize_chat(chat_results),"agent":_summarize_agents(agent_results)}
            logical=f"coding-agent|{model}|{mode}|{test_id}|{rep}"
            path=base/"chat"/test_id/f"repeat-{rep}.json"
            if is_completed(logical) and path.exists():
                chat_results.append(load_json(path)); continue
            set_current({"benchmark":"coding-agent","model":model,"mode":mode,"test":test_id,"repeat":rep,"repeats":int(cfg["repeats"])})
            payload,response,metrics,power,answer=_call(api,model,mode,task["prompt"],temperature=temperature,seed=seed,context=context)
            checks=evaluate_checks(answer,task["checks"]); passed=_chat_pass(checks)
            result={"type":"code-chat","test":test_id,"title":task["title"],"repeat":rep,"model":model,"mode":mode,"completed_at":utc_now(),
                    "pass":passed,"checks_passed":sum(c["passed"] for c in checks),"checks_total":len(checks),"checks":checks,
                    "prompt":task["prompt"],"final_answer":answer,"call":{"request":payload,"response":response,"metrics":metrics,"power":power}}
            atomic_json(path,result); mark_completed(logical,result); chat_results.append(result)

    # Agent repository tests
    fixture_root=repo_root/"benchmarks/coding-agent/fixtures/agents"
    for agent_id in cfg["agents"]:
        meta=AGENTS[agent_id]
        for rep in range(1,int(cfg["agent_repeats"])+1):
            if should_stop():
                return {"stopped":True,"model":model,"mode":mode,"profile":profile,"chat":_summarize_chat(chat_results),"agent":_summarize_agents(agent_results)}
            logical=f"coding-agent|{model}|{mode}|{agent_id}|{rep}"
            result_path=base/"agent"/agent_id/f"repeat-{rep}"/"result.json"
            if is_completed(logical) and result_path.exists():
                agent_results.append(load_json(result_path)); continue
            raw_dir=base/"agent"/agent_id/f"repeat-{rep}"
            workspace=raw_dir/"workspace"
            init_workspace(fixture_root/meta["fixture"],workspace)
            set_current({"benchmark":"coding-agent","model":model,"mode":mode,"test":agent_id,"repeat":rep,"repeats":int(cfg["agent_repeats"]),"turn":0,"turns":int(cfg["max_steps"])})
            ar=run_agent_task(workspace=workspace,raw_dir=raw_dir/"steps",issue=meta["issue"],model=model,mode=mode,api=api,
                              temperature=temperature,seed=seed,context=context,max_steps=int(cfg["max_steps"]),should_stop=should_stop,set_current=set_current,
                              current_base={"benchmark":"coding-agent","model":model,"mode":mode,"test":agent_id,"repeat":rep,"repeats":int(cfg["agent_repeats"])})
            if ar.stopped:
                return {"stopped":True,"model":model,"mode":mode,"profile":profile,"chat":_summarize_chat(chat_results),"agent":_summarize_agents(agent_results),
                        "interrupted_agent":{"test":agent_id,"repeat":rep,"raw_dir":str(raw_dir)}}
            result={"type":"agent","test":agent_id,"title":meta["title"],"repeat":rep,"model":model,"mode":mode,"completed_at":utc_now(),
                    "pass":ar.passed,"checks_passed":int(ar.passed),"checks_total":1,"issue":meta["issue"],"exit_code":ar.exit_code,
                    "steps":ar.steps,"tool_calls":ar.tool_calls,"test_runs":ar.test_runs,"files_changed":ar.files_changed,
                    "diff_added_lines":ar.diff_added_lines,"diff_removed_lines":ar.diff_removed_lines,"final_message":ar.final_message,
                    "final_test_stdout":ar.stdout,"final_test_stderr":ar.stderr,"diff_file":"steps/final.diff"}
            atomic_json(result_path,result); mark_completed(logical,result); agent_results.append(result)

    external={}
    if include_livecodebench:
        lcb=run_livecodebench(repo_root=repo_root,run_dir=run_dir,model=model,mode=mode,profile=profile,api=api,temperature=temperature,seed=seed,
                              is_completed=is_completed,mark_completed=mark_completed,should_stop=should_stop,set_current=set_current)
        external["livecodebench"]=lcb
        if lcb.get("stopped"):
            return {"stopped":True,"model":model,"mode":mode,"profile":profile,"chat":_summarize_chat(chat_results),"agent":_summarize_agents(agent_results),"external":external}

    return {"stopped":False,"model":model,"mode":mode,"profile":profile,"chat":_summarize_chat(chat_results),"agent":_summarize_agents(agent_results),"external":external}


def _summarize_chat(rows:list[dict[str,Any]])->dict[str,Any]:
    if not rows: return {"items":0}
    by={}
    for tid in sorted({r["test"] for r in rows}):
        vals=[r for r in rows if r["test"]==tid]
        by[tid]={"runs":len(vals),"passed":sum(bool(v["pass"]) for v in vals),"pass_rate":sum(bool(v["pass"]) for v in vals)/len(vals),
                 "checks_passed":sum(int(v.get("checks_passed",0)) for v in vals),"checks_total":sum(int(v.get("checks_total",0)) for v in vals)}
    calls=[r["call"] for r in rows]
    tps=[c["metrics"].get("generation_tokens_per_second") for c in calls if c["metrics"].get("generation_tokens_per_second") is not None]
    wall=[c["metrics"].get("wall_seconds") for c in calls if c["metrics"].get("wall_seconds") is not None]
    return {"items":len(rows),"passed":sum(bool(r["pass"]) for r in rows),"pass_rate":sum(bool(r["pass"]) for r in rows)/len(rows),"per_test":by,
            "performance":{"generation_tps_median":statistics.median(tps) if tps else None,"wall_seconds_median":statistics.median(wall) if wall else None}}


def _summarize_agents(rows:list[dict[str,Any]])->dict[str,Any]:
    if not rows: return {"items":0}
    by={}
    for tid in sorted({r["test"] for r in rows}):
        vals=[r for r in rows if r["test"]==tid]
        by[tid]={"runs":len(vals),"resolved":sum(bool(v["pass"]) for v in vals),"resolve_rate":sum(bool(v["pass"]) for v in vals)/len(vals)}
    return {"items":len(rows),"resolved":sum(bool(r["pass"]) for r in rows),"resolve_rate":sum(bool(r["pass"]) for r in rows)/len(rows),"per_test":by,
            "tool_calls_median":statistics.median([r["tool_calls"] for r in rows]),"test_runs_median":statistics.median([r["test_runs"] for r in rows]),
            "diff_lines_median":statistics.median([r["diff_added_lines"]+r["diff_removed_lines"] for r in rows])}
