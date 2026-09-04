#!/usr/bin/env python3
"""Small LAN control panel for the AI Benchmark command line runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import tomllib
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("core", "coding-agent", "rag", "vision", "image", "speech", "music", "web")


def results_root() -> Path:
    configured = os.environ.get("BENCH_RESULTS_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local")) / "ai-benchmark"


def load_catalog() -> list[dict[str, Any]]:
    return tomllib.loads((ROOT / "config/models.toml").read_text(encoding="utf-8")).get("models", [])


def latest_scan(*, refresh: bool = False) -> dict[str, Any]:
    root = results_root()
    scans = sorted((root / "scans").glob("*/latest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if scans and not refresh:
        return json_file(scans[0], {})
    machine = socket.gethostname().split(".", 1)[0]
    output_dir = root / "scans" / machine
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/scan_machine.py"), "--models", str(ROOT / "config/models.toml"),
         "--output", str(output_dir / "latest.json"), "--eligible-config", str(output_dir / "eligible.models.tsv")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    return json_file(output_dir / "latest.json", {})


def installed_models(scan: dict[str, Any]) -> dict[str, bool]:
    """Read local model/cache state without downloading or contacting remote services."""
    present: set[str] = set()
    if shutil.which("ollama"):
        try:
            output = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.DEVNULL, timeout=5)
            for line in output.splitlines()[1:]:
                columns = line.split()
                if columns:
                    present.add(columns[0])
        except (OSError, subprocess.SubprocessError):
            pass
    storage = (scan.get("ollama_models_storage") or {}).get("path", "~/.ollama/models")
    cache = Path(os.environ.get("BENCH_CACHE_DIR") or Path(storage) / "benchmark-cache")
    for manifest in [*cache.glob("image/models/*.json"), cache / "music/prepared.json"]:
        for row in json_file(manifest, {}).get("models", []):
            if row.get("model") and row.get("local_path") and Path(row["local_path"]).exists():
                present.add(str(row["model"]))
    speech = json_file(cache / "speech/prepared.json", {})
    for model, path in (speech.get("whisper_cpp", {}).get("models", {}) or {}).items():
        if Path(path).exists():
            present.add(str(model))
    for model, assets in (speech.get("tts_assets", {}) or {}).items():
        if isinstance(assets, dict) and assets.get("onnx") and Path(assets["onnx"]).exists():
            present.add(str(model))
    return {str(row.get("model")): str(row.get("model")) in present or f"{row.get('model')}:latest" in present for row in load_catalog()}


def installed_model_sizes(scan: dict[str, Any]) -> dict[str, float]:
    """Best-effort local disk sizes for models managed by Ollama and benchmark caches."""
    sizes: dict[str, float] = {}
    if shutil.which("ollama"):
        try:
            for line in subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.DEVNULL, timeout=5).splitlines()[1:]:
                columns = line.split()
                if len(columns) >= 4:
                    factor = {"KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1, "TB": 1024}.get(columns[3].upper())
                    if factor is not None:
                        sizes[columns[0].removesuffix(":latest")] = round(float(columns[2]) * factor, 2)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    storage = (scan.get("ollama_models_storage") or {}).get("path", "~/.ollama/models")
    cache = Path(os.environ.get("BENCH_CACHE_DIR") or Path(storage) / "benchmark-cache")
    def add_path(model: str, value: Any) -> None:
        path = Path(str(value))
        if not path.exists():
            return
        total = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        sizes[model] = round(total / 1024**3, 2)
    for manifest in [*cache.glob("image/models/*.json"), cache / "music/prepared.json"]:
        for row in json_file(manifest, {}).get("models", []):
            if row.get("model") and row.get("local_path"):
                add_path(str(row["model"]), row["local_path"])
    speech = json_file(cache / "speech/prepared.json", {})
    for model, path in (speech.get("whisper_cpp", {}).get("models", {}) or {}).items():
        add_path(str(model), path)
    for model, assets in (speech.get("tts_assets", {}) or {}).items():
        if isinstance(assets, dict):
            add_path(str(model), assets.get("onnx", ""))
    return sizes


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return str(value)


def selected_catalog(models: set[str]) -> Path:
    destination = results_root() / "ui-selections" / f"models-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.toml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Per-run selection generated by the benchmark web interface.", ""]
    for row in load_catalog():
        lines.append("[[models]]")
        for key, value in row.items():
            if key == "enabled":
                value = bool(value) and row.get("model") in models
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def precheck_path(identifier: str) -> Path:
    return results_root() / "ui-prechecks" / f"{identifier}.json"


def reconcile_precheck(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("status") in {"running", "stopping"}:
        if not record.get("pid"):
            record.update({"status": "cancelled", "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        else:
            try:
                os.kill(int(record["pid"]), 0)
            except ProcessLookupError:
                record.update({"status": "cancelled", "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return record


def finish_precheck(record_path: Path, command: list[str], env: dict[str, str]) -> None:
    record = json_file(record_path, {})
    with Path(record["log"]).open("ab") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = process.pid
        write_json(record_path, record)
        exit_code = process.wait()
    record = json_file(record_path, record)
    status = "cancelled" if record.get("status") == "stopping" else "passed" if exit_code == 0 else "failed"
    record.update({"status": status, "exit_code": exit_code, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    write_json(record_path, record)


def json_file(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def run_directories() -> list[Path]:
    root = results_root() / "runs"
    return sorted((p for p in root.glob("*/*") if (p / "state.json").is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def run_summary(directory: Path) -> dict[str, Any]:
    state = json_file(directory / "state.json", {})
    return {"id": directory.name, "machine": directory.parent.name, "state": state}


def tail(path: Path, size: int | None = None, *, compact_ollama: bool = False) -> str:
    try:
        with path.open("rb") as handle:
            if size is not None:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - size))
            text = handle.read().decode("utf-8", errors="replace")
            # Progress bars use terminal cursor controls. They are useful in a terminal but not in a browser log.
            text = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))", "", text).replace("\r", "\n")
            if not compact_ollama:
                return text
            clean: list[str] = []
            for line in text.splitlines():
                marker = re.match(r"@@PRECHECK\|model-[^|]+\|Downloading ([^|]+)\|(running|done)", line)
                if marker:
                    clean.append(f"Downloading {marker.group(1)}" + (" completed." if marker.group(2) == "done" else "…"))
                    continue
                # Ollama rewrites these lines many times per second. The overview already presents their progress.
                if "pulling manifest" in line or re.search(r"pulling [0-9a-f]{8,}:", line, re.I):
                    continue
                clean.append(line)
            return "\n".join(clean)
    except OSError:
        return "No log is available yet."


PAGE_LEGACY = r"""<!doctype html><html lang="nl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Benchmark</title><style>
body{margin:0;padding:26px;background:#101827;color:#e6edf7;font:16px system-ui,sans-serif}h1{margin:0;font-size:30px}.intro,.muted{color:#aebbd1}.app{display:grid;grid-template-columns:minmax(0,2.2fr) minmax(360px,1fr);gap:20px;align-items:start}section{background:#182235;border:1px solid #2c3b55;border-radius:12px;padding:22px;margin:18px 0}.app section:first-child{margin-top:0}h2{margin:0 0 16px}h3{margin:24px 0 10px}.metrics{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px}.metric{background:#101827;border:1px solid #2c3b55;border-radius:8px;padding:12px}.metric b{display:block;font-size:18px;margin-top:3px}.metric small{display:block;color:#aebbd1;margin-top:5px}.space-green{color:#52c78b}.space-orange{color:#ffb454}.space-red{color:#ff6b6b}.suite-title,.overview-title{display:flex;align-items:center;justify-content:space-between}.badge{border-radius:999px;padding:5px 9px;font-size:12px;font-weight:700;background:#293956;color:#e6edf7}.badge-running{background:#4b371e;color:#ffb454}.badge-passed{background:#173b2b;color:#52c78b}.badge-failed{background:#47232b;color:#ff9292}.suite-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.suite-grid label{background:#101827;border:1px solid #2c3b55;border-radius:7px;padding:10px;cursor:pointer}.suite-grid b{display:block}.suite-grid small{display:block;color:#aebbd1;margin-left:23px;margin-top:2px}.models{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.model{background:#101827;border:1px solid #2c3b55;border-radius:7px;padding:10px}.model-head{display:flex;justify-content:space-between;gap:8px}.model strong{display:block}.tag{color:#9fddb9;font-size:13px}.installed{color:#52c78b;font-size:12px;font-weight:700;white-space:nowrap}.missing{color:#ff6b6b;font-size:12px;font-weight:700;white-space:nowrap}button{background:#52c78b;border:0;border-radius:6px;padding:10px 14px;font-weight:700;cursor:pointer}.secondary{background:#293956;color:#e6edf7}button:disabled{opacity:.45;cursor:not-allowed}pre{background:#0b1220;padding:12px;overflow:auto;max-height:42vh;white-space:pre-wrap}#message{font-weight:600}.precheck{border-left:3px solid #ffb454;background:#101827;padding:11px;margin:14px 0}.passed{border-color:#52c78b}.failed{border-color:#ff6b6b}.total{background:#101827;border:1px solid #2c3b55;border-radius:8px;padding:12px;margin-bottom:18px}.total b{font-size:18px}.total progress{width:100%;height:10px;accent-color:#52c78b}.step{margin:14px 0}.step-head{display:flex;justify-content:space-between;gap:10px;font-size:14px}.step-detail{display:block;color:#aebbd1;font-size:12px;margin:3px 0 6px}.substep{margin:8px 0 0 14px;padding-left:10px;border-left:2px solid #2c3b55}.step-done{color:#52c78b}.step-running{color:#ffb454}.step progress{width:100%;height:8px;accent-color:#52c78b}.step-running progress{accent-color:#ffb454}@media(max-width:900px){.app{grid-template-columns:1fr}.metrics,.suite-grid,.models{grid-template-columns:repeat(2,1fr)}}</style>
<style>
body{--cpu-glow:.08;--ram-glow:.08;--vram-glow:.03;background:radial-gradient(circle at 14% 10%,rgba(82,199,139,var(--cpu-glow)),transparent 27%),radial-gradient(circle at 73% 8%,rgba(91,141,239,var(--ram-glow)),transparent 26%),radial-gradient(circle at 92% 80%,rgba(255,180,84,var(--vram-glow)),transparent 30%),#101827}
body::before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.28;background-image:radial-gradient(circle at 12% 18%,#52c78b 0 1px,transparent 2px),radial-gradient(circle at 68% 22%,#6d9cff 0 1px,transparent 2px),radial-gradient(circle at 34% 77%,#52c78b 0 1px,transparent 2px),linear-gradient(27deg,transparent 49.7%,rgba(82,199,139,.18) 50%,transparent 50.3%),linear-gradient(153deg,transparent 49.7%,rgba(109,156,255,.12) 50%,transparent 50.3%);background-size:190px 170px,230px 210px,260px 240px,210px 180px,260px 220px;mask-image:linear-gradient(to bottom,black,transparent 88%)}
h1,.app{position:relative;z-index:1}.metric{box-shadow:inset 0 1px 0 rgba(130,181,255,.08),0 8px 26px rgba(4,10,22,.14)}.metric:hover{border-color:#45638d;box-shadow:0 0 20px rgba(82,199,139,.12)}
@media(prefers-reduced-motion:reduce){body::before{display:none}}
#neural-core{display:none}#force-graph{height:360px;flex:none;min-height:0;border-radius:11px;overflow:hidden;background:#182235}.neural-panel{height:360px;display:flex;flex-direction:column;min-height:0;padding:0!important;overflow:hidden}.overview-tools{display:flex;align-items:center;gap:10px}.overview-title h2{margin:0}
@media(max-width:900px){.side-top{grid-template-columns:1fr}.neural-panel{display:none}}
.models{grid-template-columns:repeat(4,1fr)}
.metrics{grid-template-columns:repeat(6,minmax(130px,1fr))}
@media(min-width:901px){
body{height:100vh;box-sizing:border-box;overflow:hidden}
.app{height:calc(100vh - 52px)}
.app>section{height:100%;box-sizing:border-box;display:flex;flex-direction:column;overflow:hidden}
#models{flex:1;min-height:0;overflow:auto;padding-right:5px}
#models,aside pre{scrollbar-width:none;-ms-overflow-style:none}
#models::-webkit-scrollbar,aside pre::-webkit-scrollbar{display:none}
.actions{display:flex;justify-content:center;align-items:center;gap:10px;flex-wrap:wrap}
aside{height:100%;display:grid;grid-template-rows:auto minmax(0,1fr);gap:18px;overflow:hidden}.side-top{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(240px,.85fr);gap:18px;align-items:stretch}.side-top section,aside>section{margin:0;box-sizing:border-box}.side-top section:last-child,aside>section:last-child{display:flex;flex-direction:column;min-height:0}aside pre{flex:1;min-height:0;max-height:none;margin:0}
}
</style>
<main class="app"><section><h2>Start a run</h2><div id="machine" class="metrics">Loading machine scan…</div><div class="suite-title"><h3>Suites</h3><button id="select-all" class="secondary" type="button">Select all suites</button></div><div id="suites" class="suite-grid"></div><h3>Approved models</h3><p id="model-summary" class="muted">Loading…</p><div id="models" class="models"></div><p>Profile: <select id="profile"><option value="quick">Quick</option><option value="standard" selected>Standard</option><option value="full">Full</option></select></p><div id="precheck-report" class="precheck">Run the precheck before starting a benchmark.</div><div class="actions"><button id="precheck">Run precheck</button><button id="stop-precheck" class="secondary" disabled>Stop precheck</button><button id="start" disabled>Start benchmark</button></div><span id="message"></span></section>
<aside><div class="side-top"><section><div class="overview-title"><h2>Precheck overview</h2><div class="overview-tools"><span id="precheck-status" class="badge">Not run</span></div></div><div id="precheck-steps" class="muted">Run a precheck to see its progress.</div></section><section class="neural-panel"><div id="force-graph"></div><canvas id="neural-core" width="240" height="260" aria-hidden="true"></canvas></section></div><section><h2>Technical log</h2><pre id="log">Select a precheck to view its log.</pre></section></aside></main>
<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/dist/3d-force-graph.min.js"></script><script>
let catalog=[],scan={},installed={},modelSizes={},precheck=null,followLog=true; const suiteLabels={core:['Algemene taalmodellen','Tekst, kennis en instructies'], 'coding-agent':['Code & agents','Programmeren en zelfstandige taken'],rag:['Documenten & RAG','Zoeken en antwoorden uit documenten'],vision:['Beeld & schermen','Afbeeldingen maken'],speech:['Spraak & audio','Spraakherkenning en spraaksynthese'],music:['Muziek maken','Tekst- en melodie-naar-muziek'],web:['Webonderzoek','Zoeken, bronnen en citaten']}; const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function get(url){let r=await fetch(url);if(!r.ok)throw Error(await r.text());return r.json()}
function setLog(text){let log=document.querySelector('#log');log.textContent=text;updatePrecheckBadge();if(followLog)log.scrollTop=log.scrollHeight}
document.querySelector('#log').addEventListener('scroll',e=>{let log=e.currentTarget;followLog=log.scrollTop+log.clientHeight>=log.scrollHeight-24})
function renderPrecheckSteps(output){let events=new Map;for(let line of output.split('\n')){if(!line.startsWith('@@PRECHECK|'))continue;let [,id,label,status]=line.split('|');events.set(id,{label,status})}let target=document.querySelector('#precheck-steps'),rows=[];function row(label,detail,value,done,substep=''){let percent=Math.max(0,Math.min(100,Math.round(value)));rows.push(`<div class="step ${done?'step-done':'step-running'}"><div class="step-head"><span>${esc(label)}</span><span>${done?'Complete':percent+'%'}</span></div><span class="step-detail">${esc(detail)}</span><progress value="${percent}" max="100"></progress>${substep}</div>`)}let system=events.get('environment'),systemValue=system?.status==='done'?100:0;if(system)row('System readiness','Checking disk space, required tools and the Ollama service.',systemValue,system.status==='done');let suites=(precheck?.suites||[]).map(s=>({id:'suite-'+s,name:(suiteLabels[s]||[s])[0]})),completedSuites=suites.filter(s=>events.get(s.id)?.status==='done').length,activeSuite=suites.find(s=>events.get(s.id)?.status==='running'),suiteValue=suites.length?completedSuites/suites.length*100:0;if(suites.length)row('Preparing benchmark suites',activeSuite?`${completedSuites} of ${suites.length} complete. Now preparing ${activeSuite.name}.`:completedSuites===suites.length?'All selected suites are ready.':`${completedSuites} of ${suites.length} suites prepared.`,suiteValue,completedSuites===suites.length);let models=(precheck?.models||[]).filter(m=>catalog.find(c=>c.model===m)?.backend==='ollama'),modelEvents=models.map(m=>({model:m,event:events.get('model-'+m)})),completedModels=modelEvents.filter(x=>x.event?.status==='done').length,activeModel=modelEvents.find(x=>x.event?.status==='running'),progressMatches=[...output.matchAll(/pulling\s+[^:\s]+:\s+(\d+)%/g)],downloadPercent=progressMatches.length?Number(progressMatches.at(-1)[1]):0,modelValue=models.length?(completedModels+(activeModel?downloadPercent/100:0))/models.length*100:0;if(models.length){let detail=completedModels===models.length?'All selected Ollama models are ready.':`${completedModels} of ${models.length} model downloads complete.`,substep=activeModel?`<div class="substep step-running"><div class="step-head"><span>${esc('Downloading '+activeModel.model)}</span><span>${downloadPercent}%</span></div><progress value="${downloadPercent}" max="100"></progress></div>`:'';row('Downloading Ollama models',detail,modelValue,completedModels===models.length,substep)}if(!rows.length){target.className='muted';target.textContent='Waiting for precheck progress…';return}let parts=[];if(system)parts.push([systemValue,10]);if(suites.length)parts.push([suiteValue,30]);if(models.length)parts.push([modelValue,60]);let total=Math.round(parts.reduce((sum,p)=>sum+p[0]*p[1],0)/parts.reduce((sum,p)=>sum+p[1],0));target.className='';target.innerHTML=`<div class="total"><div class="step-head"><b>Total precheck progress</b><b>${total}%</b></div><progress value="${total}" max="100"></progress></div>`+rows.join('')}
const renderPrecheckOverview=renderPrecheckSteps;
renderPrecheckSteps=output=>renderPrecheckOverview(String(precheck?.raw_output||output).replace(/^.*?(@@PRECHECK\|)/gm,'$1'));
function selectedSuites(){return [...document.querySelectorAll('input[name="suite"]:checked')].map(x=>x.value)}
function approvedModels(){let selected=new Set(selectedSuites()),approved=new Map((scan.models||[]).filter(x=>x.allowed).map(x=>[x.model,x]));return catalog.filter(m=>approved.has(m.model)&&m.suites.some(s=>selected.has(s))).map(m=>({...m,scan:approved.get(m.model)})).sort((a,b)=>a.model.localeCompare(b.model))}
function renderModels(){let models=approvedModels();document.querySelector('#model-summary').textContent=models.length?`${models.length} model(s) approved by this machine scan will be used for the selected suite(s).`:'No scan-approved models support the selected suite(s).';document.querySelector('#models').innerHTML=models.map(m=>{let ready=installed[m.model],label=ready?'Installed':'Not installed',r=m.scan.requirements;return `<div class="model"><div class="model-head"><strong>${esc(m.model)}</strong><span class="${ready?'installed':'missing'}">${label}</span></div><span class="tag">Required: ${r.ram_gib} GiB RAM · ${r.vram_gib} GiB VRAM · ${r.free_disk_gib} GiB disk</span></div>`}).join('')}
const renderModelCards=renderModels;renderModels=()=>{renderModelCards();approvedModels().forEach((model,index)=>{let size=modelSizes[model.model];if(size!==undefined)document.querySelectorAll('#models .tag')[index].textContent+=` · Installed: ${size.toFixed(2)} GiB`})}
function updatePrecheckBadge(){let status=precheck?.status||'missing',labels={missing:'Not run',running:'Running',stopping:'Stopping',passed:'Passed',failed:'Failed',cancelled:'Stopped'},badge=document.querySelector('#precheck-status');badge.textContent=labels[status]||status;badge.className='badge '+(status==='passed'?'badge-passed':status==='running'||status==='stopping'?'badge-running':status==='failed'||status==='cancelled'?'badge-failed':'')}
function invalidatePrecheck(){precheck=null;updatePrecheckBadge();document.querySelector('#start').disabled=true;document.querySelector('#stop-precheck').disabled=true;document.querySelector('#precheck-report').className='precheck';document.querySelector('#precheck-report').textContent='Run the precheck before starting a benchmark.'}
function precheckPayload(){return {models:approvedModels().map(m=>m.model),suites:selectedSuites(),profile:document.querySelector('#profile').value}}
function saveSelection(){localStorage.setItem('ai-benchmark-selection',JSON.stringify({suites:selectedSuites(),profile:document.querySelector('#profile').value}))}
function allSelectedModelsInstalled(){return approvedModels().every(m=>installed[m.model])}
async function refreshPrecheck(){if(!precheck)return;let d=await get('/api/precheck/'+precheck.id);precheck=d;let report=document.querySelector('#precheck-report');report.className='precheck '+(d.status==='passed'?'passed':d.status==='failed'||d.status==='cancelled'?'failed':'');report.textContent=d.status==='running'?'Precheck is running. Dependencies, cache and model downloads are being prepared…':d.status==='stopping'?'Stopping the precheck and its downloads…':d.status==='passed'?'Precheck passed. You can start this exact benchmark selection.':d.status==='cancelled'?'Precheck stopped. You can adjust the selection and run it again.':`Precheck failed (exit code ${d.exit_code}). Check the log below.`;setLog(d.output||'');renderPrecheckSteps(d.output||'');if(d.status==='passed'){let overview=await get('/api/overview');installed=overview.installed||{};renderModels()}document.querySelector('#start').disabled=d.status!=='passed'||!allSelectedModelsInstalled();document.querySelector('#precheck').disabled=d.status==='running'||d.status==='stopping';document.querySelector('#stop-precheck').disabled=d.status!=='running'||!d.pid;if(d.status==='running'||d.status==='stopping')setTimeout(refreshPrecheck,2500)}
async function restorePrecheck(){let response=await fetch('/api/precheck-status',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(precheckPayload())});if(!response.ok)return;let record=await response.json();if(record.status!=='missing'){precheck=record;await refreshPrecheck()}}
function diskMetric(label,disk){let free=Number(disk.free_gib),total=Number(disk.total_gib),used=Number(disk.used_gib),percent=total?Math.round(free/total*100):0,colour=percent<15?'space-red':percent<30?'space-orange':'space-green';return `<div class="metric"><span class="muted">${label}</span><b class="${colour}">${percent}% free</b><small>${free.toFixed(2)} GiB free / ${total.toFixed(2)} GiB total (${used.toFixed(2)} GiB used)</small></div>`}
function memoryMetric(label,free,total){if(!total)return `<div class="metric"><span class="muted">${label}</span><b>0 GiB</b><small>No GPU is reported.</small></div>`;let used=total-free,percent=Math.round(free/total*100),colour=percent<15?'space-red':percent<30?'space-orange':'space-green';return `<div class="metric"><span class="muted">${label}</span><b class="${colour}">${percent}% free</b><small>${free.toFixed(2)} GiB free / ${total.toFixed(2)} GiB total (${used.toFixed(2)} GiB used)</small></div>`}
function renderMetrics(){let store=scan.ollama_models_storage||{},root=scan.root_storage||{},same=scan.ollama_models_on_root_filesystem,load=Number(scan.cpu_load_percent),cpuColour=load>85?'space-red':load>60?'space-orange':'space-green',cpu=`<div class="metric"><span class="muted">CPU (1 min load)</span><b class="${cpuColour}">${Number.isFinite(load)?load.toFixed(0):'?'}%</b><small>${esc(scan.cpu||'Unknown CPU')} · ${scan.cpu_count??'?'} logical cores</small></div>`,ram=memoryMetric('Available RAM',Number(scan.available_ram_gib),Number(scan.total_ram_gib)),vram=memoryMetric('Free VRAM',Number(scan.available_vram_gib||0),Number(scan.total_vram_gib||0)),disks=same?diskMetric('Root & Ollama disk',store):diskMetric('Root disk',root)+diskMetric('Ollama disk',store);document.querySelector('#machine').innerHTML=cpu+ram+vram+disks}
const renderResourceCards=renderMetrics;
renderMetrics=()=>{renderResourceCards();let cpu=Math.max(0,Math.min(100,Number(scan.cpu_load_percent)||0)),ramTotal=Number(scan.total_ram_gib)||0,ramFree=Number(scan.available_ram_gib)||0,vramTotal=Number(scan.total_vram_gib)||0,vramFree=Number(scan.available_vram_gib)||0,style=document.documentElement.style;style.setProperty('--cpu-glow',(0.035+cpu/100*0.13).toFixed(3));style.setProperty('--ram-glow',(0.025+(ramTotal?1-ramFree/ramTotal:0)*0.12).toFixed(3));style.setProperty('--vram-glow',(0.015+(vramTotal?1-vramFree/vramTotal:0)*0.16).toFixed(3))}
const renderMetricsWithGpu=renderMetrics;
renderMetrics=()=>{renderMetricsWithGpu();let name=scan.gpu||'No dedicated GPU reported.',vram=Number(scan.total_vram_gib||0),card=`<div class="metric"><span class="muted">GPU</span><b>${vram?`${vram.toFixed(0)} GiB VRAM`:'0 GiB VRAM'}</b><small>${esc(name)}</small></div>`;document.querySelector('#machine').insertAdjacentHTML('afterbegin',card)}
async function refreshResources(){try{let d=await get('/api/resources');scan=d.scan||scan;installed=d.installed||installed;modelSizes=d.sizes||modelSizes;renderMetrics();renderModels()}catch(_){}}
async function load(){let d=await get('/api/overview');catalog=d.catalog;scan=d.scan||{};installed=d.installed||{};modelSizes=d.sizes||{};renderMetrics();
document.querySelector('#suites').innerHTML=d.suites.map((s,i)=>{let label=suiteLabels[s]||[s,''];return `<label><input type="checkbox" name="suite" value="${s}" ${i===0?'checked':''}><b>${label[0]}</b><small>${label[1]}</small></label>`}).join('');try{let saved=JSON.parse(localStorage.getItem('ai-benchmark-selection')||'{}');if(Array.isArray(saved.suites)){document.querySelectorAll('input[name="suite"]').forEach(x=>x.checked=saved.suites.includes(x.value))}if(['quick','standard','full'].includes(saved.profile))document.querySelector('#profile').value=saved.profile}catch(_){}document.querySelectorAll('input[name="suite"]').forEach(x=>x.onchange=()=>{invalidatePrecheck();saveSelection();renderModels()});document.querySelector('#select-all').onclick=()=>{document.querySelectorAll('input[name="suite"]').forEach(x=>x.checked=true);invalidatePrecheck();saveSelection();renderModels()};document.querySelector('#profile').onchange=()=>{invalidatePrecheck();saveSelection()};renderModels();await restorePrecheck()}
async function runs(){let d=await get('/api/runs');document.querySelector('#runs').innerHTML=d.length?d.map(r=>{let p=r.state.progress||{},active=['starting','running','resuming'].includes(r.state.status);return `<p><button onclick="showLog('${r.id}')">Log</button>${active?` <button class="secondary" onclick="stopRun('${r.id}')">Stop benchmark</button>`:''}<br><b>${esc(r.id)}</b> — ${esc(r.state.status||'unknown')} — ${p.completed||0}/${p.total||0} (${p.percent||0}%)</p>`}).join(''):'No runs yet.'}
async function stopRun(id){if(!confirm(`Stop benchmark ${id}? The current model call may finish first.`))return;let r=await fetch('/api/stop-run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id})}),d=await r.json();if(!r.ok)alert(d.error);else{document.querySelector('#message').textContent=d.message;runs()}}
async function showLog(id){setLog((await get('/api/log/'+encodeURIComponent(id))).log)}
document.querySelector('#precheck').onclick=async()=>{let payload=precheckPayload(),message=document.querySelector('#message');if(!payload.models.length||!payload.suites.length){message.textContent='Choose a suite with at least one scan-approved model.';return}message.textContent='Starting precheck…';try{let r=await fetch('/api/precheck',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});let d=await r.json();if(!r.ok)throw Error(d.error);precheck=d;message.textContent='';refreshPrecheck()}catch(e){message.textContent=e.message}}
document.querySelector('#stop-precheck').onclick=async()=>{if(!precheck)return;document.querySelector('#stop-precheck').disabled=true;await fetch('/api/precheck-stop',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id:precheck.id})});refreshPrecheck()}
document.querySelector('#start').onclick=async()=>{let payload=precheckPayload(),message=document.querySelector('#message');if(!precheck||precheck.status!=='passed')return;if(!confirm(`Start ${payload.suites.join(', ')} with all ${payload.models.length} approved model(s)?`))return;message.textContent='Starting benchmark…';try{let r=await fetch('/api/start',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({...payload,precheck_id:precheck.id})});let d=await r.json();if(!r.ok)throw Error(d.error);message.textContent=d.message}catch(e){message.textContent=e.message}}
const neuralCanvas=document.querySelector('#neural-core'),neuralContext=neuralCanvas.getContext('2d'),neuralPoints=Array.from({length:18},(_,i)=>{let a=i*2.399,z=((i*37)%100)/50-1,r=Math.sqrt(1-z*z);return [Math.cos(a)*r,Math.sin(a)*r,z]}),neuralEdges=neuralPoints.flatMap((_,i)=>[[i,(i+1)%18],[i,(i+5)%18]]);
function drawNeuralCore(time=0){let c=neuralContext,w=640,h=180,raw=Math.max(Number(scan.cpu_load_percent)||0,100-(Number(scan.total_ram_gib)?Number(scan.available_ram_gib)/Number(scan.total_ram_gib)*100:100),Number(scan.total_vram_gib)?100-Number(scan.available_vram_gib)/Number(scan.total_vram_gib)*100:0)/100,speed=.24+raw*1.9,angle=time/1000*speed,scale=68+raw*29,project=([x,y,z])=>{let rx=x*Math.cos(angle)-z*Math.sin(angle),rz=x*Math.sin(angle)+z*Math.cos(angle),ry=y*Math.cos(angle*.63)-rz*Math.sin(angle*.63);rz=y*Math.sin(angle*.63)+rz*Math.cos(angle*.63);let d=3.4+rz;return [w/2+rx*scale/d,h/2+ry*scale/d,rz]},nodes=neuralPoints.map(project);c.clearRect(0,0,w,h);c.fillStyle='#070e1b';c.fillRect(0,0,w,h);let aura=c.createRadialGradient(w/2,h/2,5,w/2,h/2,Math.max(w,h)*.6);aura.addColorStop(0,`rgba(19,105,74,${.22+raw*.18})`);aura.addColorStop(1,'rgba(7,14,27,0)');c.fillStyle=aura;c.fillRect(0,0,w,h);neuralEdges.forEach(([a,b],i)=>{let p=nodes[a],q=nodes[b],t=(time/1000*speed+i*.071)%1,x=p[0]+(q[0]-p[0])*t,y=p[1]+(q[1]-p[1])*t;c.strokeStyle=`rgba(82,199,139,${.12+raw*.45})`;c.lineWidth=1.2;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(q[0],q[1]);c.stroke();let g=c.createRadialGradient(x,y,0,x,y,5+raw*8);g.addColorStop(0,'#e2fff0');g.addColorStop(.3,'#52c78b');g.addColorStop(1,'rgba(82,199,139,0)');c.fillStyle=g;c.beginPath();c.arc(x,y,5+raw*8,0,Math.PI*2);c.fill()});nodes.sort((a,b)=>a[2]-b[2]).forEach(([x,y,z],i)=>{let r=2.4+(z+1)*1.8+raw*2.3,pulse=(Math.sin(time/(430-raw*230)+i)+1)/2,g=c.createRadialGradient(x,y,0,x,y,r*(3+pulse));g.addColorStop(0,'#effff6');g.addColorStop(.22,'#52c78b');g.addColorStop(1,'rgba(82,199,139,0)');c.fillStyle=g;c.beginPath();c.arc(x,y,r*(3+pulse),0,Math.PI*2);c.fill()});c.fillStyle='#caffe1';c.font='600 12px system-ui';c.fillText(`3D NEURAL ACTIVITY  ·  ${Math.round(raw*100)}% LOAD`,18,h-16);requestAnimationFrame(drawNeuralCore)}
function drawNeuralCore3D(time=0){let c=neuralContext,w=240,h=260,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,raw=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,speed=.22+raw*2.1,angle=time/1000*speed,scale=77+raw*42,project=([x,y,z])=>{let rx=x*Math.cos(angle)-z*Math.sin(angle),rz=x*Math.sin(angle)+z*Math.cos(angle),ry=y*Math.cos(angle*.67)-rz*Math.sin(angle*.67);rz=y*Math.sin(angle*.67)+rz*Math.cos(angle*.67);let d=3.4+rz;return [w/2+rx*scale/d,h/2+ry*scale/d,rz]},nodes=neuralPoints.map(project);c.clearRect(0,0,w,h);c.fillStyle='#070e1b';c.fillRect(0,0,w,h);let aura=c.createRadialGradient(w/2,h/2,5,w/2,h/2,150);aura.addColorStop(0,`rgba(19,105,74,${.24+raw*.28})`);aura.addColorStop(1,'rgba(7,14,27,0)');c.fillStyle=aura;c.fillRect(0,0,w,h);neuralEdges.forEach(([a,b],i)=>{let p=nodes[a],q=nodes[b],t=(time/1000*speed+i*.071)%1,x=p[0]+(q[0]-p[0])*t,y=p[1]+(q[1]-p[1])*t;c.strokeStyle=`rgba(82,199,139,${.14+raw*.5})`;c.lineWidth=1.2;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(q[0],q[1]);c.stroke();let g=c.createRadialGradient(x,y,0,x,y,6+raw*9);g.addColorStop(0,'#e2fff0');g.addColorStop(.3,'#52c78b');g.addColorStop(1,'rgba(82,199,139,0)');c.fillStyle=g;c.beginPath();c.arc(x,y,6+raw*9,0,Math.PI*2);c.fill()});nodes.sort((a,b)=>a[2]-b[2]).forEach(([x,y,z],i)=>{let r=2.6+(z+1)*1.8+raw*2.6,pulse=(Math.sin(time/(450-raw*240)+i)+1)/2,g=c.createRadialGradient(x,y,0,x,y,r*(3+pulse));g.addColorStop(0,'#effff6');g.addColorStop(.22,'#52c78b');g.addColorStop(1,'rgba(82,199,139,0)');c.fillStyle=g;c.beginPath();c.arc(x,y,r*(3+pulse),0,Math.PI*2);c.fill()});c.fillStyle='#caffe1';c.font='600 11px system-ui';c.fillText(`3D NEURAL ACTIVITY`,14,h-31);c.fillStyle='#77e7ad';c.fillText(`VRAM ${Math.round(vram*100)}% · RAM ${Math.round(ram*100)}% · CPU ${Math.round(cpu*100)}%`,14,h-14);requestAnimationFrame(drawNeuralCore3D)}
const neuralScreenNodes=Array.from({length:38},(_,i)=>[.08+((i*71)%83)/100,.12+((i*37)%83)/100,((i*29)%100)/100]),neuralScreenEdges=neuralScreenNodes.flatMap((_,i)=>[[i,(i*7+5)%38],[i,(i+11)%38]]);
function drawNeuralReference(time=0){let c=neuralContext,w=240,h=340,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,load=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,speed=.18+load*1.9,nodes=neuralScreenNodes.map(([x,y,z],i)=>[x*w+Math.sin(time/950+i*1.9)*(2+load*7)*(z+.25),y*h+Math.cos(time/830+i*1.3)*(2+load*7)*(z+.25),z]);c.clearRect(0,0,w,h);c.fillStyle='#030b16';c.fillRect(0,0,w,h);c.strokeStyle='rgba(27,134,207,.06)';c.lineWidth=.5;for(let x=0;x<w;x+=8){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let y=0;y<h;y+=8){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}neuralScreenEdges.forEach(([a,b],i)=>{let p=nodes[a],q=nodes[b],t=(time/1000*speed+i*.083)%1,x=p[0]+(q[0]-p[0])*t,y=p[1]+(q[1]-p[1])*t;c.strokeStyle=`rgba(13,157,241,${.12+load*.35})`;c.lineWidth=.55+p[2]*.7;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(q[0],q[1]);c.stroke();if(i%2===0){let g=c.createRadialGradient(x,y,0,x,y,4+load*8);g.addColorStop(0,'#e4fbff');g.addColorStop(.22,'#1cb6ff');g.addColorStop(1,'rgba(28,182,255,0)');c.fillStyle=g;c.beginPath();c.arc(x,y,4+load*8,0,Math.PI*2);c.fill()}});nodes.sort((a,b)=>a[2]-b[2]).forEach(([x,y,z],i)=>{let r=1.3+z*1.7+load*1.8,p=(Math.sin(time/(490-load*260)+i)+1)/2,g=c.createRadialGradient(x,y,0,x,y,r*(3+p));g.addColorStop(0,'#dff9ff');g.addColorStop(.25,'#1bb7ff');g.addColorStop(1,'rgba(27,183,255,0)');c.fillStyle=g;c.beginPath();c.arc(x,y,r*(3+p),0,Math.PI*2);c.fill()});c.fillStyle='#6bdbff';c.font='600 11px system-ui';c.letterSpacing='2px';c.fillText('NEURAL NETWORK',14,24);c.letterSpacing='0px';c.fillStyle='#238fc4';c.fillText(`VRAM ${Math.round(vram*100)} · RAM ${Math.round(ram*100)} · CPU ${Math.round(cpu*100)}`,14,h-14);requestAnimationFrame(drawNeuralReference)}
const neuralAnchors=neuralScreenNodes.map(p=>[...p]);
function morphNeuralNetwork(){let cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,load=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,amplitude=.012+load*.18,t=performance.now()/1000;neuralScreenNodes.forEach((point,i)=>{let base=neuralAnchors[i],phase=t*(.32+load*2.1)+i*1.73;point[0]=Math.max(.035,Math.min(.965,base[0]+Math.sin(phase)*amplitude+Math.cos(phase*.47+i)*amplitude*.46));point[1]=Math.max(.08,Math.min(.92,base[1]+Math.cos(phase*1.19)*amplitude*1.55+Math.sin(phase*.39+i)*amplitude*.38));point[2]=Math.max(0,Math.min(1,base[2]+Math.sin(phase*.81)*(.04+load*.24)))})}
function drawNeuralCalm(time=0){let c=neuralContext,w=240,h=340,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,load=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,active=Math.min(neuralScreenNodes.length,Math.round(8+load*30)),nodes=neuralScreenNodes.slice(0,active).map(([x,y,z])=>[x*w,y*h,z]);c.clearRect(0,0,w,h);c.fillStyle='#030b16';c.fillRect(0,0,w,h);c.strokeStyle='rgba(27,134,207,.05)';c.lineWidth=.5;for(let x=0;x<w;x+=8){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let y=0;y<h;y+=8){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}for(let i=1;i<active;i++){let j=(i*7+3)%i,p=nodes[i],q=nodes[j];c.strokeStyle=`rgba(13,157,241,${.12+load*.35})`;c.lineWidth=.55+p[2]*.7;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(q[0],q[1]);c.stroke();if(load>.35&&i%2===0){let k=(i*3+1)%i,r=nodes[k];c.globalAlpha=(load-.35)*.45;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(r[0],r[1]);c.stroke();c.globalAlpha=1}}nodes.sort((a,b)=>a[2]-b[2]).forEach(([x,y,z],i)=>{let r=1.5+z*2+load*2.8,breath=(Math.sin(time/(1200-load*700)+i*1.4)+1)/2,g=c.createRadialGradient(x,y,0,x,y,r*(2.2+breath));g.addColorStop(0,'#dff9ff');g.addColorStop(.28,'#1bb7ff');g.addColorStop(1,'rgba(27,183,255,0)');c.fillStyle=g;c.beginPath();c.arc(x,y,r*(2.2+breath),0,Math.PI*2);c.fill()});c.fillStyle='#6bdbff';c.font='600 11px system-ui';c.letterSpacing='2px';c.fillText('NEURAL NETWORK',14,24);c.letterSpacing='0px';c.fillStyle='#238fc4';c.fillText(`VRAM ${Math.round(vram*100)} · RAM ${Math.round(ram*100)} · CPU ${Math.round(cpu*100)}`,14,h-14);requestAnimationFrame(drawNeuralCalm)}
const neuralVelocity=neuralScreenNodes.map(()=>[0,0,0]);
function morphNeuralDynamics(){let cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,load=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,energy=.00015+load*.006;neuralScreenNodes.forEach((point,i)=>{let velocity=neuralVelocity[i],home=neuralAnchors[i];velocity[0]+=(home[0]-point[0])*.004+(Math.random()-.5)*energy;velocity[1]+=(home[1]-point[1])*.004+(Math.random()-.5)*energy;velocity[2]+=(home[2]-point[2])*.004+(Math.random()-.5)*energy;velocity[0]*=.965;velocity[1]*=.965;velocity[2]*=.965;point[0]=Math.max(.03,Math.min(.97,point[0]+velocity[0]));point[1]=Math.max(.07,Math.min(.93,point[1]+velocity[1]));point[2]=Math.max(0,Math.min(1,point[2]+velocity[2]))})}
function drawNeuralCrisp(time=0){let c=neuralContext,w=240,h=340,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,load=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,active=Math.min(neuralScreenNodes.length,Math.round(8+load*30)),nodes=neuralScreenNodes.slice(0,active).map(([x,y,z])=>[x*w,y*h,z]);c.clearRect(0,0,w,h);c.fillStyle='#030b16';c.fillRect(0,0,w,h);c.strokeStyle='rgba(27,134,207,.05)';c.lineWidth=.5;for(let x=0;x<w;x+=8){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let y=0;y<h;y+=8){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}for(let i=1;i<active;i++){let j=(i*7+3)%i,p=nodes[i],q=nodes[j];c.strokeStyle=`rgba(13,157,241,${.14+load*.33})`;c.lineWidth=.55+p[2]*.65;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(q[0],q[1]);c.stroke();if(load>.35&&i%2===0){let r=nodes[(i*3+1)%i];c.globalAlpha=(load-.35)*.4;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(r[0],r[1]);c.stroke();c.globalAlpha=1}}nodes.sort((a,b)=>a[2]-b[2]).forEach(([x,y,z])=>{let r=1.35+z*1.35+load*1.65;c.fillStyle='#29c4ff';c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill();c.fillStyle='#e5fbff';c.beginPath();c.arc(x-.35,y-.35,Math.max(.55,r*.38),0,Math.PI*2);c.fill()});c.fillStyle='#6bdbff';c.font='600 11px system-ui';c.letterSpacing='2px';c.fillText('NEURAL NETWORK',14,24);c.letterSpacing='0px';requestAnimationFrame(drawNeuralCrisp)}
const neuralPresence=neuralScreenNodes.map(()=>0);
function updateNeuralPresence(){let cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,load=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,active=Math.round(8+load*30);neuralPresence.forEach((value,i)=>{let target=i<active?1:0;neuralPresence[i]+= (target-value)*(target?.055:.075)})}
function drawNeuralLifecycle(time=0){let c=neuralContext,w=240,h=340,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,load=vramTotal?vram*.6+ram*.28+cpu*.12:ram*.7+cpu*.3,nodes=neuralScreenNodes.map(([x,y,z],i)=>{let p=neuralPresence[i],cx=w/2,cy=h/2;return [cx+(x*w-cx)*p,cy+(y*h-cy)*p,z,p]}).filter(n=>n[3]>.015);c.clearRect(0,0,w,h);c.fillStyle='#030b16';c.fillRect(0,0,w,h);c.strokeStyle='rgba(27,134,207,.05)';c.lineWidth=.5;for(let x=0;x<w;x+=8){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let y=0;y<h;y+=8){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}for(let i=1;i<nodes.length;i++){let p=nodes[i],q=nodes[(i*7+3)%i];c.strokeStyle=`rgba(13,157,241,${(.11+load*.3)*Math.min(p[3],q[3])})`;c.lineWidth=.55+p[2]*.65;c.beginPath();c.moveTo(p[0],p[1]);c.lineTo(q[0],q[1]);c.stroke()}nodes.sort((a,b)=>a[2]-b[2]).forEach(([x,y,z,p])=>{let r=(1.35+z*1.35+load*1.65)*p;c.fillStyle='#29c4ff';c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill();c.fillStyle='#e5fbff';c.beginPath();c.arc(x-.35,y-.35,Math.max(.35,r*.38),0,Math.PI*2);c.fill()});c.fillStyle='#6bdbff';c.font='600 11px system-ui';c.letterSpacing='2px';c.fillText('NEURAL NETWORK',14,24);c.letterSpacing='0px';requestAnimationFrame(drawNeuralLifecycle)}
function drawResourceNetwork(time=0){let c=neuralContext,w=240,h=340,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,gpu=vramTotal?Math.max(vram,Number(scan.cpu_load_percent||0)/100*.35):0,resources=[['RAM',ram,64,105],['CPU',cpu,176,105],['VRAM',vram,65,239],['GPU',gpu,175,239]],core=[120,170];c.clearRect(0,0,w,h);c.fillStyle='#030b16';c.fillRect(0,0,w,h);c.strokeStyle='rgba(27,134,207,.05)';for(let x=0;x<w;x+=8){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let y=0;y<h;y+=8){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}let dot=(x,y,r=2)=>{c.fillStyle='#29c4ff';c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill()};resources.forEach(([label,load,x,y],ri)=>{let count=Math.floor(load*100/5)*2,cx=x+(Math.sin(time/1800+ri)*3),cy=y+(Math.cos(time/1700+ri)*3);c.strokeStyle=`rgba(13,157,241,${.3+load*.4})`;c.lineWidth=1;c.beginPath();c.moveTo(core[0],core[1]);c.lineTo(cx,cy);c.stroke();dot(cx,cy,3);c.fillStyle='#69dbff';c.font='600 10px system-ui';c.fillText(label,cx-13,cy-10);for(let i=0;i<count;i++){let a=i*2.4+time/(1200-load*700),d=12+(i%6)*7+load*20,nx=cx+Math.cos(a)*d,ny=cy+Math.sin(a)*d;c.strokeStyle=`rgba(13,157,241,${.18+load*.4})`;c.beginPath();c.moveTo(cx,cy);c.lineTo(nx,ny);c.stroke();if(i>0){let px=cx+Math.cos(a-2.4)*(12+((i-1)%6)*7+load*20),py=cy+Math.sin(a-2.4)*(12+((i-1)%6)*7+load*20);c.beginPath();c.moveTo(px,py);c.lineTo(nx,ny);c.stroke()}dot(nx,ny,1.2+load*1.4)}});dot(core[0],core[1],5);c.fillStyle='#dff9ff';c.font='600 10px system-ui';c.fillText('CORE',105,188);requestAnimationFrame(drawResourceNetwork)}
const drawResourceBase=drawResourceNetwork;
drawResourceNetwork=time=>{drawResourceBase(time);let c=neuralContext,points=[[64,105],[176,105],[65,239],[175,239]];for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){c.strokeStyle='rgba(27,168,244,.28)';c.lineWidth=.8;c.beginPath();c.moveTo(points[i][0],points[i][1]);c.lineTo(points[j][0],points[j][1]);c.stroke()}};
function drawOrganicResourceNetwork(time=0){let c=neuralContext,w=240,h=340,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,gpu=vramTotal?Math.max(vram,cpu*.35):0,t=time/1000,core=[w/2+Math.sin(t*.8)*22+Math.cos(t*1.47)*10,h/2+Math.cos(t*.69)*27+Math.sin(t*1.19)*12],resources=[['RAM',ram,60,96],['CPU',cpu,181,92],['VRAM',vram,58,244],['GPU',gpu,180,246]];c.clearRect(0,0,w,h);c.fillStyle='#030b16';c.fillRect(0,0,w,h);c.strokeStyle='rgba(27,134,207,.05)';for(let x=0;x<w;x+=8){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let y=0;y<h;y+=8){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}let dot=(x,y,r=2)=>{c.fillStyle='#29c4ff';c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill()},line=(a,b,alpha=.28)=>{c.strokeStyle=`rgba(13,157,241,${alpha})`;c.lineWidth=.75;c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.stroke()};resources.forEach(([name,load,bx,by],ri)=>{let anchor=[bx+Math.sin(t*.6+ri)*10,by+Math.cos(t*.71+ri)*11],relay1=[core[0]+(anchor[0]-core[0])*.32+Math.sin(t*1.3+ri)*9,core[1]+(anchor[1]-core[1])*.32+Math.cos(t*1.1+ri)*9],relay2=[core[0]+(anchor[0]-core[0])*.68+Math.cos(t*1.2+ri)*8,core[1]+(anchor[1]-core[1])*.68+Math.sin(t*1.4+ri)*8];line(core,relay1,.32);line(relay1,relay2,.27);line(relay2,anchor,.32);dot(relay1,1.5);dot(relay2,1.5);dot(anchor,3);c.fillStyle='#72ddff';c.font='600 10px system-ui';c.fillText(name,anchor[0]-13,anchor[1]-9);let count=Math.floor(load*100/5)*2;for(let i=0;i<count;i++){let angle=i*2.399+Math.sin(t*.7+i*1.6)*.7,radius=14+(i%7)*7+load*28,n=[anchor[0]+Math.cos(angle)*radius,anchor[1]+Math.sin(angle)*radius];line(anchor,n,.16+load*.3);if(i>0){let prevAngle=(i-1)*2.399+Math.sin(t*.7+(i-1)*1.6)*.7,prev=[anchor[0]+Math.cos(prevAngle)*(14+((i-1)%7)*7+load*28),anchor[1]+Math.sin(prevAngle)*(14+((i-1)%7)*7+load*28)];line(prev,n,.12+load*.24)}dot(n[0],n[1],1.2+load*1.2)}});dot(core[0],core[1],5);c.fillStyle='#e6fbff';c.font='600 10px system-ui';c.fillText('CORE',core[0]-15,core[1]+18);requestAnimationFrame(drawOrganicResourceNetwork)}
function drawStableResourceNetwork(time=0){let c=neuralContext,w=240,h=340,cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,gpu=vramTotal?Math.max(vram,cpu*.35):0,t=time/1000,core=[120+Math.sin(t*.45)*5,170+Math.cos(t*.38)*6],resources=[[ram,60,96],[cpu,181,92],[vram,58,244],[gpu,180,246]],dot=(x,y,r)=>{c.fillStyle='#29c4ff';c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill()},line=(a,b,a1)=>{c.strokeStyle=`rgba(13,157,241,${a1})`;c.lineWidth=.8;c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.stroke()};c.clearRect(0,0,w,h);c.fillStyle='#030b16';c.fillRect(0,0,w,h);c.strokeStyle='rgba(27,134,207,.05)';for(let x=0;x<w;x+=8){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let y=0;y<h;y+=8){c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}resources.forEach(([load,x,y],ri)=>{let anchor=[x+Math.sin(t*.35+ri)*3,y+Math.cos(t*.32+ri)*3],relay1=[core[0]+(anchor[0]-core[0])*.33,core[1]+(anchor[1]-core[1])*.33],relay2=[core[0]+(anchor[0]-core[0])*.67,core[1]+(anchor[1]-core[1])*.67],count=3+Math.floor(load*100/5)*2;line(core,relay1,.4);line(relay1,relay2,.34);line(relay2,anchor,.4);dot(relay1[0],relay1[1],2);dot(relay2[0],relay2[1],2);dot(anchor[0],anchor[1],3);for(let i=0;i<count;i++){let angle=i*2.399+Math.sin(t*.45+i)*.3,dist=14+(i%7)*7+load*22,n=[anchor[0]+Math.cos(angle)*dist,anchor[1]+Math.sin(angle)*dist];line(anchor,n,.24+load*.3);if(i>0){let pa=(i-1)*2.399+Math.sin(t*.45+i-1)*.3,pd=14+((i-1)%7)*7+load*22,prev=[anchor[0]+Math.cos(pa)*pd,anchor[1]+Math.sin(pa)*pd];line(prev,n,.18+load*.22)}dot(n[0],n[1],1.3+load*1.3)}});dot(core[0],core[1],5);requestAnimationFrame(drawStableResourceNetwork)}
let resourceGraph;function updateForceGraph(){if(!window.ForceGraph3D)return;let host=document.querySelector('#force-graph'),cpu=Math.max(0,Math.min(1,(Number(scan.cpu_load_percent)||0)/100)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?1-(Number(scan.available_ram_gib)||0)/ramTotal:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?1-(Number(scan.available_vram_gib)||0)/vramTotal:0,gpu=vramTotal?Math.max(vram,cpu*.35):0,groups=[['ram',ram],['cpu',cpu],['vram',vram],['gpu',gpu]],nodes=[{id:'core',size:8}],links=[];groups.forEach(([name,load])=>{nodes.push({id:name,size:5});links.push({source:'core',target:name});for(let i=0;i<3+Math.floor(load*20);i++){let id=`${name}-${i}`;nodes.push({id,size:2});links.push({source:name,target:id});if(i)links.push({source:`${name}-${i-1}`,target:id})}});if(!resourceGraph){resourceGraph=ForceGraph3D()(host).backgroundColor('#182235').nodeColor(()=>'#52c78b').nodeVal(n=>n.size).linkColor(()=>'rgba(82,199,139,.62)').linkWidth(.8).linkOpacity(.7).showNavInfo(false);resourceGraph.cameraPosition({z:190})}resourceGraph.width(host.clientWidth).height(host.clientHeight).graphData({nodes,links})}
const refreshForceGraph=updateForceGraph;let forceGraphBuckets='';updateForceGraph=()=>{let cpu=Math.max(0,Math.min(100,Number(scan.cpu_load_percent)||0)),ramTotal=Number(scan.total_ram_gib)||0,ram=ramTotal?100-(Number(scan.available_ram_gib)||0)/ramTotal*100:0,vramTotal=Number(scan.total_vram_gib)||0,vram=vramTotal?100-(Number(scan.available_vram_gib)||0)/vramTotal*100:0,gpu=vramTotal?Math.max(vram,cpu*.35):0,buckets=[cpu,ram,vram,gpu].map(value=>Math.floor(value/5)).join(':');if(buckets===forceGraphBuckets)return;forceGraphBuckets=buckets;refreshForceGraph()};const neuralFillRect=neuralContext.fillRect.bind(neuralContext);neuralContext.fillRect=(x,y,width,height)=>{if(neuralContext.fillStyle==='#030b16')neuralContext.fillStyle='#182235';neuralFillRect(x,y,width,height)};neuralContext.fillText=()=>{};neuralCanvas.height=340;setInterval(morphNeuralDynamics,40);setInterval(updateNeuralPresence,40);requestAnimationFrame(drawStableResourceNetwork);load().then(updateForceGraph).catch(e=>document.body.insertAdjacentHTML('beforeend','<pre>'+esc(e.message)+'</pre>'));setInterval(()=>refreshResources().then(updateForceGraph),10000)
let forceRotation=setInterval(()=>{if(resourceGraph){resourceGraph.cameraPosition({z:240});let controls=resourceGraph.controls();controls.autoRotate=true;controls.autoRotateSpeed=2.2;clearInterval(forceRotation)}},250);
let graphStateGuard=setInterval(()=>{if(!resourceGraph||resourceGraph._stateGuarded)return;let original=resourceGraph.graphData.bind(resourceGraph);resourceGraph.graphData=data=>{if(data){let previous=new Map(original().nodes.map(node=>[node.id,node]));data.nodes.forEach(node=>{let old=previous.get(node.id),parent=previous.get(String(node.id).split('-')[0])||previous.get('core');if(old)Object.assign(node,{x:old.x,y:old.y,z:old.z,vx:old.vx,vy:old.vy,vz:old.vz});else if(parent)Object.assign(node,{x:parent.x,y:parent.y,z:parent.z})})}return original(data)};resourceGraph._stateGuarded=true;clearInterval(graphStateGuard)},100);
</script></html>"""


PAGE = r"""<!doctype html><html lang="nl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Benchmark</title><style>
:root{color-scheme:dark}body{margin:0;background:#101827;color:#e6edf7;font:16px system-ui,sans-serif}.topbar{display:flex;align-items:center;gap:26px;padding:16px 26px;background:#182235;border-bottom:1px solid #2c3b55}h1{font-size:21px;margin:0}.menu{display:flex;gap:8px}.menu button{background:transparent;color:#aebbd1;border:1px solid transparent}.menu button.active{background:#293956;border-color:#45638d;color:#fff}.page{display:none;width:100%;box-sizing:border-box;padding:20px 26px}.page.active{display:block}.grid{display:grid;grid-template-columns:minmax(0,2.2fr) minmax(360px,1fr);gap:20px}section{background:#182235;border:1px solid #2c3b55;border-radius:12px;padding:20px}h2{margin:0 0 16px;font-size:21px}h3{margin:22px 0 9px}.muted{color:#aebbd1}.suite-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.suite-grid label{background:#101827;border:1px solid #2c3b55;border-radius:7px;padding:10px;cursor:pointer}.suite-grid b,.model strong{display:block}.suite-grid small,.tag{color:#aebbd1;font-size:12px;margin-left:23px}.models{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.model{background:#101827;border:1px solid #2c3b55;border-radius:7px;padding:10px}.installed{color:#52c78b}.missing{color:#ff9292}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}button{background:#52c78b;border:0;border-radius:6px;padding:10px 14px;font-weight:700;cursor:pointer}.secondary{background:#293956;color:#e6edf7}button:disabled{opacity:.45;cursor:not-allowed}.badge{border-radius:999px;padding:5px 9px;font-size:12px;font-weight:700;background:#293956}.badge-running{background:#4b371e;color:#ffb454}.badge-passed{background:#173b2b;color:#52c78b}.badge-failed{background:#47232b;color:#ff9292}.overview-title{display:flex;justify-content:space-between;align-items:center}.precheck{border-left:3px solid #ffb454;background:#101827;padding:11px;margin:14px 0}.passed{border-color:#52c78b}.failed{border-color:#ff6b6b}.total{background:#101827;border:1px solid #2c3b55;border-radius:8px;padding:12px;margin-bottom:18px}.step{margin:14px 0}.step-head{display:flex;justify-content:space-between;gap:10px;font-size:14px}.step-detail{display:block;color:#aebbd1;font-size:12px;margin:3px 0 6px}.step-done{color:#52c78b}.step-running{color:#ffb454}progress{width:100%;height:9px;accent-color:#52c78b}.step-running progress{accent-color:#ffb454}pre{margin:0;background:#0b1220;padding:12px;overflow:auto;white-space:pre-wrap}.run{border-top:1px solid #2c3b55;padding:12px 0}.run:first-child{border-top:0;padding-top:0}@media(min-width:851px){body{height:100vh;overflow:hidden}.topbar{height:64px;box-sizing:border-box}.page{height:calc(100vh - 64px)}.grid{height:100%}.grid>section{box-sizing:border-box;height:100%;display:flex;flex-direction:column;overflow:hidden}#models,#precheck-steps,#runs{overflow:auto}#models{flex:1;min-height:0;padding-right:5px}#precheck-steps{max-height:45%;padding-right:5px}pre{flex:1;min-height:0}#runs{flex:1;min-height:0}}@media(max-width:850px){.topbar{padding:14px;gap:12px;flex-wrap:wrap}.page{padding:14px}.grid{grid-template-columns:1fr}.models,.suite-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><header class="topbar"><h1>AI Benchmark</h1><nav class="menu"><button class="active" data-page="precheck-page">1. Precheck overview</button><button data-page="benchmark-page">2. Benchmark</button></nav></header>
<main id="precheck-page" class="page active"><div class="grid"><section><h2>Precheck</h2><p class="muted">Selecteer suites en bereid de vereiste onderdelen voor.</p><div class="suite-grid" data-suites></div><h3>Goedgekeurde modellen</h3><p id="model-summary" class="muted">Laden…</p><div id="models" class="models"></div><p>Profiel: <select id="profile"><option value="quick">Quick</option><option value="standard" selected>Standard</option><option value="full">Full</option></select></p><div id="precheck-report" class="precheck">Voer eerst een precheck uit.</div><div class="actions"><button id="precheck">Start precheck</button><button id="stop-precheck" class="secondary" disabled>Stop precheck</button></div><p id="message"></p></section><section><div class="overview-title"><h2>Precheck status</h2><span id="precheck-status" class="badge">Niet uitgevoerd</span></div><div id="precheck-steps" class="muted">Voer een precheck uit om de voortgang te zien.</div><h3>Technical log</h3><pre id="precheck-log">Selecteer of start een precheck om de log te bekijken.</pre></section></div></main>
<main id="benchmark-page" class="page"><div class="grid"><section><h2>Benchmark</h2><p class="muted">De suite-selectie is gelijk aan de precheck-selectie.</p><div class="suite-grid" data-suites></div><p>Profiel: <span id="benchmark-profile">Standard</span></p><div id="benchmark-report" class="precheck">Een geslaagde precheck is vereist voordat de benchmark kan starten.</div><div class="actions"><button id="start" disabled>Start benchmark</button></div><p id="benchmark-message"></p></section><section><div class="overview-title"><h2>Benchmark status</h2><button id="refresh-runs" class="secondary">Vernieuwen</button></div><div id="runs" class="muted">Laden…</div><h3>Technical log</h3><pre id="benchmark-log">Selecteer een benchmark om de log te bekijken.</pre></section></div></main>
<script>
let catalog=[],scan={},installed={},modelSizes={},precheck=null,followPrecheck=true,followBenchmark=true;
const suiteLabels={core:['Algemene taalmodellen','Tekst, kennis en instructies'],'coding-agent':['Code & agents','Programmeren en zelfstandige taken'],rag:['Documenten & RAG','Zoeken en antwoorden uit documenten'],vision:['Beeld & schermen','Afbeeldingen maken'],image:['Afbeeldingen maken','Generatie en bewerking'],speech:['Spraak & audio','Spraakherkenning en synthese'],music:['Muziek maken','Tekst- en melodie-naar-muziek'],web:['Webonderzoek','Zoeken, bronnen en citaten']};
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));async function get(url){let r=await fetch(url);if(!r.ok)throw Error(await r.text());return r.json()}
function selectedSuites(){return [...document.querySelectorAll('#precheck-page input[name="suite"]:checked')].map(x=>x.value)}
function syncSuites(){let selected=new Set(selectedSuites());document.querySelectorAll('[data-suites] input').forEach(x=>x.checked=selected.has(x.value))}
function approvedModels(){let selected=new Set(selectedSuites()),approved=new Map((scan.models||[]).filter(x=>x.allowed).map(x=>[x.model,x]));return catalog.filter(m=>approved.has(m.model)&&m.suites.some(s=>selected.has(s))).map(m=>({...m,scan:approved.get(m.model)})).sort((a,b)=>String(a.model).localeCompare(String(b.model)))}
function renderModels(){let models=approvedModels();document.querySelector('#model-summary').textContent=models.length?`${models.length} goedgekeurde model(len) voor de geselecteerde suites.`:'Geen goedgekeurde modellen voor deze suites.';document.querySelector('#models').innerHTML=models.map(m=>`<div class="model"><strong>${esc(m.model)}</strong><span class="${installed[m.model]?'installed':'missing'}">${installed[m.model]?'Geïnstalleerd':'Niet geïnstalleerd'}</span><span class="tag">RAM ${m.scan.requirements.ram_gib} GiB · VRAM ${m.scan.requirements.vram_gib} GiB</span></div>`).join('')}
function payload(){return {models:approvedModels().map(m=>m.model),suites:selectedSuites(),profile:document.querySelector('#profile').value}}
function statusLabel(status){return {missing:'Niet uitgevoerd',running:'Actief',stopping:'Stoppen',passed:'Geslaagd',failed:'Mislukt',cancelled:'Gestopt'}[status]||status}
function updatePrecheckBadge(){let status=precheck?.status||'missing',b=document.querySelector('#precheck-status');b.textContent=statusLabel(status);b.className='badge '+(status==='passed'?'badge-passed':status==='running'||status==='stopping'?'badge-running':status==='failed'||status==='cancelled'?'badge-failed':'')}
function renderPrecheckSteps(output){let events=new Map;for(let line of output.split('\n')){if(!line.startsWith('@@PRECHECK|'))continue;let [,id,label,status]=line.split('|');events.set(id,{label,status})}let target=document.querySelector('#precheck-steps'),rows=[];function row(label,detail,value,done,substep=''){let percent=Math.max(0,Math.min(100,Math.round(value)));rows.push(`<div class="step ${done?'step-done':'step-running'}"><div class="step-head"><span>${esc(label)}</span><span>${done?'Complete':percent+'%'}</span></div><span class="step-detail">${esc(detail)}</span><progress value="${percent}" max="100"></progress>${substep}</div>`)}let system=events.get('environment'),systemValue=system?.status==='done'?100:0;if(system)row('System readiness','Checking disk space, required tools and the Ollama service.',systemValue,system.status==='done');let suites=(precheck?.suites||[]).map(s=>({id:'suite-'+s,name:(suiteLabels[s]||[s])[0]})),completedSuites=suites.filter(s=>events.get(s.id)?.status==='done').length,activeSuite=suites.find(s=>events.get(s.id)?.status==='running'),suiteValue=suites.length?completedSuites/suites.length*100:0;if(suites.length)row('Preparing benchmark suites',activeSuite?`${completedSuites} of ${suites.length} complete. Now preparing ${activeSuite.name}.`:completedSuites===suites.length?'All selected suites are ready.':`${completedSuites} of ${suites.length} suites prepared.`,suiteValue,completedSuites===suites.length);let models=(precheck?.models||[]).filter(m=>catalog.find(c=>c.model===m)?.backend==='ollama'),modelEvents=models.map(m=>({model:m,event:events.get('model-'+m)})),completedModels=modelEvents.filter(x=>x.event?.status==='done').length,activeModel=modelEvents.find(x=>x.event?.status==='running'),progressMatches=[...output.matchAll(/pulling\s+[^:\s]+:\s+(\d+)%/g)],downloadPercent=progressMatches.length?Number(progressMatches.at(-1)[1]):0,modelValue=models.length?(completedModels+(activeModel?downloadPercent/100:0))/models.length*100:0;if(models.length){let detail=completedModels===models.length?'All selected Ollama models are ready.':`${completedModels} of ${models.length} model downloads complete.`,substep=activeModel?`<div class="substep step-running"><div class="step-head"><span>${esc('Downloading '+activeModel.model)}</span><span>${downloadPercent}%</span></div><progress value="${downloadPercent}" max="100"></progress></div>`:'';row('Downloading Ollama models',detail,modelValue,completedModels===models.length,substep)}if(!rows.length){target.className='muted';target.textContent='Waiting for precheck progress…';return}let parts=[];if(system)parts.push([systemValue,10]);if(suites.length)parts.push([suiteValue,30]);if(models.length)parts.push([modelValue,60]);let total=Math.round(parts.reduce((sum,p)=>sum+p[0]*p[1],0)/parts.reduce((sum,p)=>sum+p[1],0));target.className='';target.innerHTML=`<div class="total"><div class="step-head"><b>Total precheck progress</b><b>${total}%</b></div><progress value="${total}" max="100"></progress></div>`+rows.join('')}
function invalidatePrecheck(){precheck=null;updatePrecheckBadge();document.querySelector('#start').disabled=true;document.querySelector('#precheck-report').className='precheck';document.querySelector('#precheck-report').textContent='Voer eerst een precheck uit.';document.querySelector('#benchmark-report').className='precheck';document.querySelector('#benchmark-report').textContent='Een geslaagde precheck is vereist voordat de benchmark kan starten.'}
async function refreshPrecheck(){if(!precheck)return;let d=await get('/api/precheck/'+precheck.id);precheck=d;let status=d.status,report=document.querySelector('#precheck-report'),bench=document.querySelector('#benchmark-report');report.className='precheck '+(status==='passed'?'passed':status==='failed'||status==='cancelled'?'failed':'');report.textContent=status==='running'?'Precheck is bezig…':status==='stopping'?'Precheck wordt gestopt…':status==='passed'?'Precheck geslaagd. De benchmark kan worden gestart.':status==='cancelled'?'Precheck gestopt.':'Precheck mislukt; bekijk de technische log.';bench.className=report.className;bench.textContent=report.textContent;document.querySelector('#precheck-log').textContent=d.output||'';renderPrecheckSteps(d.raw_output||d.output||'');updatePrecheckBadge();document.querySelector('#precheck').disabled=status==='running'||status==='stopping';document.querySelector('#stop-precheck').disabled=status!=='running'||!d.pid;document.querySelector('#start').disabled=status!=='passed'||!approvedModels().every(m=>installed[m.model]);if(status==='running'||status==='stopping')setTimeout(refreshPrecheck,2500)}
async function restorePrecheck(){let r=await fetch('/api/precheck-status',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload())});if(r.ok){let d=await r.json();if(d.status!=='missing'){precheck=d;refreshPrecheck()}}}
async function runs(){let d=await get('/api/runs'),target=document.querySelector('#runs');target.innerHTML=d.length?d.map(r=>{let p=r.state.progress||{},active=['starting','running','resuming'].includes(r.state.status),current=p.current?`<br><span class="muted">Huidige stap: ${esc(JSON.stringify(p.current))}</span>`:'';return `<div class="run"><b>${esc(r.id)}</b><br><span class="muted">${esc(r.state.status||'unknown')} — ${p.completed||0}/${p.total||0} (${p.percent||0}%)</span>${current}<div class="actions"><button class="secondary" onclick="showLog('${r.id}')">Log</button>${active?`<button class="secondary" onclick="stopRun('${r.id}')">Stop benchmark</button>`:''}</div></div>`}).join(''):'Geen benchmarks gevonden.'}
async function showLog(id){let d=await get('/api/log/'+encodeURIComponent(id));let log=document.querySelector('#benchmark-log');log.textContent=d.log;if(followBenchmark)log.scrollTop=log.scrollHeight}
async function stopRun(id){if(!confirm(`Benchmark ${id} stoppen?`))return;let r=await fetch('/api/stop-run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id})}),d=await r.json();document.querySelector('#benchmark-message').textContent=r.ok?d.message:d.error;runs()}
function saveSelection(){localStorage.setItem('ai-benchmark-selection',JSON.stringify({suites:selectedSuites(),profile:document.querySelector('#profile').value}))}
async function load(){let d=await get('/api/overview');catalog=d.catalog;scan=d.scan||{};installed=d.installed||{};let saved={};try{saved=JSON.parse(localStorage.getItem('ai-benchmark-selection')||'{}')}catch(_){};document.querySelectorAll('[data-suites]').forEach((host,index)=>host.innerHTML=d.suites.map((s,i)=>{let label=suiteLabels[s]||[s,''];let checked=(saved.suites||[d.suites[0]]).includes(s);return `<label><input type="checkbox" name="suite" value="${s}" ${checked?'checked':''}><b>${esc(label[0])}</b><small>${esc(label[1])}</small></label>`}).join(''));document.querySelector('#profile').value=['quick','standard','full'].includes(saved.profile)?saved.profile:'standard';document.querySelector('#benchmark-profile').textContent=document.querySelector('#profile').value;document.querySelectorAll('[data-suites] input').forEach(input=>input.onchange=()=>{syncSuites();invalidatePrecheck();saveSelection();renderModels()});document.querySelector('#profile').onchange=()=>{document.querySelector('#benchmark-profile').textContent=document.querySelector('#profile').value;invalidatePrecheck();saveSelection()};renderModels();await restorePrecheck();runs()}
document.querySelectorAll('.menu button').forEach(button=>button.onclick=()=>{document.querySelectorAll('.menu button,.page').forEach(x=>x.classList.remove('active'));button.classList.add('active');document.querySelector('#'+button.dataset.page).classList.add('active');if(button.dataset.page==='benchmark-page')runs()});
document.querySelector('#precheck').onclick=async()=>{let p=payload();if(!p.models.length||!p.suites.length){document.querySelector('#message').textContent='Selecteer minstens één suite met een goedgekeurd model.';return}let r=await fetch('/api/precheck',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(p)}),d=await r.json();if(!r.ok){document.querySelector('#message').textContent=d.error;return}precheck=d;refreshPrecheck()};
document.querySelector('#stop-precheck').onclick=async()=>{if(precheck){await fetch('/api/precheck-stop',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id:precheck.id})});refreshPrecheck()}};
document.querySelector('#start').onclick=async()=>{let p=payload();if(!precheck||precheck.status!=='passed')return;let r=await fetch('/api/start',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({...p,precheck_id:precheck.id})}),d=await r.json();document.querySelector('#benchmark-message').textContent=r.ok?d.message:d.error;if(r.ok)setTimeout(runs,1500)};
document.querySelector('#refresh-runs').onclick=runs;document.querySelector('#precheck-log').addEventListener('scroll',e=>{let x=e.currentTarget;followPrecheck=x.scrollTop+x.clientHeight>=x.scrollHeight-24});document.querySelector('#benchmark-log').addEventListener('scroll',e=>{let x=e.currentTarget;followBenchmark=x.scrollTop+x.clientHeight>=x.scrollHeight-24});load().catch(e=>document.querySelector('#message').textContent=e.message);setInterval(runs,10000);
</script></html>"""


class App(BaseHTTPRequestHandler):
    server_version = "AIBenchmarkUI/1.0"

    def send_json(self, data: Any, status: int = 200) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/":
            encoded = PAGE.encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded); return
        if self.path == "/api/overview":
            try:
                scan = latest_scan()
            except (OSError, subprocess.SubprocessError) as exc:
                self.send_json({"error": f"Machine scan failed: {exc}"}, 500); return
            self.send_json({"catalog": load_catalog(), "suites": SUITES, "scan": scan, "installed": installed_models(scan), "sizes": installed_model_sizes(scan)}); return
        if self.path == "/api/resources":
            try:
                scan = latest_scan(refresh=True)
            except (OSError, subprocess.SubprocessError) as exc:
                self.send_json({"error": f"Resource refresh failed: {exc}"}, 500); return
            self.send_json({"scan": scan, "installed": installed_models(scan), "sizes": installed_model_sizes(scan)}); return
        if self.path == "/api/runs": self.send_json([run_summary(p) for p in run_directories()[:30]]); return
        if self.path.startswith("/api/precheck/"):
            identifier = Path(self.path.removeprefix("/api/precheck/")).name
            record = json_file(precheck_path(identifier), None)
            if record:
                record = reconcile_precheck(record)
                write_json(precheck_path(identifier), record)
                record["raw_output"] = tail(Path(record["log"]))
                record["output"] = tail(Path(record["log"]), compact_ollama=True)
            self.send_json(record if record else {"error": "Precheck not found"}, 200 if record else 404); return
        if self.path.startswith("/api/log/"):
            identifier = Path(self.path.removeprefix("/api/log/")).name
            match = next((p for p in run_directories() if p.name == identifier), None)
            self.send_json({"log": tail(match / "benchmark.log") if match else "Run not found."}, 200 if match else 404); return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path not in {"/api/start", "/api/precheck", "/api/precheck-status", "/api/precheck-stop", "/api/stop-run"}: self.send_json({"error": "Not found"}, 404); return
        try:
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length))
            if self.path == "/api/stop-run":
                identifier = Path(str(payload.get("id", ""))).name
                run = next((directory for directory in run_directories() if directory.name == identifier), None)
                if not run:
                    raise ValueError("Run not found.")
                (run / "stop.requested").touch()
                self.send_json({"message": f"Graceful stop requested for {identifier}."}, HTTPStatus.ACCEPTED); return
            if self.path == "/api/precheck-stop":
                identifier = Path(str(payload.get("id", ""))).name
                record_path = precheck_path(identifier)
                record = json_file(record_path, {})
                if record.get("status") != "running" or not record.get("pid"):
                    raise ValueError("No active precheck can be stopped.")
                try:
                    os.killpg(int(record["pid"]), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                record["status"] = "stopping"
                record["stop_requested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                write_json(record_path, record)
                self.send_json(record, HTTPStatus.ACCEPTED); return
            models = {str(m) for m in payload["models"]}; suites = [str(s) for s in payload["suites"]]
            if not models or not suites or any(s not in SUITES for s in suites): raise ValueError("Select valid models and suites.")
            known = {str(row.get("model")) for row in load_catalog()}
            if not models <= known: raise ValueError("Unknown model selection.")
            profile = str(payload.get("profile", "standard"))
            if profile not in {"quick", "standard", "full"}: raise ValueError("Invalid profile.")
            identity = json.dumps({"models": sorted(models), "suites": suites, "profile": profile}, separators=(",", ":"))
            identifier = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            if self.path == "/api/precheck-status":
                record = json_file(precheck_path(identifier), {"status": "missing"})
                if record.get("log"):
                    record = reconcile_precheck(record)
                    write_json(precheck_path(identifier), record)
                    record["raw_output"] = tail(Path(record["log"]))
                    record["output"] = tail(Path(record["log"]), compact_ollama=True)
                self.send_json(record); return
            if self.path == "/api/precheck":
                record_path = precheck_path(identifier)
                existing = json_file(record_path, {})
                if existing.get("status") == "running":
                    self.send_json(existing, HTTPStatus.ACCEPTED); return
                config = selected_catalog(models)
                log = results_root() / "ui-prechecks" / f"{identifier}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_bytes(b"")
                record = {"id": identifier, "status": "running", "models": sorted(models), "suites": suites, "profile": profile, "config": str(config), "log": str(log), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                write_json(record_path, record)
                env = {**os.environ, "BENCH_MODELS_CONFIG": str(config)}
                command = [str(ROOT / "scripts/precheck.sh"), ",".join(suites), str(config), "1" if "coding-agent" in suites else "0", profile]
                threading.Thread(target=finish_precheck, args=(record_path, command, env), daemon=True).start()
                self.send_json(record, HTTPStatus.ACCEPTED); return
            identifier = str(payload.get("precheck_id", ""))
            record = json_file(precheck_path(identifier), {})
            if record.get("status") != "passed": raise ValueError("Run the precheck successfully before starting a benchmark.")
            if record.get("models") != sorted(models) or record.get("suites") != suites or record.get("profile") != profile: raise ValueError("The selection changed after the precheck. Run it again.")
            config = Path(record["config"]); log = results_root() / "ui-launch.log"; log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "BENCH_MODELS_CONFIG": str(config)}
            with log.open("ab") as output:
                subprocess.Popen([str(ROOT / "benchmark"), "start", *suites, "--profile", profile], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            self.send_json({"message": "Preflight started. Refresh the Runs list in a moment."}, HTTPStatus.ACCEPTED)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def log_message(self, format: str, *args: Any) -> None: pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the AI Benchmark control panel.")
    parser.add_argument("--host", default="0.0.0.0", help="Network interface to listen on (default: all interfaces)")
    parser.add_argument("--port", default=8080, type=int, help="TCP port (default: 8080)")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), App)
    print(f"AI Benchmark web UI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAI Benchmark web UI stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
