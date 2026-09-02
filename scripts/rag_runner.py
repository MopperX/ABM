from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable
from urllib import request, error

from lib.benchlib import PowerSampler, atomic_json, call_performance_summary, distribution_summary, evaluate_checks, load_json, ollama_chat, response_metrics, utc_now

EMBED_MODEL = "embeddinggemma"
TOP_K = 5
PROFILES={
    "quick":{"tests":["R1","R3","R6"],"repeats":1,"beir_queries":5},
    "standard":{"tests":["R1","R2","R3","R4","R5","R6","R7"],"repeats":3,"beir_queries":20},
    "full":{"tests":["R1","R2","R3","R4","R5","R6","R7"],"repeats":5,"beir_queries":None},
}
SYSTEM=(
    "Answer only from the supplied retrieved sources. Do not use outside knowledge. "
    "If the sources do not establish the answer, say that explicitly. "
    "Cite supporting sources using the exact form [SOURCE:filename]. "
    "If sources conflict, state the conflict instead of silently reconciling it."
)


def rag_fixed_job_count() -> int:
    return 2  # controlled practical retrieval + external BEIR retrieval


def rag_answer_job_count(profile:str) -> int:
    c=PROFILES[profile]
    return len(c["tests"])*int(c["repeats"])


def _post_json(url:str,payload:dict[str,Any]) -> dict[str,Any]:
    body=json.dumps(payload).encode()
    req=request.Request(url,data=body,headers={"Content-Type":"application/json"},method="POST")
    try:
        with request.urlopen(req,timeout=None) as r:
            return json.loads(r.read().decode())
    except error.HTTPError as exc:
        detail=exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama embed HTTP {exc.code}: {detail}") from exc


class RagStopRequested(Exception):
    pass


def _embed(api:str,texts:list[str],model:str=EMBED_MODEL,batch_size:int=32,should_stop=None) -> tuple[list[list[float]],dict[str,Any]]:
    out=[]; calls=[]
    sampler=PowerSampler(); sampler.start(); started=time.monotonic()
    try:
        for i in range(0,len(texts),batch_size):
            if should_stop is not None and should_stop():
                raise RagStopRequested()
            batch=texts[i:i+batch_size]
            t=time.monotonic()
            resp=_post_json(api.rstrip("/")+"/api/embed",{"model":model,"input":batch,"truncate":True})
            calls.append({"items":len(batch),"wall_seconds":time.monotonic()-t,"total_duration_ns":resp.get("total_duration"),"load_duration_ns":resp.get("load_duration"),"prompt_eval_count":resp.get("prompt_eval_count")})
            out.extend(resp.get("embeddings") or [])
    finally:
        elapsed=time.monotonic()-started
        power=sampler.stop(elapsed)
    if len(out)!=len(texts): raise RuntimeError(f"Embedding count mismatch: {len(out)} != {len(texts)}")
    return out,{"model":model,"items":len(texts),"wall_seconds":elapsed,"calls":calls,"power":power}


def _pdf_text(path:Path) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def _load_practical_docs(root:Path) -> list[dict[str,str]]:
    docs=[]
    for p in sorted((root/"benchmarks/rag/fixtures/practical/docs").rglob("*")):
        if not p.is_file(): continue
        if p.suffix.lower() in {".md",".txt"}: text=p.read_text(encoding="utf-8")
        elif p.suffix.lower()==".pdf": text=_pdf_text(p)
        else: continue
        docs.append({"id":p.name,"path":str(p.relative_to(root)),"text":text.strip()})
    return docs


def _dot(a:list[float],b:list[float]) -> float:
    return sum(x*y for x,y in zip(a,b))


def _rank(q:list[float], doc_vecs:list[list[float]], docs:list[dict[str,str]], k:int=TOP_K) -> list[dict[str,Any]]:
    rows=[{"doc_id":d["id"],"score":_dot(q,v),"text":d["text"]} for d,v in zip(docs,doc_vecs)]
    rows.sort(key=lambda x:x["score"],reverse=True)
    return rows[:k]


def _ir_metrics(ranked:list[str], relevant:dict[str,float] | set[str], k:int=TOP_K) -> dict[str,float]:
    if isinstance(relevant,set): rel={x:1.0 for x in relevant}
    else: rel={k:float(v) for k,v in relevant.items() if float(v)>0}
    top=ranked[:k]; hit=[d for d in top if d in rel]
    recall=len(hit)/len(rel) if rel else 0.0
    precision=len(hit)/k if k else 0.0
    rr=0.0
    for i,d in enumerate(top,1):
        if d in rel: rr=1.0/i; break
    dcg=0.0
    for i,d in enumerate(top,1):
        gain=(2**rel.get(d,0.0)-1)
        dcg += gain/math.log2(i+1)
    ideal=sorted(rel.values(),reverse=True)[:k]
    idcg=sum((2**g-1)/math.log2(i+1) for i,g in enumerate(ideal,1))
    return {f"recall@{k}":recall,f"precision@{k}":precision,f"mrr@{k}":rr,f"ndcg@{k}":dcg/idcg if idcg else 0.0}


def _retrieval_for_questions(api:str, docs:list[dict[str,str]], doc_vecs:list[list[float]], questions:list[str],should_stop=None) -> tuple[list[list[dict[str,Any]]],dict[str,Any]]:
    qvecs,embed_meta=_embed(api,questions,should_stop=should_stop)
    return [_rank(q,doc_vecs,docs) for q in qvecs],embed_meta


def prepare_practical_retrieval(*,repo_root:Path,run_dir:Path,profile:str,api:str,is_completed,mark_completed,set_current,should_stop) -> dict[str,Any]:
    key=f"rag|retrieval|practical|{EMBED_MODEL}|{profile}"
    path=run_dir/"raw/rag/retrieval/practical/result.json"
    if is_completed(key) and path.exists(): return load_json(path)
    set_current({"benchmark":"rag","model":EMBED_MODEL,"mode":"retrieval","test":"practical-index","repeat":1,"repeats":1})
    docs=_load_practical_docs(repo_root)
    try:
        doc_vecs,index_meta=_embed(api,[d["text"] for d in docs],should_stop=should_stop)
    except RagStopRequested:
        return {"stopped":True,"type":"retrieval","dataset":"practical"}
    atomic_json(path.parent/"index.json",{"embedding_model":EMBED_MODEL,"docs":docs,"embeddings":doc_vecs})
    gt=load_json(repo_root/"benchmarks/rag/fixtures/practical/ground_truth.json")
    tests=PROFILES[profile]["tests"]
    query_rows=[]; per_test={}
    all_questions=[]; mapping=[]
    for tid in tests:
        for qi,q in enumerate(gt[tid]["questions"]):
            all_questions.append(q); mapping.append((tid,qi,q))
    try:
        rankings,qmeta=_retrieval_for_questions(api,docs,doc_vecs,all_questions,should_stop=should_stop)
    except RagStopRequested:
        return {"stopped":True,"type":"retrieval","dataset":"practical"}
    for (tid,qi,q),ranked in zip(mapping,rankings):
        relevant=set(gt[tid]["relevant"][qi])
        metrics=_ir_metrics([r["doc_id"] for r in ranked],relevant,TOP_K)
        row={"test":tid,"question_index":qi,"question":q,"relevant":sorted(relevant),"ranked":[{"doc_id":r["doc_id"],"score":r["score"]} for r in ranked],"metrics":metrics}
        query_rows.append(row); per_test.setdefault(tid,[]).append(metrics)
    avg={m:statistics.mean(r["metrics"][m] for r in query_rows) for m in [f"recall@{TOP_K}",f"precision@{TOP_K}",f"mrr@{TOP_K}",f"ndcg@{TOP_K}"]}
    result={"type":"retrieval","dataset":"practical","embedding_model":EMBED_MODEL,"profile":profile,"documents":len(docs),"queries":len(query_rows),"index":index_meta,"query_embedding":qmeta,"average":avg,"rows":query_rows,"completed_at":utc_now(),"pass":None}
    atomic_json(path,result); mark_completed(key,result); return result


def _load_scifact(data:Path):
    corpus=[]
    with (data/"corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); corpus.append({"id":str(r["_id"]),"text":((r.get("title") or "")+"\n"+(r.get("text") or "")).strip()})
    queries={}
    with (data/"queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); queries[str(r["_id"])]=r["text"]
    qrels={}
    with (data/"qrels/test.tsv").open(encoding="utf-8") as f:
        reader=csv.DictReader(f,delimiter="\t")
        for r in reader:
            qid=str(r.get("query-id") or r.get("query_id") or r.get("query")); did=str(r.get("corpus-id") or r.get("corpus_id") or r.get("doc_id")); score=float(r.get("score") or 1)
            if score>0: qrels.setdefault(qid,{})[did]=score
    return corpus,queries,qrels


def _select_qids(qrels:dict[str,Any],limit:int|None) -> list[str]:
    ids=sorted(qrels,key=lambda q:hashlib.sha256(("42|"+q).encode()).hexdigest())
    return ids if limit is None else ids[:limit]


def prepare_beir_retrieval(*,run_dir:Path,profile:str,api:str,is_completed,mark_completed,set_current,should_stop) -> dict[str,Any]:
    key=f"rag|retrieval|beir-scifact|{EMBED_MODEL}|{profile}"
    path=run_dir/"raw/rag/retrieval/beir-scifact/result.json"
    if is_completed(key) and path.exists(): return load_json(path)
    set_current({"benchmark":"rag","model":EMBED_MODEL,"mode":"retrieval","test":"BEIR-SciFact","repeat":1,"repeats":1})
    results_root=run_dir.parents[2]; data=results_root/"cache/beir/scifact"
    if not (data/"corpus.jsonl").exists(): raise RuntimeError(f"BEIR SciFact cache is missing: {data}; run preflight again")
    corpus,queries,qrels=_load_scifact(data); qids=_select_qids(qrels,PROFILES[profile]["beir_queries"])
    try:
        doc_vecs,index_meta=_embed(api,[d["text"] for d in corpus],should_stop=should_stop)
        qvecs,qmeta=_embed(api,[queries[q] for q in qids],should_stop=should_stop)
    except RagStopRequested:
        return {"stopped":True,"type":"external-retrieval","benchmark":"BEIR","dataset":"scifact"}
    rows=[]
    for qid,qv in zip(qids,qvecs):
        ranked=_rank(qv,doc_vecs,corpus,k=10); ids=[r["doc_id"] for r in ranked]
        m5=_ir_metrics(ids,qrels[qid],5); m10=_ir_metrics(ids,qrels[qid],10)
        rows.append({"query_id":qid,"query":queries[qid],"ranked":[{"doc_id":r["doc_id"],"score":r["score"]} for r in ranked],"metrics":{**m5,**m10}})
    keys=["recall@5","precision@5","mrr@5","ndcg@5","recall@10","precision@10","mrr@10","ndcg@10"]
    avg={k:statistics.mean(r["metrics"][k] for r in rows) for k in keys}
    sha=(results_root/"cache/beir/scifact.sha256").read_text().strip() if (results_root/"cache/beir/scifact.sha256").exists() else None
    result={"type":"external-retrieval","benchmark":"BEIR","dataset":"scifact","embedding_model":EMBED_MODEL,"profile":profile,"corpus_documents":len(corpus),"queries":len(rows),"selected_query_ids":qids,"dataset_zip_sha256":sha,"index":index_meta,"query_embedding":qmeta,"average":avg,"rows":rows,"completed_at":utc_now(),"pass":None}
    atomic_json(path,result); mark_completed(key,result); return result


def prepare_rag_retrieval(**kwargs) -> dict[str,Any]:
    practical=prepare_practical_retrieval(**kwargs)
    if practical.get("stopped") or kwargs["should_stop"](): return {"stopped":True,"practical":practical}
    beir=prepare_beir_retrieval(run_dir=kwargs["run_dir"],profile=kwargs["profile"],api=kwargs["api"],is_completed=kwargs["is_completed"],mark_completed=kwargs["mark_completed"],set_current=kwargs["set_current"],should_stop=kwargs["should_stop"])
    if beir.get("stopped") or kwargs["should_stop"](): return {"stopped":True,"practical":practical,"beir":beir}
    return {"stopped":False,"practical":practical,"beir":beir}


def _sources_block(ranked:list[dict[str,Any]]) -> str:
    return "\n\n".join(f"[SOURCE:{r['doc_id']}]\n{r['text']}" for r in ranked)


def _cited(answer:str) -> set[str]:
    import re
    return set(re.findall(r"\[SOURCE:([^\]]+)\]",answer))


def _answer_result(answer:str, task:dict[str,Any]) -> tuple[list[dict[str,Any]],bool]:
    checks=evaluate_checks(answer,task["checks"])
    cited=_cited(answer); required=set(task.get("required_sources") or [])
    checks.append({"name":"required source citations","kind":"all","passed":required.issubset(cited),"required":sorted(required),"cited":sorted(cited)})
    return checks,all(bool(c.get("passed")) for c in checks)


def run_rag_answers(*,repo_root:Path,run_dir:Path,model:str,mode:str,profile:str,api:str,temperature:float,seed:int,context:int,is_completed,mark_completed,should_stop,set_current) -> dict[str,Any]:
    gt=load_json(repo_root/"benchmarks/rag/fixtures/practical/ground_truth.json")
    index_path=run_dir/"raw/rag/retrieval/practical/index.json"
    if not index_path.exists(): raise RuntimeError("RAG practical index is missing; retrieval must be prepared first")
    idx=load_json(index_path); docs=idx["docs"]; doc_vecs=idx["embeddings"]
    cfg=PROFILES[profile]; slug=(model+"__"+mode).replace("/","_").replace(":","_")
    base=run_dir/"raw/rag/answers"/slug; rows=[]
    for tid in cfg["tests"]:
        task=gt[tid]
        for rep in range(1,int(cfg["repeats"])+1):
            if should_stop(): return {"stopped":True,"model":model,"mode":mode,"profile":profile,"summary":_summarize(rows)}
            key=f"rag|answer|{model}|{mode}|{tid}|{rep}"; path=base/tid/f"repeat-{rep}.json"
            if is_completed(key) and path.exists(): rows.append(load_json(path)); continue
            set_current({"benchmark":"rag","model":model,"mode":mode,"test":tid,"repeat":rep,"repeats":int(cfg["repeats"])})
            messages=[{"role":"system","content":SYSTEM}]; calls=[]; answers=[]; retrieval=[]
            for qi,q in enumerate(task["questions"]):
                try:
                    qv,qmeta=_embed(api,[q],should_stop=should_stop)
                except RagStopRequested:
                    return {"stopped":True,"model":model,"mode":mode,"profile":profile,"summary":_summarize(rows)}
                ranked=_rank(qv[0],doc_vecs,docs,TOP_K); retrieval.append({"question":q,"ranked":[{"doc_id":r["doc_id"],"score":r["score"]} for r in ranked],"embedding":qmeta})
                user=f"Retrieved sources:\n\n{_sources_block(ranked)}\n\nQuestion: {q}"
                messages.append({"role":"user","content":user})
                payload,response,elapsed,power=ollama_chat(api,model,messages,mode,temperature=temperature,seed=seed,context=context)
                answer=((response.get("message") or {}).get("content") or ""); answers.append(answer)
                calls.append({"request":payload,"response":response,"metrics":response_metrics(response,elapsed),"power":power})
                messages.append({"role":"assistant","content":answer})
            combined="\n".join(answers); checks,passed=_answer_result(combined,task)
            result={"type":"rag-answer","test":tid,"title":task["title"],"repeat":rep,"model":model,"mode":mode,"profile":profile,"completed_at":utc_now(),"pass":passed,"checks_passed":sum(bool(c["passed"]) for c in checks),"checks_total":len(checks),"checks":checks,"questions":task["questions"],"answers":answers,"retrieval":retrieval,"calls":calls}
            atomic_json(path,result); mark_completed(key,result); rows.append(result)
    return {"stopped":False,"model":model,"mode":mode,"profile":profile,"summary":_summarize(rows)}


def _summarize(rows:list[dict[str,Any]]) -> dict[str,Any]:
    if not rows:return {"items":0}
    by={}
    for tid in sorted({r["test"] for r in rows}):
        vals=[r for r in rows if r["test"]==tid]
        by[tid]={"runs":len(vals),"passed":sum(bool(v["pass"]) for v in vals),"pass_rate":sum(bool(v["pass"]) for v in vals)/len(vals)}
    tps=[]; wall=[]
    for r in rows:
        for c in r.get("calls",[]):
            x=c.get("metrics",{}).get("generation_tokens_per_second"); w=c.get("metrics",{}).get("wall_seconds")
            if x is not None:tps.append(x)
            if w is not None:wall.append(w)
    calls=[c for r in rows for c in r.get("calls",[])]
    return {"items":len(rows),"passed":sum(bool(r["pass"]) for r in rows),"pass_rate":sum(bool(r["pass"]) for r in rows)/len(rows),"per_test":by,"performance":{"generation_tps_median":statistics.median(tps) if tps else None,"wall_seconds_median":statistics.median(wall) if wall else None,"calls":call_performance_summary(calls),"repeat_generation_tps":distribution_summary(tps),"repeat_wall_seconds":distribution_summary(wall)}}
