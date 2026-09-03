#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.benchlib import parse_model_rows

GENEVAL_REV='a6e82d2289e8d418f27f0adee77908b07060eea3'
JUDGE_MODEL='qwen3-vl:4b-instruct'


def atomic(path:Path,data:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def image_config_for(machine_cfg:Path, repo:Path)->Path:
    return machine_cfg


def machine_name(machine_cfg: Path) -> str:
    """Return the machine label from the generated eligible-model config path."""
    # `benchmark` writes this file to <results>/scans/<machine>/eligible.models.tsv.
    # Retain a useful fallback for direct invocations with another config filename.
    return machine_cfg.parent.name if machine_cfg.name == 'eligible.models.tsv' else machine_cfg.stem


def parse_models(path:Path):
    out=[]
    for row in parse_model_rows(path):
        suites={x.strip() for x in row['suites'].split(',') if x.strip()}
        if 'image' not in suites or row['backend'].lower() != 'diffusers':
            continue
        out.append({
            'model':row['model'],
            'revision':row['revision'] or 'main',
            'steps':int(row['steps'] or 20),
            'guidance':float(row['guidance'] or 7.5),
            'offload':row['offload'] or 'auto',
            'notes':row['notes'],
        })
    return out

def even_subset(rows,n):
    if n>=len(rows): return list(rows)
    if n<=1: return [rows[0]]
    idx=[round(i*(len(rows)-1)/(n-1)) for i in range(n)]
    seen=[]
    for i in idx:
        if i not in seen: seen.append(i)
    return [rows[i] for i in seen]


def ensure_judge():
    print(f'Refreshing image evaluator model: {JUDGE_MODEL}',flush=True)
    subprocess.run(['ollama','pull',JUDGE_MODEL],check=True)


def prepare_models(models,cache:Path):
    from huggingface_hub import model_info, snapshot_download
    prepared=[]
    for m in models:
        repo_id=m['model']; rev=m['revision'] or 'main'
        print(f'Image-model cachen: {repo_id}@{rev}',flush=True)
        path=snapshot_download(repo_id=repo_id,revision=rev,cache_dir=str(cache/'hf-models'))
        info=model_info(repo_id,revision=rev)
        prepared.append({**m,'resolved_revision':info.sha,'local_path':path})
    return prepared


def prepare_geneval(profile,cache:Path):
    repo=cache/'geneval2/repo'
    if not (repo/'.git').exists():
        repo.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--quiet','https://github.com/facebookresearch/GenEval2.git',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'fetch','--quiet','origin'],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--quiet',GENEVAL_REV],check=True)
    data=[json.loads(x) for x in (repo/'geneval2_data.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    n={'quick':5,'standard':20,'full':len(data)}[profile]
    sel=even_subset(data,n)
    samples=[]
    for i,d in enumerate(sel):
        samples.append({'id':f'geneval2-{i:04d}','source_index':data.index(d),'prompt':d['prompt'],'atom_count':d.get('atom_count'),'vqa_list':d['vqa_list'],'skills':d.get('skills',[])})
    man={'source':'facebookresearch/GenEval2','revision':GENEVAL_REV,'profile':profile,'count':len(samples),'samples':samples}
    atomic(cache/'geneval2'/profile/'manifest.json',man)


def prepare_hps(profile,cache:Path):
    os.environ['HPS_ROOT']=str(cache/'hpsv2')
    import hpsv2
    allp=hpsv2.benchmark_prompts('all')
    per={'quick':1,'standard':5,'full':None}[profile]
    samples=[]
    for style in sorted(allp):
        prompts=list(allp[style])
        chosen=prompts if per is None else even_subset(prompts,min(per,len(prompts)))
        for i,p in enumerate(chosen): samples.append({'id':f'hps-{style}-{i:04d}','style':style,'prompt':p})
    status={'available':False,'error':None,'hps_version':'v2.1','package_version':getattr(hpsv2,'__version__',None)}
    # Force evaluator checkpoint download before headless detach. Failure does not block generation benchmark.
    try:
        from PIL import Image
        cache.mkdir(parents=True,exist_ok=True)
        dummy=cache/'hpsv2-dummy.png';Image.new('RGB',(224,224),'white').save(dummy)
        score=hpsv2.score(str(dummy),'a plain white square',hps_version='v2.1')
        status['available']=True; status['preflight_score']=float(score[0] if isinstance(score,(list,tuple)) else score)
    except Exception as e:
        status['error']=f'{type(e).__name__}: {e}'
        print('WARNING: HPS v2.1 could not be preloaded; generation remains available.',file=sys.stderr)
    atomic(cache/'hps'/profile/'manifest.json',{'source':'HPSv2 benchmark prompts','profile':profile,'count':len(samples),'samples':samples,'evaluator':status})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['quick','standard','full'],required=True);ap.add_argument('--cache-root',required=True);ap.add_argument('--machine-config',required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];cache=Path(a.cache_root)/'image';cache.mkdir(parents=True,exist_ok=True)
    cfg=image_config_for(Path(a.machine_config),repo);models=parse_models(cfg)
    ensure_judge(); prepared=prepare_models(models,cache);atomic(cache/'models'/f'{machine_name(Path(a.machine_config))}.json',{'image_model_config':str(cfg),'models':prepared})
    prepare_geneval(a.profile,cache);prepare_hps(a.profile,cache)
    print(f'Image preflight complete: {len(prepared)} image model(s), GenEval2 + HPS manifests cached.')
if __name__=='__main__': main()
