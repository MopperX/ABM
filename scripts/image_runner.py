from __future__ import annotations
import base64, csv, hashlib, json, os, re, resource, statistics, time
from pathlib import Path
from typing import Any
from urllib import request, error

from lib.benchlib import PowerSampler, atomic_json, distribution_summary, load_json, parse_model_rows, utc_now

PRACTICAL={'quick':['I1','I3','I4'],'standard':['I1','I2','I3','I4','I5','I6','I7'],'full':['I1','I2','I3','I4','I5','I6','I7']}
JUDGE_MODEL='qwen3-vl:4b-instruct'
WIDTH=512; HEIGHT=512


def parse_image_models(path:Path)->list[dict[str,Any]]:
    out=[]
    if not path.exists(): return out
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

def external_counts(run_dir:Path,profile:str)->tuple[int,int]:
    cache=run_dir.parents[2]/'cache'/'image'
    g=cache/'geneval2'/profile/'manifest.json'; h=cache/'hps'/profile/'manifest.json'
    return (int(load_json(g).get('count',0)) if g.exists() else 0,int(load_json(h).get('count',0)) if h.exists() else 0)


def job_count(run_dir:Path,profile:str,image_config:Path)->int:
    g,h=external_counts(run_dir,profile); return len(parse_image_models(image_config))*(len(PRACTICAL[profile])+g+h)


def _b64(path:Path)->str:return base64.b64encode(path.read_bytes()).decode()


def _judge(api:str,image:Path,questions:list[list[str]])->dict[str,Any]:
    prompts=[]
    for i,(q,expected) in enumerate(questions,1): prompts.append(f'{i}. {q}')
    content='Inspect the generated image. Answer every question. Return ONLY a JSON array of answer strings in the same order.\n'+'\n'.join(prompts)
    payload={'model':JUDGE_MODEL,'messages':[{'role':'user','content':content,'images':[_b64(image)]}],'stream':False,'options':{'temperature':0,'seed':42,'num_ctx':4096}}
    req=request.Request(api.rstrip('/')+'/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST');start=time.monotonic()
    try:
        with request.urlopen(req,timeout=None) as r: resp=json.loads(r.read().decode())
    except error.HTTPError as e: raise RuntimeError(f'Vision judge HTTP {e.code}: {e.read().decode(errors="replace")}') from e
    elapsed=time.monotonic()-start; text=((resp.get('message') or {}).get('content') or '').strip()
    try:
        arr=json.loads(text); assert isinstance(arr,list)
    except Exception: arr=[]
    checks=[]
    for i,(q,expected) in enumerate(questions):
        got=str(arr[i]) if i<len(arr) else ''
        passed=_norm_answer(got)==_norm_answer(expected)
        checks.append({'question':q,'expected':expected,'answer':got,'passed':passed})
    return {'model':JUDGE_MODEL,'seconds':elapsed,'raw_answer':text,'format_ok':bool(arr),'checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks),'score':sum(c['passed'] for c in checks)/len(checks) if checks else None}


def _norm_answer(s:str)->str:
    x=re.sub(r'[^a-z0-9]+',' ',str(s).lower()).strip()
    nums={'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
    return nums.get(x,x)


def _geneval_judge(api:str,image:Path,vqa:list[list[str]])->dict[str,Any]:
    return _judge(api,image,vqa)


def _sha(path:Path)->str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _rss_mb()->float:
    v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v/1024.0 if os.uname().sysname!='Darwin' else v/(1024.0*1024.0)


class DiffusersGenerator:
    def __init__(self,model_cfg:dict[str,Any],cache_root:Path):
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub import snapshot_download
        self.torch=torch; self.cfg=model_cfg
        local=snapshot_download(repo_id=model_cfg['model'],revision=model_cfg['revision'],cache_dir=str(cache_root/'hf-models'),local_files_only=True)
        self.resolved_revision=Path(local).name if Path(local).parent.name == 'snapshots' else model_cfg['revision']
        if torch.cuda.is_available(): self.device='cuda'; dtype=torch.float16
        elif getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): self.device='mps'; dtype=torch.float16
        else: self.device='cpu'; dtype=torch.float32
        start=time.monotonic()
        self.pipe=DiffusionPipeline.from_pretrained(local,torch_dtype=dtype,local_files_only=True)
        off=model_cfg.get('offload','auto')
        if self.device=='cuda':
            if off in {'cpu-offload','auto'}:
                try:self.pipe.enable_model_cpu_offload()
                except Exception:self.pipe.to('cuda')
            else:self.pipe.to('cuda')
        elif self.device=='mps':
            self.pipe.to('mps')
            try:self.pipe.enable_attention_slicing()
            except Exception:pass
        else:self.pipe.to('cpu')
        self.load_seconds=time.monotonic()-start

    def generate(self,prompt:str,seed:int,out:Path)->dict[str,Any]:
        t=self.torch
        if self.device=='cuda':
            t.cuda.empty_cache();t.cuda.reset_peak_memory_stats()
        sampler=PowerSampler(interval=0.5);sampler.start();start=time.monotonic();err=None
        try:
            gen=t.Generator(device='cpu').manual_seed(int(seed))
            kwargs={'prompt':prompt,'num_inference_steps':self.cfg['steps'],'guidance_scale':self.cfg['guidance'],'height':HEIGHT,'width':WIDTH,'generator':gen}
            image=self.pipe(**kwargs).images[0]
            out.parent.mkdir(parents=True,exist_ok=True);image.save(out)
        except Exception as e:
            err=f'{type(e).__name__}: {e}'
        elapsed=time.monotonic()-start;power=sampler.stop(elapsed)
        mem={'process_max_rss_mb':_rss_mb()}
        if self.device=='cuda':
            mem.update({'cuda_peak_allocated_mb':t.cuda.max_memory_allocated()/1048576,'cuda_peak_reserved_mb':t.cuda.max_memory_reserved()/1048576})
        elif self.device=='mps':
            try:mem.update({'mps_allocated_mb':t.mps.current_allocated_memory()/1048576,'mps_driver_allocated_mb':t.mps.driver_allocated_memory()/1048576})
            except Exception:pass
        return {'ok':err is None,'error':err,'seconds':elapsed,'images_per_minute':60/elapsed if elapsed else None,'device':self.device,'memory':mem,'power':power,'sha256':_sha(out) if out.exists() else None,'bytes':out.stat().st_size if out.exists() else None}


def _seed(base:int,kind:str,index:int)->int:
    h=hashlib.sha256(f'{kind}:{index}'.encode()).digest();return int(base)+(int.from_bytes(h[:2],'big')%10000)


def run_image(*,repo_root:Path,run_dir:Path,image_config:Path,profile:str,api:str,seed:int,is_completed,mark_completed,should_stop,set_current)->dict[str,Any]:
    cache=run_dir.parents[2]/'cache'/'image'; practical={x['id']:x for x in load_json(repo_root/'benchmarks/image/fixtures/practical/tests.json')['tests']}
    gm=load_json(cache/'geneval2'/profile/'manifest.json'); hm=load_json(cache/'hps'/profile/'manifest.json')
    configs=parse_image_models(image_config); results=[]
    for cfg in configs:
        if should_stop(): break
        slug=(cfg['model']+'__'+cfg['revision'][:12]).replace('/','_').replace(':','_');base=run_dir/'raw/image'/slug
        try:
            set_current({'benchmark':'image','model':cfg['model'],'mode':'image','test':'load-model','repeat':1,'repeats':1})
            generator=DiffusersGenerator(cfg,cache)
        except Exception as e:
            reason=f'{type(e).__name__}: {e}'; _mark_config_unavailable(cfg,profile,practical,gm,hm,reason,base,is_completed,mark_completed)
            results.append({'model':cfg['model'],'revision':cfg['revision'],'status':'unsupported','error':reason});continue
        rows=[]
        for i,tid in enumerate(PRACTICAL[profile]):
            if should_stop(): return {'stopped':True,'configurations':results+[ _summary_cfg(cfg,generator,rows)]}
            key=f'image|practical|{cfg["model"]}|{cfg["revision"]}|{tid}'; path=base/'practical'/f'{tid}.json'; png=base/'practical'/f'{tid}.png'
            if is_completed(key) and path.exists(): rows.append(load_json(path));continue
            set_current({'benchmark':'image','model':cfg['model'],'mode':'image','test':tid,'repeat':1,'repeats':1})
            task=practical[tid]; perf=generator.generate(task['prompt'],_seed(seed,'practical',i),png)
            judge=None
            if perf['ok'] and task.get('judge_questions'):
                try: judge=_judge(api,png,task['judge_questions'])
                except Exception as e: judge={'available':False,'error':f'{type(e).__name__}: {e}'}
            r={'type':'image-practical','test':tid,'title':task['title'],'model':cfg['model'],'model_revision':cfg['revision'],'resolved_revision':generator.resolved_revision,'profile':profile,'prompt':task['prompt'],'seed':_seed(seed,'practical',i),'width':WIDTH,'height':HEIGHT,'steps':cfg['steps'],'guidance':cfg['guidance'],'image':str(png),'completed_at':utc_now(),'status':'ok' if perf['ok'] else 'error','pass':None if judge is None else (judge.get('score')==1.0 if judge.get('score') is not None else None),'generation':perf,'judge':judge,'human_review':task.get('human_review',[])}
            atomic_json(path,r);mark_completed(key,r);rows.append(r)
        # GenEval2 prompt adherence with fixed local VQA judge.
        gene=[]
        for i,s in enumerate(gm['samples']):
            if should_stop():return {'stopped':True,'configurations':results+[_summary_cfg(cfg,generator,rows,gene)]}
            key=f'image|geneval2|{cfg["model"]}|{cfg["revision"]}|{s["source_index"]}';path=base/'external/geneval2'/f'{i:04d}.json';png=base/'external/geneval2'/f'{i:04d}.png'
            if is_completed(key) and path.exists():gene.append(load_json(path));continue
            set_current({'benchmark':'image','model':cfg['model'],'mode':'image','test':f'GenEval2-{i+1}/{len(gm["samples"])}','repeat':1,'repeats':1})
            perf=generator.generate(s['prompt'],_seed(seed,'geneval2',s['source_index']),png);judge=None
            if perf['ok']:
                try:judge=_geneval_judge(api,png,s['vqa_list'])
                except Exception as e:judge={'available':False,'error':f'{type(e).__name__}: {e}','score':None}
            r={'type':'GenEval2-local-VQA','source_index':s['source_index'],'model':cfg['model'],'model_revision':cfg['revision'],'dataset_revision':gm['revision'],'prompt':s['prompt'],'vqa_list':s['vqa_list'],'skills':s.get('skills',[]),'seed':_seed(seed,'geneval2',s['source_index']),'image':str(png),'generation':perf,'judge':judge,'pass':None,'checks_passed':judge.get('passed') if judge else None,'checks_total':judge.get('total') if judge else None}
            atomic_json(path,r);mark_completed(key,r);gene.append(r)
        # HPS benchmark prompts + official HPS v2.1 scorer.
        hpsrows=[];hps_available=bool((hm.get('evaluator') or {}).get('available'))
        scorer=None
        if hps_available:
            try:
                os.environ['HPS_ROOT']=str(cache/'hpsv2');import hpsv2;scorer=hpsv2
            except Exception:hps_available=False
        for i,s in enumerate(hm['samples']):
            if should_stop():return {'stopped':True,'configurations':results+[_summary_cfg(cfg,generator,rows,gene,hpsrows)]}
            key=f'image|hps|{cfg["model"]}|{cfg["revision"]}|{s["id"]}';path=base/'external/hps'/f'{i:04d}-{s["style"]}.json';png=base/'external/hps'/f'{i:04d}-{s["style"]}.png'
            if is_completed(key) and path.exists():hpsrows.append(load_json(path));continue
            set_current({'benchmark':'image','model':cfg['model'],'mode':'image','test':f'HPS-{i+1}/{len(hm["samples"])}','repeat':1,'repeats':1})
            perf=generator.generate(s['prompt'],_seed(seed,'hps',i),png);score=None;eval_s=None;eval_err=None
            if perf['ok'] and hps_available and scorer:
                st=time.monotonic()
                try:
                    val=scorer.score(str(png),s['prompt'],hps_version='v2.1');score=float(val[0] if isinstance(val,(list,tuple)) else val)
                except Exception as e:eval_err=f'{type(e).__name__}: {e}'
                eval_s=time.monotonic()-st
            r={'type':'HPS-v2.1','id':s['id'],'style':s['style'],'model':cfg['model'],'model_revision':cfg['revision'],'prompt':s['prompt'],'seed':_seed(seed,'hps',i),'image':str(png),'generation':perf,'hps_score':score,'hps_evaluation_seconds':eval_s,'hps_error':eval_err,'hps_available':hps_available,'pass':None}
            atomic_json(path,r);mark_completed(key,r);hpsrows.append(r)
        results.append(_summary_cfg(cfg,generator,rows,gene,hpsrows))
        try:
            del generator.pipe
            if generator.device=='cuda': generator.torch.cuda.empty_cache()
        except Exception:pass
    return {'stopped':should_stop(),'profile':profile,'configurations':results,'geneval2_revision':gm['revision'],'hps_evaluator':hm.get('evaluator')}


def _all_keys(cfg,profile,practical,gm,hm):
    for tid in PRACTICAL[profile]:yield f'image|practical|{cfg["model"]}|{cfg["revision"]}|{tid}','practical',tid
    for s in gm['samples']:yield f'image|geneval2|{cfg["model"]}|{cfg["revision"]}|{s["source_index"]}','geneval2',str(s['source_index'])
    for s in hm['samples']:yield f'image|hps|{cfg["model"]}|{cfg["revision"]}|{s["id"]}','hps',s['id']


def _mark_config_unavailable(cfg,profile,practical,gm,hm,reason,base,is_completed,mark_completed):
    for key,kind,ident in _all_keys(cfg,profile,practical,gm,hm):
        path=base/'unsupported'/kind/f'{str(ident).replace("/","_")}.json'
        if is_completed(key): continue
        r={'type':'image','status':'unsupported','model':cfg['model'],'model_revision':cfg['revision'],'reason':reason,'pass':None,'checks_passed':None,'checks_total':None}
        atomic_json(path,r);mark_completed(key,r)


def _summary_cfg(cfg,generator,rows,gene=None,hpsrows=None):
    gene=gene or [];hpsrows=hpsrows or []
    times=[r.get('generation',{}).get('seconds') for r in rows+gene+hpsrows if r.get('generation',{}).get('ok')]
    pows=[r.get('generation',{}).get('power',{}).get('approx_energy_wh') for r in rows+gene+hpsrows if r.get('generation',{}).get('power',{}).get('available')]
    practical_scores=[r.get('judge',{}).get('score') for r in rows if isinstance(r.get('judge'),dict) and r['judge'].get('score') is not None]
    gene_passed=sum((r.get('judge') or {}).get('passed') or 0 for r in gene if isinstance(r.get('judge'),dict))
    gene_total=sum((r.get('judge') or {}).get('total') or 0 for r in gene if isinstance(r.get('judge'),dict))
    hs=[r.get('hps_score') for r in hpsrows if r.get('hps_score') is not None]
    by_style={}
    for style in sorted({r.get('style') for r in hpsrows if r.get('style')}):
        vals=[r.get('hps_score') for r in hpsrows if r.get('style')==style and r.get('hps_score') is not None]
        by_style[style]=statistics.mean(vals) if vals else None
    return {'model':cfg['model'],'revision':cfg['revision'],'resolved_revision':getattr(generator,'resolved_revision',None),'device':getattr(generator,'device',None),'load_seconds':getattr(generator,'load_seconds',None),'status':'completed','practical':{'items':len(rows),'judge_mean':statistics.mean(practical_scores) if practical_scores else None},'geneval2_local_vqa':{'items':len(gene),'atoms_passed':gene_passed,'atoms_total':gene_total,'atom_accuracy':gene_passed/gene_total if gene_total else None},'hps_v2_1':{'items':len(hpsrows),'score_mean':statistics.mean(hs) if hs else None,'score_median':statistics.median(hs) if hs else None,'by_style':by_style},'performance':{'generated_images':len(times),'generation_seconds_median':statistics.median(times) if times else None,'generation_seconds':distribution_summary(times),'load_seconds':distribution_summary([getattr(generator,'load_seconds',None)]),'estimated_gpu_energy_wh':distribution_summary(pows),'images_per_minute_from_median':60/statistics.median(times) if times and statistics.median(times)>0 else None,'total_measured_energy_wh':sum(pows) if pows else None}}
