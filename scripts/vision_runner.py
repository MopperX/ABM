from __future__ import annotations
import base64, json, re, statistics, time
from pathlib import Path
from typing import Any, Callable
from urllib import request, error
from lib.benchlib import PowerSampler, atomic_json, evaluate_checks, load_json, mode_to_think, response_metrics, utc_now

REPEATS={'quick':1,'standard':3,'full':3}
PRACTICAL={'quick':['V1','V4','V5','V7'],'standard':['V1','V2','V3','V4','V5','V6','V7'],'full':['V1','V2','V3','V4','V5','V6','V7']}
SYSTEM='Analyseer uitsluitend wat visueel of in de prompt beschikbaar is. Verzin geen onzichtbare details. Volg het gevraagde antwoordformaat exact.'

def _chat(api,model,messages,mode,temperature,seed,context):
    payload={'model':model,'messages':messages,'stream':False,'options':{'temperature':temperature,'seed':seed,'num_ctx':context}}
    inc,val=mode_to_think(mode)
    if inc: payload['think']=val
    body=json.dumps(payload).encode(); req=request.Request(api.rstrip('/')+'/api/chat',data=body,headers={'Content-Type':'application/json'},method='POST')
    sampler=PowerSampler();sampler.start();start=time.monotonic()
    try:
        try:
            with request.urlopen(req,timeout=None) as r: resp=json.loads(r.read().decode())
        except error.HTTPError as e: raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode(errors='replace')}") from e
    finally:
        elapsed=time.monotonic()-start;power=sampler.stop(elapsed)
    return payload,resp,elapsed,power

def _img(path:Path)->str:return base64.b64encode(path.read_bytes()).decode()

def _coord(answer:str):
    try:
        x=json.loads(answer);return float(x['x']),float(x['y']),True
    except Exception:pass
    m=re.search(r'(?i)["\']?x["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)\D+["\']?y["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)',answer)
    if m: return float(m.group(1)),float(m.group(2)),False
    m=re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',answer)
    return (float(m.group(1)),float(m.group(2)),False) if m else (None,None,False)

def _practical_result(answer,task):
    if task.get('coordinate'):
        x,y,strict=_coord(answer);b=task['bbox'];inside=x is not None and b[0]<=x<=b[2] and b[1]<=y<=b[3]
        checks=[{'name':'coordinate format','passed':x is not None and y is not None,'strict_json':strict},{'name':'click inside target bbox','passed':inside,'predicted':[x,y],'bbox':b}]
    else: checks=evaluate_checks(answer,task.get('checks',[]))
    return checks, all(bool(c.get('passed')) for c in checks)

def practical_jobs(repo_root:Path,profile:str)->int:return len(PRACTICAL[profile])*REPEATS[profile]

def external_jobs(run_dir:Path,profile:str)->int:
    root=run_dir.parents[2]/'cache'/'vision'
    total=0
    for n in ['screenspot','mmmu-pro']:
        p=root/n/profile/'manifest.json'
        if p.exists(): total+=int(load_json(p).get('count',0))
    return total

def job_count(repo_root:Path,run_dir:Path,profile:str)->int:return practical_jobs(repo_root,profile)+external_jobs(run_dir,profile)

def run_vision(*,repo_root:Path,run_dir:Path,model:str,mode:str,profile:str,api:str,temperature:float,seed:int,context:int,is_completed,mark_completed,should_stop,set_current)->dict[str,Any]:
    tests={t['id']:t for t in load_json(repo_root/'benchmarks/vision/fixtures/practical/tests.json')['tests']}
    slug=(model+'__'+mode).replace('/','_').replace(':','_');base=run_dir/'raw/vision'/slug;rows=[]
    for tid in PRACTICAL[profile]:
        task=tests[tid]
        for rep in range(1,REPEATS[profile]+1):
            if should_stop():return {'stopped':True,'model':model,'mode':mode,'profile':profile,'practical':_summary(rows)}
            key=f'vision|practical|{model}|{mode}|{tid}|{rep}';path=base/'practical'/tid/f'repeat-{rep}.json'
            if is_completed(key) and path.exists():rows.append(load_json(path));continue
            set_current({'benchmark':'vision','model':model,'mode':mode,'test':tid,'repeat':rep,'repeats':REPEATS[profile]})
            image=repo_root/'benchmarks/vision/fixtures/practical'/task['image']; messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task['prompt'],'images':[_img(image)]}]
            payload,resp,elapsed,power=_chat(api,model,messages,mode,temperature,seed,context);answer=((resp.get('message') or {}).get('content') or '')
            checks,passed=_practical_result(answer,task);r={'type':'vision-practical','test':tid,'title':task['title'],'repeat':rep,'model':model,'mode':mode,'profile':profile,'image':str(image.relative_to(repo_root)),'completed_at':utc_now(),'pass':passed,'checks_passed':sum(bool(c.get('passed')) for c in checks),'checks_total':len(checks),'checks':checks,'answer':answer,'request':payload,'response':resp,'metrics':response_metrics(resp,elapsed),'power':power}
            atomic_json(path,r);mark_completed(key,r);rows.append(r)
    ext=_run_external(run_dir=run_dir,model=model,mode=mode,profile=profile,api=api,temperature=temperature,seed=seed,context=context,is_completed=is_completed,mark_completed=mark_completed,should_stop=should_stop,set_current=set_current,base=base)
    return {'stopped':bool(ext.get('stopped')),'model':model,'mode':mode,'profile':profile,'practical':_summary(rows),'external':ext}

def _run_external(*,run_dir,model,mode,profile,api,temperature,seed,context,is_completed,mark_completed,should_stop,set_current,base):
    cache=run_dir.parents[2]/'cache'/'vision';out={}
    # ScreenSpot click accuracy
    man=load_json(cache/'screenspot'/profile/'manifest.json'); ss=[]
    for i,s in enumerate(man['samples']):
        if should_stop():return {'stopped':True,'screenspot':_ext_summary(ss)}
        sid=str(i);key=f'vision|screenspot|{model}|{mode}|{profile}|{sid}';path=base/'external/screenspot'/f'{i:04d}.json'
        if is_completed(key) and path.exists():ss.append(load_json(path));continue
        set_current({'benchmark':'vision','model':model,'mode':mode,'test':f'ScreenSpot-{i+1}/{len(man["samples"])}','repeat':1,'repeats':1})
        prompt='Locate the GUI element for this instruction: '+s['instruction']+'\nReturn only JSON: {"x": integer_pixel_x, "y": integer_pixel_y} using the original image pixel coordinates.'
        messages=[{'role':'user','content':prompt,'images':[_img(Path(s['image']))]}];payload,resp,elapsed,power=_chat(api,model,messages,mode,temperature,seed,context);ans=((resp.get('message') or {}).get('content') or '')
        x,y,strict=_coord(ans); bbox=s['bbox'];w=float(s['width']);h=float(s['height']);
        # ScreenSpot variants use normalized boxes; tolerate absolute boxes as well.
        if max(bbox)<=1.5: px,py=(x/w if x is not None else None),(y/h if y is not None else None)
        else: px,py=x,y
        inside=px is not None and bbox[0]<=px<=bbox[2] and bbox[1]<=py<=bbox[3]
        r={'type':'ScreenSpot','index':i,'source_id':s['source_id'],'instruction':s['instruction'],'bbox':bbox,'image_size':[s['width'],s['height']],'predicted_pixel':[x,y],'predicted_comparison':[px,py],'format_ok':x is not None and y is not None,'strict_json':strict,'pass':inside,'checks_passed':1 if inside else 0,'checks_total':1,'request':payload,'response':resp,'metrics':response_metrics(resp,elapsed),'power':power,'dataset_revision':man['revision']}
        atomic_json(path,r);mark_completed(key,r);ss.append(r)
    out['screenspot']={**_ext_summary(ss),'revision':man['revision']}
    # MMMU-Pro accuracy
    man=load_json(cache/'mmmu-pro'/profile/'manifest.json'); mm=[]
    for i,s in enumerate(man['samples']):
        if should_stop():return {'stopped':True,**out,'mmmu_pro':_ext_summary(mm)}
        key=f'vision|mmmu-pro|{model}|{mode}|{profile}|{s["id"]}';path=base/'external/mmmu-pro'/f'{i:04d}-{s["id"].replace("/","_")}.json'
        if is_completed(key) and path.exists():mm.append(load_json(path));continue
        set_current({'benchmark':'vision','model':model,'mode':mode,'test':f'MMMU-Pro-{i+1}/{len(man["samples"])}','repeat':1,'repeats':1})
        opts='\n'.join(f'{chr(65+j)}. {o}' for j,o in enumerate(s['options']))
        prompt=f"Question: {s['question']}\nOptions:\n{opts}\nReturn only the single option letter A-J."
        images=[_img(Path(p)) for p in s['images']];messages=[{'role':'user','content':prompt,'images':images}]
        payload,resp,elapsed,power=_chat(api,model,messages,mode,temperature,seed,context);ans=((resp.get('message') or {}).get('content') or '').strip();m=re.search(r'\b([A-J])\b',ans.upper());pred=m.group(1) if m else None;gold=s['answer'].strip().upper();passed=pred==gold
        r={'type':'MMMU-Pro','id':s['id'],'subject':s.get('subject'),'gold':gold,'predicted':pred,'answer_text':ans,'pass':passed,'checks_passed':1 if passed else 0,'checks_total':1,'request':payload,'response':resp,'metrics':response_metrics(resp,elapsed),'power':power,'dataset_revision':man['revision']}
        atomic_json(path,r);mark_completed(key,r);mm.append(r)
    out['mmmu_pro']={**_ext_summary(mm),'revision':man['revision']};out['stopped']=False;return out

def _ext_summary(rows):
    if not rows:return {'items':0,'passed':0,'accuracy':None}
    p=sum(bool(x.get('pass')) for x in rows);return {'items':len(rows),'passed':p,'accuracy':p/len(rows)}

def _summary(rows):
    if not rows:return {'items':0}
    p=sum(bool(r.get('pass')) for r in rows);tps=[r['metrics'].get('generation_tokens_per_second') for r in rows if r.get('metrics',{}).get('generation_tokens_per_second') is not None]
    return {'items':len(rows),'passed':p,'pass_rate':p/len(rows),'generation_tps_median':statistics.median(tps) if tps else None}
