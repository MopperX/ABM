#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, os, shutil
from pathlib import Path
from typing import Any

SCREENSPOT_REPO='rootsautomation/ScreenSpot'
MMMU_REPO='MMMU/MMMU_Pro'
PROFILE_LIMITS={'quick':(6,14),'standard':(30,42),'full':(None,None)}

def atom(path:Path,data:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,path)

def stable(rows,key):
    return sorted(rows,key=lambda r:hashlib.sha256(('42|'+str(r.get(key,''))).encode()).hexdigest())

def save_pil(img,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if img.mode not in ('RGB','RGBA'): img=img.convert('RGB')
    img.save(path,format='PNG',optimize=True)

def parse_opts(v):
    if isinstance(v,list): return [str(x) for x in v]
    if isinstance(v,str):
        try:
            x=ast.literal_eval(v); return [str(i) for i in x] if isinstance(x,(list,tuple)) else [v]
        except Exception:return [v]
    return []

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=PROFILE_LIMITS,required=True);ap.add_argument('--cache-root',required=True);a=ap.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    root=Path(a.cache_root)/'vision'; root.mkdir(parents=True,exist_ok=True)
    src_file=root/'sources.json'; api=HfApi()
    if src_file.exists(): sources=json.loads(src_file.read_text())
    else:
        sources={
          'screenspot':{'repo':SCREENSPOT_REPO,'revision':api.dataset_info(SCREENSPOT_REPO).sha},
          'mmmu_pro':{'repo':MMMU_REPO,'revision':api.dataset_info(MMMU_REPO).sha},
        }; atom(src_file,sources)
    s_lim,m_lim=PROFILE_LIMITS[a.profile]
    # ScreenSpot: local HF cache may be ~600 MB; selected fixtures are copied to stable benchmark cache.
    sdir=root/'screenspot'/a.profile; sman=sdir/'manifest.json'
    if not sman.exists():
        ds=load_dataset(SCREENSPOT_REPO,revision=sources['screenspot']['revision'],split='test',cache_dir=str(root/'hf'))
        rows=[]
        for i,r in enumerate(ds): rows.append((i,r))
        ordered=sorted(rows,key=lambda ir:hashlib.sha256((f"42|{ir[1].get('img_filename') or ir[1].get('file_name')}|{ir[1].get('instruction')}").encode()).hexdigest())
        if s_lim is not None: ordered=ordered[:s_lim]
        out=[]
        for j,(idx,r) in enumerate(ordered):
            p=sdir/'images'/f'{j:04d}.png'; save_pil(r['image'],p)
            out.append({'sample_index':idx,'image':str(p),'width':r['image'].width,'height':r['image'].height,'instruction':r['instruction'],'bbox':[float(x) for x in r['bbox']],'data_type':r.get('data_type'),'data_source':r.get('data_source') or r.get('data_souce'),'source_id':r.get('img_filename') or r.get('file_name') or str(idx)})
        atom(sman,{'benchmark':'ScreenSpot','repo':SCREENSPOT_REPO,'revision':sources['screenspot']['revision'],'profile':a.profile,'count':len(out),'samples':out})
    # MMMU-Pro standard 10 options: ~678 MB once in HF cache.
    mdir=root/'mmmu-pro'/a.profile; mman=mdir/'manifest.json'
    if not mman.exists():
        ds=load_dataset(MMMU_REPO,'standard (10 options)',revision=sources['mmmu_pro']['revision'],split='test',cache_dir=str(root/'hf'))
        rows=stable([dict(r) for r in ds],'id')
        if m_lim is not None: rows=rows[:m_lim]
        out=[]
        for j,r in enumerate(rows):
            imgs=[]
            for n in range(1,8):
                im=r.get(f'image_{n}')
                if im is None: continue
                p=mdir/'images'/f'{j:04d}-{n}.png'; save_pil(im,p); imgs.append(str(p))
            out.append({'id':str(r['id']),'question':r['question'],'options':parse_opts(r.get('options')),'answer':str(r['answer']).strip(),'subject':r.get('subject'),'img_type':r.get('img_type'),'images':imgs})
        atom(mman,{'benchmark':'MMMU-Pro','config':'standard (10 options)','repo':MMMU_REPO,'revision':sources['mmmu_pro']['revision'],'profile':a.profile,'count':len(out),'samples':out})
    print(json.dumps({'profile':a.profile,'screenSpot':str(sman),'mmmuPro':str(mman),'sources':sources},indent=2))
if __name__=='__main__':main()
