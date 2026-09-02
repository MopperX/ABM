from __future__ import annotations

import csv
import itertools
import json
import os
import platform
import re
import statistics
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any


from lib.benchlib import PowerSampler, atomic_json, load_json, utc_now

STT_PRACTICAL={"quick":["S1","S2","S3"],"standard":["S1","S2","S3","S4","S5"],"full":["S1","S2","S3","S4","S5"]}
TTS_TESTS={"quick":["T1","T2","T4"],"standard":["T1","T2","T3","T4"],"full":["T1","T2","T3","T4"]}


def parse_speech_models(path:Path)->list[dict[str,str]]:
    out=[]
    if not path.exists(): return out
    with path.open(encoding='utf-8',newline='') as f:
        rows=(x for x in f if x.strip() and not x.lstrip().startswith('#'))
        r=csv.DictReader(rows,delimiter='\t',fieldnames=['enabled','kind','id','model','language','speaker','notes'])
        for row in r:
            if (row.get('enabled') or '').strip().lower() not in {'1','true','yes','on'}: continue
            out.append({k:(v or '').strip() for k,v in row.items()})
    return out


def _prepared(run_dir:Path)->dict[str,Any]:
    return load_json(run_dir.parents[2]/'cache'/'speech'/'prepared.json')


def job_count(run_dir:Path,profile:str,speech_config:Path)->int:
    configs=parse_speech_models(speech_config)
    prep=_prepared(run_dir)
    fleurs=load_json(Path(prep['fleurs_manifest']))
    stt=sum(1 for x in configs if x['kind']=='stt')
    tts=sum(1 for x in configs if x['kind']=='tts')
    # one fixed diarization job + model-specific ASR/TTS jobs
    return 1 + stt*(len(STT_PRACTICAL[profile])+int(fleurs.get('count',0))) + tts*len(TTS_TESTS[profile])


def _norm(text:str)->str:
    text=unicodedata.normalize('NFKC',text).lower()
    text=re.sub(r"[^\w\s]"," ",text,flags=re.UNICODE)
    return ' '.join(text.split())


def _lev(a:list[str],b:list[str])->int:
    if not a:return len(b)
    if not b:return len(a)
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def error_rates(ref:str,hyp:str)->dict[str,Any]:
    rn=_norm(ref); hn=_norm(hyp); rw=rn.split(); hw=hn.split()
    wd=_lev(rw,hw); cd=_lev(list(rn.replace(' ','')),list(hn.replace(' ','')))
    rc=max(1,len(rn.replace(' ','')))
    return {'reference_normalized':rn,'hypothesis_normalized':hn,'word_errors':wd,'reference_words':len(rw),'wer':wd/max(1,len(rw)),'character_errors':cd,'reference_characters':len(rn.replace(' ','')),'cer':cd/rc}


def _audio_duration(path:Path)->float:
    info=sf.info(path);return float(info.frames)/float(info.samplerate)


def _timing_from_whisper(log:str)->dict[str,float|None]:
    out={}
    pats={
        'load_ms':r'load time\s*=\s*([0-9.]+)\s*ms',
        'mel_ms':r'fallbacks =.*',
        'encode_ms':r'encode time\s*=\s*([0-9.]+)\s*ms',
        'decode_ms':r'decode time\s*=\s*([0-9.]+)\s*ms',
        'total_ms':r'total time\s*=\s*([0-9.]+)\s*ms',
    }
    for k,p in pats.items():
        m=re.search(p,log,re.I);out[k]=float(m.group(1)) if m and m.lastindex else None
    return out


def transcribe(cli:Path,model:Path,wav:Path,language:str,out_prefix:Path)->dict[str,Any]:
    out_prefix.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(cli),'-m',str(model),'-f',str(wav),'-l',language,'-otxt','-of',str(out_prefix)]
    sampler=PowerSampler(interval=.5);sampler.start();start=time.monotonic()
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    elapsed=time.monotonic()-start;power=sampler.stop(elapsed)
    txt=out_prefix.with_suffix('.txt')
    text=txt.read_text(encoding='utf-8',errors='replace').strip() if txt.exists() else ''
    log=(proc.stdout or '')+'\n'+(proc.stderr or '')
    return {'ok':proc.returncode==0,'returncode':proc.returncode,'command':cmd,'text':text,'wall_seconds':elapsed,'audio_seconds':_audio_duration(wav),'real_time_factor':elapsed/max(.001,_audio_duration(wav)),'whisper_timings':_timing_from_whisper(log),'power':power,'stdout':proc.stdout,'stderr':proc.stderr}


def _tts_provider()->str:
    if platform.system()=='Darwin':return 'coreml'
    if subprocess.run(['bash','-lc','command -v nvidia-smi >/dev/null 2>&1'],stdout=subprocess.DEVNULL).returncode==0:return 'cuda'
    return 'cpu'


def _make_tts(asset:dict[str,str],provider:str):
    import sherpa_onnx
    cfg=sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=asset['onnx'],tokens=asset['tokens'],data_dir=asset['data_dir']),
            provider=provider,debug=False,num_threads=max(1,min(os.cpu_count() or 2,8)),
        ),max_num_sentences=1,
    )
    if not cfg.validate():raise RuntimeError(f'Ongeldige TTS-config ({provider})')
    return sherpa_onnx.OfflineTts(cfg)


def load_tts(asset:dict[str,str])->tuple[Any,str,float]:
    preferred=_tts_provider();start=time.monotonic()
    try:
        tts=_make_tts(asset,preferred);used=preferred
    except Exception:
        if preferred=='cpu':raise
        tts=_make_tts(asset,'cpu');used='cpu'
    return tts,used,time.monotonic()-start


def synthesize(tts:Any,text:str,sid:int,out:Path)->dict[str,Any]:
    import sherpa_onnx
    cfg=sherpa_onnx.GenerationConfig();cfg.sid=sid;cfg.speed=1.0;cfg.silence_scale=.2
    sampler=PowerSampler(interval=.5);sampler.start();start=time.monotonic();audio=tts.generate(text,cfg);elapsed=time.monotonic()-start;power=sampler.stop(elapsed)
    if len(audio.samples)==0:raise RuntimeError('TTS genereerde geen audio')
    out.parent.mkdir(parents=True,exist_ok=True);sf.write(out,audio.samples,audio.sample_rate,subtype='PCM_16')
    dur=len(audio.samples)/float(audio.sample_rate)
    return {'seconds':elapsed,'audio_seconds':dur,'real_time_factor':elapsed/max(.001,dur),'generated_audio_seconds_per_second':dur/max(.001,elapsed),'sample_rate':int(audio.sample_rate),'samples':len(audio.samples),'power':power}


def _diarization_score(reference:list[dict[str,Any]],predicted:list[dict[str,Any]],duration:float,step:float=.05)->dict[str,Any]:
    n=max(1,int(np.ceil(duration/step)));ref=np.full(n,-1,dtype=int);pred=np.full(n,-1,dtype=int)
    for s in reference:
        a=max(0,int(s['start']/step));b=min(n,int(np.ceil(s['end']/step)));ref[a:b]=int(s['speaker'])
    labels=sorted(set(int(s['speaker']) for s in predicted))
    for s in predicted:
        a=max(0,int(s['start']/step));b=min(n,int(np.ceil(s['end']/step)));pred[a:b]=int(s['speaker'])
    mappings=[]
    if len(labels)<=2:
        for perm in itertools.permutations([0,1],len(labels)):
            mappings.append(dict(zip(labels,perm)))
    else:
        mappings=[{x:(i%2) for i,x in enumerate(labels)}]
    best=None
    speech=ref>=0
    ref_frames=max(1,int(speech.sum()))
    for mp in mappings or [{}]:
        mapped=np.array([mp.get(int(x),-1) if x>=0 else -1 for x in pred])
        miss=int(((ref>=0)&(mapped<0)).sum());fa=int(((ref<0)&(mapped>=0)).sum());conf=int(((ref>=0)&(mapped>=0)&(ref!=mapped)).sum())
        der=(miss+fa+conf)/ref_frames
        cand={'mapping':mp,'miss_frames':miss,'false_alarm_frames':fa,'confusion_frames':conf,'reference_speech_frames':ref_frames,'der':der}
        if best is None or der<best['der']:best=cand
    return best or {'der':None}


def run_diarization(prep:dict[str,Any],practical:dict[str,Any],run_dir:Path,is_completed,mark_completed,should_stop,set_current)->dict[str,Any]:
    import soundfile as sf
    key='speech|diarization|S4';path=run_dir/'raw/speech/diarization/S4.json'
    if is_completed(key) and path.exists():return load_json(path)
    if should_stop():return {'stopped':True}
    set_current({'benchmark':'speech','model':'sherpa-onnx-diarization','mode':'diarization','test':'S4','repeat':1,'repeats':1})
    import sherpa_onnx
    d=prep['diarization'];cfg=sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=d['segmentation'])),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=d['embedding']),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=2,threshold=.5),min_duration_on=.3,min_duration_off=.5,
    )
    if not cfg.validate():raise RuntimeError('Ongeldige speaker-diarizationconfig')
    sd=sherpa_onnx.OfflineSpeakerDiarization(cfg);wav=Path(practical['entries']['S4']['wav']);audio,sr=sf.read(wav,dtype='float32',always_2d=True);audio=audio[:,0]
    if sr!=sd.sample_rate:
        raise RuntimeError(f'S4 sample rate {sr} != diarization sample rate {sd.sample_rate}')
    sampler=PowerSampler(interval=.5);sampler.start();start=time.monotonic();segments=sd.process(audio).sort_by_start_time();elapsed=time.monotonic()-start;power=sampler.stop(elapsed)
    rows=[{'start':float(x.start),'end':float(x.end),'speaker':int(x.speaker)} for x in segments]
    score=_diarization_score(practical['entries']['S4']['turns'],rows,len(audio)/sr)
    r={'type':'speaker-diarization','test':'S4','engine':'sherpa-onnx','sherpa_version':prep.get('sherpa_onnx_version'),'reference_turns':practical['entries']['S4']['turns'],'predicted_segments':rows,'speaker_count_expected':2,'speaker_count_predicted':len(set(x['speaker'] for x in rows)),'wall_seconds':elapsed,'audio_seconds':len(audio)/sr,'real_time_factor':elapsed/max(.001,len(audio)/sr),'power':power,'score':score,'pass':score.get('der') is not None and score['der']<=.25,'checks_passed':1 if score.get('der') is not None and score['der']<=.25 else 0,'checks_total':1,'completed_at':utc_now()}
    atomic_json(path,r);mark_completed(key,r);return r


def _summary_stt(cfg:dict[str,str],rows:list[dict[str,Any]],external:list[dict[str,Any]])->dict[str,Any]:
    allrows=rows+external;wers=[r['accuracy']['wer'] for r in allrows if r.get('accuracy')];rtf=[r['inference']['real_time_factor'] for r in rows if r.get('inference',{}).get('ok')]
    ext_wer=[r['accuracy']['wer'] for r in external if r.get('accuracy')]
    return {'id':cfg['id'],'model':cfg['model'],'language':cfg['language'],'status':'completed','practical_count':len(rows),'external_count':len(external),'wer_mean_all':statistics.mean(wers) if wers else None,'fleurs_wer_mean':statistics.mean(ext_wer) if ext_wer else None,'practical_rtf_median':statistics.median(rtf) if rtf else None}


def _summary_tts(cfg:dict[str,str],rows:list[dict[str,Any]],provider:str,load_s:float)->dict[str,Any]:
    rtf=[r['generation']['real_time_factor'] for r in rows if r.get('generation')];wers=[r.get('intelligibility',{}).get('wer') for r in rows if r.get('intelligibility',{}).get('wer') is not None]
    return {'id':cfg['id'],'model':cfg['model'],'provider':provider,'load_seconds':load_s,'status':'completed','tests':len(rows),'rtf_median':statistics.median(rtf) if rtf else None,'backtranscription_wer_mean':statistics.mean(wers) if wers else None}


def run_speech(*,repo_root:Path,run_dir:Path,speech_config:Path,profile:str,is_completed,mark_completed,should_stop,set_current)->dict[str,Any]:
    prep=_prepared(run_dir);configs=parse_speech_models(speech_config);practical=load_json(Path(prep['practical_manifest']));fleurs=load_json(Path(prep['fleurs_manifest']))
    fixtures=json.loads((repo_root/'benchmarks/speech/fixtures/practical/tests.json').read_text(encoding='utf-8'));tts_defs={x['id']:x for x in fixtures['tts']}
    cli=Path(prep['whisper_cpp']['cli']);stt_paths={k:Path(v) for k,v in prep['whisper_cpp']['models'].items()};results={'profile':profile,'whisper_cpp':prep['whisper_cpp'],'fleurs_revision':fleurs['revision'],'diarization':None,'stt':[],'tts':[]}
    results['diarization']=run_diarization(prep,practical,run_dir,is_completed,mark_completed,should_stop,set_current)
    if should_stop():return {'stopped':True,**results}

    for cfg in configs:
        if cfg['kind']!='stt':continue
        model_path=stt_paths.get(cfg['model']);rows=[];ext=[];base=run_dir/'raw/speech/stt'/cfg['id']
        if not model_path or not model_path.exists():
            raise RuntimeError(f"Whisper-model ontbreekt uit preflight: {cfg['model']}")
        for tid in STT_PRACTICAL[profile]:
            if should_stop():return {'stopped':True,**results}
            key=f"speech|stt|{cfg['id']}|practical|{tid}";path=base/'practical'/f'{tid}.json'
            if is_completed(key) and path.exists():rows.append(load_json(path));continue
            set_current({'benchmark':'speech','model':cfg['id'],'mode':'stt','test':tid,'repeat':1,'repeats':1})
            item=practical['entries'][tid];prefix=base/'transcripts'/f'{tid}';inf=transcribe(cli,model_path,Path(item['wav']),cfg['language'] or 'nl',prefix);acc=error_rates(item['reference'],inf['text']) if inf['ok'] else None
            r={'type':'speech-stt-practical','test':tid,'model_id':cfg['id'],'model':cfg['model'],'language':cfg['language'],'audio':item,'reference':item['reference'],'transcription':inf['text'],'accuracy':acc,'inference':inf,'pass':None,'checks_passed':None,'checks_total':None,'completed_at':utc_now()}
            atomic_json(path,r);mark_completed(key,r);rows.append(r)
        for i,item in enumerate(fleurs['samples']):
            if should_stop():return {'stopped':True,**results}
            key=f"speech|stt|{cfg['id']}|fleurs|{item['source_index']}";path=base/'external/fleurs'/f'{i:04d}.json'
            if is_completed(key) and path.exists():ext.append(load_json(path));continue
            set_current({'benchmark':'speech','model':cfg['id'],'mode':'stt','test':f'FLEURS-{i+1}/{len(fleurs["samples"])}','repeat':1,'repeats':1})
            prefix=base/'external/fleurs/transcripts'/f'{i:04d}';inf=transcribe(cli,model_path,Path(item['wav']),cfg['language'] or 'nl',prefix);acc=error_rates(item['transcription'],inf['text']) if inf['ok'] else None
            r={'type':'FLEURS-ASR','dataset':'google/fleurs','dataset_revision':fleurs['revision'],'source_index':item['source_index'],'model_id':cfg['id'],'model':cfg['model'],'reference':item['transcription'],'transcription':inf['text'],'accuracy':acc,'inference':inf,'pass':None,'checks_passed':None,'checks_total':None}
            atomic_json(path,r);mark_completed(key,r);ext.append(r)
        results['stt'].append(_summary_stt(cfg,rows,ext))

    evaluator_model=stt_paths['small']
    for cfg in configs:
        if cfg['kind']!='tts':continue
        asset=prep['tts_assets'].get(cfg['model']);base=run_dir/'raw/speech/tts'/cfg['id'];rows=[]
        if not asset:raise RuntimeError(f"TTS asset ontbreekt: {cfg['model']}")
        try:
            set_current({'benchmark':'speech','model':cfg['id'],'mode':'tts','test':'load-model','repeat':1,'repeats':1});tts,provider,load_s=load_tts(asset)
        except Exception as e:
            reason=f'{type(e).__name__}: {e}'
            for tid in TTS_TESTS[profile]:
                key=f"speech|tts|{cfg['id']}|{tid}";p=base/'unsupported'/f'{tid}.json'
                if not is_completed(key):
                    rr={'type':'speech-tts','test':tid,'model_id':cfg['id'],'model':cfg['model'],'status':'unsupported','reason':reason,'pass':None};atomic_json(p,rr);mark_completed(key,rr)
            results['tts'].append({'id':cfg['id'],'model':cfg['model'],'status':'unsupported','error':reason});continue
        for tid in TTS_TESTS[profile]:
            if should_stop():return {'stopped':True,**results}
            key=f"speech|tts|{cfg['id']}|{tid}";path=base/'tests'/f'{tid}.json';wav=base/'audio'/f'{tid}.wav'
            if is_completed(key) and path.exists():rows.append(load_json(path));continue
            set_current({'benchmark':'speech','model':cfg['id'],'mode':'tts','test':tid,'repeat':1,'repeats':1});task=tts_defs[tid]
            try:
                gen=synthesize(tts,task['text'],int(cfg['speaker'] or 0),wav);eval_prefix=base/'backtranscription'/tid;back=transcribe(cli,evaluator_model,wav,'nl',eval_prefix);intel=error_rates(task['text'],back['text']) if back['ok'] else None
                r={'type':'speech-tts','test':tid,'title':task['title'],'model_id':cfg['id'],'model':cfg['model'],'provider':provider,'model_load_seconds':load_s,'text':task['text'],'audio':str(wav),'generation':gen,'backtranscription':back,'intelligibility':intel,'human_review':task.get('human_review',[]),'pass':None,'checks_passed':None,'checks_total':None,'completed_at':utc_now()}
            except Exception as e:
                r={'type':'speech-tts','test':tid,'model_id':cfg['id'],'model':cfg['model'],'provider':provider,'status':'error','error':f'{type(e).__name__}: {e}','pass':None}
            atomic_json(path,r);mark_completed(key,r);rows.append(r)
        results['tts'].append(_summary_tts(cfg,rows,provider,load_s))
    return {'stopped':should_stop(),**results}
