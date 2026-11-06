"""Independent, baseline-screened low-margin cohort. Enrollment precedes intervention."""
import argparse

import numpy as np
import torch

import mlp_kv_common as C
from mlp_functional_sources import prefix_hash
from mlp_kv_materials import CUT, make
from mlp_kv_behavior import score, word_indices
from mlp_kv_mediation import mediate
from mlp_fidelity_v2_collect_v4 import greedy
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease
from tools.power_analysis import check_phase

FOLDER=C.OUT/'targeted'
COUNT=896


def prepare():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer
    FOLDER.mkdir(exist_ok=True)
    prior=C.read(C.OUT/'documents.json')['docs'];selected=C.read(C.OUT/'pilot-behavior.json')['selected_profile']
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);fs=HfFileSystem(token=False)
    seen={r['prefix_sha256'] for r in prior};cohorts=[]
    plan={**C.identity(),'schema':'mlp_kv_targeted_prereg.v1','baseline_pilot_sha256':C.file_sha(C.OUT/'pilot-behavior.json'),
          'earlier_prereg_sha256':C.file_sha(C.OUT/'prereg.json'),'source_sha256':C.file_sha(__file__),
          'profile':selected,'screening_n':COUNT,'enroll_up_to':128,'criteria':'one new case per background, clean first word correct and 0<clean gap<=8; rank independent eligible cases by frozen hash then take first 128. Finish the entire screen before any edit.',
          'effect_test':'Same eight operations, same primary gate and offset as original phase. Qualified population is baseline-correct low-margin only, with baseline screening and all exclusions archived.',
          'power_plan':check_phase(.1,128),'no_retuning':'No treatment outcomes enter screening. If enrollment is smaller than n_required, mark efficacy exploratory.'}
    C.save(FOLDER/'prereg.json',plan)
    for lang in ['en','zh']:
        cache=FOLDER/f'source-{lang}.json'
        if cache.exists():cohorts.append(C.read(cache)['docs']);continue
        old=[r for r in prior if r['language']==lang];d=old[0];start=1+max(r['row_group'] for r in old);docs=[]
        with fs.open(f'datasets/{d["repo"]}@{d["revision"]}/{d["file"]}','rb',block_size=8*1024**2) as f:
            parquet=pq.ParquetFile(f)
            for group in range(start,parquet.num_row_groups):
                for ix,row in enumerate(parquet.read_row_group(group,columns=['id','title','url','text']).to_pylist()):
                    ids=tok.encode(row['text'],add_special_tokens=False)
                    if len(ids)<1040:continue
                    fp=prefix_hash(ids)
                    if fp in seen:continue
                    seen.add(fp);docs.append({**{k:d[k] for k in ['source','language','repo','revision','license','file']},
                       'id':str(row['id']),'url':row['url'],'title':row['title'],'tokens':ids[:1040],'prefix_sha256':fp,'row_group':group,'row':ix})
                print(lang,group,len(docs),flush=True)
                if len(docs)>=COUNT//2:break
        assert len(docs)>=COUNT//2
        docs=sorted(docs,key=lambda r:C.digest([C.SEED,'targeted-doc',r['id']]))[:COUNT//2]
        C.save(cache,{'docs':docs});cohorts.append(docs)
    docs=[d for pair in zip(*cohorts) for d in pair]
    assert len({d['prefix_sha256'] for d in docs})==COUNT
    rows=[]
    for i,d in enumerate(docs):
        b=i+400;t=int(C.digest([C.SEED,'targeted-T',b])[:8],16)%2
        row=make(d,b,selected,t,tok);row['id']='targeted:'+row['id'];row['split']='targeted'
        rows.append(row)
    C.save(FOLDER/'inputs.json',{**C.identity(),'rows':rows,'docs':docs,'prereg_sha256':C.file_sha(FOLDER/'prereg.json')})
    print('screening cases frozen',len(rows),flush=True)


def screen():
    dest=FOLDER/'screen.json'
    if dest.exists():return
    data=C.read(FOLDER/'inputs.json');assert data['prereg_sha256']==C.file_sha(FOLDER/'prereg.json')
    torch.set_num_threads(4);rows=[]
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();words=word_indices(tok)
        for lo in range(0,len(data['rows']),4):
            batch=data['rows'][lo:lo+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
            past=model(input_ids=x[:,:CUT],use_cache=True,logits_to_keep=1).past_key_values
            out=model(input_ids=x[:,CUT:],past_key_values=past,use_cache=True,logits_to_keep=1)
            logits=out.logits[:,-1].float();generated=greedy(model,out.past_key_values,logits,eos,limit=4)
            for j,r in enumerate(batch):
                value=score(r,logits[j],generated[j],tok,words)
                rows.append({'index':lo+j,'id':r['id'],'source_id':r['source_id'],'expected':r['expected'],'clean':value,
                             'eligible':value['correct'] and 0<value['gap']<=8})
            if (lo+len(batch))%128==0:print('screen',lo+len(batch),'/',len(data['rows']),flush=True)
    elig=[r for r in rows if r['eligible']]
    ranked=sorted(elig,key=lambda r:C.digest([C.SEED,'enroll',r['id']]))[:128]
    record={**C.identity(),'schema':'mlp_kv_low_margin_enrollment.v1','inputs_sha256':C.file_sha(FOLDER/'inputs.json'),
            'eligible':len(elig),'enrolled':[r['index'] for r in ranked],'n_screened':len(rows),'rows':rows,'source_sha256':C.file_sha(__file__)}
    C.save(dest,record);print('low margin eligible',len(elig),'enrolled',len(ranked),flush=True)


def intervene():
    dest=FOLDER/'result.json'
    if dest.exists():return
    enrollment=C.read(FOLDER/'screen.json');source=FOLDER/'inputs.json';data=C.read(source)
    assert enrollment['inputs_sha256']==C.file_sha(source)
    indices=enrollment['enrolled'];plan=C.read(FOLDER/'prereg.json')
    if not indices:
        C.save(dest,{**C.identity(),'status':'insufficient','rows':[],'summary':{},'checks':{}});return
    estimate={**C.identity(),'schema':'mlp_kv_screened_execution.v1','enrollment_sha256':C.file_sha(FOLDER/'screen.json'),
              'source_sha256':C.file_sha(__file__),'core_sha256':C.file_sha(C.ROOT/'tests/glm5/mlp_kv_mediation.py'),
              'n_enrolled':len(indices),'power_plan':check_phase(.1,len(indices))}
    C.save(FOLDER/'execution-prereg.json',estimate)
    cohort=[data['rows'][i] for i in indices]
    torch.set_num_threads(4)
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        C.OUT=FOLDER
        result=mediate(model,tok,eos,cohort,{'passed':True},{'power_plan':{'all_cases':plan['power_plan']}})
    baseline=enrollment['rows']
    matches=sum(r['conditions']['clean']['word']==baseline[i]['clean']['word'] and r['conditions']['clean']['gap']==baseline[i]['clean']['gap'] for i,r in zip(indices,result['rows']))
    if matches!=len(indices):
        for summary in result['summary'].values():summary.update(eligible=False,passed=False)
    C.save(dest,{**estimate,'status':'complete','screen_to_replay_exact':matches,**result})
    print('targeted',len(indices),'baseline replay exact',matches,'checks',result['checks'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['prepare','screen','intervene'])
    {'prepare':prepare,'screen':screen,'intervene':intervene}[p.parse_args().part]()
