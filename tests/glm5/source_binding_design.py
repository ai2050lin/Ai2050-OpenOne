"""Source binding v1: independent family material and baseline-first admission."""
from __future__ import annotations
import argparse,hashlib,json,os,re,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from deploy.storage_policy import check_write
OUT=ROOT/'tests/glm5/result/source_binding_v1'
RUN='SOURCE-BINDING-20261110-v1';SEED=202611100393
MODELS={'qwen14':ROOT/'models/hf/Qwen3-14B','glm9':ROOT/'models/hf/glm4-9b-chat-hf'}
WORDS='cat dog fox wolf bear lion bird fish tree leaf rain snow wind fire stone metal wood paper glass water salt bread rice silver'.split()
COUNTS={'discovery':64,'confirm':96,'lockbox_frame':96}
SOURCES=['Orion','Vega']
CODE=['source_binding_design.py','source_binding_runtime.py','source_binding_run.py','source_binding_resources.py']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(8<<20),b''):h.update(a)
    return h.hexdigest()
def rng(key):return np.random.default_rng(np.frombuffer(hashlib.sha256(f'{RUN}/{SEED}/{key}'.encode()).digest(),dtype='<u4'))
def git(*a):return subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',*a],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout.strip()
def save(name,data,model=None,immutable=True):
    p=OUT/name;check_write(p,1<<20,operation='来源绑定产物');p.parent.mkdir(parents=True,exist_ok=True)
    if immutable:assert not p.exists(),f'Immutable artifact exists: {p}'
    modelpath=MODELS.get(model);identity=OUT/f'{model}-identity.json'
    data={**data,'provenance':{'git_sha':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')),
         'model_path':str(modelpath) if modelpath else 'multi-model-controlled-material',
         'model_revision':read(identity)['model_snapshot'] if model and identity.exists() else data.get('model_snapshot','not-a-model-result'),
         'seed':SEED,'tool':RUN,'created_at':datetime.now().astimezone().isoformat()}}
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(data,ensure_ascii=False,indent=1,default=lambda x:x.item()),encoding='utf-8');t.replace(p)
    return p

def log(stage,done,total,start,**kw):
    from source_binding_resources import resources
    elapsed=time.monotonic()-start;row={'time':datetime.now().astimezone().isoformat(),'stage':stage,'done':done,'total':total,
        'elapsed_seconds':elapsed,'eta_seconds':elapsed/done*(total-done) if done else None,**resources(),**kw}
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t=sys.modules['torch'];row.update(cuda_allocated=t.cuda.memory_allocated(),cuda_peak=t.cuda.max_memory_allocated())
    print(json.dumps(row,ensure_ascii=False),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    assert row['ram_available_bytes']>2*2**30 and row['commit_available_bytes']>4*2**30,'RAM/commit resource stop'
    if row.get('shared_gpu_bytes') is not None:assert row['shared_gpu_bytes']<1*2**30,'Unexpected GPU shared residency above 1GiB'
    if 'cuda_allocated' in row:assert row['cuda_allocated']<10*2**30,'CUDA allocation budget exceeded'
    assert elapsed<5400,'90 minute per-process budget'

def behavior(rows,items):
    assert [r['id'] for r in rows]==[r['id'] for r in items],'Missing, duplicate or reordered behavior rows'
    cells={}
    for s in [0,1]:
        for e in [0,1]:
            for o in [0,1]:cells[f'S{s}E{e}O{o}']=[i for i,r in enumerate(rows) if (r['source'],r['entity'],r['order'])==(s,e,o)]
    for f in sorted({r['frame'] for r in rows}):cells[f'frame{f}']=[i for i,r in enumerate(rows) if r['frame']==f]
    stats={k:{'n':len(v),'forced_accuracy':float(np.mean([items[i]['forced_correct'] for i in v])),
                 'top1_accuracy':float(np.mean([items[i]['top1_correct'] for i in v]))} for k,v in cells.items()}
    ok=all(x['n']>=16 and x['forced_accuracy']>=.95 and x['top1_accuracy']>=.90 for x in stats.values())
    return {'passed':ok,'n':len(rows),'cells':stats,'forced_accuracy':float(np.mean([x['forced_correct'] for x in items])),
            'top1_accuracy':float(np.mean([x['top1_correct'] for x in items])),
            'rule':'All declared SxExO and frame cells: n>=16, forced>=.95 and top1>=.90; no entity/row deletion.'}

def checked(model):
    p=read(OUT/'plan.json')
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f'Frozen code modified: {f}'
    assert sha(ROOT/p['grammar_path'])==p['grammar_sha256']
    return p

def admission(model,split):
    p=checked(model);folder=OUT/model
    gatepath=folder/f'{split}-gate.json';behpath=folder/f'{split}-behavior.json'
    assert gatepath.exists() and behpath.exists(),'Complete baseline must exist before any internal collection'
    gate=read(gatepath);beh=read(behpath);mat=read(OUT/f'{model}-materials.json')['splits'][split]
    assert gate['behavior_sha256']==sha(behpath) and gate['materials_sha256']==sha(OUT/f'{model}-materials.json')
    assert gate['model_snapshot']==read(OUT/f'{model}-identity.json')['model_snapshot']
    assert gate['plan_sha256']==sha(OUT/'plan.json') and beh['complete'] and gate['stage']=='behavior_complete'
    recomputed=behavior(mat,beh['rows']);assert recomputed['passed'] and gate['passed'],'Behavior gate failed: internal execution forbidden'
    return {'gate_sha256':sha(gatepath),'behavior_sha256':sha(behpath),'model_snapshot':gate['model_snapshot'],'n':len(mat)}

def sentence(frame,name,value):
    forms=[f'{name} has code {value}.',f'The code for {name} is {value}.',f'{name}: {value}.',f'{name} is assigned the code {value}.',
           f'The entry for {name} gives {value} as its code.',f'The code listed beside {name} is {value}.',
           f'For {name}, the registered code reads {value}.',f'{value} is the code attached to {name}.']
    return forms[frame]

def material(family,split,s,e,o,frame,names,values):
    pieces=[];records={};heads={};offset=0
    def add(text):
        nonlocal offset
        span=(offset,offset+len(text));pieces.append(text);offset+=len(text);return span
    add('Two independent archives assign codes to the same entities. Use only the archive named in the question.\n')
    for src in ([0,1] if o==0 else [1,0]):
        heads[src]=add(f'Archive {SOURCES[src]}:\n')
        entorder=[0,1] if (family+src)%2==0 else [1,0]
        for ent in entorder:
            text=sentence(frame,names[ent],values[src*2+ent]);a,b=add(text+'\n')
            valstart=a+text.index(values[src*2+ent]);records[src*2+ent]={'span':[a,b-1],'value_span':[valstart,valstart+len(values[src*2+ent])],
                'name_span':[a+text.index(names[ent]),a+text.index(names[ent])+len(names[ent])]}
    query_forms=[f'According to archive {SOURCES[s]}, what is the code for {names[e]}?',
                 f'Find the code assigned to {names[e]} in archive {SOURCES[s]}.',
                 f'Which code does archive {SOURCES[s]} list for {names[e]}?',
                 f'Read archive {SOURCES[s]} and give the code belonging to {names[e]}.',
                 f'Use the entry for {names[e]} in archive {SOURCES[s]}. What code does it specify?',
                 f'What code should be returned for {names[e]} when the selected archive is {SOURCES[s]}?',
                 f'The requested archive is {SOURCES[s]} and the requested entity is {names[e]}. Give its code.',
                 f'Retrieve the code associated with {names[e]} from archive {SOURCES[s]}.']
    q=query_forms[frame];qa,qb=add('Question: '+q+'\nReturn only the code word, with no explanation.')
    content=''.join(pieces)
    return {'id':f'{split}-{family:04d}-S{s}E{e}O{o}','family':f'{split}-{family:04d}','frame':frame,'source':s,'entity':e,'order':o,
            'names':names,'values':values,'gold':values[2*s+e],'target_record':2*s+e,'content':content,'records':records,'headers':heads,
            'query_span':[qa,qb],'query_name_span':[qa+10+q.index(names[e]),qa+10+q.index(names[e])+len(names[e])],
            'query_source_span':[qa+10+q.index(SOURCES[s]),qa+10+q.index(SOURCES[s])+len(SOURCES[s])]}

def prepare():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    from shared.dist_protocol.grammar import validate
    check_write(OUT,32*2**30,operation='来源绑定完整阶段32GiB预算');OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'plan.json').exists()
    fillers=[];materialrows={};used=set()
    for split,nf in COUNTS.items():
        rows=[];pool=WORDS[:16] if split=='discovery' else WORDS[16:]
        for family in range(nf):
            names=[];random=rng(f'{split}/{family}')
            while len(names)<2:
                name=''.join(random.choice(list('bdfgklmnprstvz'))+random.choice(list('aeiou')) for _ in range(3)).capitalize()
                if name not in used:used.add(name);names.append(name)
            values=random.choice(pool,4,replace=False).tolist();frame=family%4+(4 if split=='lockbox_frame' else 0)
            fillers.append({'id':f'{split}-{family}','slots':{'N0':names[0],'N1':names[1],**{f'V{i}':v for i,v in enumerate(values)},'TAIL':'Answer:'},'partition':split})
            for s in [0,1]:
                for e in [0,1]:
                    for o in [0,1]:rows.append(material(family,split,s,e,o,frame,names,values))
        materialrows[split]=rows
    # Audit/interoperability grammar exposes the factor product and frozen family partitions.
    frames=[]
    for frame in range(8):
        for s in [0,1]:
            for e in [0,1]:
                for o in [0,1]:
                    r=material(0,'grammar',s,e,o,frame,['{N0}','{N1}'],[f'{{V{i}}}' for i in range(4)])
                    text=r['content']+'\n{TAIL}'
                    frames.append({'id':f'f{frame}-S{s}E{e}O{o}','levels':{'source':str(s),'entity':str(e),'order':str(o),'paraphrase':str(frame)},
                        'text':text,'target':'TAIL','probe':{'prompt':text,'answer':f'{{V{s*2+e}}}','contrast':[f'{{V{i}}}' for i in range(4) if i!=s*2+e]}})
    grammar={'schema_version':'template_grammar.v1','family_id':'SOURCE-BINDING-20261110','grammar_version':1,
        'factors':['source','entity','order','paraphrase'],'frames':frames,'fillers':fillers,'layers':'all','readouts':['final','probe_answer'],
        'splits':{'fractions':{'discovery':.25,'confirm':.375,'lockbox':.375},'heldout_frames':[x['id'] for x in frames if x['id'].startswith(('f4','f5','f6','f7'))]},
        'sampling_note':'Custom frozen runner uses explicit disjoint filler partition, one balanced paraphrase per family, all8 SxExO cells and entity-line-order rotation by family parity. Generic grammar Cartesian expansion is not the executed sample list.',
        'prereg':{'min_behavior_acc':.95,'min_cell_n':16,'min_lockbox_cos':.0}}
    assert not validate(grammar),validate(grammar)
    gp=ROOT/'shared/dist_protocol/grammars/SOURCE-BINDING-20261110.json';assert not gp.exists();check_write(gp,1<<20,operation='冻结来源绑定文法')
    gp.write_text(json.dumps(grammar,ensure_ascii=False,indent=1),encoding='utf-8')
    for model,path in MODELS.items():
        start=time.monotonic();log(model+'-hash',0,1,start)
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True);compiled={}
        for split,rows in materialrows.items():
            compiled[split]=[]
            for r in rows:
                prompt=tok.apply_chat_template([{'role':'user','content':r['content']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:\n'
                encoded=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);off=prompt.index(r['content']);offsets=encoded['offset_mapping']
                def indices(span):
                    a,b=span;out=[i for i,(lo,hi) in enumerate(offsets) if hi>off+a and lo<off+b];assert out;return out
                recs={str(k):{key:indices(v) for key,v in info.items()} for k,info in r['records'].items()}
                positions=[indices(r['headers'][s])[-1] for s in [0,1]]+[indices(r['query_source_span'])[-1],indices(r['query_name_span'])[-1],len(encoded['input_ids'])-1]
                vpos=[recs[str(k)]['value_span'][-1] for k in range(4)]
                assert all(len(recs[str(k)]['value_span'])==1 for k in range(4))
                choiceids=[]
                for val in r['values']:
                    suffix=tok.encode(prompt+val,add_special_tokens=False)
                    assert suffix[:-1]==encoded['input_ids'],'Answer word must be one token with stable prefix'
                    choiceids.append(suffix[-1])
                compiled[split].append({**r,'records':recs,'headers':{str(k):indices(v) for k,v in r['headers'].items()},
                    'query_positions':indices(r['query_span']),'positions':positions,'value_positions':vpos,'prompt':prompt,
                    'tokens':encoded['input_ids'],'choice_ids':choiceids,'gold_id':choiceids[r['target_record']]})
        save(f'{model}-identity.json',{'model_snapshot':snapshot(path),'config':read(path/'config.json')},model)
        save(f'{model}-materials.json',{'splits':compiled,'origin':'New constructed controlled-language prompts; all reported model responses must be real CUDA runs.'},model)
        log(model+'-prepared',1,1,start,max_tokens=max(len(r['tokens']) for rs in compiled.values() for r in rs))
    from source_binding_resources import resources
    plan={'schema':'source_binding_plan.v1','run_id':RUN,'created_at':datetime.now().astimezone().isoformat(),
        'code_sha256':{str((ROOT/'tests/glm5'/f).relative_to(ROOT)):sha(ROOT/'tests/glm5'/f) for f in CODE},
        'grammar_path':str(gp.relative_to(ROOT)),'grammar_sha256':sha(gp),'materials':{m:sha(OUT/f'{m}-materials.json') for m in MODELS},
        'models':{m:read(OUT/f'{m}-identity.json')['model_snapshot'] for m in MODELS},
        'family_counts':COUNTS,'pilot':'first16 discovery families (128 rows); complete pilot gate before discovery internal work',
        'behavior':'All SxExO and frame cells n>=16, forced .95 and vocabulary top1 .90. No exclusion or filtering; full baseline pass precedes every internal collection.',
        'sources':'Two conflicting sources, identical entities; source order and queried entity crossed. New disjoint names in each split; discovery16 vs heldout8 answer words. Frame holdout uses new forms.',
        'native':'All41 raw boundaries at5 positions, separate final norm; whole native coordinates. Discovery full QK/V, head contexts and MLP G/V/A; confirmation only frozen component details plus all hidden boundaries. Internal implementation separately hash-frozen before first internal observation.',
        'head_selection':'Discovery-only; rank original correct-record attention ratio then absolute mass; top4 heads, at most one per block, blocks8..35. No target logits used.',
        'sparse_qk':'32 of128 post-RoPE query coordinates ranked by discovery energy of q_j*k_j across all token positions; freeze. Compare native full dot, same32 random coordinates, and discovery mean head context.',
        'mlp_selection':'From discovery natural full-coordinate MLP response: choose2 blocks with largest across-family unit variance weighted by Down-column norm; top256 units. Predict centered MLP write with their exact native G/V/Down and frozen mean remainder. Compare equal-count random units.',
        'prediction_thresholds':{'head_or_mlp_MSE_improvement_over_discovery_mean':.10,'head_binding_record_accuracy':.65,'qkv_replay_relative_max':.02},
        'causal':'Natural source attention-edge exclusion versus wrong-source matched record; sparse-QK keep versus random coordinates; native MLP-unit suppression versus discovery norm-matched units. Last query position only; no answer vector injection.',
        'glm_replication':'Only after Qwen complete behavior gates, replay calibration and >=1 frozen candidate with >=10% component MSE improvement over mean on BOTH confirm and new-frame panels. GLM uses own discovery candidates; no index transfer.',
        'resources':{'disk_peak_GiB':32,'reserve_GiB':20,'per_process_minutes':90,'campaign_GPU_minutes':240,'models_concurrent':1,'batch':8,'chunk':128,'CUDA_GiB_max':10,'shared_GPU_GiB_stop':1},
        'preflight_resources':resources(),'formal_evidence_promotion':0}
    save('plan.json',plan)
    print('Frozen plan '+str(OUT/'plan.json'),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare']);a=ap.parse_args();prepare()
