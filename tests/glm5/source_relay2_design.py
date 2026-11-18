"""Frozen new-family experiment: relation routing, content prediction, and relay locations."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as B
import source_formation_design as F
ROOT=B.ROOT
OUT=ROOT/'tests/glm5/result/source_relay_v2'
RUN='SOURCE-RELAY-20261110-v2'
SEED=202611100430
MODEL='qwen14'
MODELS=B.MODELS
COUNTS={'discovery':64,'confirm':96,'lockbox_frame':96}
HEADS=[{'block':28,'head':21,'kv_head':4},{'block':29,'head':10,'kv_head':2}]
EARLY=20
read,sha,check_write=B.read,B.sha,B.check_write
gate=F.gate
normalized=F.normalized

def rng(key):return np.random.default_rng(np.frombuffer(hashlib.sha256(f'{RUN}/{SEED}/{key}'.encode()).digest(),dtype='<u4'))

def save(name,data,model=MODEL):
    p=OUT/name;check_write(p,1<<20,operation='来源接力研究产物');p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),p
    identity=OUT/f'{model}-identity.json'
    data={**data,'provenance':{'run':RUN,'seed':SEED,'created_at':datetime.now().astimezone().isoformat(),
        'git_sha':B.git('rev-parse','HEAD'),'git_dirty':True,'model':model,
        'model_snapshot':read(identity)['model_snapshot'] if identity.exists() else None}}
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=1,default=lambda x:x.item()),encoding='utf-8');tmp.replace(p)
    return p

def log(stage,done,total,start,**kw):
    assert not (OUT/'stop-request.json').exists(),'Explicit campaign stop marker present'
    old=B.OUT;B.OUT=OUT
    try:
        check_write(OUT,0,operation='来源接力运行空间复查')
        B.log(stage,done,total,start,**kw)
    finally:B.OUT=old

@contextlib.contextmanager
def gpu_lease(stage):
    from shared.dist_protocol.mechanism_trace import GpuLease
    lease=GpuLease();start=time.monotonic()
    while True:
        try:lease.acquire();break
        except RuntimeError as exc:
            if 'GPU is leased' not in str(exc):raise
            log(stage+'/gpu_wait',0,1,start);time.sleep(30)
    try:yield
    finally:lease.release()

def raw_rows():
    used=set();prior=read(F.OUT/'qwen14-materials.json')
    for rs in prior['splits'].values():
        for r in rs:used.update(r['names']);used.update(r['sources'])
    for rs in read(ROOT/'tests/glm5/result/source_relay_v1/qwen14-materials.json')['splits'].values():
        for r in rs:used.update(r['names']);used.update(r['sources'])
    old=read(B.OUT/'qwen14-materials.json')
    for rs in old['splits'].values():
        for r in rs:used.update(r['names'])
    used.update(B.SOURCES);result={}
    for split,n in COUNTS.items():
        result[split]=[]
        for f in range(n):
            random=rng(f'{split}/{f}')
            def names(count):
                out=[]
                while len(out)<count:
                    word=''.join(random.choice(list('bdfgklmnprstvz'))+random.choice(list('aeiou')) for _ in range(3)).capitalize()
                    if word not in used:used.add(word);out.append(word)
                return out
            entities=names(2);sources=names(3)
            vals=random.choice(B.WORDS[:16] if split=='discovery' else B.WORDS[16:],6,replace=False).tolist()
            orders={'sources':random.permutation(3).tolist(),'entities':[random.permutation(2).tolist() for _ in range(3)]}
            for s in range(3):
                for e in range(2):
                    for o in range(2):
                        r=F.material(split,f,f%4,entities,sources,vals,s,e,o,orders,(f//4)%2,['code','tag'][(f//8)%2])
                        if split=='lockbox_frame':
                            # New query frames not used in the preceding experiment; preserve record spans.
                            q=[f'In ledger {sources[s]}, identify the {r["label"]} linked to the entity {entities[e]}.',
                               f'Consult ledger {sources[s]} for entity {entities[e]}; which {r["label"]} is stored there?',
                               f'For entity {entities[e]}, report the {r["label"]} from the ledger named {sources[s]}.',
                               f'Which value is recorded as the {r["label"]} of {entities[e]} within ledger {sources[s]}?'][f%4]
                            a=r['query_span'][0];r['content']=r['content'][:a]+'Question: '+q+'\nReturn only the one-word value, with no explanation.'
                            r['query_span']=[a,len(r['content'])];r['frame']=8+f%4
                            r['query_source_span']=[a+10+q.index(sources[s]),a+10+q.index(sources[s])+len(sources[s])]
                            r['query_name_span']=[a+10+q.index(entities[e]),a+10+q.index(entities[e])+len(entities[e])]
                        result[split].append(r)
    return result

def groups(r):
    """Disjoint prefix groups. Tokens touching a boundary belong to the earlier group."""
    used=set();result={}
    for name,idx in [('records',[t for j in range(6) for t in r['records'][str(j)]['span']]),
                     ('question',r['question_tokens']),('instruction',r['instruction_tokens']),
                     ('assistant_tail',r['assistant_tail_tokens'])]:
        result[name]=sorted(set(idx)-used-{len(r['tokens'])-1});used.update(result[name])
    result['other']=[t for t in range(len(r['tokens'])-1) if t not in used]
    result['self']=[len(r['tokens'])-1]
    return result

def compile_rows(tok,raw):
    compiled=F.compile_rows(tok,raw)
    for rows in compiled.values():
        for r in rows:
            enc=tok(r['prompt'],add_special_tokens=False,return_offsets_mapping=True);offset=r['prompt'].index(r['content'])
            def indices(a,b):return [i for i,(lo,hi) in enumerate(enc['offset_mapping']) if hi>a and lo<b]
            split=r['content'].index('\nReturn only',r['query_span'][0])
            r['question_tokens']=indices(offset+r['query_span'][0],offset+split)
            r['instruction_tokens']=indices(offset+split,offset+len(r['content']))
            r['assistant_tail_tokens']=indices(offset+len(r['content']),len(r['prompt']))
            r['positions']=[r['positions'][3],r['positions'][4],r['question_tokens'][-1],len(r['tokens'])-1]
            target=r['target_record'];same_source=2*r['source']+1-r['entity'];other_source=2*((r['source']+1)%3)+r['entity']
            r['trace_positions']=[r['records'][str(j)]['value_span'][-1] for j in [target,same_source,other_source]]+[r['question_tokens'][-1],len(r['tokens'])-1]
            r['trace_roles']=['correct_record_value','same_source_other_entity_value','other_source_same_entity_value','question_end','final']
            r['groups']=groups(r)
    return compiled

def shapes(n,L,dim=5120,hd=128):
    return {'hidden':((n,41,4,dim),np.uint16),'early':((n,L,dim),np.uint16),
        'trace_state':((n,11,5,dim),np.uint16),'trace_postattn':((n,10,5,dim),np.uint16),
        'qpre':((n,2,hd),np.uint16),'kpre':((n,2,L,hd),np.uint16),'vproj':((n,2,L,hd),np.uint16),
        'attention':((n,2,L),np.float32),'head_context':((n,2,hd),np.uint16),
        'final_norm':((n,dim),np.uint16)}

def verify():
    p=read(OUT/'plan.json')
    revision=read(OUT/'topology-correction.json') if (OUT/'topology-correction.json').exists() else None
    overrides={}
    if revision:
        assert revision['base_plan_sha256']==sha(OUT/'plan.json')
        overrides=revision['code_overrides']
        assert revision['corrected_heads']==HEADS
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==overrides.get(path,digest),f'Frozen source changed: {path}'
    assert sha(OUT/'qwen14-materials.json')==p['materials_sha256']
    return p

def verify_internal():
    verify();p=read(OUT/'internal-freeze.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,f'Frozen internal source changed: {path}'
    return p

def admission(split):
    p=verify();root=OUT/MODEL;b=read(root/f'{split}-behavior.json');g=read(root/f'{split}-gate.json')
    assert b['complete'] and g['behavior_sha256']==sha(root/f'{split}-behavior.json')
    assert g['plan_sha256']==sha(OUT/'plan.json') and g['model_snapshot']==p['model_snapshot']
    rows=read(OUT/'qwen14-materials.json')['splits'][split]
    assert g['passed'] and gate(rows,b['rows'])['passed'],'Behavior gate failed; no internal measurement allowed'
    return {'gate_sha256':sha(root/f'{split}-gate.json'),'n':len(rows),'model_snapshot':p['model_snapshot']}

def prepare():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    check_write(OUT,22*2**30,operation='来源接力阶段22GiB峰值预算');OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'plan.json').exists();start=time.monotonic();log('prepare',0,1,start)
    diagnostic=ROOT/'tests/glm5/result/source_relay_v1/padding-diagnostic.json'
    assert read(diagnostic)['canonical_test_passed']
    save('measurement-calibration.json',{'source':str(diagnostic.relative_to(ROOT)),'source_sha256':sha(diagnostic),
        'arrays_sha256':read(diagnostic)['arrays_sha256'],'canonical_test_passed':True,
        'change':'Use fixed global width and right padding, gather each real final token. Fresh family names; original prediction recipe and thresholds retained.'})
    tok=AutoTokenizer.from_pretrained(MODELS[MODEL],local_files_only=True);raw=raw_rows();compiled=compile_rows(tok,raw)
    cfg=read(MODELS[MODEL]/'config.json');sizes={}
    for split,rows in compiled.items():
        L=max(len(r['tokens']) for r in rows);sizes[split]=sum(int(np.prod(s))*np.dtype(t).itemsize for s,t in shapes(len(rows),L,cfg['hidden_size'],cfg['head_dim']).values())
    assert sum(sizes.values())<20*2**30,(sizes,'Raw field budget exceeded')
    save('qwen14-identity.json',{'model_snapshot':snapshot(MODELS[MODEL]),'config':cfg})
    save('qwen14-materials.json',{'splits':compiled,'origin':'new constructed families, actual local-model behavior; prior values deliberately reused'})
    import torch,transformers,platform
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.integrations import sdpa_attention
    save('runtime-environment.json',{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
        'numpy':np.__version__,'cuda_build':torch.version.cuda,'modeling_source_sha256':sha(Path(modeling_qwen3.__file__)),
        'sdpa_source_sha256':sha(Path(sdpa_attention.__file__)),'dtype':'original BF16 weights; no quantization; CUDA layer streaming'})
    files=['source_relay2_design.py','source_relay2_behavior.py','source_relay2_runtime.py','source_binding_runtime.py','source_binding_resources.py','source_binding_design.py','source_formation_design.py']
    save('plan.json',{'schema':'source_relay_plan.v2','fixed_length':max(len(r['tokens']) for rs in compiled.values() for r in rs)+8,
        'padding':'Fixed global width, right padding; actual last token gathered by index; padding keys masked. Exact identical-prefix K gate required.',
        'family_counts':COUNTS,'raw_array_bytes':sizes,'model_snapshot':read(OUT/'qwen14-identity.json')['model_snapshot'],
        'calibration_sha256':sha(OUT/'measurement-calibration.json'),'environment_sha256':sha(OUT/'runtime-environment.json'),
        'materials_sha256':sha(OUT/'qwen14-materials.json'),'code_sha256':{str((Path(__file__).parent/f).relative_to(ROOT)):sha(Path(__file__).parent/f) for f in files},
        'hypotheses':[
            'Query-conditioned attention/write loss improves heldout routing over vector-MSE training at the same128 native inputs; low-rank PCA and random inputs control parameter budget.',
            'Earlier-state predicted V retains useful local writing and original-model output, with no true future Q/K/V at inference.',
            'Record-edge suppression can cause wrong-record interference; blocking all records differs from blocking only correct records. Question and assistant-prefix relay groups are distinct.',
            'Native K projections admit a traceable algebraic decomposition over hs20 and subsequent attention/MLP effective residual increments; this is not causal uniqueness.'],
        'behavior':'Every row greedy max4 and EOS; normalizes only whitespace/case/final periods. All SxExO/frame/distance/label cells n>=64, forced+normalized>=.90; pooled>=.95; strict and full-vocabulary first-token scored separately.',
        'heads':HEADS,'early_boundary':EARLY,'measurement_revision':'V1 stopped on discovery numerical prefix dependence. All v2 families freshly generated; v1 partially started confirm excluded. Prediction objectives,400 steps,thresholds andfixedheads unchanged. V1 discovery results are exposed; v2 candidate selection remains discovery-only.',
        'disclosure':'Heads and hs20 chosen from prior phase. All prior panels exposed. New entity/source families, heldout query frames8..11; answer vocabulary and basic record forms previously seen. No claim of novel answer words or world knowledge.',
        'fit_recipe':'discovery only; native/random/PCA128 inputs; q/k/v ridge lambda1. Query-aware model initialized at ridge,400 Adam steps lr.003,batch32,loss KL attention+.1 normalized full-write MSE. No heldout selection. Same fixed two heads; no head search.',
        'primary':'For each of the two fixed heads, attention-native must reduce attention KL and full-write MSE vs ridge-native by>=5% on BOTH panels, and per-family paired write-MSE gain95% interval lower>0. Holm correction across two heads for primary gain permutation tests. Failed claims retained.',
        'causal':'All samples, complete behavioral admission first. From hs20 suppress final-position attention to correct/all/wrong records, question, instruction, assistant tail, all past; original model recomputed. These are total path-blocking effects, not norm-matched semantic selectivity. Forecast arms replace only two head contexts; future true V versus predicted V tracked separately.',
        'replication':'This phase targets Qwen14 and the identified relay ambiguity. GLM extension only after independent Qwen dual-panel forecast gate and sufficient storage; requires its own frozen behavior/material protocol, not automatic reuse of failed GLM panel.',
        'resources':{'disk_peak_GiB':22,'reserve_GiB':20,'models_concurrent':1,'resident_behavior':128,'native_chunk':128,'causal_chunk':64,'forward_batch':8,'GPU_minutes_budget':180},
        'power':'64 discovery,96 families each heldout x12 queries; family bootstrap2000. Operational point gates are not confidence-bound guarantees.',
        'scope':'custom local controlled research, not formal dist_protocol v3 shard; no publication or evidence auto-promotion'})
    log('prepare',1,1,start,raw_GiB=sum(sizes.values())/2**30)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare']);ap.parse_args();prepare()
