"""Fresh source/entity families, complete behavioral admission and frozen provenance."""
from __future__ import annotations
import argparse,copy,hashlib,json,re,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as B
ROOT=B.ROOT
OUT=ROOT/'tests/glm5/result/source_formation_v1'
RUN='SOURCE-FORMATION-20261110-v1'
SEED=202611100394
MODELS=B.MODELS
COUNTS={'discovery':64,'confirm':96,'lockbox_frame':96}
BLOCKS=[27,28,29,30]
EARLY=20
read=B.read
sha=B.sha
check_write=B.check_write
def rng(key):return np.random.default_rng(np.frombuffer(hashlib.sha256(f'{RUN}/{SEED}/{key}'.encode()).digest(),dtype='<u4'))
def save(name,data,model=None):
    p=OUT/name;check_write(p,1<<20,operation='来源形成产物');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
    data={**data,'provenance':{'run':RUN,'seed':SEED,'created_at':datetime.now().astimezone().isoformat(),
        'git_sha':B.git('rev-parse','HEAD'),'git_dirty':True,'model':model,
        'model_snapshot':read(OUT/f'{model}-identity.json')['model_snapshot'] if model else None}}
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=1,default=lambda x:x.item()),encoding='utf-8');tmp.replace(p)
    return p
def log(stage,done,total,start,**kw):
    old=B.OUT;B.OUT=OUT
    try:B.log(stage,done,total,start,**kw)
    finally:B.OUT=old
def normalized(text):return text.strip().rstrip('.').strip().casefold()

def material(split,fam,frame,names,sources,values,s,e,o,orders,distant,label):
    pieces=[];records={};headers={};cursor=0
    def add(text):
        nonlocal cursor
        a=cursor;cursor+=len(text);pieces.append(text);return [a,cursor]
    add(f'Three independent ledgers assign {label}s. An entry is identified by BOTH its ledger and its entity. Use the ledger requested in the question.\n')
    seq=orders['sources'][::1 if o==0 else -1]
    for src in seq:
        headers[src]=add(f'Ledger {sources[src]}:\n')
        for ent in orders['entities'][src]:
            name,value=names[ent],values[2*src+ent]
            forms=[f'{name} has {label} {value}.',f'The {label} for {name} is {value}.',
                f'{name}: {value}.',f'{name} is assigned the {label} {value}.',
                f'The entry for {name} gives {value} as its {label}.',f'For {name}, the recorded {label} is {value}.',
                f'{value} is the {label} attached to {name}.',f'The {label} registered beside {name} reads {value}.']
            line=forms[frame];a,b=add(line+'\n')
            records[2*src+ent]={'span':[a,b-1],'value_span':[a+line.index(value),a+line.index(value)+len(value)],
                'name_span':[a+line.index(name),a+line.index(name)+len(name)]}
    if distant:add('Note: these ledgers are separate records. Their entries may disagree. No ledger overrides another ledger, and the order of presentation does not establish priority.\n')
    queries=[f'According to ledger {sources[s]}, what is the {label} for {names[e]}?',
        f'Find the {label} assigned to {names[e]} in ledger {sources[s]}.',
        f'Which {label} does ledger {sources[s]} list for {names[e]}?',
        f'Read ledger {sources[s]} and give the {label} belonging to {names[e]}.',
        f'Use the entry for {names[e]} in ledger {sources[s]}. What {label} does it specify?',
        f'What {label} should be returned for {names[e]} when the selected ledger is {sources[s]}?',
        f'The requested ledger is {sources[s]} and the requested entity is {names[e]}. Give its {label}.',
        f'Retrieve the {label} associated with {names[e]} from ledger {sources[s]}.']
    q=queries[frame];a,b=add('Question: '+q+'\nReturn only the one-word value, with no explanation.')
    return {'id':f'{split}-{fam:04d}-S{s}E{e}O{o}','family':f'{split}-{fam:04d}','frame':frame,'source':s,'entity':e,'order':o,
        'distant':distant,'label':label,'names':names,'sources':sources,'values':values,'gold':values[2*s+e],'target_record':2*s+e,
        'content':''.join(pieces),'records':records,'headers':headers,'query_span':[a,b],
        'query_source_span':[a+10+q.index(sources[s]),a+10+q.index(sources[s])+len(sources[s])],
        'query_name_span':[a+10+q.index(names[e]),a+10+q.index(names[e])+len(names[e])]}

def source_rows():
    used=set()
    old=read(B.OUT/'qwen14-materials.json')
    for rs in old['splits'].values():
        for r in rs:used.update(r['names']);used.update(B.SOURCES)
    result={}
    for split,n in COUNTS.items():
        rows=[]
        for f in range(n):
            random=rng(f'{split}/{f}')
            def names(count):
                out=[]
                while len(out)<count:
                    v=''.join(random.choice(list('bdfgklmnprstvz'))+random.choice(list('aeiou')) for _ in range(3)).capitalize()
                    if v not in used:used.add(v);out.append(v)
                return out
            entities=names(2);sources=names(3);pool=B.WORDS[:16] if split=='discovery' else B.WORDS[16:]
            values=random.choice(pool,6,replace=False).tolist();frame=f%4+(4 if split=='lockbox_frame' else 0)
            # Every frame has both distances and both relation labels; orders have independent RNG.
            distant=(f//4)%2;label=['code','tag'][(f//8)%2]
            orders={'sources':random.permutation(3).tolist(),'entities':[random.permutation(2).tolist() for _ in range(3)]}
            for s in range(3):
                for e in range(2):
                    for o in range(2):rows.append(material(split,f,frame,entities,sources,values,s,e,o,orders,distant,label))
        result[split]=rows
    return result

def compile_rows(tok,raw):
    result={}
    for split,rows in raw.items():
        result[split]=[]
        for r in rows:
            prompt=tok.apply_chat_template([{'role':'user','content':r['content']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:\n'
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);off=prompt.index(r['content'])
            def ix(span):
                a,b=span;v=[i for i,(lo,hi) in enumerate(enc['offset_mapping']) if hi>off+a and lo<off+b];assert v;return v
            records={str(k):{key:ix(v) for key,v in info.items()} for k,info in r['records'].items()}
            positions=[ix(r['headers'][s])[-1] for s in range(3)]+[ix(r['query_source_span'])[-1],ix(r['query_name_span'])[-1],len(enc['input_ids'])-1]
            choices=[]
            for val in r['values']:
                ids=tok.encode(prompt+val,add_special_tokens=False);assert ids[:-1]==enc['input_ids'];choices.append(ids[-1])
            result[split].append({**r,'prompt':prompt,'tokens':enc['input_ids'],'records':records,'positions':positions,
                'headers':{str(k):ix(v) for k,v in r['headers'].items()},'query_positions':ix(r['query_span']),
                'choice_ids':choices,'gold_id':choices[r['target_record']]})
    return result

def gate(rows,items):
    assert [r['id'] for r in rows]==[r['id'] for r in items],'Rows must be complete and in frozen order'
    cells={f'S{s}E{e}O{o}':[i for i,r in enumerate(rows) if (r['source'],r['entity'],r['order'])==(s,e,o)]
        for s in range(3) for e in range(2) for o in range(2)}
    for key in ['frame','distant','label']:
        for value in sorted(set(r[key] for r in rows)):cells[f'{key}={value}']=[i for i,r in enumerate(rows) if r[key]==value]
    def stat(ii):
        return {'n':len(ii),'forced_accuracy':float(np.mean([items[i]['forced_correct'] for i in ii])),
            'normalized_generation_accuracy':float(np.mean([items[i]['normalized_correct'] and items[i]['terminated'] for i in ii])),
            'strict_generation_accuracy':float(np.mean([items[i]['strict_correct'] and items[i]['terminated'] for i in ii])),
            'top1_accuracy':float(np.mean([items[i]['top1_correct'] for i in ii]))}
    stats={k:stat(v) for k,v in cells.items()};allstat=stat(list(range(len(rows))))
    passed=all(x['n']>=64 and x['forced_accuracy']>=.90 and x['normalized_generation_accuracy']>=.90 for x in stats.values())
    passed &= allstat['forced_accuracy']>=.95 and allstat['normalized_generation_accuracy']>=.95
    return {'passed':bool(passed),'cells':stats,**allstat}

def verify():
    p=read(OUT/'plan.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,f'Frozen code changed: {path}'
    return p
def admission(model,split):
    p=verify();root=OUT/model;b=read(root/f'{split}-behavior.json');g=read(root/f'{split}-gate.json')
    assert sha(OUT/f'{model}-materials.json')==p['materials'][model]
    assert b['complete'] and g['behavior_sha256']==sha(root/f'{split}-behavior.json')
    assert g['plan_sha256']==sha(OUT/'plan.json') and g['model_snapshot']==p['models'][model]
    rows=read(OUT/f'{model}-materials.json')['splits'][split]
    assert g['passed'] and gate(rows,b['rows'])['passed'],'Full behavioral admission failed'
    return {'gate_sha256':sha(root/f'{split}-gate.json'),'n':len(rows),'model_snapshot':p['models'][model]}

def prepare():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    check_write(OUT,56*2**30,operation='来源形成阶段56GiB总预算');OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'plan.json').exists()
    raw=source_rows()
    # This explicit family grammar extends the existing protocol vocabulary; the executed lists are canonical.
    grammar={'schema':'source_formation_family.v1','run':RUN,'source_count':3,'entity_count':2,'source_query':[0,1,2],
        'entity_query':[0,1],'reversal':[0,1],'families':COUNTS,'source':str(Path(__file__).relative_to(ROOT)),
        'heldout_frames':[4,5,6,7],'split_rule':'disjoint source and entity name families; discovery16 vs heldout8 value words',
        'scope':'custom local experiment, not a dist_protocol v3 audited shard'}
    save('family-grammar.json',grammar)
    for model,path in MODELS.items():
        start=time.monotonic();log(model+'/prepare',0,1,start)
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True);compiled=compile_rows(tok,raw)
        save(f'{model}-identity.json',{'model_snapshot':snapshot(path),'config':read(path/'config.json')})
        save(f'{model}-materials.json',{'splits':compiled,'origin':'constructed controlled language, real-model responses only'},model)
        log(model+'/prepare',1,1,start,max_length=max(len(r['tokens']) for rs in compiled.values() for r in rs))
    paths=[Path(__file__),Path(__file__).with_name('source_formation_behavior.py'),Path(__file__).with_name('source_binding_runtime.py'),
        Path(__file__).with_name('source_binding_resources.py'),Path(__file__).with_name('source_binding_design.py')]
    save('plan.json',{'schema':'source_formation_plan.v1','run':RUN,'family_counts':COUNTS,'start_boundary':EARLY,
        'candidate_blocks':BLOCKS,'models':{m:read(OUT/f'{m}-identity.json')['model_snapshot'] for m in MODELS},
        'materials':{m:sha(OUT/f'{m}-materials.json') for m in MODELS},
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'behavior':'All rows greedy max4 tokens, terminated normalized one-word answer; normalization only whitespace/case/terminal periods. Every SxExO and frame/distance/label cell n>=64 and forced+normalized>=.90; pooled both>=.95. Strict exact and first-token retained. No filtering.',
        'power':'64 discovery and96 heldout families per factor cell; binomial worst SE .0625/.0510, correlated within family. Operational gates are point criteria, not proof lower confidence bound exceeds threshold. Family-bootstrap intervals and failure cells reported.',
        'hypotheses':[
            'hs20 native-coordinate ridge predicts postnorm preRoPE Q/K at blocks27..30 on unseen source/entity names and frames better than embedding/mean/random-coordinate controls.',
            'Source-match x entity-match external interaction predicts record-routing beyond additive source/entity/position controls.',
            'Preserved original V plus forecast QK improves step and recurrent state/output prediction over same-width state-only updates; not a full model replacement.',
            'Norm-equal partial native head/MLP suppression has distinct effects from corresponding controls; mismatch stops those claims.'],
        'fit':'discovery-only selection; ridge lambda1 standardized128 features; no heldout hyperparameter/layer search; native full-coordinate outputs scored',
        'confirmation':'candidate before holdout internal; >=.10 write MSE improvement vs discovery mean AND improvement vs embedding baseline in BOTH panels; query/key/attention/write scores separate',
        'GLM':'Only after Qwen behavior/replay pass and >=1 formation candidate passes both panels. Own discovery heads in fixed27..30 block window, not coordinate transfer. Complete GLM discovery gate before internal.',
        'controls':'per-example actual output norm matching by partial damping to common minimum norm; direction and dose separately reported; same-block random units fixed discovery; no answer vector transfer',
        'resources':{'campaign_GPU_minutes':300,'per_process_minutes':90,'disk_peak_GiB':56,'reserve_GiB':20,'chunk':64,'batch':8,'max_new_tokens':4,'models_concurrent':1},
        'evidence_status':'local experiment; no automatic formal evidence promotion'})
    print('FROZEN '+str(OUT/'plan.json'),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare']);ap.parse_args();prepare()
