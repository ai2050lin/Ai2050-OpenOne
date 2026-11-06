"""Real Qwen3-4B research-to-client loop. No synthetic fallback or automatic evidence promotion.

prepare -> pilot -> collect -> discover -> validate -> publish.
Uses template_grammar.v1 and the frozen dist_protocol v3 runner. All choices in
DESIGN are written before inference. Holdout results are not read in discover.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import grammar as GR

RUN = 'P2768-KN-FRUIT-TOOL-v1'
OUT = ROOT / 'tests/glm5/result/phase2768_knowledge_loop'
PUBLIC = ROOT / 'research/phase2761' / RUN
MODEL = Path('D:/develop/model/Qwen/Qwen3-4B')
BLOCK = 18
DESIGN = {
    'run_id': RUN, 'version': 1, 'model': str(MODEL), 'precision': 'bf16',
    'primary_block': BLOCK, 'readout': 'final', 'feature': 'silu(G)*V',
    'selection': 'largest absolute pooled Cohen d over all 9728 units, discovery entities averaged across frames',
    'permutations': 1000, 'seed': 2768, 'min_std': 1e-4,
    'discovery_fwer_alpha': .05, 'heldout_min_auc': .8, 'heldout_min_abs_d': .8,
    'min_behavior_acc': .8, 'min_tokens': 256, 'max_tokens': 768,
    'splits': 'entity-family deterministic SHA256; 50/25/25; third frame never used for selection',
    'claim': 'category-associated single-unit candidate, not a causal mechanism or independent graph predictor',
    'not_measured': ['necessity', 'sufficiency', 'natural free generation', 'graph-only next-token KL', 'other models', 'other languages'],
}
FRUITS = 'apple banana orange pear peach plum cherry grape lemon lime mango papaya pineapple coconut apricot nectarine tangerine grapefruit raspberry blueberry blackberry cranberry strawberry watermelon cantaloupe honeydew kiwi fig date guava pomegranate lychee'.split()
TOOLS = ['hammer','screwdriver','wrench','pliers','chisel','saw','drill','clamp','vise','file','rasp','awl','mallet','hatchet','crowbar','spanner','ratchet','socket wrench','soldering iron','trowel','shovel','rake','hoe','pickaxe','pruning shears','wire cutters','bolt cutters','handsaw','hacksaw','plane','screw clamp','paint scraper']
BACKGROUND = (
    'A documentation team is preparing a careful catalogue for readers who have never visited its collection. '
    'Each record must distinguish the name of an item from its description, the purpose of the current question, '
    'and the evidence used to support an answer. The team keeps the original spelling so that later readers can '
    'check the source without guessing what an editor intended. A short answer is useful only when it addresses '
    'the requested distinction. Extra comments about appearance, location, ownership, or price should not replace '
    'the requested classification.\n\n'
    'During review, two editors read every record independently. They compare ordinary usage, note possible '
    'ambiguities, and retain disagreements rather than silently changing the observations. Their notes explain '
    'that an unfamiliar name is not evidence that an item belongs to a different group. Likewise, a familiar '
    'name does not remove the need to read the question. Records are organised into separate sections, but the '
    'position of a record is an administrative choice and is not part of its meaning. Nothing in the preceding '
    'discussion supplies the answer for the particular entry below.\n\n'
    'For the final entry, use the ordinary concrete meaning of the name. Do not treat it as a company, a surname, '
    'a metaphor, or a software command. Give the requested common category rather than repeating the name. '
    'The review form accepts a concise category word; explanations can be written in a separate document. '
    'The entry and the question follow this background so that the answer can depend on all information that '
    'has actually been supplied.\n\n'
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, doc, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        if read(path) != doc: raise ValueError('Immutable artifact differs: ' + str(path))
        return
    temp = path.with_suffix(path.suffix + '.tmp'); temp.write_text(raw, encoding='utf-8'); os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def status(stage, state='done', **kw):
    p = PUBLIC / '00-progress.research.json'
    d = read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'research_plan','run_id':RUN,
        'title':'长文本水果/工具类别响应 · 首个图谱拼图闭环','stages':{},'design':DESIGN}
    d['stages'][stage] = {'state':state,'time':time.time(),**kw}; d['updated_at']=time.time()
    save(p,d)
    print(json.dumps({'stage':stage,'state':state,**kw},ensure_ascii=False),flush=True)


def spec():
    frames=[]
    endings=[('catalogue','The catalogue entry is {X}. What common category does this item belong to?\nCategory:'),
             ('review','A reviewer is examining {X}. Classify the named item by its ordinary category.\nCategory:'),
             ('heldout','For {X}, supply the ordinary category rather than a description of the individual item.\nCategory:')]
    for fid,text in endings:
        text=BACKGROUND+text
        frames.append({'id':fid,'levels':{'frame':fid},'text':text,'target':'X',
            'probe':{'prompt':text,'answer':'{Y}','answers':['{Y}'],'contrast':'auto:other'}})
    fillers=[{'id':f'{category}-{i:02d}','levels':{'category':category},'slots':{'X':word,'Y':' '+category}}
             for category,words in [('fruit',FRUITS),('tool',TOOLS)] for i,word in enumerate(words)]
    return {'schema_version':'template_grammar.v1','family_id':'KN-LOOP','grammar_version':1,'module':'KN',
        'name':'长文本类别响应闭环','factors':['category','frame'],'layers':'all','readouts':['target','final'],
        'frames':frames,'fillers':fillers,'splits':{'unit':'filler','fractions':{'discovery':.5,'confirm':.25,'lockbox':.25},
        'salt':RUN,'heldout_frames':['heldout']},'prereg':{'min_behavior_acc':.8,'min_cell_n':8,'min_lockbox_cos':.5},
        'mlp_units':{'readouts':['target','final'],'layers':[BLOCK]},'shards':1,
        'concepts':{'category=fruit':'水果类别','category=tool':'工具类别'}}


def prepare():
    from transformers import AutoTokenizer
    from shared.dist_protocol import runner_v3 as runner
    OUT.mkdir(parents=True,exist_ok=True)
    g=spec(); errors=GR.validate(g)
    if errors: raise ValueError(errors)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    corpus=GR.generate(g); corpus.update(tm_id=RUN,version=1)
    pilot=GR.generate(g,per_cell_limit=3);pilot.update(tm_id=RUN+'-P',version=1)
    study=[i for i in corpus['items'] if not i.get('canary')]
    tokens={i['id']:tok(i['text'],add_special_tokens=False)['input_ids'] for i in study}
    lengths=[len(t) for t in tokens.values()]
    assert len(FRUITS)==len(TOOLS)==32 and len(study)==192
    assert min(lengths)>=DESIGN['min_tokens'] and max(lengths)<=DESIGN['max_tokens']
    assert len({t[-1] for t in tokens.values()})==1, 'Final-token embedding baseline must be identical'
    assert all(i['split']=='discovery' for i in pilot['items'] if not i.get('canary'))
    frozen={'design':DESIGN,'grammar_sha256':GR.spec_sha(g),'script_sha256':sha(__file__),
        'runner_sha256':sha(runner.__file__),'length_range': [min(lengths),max(lengths)],
        'final_token_id':next(iter(tokens.values()))[-1],'created_at':time.time()}
    if (OUT/'freeze.json').exists(): raise ValueError('Use a new run/version; freeze exists')
    save(ROOT/'shared/dist_protocol/grammars/KN-LOOP-v1.json',g,True)
    save(OUT/'corpus.json',corpus,True);save(OUT/'pilot-corpus.json',pilot,True)
    save(OUT/'freeze.json',frozen,True)
    save(OUT/'tokens.json',tokens,True)
    status('freeze',samples=len(study),entities=64,length_range=frozen['length_range'])


def collect(pilot):
    import torch
    from shared.dist_protocol import runner_v3 as runner
    assert torch.cuda.is_available(), 'Real CUDA required; no CPU or synthetic fallback'
    freeze=read(OUT/'freeze.json');assert freeze['runner_sha256']==sha(runner.__file__)
    assert freeze['script_sha256']==sha(__file__), 'Collector changed after freeze; create a new version'
    if not pilot:
        assert read(OUT/'pilot-gate.json')['passed'], 'Pilot behavior gate must pass before full collection'
    stage='pilot' if pilot else 'collection'; target=OUT/stage
    assert not (target/'summary.json').exists(), 'Collection is immutable'
    target.mkdir(parents=True,exist_ok=True)
    corpus=read(OUT/('pilot-corpus.json' if pilot else 'corpus.json'))
    items=[i for i in corpus['items'] if not i.get('canary')];canaries=[i for i in corpus['items'] if i.get('canary')]
    os.environ['AI2050_RUNNER_MODEL']=str(MODEL);os.environ['AI2050_NATIVE_DIR']=str(OUT/'native')
    os.environ['AI2050_PRECISION']='bf16'
    summary={'tm_id':corpus['tm_id'],'family_id':corpus['family_id'],'shard':'0/1','precision':'bf16',
        'corpus_fingerprint':GR.spec_sha(spec())[:16], 'protocol':'dist_protocol.v3','n_items':len(items)}
    status(stage,'running',samples=len(items))
    with torch.inference_mode():runner.run_real(corpus,items,items,canaries,corpus['factors'],'bf16',target,summary)
    assert summary['status']=='real' and summary['device']=='cuda'
    save(target/'summary.json',summary,True)
    behavior=read(target/'behavior.json')
    rows=behavior if isinstance(behavior,list) else behavior['items']
    cells={}
    for row in rows: cells.setdefault(row['cell'],[]).append(bool(row['correct']))
    gate={k:{'n':len(v),'accuracy':sum(v)/len(v)} for k,v in cells.items()}
    passed=bool(gate) and all(v['accuracy']>=DESIGN['min_behavior_acc'] for v in gate.values())
    save(OUT/('pilot-gate.json' if pilot else 'behavior-gate.json'),{'passed':passed,'cells':gate},True)
    status(stage,'done' if passed else 'behavior_failed',gate=gate)


def data():
    import numpy as np
    summary=read(OUT/'collection/summary.json')
    native=Path(summary['native_field']['dir']);manifest=read(native/'manifest.json')
    mlp=summary['mlp_units']['native']; mp=Path(mlp['dir']);mm=read(mp/'mlp.json')
    assert sha(mp/'mlp.json')==mlp['manifest_sha256']
    part=mm['files'][0];assert sha(mp/part['path'])==part['sha256']
    arr=np.load(mp/part['path']);p=mm['readouts'].index('final')
    G,V=arr['G'][p].astype('float64'),arr['V'][p].astype('float64')
    A=G/(1+np.exp(-np.clip(G,-50,50)))*V
    ids=mm['ids'];items={i['id']:i for i in read(OUT/'corpus.json')['items'] if not i.get('canary')}
    return summary,native,manifest,ids,items,A


def entity_rows(ids,items,A,split):
    import numpy as np
    grouped={}
    original=GR.assign_splits(spec())
    for i,key in enumerate(ids):
        it=items[key]
        use=(it['split']==split) if split!='unseen_frame_and_entities' else (it['split']=='lockbox_frame' and original[it['filler_id']]=='lockbox')
        if use:grouped.setdefault(it['filler_id'],[]).append(i)
    names=sorted(grouped)
    X=np.stack([A[grouped[n]].mean(0) for n in names])
    y=np.array([items[ids[grouped[n][0]]]['cond']['category']=='fruit' for n in names],int)
    return names,X,y


def effect(X,y):
    import numpy as np
    a,b=X[y==1],X[y==0];s=np.sqrt(((len(a)-1)*a.var(0,ddof=1)+(len(b)-1)*b.var(0,ddof=1))/(len(X)-2))
    return (a.mean(0)-b.mean(0))/np.maximum(s,DESIGN['min_std'])


def auc(v,y):
    import numpy as np
    delta=v[y==1,None]-v[y==0][None,:]
    return float(((delta>0)+.5*(delta==0)).mean())


def discover():
    import numpy as np
    summary,native,manifest,ids,items,A=data()
    names,X,y=entity_rows(ids,items,A,'discovery');d=effect(X,y)
    unit=int(np.argmax(np.abs(d)));sign=1 if d[unit]>=0 else -1
    rng=np.random.default_rng(DESIGN['seed']);null=np.array([np.abs(effect(X,rng.permutation(y))).max() for _ in range(DESIGN['permutations'])])
    p=float((1+(null>=abs(d[unit])).sum())/(len(null)+1))
    result={'unit':unit,'block':BLOCK,'sign':sign,'d':float(d[unit]),'fwer_p':p,'entities':names,
        'threshold':float((X[y==1,unit].mean()+X[y==0,unit].mean())/2),'all_units':A.shape[1],
        'selection_used_holdout':False,'model_snapshot':summary['model_snapshot']['sha256']}
    save(OUT/'discovery.json',result,True)
    np.savez_compressed(OUT/'discovery-full-array.npz',d=d,null_max_abs_d=null)
    status('discovery',unit=unit,block=BLOCK,fwer_p=p)
    # Publish discovery separately before computing any heldout statistic.
    publish('discovery',result)


def validate():
    import numpy as np
    result=read(OUT/'discovery.json');j=result['unit'];s=result['sign']
    summary,native,manifest,ids,items,A=data();metrics={}
    for split in ['confirm','lockbox','unseen_frame_and_entities']:
        names,X,y=entity_rows(ids,items,A,split);v=s*X[:,j]
        score=auc(v,y); rng=np.random.default_rng(DESIGN['seed']+len(names));boots=[]
        for _ in range(1000):
            pick=np.r_[rng.choice(np.flatnonzero(y==1),sum(y==1)),rng.choice(np.flatnonzero(y==0),sum(y==0))]
            boots.append(auc(v[pick],y[pick]))
        metrics[split]={'n_entities':len(names),'auc':score,'d_oriented':float(s*effect(X,y)[j]),
            'threshold_accuracy':float(((s*(X[:,j]-result['threshold'])>0)==y).mean()),
            'auc_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'entities':names}
    # Same-final-token hs[0] baseline, and target-token lexical and length baselines.
    emb=np.load(native/'L0.npz')['X'].astype('float64');p=manifest['readouts'].index('final')
    final_emb=emb[p];assert np.max(np.abs(final_emb-final_emb[:1]))==0
    rawids=manifest['ids'];positions={key:i for i,key in enumerate(rawids)}
    target=emb[manifest['readouts'].index('target')][[positions[key] for key in ids]]
    dn,DX,dy=entity_rows(ids,items,target,'discovery');direction=DX[dy==1].mean(0)-DX[dy==0].mean(0)
    tokens=read(OUT/'tokens.json');length=np.array([[len(tokens[key])] for key in ids],float)
    _,LX,ly=entity_rows(ids,items,length,'discovery');ls=1 if effect(LX,ly)[0]>=0 else -1
    baselines={}
    for split in metrics:
        _,TX,ty=entity_rows(ids,items,target,split);_,LX,ly=entity_rows(ids,items,length,split)
        baselines[split]={'final_embedding_auc':.5,'target_token_centroid_auc':auc(TX@direction,ty),'length_auc':auc(ls*LX[:,0],ly)}
    passed=(read(OUT/'behavior-gate.json')['passed'] and result['fwer_p']<=DESIGN['discovery_fwer_alpha']
        and all(v['auc']>=DESIGN['heldout_min_auc'] and v['d_oriented']>=DESIGN['heldout_min_abs_d'] for v in metrics.values()))
    counter=[]
    for i,key in enumerate(ids):
        it=items[key]
        if it['split']=='discovery':continue
        predicted='fruit' if s*(A[i,j]-result['threshold'])>0 else 'tool'
        if predicted!=it['cond']['category']:
            counter.append({'id':key,'entity':it['target'],'frame':it['frame_id'],'split':it['split'],'expected':it['cond']['category'],'predicted':predicted,'activation':float(A[i,j])})
    result.update(metrics=metrics,baselines=baselines,passed=passed,counterexamples=counter,
        outcome='heldout_category_association' if passed else 'not_supported_under_preregistered_gates',
        limitations=['One selected MLP unit is a response association; no causal intervention.',
        'Entity names carry category information; success above constant final embedding does not eliminate lexical explanations.',
        'Same source grammar and English only; entity bootstrap does not cover all template uncertainty.',
        'No graph-only full-vocabulary prediction, no claim of mechanism closure.'])
    save(OUT/'validation.json',result,True);status('validation','done' if passed else 'not_supported',outcome=result['outcome'])
    publish('validated',result)


def publish(revision,result):
    from deploy.research_library import Library
    from deploy.parameter_reader import read_slice,catalog
    lib=Library(ROOT/'distributed_data/research_library')
    summary=read(OUT/'collection/summary.json');unit=result['unit'];snapshot=result['model_snapshot']
    local=next(m for m in catalog([MODEL]) if m['_path']==MODEL.resolve())
    params={name:read_slice(MODEL,f'model.layers.{BLOCK}.mlp.{name}.weight',unit if name!='down_proj' else 0,0 if name!='down_proj' else unit,1,1)
        for name in ['gate_proj','up_proj','down_proj']}
    raw_files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [OUT/'corpus.json',OUT/'freeze.json',OUT/'discovery.json',OUT/'collection/summary.json',OUT/'collection/behavior.json']}
    if revision=='validated':raw_files[str((OUT/'validation.json').relative_to(ROOT)).replace('\\','/')]=sha(OUT/'validation.json')
    report={'schema':'research_evidence.v1','kind':'real','run_id':RUN,'revision':revision,'model_snapshot':snapshot,
        'model_name':'Qwen3-4B','title':'水果与工具 · 长文本类别响应','design':DESIGN,'result':result,
        'raw_artifacts':raw_files,'native_field':summary['native_field'],'mlp_units':summary['mlp_units'],
        'snapshot_manifest':summary['model_snapshot'],'parameter_checks':params,'software_only':False,
        'sample_count':192,'entity_count':64,'token_length_range':read(OUT/'freeze.json')['length_range'],
        'model_binding':{'local_model_id':local['id'],'snapshot':snapshot},'reproduction':str(Path(__file__).relative_to(ROOT))}
    reportpath=PUBLIC/f'{revision}-evidence.research.json';save(reportpath,report,True)
    evidence=lib.ingest(report,str(reportpath.relative_to(ROOT)),kind='local_report')['sha256']
    fid=f'{snapshot}:B{BLOCK}:mlp:{unit}:final:{RUN}'
    level='observation' if revision=='discovery' else ('heldout' if result['passed'] else 'negative')
    card={'id':fid,'label':f'类别响应候选 · B{BLOCK} / U{unit}','layer':BLOCK,'boundary':'mlp.down_proj.input',
        'index':{'space':'mlp','unit':unit},'method':'entity_grouped_maxT.v1','run_id':RUN,
        'scope':{'language':'en','categories':['fruit','tool'],'readout':'final','context':'long catalogue/review prompts'},
        'evidence':{'level':level,'artifact_sha256':evidence},'counterexamples':result.get('counterexamples',[]),
        'summary':result,'causal_tests':'not_measured','natural_contexts':'Existing census available; this candidate was selected on controlled discovery, not a learned dictionary.',
        'parameter_path':{name:f'model.layers.{BLOCK}.mlp.{name}.weight' for name in params},
        'model_binding':report['model_binding']}
    graph={'schema':'research_graph.v1','kind':'real','run_id':RUN,'revision':revision,'model_snapshot':snapshot,
        'feature_cards':[card],'graph_edges':[{'id':fid+':category','source':'language:knowledge:fruit-vs-tool',
        'target':fid,'relation':'heldout_response_association' if level=='heldout' else 'candidate_response_association' if level=='observation' else 'not_supported',
        'run_id':RUN,'evidence':{'level':level,'artifact_sha256':evidence}}]}
    graphpath=PUBLIC/f'{revision}-graph.research.json';save(graphpath,graph,True)
    registered=lib.ingest(graph,str(graphpath.relative_to(ROOT)),kind='registered')
    save(OUT/f'{revision}-registration.json',{'report_sha256':evidence,'graph_sha256':registered['sha256'],'feature_id':fid},True)
    status('registration_'+revision,feature_id=fid,graph_sha256=registered['sha256'],evidence_level=level)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','collect','discover','validate']);args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage in ('pilot','collect'):collect(args.stage=='pilot')
    elif args.stage=='discover':discover()
    else:validate()
