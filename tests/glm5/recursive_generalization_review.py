"""Read-only retrospective candidate review; no model execution or evidence promotion."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import unbits,save_json
OUT=ROOT/'tests/glm5/result/recursive_generalization_review_20261113'
SOURCES=[
 ('qwen4','goodfire_middle_20261113_v2/qwen4',range(18,29),'canonical'),
 ('qwen4','mlp_tail_b28_b31_v1',range(29,32),'legacy'),
 ('qwen4','mlp_tail_goodfire_v1',range(32,36),'legacy'),
 ('qwen14','goodfire_middle_20261113_v2/qwen14',range(18,29),'canonical'),
 ('qwen14','qwen14_mlp_atlas_v1',range(29,36),'legacy'),
 ('qwen14','qwen14_mlp_b28_b39_v2',range(36,40),'legacy'),
]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()
def write(name,data):
    p=check_write(OUT/name);save_json(p,dict(data,timestamp=datetime.now().astimezone().isoformat(),
        analysis_kind='retrospective_existing_real_artifacts',formal_promotion=False))


def inventory():
    check_write(OUT,500*2**20);OUT.mkdir(exist_ok=True)
    rows=[];start=time.monotonic()
    for model,name,blocks,execution in SOURCES:
        folder=ROOT/'tests/glm5/result'/name
        for b in blocks:
            p=folder/f'B{b}-profile.json';d=read(p);q=folder/f'B{b}-cues.json';c=read(q)
            width=d['width'];assert len(c['units'])==width
            assert all(len(d['fields'][f]['mean'])==width for f in ['G','V','A'])
            rows.append(dict(model=model,block=b,width=width,run_id=d['run_id'],snapshot=d['model_snapshot'],
                execution=execution,path=str(folder),profile_sha256=sha(p),cues_sha256=sha(q),
                raw_A=(folder/f'B{b}-A.npy').is_file(),pole_values=(folder/f'B{b}-pole-values.npy').is_file()))
            print(f'inventory {len(rows)}/40 elapsed={time.monotonic()-start:.1f}s {model} B{b}',flush=True)
    write('inventory.json',dict(layers=rows,unique_units={m:sum(r['width'] for r in rows if r['model']==m) for m in ['qwen4','qwen14']},
        overlap_policy='Prefer canonical B28; do not count legacy B28 twice or merge numerical execution versions.',
        limits=['Static checkpoints cannot establish temporal growth during training.',
                'Previously exposed corpora; every new semantic reading is exploratory.',
                'All coordinate coverage is an inventory, not an audit of every semantic explanation.']))
    write('review-plan.json',dict(source_sha256=sha(__file__),budget=dict(new_disk_peak_MiB=500,GPU=False),
        procedure=['Inventory all existing unique layer/unit profiles.',
            'Retrieve repeated lexical cues as search hints; diversify manual candidate review.',
            'Inspect same unit/pole across contexts, branches, weak responses and counterexamples where raw fields survive.',
            'Distinguish related-category generalization, plausible bridged branches, token/source artifacts, and unexplained mixtures.',
            'No training-history inference or population prevalence estimate from selected cases.'],
        evidence_requirements=dict(candidate='A named common core and at least two related but nonidentical observed branches, with explicit alternatives.',
            stronger_static_support='New frozen family material, lexical/position/source controls, branch-specific prediction and intervention.',
            developmental_claim='Training checkpoints or controlled training showing branch emergence and its local connections.')))


def screen():
    rows=[];start=time.monotonic()
    for i,source in enumerate(read(OUT/'inventory.json')['layers']):
        d=read(Path(source['path'])/f"B{source['block']}-cues.json")
        for c in d['units']:
            rep=c['repeat'];ratios=[];docs=[]
            for split in ['confirm','holdout']:
                s=rep[split];ratios.append(s.get('hits',s.get('cue_hit_count',0))/s.get('n',s.get('n_pole_positions',1)))
                docs.append(s.get('distinct_documents',0))
            rows.append(dict(model=source['model'],block=source['block'],unit=c['unit'],keywords=c['positive_keywords'],
                negative=c['negative_keywords'],repeat=ratios,documents=docs,score=min(ratios),execution=source['execution']))
        print(f'screen {i+1}/40 elapsed={time.monotonic()-start:.1f}s',flush=True)
    # Retain every search row; use sorted, diverse summaries only as reading aids.
    save_json(check_write(OUT/'cue-index.json'),rows)
    selected={}
    for model in ['qwen4','qwen14']:
        ranked=sorted((r for r in rows if r['model']==model and r['keywords']),key=lambda r:(-r['score'],-min(r['documents']),r['block'],r['unit']))
        used=Counter();chosen=[]
        for r in ranked:
            key=tuple(sorted(r['keywords'][:3]))
            if used[key]>=1:continue
            if any(used['word:'+w]>=3 for w in r['keywords'][:2]):continue
            chosen.append(r);used[key]+=1
            for w in r['keywords'][:2]:used['word:'+w]+=1
            if len(chosen)==80:break
        selected[model]=chosen
    write('search-shortlist.json',dict(shortlist=selected,scope='Biased lexical retrieval aid; not a recursive-generalization score or prevalence sample.'))


def extract(specs):
    from tokenizers import Tokenizer
    sources=read(OUT/'inventory.json')['layers'];result=[];cache={}
    for query in specs.split(','):
        model,block,unit=query.split(':');b=int(block);u=int(unit)
        source=next(s for s in sources if s['model']==model and s['block']==b);folder=Path(source['path'])
        p=read(folder/f'B{b}-profile.json');assert 0<=u<source['width']
        docroot=folder
        if not (docroot/'documents.json').is_file():
            docroot=ROOT/'tests/glm5/result'/('mlp_tail_goodfire_v1' if model=='qwen4' else 'qwen14_mlp_atlas_v1')
        if str(docroot) not in cache:
            d=read(docroot/'documents.json');tp=docroot/'tokenizer.json'
            if not tp.is_file():tp=ROOT/'models/hf/qwen3-4b/tokenizer.json'
            cache[str(docroot)]=(d,Tokenizer.from_file(str(tp)))
        docs,tok=cache[str(docroot)];positions=docs['positions'];examples=[]
        maps={f:np.load(folder/f'B{b}-{f}.npy',mmap_mode='r') for f in ['G','V','A']} if source['raw_A'] else None
        poles=np.load(folder/f'B{b}-pole-values.npy',mmap_mode='r') if source['pole_values'] else None
        for field in ['A','G','V']:
            for si,split in enumerate(['discovery','confirm','holdout']):
                for pi,pole in enumerate(['positive','negative']):
                    indices=p['fields'][field][split][pole+'_indices'][u]
                    for rank,idx in enumerate(indices):
                        di,slot=divmod(idx,len(positions));pos=positions[slot];d=docs['docs'][di];ids=d['tokens']
                        values={f:float(unbits(a[di,slot,u])) for f,a in maps.items()} if maps else (dict(zip(['G','V','A'],map(float,unbits(poles[['G','V','A'].index(field),si,pi,u,rank])))) if poles is not None else dict(G=None,V=None,A=None))
                        examples.append(dict(field=field,pole=pole,rank=rank,split=split,document=di,position=pos,
                            doc_id=d['doc_id'],source=d['source'],title=d.get('title'),url=d.get('url'),**values,
                            prefix=tok.decode(ids[:pos+1]),window=tok.decode(ids[max(0,pos-120):pos+1]),
                            current=tok.decode([ids[pos]]),following_not_causal=tok.decode(ids[pos+1:pos+33])))
        result.append(dict(**source,unit=u,statistics={f:{k:p['fields'][f][k][u] for k in ['mean','std','min','max']} for f in ['G','V','A']},
            examples=examples))
        print('extracted',query,flush=True)
    write('contexts-'+hashlib.sha256(specs.encode()).hexdigest()[:10]+'.json',dict(units=result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','screen','extract']);parser.add_argument('--units')
    args=parser.parse_args()
    if args.stage=='extract':extract(args.units)
    else:globals()[args.stage]()
