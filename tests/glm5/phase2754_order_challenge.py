"""Independent follow-up: preserve graph, reverse fact presentation order."""
import argparse,json,re
from pathlib import Path
from collections import Counter
import numpy as np
import phase2754_relation_stability as native
from phase2754_relation_stability import ROOT,OUT,FAMILIES,write,sha,now,snapshot,render

DEST=OUT/'fact_order_challenge'

def prepare():
    if (DEST/'material.json').exists():return json.loads((DEST/'material.json').read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    used={n for w in json.loads((OUT/'material.json').read_text(encoding='utf-8'))['worlds'] for n in w['names']}
    syll=('ba','ce','di','fo','gu','ha','ji','ko','lu','me','ni','po','ra','se','ti','vu')
    pool=[(''.join(syll[(i//16**k)%16] for k in (3,2,1,0))+'z').capitalize() for i in range(65536)]
    pool=[p for p in pool if p not in used];np.random.default_rng(2754009).shuffle(pool)
    worlds=[];rows=[]
    for fam in FAMILIES:
        for j in range(16):
            wid=f'{fam}_order_challenge_{j:02d}';names=pool[len(worlds)*4:len(worlds)*4+4];template=j%2
            worlds.append(dict(id=wid,family=fam,names=names,depth=3,index=j,template=template))
            for order in (0,1):
                group=f'{wid}_o{order}';split='ordered' if order==0 else 'reversed'
                for theta in (-1,1):
                    path=names if theta==1 else names[::-1]
                    for rho in (-1,1):
                        a,b=(names[0],names[-1]) if rho==1 else (names[-1],names[0])
                        for eta in (-1,1):
                            text,_=render(fam,path,a,b,template,eta)
                            facts=re.findall(r'(?:Fact|Event) \d+: [^.]+\.',text);assert len(facts)==3
                            if order:text=text.replace(' '.join(facts),' '.join(facts[::-1]),1)
                            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);start=text.index(' the case that ')+len(' the case that ');positions=[]
                            for name in (a,b):
                                lo=text.index(name,start);hi=lo+len(name);ii=[k for k,(x,y) in enumerate(enc['offset_mapping']) if x<hi and y>lo];positions.append(ii[-1])
                            cell=int(theta==1)*4+int(rho==1)*2+int(eta==1)
                            rows.append(dict(id=f'{group}_c{cell}',group=group,world=wid,family=fam,split=split,index=j,template=template,depth=3,order=order,
                                theta=theta,rho=rho,eta=eta,cell=cell,predicate_sign=theta*rho,truth_sign=theta*rho*eta,expected='yes' if theta*rho*eta==1 else 'no',
                                text=text,edges=list(zip(path[:-1],path[1:])),query=[a,b],replicate14B=False,
                                tokenization={'4B':dict(token_ids=enc['input_ids'],length=len(enc['input_ids']),query_source_target_last_tokens=positions)}))
    assert len(worlds)==64 and len(rows)==1024
    for w in worlds:
        for c in range(8):
            rr=[r for r in rows if r['world']==w['id'] and r['cell']==c];assert len(rr)==2
            assert Counter(rr[0]['tokenization']['4B']['token_ids'])==Counter(rr[1]['tokenization']['4B']['token_ids'])
            assert rr[0]['truth_sign']==rr[1]['truth_sign'] and rr[0]['edges']==rr[1]['edges']
    mat=dict(created_utc=now(),worlds=worlds,rows=rows,sampling_unit='64 new worlds, paired fact order,8 conditions per order')
    write(DEST/'material.json',mat)
    write(DEST/'design.json',dict(created_utc=now(),question='Does frozen relation decoding survive unchanged graph with reversed fact order?',
        status='New follow-up designed after primary confirmation, independently new64worlds; no refit or threshold choice.',
        design='4families x16newworlds x2factorders x8conditions; all depth3, active/passive assigned byworld index. Exact token bags and known graph unchanged across order.',
        confound='Role events retain numerical event IDs, so event semantics do not change. This tests a specific presentation shortcut, not arbitrary graph topology.',
        frozen_inputs='Original4B probe fits and selection; end/slot states only. Labels are scoring targets.',budget='One4B sequential run1024prompts; inherited900s hardcap, expected<90s. No14B challenge allocated.'))
    (DEST/'selection.json').write_bytes((OUT/'selection.json').read_bytes())
    write(DEST/'pre_capture_seal.json',dict(created_utc=now(),source=snapshot(Path(__file__)),material_sha256=sha(DEST/'material.json'),design_sha256=sha(DEST/'design.json'),parent_selection_sha256=sha(OUT/'selection.json'),same_bag_graph_truth=True))
    print(dict(worlds=64,prompts=1024),flush=True);return mat

def collect():
    mat=prepare();native.OUT=DEST;native.prepare=lambda:mat
    write(DEST/'wrapper_execution.json',dict(created_utc=now(),source=snapshot(Path(__file__)),native_collector_sha256=sha(Path(native.__file__)),parent_selection_sha256=sha(OUT/'selection.json')))
    native.collect('4B','confirmation')

def evaluate():
    import phase2754_probes as probe
    from phase2754_analysis import bootstrap
    probe.OUT=DEST;rows,h,q,meta=probe.load('confirmation')
    sel=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));methods=[sel['selected']['semantic'],'finalnorm'];preds={}
    for method in methods:
        p=OUT/'probe_fits'/f'{method}.npz';assert sha(p)==sel['fits'][p.name]
        blocks=probe.input_blocks(rows,h,q,method)
        with np.load(p) as z:
            xx,_=probe.standardize(blocks,parameters=[(z[f'mu{j}'],float(z[f'sd{j}'])) for j in range(len(blocks))]);preds[method]=xx@z['coef']+z['ym']
    y=np.array([r['truth_sign'] for r in rows]);predicate=np.array([r['predicate_sign'] for r in rows]);records=[]
    for i,(r,m) in enumerate(zip(rows,meta)):
        records.append(dict(world=r['world'],family=r['family'],split=r['split'],native_accuracy=float(m['prediction_text'].strip().lower()==r['expected']),
            truth_effect=y[i]*m['margin'],early_truth=float((preds[methods[0]][i,1]>=0)==(y[i]>0)),early_predicate=float((preds[methods[0]][i,0]>=0)==(predicate[i]>0)),
            final_truth=float((preds['finalnorm'][i,1]>=0)==(y[i]>0))))
    summaries={s:{k:bootstrap([r for r in records if r['split']==s],k) for k in ('native_accuracy','truth_effect','early_truth','early_predicate','final_truth')} for s in ('ordered','reversed')}
    diffs=[]
    for w in sorted({r['world'] for r in records}):
        rr=[r for r in records if r['world']==w];d=dict(world=w,family=rr[0]['family'])
        for k in summaries['ordered']:d[k]=float(np.mean([r[k] for r in rr if r['split']=='reversed'])-np.mean([r[k] for r in rr if r['split']=='ordered']))
        diffs.append(d)
    summaries['reversed_minus_ordered']={k:bootstrap(diffs,k) for k in summaries['ordered']}
    write(DEST/'summary.json',dict(created_utc=now(),source=snapshot(Path(__file__)),worlds=64,prompts=1024,method=methods[0],results=summaries,
        no_refit=True,scope='Independent newworld follow-up; fact order only paired withinworld. Fixed depth3 and2 known grammar templates.'))
    np.savez(DEST/'predictions.npz',**preds);write(DEST/'metric_rows.json',records)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect','evaluate']);a=p.parse_args();{'prepare':prepare,'collect':collect,'evaluate':evaluate}[a.mode]()
