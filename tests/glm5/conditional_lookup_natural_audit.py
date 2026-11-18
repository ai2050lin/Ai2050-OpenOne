"""Independent artifact checks after the candidate seal and confirmation run."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import source_binding_design as D
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
pre=D.read(OUT/'prereg.json');seal=D.read(OUT/'candidate.json');plan=D.read(OUT/'natural-plan.json')
assert seal['analysis_sha']==D.sha(ROOT/'tests/glm5/conditional_lookup_analyze.py')
for n,h in seal['files'].items():assert D.sha(OUT/n)==h
for n,h in plan['code'].items():assert D.sha(ROOT/'tests/glm5'/n)==h
assert (OUT/'candidate.json').stat().st_mtime < (OUT/'confirm-00.npz').stat().st_mtime
report={'provenance':pre['provenance'],'seal_valid':True,'splits':{}}
names={}
for split in ('discovery','confirm'):
    assert D.sha(OUT/f'{split}-materials.json')==pre['materials'][split]['sha256']
    rows=D.read(OUT/f'{split}-materials.json');c=D.read(OUT/f'{split}-collection.json')
    names[split]={name for r in rows for name in r['names']};scores=[];off=0;eq=0
    for fn,h in sorted(c['shards'].items()):
        assert D.sha(OUT/fn)==h
        summary=D.read((OUT/fn).with_suffix('.json'));scores.extend(summary['scores'])
        rr=rows[off:off+len(summary['ids'])];assert [r['id'] for r in rr]==summary['ids']
        index={(r['family'],r['r0'],r['r1'],r['b'],r['late']):i for i,r in enumerate(rr)}
        pairs=[(i,index[r['family'],r['r0'],r['r1'],1,1]) for i,r in enumerate(rr) if r['late'] and r['b']==0]
        with np.load(OUT/fn) as z:
            for layer in range(41):
                hs=z[f'h{layer:02}'];assert hs.dtype==np.uint16
                assert all(np.array_equal(hs[i],hs[j]) for i,j in pairs)
        eq+=len(pairs);off+=len(rr)
    assert len(scores)==len(rows)
    for r,s in zip(rows,scores):
        assert s['id']==r['id'];gold=2*r['b']+(r['r0'],r['r1'])[r['b']]
        assert gold==r['target_record'] and r['gold_id']==r['choice_ids'][gold]
        vals=s['scores'];assert s['gold_margin']==vals[gold]-max(v for j,v in enumerate(vals) if j!=gold)
        assert s['top1_correct']==(s['top1']==r['gold_id'])
    report['splits'][split]={'n':len(rows),'families':len({r['family'] for r in rows}),'strict_correct':sum(s['gold_margin']>0 for s in scores),'top1_correct':sum(s['top1_correct'] for s in scores),'bitwise_equal_pairs_all41':eq,'behavior_pass':c['behavior_pass'],'technical_pass':c['technical_pass'],'source_max_abs':c['source_audit_max']}
    print(split+' hashes, scores and all41-prefix comparisons verified',flush=True)
assert not names['discovery']&names['confirm']
report['disjoint_names']=True
with np.load(OUT/'confirm-predictions.npz') as z:
    correct=(z['row_scores']>=.5)==z['truth'];actual=[float(correct[:,0].mean()),float(correct[:,1].mean()),float(correct.all(1).mean())]
    result=D.read(OUT/'confirm-analysis.json');assert actual==[result['encoding'][k] for k in ('bit0','bit1','joint')]
    report['confirm_encoding_recomputed']=actual
report['scope']='artifact and arithmetic audit, not independent model rerun or causal certification'
p=OUT/'natural-audit.json';assert not p.exists();p.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
