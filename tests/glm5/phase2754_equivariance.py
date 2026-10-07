"""Post-hoc description of role-swap equivariance of the frozen predicate score."""
import json
from pathlib import Path
import numpy as np
from phase2754_relation_stability import OUT,write,now,snapshot
from phase2754_analysis import PHI,bootstrap

def run():
    selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']['semantic'];records=[]
    for root in (OUT,OUT/'fact_order_challenge'):
        if root==OUT:
            rows=json.loads((OUT/'probe_rows.json').read_text(encoding='utf-8'));path=root/'probe_predictions.npz'
        else:
            src={r['id']:r for r in json.loads((root/'material.json').read_text(encoding='utf-8'))['rows']}
            rows=[src[r['id']] for p in sorted((root/'4B/confirmation').glob('chunk_*.json')) for r in json.loads(p.read_text(encoding='utf-8'))];path=root/'predictions.npz'
        with np.load(path) as z:score=z[selected][:,0]
        for i in range(0,len(rows),8):
            rr=rows[i:i+8];assert [r['cell'] for r in rr]==list(range(8)) and len({r['group'] for r in rr})==1
            p=score[i:i+8];coef=PHI.T@p/8;gain=abs(coef[4]);other=coef.copy();other[4]=0
            def error(mask,sign):return float(np.sqrt(np.mean((p[np.arange(8)^mask]-sign*p)**2))/max(np.sqrt(np.mean(p*p)),1e-12))
            r=rr[0];records.append(dict(world=r['world'],family=r['family'],split=r['split'],group=r['group'],predicate_gain=float(coef[4]),
                nuisance_over_predicate=float(np.linalg.norm(other)/max(gain,1e-12)),fact_flip_error=error(4,-1),query_flip_error=error(2,-1),double_flip_error=error(6,1),negation_invariance_error=error(1,1)))
    summary={s:{k:bootstrap([r for r in records if r['split']==s],k) for k in ('predicate_gain','nuisance_over_predicate','fact_flip_error','query_flip_error','double_flip_error','negation_invariance_error')} for s in sorted({r['split'] for r in records})}
    write(OUT/'equivariance_diagnostic.json',dict(created_utc=now(),source=snapshot(Path(__file__)),status='Post-hoc empirical score analysis; label algebra is known, not new mathematics.',
        interpretation='One fixed decoder of role-position states yields an approximately sign-equivariant predicate score across entity/grammar/depth/order. This describes the extracted score, not a proven group action on all native hidden coordinates.',splits=summary))
    print({s:{k:v[k]['mean'] for k in ('predicate_gain','nuisance_over_predicate','negation_invariance_error')} for s,v in summary.items()},flush=True)

if __name__=='__main__':run()
