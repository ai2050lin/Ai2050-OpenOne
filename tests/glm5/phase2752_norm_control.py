"""Posthoc boundary diagnostic: does final RMSNorm alone explain interaction?

Uses the three raw source residuals, never the target residual as input. The
known norm formula is an architectural baseline, not a new extracted mechanism.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from phase2752_context_interaction import ROOT, OUT, now, sha, write, snapshot


def run(side, assertion=False):
    from safetensors import safe_open
    root=OUT/('assertion_control' if assertion else '')
    dest=root/side
    done=json.loads((dest/'capture_done.json').read_text(encoding='utf-8'))
    material=json.loads((root/'material.json').read_text(encoding='utf-8'))
    byid={r['id']:r for r in material['rows']}
    mdir=ROOT/'models/hf'/('qwen3-4b' if side=='4B' else 'Qwen3-14B')
    config=json.loads((mdir/'config.json').read_text(encoding='utf-8'))
    ix=json.loads((mdir/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    with safe_open(mdir/ix['model.norm.weight'],framework='pt',device='cpu') as f:
        gamma=f.get_tensor('model.norm.weight').double().numpy()
    def norm(x):
        return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+config['rms_norm_eps'])*gamma
    groups=[]
    for path in sorted(dest.glob('chunk_*.npz')):
        rows=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        with np.load(path) as z:
            h=z['hidden']
        for start in range(0,len(rows),4):
            rr=rows[start:start+4]
            assert len({r['group'] for r in rr})==1
            assert [byid[r['id']]['cond'] for r in rr]==[0,1,2,3]
            post=h[start:start+4,done['last_norm_index']].astype(np.float64)
            raw=h[start:start+4,done['raw_last_index']].astype(np.float64)
            inp=raw[2]+raw[1]-raw[0]
            prediction=norm(inp)
            target_i=post[3]-post[2]-post[1]+post[0]
            err=np.linalg.norm(prediction-post[3])
            true_formula=norm(raw)
            groups.append(dict(group=rr[0]['group'],world=rr[0]['world'],split=rr[0]['split'],
                interaction_error=float(err/np.linalg.norm(target_i)),
                combined_error=float(err/np.linalg.norm(post[3]-post[0])),
                norm_float64_vs_native_relative=float(np.max(np.linalg.norm(true_formula-post,axis=-1)/np.linalg.norm(post,axis=-1)))))
    summary={}
    for split in sorted({g['split'] for g in groups}):
        gg=[g for g in groups if g['split']==split]
        summary[split]=dict(n=len(gg),interaction_error=float(np.mean([g['interaction_error'] for g in gg])),
                           combined_error=float(np.mean([g['combined_error'] for g in gg])))
    write(dest/'norm_control.json',dict(created_utc=now(),status='Posthoc architectural baseline, no independent discovery claim',
        source=snapshot(Path(__file__)),input='raw r00,r10,r01 and learned final gamma; no r11 input',
        formula='predicted postnorm h11 = RMSNorm(r10+r01-r00)',
        numeric='FP64 analytic norm of actual BF16 source residuals; not bit-identical to native BF16. Reference discrepancy measured on all observed inputs.',
        max_norm_reference_relative=max(g['norm_float64_vs_native_relative'] for g in groups),splits=summary,groups=groups))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='4B',choices=['4B','14B'])
    p.add_argument('--assertion',action='store_true')
    a=p.parse_args()
    run(a.model,a.assertion)
