"""Explicit post-4B-result diagnostic: distinguish source information from products.

Does not modify the preregistered candidates or select a new primary winner.
The 14B specification is sealed before its formal capture/outcome inspection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from phase2752_context_interaction import OUT, now, sha, write, snapshot
from phase2752_predict_interaction import load, relative, interaction, coordinate_input, coordinate_fit_predict, finite_mean, SPLITS, ALPHAS


def run(side):
    seal = OUT/'source_ablation_design.json'
    if not seal.exists():
        write(seal,dict(created_utc=now(),source=snapshot(Path(__file__)),status=('Reuse of primary-corpus diagnostic on the assertion control; no independent confirmation claim' if OUT.name=='assertion_control' else 'Posthoc diagnostic after seeing main4B test results; before14B formal outcomes'),
            reason='Primary source-product method also adds single-condition inputs. Need linear-source and h01/h10 carry-forward controls before attributing benefit to nonlinear products.',
            methods=['negative-only state h01 (Ihat=-delta_style)','style-only state h10 (Ihat=-delta_negation)',
                     'source_linear [h00,delta_style,delta_negation]', 'source_linear_shuffled_train_targets_within_family_and_wording'],
            selection='Alpha via validation final-norm interaction mean only; no change to original primary winner or results.'))
    write(OUT/side/'source_ablation_execution.json',dict(created_utc=now(),source=snapshot(Path(__file__)),shared_design_sha256=sha(seal)))
    rows,h,_,done,_ = load(side)
    train = np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    val = np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    last = done['last_norm_index']
    # Same permutation of worlds for every wording, preserving repeated-measure structure.
    rng = np.random.default_rng(2752004)
    source_map = {}
    for family in sorted({r['family'] for r in rows}):
        worlds = sorted({rows[i]['world'] for i in train if rows[i]['family']==family})
        source_map.update(dict(zip(worlds,rng.permutation(worlds))))
    lookup = {(rows[i]['world'],rows[i]['wording']):k for k,i in enumerate(train)}
    perm = np.array([lookup[(source_map[rows[i]['world']],rows[i]['wording'])] for i in train])
    target = interaction(h[:,:,last])
    z = coordinate_input(h[:,:,last])[...,:3]
    choices,validation = {},{}
    for name in ('source_linear','shuffled_linear'):
        yt = target[train][perm] if name=='shuffled_linear' else target[train]
        vals = {a:finite_mean(relative(coordinate_fit_predict(z,yt,train,a),target)[val]) for a in ALPHAS}
        choices[name] = min(vals,key=vals.get)
        validation[name] = vals
    metrics = {m:np.empty((len(rows),h.shape[2]),dtype=np.float32) for m in ('negative_only','style_only','source_linear','shuffled_linear')}
    final = {}
    for layer in range(h.shape[2]):
        hl = h[:,:,layer]
        target = interaction(hl)
        pred = dict(negative_only=hl[:,0].astype(np.float64)-hl[:,2],style_only=hl[:,0].astype(np.float64)-hl[:,1])
        z = coordinate_input(hl)[...,:3]
        pred['source_linear'] = coordinate_fit_predict(z,target[train],train,choices['source_linear'])
        pred['shuffled_linear'] = coordinate_fit_predict(z,target[train][perm],train,choices['shuffled_linear'])
        for m,v in pred.items():
            metrics[m][:,layer] = relative(v,target)
        if layer==last:
            final=pred
    with np.load(OUT/side/'prediction_metrics.npz') as original:
        source_product=original['source_product_interaction']
    summary = dict(created_utc=now(),status='diagnostic_not_independent_confirmation',alpha=choices,validation=validation,splits={})
    for split in SPLITS:
        ix=[i for i,r in enumerate(rows) if r['split']==split]
        worlds=sorted({rows[i]['world'] for i in ix})
        diff=source_product[:,last]-metrics['source_linear'][:,last]
        block=np.array([np.mean([diff[i] for i in ix if rows[i]['world']==w]) for w in worlds])
        boot=block[rng.integers(0,len(block),(2000,len(block)))].mean(1)
        summary['splits'][split]=dict(methods={m:finite_mean(v[ix,last]) for m,v in metrics.items()},
            product_minus_linear=dict(mean=float(block.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    write(OUT/side/'source_ablation_summary.json',summary)
    np.savez(OUT/side/'source_ablation_metrics.npz',**metrics)
    np.savez(OUT/side/'source_ablation_final.npz',**final)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['4B','14B'],default='4B')
    run(p.parse_args().model)
