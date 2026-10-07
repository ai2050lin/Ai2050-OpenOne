"""Post-confirmation diagnostics; not new independent confirmation claims."""
import json
from pathlib import Path
import numpy as np
from phase2753_forecast import data,ROOT,OLD,OUT,FAMILIES,write,sha,now,snapshot
import phase2752_predict_interaction as old_algorithm

def run():
    # Prior phase's selected alpha is fixed. Late inputs are a privileged reference,
    # explicitly excluded from the current early-input candidate selection.
    old_summary=json.loads((OLD/'4B/prediction_summary.json').read_text(encoding='utf-8'))
    alpha=old_summary['alpha_choices']['source_product']
    rr,hh,_,_=data(OLD,True);train=np.array([i for i,r in enumerate(rr) if r['split']=='train'])
    y=old_algorithm.interaction(hh[:,:,36])[train]
    source=hh[train,:,36].copy();del hh
    rows,h,meta,mat=data(OUT)
    combined=np.concatenate([source,h[:,:,36]])
    pred=old_algorithm.coordinate_fit_predict(old_algorithm.coordinate_input(combined),y,np.arange(len(train)),alpha)[len(train):]
    truth=old_algorithm.interaction(h[:,:,36]);error=old_algorithm.relative(pred,truth)
    np.savez(OUT/'late_source_reference.npz',prediction=pred,error=error)
    with np.load(OUT/'confirmation_metrics.npz') as z:errors={m:z[m+'_interaction'][:,36] for m in ('family_mean','graph_surface','static_bag','static_binding','target8','base8','four8')}
    errors['late_source_reference']=error
    rng=np.random.default_rng(2753003)
    source_rows={r['id']:r for r in mat['rows']}
    ok={r['id']:r['prediction_text'].strip().lower()==source_rows[r['id']]['expected'] for r in meta}
    grouped={}
    for r in meta:grouped.setdefault(r['group'],[]).append(ok[r['id']])
    output={}
    for split in sorted({r['split'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['split']==split];pairs={}
        for a,b in [('target8','graph_surface'),('target8','base8'),('static_binding','static_bag'),('late_source_reference','target8')]:
            dif=errors[a]-errors[b];draws=[]
            for fam in FAMILIES:
                ids=[i for i in ix if rows[i]['family']==fam];worlds=sorted({rows[i]['world'] for i in ids})
                means=np.array([np.mean([dif[i] for i in ids if rows[i]['world']==w]) for w in worlds])
                draws.append(means[rng.integers(len(means),size=(2000,len(means)))].mean(1))
            pairs[a+'-minus-'+b]=dict(mean=float(dif[ix].mean()),ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist())
        good=[i for i in ix if all(grouped[rows[i]['group']])]
        output[split]=dict(pairs=pairs,late_source_reference=float(error[ix].mean()),all_four_answers_correct_groups=len(good),
            correct_only={m:float(e[good].mean()) if good else None for m,e in errors.items()},
            correct_subset_note='Post-outcome subgroup, not randomized or an independent confirmatory estimand.')
    readout=json.loads((OUT/'readout_rows.json').read_text(encoding='utf-8'))
    # A diagnostic lower bound for the dual-ridge class, using test targets as an
    # oracle. This is never a deployable prediction or a proposed latent basis.
    with np.load(OUT/'discovery_targets.npz') as z:train_i=z['interaction'][:,36].astype(np.float64)
    im=train_i.mean(0)
    _,singular,vh=np.linalg.svd(train_i-im,full_matrices=False)
    tol=np.finfo(float).eps*max(train_i.shape)*singular[0]
    rank=int(np.sum(singular>tol));basis=vh[:rank]
    projection=(truth-im)@basis.T@basis+im
    lower=old_algorithm.relative(projection,truth)
    for split in output:
        ix=[i for i,r in enumerate(rows) if r['split']==split]
        output[split]['oracle_train_affine_span_floor']=float(lower[ix].mean())
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    counts={}
    for split in output:
        ix=[i for i,r in enumerate(rows) if r['split']==split];counts[split]={}
        for m,dd in readout['methods'].items():
            d={}
            for i in ix:
                token=tok.decode([dd['predicted_ids'][i]]);d[token]=d.get(token,0)+1
            counts[split][m]=d
    write(OUT/'post_confirmation_diagnostics.json',dict(created_utc=now(),source=snapshot(Path(__file__)),status='post-confirmation exploratory diagnostics',
        dual_ridge_capacity=dict(centered_training_output_rank=rank,output_dimension=2560,numerical_rank_threshold=float(tol),
            interpretation='All dual-ridge I predictions lie in the affine span of256 training targets. Oracle projection is a target-informed unattainable lower error bound, not a forecast, PCA backbone, or selected latent structure.'),
        late_reference=dict(method='Phase2752 source_product',alpha=alpha,training='Same old256 train groups; no refit on fresh data; target layer h00/h10/h01 REQUIRED',
            prior_summary_sha256=sha(OLD/'4B/prediction_summary.json'),source_sha256=sha(Path(old_algorithm.__file__))),splits=output,readout_prediction_counts=counts))
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':run()
