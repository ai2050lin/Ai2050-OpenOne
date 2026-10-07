"""Train/validate/test native-coordinate interaction predictors without target leakage."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import argparse
import json
import time
from pathlib import Path
import numpy as np
from phase2752_context_interaction import OUT, ROOT, FAMILIES, now, sha, write, snapshot

METHODS = ('zero', 'family_mean', 'surface', 'graph_only', 'graph_surface', 'base_coordinate', 'source_product', 'hybrid')
ALPHAS = (0.01, 0.1, 1., 10.)
SPLITS = ('train', 'validation', 'entity', 'wording', 'joint_wording', 'role_order', 'depth')


def seal():
    path = OUT/'predictor_pre_fit_seal.json'
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['source']['sha256'] == sha(Path(__file__)), 'Changing fitted algorithm requires a named revision.'
        return
    write(path, dict(created_utc=now(), source=snapshot(Path(__file__)), methods=METHODS, alpha=ALPHAS,
        primary_layer='final norm output; raw final residual and all intermediate fields secondary',
        coordinate_inputs=['h00', 'h10-h00', 'h01-h00', '(h10-h00)*(h01-h00)', '(h10-h00)^2', '(h01-h00)^2'],
        fitting='Per-coordinate mean and scale learned on train only. Ridge penalty alpha on mean-square objective; intercept unpenalized.',
        graph='Input-family/role/order/depth/question syntax/context and specified pairwise products. No truth, expected, world identity or tokenizer IDs.',
        surfaces='Baseline token count, question offset, query span, fact count and squared lengths; no condition target state.',
        selection='Each method alpha chosen by validation mean interaction relative L2 at final norm; overall method chosen by same validation metric, ties METHODS order.',
        no_post_selection_refit=True, bootstrap='2000 world resamples within each split, mean of world-level paired error differences; percentile intervals; not adjusted hypothesis tests.'))


def load(side):
    mat = json.loads((OUT/'material.json').read_text(encoding='utf-8'))
    source = {r['id']:r for r in mat['rows']}
    rows, parts = [], []
    for path in sorted((OUT/side).glob('chunk_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            parts.append(z['hidden'])
        rows.extend(json.loads(path.with_suffix('.json').read_text(encoding='utf-8')))
    done = json.loads((OUT/side/'capture_done.json').read_text(encoding='utf-8'))
    assert len(rows) == done['count']
    h = np.concatenate(parts)
    del parts
    indices = {}
    for i,row in enumerate(rows):
        indices.setdefault(row['group'], {})[source[row['id']]['cond']] = i
    assert all(set(v) == {0,1,2,3} for v in indices.values())
    groups = [source[rows[ids[0]]['id']] for ids in indices.values()]
    fields = h[np.array([[d[c] for c in range(4)] for d in indices.values()])]
    return groups, fields, rows, done, source


def features(rows, side):
    graph, surface = [], []
    for r in rows:
        f = np.array([int(r['family']==x) for x in FAMILIES], dtype=float)
        role, order, depth, wording = r['role'], r['order'], r['depth']-2, r['wording']
        modal = int(wording in (1,3))
        passive = int(wording==1)
        listed = int(wording==2)
        contextual = int(wording in (2,3))
        basic = np.array([role, order, depth, modal, passive, listed, contextual, role*order], dtype=float)
        graph.append(np.concatenate([f, basic, np.outer(f, basic[:5]).ravel()]))
        tm = r['tokenization'][side]
        length, pos = tm['length']/100, tm['question_start']/100
        surface.append([length, pos, length-pos, r['fact_count'], length**2, pos**2, length*pos])
    return np.asarray(graph), np.asarray(surface)


def influence(x, train, alpha):
    """Small-feature multi-output ridge; an affine train-target influence matrix."""
    mu, sd = x[train].mean(0), x[train].std(0)
    sd[sd < 1e-8] = 1.
    z = (x-mu)/sd
    xt = z[train]
    n = len(train)
    # Centered training rows sum to zero, so y centering cancels in this product.
    return np.full((len(x),n),1/n) + z @ np.linalg.solve(xt.T @ xt/n + alpha*np.eye(x.shape[1]), xt.T/n)


def coordinate_input(h, base=False):
    # Critical input boundary: this function never accesses h[:,3].
    b = h[:,0].astype(np.float64)
    if base:
        return b[...,None]
    s = h[:,2].astype(np.float64)-b
    n = h[:,1].astype(np.float64)-b
    return np.stack([b,s,n,s*n,s*s,n*n],-1)


def coordinate_fit_predict(z, target_train, train, alpha):
    mu, sd = z[train].mean(0), z[train].std(0)
    sd[sd < 1e-8] = 1.
    zz = (z-mu)/sd
    xt = zz[train]
    ym = target_train.mean(0)
    yt = target_train-ym
    n = len(train)
    cov = np.einsum('ndk,ndj->dkj',xt,xt)/n
    rhs = np.einsum('ndk,nd->dk',xt,yt)/n
    coef = np.linalg.solve(cov+alpha*np.eye(xt.shape[-1])[None],rhs[...,None])[...,0]
    return (np.einsum('ndk,dk->nd',zz,coef)+ym).astype(np.float32)


def interaction(h):
    return h[:,3].astype(np.float64)-h[:,2]-h[:,1]+h[:,0]


def relative(pred, target):
    den = np.linalg.norm(target, axis=-1)
    result = np.full(len(target), np.nan)
    np.divide(np.linalg.norm(pred-target,axis=-1), den, out=result, where=den>1e-10)
    return result


def finite_mean(a):
    a = np.asarray(a)
    return float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else None


def predict(method, alpha, h, target_train, train, rows, matrices):
    if method == 'zero':
        return np.zeros((len(h),h.shape[-1]),dtype=np.float32)
    if method == 'family_mean':
        means = {f:target_train[[rows[t]['family']==f for t in train]].mean(0) for f in FAMILIES}
        return np.stack([means[r['family']] for r in rows]).astype(np.float32)
    if method in ('surface','graph_only','graph_surface'):
        return (matrices[(method,alpha)] @ target_train).astype(np.float32)
    if method in ('base_coordinate','source_product'):
        return coordinate_fit_predict(coordinate_input(h,method=='base_coordinate'),target_train,train,alpha)
    gp = (matrices[('graph_surface',alpha)] @ target_train).astype(np.float32)
    return gp + coordinate_fit_predict(coordinate_input(h),target_train-gp[train],train,alpha)


def selftest():
    rng = np.random.default_rng(14)
    h = rng.normal(size=(18,4,7))
    changed = h.copy()
    changed[:,3] += 1000
    assert np.array_equal(coordinate_input(h),coordinate_input(changed))
    target = interaction(h)
    additive = h[:,2]+h[:,1]-h[:,0]
    assert np.allclose(additive+target,h[:,3])
    # Native coordinate implementation matches independently solved explicit ridge.
    z = coordinate_input(h)
    train = np.arange(12)
    pred = coordinate_fit_predict(z,target[train],train,.1)
    x = z[:,3]
    x = (x-x[train].mean(0))/x[train].std(0)
    y = target[train,3]
    coef = np.linalg.solve(x[train].T@x[train]/12 + .1*np.eye(6), x[train].T@(y-y.mean())/12)
    assert np.allclose(pred[:,3],x@coef+y.mean(),atol=1e-6)
    return dict(target_mutation_does_not_change_source_features=True, interaction_reconstruction=True, independent_ridge_reference=True)


def summarize(rows, metrics, choices, done):
    layer = done['last_norm_index']
    by_split = {}
    rng = np.random.default_rng(2752002)
    for split in SPLITS:
        ix = [i for i,r in enumerate(rows) if r['split']==split]
        worlds = sorted({rows[i]['world'] for i in ix})
        stats = {}
        for method in METHODS:
            a = metrics[method]['interaction'][ix,layer]
            b = metrics[method]['combined'][ix,layer]
            stats[method] = dict(interaction_mean=finite_mean(a), combined_change_mean=finite_mean(b),
                interaction_median=float(np.nanmedian(a)) if np.isfinite(a).any() else None)
        # All candidates are compared with both simple baselines, not just the winner.
        paired = {}
        for method in METHODS:
            for baseline in ('zero','family_mean','surface'):
                if method == baseline:
                    continue
                diff = metrics[method]['interaction'][:,layer]-metrics[baseline]['interaction'][:,layer]
                blocks = np.array([np.nanmean([diff[i] for i in ix if rows[i]['world']==w]) for w in worlds])
                draws = np.mean(blocks[rng.integers(0,len(blocks),(2000,len(blocks)))],axis=1)
                paired[f'{method}-minus-{baseline}'] = dict(mean=float(blocks.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
        by_family = {f:{m:dict(interaction_mean=finite_mean(metrics[m]['interaction'][[i for i in ix if rows[i]['family']==f],layer]),
                               combined_change_mean=finite_mean(metrics[m]['combined'][[i for i in ix if rows[i]['family']==f],layer]))
                        for m in METHODS} for f in FAMILIES}
        by_split[split] = dict(groups=len(ix),worlds=len(worlds),methods=stats,paired_world_bootstrap=paired,families=by_family)
    # Choice is made solely from validation, with a deterministic predeclared tie order.
    winner = min(METHODS,key=lambda m:by_split['validation']['methods'][m]['interaction_mean'])
    return dict(created_utc=now(),status='completed_bounded_experiment',alpha_choices=choices,selected_method=winner,
                selection_scope='validation only, no test selection or refitting',primary_layer=layer,splits=by_split,
                evidence_level='heldout prediction under a fixed synthetic grammar; not mechanistic identification or universal theory')


def run(side):
    seal()
    start = time.time()
    dest = OUT/side
    assert not (dest/'prediction_summary.json').exists(), 'Retain completed prediction; use a named revision for changes.'
    rows, h, behavior, done, source = load(side)
    train = np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    val = np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    graph,surface = features(rows,side)
    matrices = {(m,a):influence(x,train,a) for m,x in [('surface',surface),('graph_only',graph),('graph_surface',np.c_[graph,surface])] for a in ALPHAS}
    last = done['last_norm_index']
    final_target = interaction(h[:,:,last])
    choices,validation = {},{}
    for method in METHODS:
        vals = {}
        for alpha in ([1.] if method in ('zero','family_mean') else ALPHAS):
            pred = predict(method,alpha,h[:,:,last],final_target[train],train,rows,matrices)
            vals[str(alpha)] = finite_mean(relative(pred,final_target)[val])
        choices[method] = float(min(vals,key=vals.get))
        validation[method] = vals
    # Selection persisted before any test metrics are computed.
    write(dest/'validation_selection.json',dict(created_utc=now(), choices=choices,validation_errors=validation,
        predictor_seal_sha256=sha(OUT/'predictor_pre_fit_seal.json'),selftest=selftest()))
    n,_,layers,d = h.shape
    metrics = {m:{k:np.empty((n,layers),dtype=np.float32) for k in ('interaction','combined')} for m in METHODS}
    fields = {m:np.empty((layers,d),dtype=np.float32) for m in METHODS}
    observed = np.empty((layers,d),dtype=np.float32)
    magnitudes = np.empty((n,layers),dtype=np.float32)
    final_predictions = {}
    test_ix = np.array([i for i,r in enumerate(rows) if r['split'] not in ('train','validation')])
    for layer in range(layers):
        hl = h[:,:,layer]
        target = interaction(hl)
        change = hl[:,3].astype(np.float64)-hl[:,0]
        magnitudes[:,layer] = np.linalg.norm(target,axis=-1)
        observed[layer] = target[test_ix].mean(0)
        for method in METHODS:
            pred = predict(method,choices[method],hl,target[train],train,rows,matrices)
            metrics[method]['interaction'][:,layer] = relative(pred,target)
            # prediction error is identical; denominator is total h11-h00 change.
            metrics[method]['combined'][:,layer] = relative(pred-target+change,change)
            fields[method][layer] = pred[test_ix].mean(0)
            if layer == last:
                final_predictions[method] = pred
        if layer % 8 == 0:
            print(f'{side} predicted layer {layer}/{layers-1} elapsed={time.time()-start:.1f}s',flush=True)
    summary = summarize(rows,metrics,choices,done)
    summary['execution_seconds'] = time.time()-start
    summary['first_token_behavior'] = {split:dict(n=sum(r['split']==split for r in behavior),
        exact_accuracy=np.mean([r['prediction_text'].strip().lower()==source[r['id']]['expected'] for r in behavior if r['split']==split]).item(),
        forced_yesno_accuracy=np.mean([max(r['yesno_logmass'],key=r['yesno_logmass'].get)==source[r['id']]['expected'] for r in behavior if r['split']==split]).item(),
        mean_yesno_mass=np.mean([sum(np.exp(v) for v in r['yesno_logmass'].values()) for r in behavior if r['split']==split]).item()) for split in SPLITS}
    summary['zero_interaction_groups_by_layer'] = (magnitudes<=1e-10).sum(0).tolist()
    write(dest/'prediction_summary.json',summary)
    write(dest/'prediction_groups.json',[{k:r[k] for k in ('id','group','world','family','split','wording','role','order','depth')} for r in rows])
    np.savez(dest/'prediction_metrics.npz',**{f'{m}_{k}':v for m,mm in metrics.items() for k,v in mm.items()},interaction_norm=magnitudes)
    np.savez(dest/'predicted_final_interactions.npz',observed=final_target.astype(np.float32),**final_predictions)
    np.savez(dest/'full_coordinate_fields.npz',observed=observed,**fields)
    write(dest/'field_metadata.json',dict(native_coordinate_order=True,sort=False,scope='Test-group mean interaction by boundary and coordinate; raw arrays retained without selection',
        boundaries=done['boundary'],shape=[layers,d],aggregation='All test groups equally weighted (wording test has more worlds); split/family metrics available separately',
        zero_interaction_embedding=True))
    print(json.dumps(dict(model=side,selected=summary['selected_method'],seconds=summary['execution_seconds'],test={s:summary['splits'][s]['methods'][summary['selected_method']] for s in SPLITS[2:]}),indent=2),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=['seal','run','selftest'])
    p.add_argument('--model',default='4B',choices=['4B','14B'])
    a = p.parse_args()
    if a.mode=='seal':seal()
    elif a.mode=='selftest':print(selftest())
    else:run(a.model)
