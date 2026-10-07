"""Frozen early semantic decoding and future model-margin forecasts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8');os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,json,time
from pathlib import Path
import numpy as np
from phase2754_relation_stability import ROOT,OUT,FAMILIES,CUTS,decode,write,sha,now,snapshot

ALPHAS=(.001,.01,.1,1.,10.)
METHODS=['surface']+[f'{kind}{c}' for c in CUTS for kind in ('end','slots')]+['finalnorm']

def load(stage):
    source={r['id']:r for r in json.loads((OUT/'material.json').read_text(encoding='utf-8'))['rows']}
    hs=[];qs=[];metadata=[]
    for p in sorted((OUT/'4B'/stage).glob('chunk_*.npz')):
        with np.load(p,allow_pickle=False) as z:
            hs.append(decode(z['hidden'][:,list(CUTS)+[36]]));qs.append(decode(z['query_slots']))
        metadata.extend(json.loads(p.with_suffix('.json').read_text(encoding='utf-8')))
    return [source[r['id']] for r in metadata],np.concatenate(hs),np.concatenate(qs),metadata

def input_blocks(rows,h,q,method):
    # No expected,truth,predicate,orientation,late margin,or world ID is accessed.
    if method=='surface':
        xx=[]
        for r in rows:
            tm=r['tokenization']['4B'];a,b=tm['query_source_target_last_tokens']
            xx.append([*[float(r['family']==f) for f in FAMILIES],r['depth'],tm['length']/100,(a-b)/100,
                int('It is not the case' in r['text']),int(r['template'] in (1,3))])
        return [np.asarray(xx,dtype=float)]
    if method=='finalnorm':return [h[:,-1].astype(float)]
    cut=int(''.join(c for c in method if c.isdigit()));idx=CUTS.index(cut)
    blocks=[h[:,idx].astype(float)]
    if method.startswith('slots'):
        x=q[:,idx].astype(float);blocks.extend([x[:,0]-x[:,1],x[:,0]+x[:,1]])
    return blocks

def standardize(blocks,train=None,parameters=None):
    if parameters is None:
        parameters=[]
        for x in blocks:
            mu=x[train].mean(0);sd=max(float(np.sqrt(np.mean(np.sum((x[train]-mu)**2,axis=1)))),1e-10)
            parameters.append((mu,sd))
    return np.concatenate([(x-m)/s for x,(m,s) in zip(blocks,parameters)],axis=1),parameters

def ridge_path(z,y,train):
    xt=z[train];ym=y[train].mean(0);yy=y[train]-ym
    # All native coordinates retained. Eigendecomposition only solves a ridge system.
    ev,u=np.linalg.eigh(xt@xt.T);ev=np.maximum(ev,0);uy=u.T@yy
    for a in ALPHAS:
        coef=xt.T@(u@(uy/(ev[:,None]+len(train)*a)))
        yield a,coef,ym,z@coef+ym

def accuracy(p,y):return float(np.mean((p>=0)==(y>0)))

def discover():
    assert not (OUT/'selection.json').exists(),'Discovery already frozen.'
    start=time.time()
    write(OUT/'probe_algorithm_seal.json',dict(created_utc=now(),source=snapshot(Path(__file__)),methods=METHODS,alpha=ALPHAS,
        inputs='End state or end plus two known semantic query-slot token states at4/8/12/16. The slot annotations are supplied, not learned language parsing.',
        surface_control='Family,depth,length,semantic-slot position difference,observed negation marker,and passive syntax. Added explicit bias controls before fitting; no theta/rho/labels.',
        outputs=['predicate theta*rho','statement truth theta*rho*eta','future native canonical yes-minus-no margin'],
        selection='Per-target alpha by validation accuracy (predicate,truth) or MSE(margin). Overall early semantic and margin winners chosen separately. Finalnorm is descriptive decoding ceiling, excluded from early selection.',
        baseline='A token-bag-only rule cannot distinguish balanced truth within each matched set, as verified for both tokenizers.',
        fits='Train means/scales and intercept; no post-selection refit; no PCA/TopK. Ridge eigensolver retains all eigenvectors.',
        tolerance='Stable directional/ranking/decoding effects retained without requiring perfect answers or exact vector equality. Errors stay in the primary population.'))
    rows,h,q,meta=load('discovery');train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    y=np.array([[r['predicate_sign'],r['truth_sign'],m['margin']] for r,m in zip(rows,meta)],dtype=float)
    records={};(OUT/'probe_fits').mkdir(exist_ok=True)
    for method in METHODS:
        blocks=input_blocks(rows,h,q,method);z,params=standardize(blocks,train)
        changed=[dict(r,expected='MUTATED',truth_sign=999,predicate_sign=999,theta=999,rho=999) for r in rows[:2]]
        h2=h[:2].copy();q2=q[:2].copy()
        if method not in ('surface','finalnorm'):
            ci=CUTS.index(int(''.join(c for c in method if c.isdigit())));h2[:,ci+1:]+=10000;q2[:,ci+1:]+=10000
        if method=='surface':h2+=10000;q2+=10000
        mutated=input_blocks(changed,h2,q2,method)
        assert all(np.array_equal(a[:2],b) for a,b in zip(blocks,mutated))
        candidates=[];saved={}
        for a,coef,ym,pred in ridge_path(z,y,train):
            scores=[accuracy(pred[val,0],y[val,0]),accuracy(pred[val,1],y[val,1]),float(np.mean((pred[val,2]-y[val,2])**2))]
            candidates.append(dict(alpha=a,predicate_accuracy=scores[0],truth_accuracy=scores[1],margin_mse=scores[2]));saved[a]=(coef,ym)
        choices=[max(candidates,key=lambda s:s['predicate_accuracy'])['alpha'],max(candidates,key=lambda s:s['truth_accuracy'])['alpha'],min(candidates,key=lambda s:s['margin_mse'])['alpha']]
        coef=np.stack([saved[a][0][:,j] for j,a in enumerate(choices)],axis=1);ym=saved[choices[0]][1]
        payload=dict(coef=coef,ym=ym)
        for j,(mu,sd) in enumerate(params):payload[f'mu{j}']=mu;payload[f'sd{j}']=np.array(sd)
        np.savez(OUT/'probe_fits'/f'{method}.npz',**payload)
        pred=z@coef+ym
        records[method]=dict(alpha_by_target=choices,validation=candidates,predicate_accuracy=accuracy(pred[val,0],y[val,0]),truth_accuracy=accuracy(pred[val,1],y[val,1]),margin_mse=float(np.mean((pred[val,2]-y[val,2])**2)),input_invariance_check=True)
        print(method,records[method]['truth_accuracy'],records[method]['margin_mse'],flush=True)
    eligible=[m for m in METHODS if m!='finalnorm']
    selected=dict(semantic=max(eligible,key=lambda m:records[m]['truth_accuracy']),margin=min(eligible,key=lambda m:records[m]['margin_mse']))
    write(OUT/'selection.json',dict(created_utc=now(),records=records,selected=selected,train_worlds=96,validation_worlds=32,train_rows=len(train),validation_rows=len(val),
        elapsed_seconds=time.time()-start,source_sha256=sha(Path(__file__)),algorithm_seal_sha256=sha(OUT/'probe_algorithm_seal.json'),
        fits={p.name:sha(p) for p in sorted((OUT/'probe_fits').glob('*.npz'))},no_refit=True))
    print(selected,flush=True)

def confirm():
    start=time.time();sel=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));assert sel['source_sha256']==sha(Path(__file__))
    rows,h,q,meta=load('confirmation');y=np.array([[r['predicate_sign'],r['truth_sign'],m['margin']] for r,m in zip(rows,meta)],dtype=float)
    predictions={}
    for method in METHODS:
        p=OUT/'probe_fits'/f'{method}.npz';assert sha(p)==sel['fits'][p.name]
        blocks=input_blocks(rows,h,q,method)
        with np.load(p) as z:
            params=[(z[f'mu{j}'],float(z[f'sd{j}'])) for j in range(len(blocks))];xx,_=standardize(blocks,parameters=params);pred=xx@z['coef']+z['ym']
        predictions[method]=pred
    np.savez(OUT/'probe_predictions.npz',truth=y,**predictions);write(OUT/'probe_rows.json',[dict(id=r['id'],world=r['world'],group=r['group'],family=r['family'],split=r['split'],cell=r['cell'],eta=r['eta']) for r in rows])
    rng=np.random.default_rng(2754002);report={}
    for split in ('entity','surface','depth'):
        ix=np.array([i for i,r in enumerate(rows) if r['split']==split]);dd={}
        for method,p in predictions.items():
            correct=(p[:,1]>=0)==(y[:,1]>0);draws=[]
            for fam in FAMILIES:
                worlds=sorted({rows[i]['world'] for i in ix if rows[i]['family']==fam});means=np.array([correct[[i for i in ix if rows[i]['world']==w]].mean() for w in worlds])
                draws.append(means[rng.integers(len(means),size=(2000,len(means)))].mean(1))
            byeta={str(eta):accuracy(p[[i for i in ix if rows[i]['eta']==eta],1],y[[i for i in ix if rows[i]['eta']==eta],1]) for eta in (-1,1)}
            nativeok=np.array([m['prediction_text'].strip().lower()==r['expected'] for r,m in zip(rows,meta)])
            bad=ix[~nativeok[ix]];good=ix[nativeok[ix]]
            dd[method]=dict(predicate_accuracy=accuracy(p[ix,0],y[ix,0]),truth_accuracy=float(correct[ix].mean()),truth_world_ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist(),
                margin_mse=float(np.mean((p[ix,2]-y[ix,2])**2)),margin_sign_fidelity=float(np.mean((p[ix,2]>=0)==(y[ix,2]>=0))),by_negation=byeta,
                native_wrong_rows=len(bad),truth_on_native_wrong=float(correct[bad].mean()) if len(bad) else None,truth_on_native_correct=float(correct[good].mean()) if len(good) else None,
                by_family={f:accuracy(p[[i for i in ix if rows[i]['family']==f],1],y[[i for i in ix if rows[i]['family']==f],1]) for f in FAMILIES})
        report[split]=dd
    write(OUT/'probe_confirmation.json',dict(created_utc=now(),selected=sel['selected'],splits=report,elapsed_seconds=time.time()-start,
        scope='Frozen early decoders and margin forecasts; finalnorm is descriptive. Wrong/correct-output subgroups are post-outcome diagnostics, not causal populations.',selection_sha256=sha(OUT/'selection.json')))
    print(json.dumps({s:{m:report[s][m]['truth_accuracy'] for m in [sel['selected']['semantic'],'surface','finalnorm']} for s in report},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['discover','confirm']);a=p.parse_args();discover() if a.mode=='discover' else confirm()
