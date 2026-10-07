"""Frozen early-input forecasts: all coordinates, separate input-cost classes."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse, json, re, time
from pathlib import Path
import numpy as np
from phase2753_early_scope import ROOT, OLD, OUT, FAMILIES, write, sha, now, snapshot
import phase2752_predict_interaction as legacy

ALPHAS=(.001,.01,.1,1.,10.)
SPEC={
 'surface':['surface'], 'graph':['graph'], 'graph_surface':['graph','surface'],
 'static_bag':['graph','surface','bag','claim'],
 'static_binding':['graph','surface','bag','claim','query0','query1','out0','in0','out1','in1','edge','negative_edge','query_product'],
 'target4':['graph','surface','target4'], 'target8':['graph','surface','target8'],
 'target12':['graph','surface','target12'], 'target8_only':['target8'],
 'base8':['graph','surface','base8'],
 'four8':['graph','surface','base8','ds8','dn8','interaction8']}
METHODS=['zero','family_mean']+list(SPEC)+['native8']
SCOPES={m:('four_prefix_8' if m=='four8' else 'base_prefix_8' if m=='base8' else 'target_prefix_early' if m.startswith('target') or m=='native8' else 'static') for m in METHODS}

def data(root, discovery=False):
    """Load only train/validation for discovery; old test responses never enter RAM."""
    mat=json.loads((root/'material.json').read_text(encoding='utf-8'))
    byid={r['id']:r for r in mat['rows']}
    meta,parts=[],[]
    for path in sorted((root/'4B').glob('chunk_*.npz')):
        rr=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        ix=[i for i,r in enumerate(rr) if not discovery or byid[r['id']]['split'] in ('train','validation')]
        if ix:
            with np.load(path) as z:parts.append(z['hidden'][ix])
            meta.extend(rr[i] for i in ix)
    hh=np.concatenate(parts);del parts
    mapping={}
    for i,r in enumerate(meta):mapping.setdefault(r['group'],{})[byid[r['id']]['cond']]=i
    assert all(set(v)=={0,1,2,3} for v in mapping.values())
    index=np.array([[v[c] for c in range(4)] for v in mapping.values()])
    rows=[byid[meta[v[3]]['id']] for v in mapping.values()]
    return rows,hh[index],meta,mat

def embedding():
    import torch
    from safetensors import safe_open
    p=ROOT/'models/hf/qwen3-4b'
    idx=json.loads((p/'model.safetensors.index.json').read_text())
    with safe_open(p/idx['weight_map']['model.embed_tokens.weight'],framework='pt',device='cpu') as f:
        emb=f.get_tensor('model.embed_tokens.weight')
    return emb

def features(rows,h,mat,emb,tok):
    """No expected/truth/prediction/late state read here. Role spans come from input."""
    worlds={w['id']:w for w in mat['worlds']}
    b={k:[] for k in ['graph','surface','bag','claim','query0','query1','out0','in0','out1','in1','edge','negative_edge','query_product']}
    annotations=[]
    for r in rows:
        text=r['text']; w=worlds[r['world']]
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        assert enc['input_ids']==r['tokenization']['4B']['token_ids']
        ids=np.asarray(enc['input_ids']); offsets=enc['offset_mapping']
        ee=emb[ids.tolist()].float().numpy()
        def span(a,z):
            ix=[i for i,(x,y) in enumerate(offsets) if x<z and y>a]
            assert ix
            return ee[ix].mean(0)
        boundary=text.index(' the case that ')+len(' the case that ')
        end=text.index('.\n',boundary)
        clause=text[boundary:end]
        names=w['entities']; path=names[:r['depth']+1]
        present={n:span(m.start(),m.end()) for n in names if (m:=re.search(r'\b'+re.escape(n)+r'\b',text))}
        qnames=re.findall(r'\b(?:'+ '|'.join(map(re.escape,present))+r')\b',clause)
        assert len(qnames) in (1,2)
        q0=present[qnames[0]]
        q1=present[qnames[1]] if len(qnames)>1 else span(boundary+clause.index('was the ')+8,end)
        edges=list(zip(path[:-1],path[1:]))
        assert all(a in present and z in present for a,z in edges)
        zero=np.zeros_like(q0)
        def neighbor(n,direction):
            ns=[z if direction=='out' else a for a,z in edges if (a if direction=='out' else z)==n]
            return np.mean([present[x] for x in ns],axis=0) if ns else zero
        values=dict(bag=ee.mean(0),claim=span(boundary,end),query0=q0,query1=q1,
            out0=neighbor(qnames[0],'out'),in0=neighbor(qnames[0],'in'),
            out1=neighbor(qnames[1],'out') if len(qnames)>1 else zero,
            in1=neighbor(qnames[1],'in') if len(qnames)>1 else zero,
            edge=np.mean([present[a]-present[z] for a,z in edges],axis=0),
            negative_edge=present[path[1]]-present[names[4]] if r['family']=='category' else zero,
            query_product=q0*q1)
        family=np.array([r['family']==f for f in FAMILIES],dtype=float)
        # Metadata describe supplied graph and syntax, never relation truth or label.
        basic=np.array([r['role'],r['order'],r['depth']-2,int(r['wording'] in (1,5)),r['role']*r['order']],dtype=float)
        values['graph']=np.r_[family,basic,np.outer(family,basic).ravel()]
        length=len(ids)/100; pos=r['tokenization']['4B']['question_start']/100
        values['surface']=np.array([length,pos,length-pos,r['fact_count'],length**2,pos**2,length*pos])
        for k,v in values.items():b[k].append(v)
        annotations.append(dict(group=r['group'],query_entities=qnames,edges=edges,query_span=[boundary,end],
            negative_edge=[path[1],names[4]] if r['family']=='category' else None))
    b={k:np.asarray(v,dtype=np.float64) for k,v in b.items()}
    for cutoff in (4,8,12):b[f'target{cutoff}']=h[:,3,cutoff].astype(np.float64)
    b['base8']=h[:,0,8].astype(np.float64)
    b['ds8']=h[:,2,8].astype(np.float64)-b['base8']
    b['dn8']=h[:,1,8].astype(np.float64)-b['base8']
    b['interaction8']=h[:,3,8].astype(np.float64)-h[:,2,8]-h[:,1,8]+h[:,0,8]
    return b,annotations

def kernel_features(blocks,method,train=None,scale=None):
    if scale is None:
        scale={}
        for k in SPEC[method]:
            x=blocks[k][train]; mu=x.mean(0)
            # Each whole block has unit mean squared Euclidean length on train.
            sd=max(float(np.sqrt(np.mean(np.sum((x-mu)**2,axis=1)))),1e-10)
            scale[k]=(mu,sd)
    z=np.concatenate([(blocks[k]-scale[k][0])/scale[k][1] for k in SPEC[method]],axis=1)
    return z,scale

def weights(z,zt,alpha):
    n=len(zt)
    return np.full((len(z),n),1/n)+(z@zt.T)@np.linalg.solve(zt@zt.T+n*alpha*np.eye(n),np.eye(n)-np.ones((n,n))/n)

def native_fit(x,y,alpha):
    z=np.stack([x,x*x],axis=-1);mu=z.mean(0);sd=z.std(0);sd[sd<1e-10]=1
    zz=(z-mu)/sd; ym=y.mean(0)
    coef=np.linalg.solve(np.einsum('ndk,ndj->dkj',zz,zz)/len(x)+alpha*np.eye(2),np.einsum('ndk,nd->dk',zz,y-ym)[...,None]/len(x))[...,0]
    return dict(mu=mu,sd=sd,ym=ym,coef=coef)

def native_predict(x,fit):
    z=(np.stack([x,x*x],-1)-fit['mu'])/fit['sd']
    return np.einsum('ndk,dk->nd',z,fit['coef'])+fit['ym']

def selftest():
    rng=np.random.default_rng(9);x=rng.normal(size=(19,7)); y=rng.normal(size=(12,3)); x-=x[:12].mean(0)
    a=.1; w=weights(x,x[:12],a)
    coef=np.linalg.solve(x[:12].T@x[:12]+12*a*np.eye(7),x[:12].T@(y-y.mean(0)))
    assert np.allclose(w@y,x@coef+y.mean(0),atol=1e-10)
    return dict(dual_primal_ridge_max=float(np.max(abs(w@y-(x@coef+y.mean(0))))))

def discover():
    assert not (OUT/'selection.json').exists(),'Sealed discovery cannot be silently overwritten.'
    start=time.time()
    write(OUT/'algorithm_seal.json',dict(created_utc=now(),source=snapshot(Path(__file__)),methods=METHODS,spec=SPEC,scopes=SCOPES,alpha=ALPHAS,
        block_scaling='Train mean plus ONE train RMS Euclidean scale per feature block; unit total energy per block. All coordinates retained.',
        graph='Generator-known typed edge paths and actual query token spans. No automatic parser, no computed truth or answer features. Unused entity names excluded.',
        selection='Each method alpha by old validation E_I; primary global winner, static winner, and target-only early winner declared separately. Same-input h11 alpha by validation error relative to train-mean-centered true h11.',
        native8='Per-coordinate [h11_8, h11_8 squared], native coordinates only; no cross-coordinate mixing.',
        uncertainty='Family-stratified world bootstrap paired with family mean, 2000 draws, fixed syntax protocols.',
        selftest=selftest(),material_seal_sha256=sha(OUT/'material_seal.json')))
    rows,h,meta,mat=data(OLD,True)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    emb=embedding();blocks,annotations=features(rows,h,mat,emb,tok);del emb
    # Mutate all late states and labels, leaving the contractual early states fixed.
    sample=h[:2].copy();sample[:,:,13:]+=10000
    rr=[dict(r,expected='MUTATED') for r in rows[:2]]
    emb=embedding();changed,_=features(rr,sample,mat,emb,tok);del emb
    assert all(np.array_equal(changed[k],v[:2]) for k,v in blocks.items())
    write(OUT/'feature_contract_check.json',dict(late_state_and_label_mutation_invariant=True,maximum_boundary_read=12,early_prefixes=SCOPES,
        discovery_groups=len(rows),train_worlds=len({r['world'] for r in rows if r['split']=='train'}),validation_worlds=len({r['world'] for r in rows if r['split']=='validation'})))
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']); val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    interaction=h[:,3].astype(np.float64)-h[:,2]-h[:,1]+h[:,0]
    y=interaction[:,36]; state=h[:,3,36].astype(np.float64); center=state[train].mean(0)
    np.savez(OUT/'discovery_targets.npz',interaction=interaction[train].astype(np.float32),full=state[train].astype(np.float32),center=center,
        families=np.array([r['family'] for r in rows])[train])
    records={}; fitted={}; cache={}
    for method in METHODS:
        if method in SPEC:
            z,scale=kernel_features(blocks,method,train); cache[method]=(z,scale)
        scores=[]
        for a in (ALPHAS if method not in ('zero','family_mean') else [0.]):
            if method=='zero':ip=np.zeros_like(y); hp=np.broadcast_to(center,state.shape)
            elif method=='family_mean':
                ip=np.stack([y[[i for i in train if rows[i]['family']==r['family']]].mean(0) for r in rows])
                hp=np.stack([state[[i for i in train if rows[i]['family']==r['family']]].mean(0) for r in rows])
            elif method=='native8':
                ip=native_predict(blocks['target8'],native_fit(blocks['target8'][train],y[train],a))
                hp=native_predict(blocks['target8'],native_fit(blocks['target8'][train],state[train],a))
            else:
                ww=weights(z,z[train],a); ip=ww@y[train];hp=ww@state[train]
            scores.append(dict(alpha=a,interaction=float(legacy.relative(ip[val],y[val]).mean()),full_centered=float(legacy.relative(hp[val]-center,state[val]-center).mean())))
        ai=min(scores,key=lambda q:q['interaction'])['alpha']; ah=min(scores,key=lambda q:q['full_centered'])['alpha']
        records[method]=dict(alpha_interaction=ai,alpha_full=ah,validation=scores,scope=SCOPES[method])
        print(method,ai,min(q['interaction'] for q in scores),flush=True)
        if method in SPEC:
            z,scale=cache[method]
            payload={'zt':z[train]}
            for k,(mu,sd) in scale.items():payload['mu_'+k]=mu;payload['sd_'+k]=np.array(sd)
            fitted[method]=payload
        elif method=='native8':
            fitted[method]={f'{kind}_{k}':v for kind,tgt,a in [('i',y,ai),('h',state,ah)] for k,v in native_fit(blocks['target8'][train],tgt[train],a).items()}
    (OUT/'fits').mkdir(exist_ok=True)
    for m,payload in fitted.items():np.savez(OUT/'fits'/f'{m}.npz',**payload)
    def winner(candidates):return min(candidates,key=lambda m:min(s['interaction'] for s in records[m]['validation']))
    selected=dict(global_winner=winner(METHODS),static_winner=winner([m for m in METHODS if SCOPES[m]=='static']),
        target_early_winner=winner([m for m in METHODS if SCOPES[m]=='target_prefix_early']))
    write(OUT/'discovery_annotations.json',annotations)
    write(OUT/'selection.json',dict(created_utc=now(),records=records,selected=selected,no_refit=True,discovery_root=str(OLD.relative_to(ROOT)),
        elapsed_seconds=time.time()-start,code_sha256=sha(Path(__file__)),algorithm_seal_sha256=sha(OUT/'algorithm_seal.json'),
        fits={p.name:sha(p) for p in sorted((OUT/'fits').glob('*.npz'))},targets_sha256=sha(OUT/'discovery_targets.npz')))
    print(selected,flush=True)

def confirm():
    start=time.time();selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    assert selection['code_sha256']==sha(Path(__file__)),'Do not change the frozen algorithm after discovery.'
    for name,digest in selection['fits'].items():assert sha(OUT/'fits'/name)==digest
    assert sha(OUT/'discovery_targets.npz')==selection['targets_sha256']
    rows,h,meta,mat=data(OUT)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    emb=embedding();blocks,annotations=features(rows,h,mat,emb,tok);del emb
    with np.load(OUT/'discovery_targets.npz') as z:yt=z['interaction'].astype(np.float64);ht=z['full'].astype(np.float64);center=z['center'];families=z['families']
    truth=h[:,3].astype(np.float64)-h[:,2]-h[:,1]+h[:,0]
    state=h[:,3,36].astype(np.float64)
    metrics={}; ipreds={}; hpreds={}
    for method in METHODS:
        rec=selection['records'][method]
        if method=='zero':ip=np.zeros((len(rows),2560));hp=np.broadcast_to(center,state.shape);wi=None
        elif method=='family_mean':
            wi=np.stack([(families==r['family'])/np.sum(families==r['family']) for r in rows]);ip=wi@yt[:,36];hp=wi@ht
        elif method=='native8':
            with np.load(OUT/'fits'/f'{method}.npz') as z:fit=dict(z)
            ip=native_predict(blocks['target8'],{k:fit['i_'+k] for k in ('mu','sd','ym','coef')})
            hp=native_predict(blocks['target8'],{k:fit['h_'+k] for k in ('mu','sd','ym','coef')});wi=None
        else:
            with np.load(OUT/'fits'/f'{method}.npz') as z:
                scale={k:(z['mu_'+k],float(z['sd_'+k])) for k in SPEC[method]};zt=z['zt']
            zz,_=kernel_features(blocks,method,scale=scale)
            wi=weights(zz,zt,rec['alpha_interaction']);wh=weights(zz,zt,rec['alpha_full'])
            ip=wi@yt[:,36];hp=wh@ht
        profile=np.full((len(rows),38),np.nan)
        if wi is not None:
            for layer in range(38):profile[:,layer]=legacy.relative(wi@yt[:,layer],truth[:,layer])
        elif method=='zero':
            for layer in range(38):profile[:,layer]=legacy.relative(np.zeros_like(truth[:,layer]),truth[:,layer])
        # native8 only fits boundary36; do not imply it was fit at other boundaries.
        profile[:,36]=legacy.relative(ip,truth[:,36])
        metrics[method]=dict(interaction=profile,full_centered=legacy.relative(hp-center,state-center),full_absolute=legacy.relative(hp,state))
        ipreds[method]=ip.astype(np.float32);hpreds[method]=hp.astype(np.float32)
        print('confirmed',method,flush=True)
    np.savez(OUT/'confirmation_predictions.npz',**{'i_'+m:v for m,v in ipreds.items()},**{'h_'+m:v for m,v in hpreds.items()},true_h=state.astype(np.float32),true_i=truth[:,36].astype(np.float32))
    np.savez(OUT/'confirmation_metrics.npz',**{m+'_'+k:v for m,d in metrics.items() for k,v in d.items()})
    write(OUT/'confirmation_groups.json',rows);write(OUT/'confirmation_annotations.json',annotations)
    rng=np.random.default_rng(2753002); splits={}
    for split in sorted({r['split'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['split']==split]; report={}
        for method in METHODS:
            d=metrics[method]; dif=d['interaction'][:,36]-metrics['family_mean']['interaction'][:,36]
            draws=[];family_results={}
            for family in FAMILIES:
                jj=[i for i in ix if rows[i]['family']==family]; worlds=sorted({rows[i]['world'] for i in jj})
                means=np.array([np.nanmean([dif[i] for i in jj if rows[i]['world']==w]) for w in worlds])
                draws.append(means[rng.integers(len(means),size=(2000,len(means)))].mean(1))
                family_results[family]=float(np.nanmean(d['interaction'][jj,36]))
            report[method]=dict(interaction=float(np.nanmean(d['interaction'][ix,36])),full_centered=float(np.nanmean(d['full_centered'][ix])),
                full_absolute=float(np.nanmean(d['full_absolute'][ix])),paired_family_mean_difference=float(np.nanmean(dif[ix])),
                paired_world_ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist(),families=family_results)
        splits[split]=dict(worlds=len({rows[i]['world'] for i in ix}),groups=len(ix),methods=report)
    source={r['id']:r for r in mat['rows']}; behaviors={}
    for split in splits:
        mm=[r for r in meta if r['split']==split]
        behaviors[split]=dict(prompts=len(mm),first_token_accuracy=float(np.mean([r['prediction_text'].strip().lower()==source[r['id']]['expected'] for r in mm])),
            target11_accuracy=float(np.mean([r['prediction_text'].strip().lower()==source[r['id']]['expected'] for r in mm if source[r['id']]['cond']==3])))
    write(OUT/'confirmation_summary.json',dict(created_utc=now(),selection=selection['selected'],splits=splits,behavior=behaviors,
        elapsed_seconds=time.time()-start,worlds=len(mat['worlds']),prompts=len(mat['rows']),selection_sha256=sha(OUT/'selection.json'),
        limitations=['Known synthetic graph families; static graph annotations supplied.','Early h11 sees the doubly conditioned input but no later states.','Four-prefix8 and target-prefix12 have different compute scopes.','Fixed discovery and syntax protocols; world bootstrap excludes training/template uncertainty.']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['discover','confirm']);a=p.parse_args()
    discover() if a.mode=='discover' else confirm()
