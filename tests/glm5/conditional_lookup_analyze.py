"""Frozen natural CLT analysis. Readouts diagnose information, never causal use.

Discovery alone selects indices/regularization/heads. Confirmation uses saved
arrays verbatim. Parameter surrogate uses original gate/up weights and only
32 normalized-input coordinates, with omitted coordinates fixed at discovery
means. This is a local approximation, not a closed language mechanism.
"""
import os,sys,json,gc,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import source_binding_design as D
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
SEED=202611170501;ALPHAS=[.01,.1,1,10,100]

def save(name,obj):
    p=OUT/name;assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def decode(a):return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a

class Data:
    def __init__(self,split):
        self.split=split;self.collection=D.read(OUT/f'{split}-collection.json')
        assert self.collection['technical_pass'],'technical gate failed'
        assert self.collection['behavior_pass'],'behavior gate failed'
        self.rows=D.read(OUT/f'{split}-materials.json')
        self.paths=sorted(self.collection['shards'])
        for name,h in self.collection['shards'].items():assert D.sha(OUT/name)==h
        self.keep=np.array([i for i,r in enumerate(self.rows) if r['late'] and r['b']==0])
        self.rr=[self.rows[i] for i in self.keep]
        self.y=np.array([[r['r0'],r['r1']] for r in self.rr],dtype=int)
        self.fam=np.array([r['family'] for r in self.rr])
        self.folds=np.searchsorted(np.unique(self.fam),self.fam)%5
    def get(self,key,only=True):
        chunks=[]
        for p in self.paths:
            with np.load(OUT/p) as z:chunks.append(z[key])
        a=np.concatenate(chunks);return decode(a[self.keep] if only else a)

def rank(x,y,k):
    xx=x-x.mean(0);yy=y-y.mean(0)
    cov=xx.T@yy;den=np.maximum((xx*xx).sum(0)[:,None]*(yy*yy).sum(0)[None,:],1e-20)
    score=(cov*cov/den).sum(1)
    return np.argsort(-score,kind='stable')[:k]

def fit(x,y,idx,alpha):
    xx=x[:,idx].astype(np.float64);mu=xx.mean(0);sd=xx.std(0);sd[sd<1e-6]=1
    z=(xx-mu)/sd;ym=y.mean(0)
    w=np.linalg.solve(z.T@z+alpha*np.eye(len(idx)),z.T@(y-ym))
    return dict(idx=np.asarray(idx),mu=mu,sd=sd,w=w,ym=ym)

def pred(m,x):return ((x[:,m['idx']]-m['mu'])/m['sd'])@m['w']+m['ym']
def metric(p,y):
    ok=(p>=.5)==y
    return dict(bit0=float(ok[:,0].mean()),bit1=float(ok[:,1].mean()),joint=float(ok.all(1).mean()))
def correct(p,y):return ((p>=.5)==y).all(1)

def cv(x,y,folds,k,seed=None):
    # Coordinate choice is trained inside each family fold, never on held-out rows.
    fixed=np.random.default_rng(seed).choice(x.shape[1],min(k,x.shape[1]),replace=False) if seed is not None else None
    prepared=[]
    for f in range(5):
        tr=folds!=f;te=~tr;idx=rank(x[tr],y[tr],k) if fixed is None else fixed
        prepared.append((tr,te,idx))
    options=[]
    for alpha in ALPHAS:
        p=np.empty(y.shape,dtype=float)
        for tr,te,idx in prepared:p[te]=pred(fit(x[tr],y[tr],idx,alpha),x[te])
        options.append((metric(p,y)['joint'],alpha,p))
    _,alpha,p=max(options,key=lambda z:(z[0],-z[1]))
    idx=rank(x,y,k) if fixed is None else fixed
    return fit(x,y,idx,alpha),p,dict(alpha=alpha,k=k,seed=seed,**metric(p,y))

def putmodel(arrays,prefix,m):
    for k,v in m.items():arrays[prefix+'_'+k]=v
def getmodel(z,prefix):return {k:z[prefix+'_'+k] for k in ('idx','mu','sd','w','ym')}

def nuisance(rows,vocab):
    # Family-level nuisance covariates; none explicitly computes row order.
    x=np.zeros((len(rows),4+4*len(vocab)),dtype=np.float32);ix={v:i for i,v in enumerate(vocab)}
    for i,r in enumerate(rows):
        x[i,:4]=[len(r['tokens']),r['entity_pos'],len(r['names'][0]),r['titles'][0]=='Orion']
        for s,v in enumerate(r['choice_ids']):
            if v in ix:x[i,4+s*len(vocab)+ix[v]]=1
    return x

def interval(d,fam):
    vals=np.array([np.mean(d[fam==f]) for f in np.unique(fam)])
    rng=np.random.default_rng(SEED);means=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(1)
    return dict(mean=float(vals.mean()),ci975=np.quantile(means,[.0125,.9875]).tolist(),families=len(vals))

def source(data,head=None):
    rows=data.rows;gold=np.array([r['target_record'] for r in rows]);b=np.array([r['b'] for r in rows])
    late=np.array([r['late'] for r in rows],dtype=bool);allstats=[];preds={}
    for layer in range(20,31):
        mass=data.get(f'mass{layer}',False)[:,-1];write=data.get(f'write{layer}',False)[:,-1]
        for h in range(mass.shape[1]):
            p=mass[:,h,:4].argmax(-1);preds[layer,h]=p
            record=mass[:,h,:4];selected=record[np.arange(len(rows)),gold]
            # Actual same-head native sources. This sees measured attention and is NOT a future predictor.
            stats=dict(layer=layer,head=h,late_slot_acc=float((p[late]==gold[late]).mean()),early_slot_acc=float((p[~late]==gold[~late]).mean()),record_mass=float(record.mean(0).sum()),entity_mass=float(mass[:,h,4].mean()),gold_slot_mass=float(selected.mean()))
            margins=write[:,h,:].copy()
            # Raw unembedding contrast of each source's OV write, no final normalization or causal attribution.
            target=np.take_along_axis(margins,gold[:,None,None],axis=2).squeeze(2)
            contrast=(margins.sum(-1)-target)/3
            delta=target-contrast
            stats['local_raw_gold_contrast_by_source']=delta.mean(0).tolist()
            # Within same (family,r0,r1,late) b-pair, how often source argmax follows both requests?
            pairs={}
            for i,r in enumerate(rows):pairs.setdefault((r['family'],r['r0'],r['r1'],r['late']),[]).append(i)
            eligible=[v for key,v in pairs.items() if key[-1] and len(v)==2]
            stats['late_both_requests_correct']=float(np.mean([np.all(p[ii]==gold[ii]) for ii in eligible]))
            allstats.append(stats)
    if head is None:head=max(allstats,key=lambda s:(s['late_slot_acc'],-s['layer'],-s['head']))
    chosen=next(s for s in allstats if s['layer']==head['layer'] and s['head']==head['head'])
    return dict(chosen=chosen,all_heads=allstats,scope='observed QK routing and unnormalized local OV accounting; not causal necessity and not a compressed mechanism prediction')

def parameter_fit(data,key,units):
    import torch
    from source_binding_runtime import Runtime
    rt=Runtime('qwen14');b=int(key[1:]);x=data.get('z'+str(b));a=data.get(key)[:,units]
    weights={}
    for name in ('gate_proj','up_proj'):
        full=rt.weight(f'model.layers.{b}.mlp.{name}.weight','cpu');weights[name]=full[units].float().numpy().copy();del full
    full=rt.weight(f'model.layers.{b}.mlp.down_proj.weight','cpu');down=full[:,units].float().numpy().copy();del full
    g,u=weights['gate_proj'],weights['up_proj'];mu=x.mean(0)
    score=((g*g+u*u).sum(0))*x.var(0);idx=np.argsort(-score,kind='stable')[:32]
    rnd=np.random.default_rng(SEED+900).choice(x.shape[1],32,replace=False)
    arrays=dict(units=units,input_idx=idx,random_idx=rnd,input_mean=mu,gate=g,up=u,down=down,mean_a=a.mean(0))
    np.savez_compressed(OUT/'native-parameters.npz',**arrays)
    return dict(layer=b,units=units.tolist(),input_idx=idx.tolist(),random_idx=rnd.tolist(),scope='native gate/up rows and down columns; 32 input coordinates, omitted coordinates fixed to discovery mean; same-position conditional local surrogate only')

def parameter_eval(data,meta):
    b=meta['layer'];z=np.load(OUT/'native-parameters.npz');x=data.get('z'+str(b));a=data.get('a'+str(b))[:,z['units']]
    result={};errors={}
    for name,idx in [('native32',z['input_idx']),('random32',z['random_idx'])]:
        g=(x[:,idx]-z['input_mean'][idx])@z['gate'][:,idx].T+z['input_mean']@z['gate'].T
        u=(x[:,idx]-z['input_mean'][idx])@z['up'][:,idx].T+z['input_mean']@z['up'].T
        estimate=(g/(1+np.exp(-np.clip(g,-60,60))))*u
        errors[name]=((estimate-a)**2).mean(1)
    errors['mean']=((a-z['mean_a'])**2).mean(1)
    for name,e in errors.items():result[name]=float(e.mean())
    result['relative_mse_gain_vs_mean']=1-result['native32']/max(result['mean'],1e-20)
    result['relative_mse_gain_vs_random']=1-result['native32']/max(result['random32'],1e-20)
    result['paired_gain_vs_mean']=interval(errors['mean']-errors['native32'],data.fam)
    result['paired_gain_vs_random']=interval(errors['random32']-errors['native32'],data.fam)
    result['pass_20percent']=bool(min(result['relative_mse_gain_vs_mean'],result['relative_mse_gain_vs_random'])>=.2 and result['paired_gain_vs_mean']['ci975'][0]>0 and result['paired_gain_vs_random']['ci975'][0]>0)
    # Separate numerical reconstruction diagnostic from reduced-model prediction.
    gg=data.get('g'+str(b))[:,z['units']];uu=data.get('u'+str(b))[:,z['units']]
    import torch
    reconstructed=(torch.nn.functional.silu(torch.tensor(gg).bfloat16())*torch.tensor(uu).bfloat16()).float().numpy()
    result['gua_bitwise_equivalent']=bool(np.array_equal(reconstructed,a))
    result['gua_max_abs']=float(np.abs(reconstructed-a).max())
    return result

def cache_fit(x,rows,model):
    # Independent code-label readout on same frozen native coordinate budget.
    vocab=sorted({v for r in rows for v in r['choice_ids']});ix={v:i for i,v in enumerate(vocab)}
    y=np.zeros((len(rows),2*len(vocab)))
    for i,r in enumerate(rows):
        for b in range(2):y[i,b*len(vocab)+ix[r['choice_ids'][2*b+r['r'+str(b)]]]]=1
    return fit(x,y,model['idx'],1.0),vocab

def cache_eval(m,vocab,x,rows):
    p=pred(m,x).reshape(len(rows),2,len(vocab)).argmax(-1);v=np.array(vocab);predicted=v[p]
    gold=np.array([[r['choice_ids'][r['r0']],r['choice_ids'][2+r['r1']]] for r in rows]);ok=predicted==gold
    return dict(code0=float(ok[:,0].mean()),code1=float(ok[:,1].mean()),joint=float(ok.all(1).mean()),vocab=len(vocab),scope='fixed alpha=1 diagnostic only; failure does not rule out nonlinear or distributed answer cache')

def discover():
    start=time.time();data=Data('discovery');models={};grid=[];best=None;best_a=None
    keys=[f'h{i:02}' for i in range(1,41)]+['a20','a21','a22']
    for key in keys:
        x=data.get(key)
        for k in (1,32):
            m,p,stat=cv(x,data.y,data.folds,k);stat['key']=key;grid.append(stat)
            item=(stat,m,p)
            score=lambda z:(z[0]['joint'],-z[0]['k'],-keys.index(z[0]['key']))
            if best is None or score(item)>score(best):best=item
            if key.startswith('a') and k==32 and (best_a is None or score(item)>score(best_a)):best_a=item
        print(json.dumps(dict(stage='discovery_decode',key=key,completed=keys.index(key)+1,total=len(keys),elapsed=time.time()-start,latest=grid[-1])),flush=True)
    stat,m,p=best;x=data.get(stat['key']);putmodel(models,'row',m);random=[]
    for seed in (101,202,303):
        rm,rp,rs=cv(x,data.y,data.folds,stat['k'],seed);putmodel(models,'random'+str(seed),rm);random.append((rs,rp))
    vocab=sorted({v for r in data.rr for v in r['choice_ids']});nx=nuisance(data.rr,vocab)
    nm,npred,ns=cv(nx,data.y,data.folds,min(32,nx.shape[1]),404);putmodel(models,'nuisance',nm)
    # Embedding is separately reported; identical queried token cannot encode row order.
    _,ep,es=cv(data.get('h00'),data.y,data.folds,32)
    cm,cvocab=cache_fit(x,data.rr,m);putmodel(models,'cache',cm)
    np.savez_compressed(OUT/'readout-models.npz',**models)
    parameter=parameter_fit(data,best_a[0]['key'],best_a[1]['idx'])
    routing=source(data)
    baselines=[(str(rs['seed']),rp,rs) for rs,rp in random]+[('nuisance',npred,ns)]
    bn,bp,bs=max(baselines,key=lambda z:z[2]['joint'])
    report=dict(provenance=D.read(OUT/'prereg.json')['provenance'],grid=grid,chosen=stat,chosen_indices=m['idx'].tolist(),random=[s for s,_ in random],nuisance=ns,embedding=es,strongest_baseline=bn,advantage=interval(correct(p,data.y).astype(float)-correct(bp,data.y),data.fam),parameter=parameter_eval(data,parameter),routing=routing,elapsed=time.time()-start,limitations='OOF used to select layer/dimension/alpha; discovery metrics selected and optimistic, not independent evidence. Confirmation untouched until seal below.')
    save('discovery-analysis.json',report)
    files={n:D.sha(OUT/n) for n in ['natural-plan.json','calibration.json','discovery-collection.json','discovery-analysis.json','readout-models.npz','native-parameters.npz']}
    seal=dict(provenance=report['provenance'],utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),files=files,analysis_sha=D.sha(Path(__file__)),row_key=stat['key'],row_k=stat['k'],nuisance_vocab=vocab,cache_vocab=cvocab,random_seeds=[101,202,303],parameter=parameter,head={k:routing['chosen'][k] for k in ('layer','head')},discovery_encoding_gate=bool(min(stat['bit0'],stat['bit1'])>=.9 and stat['joint']>=.85 and report['advantage']['mean']>=.1 and report['advantage']['ci975'][0]>0),scope='information readout, observed routing, local native-parameter surrogate. No executable closed selector or causal claim. Negative confirmation remains negative; no post-confirm selection.')
    save('candidate.json',seal);print('CANDIDATES SEALED; confirmation may now be collected',flush=True)

def confirm():
    seal=D.read(OUT/'candidate.json');assert seal['analysis_sha']==D.sha(Path(__file__))
    for name,h in seal['files'].items():assert D.sha(OUT/name)==h
    data=Data('confirm');z=np.load(OUT/'readout-models.npz');x=data.get(seal['row_key'])
    p=pred(getmodel(z,'row'),x);result=metric(p,data.y);baselines=[]
    for seed in seal['random_seeds']:
        rp=pred(getmodel(z,'random'+str(seed)),x);baselines.append((str(seed),rp,metric(rp,data.y)))
    nx=nuisance(data.rr,seal['nuisance_vocab']);npred=pred(getmodel(z,'nuisance'),nx);baselines.append(('nuisance',npred,metric(npred,data.y)))
    comparisons={name:dict(**s,advantage=interval(correct(p,data.y).astype(float)-correct(bp,data.y),data.fam)) for name,bp,s in baselines}
    # Gate must beat EVERY frozen comparison; no choice of a favorable baseline.
    gate=bool(min(result['bit0'],result['bit1'])>=.9 and result['joint']>=.85 and all(s['advantage']['mean']>=.1 and s['advantage']['ci975'][0]>0 for s in comparisons.values()))
    np.savez_compressed(OUT/'confirm-predictions.npz',ids=np.array([r['id'] for r in data.rr]),truth=data.y,row_scores=p,**{name:bp for name,bp,_ in baselines})
    report=dict(provenance=seal['provenance'],candidate_sha=D.sha(OUT/'candidate.json'),encoding=result,comparisons=comparisons,encoding_gate=gate,cache=cache_eval(getmodel(z,'cache'),seal['cache_vocab'],x,data.rr),parameter=parameter_eval(data,seal['parameter']),routing=source(data,seal['head']),families=len(np.unique(data.fam)),independent_entity_prefixes=len(data.rr),mechanism_accepted=False,causal_tested=False,limitation='No frozen compressed native selector with upstream-only prediction and matched intervention; natural readout/attention accounting cannot distinguish exclusive table vs cache vs reread. No tuning after confirmation.')
    save('confirm-analysis.json',report);print(json.dumps({k:v for k,v in report.items() if k!='routing'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    discover() if sys.argv[1]=='discovery' else confirm()
