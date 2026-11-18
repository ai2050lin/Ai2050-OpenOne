"""Compare frozen text, native-coordinate and sparse-feature forecasts on the same queries."""

from collections import defaultdict
import time

import numpy as np

import kv_chain_material
import kv_chain_native as K
import record_update_material as M
import record_update_run as R
from deploy.storage_policy import check_write, staging_file
from kv_origin_audit import summarize
from shared.dist_protocol.transcoder import TopKTranscoder, train


C=R.C
D=5120
GROUPS=('intro','rule','first','second','key','instruction','state','other')
NAMES=('state','code')
METHODS=('text','native','dictionary')


def save_np(name, with_, plan):
    payload=dict(provenance=np.array(M.canonical(plan['provenance']).decode()),**with_)
    # Raw byte size plus ZIP overhead is a safe bound for compressed NumPy output.
    n=sum(x.nbytes for x in payload.values())
    size=n+n//20+2**17
    target=check_write(R.OUT/name,size,operation='记录更新原生摘要和预测')
    with staging_file(target) as temp, temp.open('wb') as f:
        np.savez_compressed(f,**payload)
    return C.A.digest(target)


def load_np(name):
    with np.load(R.OUT/name,allow_pickle=False) as f:
        return {k:f[k] for k in f.files if k!='provenance'}


def caption(tok, row, sample):
    """Find causal query boundaries in the measured history, including wrong answers."""
    k=row['token']; ids=k['prompt']; observed=sample['actual']
    token_prefix=[]; next_=[observed[0],-1]; e=len(ids)-1
    g=c=-1
    if sample['format'] and sample['state']:
        matches=[i for i in range(1,len(observed)) if tok.decode(observed[:i],skip_special_tokens=False).strip().endswith('Code:')]
        if len(matches)==1:
            pos=matches[0]
            word=tok.decode(observed[:1],skip_special_tokens=False).strip()
            if word==sample['state']:
                g=len(ids); c=e+pos; token_prefix=observed[:pos+1]; next_[1]=observed[pos]
    return dict(row,query=(e,c),expected_top=next_,g=g,
                token=dict(k,ids=ids+token_prefix,slots=[g if g>=0 else e]))


def sites(model,p):
    if 'query_locations' in p:
        return p['query_locations']
    t=model.t; r=p['rows']; b=len(r); w=p['x'].shape[1]
    j=t.arange(b,device='cuda')
    q=t.tensor([[x['query'][0],max(x['query'][0],x['query'][1])] for x in r],device='cuda')
    loc=t.zeros((b,len(GROUPS),w),dtype=t.bool,device='cuda')
    for i,row in enumerate(r):
        for k,positions in enumerate(row['token']['sections']):loc[i,k,positions]=True
        if row['g']>=0:loc[i,6,row['g']]=True
        loc[i,7,:len(row['token']['ids'])]=~loc[i,:7,:len(row['token']['ids'])].any(0)
    p['query_locations']=j,q,loc
    return p['query_locations']


def bits(t,x):
    return x.contiguous().view(t.int16).cpu().numpy().copy()


def fp32(t,x,device):
    return t.as_tensor(np.ascontiguousarray(x),device=device).view(t.bfloat16).float()


def h18(model,layer,i,p,x,q,k,v,o,mask):
    audit=[k for k,r in enumerate(p['rows']) if r.get('audit')]
    if not (audit or i in (M.READ_LAYER,M.SOURCE_LAYER)):
        return
    t=model.t
    j,pos,loc=sites(model,p)
    if audit:
        p.setdefault('field',[]).append(bits(t,x[j[:,None],pos][audit]))
        p['sample']=audit
    if i==M.READ_LAYER:
        p['early']=bits(t,x[j[:,None],pos])
    if i!=M.SOURCE_LAYER:
        return
    h=M.SOURCE_HEAD; masses=[]
    for qid in pos.T:
        visible=mask[j,:,qid].unsqueeze(2)
        a=model.read(q[j,:,qid].unsqueeze(2),k,visible)[:,h]
        masses.append((a[:,None]*loc).sum(-1).cpu().numpy())
    p['masses']=np.stack(masses,1)
    if p['rows'][0]['split']=='discovery':
        which=pos[:,1]
        qc=q[j,:,which].unsqueeze(2); vis=mask[j,:,which].unsqueeze(2)
        hs=tuple(range(model.heads))
        shift=t.stack([K.raw_change(model,layer,h0,qc,k,v,vis,loc[:,6],'value') for h0 in hs],1)
        p['norms']=model.projections(layer,hs,shift).norm(dim=-1).cpu().numpy()


def packed_output(model,p):
    t=model.t; j,pos,_=sites(model,p)
    last=p['x'].cuda()[j[:,None],pos]
    z=model.decoder.norm(last)
    scores=model.model.lm_head(z)
    output=dict(early=p['early'],target=bits(t,z),masses=p['masses'],
                argmax=scores.argmax(-1).cpu().numpy(),ids=np.array([r['id'] for r in p['rows']]))
    if 'norms' in p:output['norms']=p['norms']
    if 'field' in p:
        output['field']=np.stack((*p['field'],bits(t,last[p['sample']])),axis=1)
        output['field_ids']=output['ids'][p['sample']]
    return output


def collect(model,rows,behavior,split,plan):
    behaviors={r['id']:r for r in behavior['records']}
    observed=[caption(model.tok,row,behaviors[row['id']]) for row in rows]
    name=split+'-native.npz'; record=split+'-native-binding.json'
    if (R.OUT/record).exists():
        assert all(C.A.digest(R.OUT/path)==sha for path,sha in R.read(record)['links'].items())
        return observed,load_np(name)
    for row in observed[:2]:row['audit']=True
    packets=list(C.forward(model,observed,split+'-readouts',h18,packed_output))
    z={k:np.concatenate([p[k] for p in packets if k in p]) for k in set().union(*packets)}
    assert list(z['ids'])==[r['id'] for r in observed]
    assert np.allclose(z['masses'].sum(-1),1,atol=.001)
    save_np(name,z,plan)
    files=(name,'plan.json',split+'-behavior.json')
    R.write(record,dict(at=C.stamp(),provenance=plan['provenance'],links={p:C.A.digest(R.OUT/p) for p in files}))
    return observed,z


def text_features(row,site):
    ids=np.asarray(row['token']['ids'][:site+1],dtype=np.int64)
    n=len(ids)
    unigrams=np.bincount(ids%64,minlength=64)/n
    bigrams=np.bincount((ids[1:]*65537+ids[:-1])%64,minlength=64)/max(n-1,1)
    last=np.bincount(ids[-12:]%32,minlength=32)/min(n,12)
    return np.r_[1.,n/M.WIDTH,row['g']>=0 and site>row['g'],unigrams,bigrams,last]


def coordinates(rows,z,model,dictionary,j):
    t=model.t
    x=fp32(t,z['early'][:,j],'cuda')
    valid=np.array([r['query'][j]>=0 for r in rows])
    exact=valid&(np.array([r['expected_top'][j] for r in rows])==z['argmax'][:,j])
    baseline=np.asarray([text_features(r,max(r['query'][j],r['query'][0])) for r in rows])
    return dict(valid=valid,exact=exact,text=baseline,native=x.cpu().numpy(),
                dictionary=dictionary.encode(x).cpu().numpy(),target=fp32(t,z['target'][:,j],'cpu').numpy(),
                source=np.log(np.maximum(z['masses'][:,j],1e-8)))


def linear(x,y,ridge):
    mu=x.mean(0); scale=np.maximum(x.std(0),.01)
    design=(x-mu)/scale
    intercept=y.mean(0)
    weights=np.linalg.solve(design.T@design+ridge*np.eye(len(mu)),design.T@(y-intercept))
    return {k:np.asarray(v,np.float32) for k,v in dict(mu=mu,scale=scale,intercept=intercept,weights=weights).items()}


def evaluate(fit,x):
    return ((np.asarray(x,np.float32)-fit['mu'])/fit['scale'])@fit['weights']+fit['intercept']


def selected(x,resid,limit):
    x=np.asarray(x,np.float64); x-=x.mean(0)
    scale=np.linalg.norm(x,axis=0)
    measure=np.linalg.norm(x.T@resid,axis=1)/np.maximum(scale,1e-8)
    return np.argsort(-measure,kind='stable')[:limit]


def x_for(data,method,feature):
    return data['text'] if method=='text' else np.c_[data['text'],data[method][:,feature]]


def choose_controls(z,rows,model):
    h=M.SOURCE_HEAD
    mask=np.array([r['query'][1]>=0 for r in rows])
    norms=z['norms'][mask]
    ratio=np.median(np.minimum(norms,norms[:,h,None])/np.maximum(np.maximum(norms,norms[:,h,None]),1e-6),axis=0)
    pool=sorted((j for j in range(model.heads) if j//model.groups != h//model.groups),key=lambda j:(-ratio[j],j))[:8]
    rng=np.random.default_rng(M.SEED)
    # Reuse the same matched control for both edit types where applicable.
    return dict(head=int(rng.choice(pool)),eligible=pool,discovery_norm_ratio=ratio.tolist())


def freeze_models(model,p,rows,z):
    t=model.t; cfg=p['design']; names=('dictionary.npz','forecast.npz','prediction-freeze.json')
    if (R.OUT/names[2]).exists():
        frozen=R.read(names[2])
        assert all(C.A.digest(R.OUT/k)==v for k,v in frozen['links'].items())
        return TopKTranscoder.from_state(load_np(names[0]),device='cuda'),load_np(names[1]),frozen
    valid=np.array([r['query'][1]>=0 for r in rows])
    raw=fp32(t,z['early'],'cuda')
    examples=t.cat((raw[:,0],raw[valid,1]),dim=0)
    sampler=t.Generator(device='cuda').manual_seed(M.SEED)
    start=time.monotonic()

    def sample(n):
        x=examples[t.randint(len(examples),(n,),device='cuda',generator=sampler)]
        return x,x
    with t.enable_grad():
        dictionary,history=train(sample,d_in=D,d_out=D,m=cfg['dictionary_atoms'],k=cfg['dictionary_active'],
                                 steps=cfg['dictionary_steps'],batch=96,device='cuda',seed=M.SEED,
                                 log=lambda x:C.N.progress('discovery-sparse-dictionary',x['step']+1,cfg['dictionary_steps'],start,fvu=x['fvu']),
                                 log_every=20)
    fits={}
    for i,query in enumerate(NAMES):
        d=coordinates(rows,z,model,dictionary,i); use=d['exact']
        assert int(use.sum())>=300,(query,int(use.sum()))
        response=np.c_[d['target'],d['source']]
        base=linear(d['text'][use],response[use],cfg['ridge_regularization'])
        residual=response[use,:D]-evaluate(base,d['text'][use])[:,:D]
        for method in METHODS:
            ind=selected(d[method][use],residual,cfg['added_inputs']) if method!='text' else np.array([],np.int64)
            fit=base if method=='text' else linear(x_for(d,method,ind)[use],response[use],cfg['ridge_regularization'])
            for key,value in (fit|{'features':ind}).items():
                fits[f'{query}.{method}.{key}']=value
    save_np(names[0],dictionary.state(),p)
    save_np(names[1],fits,p)
    files=(*names[:2],'plan.json','discovery-native-binding.json')
    frozen=dict(at=C.stamp(),provenance=p['provenance'],control=choose_controls(z,rows,model),dictionary_train=history,
                links={k:C.A.digest(R.OUT/k) for k in files})
    R.write(names[2],frozen)
    return dictionary,fits,frozen


def score_vocab(model,estimate,target,expected):
    t=model.t; head=model.model.lm_head
    scores=[]
    for lo in range(0,len(target),12):
        observed=head(t.as_tensor(target[lo:lo+12],device='cuda',dtype=t.bfloat16)).float()
        guessed=head(t.as_tensor(estimate[lo:lo+12],device='cuda',dtype=t.bfloat16)).float()
        p=observed.log_softmax(-1); q=guessed.log_softmax(-1)
        index=t.as_tensor(np.maximum(expected[lo:lo+12],0),device='cuda',dtype=t.int64)[:,None]
        scores.append(t.stack(((p.exp()*(p-q)).sum(-1), guessed.argmax(-1),
                               -q.gather(1,index).squeeze(1)),1).cpu().numpy())
    return np.concatenate(scores,axis=0)


def paired(values,rows,mask,confidence=.99):
    bins=defaultdict(list)
    for v,r,ok in zip(values,rows,mask,strict=True):
        if ok:bins[r['family']].append(v)
    return summarize([np.mean(x) for x in bins.values()],confidence) if bins else dict(n=0,mean=None,ci=None)


def forecast(model,p,rows,z,dict_,fits,stage):
    name=stage+'-predictions.npz'; binding=stage+'-predictions.json'
    if (R.OUT/binding).exists():
        b=R.read(binding)
        assert all(C.A.digest(R.OUT/k)==v for k,v in b['links'].items())
        return b
    if (R.OUT/(stage+'-intervention.npz')).exists():raise ValueError('Predictions must precede intervention')
    metric={}; arrays={'ids':z['ids']}
    for qi,query in enumerate(NAMES):
        d=coordinates(rows,z,model,dict_,qi)
        use=d['exact']; target=np.array([r['expected_top'][qi] for r in rows])
        scores={}
        for method in METHODS:
            pref=f'{query}.{method}.'
            fit={k.removeprefix(pref):v for k,v in fits.items() if k.startswith(pref)}
            yhat=evaluate(fit,x_for(d,method,fit['features']))
            val=score_vocab(model,yhat[:,:D],d['target'],target)
            shat=yhat[:,D:]
            psrc=np.exp(d['source']); logq=shat-np.logaddexp.reduce(shat,axis=-1,keepdims=True)
            src_kl=np.sum(psrc*(d['source']-logq),axis=-1)
            scores[method]=val[:,0]
            arrays[f'{query}.{method}.metrics']=np.c_[val,src_kl]
            arrays[f'{query}.{method}.read_probs']=np.exp(logq)
            metric[f'{query}.{method}']=dict(kl=paired(val[:,0],rows,use),
                source_kl=paired(src_kl,rows,use),full_vocabulary_match=paired(val[:,1]==target,rows,use),
                hidden_mse=paired(((d['target']-yhat[:,:D])**2).mean(1),rows,use),
                read_count=int(use.sum()),unstable_ids=z['ids'][d['valid']&~use].tolist())
        for method in METHODS[1:]:
            gain=paired(scores['text']-scores[method],rows,use)
            denominator=metric[f'{query}.text']['kl']['mean']
            gain['relative']=gain['mean']/max(denominator,1e-6) if denominator is not None else None
            gain['passed']=bool(gain['n']>=p['design']['minimum_power_families'] and gain['ci'][0]>0 and gain['relative']>=.10)
            metric[f'{query}.{method}']['improvement']=gain
    save_np(name,arrays,p)
    files=(name,stage+'-native-binding.json','prediction-freeze.json')
    b=dict(at=C.stamp(),provenance=p['provenance'],metric=metric,
           links={k:C.A.digest(R.OUT/k) for k in files})
    R.write(binding,b)
    return b


def perturb(model,p,rows,z,behavior,frozen,split):
    """Apply State V blocking, matched random write, and exact native restoration."""
    key=split+'-intervention.npz'
    if (R.OUT/key).exists():return load_np(key)
    direct=[r for r in rows if r['query'][1]>=0 and r['g']>=0]
    paired_rows=[]
    for r in direct:
        token=dict(r['token'],ids=r['token']['ids'][:r['query'][1]+1])
        named={site:[] for site in kv_chain_material.SITES}
        named['gen']=[r['g']]; named['note']=r['token']['sections'][3]
        paired_rows.append(dict(r,token=token,sites=named))
    head=frozen['control']['head']
    arms=K.edits(model,dict(address=head,value=head))
    result=list(C.forward(model,paired_rows,split+'-native-edit',arms,K.edited_output))
    edit={k:np.concatenate([a[k] for a in result]) for k in result[0]}
    save_np(key,edit,p)
    old={r['id']:r for r in behavior['records']}
    raw=np.array([old[k]['actual'][r['query'][1]-r['query'][0]] for k,r in zip(edit['ids'],paired_rows,strict=True)])
    idmap={k:j for j,k in enumerate(z['ids'])}
    target=np.array([z['argmax'][idmap[k],1] for k in edit['ids']])
    take=(edit['top'][:,0]==target)&(raw==target)
    letters=np.array(model.letters); expected=np.array([r['answer']=='Q' for r in paired_rows])
    sign=np.where(expected,1.,-1.)
    margin=(edit['pq'][:,:,1]-edit['pq'][:,:,0])*sign[:,None]
    a=K.ARMS.index
    v=abs(edit['dose'][:,2,2]-edit['dose'][:,3,2])/np.maximum(edit['dose'][:,2,1],.02)
    p95=float(np.quantile(v[take],.95)) if take.any() else None
    loss=paired(margin[:,a('random-value')]-margin[:,a('matched-value')],paired_rows,take)
    restore=paired(margin[:,a('restore-gen')]-margin[:,a('joint')],paired_rows,take)
    dose_admitted=p95 is not None and p95<=p['design']['dose_rel_p95']
    causal_pass=dose_admitted and all(stat['n']>=p['design']['minimum_power_families'] and stat['ci'][0]>0 and stat['mean']>=p['design'][minimum]
                                     for stat,minimum in ((loss,'minimum_matched_margin'),(restore,'minimum_restore_margin')))
    phase=dict(at=C.stamp(),provenance=p['provenance'],dose_error_p95=p95,dose_admitted=dose_admitted,
               passed=causal_pass,
               heads=dict(target=M.SOURCE_HEAD,control=head),loss_vs_matched=loss,partial_native_restore=restore,
               full_native_restore_matches_sham=bool(np.array_equal(edit['top'][:,a('restore-all')],edit['top'][:,0])),
               eligible=int(take.sum()),failed_sham=edit['ids'][~take].tolist(),
               real_code_changes_by_arm={arm:int((take & (edit['top'][:,i]!=edit['top'][:,0]) & np.isin(edit['top'][:,i],letters)).sum())
                                         for i,arm in enumerate(K.ARMS)},
               input_predictions_sha=C.A.digest(R.OUT/(split+'-predictions.json')),effects_sha=C.A.digest(R.OUT/key))
    R.write(split+'-intervention.json',phase)
    return edit


def run(model,p,discovery,behavior):
    rows,z=collect(model,discovery,behavior,'discovery',p)
    dictionary,fits,frozen=freeze_models(model,p,rows,z)
    hold={}; stages={}
    for split in R.HOLDOUTS:
        material,b=R.stage(model,p,split)
        if not b['gate']['passed']:
            stages[split]=dict(behavior='failed',cells=b['gate']['cells'])
            continue
        observed,native=collect(model,material,b,split,p)
        result=forecast(model,p,observed,native,dictionary,fits,split)
        hold[split]=(observed,native,b)
        stages[split]=dict(behavior='passed',metric=result['metric'])
    admitted=[method for method in METHODS[1:] if set(hold)==set(R.HOLDOUTS) and
              all(stages[s]['metric'][f'{q}.{method}']['improvement']['passed'] for s in R.HOLDOUTS for q in NAMES)]
    R.write('stage-decision.json',dict(at=C.stamp(),provenance=p['provenance'],admitted=admitted,
            panels=stages,lockbox='unopened',intervention=('admitted' if admitted else 'blocked_by_frozen_gate')))
    if admitted:
        for split,(observed,native,b) in hold.items():perturb(model,p,observed,native,b,frozen,split)
