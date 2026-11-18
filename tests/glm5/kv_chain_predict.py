"""Family-level tests, pre-intervention predictions, and reproducible case records."""

from collections import defaultdict

import numpy as np
from scipy.special import ndtr

import kv_chain_run as C
import kv_chain_native as V
from kv_orth_audit import summarize


CONF=1-.05/8
NAMES=('address','value')


def text_matrix(rows):
    output=[]
    for r in rows:
        k=r['token']; s=r['sites']
        factor=[r['text'],r['distance'],r['compete'],r['map'],r['fact'],int(r['fact']!=r['desired'])]
        pairs=[a*b for j,a in enumerate(factor) for b in factor[j+1:]]
        hashes=np.bincount(np.asarray(k['ids'])%64,minlength=64)/len(k['ids'])
        geometry=[len(s[q]) for q in C.M.SITES]+[(len(k['ids'])-max(s[q]))/100 for q in C.M.SITES]
        output.append(np.r_[factor,pairs,hashes,geometry])
    return np.array(output)


def lookup(z,rows):
    all_rows={k:j for j,k in enumerate(z['ids'])}
    assert len(all_rows)==len(z['ids'])
    return np.array([all_rows[r['id']] for r in rows],dtype=int)


def features(rows,z):
    text=text_matrix(rows); j=lookup(z,rows)
    earlier=z['masses'][j,:-1].reshape(len(j),-1)
    clean=z['pq'][j,1]-z['pq'][j,0]
    # These are measured from a clean forward before editing. The read-position predictor only sees earlier layers.
    impact=np.c_[earlier,z['masses'][j,-1,1],z['direct'][j],z['dose'][j],z['score'][j]]
    return text,np.c_[text,earlier],np.c_[text,clean],np.c_[text,clean,impact]


def fit(x,y):
    mu=x.mean(0); scale=np.maximum(x.std(0),.01); z=(x-mu)/scale; reg=2.
    intercept=y.mean(0); w=np.linalg.solve(z.T@z+reg*np.eye(z.shape[1]),z.T@(y-intercept))
    mdl=dict(mu=mu.tolist(),scale=scale.tolist(),intercept=intercept.tolist(),coef=w.tolist())
    return mdl


def predict(m,x):
    return ((x-m['mu'])/m['scale'])@m['coef']+m['intercept']


def truth(z,rows):
    j=lookup(z,rows)
    return z['masses'][j,-1,1].argmax(-1)


def clean_match(rows,known,z):
    j=lookup(z,rows)
    top=np.array([known[x['id']]['top'] for x in rows]); pq=np.array([known[x['id']]['pq'] for x in rows])
    return (z['top'][j]==top)&(abs(z['pq'][j]-pq).max(-1)<=1.)


def paired_data(rows,known,z,edit):
    clean=clean_match(rows,known,z)
    assert [r['id'] for r in rows]==list(edit['ids'])
    j=lookup(z,rows)
    clean &= (edit['top'][:,0]==z['top'][j])&(abs(edit['pq'][:,0]-z['pq'][j]).max(-1)<=1.)
    return clean


def freeze(p,rows,compare,known,z,edit,control):
    name='frozen-models.json'
    if (C.OUT/name).exists():return verify_freeze(p)
    good=clean_match(rows,known,z); okay=paired_data(compare,known,z,edit)
    assert len({r['family'] for r,o in zip(compare,okay,strict=True) if o})>=24
    x,r,_,_=features(rows,z); _,_,ft,fn=features(compare,z); y=truth(z,rows)
    models={key:fit(view[good],np.eye(len(C.M.SITES))[y[good]]) for key,view in (('text_read',x),('native_read',r))}
    counts=np.bincount(y[good],minlength=len(C.M.SITES)); models['most']=int(counts.argmax())
    margin=edit['pq'][:,:,1]-edit['pq'][:,:,0]
    for name_,xi in (('text_flip',ft),('native_flip',fn)):
        block={}
        for target in NAMES:
            delta=(margin[:,V.ARMS.index(target)]-margin[:,0])[okay]
            m=fit(xi[okay],delta); s=np.sqrt(np.mean((predict(m,xi[okay])-delta)**2))
            block[target]=dict(fit=m,noise=max(.20,float(s)))
        models[name_]=block
    names=('plan.json','random-controls.json','discovery-greedy.json','discovery-donors.npz','discovery-conflict.json',
           'discovery-native.npz','discovery-edits.npz','discovery-status.json')
    C.A.save(name,dict(time=C.stamp(),binding={k:C.B.sha(k) for k in names},models=models,controls=control['controls'],
                      discovery=dict(read_n=int(good.sum()),counterfactual_n=int(okay.sum()),named_sites=C.M.SITES)))
    return verify_freeze(p)


def verify_freeze(p):
    r=C.A.load('frozen-models.json')
    assert all(C.B.sha(name)==h for name,h in r['binding'].items())
    assert all(C.A.digest(C.ROOT/k)==v for k,v in p['sources'].items())
    return r


def prior_predictions(stage,fixed,rows,compare,z):
    name=stage+'-predictions.npz'
    if (C.OUT/name).exists():
        binding=C.A.load(stage+'-predictions.json')
        assert binding['sha']==C.B.sha(name) and binding['frozen']==C.B.sha('frozen-models.json') and binding['clean']==C.B.sha(stage+'-native.npz')
        return
    assert not (C.OUT/(stage+'-edits.npz')).exists(), 'Predictions must precede held-out edits'
    m=fixed['models']; x,r,_,_=features(rows,z)
    result=dict(read_ids=z['ids'],read_base=predict(m['text_read'],x).argmax(-1),read_native=predict(m['native_read'],r).argmax(-1),
                majority=np.full(len(rows),m['most'],dtype=int))
    if compare:
        _,_,ft,fn=features(compare,z); j=lookup(z,compare)
        pq=z['pq'][j]; clean=pq[:,1]-pq[:,0]; sign=np.where(clean>0,1.,-1.)
        result.update(flip_ids=z['ids'][j],never=np.zeros((len(compare),2),dtype=np.float32))
        for model,x_ in (('text_flip',ft),('native_flip',fn)):
            shift=np.column_stack([predict(m[model][k]['fit'],x_) for k in NAMES])
            noise=[m[model][k]['noise'] for k in NAMES]
            result[model]=ndtr(-(clean[:,None]+shift)*sign[:,None]/noise)
            result[model+'_margin']=clean[:,None]+shift
    C.B.write_np(name,result)
    C.A.save(stage+'-predictions.json',dict(time=C.stamp(),sha=C.B.sha(name),frozen=C.B.sha('frozen-models.json'),clean=C.B.sha(stage+'-native.npz'),
           read_inputs='Visible text and optionally earlier attention groups through L24',flip_inputs='Clean native geometry and clean output margin, calibration only on discovery'))


def families(data,rows,mask):
    by=defaultdict(list)
    for x,r,ok in zip(data,rows,mask,strict=True):
        if ok:by[r['family']].append(float(x))
    return np.asarray([np.mean(v) for v in by.values()])


def interval(x,rows,valid,threshold):
    v=families(x,rows,valid)
    if not len(v):return dict(n=0,passed=False,mean=None,ci=None)
    y=summarize(v,CONF); y['passed']=bool(y['n']>=50 and y['mean']>=threshold and y['ci'][0]>0)
    return y


def quality(z,pred,label,clean):
    return {k:float((pred[k][clean]==label[clean]).mean()) for k in ('majority','read_base','read_native')}


def describe_cases(rows,valid,native,z,known):
    names=V.ARMS; cases=[]
    if not rows:return cases
    for label,condition in (('confirmed flip',z['top'][:,0]!=z['top'][:,1]),('block without flip',z['top'][:,0]==z['top'][:,1]),
                            ('original partial rescue changes code',z['top'][:,names.index('joint')]!=z['top'][:,names.index('restore-gen')]),
                            ('sham disagreement',~valid)):
        found=np.flatnonzero(condition & (valid if label!='sham disagreement' else ~valid))
        for i in found[:2]:
            r=rows[i]; j=lookup(native,[r])[0]
            cases.append(dict(case=label,id=r['id'],prompt=r['prompt'],fact=r['words'][r['fact']],state=r['values']['State'],
                natural=known[r['id']]['text'],truth=r['gold'],read_site=C.M.SITES[native['masses'][j,-1,1].argmax()],
                code={k:(r['token']['letters'].index(int(z['top'][i,a])) if int(z['top'][i,a]) in r['token']['letters'] else int(z['top'][i,a])) for a,k in enumerate(names)},
                logits=z['pq'][i].tolist(),dose=z['dose'][i].tolist()))
    return cases


def assess(stage,fixed,rows,compare,known,z,edit):
    from kv_chain_study import load_np
    pred=load_np(stage+'-predictions.npz'); label=truth(z,rows)
    assert list(z['ids'])==list(pred['read_ids'])
    main={}; clean=clean_match(rows,known,z)
    assert clean.any()
    difference=(pred['read_native']==label).astype(float)-(pred['read_base']==label)
    main['read']=interval(difference,rows,clean,.03)
    count=np.bincount(label[clean],minlength=len(C.M.SITES)); region={s:int(n) for s,n in zip(C.M.SITES,count,strict=True)}
    source_in_site=np.array([int(i) in {j for positions in row['sites'].values() for j in positions} for row,i in zip(rows,z['most'],strict=True)])
    stat=dict(primary=main,read=dict(accuracy=quality(z,pred,label,clean), named_top_counts=region,valid_n=int(clean.sum()),
              global_top_is_annotated=int(source_in_site[clean].sum()), mean_mass=z['masses'][clean,-1,1].mean(0).tolist()),
              study_sha=C.B.sha('frozen-models.json'),predictions_sha=C.B.sha(stage+'-predictions.npz'))
    cells={}
    for q in sorted({(r['frame'],r['text'],r['distance'],r['compete'],r['map']) for r in rows}):
        use=np.array([(r['frame'],r['text'],r['distance'],r['compete'],r['map'])==q for r in rows])&clean
        if use.any():cells[''.join(map(str,q))]=dict(n=int(use.sum()),state_attention=float(z['masses'][use,-1,1,2].mean()))
    stat['factor_cells']=cells
    if edit is not None:
        assert list(edit['ids'])==list(pred['flip_ids'])
        usable=paired_data(compare,known,z,edit); top=edit['top']; pq=edit['pq']; margin=pq[:,:,1]-pq[:,:,0]
        pair_ids=np.array([r['token']['letters'] for r in compare]); right=top[:,0]==pair_ids[:,1]; sign=np.where(right,1.,-1.)
        arms=[V.ARMS.index(k) for k in NAMES]
        change=(top[:,arms]!=top[:,0,None])&((top[:,arms]==pair_ids[:,0,None])|(top[:,arms]==pair_ids[:,1,None]))
        predkeys=('never','text_flip','native_flip')
        losses={k:((pred[k]-change)**2).mean(-1) for k in predkeys}
        main['flip']=interval(losses['text_flip']-losses['native_flip'],compare,usable,.01)
        stats={}
        for k in predkeys:
            prediction=pred[k]>=.5; common=usable[:,None]
            tp=int((prediction&change&common).sum());fp=int((prediction&~change&common).sum()); fn=int((~prediction&change&common).sum())
            stats[k]=dict(brier=float(losses[k][usable].mean()),true_positive=tp,false_positive=fp,false_negative=fn,
                          precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
        effect={}; dose={}; a=V.ARMS.index; full=margin*sign[:,None]
        for i,name in enumerate(NAMES):
            z0,z1=edit['dose'][:,i*2],edit['dose'][:,i*2+1]
            err=np.abs(z0[:,2]-z1[:,2])/np.maximum(z0[:,1],.02)
            dose[name]=dict(p95=float(np.quantile(err[usable],.95)), target_median=float(np.median(z0[usable,2])),
                            control_median=float(np.median(z1[usable,2])))
            effect[name]=full[:,a('random-'+name)]-full[:,a('matched-'+name)]
        effect['joint']=full[:,0]-full[:,a('joint')]
        effect['rescue']=full[:,a('restore-gen')]-full[:,a('joint')]
        raw={k:interval(v,compare,usable,.10) for k,v in effect.items()}
        for k in NAMES:
            raw[k]['dose_admitted']=dose[k]['p95']<=.03; raw[k]['passed']&=raw[k]['dose_admitted']
        stat.update(impact=dict(valid=int(usable.sum()),attempted=len(compare),brier=stats,dose=dose,effects=raw,
                      real_flips={k:int(((top[:,a(k)]!=top[:,0])&usable).sum()) for k in V.ARMS},
                      full_restore_exact=bool(np.array_equal(pq[:,0],pq[:,a('restore-all')]) and np.array_equal(top[:,0],top[:,a('restore-all')]))),
                    cases=describe_cases(compare,usable,z,edit,known))
        C.B.write_np(stage+'-effect-detail.npz',dict(ids=edit['ids'],eligible=usable,**effect))
    C.A.save(stage+'-analysis.json',stat)
    print(stage,main,stat.get('impact',{}).get('valid',0),flush=True)
