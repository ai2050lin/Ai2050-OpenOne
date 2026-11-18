"""Freeze on discovery and score new vocabulary and new prompt frames."""

import numpy as np
from scipy.special import ndtr

import kv_receipt_native as V
import kv_receipt_run as R
from kv_chain_predict import fit,predict


PLAN = dict(behavior=.85, min_panel=40, confidence=.99, read_gain=.03, brier_gain=.01,
            dose_error=.03, sham_pq=.5, effect=.10, match_seed=295517,
            read_truth='H18 Receipt group mass greater than its State-token mass',
            flip_truth='full-vocabulary argmax changes between P and Q from same-family zero-insert baseline',
            controls='scan same layer in discovery; sample from six nearest physical doses outside shared KV group',
            prediction='fit insertion-branch log mass ratio and signed final margin; final margin baseline from zero insertion only',
            early='Read ratios and changes at L20H5, L21H12, L24H18; branch H18 and answer are labels only',
            thresholds='each holdout independently passes gain, family-wise interval excluding zero, and minimum families',
            eligibility='per frame and State mode, each cell >= .85 verified and P/Q; natural baseline also correct; no exclusion by which letter wins after insertion',
            history='greedy verification except externally supplied insertion and (conflict only) reverse-fact donor State')


def read_np(name):
    with np.load(R.OUT/name,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def plan(model,identity):
    r=R.Chain; a=R.A; src=r.sources()
    for name in ('kv_receipt_material','kv_receipt_run','kv_receipt_native','kv_receipt_study'):
        path=f'tests/glm5/{name}.py'; src[path]=a.digest(R.ROOT/path)
    current=dict(run=R.M.RUN, snapshot=identity,sources=src,pilot=R.B.sha('pilot.json'),
                 vocabulary=R.M.vocabulary(model.tok),tracks=R.M.sequence(model.tok),design=PLAN,
                 sizes=R.M.SIZES,readers=V.V.READERS,padding=R.WIDTH,batch=R.CONSORTIUM,arms=V.NAMES)
    if (R.OUT/'plan.json').is_file():
        previous=a.load('plan.json'); time=previous.pop('frozen_at'); assert R.digest(previous)==R.digest(current)
    else:
        a.save('plan.json',dict(current,frozen_at=r.stamp()))
    return R.B.sha('plan.json')


def collect(model,stage):
    rows=R.panels(model,stage)
    material=stage+'-material.json'; native=stage+'-native.npz'; behavior=stage+'-behavior.json'
    receipt=stage+'-observation.json'
    if (R.OUT/receipt).exists():
        b=R.A.load(receipt)
        assert all(R.B.sha(k)==v for k,v in b['sha'].items()) and b['material_digest']==R.digest(rows)
        return rows,read_np(native),R.A.load(behavior)
    n=R.digest(rows)
    R.A.save(material,rows)
    model.receipt_selector=R.donors(model,rows,stage)
    seen=set()
    for i in range(0,len(rows),R.CONSORTIUM):
        key=rows[i]['frame'],rows[i]['conflict']
        if key not in seen:
            for r in rows[i:i+R.CONSORTIUM]:r['capture']=True
            seen.add(key)
    gathered=list(R.Chain.forward(model,rows,stage+'-native',V.read_trace(scan=stage=='discovery'),V.readings))
    records=[each for entries,_ in gathered for each in entries]
    arrays=[z for _,z in gathered]
    z={k:np.concatenate([a[k] for a in arrays if k in a]) for k in set().union(*arrays)}
    a=np.array([r['family'] for r in rows]); assert np.array_equal(z['ids'],[r['id'] for r in rows]);
    # All paths precede the insertion at the same absolute State position.
    for field in ('source_k','source_v'):
        x=z[field].reshape(-1,R.CONSORTIUM,z[field].shape[1])
        assert np.all(x==x[:,:1]),f'{field} changed before insertion'
    assert np.allclose(z['masses'][:,:,:6].sum(-1),1.,atol=.002)
    g=R.gate(rows,records)
    check=dict(records=records,gate=g)
    R.B.write_np(native,z); R.A.save(behavior,check)
    files=(material,native,behavior,stage+'-donors.npz')
    R.A.save(receipt,dict(at=R.Chain.stamp(),material_digest=n,sha={k:R.B.sha(k) for k in files}))
    print(stage,'admitted',len(g['eligible']),'complete',len(g['complete']),'of',len(a)//R.CONSORTIUM,g['frame_modes'],flush=True)
    return rows,z,check


def covariates(rows,z,b):
    records=b['records']; eligible=set(b['gate']['eligible'])
    seeds={r['family']:i for i,r in enumerate(rows) if r['n']==0}
    base=np.array([seeds[r['family']] for r in rows]); idx=np.arange(len(rows))
    count=np.array([len(r['sites']['receipt']) for r in rows])
    text=np.array([[r['n']/7,(r['n']/7)**2, r['track']==0,r['track']==1,r['track']==2,
                   r['conflict'],r['mapping'],r['fact'],len(r['token']['ids'])/100,
                   (len(r['token']['ids'])-r['token']['slots'][0])/100] for r in rows]);
    earlier=z['masses'][:,:-1]
    ratio=np.log(np.maximum(earlier[:,:,0],1e-9))-np.log(np.maximum(earlier[:,:,1],1e-9))
    top=np.array([r['code_top'] for r in records]); codes=np.array([r['code_ids'] for r in records])
    pq=np.asarray([r['pq'] for r in records]); margin=pq[:,1]-pq[:,0]; sign=np.where(top[base]==codes[:,1],1.,-1.)
    real=sign*margin
    h18=z['score_pair'][:,0]-z['score_pair'][:,1]
    baseline=real[base]
    natural=idx!=base
    admitted=np.array([r['family'] in eligible for r in rows])
    return dict(read_text=text,read_native=np.c_[text,ratio,ratio-ratio[base]],
                flip_text=np.c_[text,baseline,baseline*text[:,0]],
                flip_native=np.c_[text,baseline,baseline*text[:,0],ratio,ratio-ratio[base]],
                h18=h18,base=base,top=top,real=real,flip=(top!=top[base]),sign=sign,
                split=natural,admitted=admitted,count=count)


def freeze(rows,z,b,plan_sha):
    target='frozen.json'; source=R.B.sha('discovery-observation.json')
    if (R.OUT/target).exists():
        f=R.A.load(target); assert f['observation']==source and f['plan']==plan_sha
        return f
    x=covariates(rows,z,b); use=x['admitted']&x['split']; assert int(use.sum())>=180
    result={}
    for name in ('read_text','read_native','flip_text','flip_native'):
        inputs=x[name][use]; truth=x['h18'][use] if name.startswith('read') else x['real'][use]
        model=fit(inputs,truth); residual=truth-predict(model,inputs)
        result[name]=dict(fit=model,noise=max(.20,float(np.sqrt(np.mean(residual**2)))))
    f=dict(models=result,controls=V.controls(rows,z,set(b['gate']['eligible'])),
           observation=source,plan=plan_sha,at=R.Chain.stamp(),discovery_n=int(use.sum()),
           majority=float((x['h18'][use]>0).mean()))
    R.A.save(target,f)
    return f


def forecast(stage,rows,z,b,f):
    name=stage+'-predictions.npz'; receipt=stage+'-prediction-binding.json'
    tar='frozen.json'; obs=stage+'-observation.json'
    if (R.OUT/receipt).exists():
        b=R.A.load(receipt)
        assert all(R.B.sha(k)==v for k,v in b['sha'].items())
        return
    assert not (R.OUT/(stage+'-edits.npz')).exists()
    x=covariates(rows,z,b); output=dict(ids=z['ids'])
    for key,m in f['models'].items():
        pred=predict(m['fit'],x[key]); output[key]=pred
        if key.startswith('flip'):
            output[key+'_prob']=ndtr(-pred/m['noise'])
    R.B.write_np(name,output)
    R.A.save(receipt,dict(at=R.Chain.stamp(),sha={k:R.B.sha(k) for k in (name,tar,obs)}))


def intervention(model,stage,rows,b,f):
    name=stage+'-edits.npz'; final=stage+'-intervention-binding.json'
    if (R.OUT/final).exists():
        assert all(R.B.sha(k)==v for k,v in R.A.load(final)['sha'].items())
        return
    families=set(b['gate']['eligible'])
    if not families:
        R.A.save(final,dict(at=R.Chain.stamp(),sha={},reason='No strata passed frozen behavior gate'))
        return
    selected=[r for r in rows if r['family'] in families]
    R.B.write_np(name,V.run_edits(model,selected,f['controls'],stage))
    req=(name,stage+'-prediction-binding.json','frozen.json')
    R.A.save(final,dict(at=R.Chain.stamp(),sha={k:R.B.sha(k) for k in req}))


def smoke(model):
    rows=[r for i,g in enumerate(R.M.families(model.tok,'pilot')) if i in (0,4) for r in g]
    model.receipt_selector=R.donors(model,rows,'smoke')
    for r in rows[:R.CONSORTIUM]:r['capture']=True
    tests=list(R.Chain.forward(model,rows,'smoke-native',V.read_trace(scan=True),V.readings))
    assert all(t['alpha'].shape==(R.CONSORTIUM,R.WIDTH) for _,t in tests)
    assert all(np.all(t['source_k']==t['source_k'][:1]) for _,t in tests)
    assert tests[0][1]['field'].shape[:3]==(R.CONSORTIUM,41,3)
    z=V.run_edits(model,rows[:R.CONSORTIUM],dict(address={'head':14},value={'head':14}),'smoke')
    assert z['pq'].shape==(R.CONSORTIUM,len(V.NAMES),2)
    R.A.save('smoke.json',dict(n=len(rows),field_shape=tests[0][1]['field'].shape,edit_shape=z['pq'].shape,
             max_sham_pq=float(np.max(np.abs(z['pq'][:,0]-np.array([x['pq'] for x in tests[0][0]]))))))


def run(model,stage,identity):
    if not (R.OUT/'smoke.json').exists():smoke(model)
    plan_sha=plan(model,identity)
    rows,z,b=collect(model,'discovery')
    f=freeze(rows,z,b,plan_sha)
    if stage=='discovery':return
    assert stage in ('all','words','frames')
    for split in (('words','frames') if stage=='all' else (stage,)):
        rows,z,b=collect(model,split)
        forecast(split,rows,z,b,f)
        intervention(model,split,rows,b,f)
