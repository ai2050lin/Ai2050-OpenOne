"""Freeze discovery models before scoring independent word and frame panels."""

from collections import defaultdict
from itertools import batched
import json

import numpy as np

import kv_chain_run as C
import kv_chain_native as V
import kv_chain_predict as F


STAGES=('discovery','words','frames')


def stable_identity(rows):
    import hashlib
    return hashlib.sha256(json.dumps(rows,separators=(',',':'),ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def plan(model,snapshot):
    p=C.OUT/'plan.json'
    if p.exists():return C.A.load(p.name)
    initial=C.A.load('pilot-greedy.json')
    assert initial['gate']['passed']
    panels={k:C.M.payload(model.tok,k) for k in STAGES}
    lex={key:sorted({w for r in rows for w in r['words']}) for key,rows in panels.items()}
    assert sum(map(len,lex.values()))==len(set().union(*lex.values()))
    lengths={s:sorted({len(r['token']['ids']) for r in rs}) for s,rs in panels.items()}
    c_gap={f'{t}{d}':sorted({r['token']['gap'][0] for rs in panels.values() for r in rs if (r['text'],r['distance'])==(t,d)})
           for t in (0,1) for d in (0,1)}
    assert max(max(x) for x in lengths.values())<=C.WIDTH
    C.A.save(p.name,dict(run_id=C.RUN,time=C.stamp(),snapshot=snapshot,runtime=C.A.Base.versions(),sources=C.sources(),
        settings=dict(width=C.WIDTH,batch=C.BATCH,precision='BF16',sdpa='original checkpoint, right padding, explicit causal mask',
                      pad_and_natural_code_tolerance=1.,bins=['fact','book','gen','note','extra','receipt']),
        panels={key:dict(lexicon=lex[key],n=len(rows),material_sha=stable_identity(rows),frames=sorted({r['frame'] for r in rows}),
                         family_count=len(rows)//16,lengths=lengths[key]) for key,rows in panels.items()},
        receipt=dict(texts=C.M.RECEIPTS,state_code_gap_planned=c_gap,exposure='copy instruction contains the assigned text; prompt and generated length/contents co-vary'),
        prereg=dict(clean_cell=.85,role_frame=.80,min_families_heldout=50,min_families_discovery=24,threshold_read_gain=.03,
            threshold_flip_brier_gain=.01,threshold_effect_logit=.10,threshold_dose_p95=.03,adjusted_confidence=1-.05/8,
            primary=['read accuracy augmented vs text only for both panels','flip Brier improvement vs clean-margin+text for both panels',
                     'two pooled equal-dose matched contrasts','pooled joint-ablation margin loss','pooled original partial restoration'],
            read='Predict biggest named-group attention mass at H25.18, from prefix features alone or plus L20/L21/L24 original attention at C preceding H25. Report global top token and named mass separately.',
            flip='Predict full-vocabulary P↔Q changes with natural clean margin plus discovery-trained textual or clean native QK/VO summaries. Save genuine predictions before scoring original holdout interventions.',
            role='Sample four predetermined views per family. Reverse just the fact statements on the role donor; keep distractor/receipt/map. Only the first State-token argmax is selected by the packed native donor terminal residual. Remaining tokens must pass genuine greedy causal verification.',
            sample='Family bootstrap, 80 independently generated families per panel; ideal paired normal power about .79 at standardized effect .4 and two-sided alpha .00625, before behavior and technical exclusions. Same templates and reused lexemes induce additional dependence. Family is not token count.',
            source='Original authored fictional tasks. Word pools disjoint across pilot/discovery/word/frame panels and checked against Phase 2952 geography pool. New-frame assessment includes independently held-out words. No model pretraining exclusion claim.',
            failure='Causal gate after real State validation; if a frame misses an ordinary cell, or donor challenge aggregate is <80%, no edit claims in that frame. Same source permutations and no post-holdout threshold tune. Match tolerance 3% for normalized dose p95. Preserve all diagnostics and unmatched views.')))
    return C.A.load(p.name)


def panels(model,p,k):
    rows=C.M.payload(model.tok,k)
    assert stable_identity(rows)==p['panels'][k]['material_sha']
    return rows


def verify(p,snapshot):
    assert p['snapshot']==snapshot and p['runtime']==C.A.Base.versions()
    assert all(C.A.digest(C.ROOT/k)==v for k,v in p['sources'].items())


def save_np(name,rows):
    ordinary=[key for key in rows[0] if key not in ('field','field_ids')]
    doc={key:np.concatenate([p[key] for p in rows]) for key in ordinary}
    if any('field' in p for p in rows):
        for key in ('field','field_ids'):
            doc[key]=np.concatenate([p[key] for p in rows if key in p])
    C.B.write_np(name,doc)
    return doc


def load_np(name):
    with np.load(C.OUT/name,allow_pickle=False) as value:return {k:value[k] for k in value.files}


def captured_donors(model, challenge, name):
    if (C.OUT/(name+'-donors.npz')).exists():return load_np(name+'-donors.npz')
    donor=[C.M.build_row(model.tok,dict(r,base_fact=r['fact']),1-r['fact'],1-r['fact']) for r in challenge]
    for alt,parent in zip(donor,challenge,strict=True):
        left=lambda r:r['prompt'].split('\n\n',1)[0]
        right=lambda r:r['prompt'].split('\nNote entry:',1)[1]
        assert left(alt)==left(parent) and right(alt)==right(parent)
    def extract(m,p):
        t=m.t; x=p['x'][t.arange(len(p['rows'])),p['g']-1]; logits=m.logits(x.cuda())
        return dict(ids=np.array([r['id'] for r in p['rows']]),top=logits.argmax(-1).cpu().numpy(),
                    expected=np.array([r['token']['ids'][r['token']['slots'][0]] for r in p['rows']]),
                    native=x.view(t.int16).numpy().copy())
    return save_np(name+'-donors.npz',list(C.forward(model,donor,name+'-role-donor',output=extract)))


def valid_by_frame(gate):
    good=defaultdict(list)
    for c in gate['cells']:good[c['key'][0]].append(c['code']/c['n']>=.85)
    return {f for f,a in good.items() if len(a)==16 and all(a)}


def clean_stage(model,p,stage):
    rows=panels(model,p,stage)
    b=C.behavior(model,rows,stage); seen=b['records']; gate=b['gate']
    included=valid_by_frame(gate)
    clean=C.attach(model,rows,seen); allowed=set(gate['eligible'])
    tested=[r for r in clean if r['id'] in allowed and r['frame'] in included]
    challenge=[r for r in tested if r['challenge']]
    if not tested:
        C.A.save(stage+'-status.json',dict(scope='behavior gate failed for all frames',eligible_frames=[],clean=0))
        return [],[],{}
    if stage=='discovery':assert len({r['family'] for r in tested})>=24
    valid_challenge=[]; control=[]
    natural={r['id']:r for r in seen}
    if challenge:
        donors=captured_donors(model,challenge,stage)
        assert list(donors['ids'])==[r['id'] for r in challenge]
        conflict=[C.M.build_row(model.tok,dict(r,id=r['id']+'-switch'),r['fact'],1-r['fact'],r['token']['breaks']) for r in challenge]
        select={r['id']:dict(top=int(pred)) for r,pred in zip(conflict,donors['top'],strict=True)}
        loc=stage+'-conflict.json'
        if not (C.OUT/loc).exists():
            rec=C.verify_generation(model,conflict,stage+'-conflict',select)
            C.A.save(loc,dict(records=rec,gate=C.cells(conflict,rec,True),binding=C.B.sha(stage+'-donors.npz')))
        switched=C.A.load(loc); assert switched['binding']==C.B.sha(stage+'-donors.npz')
        rec=switched['records']; natural.update({r['id']:r for r in rec})
        valid=set(switched['gate']['eligible']); total=defaultdict(int); count=defaultdict(int)
        for r in conflict:
            total[r['frame']]+=1; count[r['frame']]+=int(r['id'] in valid)
        frame_ok={k for k,n in total.items() if count[k]/n>=.80}
        valid_challenge=[r for r in C.attach(model,conflict,rec) if r['frame'] in frame_ok and r['id'] in valid]
        heads={r['id'].removesuffix('-switch') for r in valid_challenge}
        control=[r for r in challenge if r['id'] in heads]
    augmented=[*tested,*valid_challenge]
    for r in (augmented[0],):r['capture']=True
    outcome={r['id']:natural[r['id']] for r in augmented}
    source=stage+'-native.npz'
    native=load_np(source) if (C.OUT/source).exists() else save_np(source,list(C.forward(model,augmented,stage+'-trace',V.trace,V.readouts)))
    assert list(native['ids'])==[r['id'] for r in augmented]
    compare=[r for pair in zip(control,valid_challenge,strict=True) for r in pair]
    assert all(a['id']+'-switch'==b['id'] for a,b in batched(compare,2))
    C.A.save(stage+'-status.json',dict(eligible_frames=sorted(included),clean=len(tested),conflict=len(valid_challenge),
        planned=len(rows),state=sum(s['state_verified'] for s in seen),greedy=len(gate['eligible']),scoped_families=len({r['family'] for r in tested}),
        source=C.B.sha(source),report_cut='Complete named-group/causal sites only; role donor is logged separately.'))
    return augmented,compare,outcome


def clamp_controls(model,compare):
    key='random-controls.json'
    if (C.OUT/key).exists():return C.A.load(key)
    samples=compare[:64]
    scan=np.concatenate(list(C.forward(model,samples,'discovery-match-dose',V.scan_step,V.scan_output)))
    med=np.median(scan,axis=0); h=V.HEAD[1]; controls={}; results={}
    import random
    rng=random.Random(2954)
    for j,mode in enumerate(('address','value')):
        ordered=sorted((i for i in range(model.heads) if i!=h),key=lambda i:abs(float(np.log((med[j,i]+1e-6)/(med[j,h]+1e-6)))))
        pool=[i for i in ordered if med[j,i]>=.1*med[j,h]][:8] or ordered[:8]
        controls[mode]=rng.choice(pool)
        results[mode]=dict(norms=med[j].tolist(),pool=pool,selected=controls[mode])
    C.A.save(key,dict(controls=controls,method=results,n=len(samples),time=C.stamp()))
    return C.A.load(key)


def causals(model,compare,stage,control):
    name=stage+'-edits.npz'
    values=load_np(name) if (C.OUT/name).exists() else save_np(name,list(C.forward(model,compare,stage+'-edits',V.edits(model,control),V.edited_output)))
    assert list(values['ids'])==[r['id'] for r in compare]
    return values


def run(model,stage,snapshot):
    if stage=='smoke':
        rows=C.M.payload(model.tok,'pilot')[:16]; clean=C.attach(model,rows,C.A.load('pilot-greedy.json')['records'][:16]); clean[0]['capture']=True
        q=list(C.forward(model,clean,'pilot-native',V.trace,V.readouts)); assert q and q[0]['field'].shape==(1,41,3,5120)
        scan=list(C.forward(model,clean[:8],'pilot-controls',V.scan_step,V.scan_output))
        z=list(C.forward(model,clean,'pilot-edits',V.edits(model,{'value':15,'address':33}),V.edited_output))
        C.A.save('pilot-native.json',dict(read_shape=q[0]['masses'].shape,shape=z[0]['pq'].shape,scan_shape=scan[0].shape,source_ok=True))
        return
    p=plan(model,snapshot); verify(p,snapshot)
    if stage in ('discovery','all'):
        assert (C.OUT/'pilot-native.json').is_file()
        rows,compare,known=clean_stage(model,p,'discovery')
        assert len({r['family'] for r in compare})>=24,len(compare)
        control=clamp_controls(model,compare)
        edges=causals(model,compare,'discovery',control['controls'])
        F.freeze(p,rows,compare,known,load_np('discovery-native.npz'),edges,control)
    if stage in ('words','frames','all'):
        fixed=F.verify_freeze(p)
        targets=('words','frames') if stage=='all' else (stage,)
        for subset in targets:
            rows,compare,known=clean_stage(model,p,subset)
            if not rows:continue
            native=load_np(subset+'-native.npz')
            F.prior_predictions(subset,fixed,rows,compare,native)
            if compare:causals(model,compare,subset,fixed['controls'])
            F.assess(subset,fixed,rows,compare,known,native,load_np(subset+'-edits.npz') if compare else None)
