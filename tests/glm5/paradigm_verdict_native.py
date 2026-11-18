"""Matched native residual tests on both discovery-frozen directions."""

from itertools import batched
import time

import numpy as np

import paradigm_verdict_run as V
from deploy.storage_policy import check_write

W,N,M,R,C,S=V.W,V.N,V.M,V.R,V.C,V.S
OUT=V.OUT
SLAB=128
ARMS=('sham','restore','full','last20','full_perm','full_natural','last20_perm','last20_natural')
FIRST_MATCH=N.fit_direction


def repair_norm(t,state,ray,goal,mask):
    """Repair failed tokens by bracketing the actual monotone BF16 dose."""
    first,receipt=FIRST_MATCH(t,state,ray,goal,mask)
    measured=receipt[...,1]
    gap=(goal-measured).abs()/t.maximum(goal,measured).clamp_min(1e-12)
    take=mask&(goal>0)&(receipt[...,3]>0)&((gap>.03)|(receipt[...,2]<.90))
    if not take.any():
        return first,receipt
    base=state[take].float()
    u=ray[take]/receipt[...,3][take][:,None]
    target=goal[take]

    def candidate(scale):
        value=(base+u*scale[:,None]).to(state.dtype)
        delta=value.float()-base
        dose=delta.norm(dim=-1)
        cos=(delta*u).sum(-1)/dose.clamp_min(1e-12)
        return value,dose,cos

    lo=t.zeros_like(target);hi=2*target
    for _ in range(6):
        _,dose,_=candidate(hi)
        hi=t.where(dose<target,2*hi,hi)
    for _ in range(17):
        mid=(lo+hi)/2
        _,dose,_=candidate(mid)
        lo=t.where(dose<target,mid,lo)
        hi=t.where(dose>=target,mid,hi)
    picked=first[take]; n=measured[take]; c=receipt[...,2][take]
    def score(dose,cos):
        return (dose-target).abs()/t.maximum(dose,target).clamp_min(1e-12)+(cos<.90).float()
    for level in (lo,hi):
        new,achieved,angle=candidate(level)
        better=score(achieved,angle)<score(n,c)
        picked=t.where(better[:,None],new,picked)
        n=t.where(better,achieved,n)
        c=t.where(better,angle,c)
    first[take]=picked
    receipt[...,1][take]=n
    receipt[...,2][take]=c
    return first,receipt


def rows(tokenizer,count,corners):
    out=[]
    with M.scope():
        for pair in batched(M.families('confirm')[:count],2,strict=True):
            for corner in corners:
                for layout in M.LAYOUTS:
                    for f in pair:
                        ref,code,comparison=(M.case(tokenizer,f,corner,layout,i) for i in range(3))
                        assert ref['query']==code['query']==comparison['query'] and ref['source']==code['source']==comparison['source']
                        changed={i for i,(a,b) in enumerate(zip(code['free_ids'],comparison['free_ids'],strict=True)) if a!=b}
                        assert changed and changed<=set(code['source'])
                        out.extend((ref,code,comparison,code))
    return out


def qualify(model,views,corners,weights,binding):
    name='heldout-behavior.json'
    if (OUT/name).exists():
        p=R.read(name)
        assert p['holdout']==binding and p['measurements']==W.sha(OUT/'heldout-style-files.json')
        return p['passed']
    chosen=M.families('confirm')[:32]
    ids={r['family'] for r in chosen}
    baseline=[r for r in views if r['family'] in ids and r['corner'] in corners]
    with M.scope():
        alt=[M.case(model.tok,r,r['corner'],r['layout'],r['archive'],11) for r in baseline]
    W.clean_pass(model,alt,weights,'heldout-style',binding)
    clean=S.load('confirm',binding)
    by=dict(zip(clean['ids'],clean['margin'][:,1],strict=True))
    alt_val=S.load('heldout-style',binding)['margin'][:,1]
    diff=(np.array([by[r['id']] for r in baseline])-alt_val).reshape(-1,2,2,2)
    lexical=np.array([r['lex'] for r in chosen])
    rates=np.stack([(diff[lexical==i]>0).mean(0) for i in (0,1)])
    passed=bool((rates>=V.GATE).all())
    R.write(name,dict(at=C.stamp(),holdout=binding,passed=passed,minimum=float(rates.min()),rates=rates.tolist(),
                      n=len(chosen),measurements=W.sha(OUT/'heldout-style-files.json')))
    print('Selected-corners heldout behavior',passed,'minimum',rates.min(),flush=True)
    return passed


def positive(model,panels,weights):
    t=model.t
    out=[]
    for p in panels:
        x=p['x'].cuda();j=p['recipient'];q=p['q'][j,0]; y=x.clone()
        y[j,q]=x[j-1,q]
        original=V.W.P.full_logits(model,p,x)[:,0]
        edited=V.W.P.full_logits(model,p,y)[:,0]
        assert t.equal(edited[j],original[j-1]) and t.equal(edited[j-1],original[j-1])
        before=V.F.opening_scores(model,original[j],weights[:2]);after=V.F.opening_scores(model,edited[j],weights[:2])
        out.extend((before-after).cpu().numpy())
    return np.asarray(out)


def run():
    d,c,binding=V.checked_plan()
    N.fit_direction=repair_norm
    with W.engine(d,'verdict-native') as (model,weights):
        main,*_=V.data(model,d,'confirm')
        if not qualify(model,main,c['corners'],weights,binding):
            return
        source=rows(model.tok,M.CAUSAL,c['corners'])
        started=time.monotonic();tally=(len(source)+SLAB-1)//SLAB;files=[]
        for idx,sample in enumerate(batched(source,SLAB)):
            name=f'native-{idx:02}.npz';receipt=name.replace('.npz','.json')
            if (OUT/receipt).exists():
                prior=R.read(receipt)
                assert prior['sha']==W.sha(OUT/name) and prior['holdout']==binding
            else:
                check_write(OUT/name,10*2**20)
                result,panels=N.rollout(model,sample,weights,ARMS,
                    lambda step:C.N.progress('verdict-native',idx*40+step,tally*40,started))
                result['positive_control']=positive(model,panels,weights)
                result['corner']=np.asarray([int(x.rsplit('-c',1)[1]) for x in result['ids']])
                R.write(receipt,dict(at=C.stamp(),sha=S.artifact(name,result),design=W.sha(OUT/'design.json'),holdout=binding,views=len(sample)))
            files.append(name)
        S.bind_label('native',files,binding)


def generate():
    import paradigm_window_generate as G

    d,c,binding=V.checked_plan()
    verdict=R.read('analysis.json')
    assert verdict['binding']==binding and verdict['decision']=='prospective_gate_pass'
    raw=S.load('native',binding)
    match=dict(zip(raw['ids'],raw['top'],strict=True))
    with W.engine(d,'verdict-generate') as (model,weights):
        source=rows(model.tok,16,c['corners']);files=[]
        for idx,slab in enumerate(batched(source,128)):
            start=time.monotonic()
            # Check the exact cached prefill against the earlier native readout.
            raw_part,panels=N.rollout(model,slab,weights,N.GEN_ARMS,
                lambda step:C.N.progress('verdict-prefill-cache',step,40,start,part=idx),cache=True)
            for j,key in enumerate(raw_part['ids']):
                for arm in ('sham','full','last20'):
                    assert raw_part['top'][j,N.GEN_ARMS.index(arm)]==match[key][ARMS.index(arm)]
            start=time.monotonic()
            result=G.decode(model,panels,lambda step:C.N.progress('verdict-decode',step,G.CAP,start,part=idx))
            name=f'generation-{idx:02}.json'
            files.append(dict(name=name,sha=R.write(name,dict(holdout=binding,views=result))))
        R.write('generation-files.json',dict(files=files,holdout=binding))
