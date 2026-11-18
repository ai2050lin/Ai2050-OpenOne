"""Freeze a decisive conditional prediction and native-path test on Qwen3-14B."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
if __name__=='__main__':
    sys.modules['paradigm_verdict_run']=sys.modules[__name__]
import paradigm_verdict_material as M
import paradigm_window_run as W
import paradigm_window_native as N
from paradigm_four_report import behavior
from tools.power_analysis import check_phase

R,C,S,F=W.R,W.C,W.S,W.F
PREVIOUS=W.OUT.with_name('paradigm_window_20261118_v2')
OUT=W.OUT.with_name('paradigm_verdict_20261118_v1')
ASSETS=W.sources
PREVIOUS_PROTOCOL=('design.json','candidate.json','analysis.json','manifest.json','generation-files.json')
LAYERS=(0,1,2,6)
GATE=.75
RULES=dict(sign_accuracy=.75,sign_gain=.05,minimum_margin_advantage=.10,minimum_closure=.10,dose=.03,cover=.98,confidence=.99)
FACTORS=np.asarray([[1 if c&b else -1 for b in (4,2,1)] for c in range(8)])
PROGRAMS=('paradigm_verdict_native.py','paradigm_verdict_report.py')


def setup():
    W.OUT,W.TEMP,W.PREVIOUS=OUT,W.TEMP.with_name(OUT.name),PREVIOUS
    W.M.RUN,W.M.SEED=M.RUN,M.SEED
    W.PEAK=157*2**20
    W.SLAB=256
    N.OUT=OUT
    W.sources=sources
    W.setup()


def sources():
    own=(Path(__file__),Path(M.__file__))
    return ASSETS()|{path.relative_to(R.ROOT).as_posix():W.sha(path) for path in own}


def freeze():
    from transformers import AutoTokenizer

    if (OUT/'design.json').exists():
        raise FileExistsError('Design already frozen')
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    power=check_phase(.14,M.SIZES['confirm'],.20,.01)
    assert power['pass']
    d=dict(at=C.stamp(),run=M.RUN,model=C.Prior.identity(),software=R.versions(),provenance=R.metadata(),
           code=sources(),previous={name:W.sha(PREVIOUS/name) for name in PREVIOUS_PROTOCOL},
           materials=M.inventory(tok),anchors=F.prior_material.alphabet(tok),thresholds=RULES,power=power,
           scope='First-token code/prose archive effect of shifting request with archive fixed late. Does the conditional mechanistic route generalize on original Qwen3-14B BF16?',
           method=dict(
                causal_geometry='Within every family and realization, length, all slots, readout and position ids fixed. Three independent token domains are headers, facts, instruction. Exact preceding frames as separate bridge on 64 new families, no candidate fitting on bridge.',
                populations='24 pilot families (both lex realizations); 32 discovery families (lex 0); 128 confirmation families (64 per lex, half lexically unseen in fitting); first 64 for causal, 16 for generation only if predictive and causal gates both pass; 128 lockbox hash only.',
                behavior='Four-axis pilot and every lex x corner x request-layout x archive paired format margin must pass .75. Relative test, not absolute script conformity. Only exposure after fit freeze.',
                predictor='Discovery single ridge penalty 8 per coefficient after training standardization. Position-only uses token geometry, bank and marker. Text-only adds all three crossed phrase domains and interactions. Native adds four original residual projections minus prose/archive at E-before positions; B0, B8, B20, B31 last20. No target logits or raw anchors. Sign is Y>0. Mechanism prediction >=75% and >=5pp higher than text means with positive family 99% CI pooled and no reversal of advantage in either wording panel. Text is separately compared to geometry.',
                candidate='Select only lowest and highest discovery mean among the 8 equal-length corners, save both. Confirmation must reproduce opposite natural CIs. No reselection. Preserve original wide B0/B8/B20/B31 and compact B31 last20 schedules with same early stages; compare donors and controls at the same positions. Both schedules counted as two planned candidates; test with 99% family intervals.',
                causal='Native donor is same-family prose vs code, direction control from permuted orthogonal and a same-format archive alternate natural payload; quantities matched by per-token monotone BF16 bracketing and whole-stage verification within 3%. Require >=98% complete family coverage per control and corner. Each of two opposite corners requires >=10% point attenuation, lower 99% slope on attenuation >0, >=0.10 margin vs each control with 99% lower bound above zero. Exact sham/native restore and final E swap positive control are technical prerequisites. A route passes only when a single frozen arm passes both corners.',
                exit='Decide about a usable shared mechanism in these controlled tasks at the frozen useful-effect criteria, with no unplanned reselecting. If raw bridge fails we cannot attribute a localized disagreement to new wording alone. If neither tested native path works and prediction gate fails, conclude do not adopt as general mechanism; retain any valid conditional law. Free decoding only when main predictive and causal tests both pass, because Phase 2978 already tested the failed reference paths on full answers.'))
    print('Frozen material and decision contract',R.write('design.json',d),flush=True)


def data(model,design,split):
    material=M.build(model.tok,split)
    assert M.W.digest(material)==design['materials'][split]['sha256']
    return material


def pilot():
    d=R.read('design.json')
    with W.engine(d,'verdict-pilot') as (model,weights):
        main,_,opposite,factor=data(model,d,'pilot')
        W.clean_pass(model,[*main,*opposite],weights,'pilot-format')
        raw=S.load('pilot-format')
        a,b=np.split(raw['margin'][:,1],2)
        groups=np.asarray([f['lex'] for f in M.families('pilot')])
        rates=np.stack([((a-b).reshape(-1,8,2,2)[groups==i]>0).mean(0) for i in (0,1)])
        W.clean_pass(model,factor,weights,'pilot-axes')
        gates=behavior(S.load('pilot-axes'),factor,model.tok)
        ok=bool((rates>=GATE).all() and all(x['passed'] for x in gates.values()))
        R.write('behavior.json',dict(at=C.stamp(),passed=ok,minimum=float(rates.min()),rates=rates.tolist(),axes=gates))
        print('Behavior gate',ok,'minimum',rates.min(),flush=True)
        if ok:
            main,*_=data(model,d,'discovery')
            W.clean_pass(model,main,weights,'discovery')


def movement(raw,corners=8):
    groups=len(raw['ids'])//(corners*4)
    z=raw['margin'][:,1].reshape(groups,corners,2,2)
    x=raw['feature'].reshape(groups,corners,2,2,-1)
    assert (raw['layout'].reshape(groups,corners,2,2)==np.array(M.LAYOUTS)[None,None,:,None]).all()
    assert (raw['archive'].reshape(groups,corners,2,2)==[0,1]).all()
    y=z[:,:,1,1]-z[:,:,1,0]-z[:,:,0,1]+z[:,:,0,0]
    v=x[:,:,1,1]-x[:,:,1,0]-x[:,:,0,1]+x[:,:,0,0]
    f=raw['frame'].reshape(groups,corners,2,2)[:,:,0,0]
    return y,v,f


def predictors(raw):
    y,v,f=movement(raw)
    n=len(y)
    def property(name,sub=None):
        a=raw[name]
        if sub is not None:
            a=a[:,sub]
        a=a.reshape(n,8,2,2)[:,:,0,0]
        return a[...,None]
    geo=np.concatenate([property('marker'),property('bank'),property('query',0),property('source',-1)],2)
    three=FACTORS[f%8]
    h,s,i=three.transpose(2,0,1)
    text=np.concatenate((geo,three,np.stack((h*s,h*i,s*i,h*s*i),2)),2)
    return y,dict(geometry=geo,text=text,native=np.concatenate([text,v[:,:,LAYERS]],2),
                  residual=np.concatenate([geo,v[:,:,LAYERS]],2))


def fit(x,target):
    v=x.reshape(-1,x.shape[-1]).astype(np.float64)
    y=target.ravel().astype(np.float64)
    mean=v.mean(0);std=v.std(0).clip(1e-5)
    z=(v-mean)/std
    intercept=float(y.mean())
    b=np.linalg.solve(z.T@z+8*np.eye(z.shape[1]),z.T@(y-intercept))
    return dict(mean=mean.tolist(),spread=std.tolist(),intercept=intercept,beta=b.tolist())


def predict(x,p):
    return p['intercept']+((x-np.asarray(p['mean']))/np.asarray(p['spread']))@np.asarray(p['beta'])


def candidate():
    assert R.read('behavior.json')['passed'] and not (OUT/'candidate.json').is_file()
    y,x=predictors(S.load('discovery'))
    avg=y.mean(0)
    low,high=int(avg.argmin()),int(avg.argmax())
    assert low!=high
    execution={p:W.sha(Path(__file__).with_name(p)) for p in PROGRAMS}
    payload=dict(at=C.stamp(),design=W.sha(OUT/'design.json'),behavior=W.sha(OUT/'behavior.json'),
                 discovery=W.sha(OUT/'discovery-files.json'),execution=execution,
                 fits={k:fit(v,y) for k,v in x.items()},means=avg.tolist(),corners=[low,high],directions=[-1,1])
    print('Frozen candidate',R.write('candidate.json',payload),payload['corners'],flush=True)


def checked_plan():
    d=R.read('design.json');c=R.read('candidate.json');sha=W.sha(OUT/'candidate.json')
    assert d['thresholds']==RULES and c['design']==W.sha(OUT/'design.json') and c['behavior']==W.sha(OUT/'behavior.json')
    assert c['discovery']==W.sha(OUT/'discovery-files.json') and R.read('behavior.json')['passed']
    assert all(W.sha(Path(__file__).with_name(k))==v for k,v in c['execution'].items())
    return d,c,sha


def confirmation():
    d,_,binding=checked_plan()
    with W.engine(d,'verdict-confirm') as (model,weights):
        main,anchors,*_=data(model,d,'confirm')
        W.clean_pass(model,main,weights,'confirm',binding)
        W.clean_pass(model,anchors,weights,'bridge',binding)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('freeze','pilot','candidate','confirm','native','generate','report'))
    a=p.parse_args().stage
    setup()
    if a in ('native','generate'):
        import paradigm_verdict_native as native
        (native.run if a=='native' else native.generate)()
    elif a=='report':
        import paradigm_verdict_report as report
        report.run()
    else:
        {'freeze':freeze,'pilot':pilot,'candidate':candidate,'confirm':confirmation}[a]()
