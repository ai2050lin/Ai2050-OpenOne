"""Freeze behavioral and multi-stage placement predictions before native validation."""

import argparse
from pathlib import Path
import time

import numpy as np

import paradigm_position_field as S
import paradigm_order_material as M
from paradigm_four_report import behavior
from tools.power_analysis import check_phase


R,C,F,X,P=S.R,S.C,S.F,S.X,S.P
PREVIOUS=S.OUT
OUT=R.ROOT/'tests/glm5/result/paradigm_order_20261118_v1'
TEMP=R.ROOT/'tests/glm5_temp'/OUT.name
PEAK=490*2**20
MIDDLE=(8,12,16,20)
LATE=(24,28,30)
SITES=tuple(range(4,10))
EARLY=[0,-1]  # -1 means the first token after the active archive.


def setup():
    S.OUT,S.TMP=OUT,TEMP
    R.OUT,R.TEMP=OUT,TEMP
    R.M.RUN,R.M.SEED,R.M.WIDTH=M.RUN,M.SEED,M.P.F.WIDTH
    R.BATCH,R.SLAB=8,S.SLAB
    R.configure(PEAK)


def sha(path):
    return S.sha(path)


def identity():
    paths=[Path(__file__).resolve(),Path(M.__file__).resolve()]
    return S.registry()|{p.relative_to(R.ROOT).as_posix():sha(p) for p in paths}


def prior_identity():
    return {k:sha(PREVIOUS/k) for k in ('manifest.json','analysis.json','candidate.json','residual-first-write.png')}


def freeze():
    from transformers import AutoTokenizer

    assert not (OUT/'design.json').exists()
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    power=check_phase(.14,128,.20,.01)
    assert power['pass']
    R.write('design.json',dict(at=C.stamp(),run=M.RUN,model=C.Prior.identity(),software=R.versions(),
        provenance=R.metadata(),code=identity(),previous=prior_identity(),materials=M.inventory(tok),
        anchors=F.prior_material.alphabet(tok),power=power,
        geometry=dict(layouts=M.LAYOUTS,edges=M.EDGES,archive_placeholder=M.EMPTY_A,request_placeholder=M.EMPTY_R,
                      archive_width=6,request_width=5,shared_template='Repeated slot headers, archive/facts/role/guide shared with Phase 2974'),
        procedure=dict(
            gate='16 development families. Each paired factor nuisance stratum must be >=75%; every layout by archive 1011->1111 request-style pair must be >=75%. Natural sign reversal is never a gate.',
            trace='Exactly aligned full B0–B32 paired fields, one full valid prefix pair of 41 BF16 boundaries per traced corpus, existing native streaming model. Early common point is first post-archive token at after-B0 boundary.',
            selection='For each of the two moves, score code-prose residual contrasts as change from layout i to layout j with bank-consistent standardized covariance against its output effect. Highest middle block/anchor and highest late block/different anchor; both legal for all layouts. Include fixed B0 point. Score and freeze both even if routes overlap. Same original B32 earliest-output 32 G/U receivers ranked by write contrast; fixed disjoint random pool.',
            predictions='Two outcomes are paired moves D2-D0 and D3-D2 where D=print margin for code archive minus prose archive. For each estimate constant discovery mean, B0-only ridge, and two route-specific three-point ridge predictors before seeing holdout. Report MAE/MSE plus between-route differences. Native effects evaluated independently.',
            checks='99% family bootstrap for natural signs, edge attenuation and matched contrasts. A shared path needs both expected edges and on both a positive 99% raw joint closure of at least 10% and positive equal-dosed specificity versus same-position sign-scrambled control. Compare one-stage common B0, nearby native B0 control, and later B32 residual/MLP and G/U separately. Freeze doses and schedules before holdout.',
            resources='Local raw BF16 one GPU with CPU weight streaming. Batch 8 width 160. D-only disk preflight per shard. 128 new clean families, causal on first 64 families, 128 lockbox hashes without model access.'
        )))
    print('Frozen design',sha(OUT/'design.json'),flush=True)


def pairs(data):
    layout=data['layout'][::2].reshape(-1,3)
    assert (layout==M.LAYOUTS).all()
    assert (data['archive'].reshape(-1,2)==[0,1]).all()
    assert (data['cell']==15).all()
    assert (data['family'][::2].reshape(-1,3)==data['family'][::6,None]).all()
    output=(data['margin'][1::2,1]-data['margin'][::2,1]).reshape(-1,3)
    return output,data['bank'][::6]


def at_point(data,point):
    block,site=point
    pos=(data['source'][::2,-1]+1) if site==-1 else data['sites'][::2,site]
    assert (pos>data['source'][::2,-1]).all() and (pos<data['query'][::2,0]).all()
    return data['trace'][np.arange(len(pos)),block,pos,3].reshape(-1,3)


def edge_matrix(data,edge,points):
    _,a,b=edge
    ia,ib=M.LAYOUTS.index(a),M.LAYOUTS.index(b)
    values=[at_point(data,pt) for pt in points]
    return np.stack([f[:,ib]-f[:,ia] for f in values],1)


def fit(a,y):
    mean=a.mean(0)
    scale=a.std(0).clip(1e-5)
    z=(a-mean)/scale
    b=np.linalg.solve(z.T@z+2*np.eye(a.shape[1]),z.T@(y-y.mean()))
    return dict(y=float(y.mean()),mean=mean.tolist(),scale=scale.tolist(),beta=b.tolist())


def discover():
    data=S.load('discover')
    y,bank=pairs(data)
    assert len(y)==M.SIZES['discovery']
    routes={}
    summary={}
    for edge in M.EDGES:
        name,a,b=edge
        difference=y[:,M.LAYOUTS.index(b)]-y[:,M.LAYOUTS.index(a)]
        def ranked(blocks):
            coords=[[layer,site] for layer in blocks for site in SITES]
            features=edge_matrix(data,edge,coords)
            values=[]
            for code in (0,1):
                z=features[bank==code]; h=difference[bank==code]
                values.append(((z-z.mean(0))*(h-h.mean())[:,None]).mean(0)/z.std(0).clip(1e-5))
            shares=np.minimum(np.abs(values[0]),np.abs(values[1]))*((values[0]*values[1])>0)
            order=np.argsort(-shares,kind='stable')
            return [(coords[i],float(shares[i])) for i in order]
        mid,ms=ranked(MIDDLE)[0]
        end,ls=next((p,s) for p,s in ranked(LATE) if p[1]!=mid[1])
        routes[name]=[EARLY,mid,end]
        summary[name]=dict(mean=float(difference.mean()),point_scores=[ms,ls])
    predicted={}
    for edge in M.EDGES:
        name,a,b=edge
        actual=y[:,M.LAYOUTS.index(b)]-y[:,M.LAYOUTS.index(a)]
        predicted[name]={key:fit(edge_matrix(data,edge,value),actual)
                         for key,value in {'early':[EARLY],**routes}.items()}
    scores=np.linalg.norm(data['units'].sum(0),axis=0)
    order=np.argsort(-scores,kind='stable')
    rng=np.random.default_rng(M.SEED)
    R.write('candidate.json',dict(at=C.stamp(),design=sha(OUT/'design.json'),behavior=sha(OUT/'behavior.json'),
        items=sha(OUT/'discover-files.json'),routes=routes,summary=summary,layout_means=y.mean(0).tolist(),
        forecast=predicted,receivers=order[:32].tolist(),random=rng.choice(order[64:],32,replace=False).tolist()))
    print('Frozen routes',routes,flush=True)


def discovery_run():
    import torch
    design=R.read('design.json')
    assert design['code']==identity() and design['software']==R.versions() and design['previous']==prior_identity()
    assert not (OUT/'candidate.json').exists()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic();C.N.progress('order-load',0,2,start)
        assert C.Prior.identity()==design['model']
        model=C.N.Engine();C.N.progress('order-load',2,2,start)
        try:
            weights=X.anchors(model,design)
            data=M.build(model.tok,'pilot')
            assert M.digest(data)==design['materials']['pilot']['identity']
            S.collect(model,data[1],'pilot-factor',weights)
            S.collect(model,data[0],'pilot-position',weights)
            all_factors=behavior(S.load('pilot-factor'),data[1],model.tok)
            margins=S.load('pilot-position')['margin'][:,1].reshape(16,2,3,2)
            rates=((margins[:,1]-margins[:,0])>0).mean(0)
            passed=bool((rates>=.75).all() and all(v['passed'] for v in all_factors.values()))
            R.write('behavior.json',dict(at=C.stamp(),passed=passed,rates=rates.tolist(),factors=all_factors))
            print('Gates',passed,rates.tolist(),flush=True)
            if passed:
                data=M.build(model.tok,'discovery')
                assert M.digest(data)==design['materials']['discovery']['identity']
                S.collect(model,data[0],'discover',weights,trace=True)
                discover()
        finally:
            model.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','discover'))
    args=cli.parse_args()
    setup()
    (freeze if args.action=='freeze' else discovery_run)()
