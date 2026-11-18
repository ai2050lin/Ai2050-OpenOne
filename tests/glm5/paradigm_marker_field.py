"""Freeze the independent marker experiment and a wider common residual path."""

import argparse
from itertools import batched
from pathlib import Path
import math
import time

import numpy as np

import paradigm_marker_material as M
import paradigm_order_field as O
from paradigm_four_report import behavior
from deploy.storage_policy import check_write
from tools.power_analysis import check_phase


S,R,C,F,X,P=O.S,O.R,O.C,O.F,O.X,O.P
PREVIOUS=O.OUT
OUT=R.ROOT/'tests/glm5/result/paradigm_marker_20261118_v1'
TEMP=R.ROOT/'tests/glm5_temp'/OUT.name
PEAK=368*2**20
BANDS=((4,8,12),(16,20,24,28))
VIEWS=24
RMS_LINE=.0005


def setup():
    S.OUT,S.TMP=OUT,TEMP
    R.OUT,R.TEMP=OUT,TEMP
    R.M.RUN,R.M.SEED,R.M.WIDTH=M.RUN,M.SEED,M.P.F.WIDTH
    R.BATCH,R.SLAB=8,S.SLAB
    R.configure(PEAK)


def sha(path):
    return O.sha(path)


def source_id():
    paths=[Path(__file__).resolve(),Path(M.__file__).resolve()]
    return O.identity()|{p.relative_to(R.ROOT).as_posix():sha(p) for p in paths}


def previous():
    return {p:sha(PREVIOUS/p) for p in ('manifest.json','analysis.json','marker-plan.json','candidate.json')}


def freeze():
    from transformers import AutoTokenizer

    assert not (OUT/'design.json').exists()
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    power=check_phase(.14,128,.20,.01)
    assert power['pass']
    R.write('design.json',dict(at=C.stamp(),run=M.RUN,model=C.Prior.identity(),software=R.versions(),
        provenance=R.metadata(),code=source_id(),previous=previous(),materials=M.inventory(tok),power=power,
        anchors=F.prior_material.alphabet(tok),markers={'old':M.OLD,'new':M.NEW},bands=BANDS,
        plan=dict(
            factors='2x2 inactive archive/request markers with identical active samples and token geometry. All three layouts and both archive styles per marker setting. Contrast B2-B0 archive movement and B3-B2 request movement, 1111. Report independent effects and their interaction for all layouts and movement contrasts, family 99% intervals.',
            behavior='16 disjoint pilot families. Four-factor paired per-nuisance accuracy and 4 markers x 3 layouts x 2 archive condition paired format response each >=.75. No natural reversal requirement.',
            candidates='One schedule applied to both edges: post-B0 active archive next four tokens, two earlier/later post-block 8-token windows chosen from common post-late-archive suffix via cross-bank standardized residual projection covariance in discovery, and all common suffix states at post-B31; exclude archive source and free onset E. 32 shared B32 E G/U units and random controls.',
            trials='All 128 holdout families for the behavior factorial and forecasts. Freeze first 64 families for original markers and the discovery-frozen most weakening single marker (higher layout-2 effect among m1 and m2). Other two marker settings measured in clean factorial only.',
            predictions='For both movements compare discovery means per marker with B0 four-token pooled forecast and four-boundary pooled residual forecast. Use marginal code/prose residual contrasts and both layouts before seeing output labels. State feature input and report by marker and pooled family.',
            causal='Whole-prefix same-shape rerun under native same-family donor edits. Direction control at exactly the same positions and per-boundary Frobenius matched dose. Narrow B0-only, dense B31-only, joint schedule, direction controls, exact undo, local B32 residual/MLP split and G/U recovery. Need >=10% signed closure and positive 99% CI versus equal-dose controls (>=98% coverage) for both natural edges to accept a shared pathway.',
            resources='Serial BF16 local CUDA/CPU. Keep one 41-boundary full discovery pair plus all B0–B32 discovery summaries. Confirmation has native projections of selected boundaries only. D reserve checked by shard. Lockbox no model execution.'
        )))
    print('Frozen material and gate',sha(OUT/'design.json'),flush=True)


def shape(data):
    markers=np.array([int(x[1]) for x in data['ids']])
    n=len(markers)//VIEWS
    assert (markers.reshape(n,4,6)==np.arange(4)[None,:,None]).all()
    assert (data['layout'].reshape(n,4,3,2)==np.array(M.LAYOUTS)[None,None,:,None]).all()
    assert (data['archive'].reshape(n,4,3,2)==[0,1]).all()
    assert (data['cell']==15).all()
    assert (data['family'].reshape(n,VIEWS)==data['family'][::VIEWS,None]).all()
    return n


def increments(data):
    n=shape(data)
    m=data['margin'][:,1].reshape(n,4,3,2)
    return m[...,1]-m[...,0]


def positions(t,p,group):
    act=t.as_tensor([r['source'][-1] for r in p['rows']],device='cuda')
    sites=t.as_tensor([r['sites'] for r in p['rows']],device='cuda')
    end=p['q'][:,0]
    kind=group['kind']
    if kind=='proximal':
        out=act[:,None]+t.arange(1,5,device='cuda')
        mask=t.ones_like(out,dtype=t.bool)
    elif kind=='windows':
        start=sites[:,group['sites']]
        out=(start[:,:,None]+t.arange(2,device='cuda')).flatten(1)
        mask=t.ones_like(out,dtype=t.bool)
    else:
        first=sites[:,4]
        window=end-first
        out=first[:,None]+t.arange(int(window.max()),device='cuda')
        mask=out<end[:,None]
    assert (out[mask]>act[:,None].expand_as(out)[mask]).all() and (out[mask]<end[:,None].expand_as(out)[mask]).all()
    if kind!='suffix':
        sorted_=out.sort(1).values
        assert (sorted_[:,1:]!=sorted_[:,:-1]).all()
    return out,mask


def discover_project(data,group):
    n=shape(data)
    branch=data['trace'][:,group['block'],:,3]
    sites=data['sites'][::2]
    if group['kind']=='proximal':
        where=data['source'][::2,-1,None]+np.arange(1,5)
    elif group['kind']=='windows':
        where=(sites[:,group['sites'],None]+np.arange(2)).reshape(len(sites),-1)
    else:
        valid=(np.arange(branch.shape[1])[None,:]>=sites[:,4,None])&(np.arange(branch.shape[1])[None,:]<data['query'][::2,0,None])
        return ((branch*valid).sum(1)/valid.sum(1)).reshape(n,4,3)
    result=branch[np.arange(len(sites))[:,None],where].mean(1)
    return result.reshape(n,4,3)


def fit_pack(x,actual):
    mean=x.mean(0)
    y=actual.mean(0)
    z=(x-mean).reshape(-1,x.shape[-1])
    target=(actual-y).ravel()
    spread=z.std(0).clip(1e-5)
    z=z/spread
    beta=np.linalg.solve(z.T@z+2*np.eye(z.shape[-1]),z.T@target)
    return dict(baseline=y.tolist(),mean=mean.tolist(),spread=spread.tolist(),beta=beta.tolist())


def choose():
    data=S.load('discover')
    raw=increments(data)
    banks=data['bank'][::VIEWS]
    archive=raw[:,:,1]-raw[:,:,0]
    request=raw[:,:,2]-raw[:,:,1]
    labels=[archive-archive.mean(0),request-request.mean(0)]
    stages=[dict(block=0,kind='proximal')]
    ranked=[]
    for band in BANDS:
        best=None
        for b in band:
            features=[]
            for site in range(4,10):
                features.append(discover_project(data,dict(block=b,kind='windows',sites=[site])))
            z=np.stack(features,-1)
            score=np.zeros(z.shape[-1])
            for e,y in enumerate(labels):
                x=z[:,:,e+1]-z[:,:,e]
                x-=x.mean(0)
                split=[]
                for bank in (0,1):
                    a=x[banks==bank]
                    split.append((a*y[banks==bank,:,None]).mean((0,1))/a.std((0,1)).clip(1e-5))
                score+=np.minimum(abs(split[0]),abs(split[1]))*((split[0]*split[1])>0)
            order=np.argsort(-score,kind='stable')
            # A two-token window cannot overlap another at any constituent input.
            selected=[]
            all_sites=data['sites'][::2]
            for index in order+4:
                if all((abs(all_sites[:,index]-all_sites[:,other])>=2).all() for other in selected):
                    selected.append(int(index))
                if len(selected)==4:
                    break
            assert len(selected)==4
            candidate=(float(score[np.array(selected)-4].sum()),b,selected)
            if best is None or candidate[0]>best[0]:
                best=candidate
        rank,b,sites=best
        ranked.append(rank)
        stages.append(dict(block=b,kind='windows',sites=sites))
    stages.append(dict(block=31,kind='suffix'))
    x=np.stack([discover_project(data,g) for g in stages],-1)
    fits={}
    for e,(name,_,_) in enumerate(M.EDGES):
        a=raw[:,:,e+1]-raw[:,:,e]
        f=x[:,:,e+1]-x[:,:,e]
        fits[name]=dict(early=fit_pack(f[:,:,:1],a),joint=fit_pack(f,a))
    weaker=max((1,2),key=lambda marker:raw[:,marker,1].mean())
    units=np.linalg.norm(data['units'].sum(0),axis=0)
    order=np.argsort(-units,kind='stable')
    rng=np.random.default_rng(M.SEED)
    R.write('candidate.json',dict(at=C.stamp(),design=sha(OUT/'design.json'),behavior=sha(OUT/'behavior.json'),
        discovery=sha(OUT/'discover-files.json'),stages=stages,scores=ranked,markers=[0,weaker],
        marker_layout_means=raw.mean(0).tolist(),predictions=fits,
        receivers=order[:32].tolist(),random=rng.choice(order[64:],32,replace=False).tolist()))
    print('Frozen selected markers/stages',[0,weaker],stages,flush=True)


def confirm(model,rows,weights,candidate,binding):
    t=model.t
    d=S.contrast(model,weights)
    count=math.ceil(len(rows)/S.SLAB);start=time.monotonic();files=[]
    by_block={g['block']:g for g in candidate['stages']}
    for part,group in enumerate(batched(rows,S.SLAB)):
        file=f'confirm-{part:02}.npz';receipt=f'confirm-{part:02}.json'
        design=sha(OUT/'design.json')
        if (OUT/receipt).exists():
            found=R.read(receipt)
            assert found['design']==design and found['holdout']==binding and found['sha']==sha(OUT/file)
        else:
            check_write(OUT/file,10*2**20,operation='四标记留出逐批空间')
            panels=X.panels(model,group,0)
            for p in panels:
                p['features']=[]
            C.N.progress('marker-confirm',part*40,count*40,start)
            for b in range(40):
                with model.block(b) as block:
                    for p in panels:
                        x=p['x'].cuda()
                        q,k,v=model.qkv(block,x,[s.cuda() for s in p['angles']])
                        o=model.attention(block,q,k,v,p['mask'].cuda())
                        out=model.residual(block,x,o)
                        if b in by_block:
                            places,use=positions(t,p,by_block[b])
                            j=t.arange(len(x),device='cuda')[:,None]
                            pro=(out[j,places].float()@d)*use
                            p['features'].append((pro.sum(1)/use.sum(1)).cpu().numpy())
                        p['x']=out.cpu()
                if (b+1)%5==0:
                    C.N.progress('marker-confirm',part*40+b+1,count*40,start)
            data=[]
            for p in panels:
                x=P.full_logits(model,p,p['x'].cuda())
                row=S.meta(p)
                row.update(margin=F.probe(model,p,x,weights),top=x.argmax(-1).cpu().numpy(),
                           project=np.stack(p['features'],1))
                data.append(row)
            raw={k:np.concatenate([r[k] for r in data]) for k in data[0]}
            R.write(receipt,dict(sha=S.artifact(file,raw),design=design,holdout=binding,
                                 at=C.stamp(),views=len(group)))
        files.append(file)
    S.bind_label('confirm',files,binding)


def discovery():
    import torch
    design=R.read('design.json')
    assert design['code']==source_id() and design['previous']==previous() and design['software']==R.versions()
    assert not (OUT/'candidate.json').exists()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        tic=time.monotonic();C.N.progress('marker-factor-load',0,2,tic)
        assert design['model']==C.Prior.identity()
        model=C.N.Engine();C.N.progress('marker-factor-load',2,2,tic)
        try:
            w=X.anchors(model,design)
            inp=M.build(model.tok,'pilot')
            assert M.O.digest(inp)==design['materials']['pilot']['identity']
            S.collect(model,inp[1],'pilot-factor',w)
            S.collect(model,inp[0],'pilot-position',w)
            gate=behavior(S.load('pilot-factor'),inp[1],model.tok)
            y=S.load('pilot-position')['margin'][:,1].reshape(16,4,2,3,2)
            rates=(y[:,:,1]>y[:,:,0]).mean(0)
            ok=bool((rates>=.75).all() and all(s['passed'] for s in gate.values()))
            R.write('behavior.json',dict(at=C.stamp(),passed=ok,rates=rates.tolist(),factors=gate))
            print('Gate',ok,'minimum',float(rates.min()),flush=True)
            if ok:
                inp=M.build(model.tok,'discovery')
                assert M.O.digest(inp)==design['materials']['discovery']['identity']
                S.collect(model,inp[0],'discover',w,trace=True)
                choose()
        finally:
            model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','discover'))
    args=parser.parse_args()
    setup()
    (freeze if args.action=='freeze' else discovery)()
