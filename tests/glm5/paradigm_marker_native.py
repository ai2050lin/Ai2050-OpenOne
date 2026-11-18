"""Native multi-position cuts with equal geometry and chunked BF16 dose matching."""

import argparse
from itertools import batched
from pathlib import Path
import time

import numpy as np

import paradigm_marker_field as A
import paradigm_order_native as K
from deploy.storage_policy import check_write


R,C,M,S,F,X,P,OUT=A.R,A.C,A.M,A.S,A.F,A.X,A.P,A.OUT
PLAN='holdout-plan.json'
SLAB=64
EARLY=('sham','restore','proximal','joint','matched','scramble','suffix')
ARMS=EARLY+tuple('joint_'+key for key in K.FORKS)


def sha(name):
    return A.sha(OUT/name)


def code():
    return A.sha(Path(__file__).resolve())


def freeze():
    assert not (OUT/PLAN).exists()
    design=R.read('design.json');candidate=R.read('candidate.json')
    assert R.read('behavior.json')['passed'] and candidate['design']==sha('design.json')
    assert candidate['behavior']==sha('behavior.json') and candidate['discovery']==sha('discover-files.json')
    assert [x['kind'] for x in candidate['stages']]==['proximal','windows','windows','suffix']
    R.write(PLAN,dict(at=C.stamp(),design=sha('design.json'),candidate=sha('candidate.json'),code=code(),
        reused_native=A.sha(Path(K.__file__).resolve()),material=design['materials']['confirm'],arms=ARMS,
        markers=candidate['markers'],families=M.CAUSAL,method=dict(
            sample='First 64 unopened holdout families. Old/old and the discovery-frozen weakening single marker. All three layouts each with aligned code recipient and prose donor. Initial 4 families run same-shape numeric controls before any subsequent native slab.',
            positions='Full post-block native donor states at B0 next-four, two disjoint-window layers, and at B31 every common suffix token after late archive slot through E-1. E and direct archive source never overwritten. Narrow B0 and B31-only controls are separately logged.',
            matched='Same example, same block and exact same set of token indices, so both distances exactly match per token. Frozen signed coordinate permutation per token, orthogonalized to native donor ray as a stage vector, own branch state used at each next edit. Match actual BF16 Frobenius dose at each boundary, <=3% gap, >=.003, cosine >=.90 against own ray. Same .5–8x grid as earlier, evaluated in chunks. Require >=98% of pairs across all relevant stages; log actual orthogonality to donor direction.',
            restoration='Apply donor write to the restore arm and reinstall original same-location values at each boundary. Unedited and restored full-vocabulary logits, plus untouched donors, must be bit-identical. Propagate the whole causally affected sequence with native BF16 shape.',
            downstream='At B32 E on joint branch take post-attention residual only, MLP write only, or their native combination. In these reference arms other positions are original. On joint continued suffix restore candidate G, U, or G/U, disjoint random and matched equal-dose directions at B32 E. Log sign, phase-specific natural denominator and absolute restoration versus random.',
            claims='99 percent family bootstrap, direction taken from predeclared move. Shared route must on both naturally sign-flipping edges remove >=10% of natural mean contrast with positive 99% interval and beat same-geometry matched control with coverage. Smaller supported local contributions and opposite effects get separate status.'
        )))
    print('Frozen wider native intervention',sha(PLAN),flush=True)


def grid_stats(t,base,ray,goal,grid):
    """Keep only norm and cosine across the grid, limiting dense-suffix memory."""
    unit=ray/ray.norm(dim=-1).clamp_min(1e-12)[:,None]
    original=base.float()
    norms=[];cosines=[]
    for levels in grid.split(12):
        d=(goal[:,None]*levels[None,:])[:,:,None]*unit[:,None]
        shift=(original[:,None]+d).to(base.dtype).float()-original[:,None]
        n=shift.norm(dim=-1)
        norms.append(n)
        cosines.append((shift*unit[:,None]).sum(-1)/n.clamp_min(1e-12))
    return unit,t.cat(norms,1),t.cat(cosines,1)


def match(t,left,right,ordinary,rotated):
    d1=ordinary.float()-left.float()
    d2=rotated.float()-right.float()
    n1=d1.norm(dim=-1);n2=d2.norm(dim=-1)
    goal=t.maximum(n1,n2)
    grid=t.as_tensor(P.GRID,device=left.device,dtype=t.float32)
    u,a,ca=grid_stats(t,left,d1,goal,grid)
    v,b,cb=grid_stats(t,right,d2,goal,grid)
    larger=t.maximum(a[:,:,None],b[:,None,:])
    gaps=(a[:,:,None]-b[:,None,:]).abs()/larger.clamp_min(1e-12)
    eligible=(gaps<=P.TOL)&(t.minimum(a[:,:,None],b[:,None,:])>=P.ROW_DOSE)&(ca[:,:,None]>=P.COS_MIN)&(cb[:,None,:]>=P.COS_MIN)
    distance=(a[:,:,None]-goal[:,None,None]).abs()+(b[:,None,:]-goal[:,None,None]).abs()
    alternative=gaps+distance/goal[:,None,None].clamp_min(P.ROW_DOSE)
    obj=t.where(eligible.any((1,2))[:,None,None],distance.masked_fill(~eligible,t.inf),alternative)
    pick=obj.flatten(1).argmin(-1)
    i,j=pick//len(grid),pick%len(grid)
    ix=t.arange(len(left),device=left.device)
    p=(left.float()+(grid[i]*goal)[:,None]*u).to(left.dtype)
    q=(right.float()+(grid[j]*goal)[:,None]*v).to(right.dtype)
    receipt=t.stack((n1,n2,a[ix,i],b[ix,j],gaps[ix,i,j],eligible[ix,i,j],
                    ca[ix,i],cb[ix,j],n1,n2,n1,n2),1)
    return p,q,receipt


def panel(model,p,c):
    t=model.t
    assert len(p['rows'])==8 and [r['archive'] for r in p['rows']]==[0,1]*4
    p['pair']=t.arange(1,8,2,device='cuda')
    p['src']=t.as_tensor([r['source'][-1] for r in p['rows'][1::2]],device='cuda')
    p['lo']=int(p['src'].min())+1
    p['hi']=int(p['q'][:,0].max())+1
    p['where']={st['block']:A.positions(t,p,st) for st in c['stages']}
    p['doses']={};p['geometry']={};p['gu_doses']={};p['gua']={}


def apply(model,p,states,original,block,signs,perm):
    t=model.t
    j=p['pair'][:,None]
    places,valid=p['where'][block]
    q=places[p['pair']]
    keep=valid[p['pair']]
    donor=original[j-1,q]
    basis=states['matched'][j,q]
    carrier=states['scramble'][j,q]
    ordinary=t.where(keep[:,:,None],donor,basis)
    d=(donor.float()-carrier.float())*keep[:,:,None]
    vector=d[:,:,perm]*signs
    d=d.flatten(1)
    v=vector.flatten(1)
    v=v-d*(v*d).sum(-1,keepdim=True)/d.square().sum(-1,keepdim=True).clamp_min(1e-12)
    wrong=(carrier.float()+v.view_as(carrier)).to(carrier.dtype)
    first,second,dose=match(t,basis.flatten(1),carrier.flatten(1),ordinary.flatten(1),wrong.flatten(1))
    cosine=((second.float()-carrier.float().flatten(1))*d).sum(-1)/(
        (second.float()-carrier.float().flatten(1)).norm(dim=1)*d.norm(dim=1)).clamp_min(1e-12)
    actual=p['q'][p['pair'],0,None]-q
    dist=q-p['src'][:,None]
    p['doses'][block]=t.cat((dose,cosine[:,None]),1).cpu().numpy()
    p['geometry'][block]=t.stack((keep.sum(1),dist.masked_fill(~keep,1000).min(1).values,
        dist.masked_fill(~keep,-1).max(1).values,actual.masked_fill(~keep,1000).min(1).values,
        actual.masked_fill(~keep,-1).max(1).values),1).cpu().numpy()
    states['matched']=K.put(states['matched'],j,q,first.view_as(basis))
    states['scramble']=K.put(states['scramble'],j,q,second.view_as(carrier))
    states['joint']=K.put(states['joint'],j,q,t.where(keep[:,:,None],donor,states['joint'][j,q]))
    restored=K.put(states['restore'],j,q,t.where(keep[:,:,None],donor,original[j,q]))
    restored[j,q]=original[j,q]
    assert t.equal(restored,original)
    states['restore']=restored
    if block==0:
        states['proximal']=states['joint']
    if block==31:
        states['suffix']=K.put(original,j,q,t.where(keep[:,:,None],donor,original[j,q]))


def slab(model,rows,weights,c,part,total,started):
    t=model.t
    panels=X.panels(model,rows,0)
    for p in panels:
        panel(model,p,c)
    rng=np.random.default_rng(M.SEED+1)
    choices={b['block']:(t.as_tensor(rng.choice([-1,1],size=model.width),device='cuda',dtype=t.float32),
                         t.as_tensor(rng.permutation(model.width),device='cuda')) for b in c['stages']}
    C.N.progress('marker-native',part*40,total,started)
    for i in range(40):
        with model.block(i) as block:
            for p in panels:
                x=p['x'].cuda()
                angles=[z.cuda() for z in p['angles']]
                mask=p['mask'].cuda()
                q,k,v=model.qkv(block,x,angles)
                o=model.attention(block,q,k,v,mask)
                clean=model.residual(block,x,o)
                lo,hi=p['lo'],p['hi']
                if i==0:
                    states=dict.fromkeys(EARLY,clean)
                else:
                    states={};cache=[];fork={}
                    names,past=p['live']
                    if i==S.GULAYER:
                        base=P.extract(model,block,x,o,p['q'][:,0])
                        assert t.equal(base['output'],clean)
                    for key,a in zip(names,past.cuda(),strict=True):
                        next_=next((val for earlier,val in cache if t.equal(earlier,a)),None) if not (key=='joint' and i==S.GULAYER) else None
                        if next_ is None:
                            changed=x.clone()
                            changed[:,lo:hi]=a
                            q1,k1,v1=model.qkv(block,changed,angles)
                            heads=model.attention(block,q1,k1,v1,mask)
                            next_=model.residual(block,changed,heads)
                            cache.append((a,next_))
                        states[key]=next_
                        if key=='joint' and i==S.GULAYER:
                            fork=K.variants(model,block,p,base,clean,changed=changed,heads=heads,route='joint',c=c)
                    states.update(fork)
                if i in choices:
                    apply(model,p,states,clean,i,*choices[i])
                assert t.equal(states['sham'],clean) and t.equal(states['restore'],clean)
                names=ARMS if i>=S.GULAYER else EARLY
                p['live']=(names,t.stack([states[key][:,lo:hi] for key in names]).cpu())
                p['x']=clean.cpu()
        if (i+1)%5==0:
            C.N.progress('marker-native',part*40+i+1,total,started)
    final=[]
    for p in panels:
        x=p['x'].cuda()
        z=P.full_logits(model,p,x)[:,0]
        m=F.opening_scores(model,z,weights[:2]).cpu().numpy().reshape(-1,2)
        top=z.argmax(-1).cpu().numpy().reshape(-1,2)
        base=z[1::2].log_softmax(-1)
        scores=[];ids=[];kls=[]
        for branch,a in zip(*p['live'],strict=True):
            y=x.clone()
            y[:,p['lo']:p['hi']]=a.cuda()
            revised=P.full_logits(model,p,y)[:,0]
            assert t.equal(revised[::2],z[::2])
            if branch in ('sham','restore'):
                assert t.equal(revised,z)
            out=revised[1::2]
            scores.append(F.opening_scores(model,out,weights[:2]).cpu().numpy())
            ids.append(out.argmax(-1).cpu().numpy())
            kls.append((base.exp()*(base-out.log_softmax(-1))).sum(-1).cpu().numpy())
        meta={k:v[1::2] for k,v in S.meta(p).items()}
        meta.update(natural=m,original=top,edit=np.stack(scores,1),top=np.stack(ids,1),kl=np.stack(kls,1),
            dose=np.stack([p['doses'][g['block']] for g in c['stages']],1),
            geometry=np.stack([p['geometry'][g['block']] for g in c['stages']],1),
            gu_dose=p['gu_doses']['joint'],gua=p['gua']['joint'])
        final.append(meta)
    return {k:np.concatenate([v[k] for v in final]) for k in final[0]}


def native(model,rows,weights,c,stamp):
    sample=[r for r in rows[:M.CAUSAL*A.VIEWS] if r['marker'] in c['markers']]
    assert len(sample)==M.CAUSAL*12
    groups=[sample[:48],*batched(sample[48:],SLAB)]
    files=[];started=time.monotonic()
    for part,group in enumerate(groups):
        name=f'native-{part:02}.npz';receipt=f'native-{part:02}.json'
        if (OUT/receipt).exists():
            previous=R.read(receipt)
            assert previous['holdout']==stamp and previous['sha']==sha(name)
        else:
            check_write(OUT/name,12*2**20,operation='宽位置路径逐批预检')
            result=slab(model,group,weights,c,part,40*len(groups),started)
            R.write(receipt,dict(sha=S.artifact(name,result),design=sha('design.json'),
                                 holdout=stamp,views=len(group),at=C.stamp()))
        files.append(name)
    S.bind_label('native',files,stamp)


def run():
    import torch
    d=R.read('design.json');c=R.read('candidate.json');plan=R.read(PLAN)
    stamp=sha(PLAN)
    assert plan['code']==code() and plan['candidate']==sha('candidate.json') and plan['design']==sha('design.json')
    assert plan['reused_native']==A.sha(Path(K.__file__).resolve()) and d['code']==A.source_id()
    assert d['software']==R.versions() and d['previous']==A.previous()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic();C.N.progress('marker-holdout-load',0,2,start)
        assert C.Prior.identity()==d['model']
        model=C.N.Engine();C.N.progress('marker-holdout-load',2,2,start)
        try:
            inp=M.build(model.tok,'confirm')
            assert M.O.digest(inp)==plan['material']['identity']
            weights=X.anchors(model,d)
            A.confirm(model,inp[0],weights,c,stamp)
            native(model,inp[0],weights,c,stamp)
        finally:
            model.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','run'))
    args=cli.parse_args()
    A.setup()
    (freeze if args.action=='freeze' else run)()
