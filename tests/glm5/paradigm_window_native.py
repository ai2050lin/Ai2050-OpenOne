"""Post-block native suffix narrowing, stage factorial, and dose controls."""

from itertools import batched
import time

import numpy as np

import paradigm_window_run as W
from deploy.storage_policy import check_write

R,C,S,F,X,P,M,OUT=W.R,W.C,W.S,W.F,W.X,W.P,W.M,W.OUT
B = (0,8,20,31)
# Ordinary arms use the exact BF16 donor. Companion directions are adjusted
# token by token to match that arm's realized displacement at each boundary.
COMPARISONS = ('full','last20','suffix')
BASE = dict(full=B,first40=B,last40=B,last20=B,suffix=(31,),drop8=(0,20,31),drop20=(0,8,31),
            dropboth=(0,31),late20_restore=B,restore=B,sham=())
ARMS = (*BASE,*(f'{key}_{kind}' for key in COMPARISONS for kind in ('perm','natural')))
CTRL = {f'{key}_{kind}':(key,kind) for key in COMPARISONS for kind in ('perm','natural')}
SLAB = 96
TOL = .03
COS = .90
GEN_ARMS = ('sham','full','last20','restore')


def material(views,families,layouts):
    chosen=list(dict.fromkeys(x['family'] for x in views))[:families]
    by={(x['family'],x['marker'],x['layout'],x['archive']):x for x in views}
    rows=[]
    for left,right in batched(chosen,2,strict=True):
        for marker in range(2):
            for layout in layouts:
                for family in (left,right):
                    ref,code,natural=(by[family,marker,layout,i] for i in range(3))
                    assert ref['query']==code['query']==natural['query']
                    rows.extend((ref,code,natural,code))
    return rows


def fit_direction(t, state, ray, goal, mask):
    """Find a positive scalar per token by measuring the actual BF16 edit."""
    base=state.float()
    norm=ray.norm(dim=-1)
    direction=ray/norm.clamp_min(1e-12)[...,None]
    scale=goal.clone()
    best=state.clone()
    error=t.full_like(goal,t.inf)
    best_norm=t.zeros_like(goal)
    best_cos=t.zeros_like(goal)
    for _ in range(5):
        new=(base+direction*scale[...,None]).to(state.dtype)
        delta=new.float()-base
        dose=delta.norm(dim=-1)
        cosine=(delta*direction).sum(-1)/dose.clamp_min(1e-12)
        relative=(dose-goal).abs()/t.maximum(goal,dose).clamp_min(1e-12)
        loss=relative+(cosine<COS).float()
        take=loss<error
        best=t.where(take[...,None],new,best)
        best_norm=t.where(take,dose,best_norm)
        best_cos=t.where(take,cosine,best_cos)
        error=t.minimum(error,loss)
        scale=scale*(goal/dose.clamp_min(1e-12)).clamp(.5,2.)
    noop=goal==0
    active=mask&~noop
    best=t.where(active[...,None],best,state)
    return best,t.stack((goal,best_norm,best_cos,norm),-1)*active[...,None]


def suffix_key(name):
    return name if name in W.WIDTHS else 'all'


def start_panel(p,model,keep_cache):
    t=model.t
    p['torch']=t
    assert [x['archive'] for x in p['rows']]==[0,1,2,1]*2
    p['recipient']=t.tensor([1,5],device='cuda')
    p['lo']=min(r['source'][-1] for r in p['rows'])+1
    p['hi']=int(p['q'][:,0].max())+1
    p['places']={i:W.positions(p,i) for i in B}
    p['live']={};p['doses']={};p['updates']={}
    if keep_cache:
        p['cache']={}
        p['cache_end']=p['hi']
        p['hi']+=1


def transplant(t,p,states,clean,layer,arms,permutation):
    j=p['recipient'][:,None]
    after={}
    choice=p['places'][layer]
    edits={};computed={}
    for name in arms:
        target,kind=CTRL.get(name,(name,'raw'))
        current=states[name]
        if layer not in BASE[target]:
            after[name]=current
            continue
        suffix=suffix_key(target) if layer==31 else next(iter(choice))
        where=choice[suffix][p['recipient']]
        valid=where>=0
        ix=where.clamp_min(0)
        own=current[j,ix]
        donor=clean[j-1,ix]
        cache_key=(id(current),suffix,kind,id(states[target])) if kind!='raw' else (id(current),suffix,kind)
        unique=name=='restore' or (name=='late20_restore' and layer==31)
        if not unique and cache_key in computed:
            out,edit,dose=computed[cache_key]
            if kind=='raw':
                edits[name]=edit
            else:
                p['doses'][layer,name]=dose
            after[name]=out
            continue
        if kind=='raw':
            new=t.where(valid[...,None],donor,own)
            edits[name]=(new.float()-own.float()).norm(dim=-1)
        else:
            goal=edits[target]
            vector=donor.float()-own.float()
            if kind=='natural':
                ray=clean[j+1,ix].float()-clean[j,ix].float()
            else:
                perm,sign=permutation
                ray=vector[...,perm]*sign
                ray-=vector*((ray*vector).sum(-1)/vector.square().sum(-1).clamp_min(1e-12))[...,None]
            new,receipt=fit_direction(t,own,ray,goal,valid)
            realized=new.float()-own.float()
            overlap=(realized*vector).sum(-1)/(realized.norm(dim=-1)*vector.norm(dim=-1)).clamp_min(1e-12)
            receipt=t.cat((receipt,overlap[...,None]),-1)
            p['doses'][layer,name]=receipt.cpu().numpy()
        # Padded positions are -1 before indexing. Advanced assignment must
        # skip them rather than duplicating writes at token zero.
        out=current.clone()
        jj=j.expand_as(ix)[valid];qq=ix[valid]
        out[jj,qq]=new[valid]
        if name=='restore':
            out[jj,qq]=clean[jj,qq]
            assert t.equal(out,clean)
            out=clean
        if name=='late20_restore' and layer==31:
            yy=choice['last20'][p['recipient']]
            assert (yy>=0).all()
            out[j,yy]=clean[j,yy]
        after[name]=out
        if not unique:
            computed[cache_key]=(out,edits.get(name),p['doses'].get((layer,name)))
    return after


def rollout(model,rows,weights,arms,progress,cache=False):
    t=model.t
    panels=X.panels(model,rows,0)
    for p in panels:
        start_panel(p,model,cache)
    rng=np.random.default_rng(M.SEED+19)
    perm={i:(t.as_tensor(rng.permutation(model.width),device='cuda'),
             t.as_tensor(rng.choice((-1.,1.),size=model.width),device='cuda',dtype=t.float32)) for i in B}
    for layer in range(40):
        with model.block(layer) as block:
            for p in panels:
                x=p['x'].cuda()
                angles=[a.cuda() for a in p['angles']]
                mask=p['mask'].cuda()

                def step(x):
                    q,k,v=model.qkv(block,x,angles)
                    h=model.residual(block,x,model.attention(block,q,k,v,mask))
                    prefix=(k,v) if cache else None
                    return h,prefix

                clean,cleankv=step(x)
                states={};done=[];branchkv={}
                lo,hi=p['lo'],p['hi']
                for name in arms:
                    past=p['live'].get(name)
                    reused=next((match for old,match in done if old is past),None) if past is not None else None
                    if past is None:
                        stem=clean,cleankv
                    elif reused is not None:
                        stem=reused
                    else:
                        modified=x.clone()
                        modified[:,lo:hi]=past.cuda()
                        stem=step(modified)
                        done.append((past,stem))
                    states[name],branchkv[name]=stem
                if cache:
                    # Keep the exact prefix K/V for free greedy continuation;
                    # by causality the probe tokens after E are never visible.
                    j=p['recipient']
                    picks=[(cleankv,j-1),(cleankv,j),*( (branchkv[k],j) for k in ('full','last20') )]
                    p['cache'][layer]=tuple(t.cat([pair[n][rows,:,:p['cache_end']].cpu() for pair,rows in picks]) for n in (0,1))
                if layer in B:
                    states=transplant(t,p,states,clean,layer,arms,perm[layer])
                assert t.equal(states['restore'],clean) and t.equal(states['sham'],clean)
                new_live={};known={}
                for name,h in states.items():
                    if h is not clean:
                        if id(h) not in known:
                            known[id(h)]=h[:,lo:hi].cpu()
                        new_live[name]=known[id(h)]
                p['live']=new_live;p['x']=clean.cpu()
        if layer%5==0 or layer==39:
            progress(layer+1)
    final=[]
    for p in panels:
        x=p['x'].cuda()
        logits=P.full_logits(model,p,x)[:,0]
        j=p['recipient']
        original=logits[j]
        ref=F.opening_scores(model,logits,weights[:2])
        assert t.equal(logits[j],logits[j+2])
        y=[];tops=[];next_at_forced={}
        at=p['q'][j,0]+1
        if cache:
            next_at_forced['prose']=model.logits(x[j-1,at]).argmax(-1).cpu().numpy()
        for name in arms:
            full=x.clone()
            if name in p['live']:
                full[:,p['lo']:p['hi']]=p['live'][name].cuda()
            z=P.full_logits(model,p,full)[:,0]
            assert t.equal(z[j-1],logits[j-1]) and t.equal(z[j+1],logits[j+1])
            if name in ('restore','sham'):
                assert t.equal(z,logits)
            y.append(F.opening_scores(model,z[j],weights[:2]).cpu().numpy());tops.append(z[j].argmax(-1).cpu().numpy())
            if cache:
                next_at_forced['code' if name=='sham' else name]=model.logits(full[j,at]).argmax(-1).cpu().numpy()
        jcpu=j.cpu().numpy()
        meta={key:value[jcpu] for key,value in S.meta(p).items()}
        meta.update(marker=[p['rows'][i]['marker'] for i in jcpu],frame=[p['rows'][i]['frame'] for i in jcpu],
                    original=ref[j].cpu().numpy(),reference=ref[j-1].cpu().numpy(),
                    original_top=original.argmax(-1).cpu().numpy(),reference_top=logits[j-1].argmax(-1).cpu().numpy(),
                    edit=np.stack(y,1),top=np.stack(tops,1))
        if not cache:
            for (layer,name),value in p['doses'].items():
                # Absolute padding prevents a different E or active source from
                # changing shard shape. Zero rows represent masked token slots.
                meta[f'dose-{layer}-{name}']=np.pad(value,((0,0),(0,M.WIDTH-value.shape[1]),(0,0)))
        else:
            names=('prose','code','full','last20')
            p['initial']=np.stack((meta['reference_top'],meta['original_top'],*(meta['top'][:,arms.index(k)] for k in names[2:])))
            p['expected_next']=np.stack([next_at_forced[k] for k in names])
        final.append(meta)
    combined={key:np.concatenate([q[key] for q in final]) for key in final[0]}
    return combined,panels


def run(model,views,weights,binding):
    if not qualify_frames(model,views,weights,binding):
        raise RuntimeError('New-frame absolute layout panel failed the original paired format gate')
    rows=material(views,M.CAUSAL,M.LAYOUTS)
    files=[];started=time.monotonic()
    total=(len(rows)+SLAB-1)//SLAB
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'native-{part:02}.npz';receipt=name.replace('.npz','.json')
        if (OUT/receipt).exists():
            meta=R.read(receipt)
            assert meta['sha']==W.sha(OUT/name) and meta['holdout']==binding
        else:
            check_write(OUT/name,16*2**20)
            result,_=rollout(model,group,weights,ARMS,lambda n:C.N.progress('window-native',part*40+n,total*40,started))
            R.write(receipt,dict(at=C.stamp(),sha=S.artifact(name,result),design=W.sha(OUT/'design.json'),holdout=binding,views=len(group)))
        files.append(name)
    S.bind_label('native',files,binding)


def qualify_frames(model,views,weights,binding):
    """Before edits, run the frozen same-family 1011 contrast on each new frame."""
    name='confirm-frame-gate.json'
    if (OUT/name).exists():
        gate=R.read(name)
        assert gate['holdout']==binding and gate['measurements']==W.sha(OUT/'confirm-style-gate-files.json')
        return gate['passed']
    frozen=M.families('confirm')[:32]
    ids={r['family'] for r in frozen}
    donors=[r for r in views if r['family'] in ids and r['archive']!=2]
    foils=[M.case(model.tok,r,r['marker'],r['layout'],r['archive'],11) for r in donors]
    W.clean_pass(model,foils,weights,'confirm-style-gate',binding)
    d=S.load('confirm',binding);s=S.load('confirm-style-gate',binding)
    baseline={str(k):v for k,v in zip(d['ids'],d['margin'][:,1],strict=True)}
    values=(np.array([baseline[r['id']] for r in donors])-s['margin'][:,1]).reshape(32,2,3,2)
    frames=np.array([r['frame'] for r in frozen]);assert (np.unique(frames,return_counts=True)[1]==16).all()
    rates={str(key):(values[frames==key]>0).mean(0).tolist() for key in np.unique(frames)}
    minimum=min(np.min(x) for x in rates.values())
    passed=bool(minimum>=W.MIN_BEHAVIOR)
    R.write(name,dict(at=C.stamp(),holdout=binding,families=len(frozen),threshold=W.MIN_BEHAVIOR,
                      measurements=W.sha(OUT/'confirm-style-gate-files.json'),minimum=float(minimum),rates=rates,passed=passed))
    print('Held-out frame paired format gate',passed,'minimum',float(minimum),flush=True)
    return passed
