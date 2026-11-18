"""Original BF16 one-query support, K/Q projection, V write, and parallel recovery."""

from itertools import batched
import time

import numpy as np

import kv_length_run as C
from kv_orth_native import dose,site_masks
from kv_prefix_native import F


TYPES=("address","value","fact","book")
COL={"gen":0,"fact":1,"book":2}


def scan(m,l,p,index,q,k,v,o,mask):
    t=m.t
    hidx=next(j for j,r in enumerate(C.ARMS) if r[0]==index)
    ms=site_masks(m,p)[:,hidx]
    a=m.read(q[:,:,-1:],k,mask[:,:,-1:])
    vv=v.repeat_interleave(m.groups,dim=1).float()
    weighted=a*ms[:,None]
    chosen=(weighted[...,None]*vv).sum(-2)
    hs=tuple(range(m.heads))
    proj=m.projections(l,hs,chosen).norm(dim=-1)
    if hidx==0:
        alpha=weighted.sum(-1)
        rest=(o[:,-1].float()*alpha[...,None]-chosen)/(1-alpha).clamp_min(1e-5)[...,None]
        p["scan"]["address"]=F(m.projections(l,hs,rest).norm(dim=-1))
    p["scan"][TYPES[hidx+1]]=F(proj)


def clone_delta(m,l,h,key,mode,q,k,v,mask,site,geometry=False):
    t=m.t;group=h//m.groups;orig_q=q;orig_k=k;orig_v=v
    rows=t.arange(len(k),device="cuda")
    v2=v.clone();k2=k.clone();q2=q.clone();mask2=mask.clone()
    if mode in ("address","address-self"):
        mask2[:,0,0] &= ~site
        if mode=="address-self":
            mask2.copy_(mask)
    if mode in ("value","value-self","kv","restore-k","restore-v","restore-kv"):
        v2[:,group].masked_fill_(site[...,None],0)
        if mode in ("value-self","restore-v","restore-kv"):
            v2[:,group]=t.where(site[...,None],orig_v[:,group],v2[:,group]);assert t.equal(v,v2)
    g=None
    if key=="gen":
        assert bool((site.sum(-1)==1).all())
        g=site.to(t.int64).argmax(-1)
        q0,k0=q[rows,h,0].float(),k[rows,group,g].float()
        c=(q0*k0).sum(-1,keepdim=True)
        if mode in ("key","kv","restore-k","restore-v","restore-kv"):
            k2[rows,group,g]=(k0-q0*(c/q0.square().sum(-1,keepdim=True).clamp_min(1e-8))).to(k.dtype)
            if mode in ("restore-k","restore-kv"):
                k2[rows,group,g]=k[rows,group,g];assert t.equal(k,k2)
        if mode in ("query","query-self"):
            q2[rows,h,0]=(q0-k0*(c/k0.square().sum(-1,keepdim=True).clamp_min(1e-8))).to(q.dtype)
            if mode=="query-self":
                q2.copy_(q)
    base=m.foreground(l,h,q,k,v,mask).float()
    modified=m.foreground(l,h,q2,k2,v2,mask2).float()
    info=None
    if geometry and g is not None:
        before=m.read(q,k,mask)[rows,h,g]
        after=m.read(q2,k2,mask2)[rows,h,g]
        score0=(q0*k0).sum(-1)*128**-.5
        qa,ka=q2[rows,h,0].float(),k2[rows,group,g].float()
        score1=(qa*ka).sum(-1)*128**-.5
        values=[before,after,score0,score1,(qa-q0).norm(dim=-1),(ka-k0).norm(dim=-1),
                (v2[rows,group,g].float()-orig_v[rows,group,g].float()).norm(dim=-1)]
        if mode in ("value","value-self"):
            assert t.equal(before,after)
        if mode=="address":
            assert not bool(after.any()) and t.equal(q2,orig_q) and t.equal(k2,orig_k) and t.equal(v2,orig_v)
        if mode in ("key","query"):
            assert t.equal(v2,orig_v)
        info=F(t.stack(values,-1))
    if mode in ("address-self","value-self","query-self","restore-kv"):
        assert t.equal(base,modified)
    return modified-base,info


def process(m,l,i,p,cfg,mode):
    t=m.t
    if i==0:
        if mode=="scan":p["scan"]={}
        else:
            p["first"]=min(o[0] for arm in cfg.values() for o in arm)
            p["dose"]=np.zeros((len(p["rows"]),len(cfg),3,3),dtype=np.float32)
            p["geom"]=np.zeros((len(p["rows"]),len(cfg),7),dtype=np.float32)
            p["trace"]={}
    x=p["x"].cuda();mask=p["mask"].cuda();cs=tuple(y.cuda() for y in p["cs"])
    q0,k0,v0=m.qkv(l,x,cs);o0=m.attention(l,q0,k0,v0,mask)
    p["x"]=m.residual(l,x,o0).cpu()
    if mode=="scan":
        if i in {z[0] for z in C.ARMS}:scan(m,l,p,i,q0,k0,v0,o0,mask)
        return
    if i<p["first"]:return
    a,b=len(cfg),len(x);vis=mask[:,:,-1:]
    live=p["branch"].cuda() if "branch" in p else x[None,:,-1].expand(a,-1,-1)
    angle=tuple(y[:,-1:].repeat(a,1,1) for y in cs)
    qq,kk,vv=m.branch_qkv(l,live.reshape(a*b,m.width),k0.repeat(a,1,1,1),v0.repeat(a,1,1,1),angle)
    z=m.attention(l,qq,kk,vv,vis.repeat(a,1,1,1)).reshape(a,b,m.heads,128)
    q,k,v=(y.reshape(a,b,*y.shape[1:]) for y in (qq,kk,vv))
    do=[(j,*o[1:]) for j,arm in enumerate(cfg.values()) for o in arm if o[0]==i]
    if do:masks=site_masks(m,p)
    for j,head,key,how,peer in do:
        idx=COL[key];site=masks[:,idx];old=z[j,:,head].clone()
        if how=="head-restore":
            z[j,:,head]=z[0,:,head]
            assert t.equal(z[j,:,head],z[0,:,head])
        else:
            d,info=clone_delta(m,l,head,key,how,q[j],k[j],v[j],vis,site,geometry=key=="gen" and peer is None)
            raw=dose(m,l,head,d);goal=raw
            if peer is not None:
                assert t.equal(old,z[0,:,head])
                match,_=clone_delta(m,l,peer,key,how,q[0],k[0],v[0],vis,site)
                goal=t.minimum(raw,dose(m,l,peer,match))
                d=d*t.where(raw>0,goal/raw,t.zeros_like(raw))[:,None]
            z[j,:,head]=(old.float()+d).to(old.dtype)
            p["dose"][:,j,idx,:2]=F(t.stack((raw,goal),-1))
            if info is not None:p["geom"][:,j]=info
            actual=z[j,:,head].float()-old.float()
            err=(actual-d).norm(dim=-1)/(1+d.norm(dim=-1))
            assert float(err.max())<.05,(head,how,float(err.max()))
            if key=="gen" and head==18 and how in ("address","value") and peer is None and how not in p["trace"]:
                r=t.arange(b,device="cuda");g=p["slots"][:,0].cuda();group=head//m.groups
                vg=v[j,r,group,g].float()
                alpha=t.as_tensor(info[:,0],device="cuda")
                pool=m.read(q[j],k[j],vis)[:,head].masked_fill(site,0)
                other=(pool[...,None]*v[j,:,group].float()).sum(-2)/(1-alpha).clamp_min(1e-5)[:,None]
                p["trace"][how]=F(t.stack((vg,other,actual),1))
        p["dose"][:,j,idx,2]=F(dose(m,l,head,z[j,:,head].float()-old.float()))
    p["branch"]=m.residual(l,live,z).cpu()


def run(m,rows,cfg,mode,known=None):
    output=[]
    nlayer=29 if mode=="scan" else 40
    total=(len(rows)+C.SLAB-1)//C.SLAB
    for i,part in enumerate(batched(rows,C.SLAB),1):
        clock=time.monotonic();packets=[m.packet(r) for r in batched(part,8)]
        C.N.progress(mode,0,nlayer,clock,part=i,parts=total)
        for j in range(nlayer):
            with m.block(j) as l:
                for p in packets:process(m,l,j,p,cfg,mode)
            C.N.progress(mode,j+1,nlayer,clock,part=i,parts=total)
        for p in packets:
            ids=np.array([r["id"] for r in p["rows"]]);z=dict(ids=ids)
            if mode=="scan":z["scan"]=np.stack([p["scan"][k] for k in TYPES],1)
            else:
                scores=[m.logits(w.cuda()) for w in p["branch"]]
                pq=np.stack([F(x[:,m.letters]) for x in scores],1)
                top=np.stack([x.argmax(-1).cpu().numpy() for x in scores],1)
                known_pq=np.array([known[k]["pq"] for k in ids]);error=abs(pq[:,0]-known_pq).max(-1)
                technical=(error<=C.N.SHAM_TOL)&(top[:,0]==[known[k]["top"] for k in ids])
                z.update(pq=pq,top=top,technical=technical,drift=error,dose=p["dose"],geom=p["geom"],**p["trace"])
            output.append(z)
    return {k:np.concatenate([x[k] for x in output]) for k in output[0]}
