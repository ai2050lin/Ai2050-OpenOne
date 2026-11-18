"""Original BF16 early-slot readout and isolated L25H18 cache restoration."""

from itertools import batched
import time

import numpy as np

import kv_early_run as C
import kv_temporal_native as N
import kv_origin_trace as O


F=N.F


def arm(early=(),relay=None,writes=(),suppress=False):
    return dict(early=list(early),relay=relay,writes=list(writes),memory=None,code=None,suppress=suppress)


def read(head,type_,donor=0):
    return dict(head=head,type=type_,donor=donor)


def reshape_rows(rows,scope):
    if scope=="early":
        return N.scope_plan(rows,scope)
    if scope=="probe":
        return [dict(r,token=dict(r["token"],ids=r["token"]["ids"][:r["token"]["slots"][0]+1])) for r in rows]
    return rows


def kv25(m,l,qq,kk,vv,vis,head,j,meta,slot,native,base):
    t=m.t
    r=t.arange(len(slot),device=kk.device)
    group=head//m.groups
    origk=kk[r,group,slot].clone()
    origv=vv[r,group,slot].clone()
    k1,v1=kk.clone(),vv.clone()
    kind=meta["type"]
    k_back,v_back=native
    if kind in ("k","kv","reverse"):
        k1[r,group,slot]=k_back[meta["donor"],r,group,slot]
    if kind in ("v","kv","reverse","joint"):
        v1[r,group,slot]=v_back[meta["donor"],r,group,slot]
    if kind in ("zero","self","joint"):
        v1[r,group,slot]=0
        if kind in ("self","joint"):
            v1[r,group,slot]=origv if kind=="self" else v_back[meta["donor"],r,group,slot]
        if kind=="joint":
            k1[r,group,slot]=k_back[meta["donor"],r,group,slot]
    if kind=="zero":
        assert not bool(v1[r,group,slot].any()) and t.equal(k1[r,group,slot],origk)
    if kind=="self":
        assert t.equal(v1[r,group,slot],origv)
    if kind in ("v","kv","reverse","joint"):
        assert t.equal(v1[r,group,slot],v_back[meta["donor"],r,group,slot])
    if kind in ("k","kv","reverse","joint"):
        assert t.equal(k1[r,group,slot],k_back[meta["donor"],r,group,slot])
    before=m.foreground(l,head,qq,kk,vv,vis)
    after=m.foreground(l,head,qq,k1,v1,vis)
    edited=(base.float()+(after.float()-before.float())).to(base.dtype)
    logit=(qq[:,head].float()*k1[:,group].float()).sum(-1)*(128**-.5)
    weight=logit.masked_fill(~vis[:,0,0],-t.inf).softmax(-1)[r,slot]
    return edited,weight,k1[r,group,slot],v1[r,group,slot]


def summarize_kv(m,l,k,v,slot):
    t=m.t
    r=t.arange(len(slot),device=k.device)
    group=18//m.groups
    key=k[:,r,group,slot].float()
    value=v[:,r,group,slot].float()
    proj=l.self_attn.o_proj.weight.reshape(m.width,m.heads,128)[:,18].float()
    written=value@proj.T
    dr=[(x-x[:1]).norm(dim=-1)/x[:1].norm(dim=-1).clamp_min(1e-4) for x in (key,written)]
    return dict(kv_ratios=F(t.stack(dr,-1)).swapaxes(0,1),k25_before=F(key).swapaxes(0,1),v25_before=F(value).swapaxes(0,1))


def middle(m,l,p,mode,cfg):
    t=m.t
    if "pivot" not in p:
        N.init(m,p,mode,cfg)
    x=p["x"].cuda(); cs=tuple(z.cuda() for z in p["cs"]); mask=p["mask"].cuda()
    q0,k0,v0=m.qkv(l,x,cs)
    o0=m.attention(l,q0,k0,v0,mask)
    p["x"]=m.residual(l,x,o0).cpu()
    raws=list(cfg["arms"].values())
    a,b=len(raws),len(x)
    pivot=p["pivot"]
    width=x.shape[1]-pivot
    vis=mask[:,:,pivot:]
    live=p["branches"].cuda() if "branches" in p else x[:,pivot:].unsqueeze(0).expand(a,-1,-1,-1)
    angles=tuple(z[:,pivot:].repeat(a,1,1) for z in cs)
    q,k,v=N.branch_qkv(m,l,live,k0,v0,angles,pivot)
    slot=p["g"].cuda()
    p["actual_kv"]=summarize_kv(m,l,k,v,slot)
    if mode=="probe":
        return
    z=m.attention(l,q.flatten(0,1),k.flatten(0,1),v.flatten(0,1),vis.repeat(a,1,1,1)).reshape(a,b,width,m.heads,128)
    r=t.arange(b,device=x.device)
    qcode=q[:,:,:,-1:]
    viscode=vis[:,:,-1:]
    weights=[]; keys=[]; vals=[]; h_outputs=[]
    for j,op in enumerate(raws):
        patch=op["relay"]
        head=18 if patch is None else patch["head"]
        group=head//m.groups
        if patch is None:
            logit=(qcode[j,:,head].float()*k[j,:,group].float()).sum(-1)*(128**-.5)
            weight=logit.masked_fill(~viscode[:,0,0],-t.inf).softmax(-1)[r,slot]
            key,val=k[j,r,group,slot],v[j,r,group,slot]
        else:
            z[j,:,-1,head],weight,key,val=kv25(m,l,qcode[j],k[j],v[j],viscode,head,j,patch,slot,(k,v),z[j,:,-1,head])
        weights.append(weight); keys.append(key); vals.append(val); h_outputs.append(z[j,:,-1,head])
    p["actual_kv"].update(g_alpha25=F(t.stack(weights,1)),applied_k25=F(t.stack(keys,1)),applied_v25=F(t.stack(vals,1)),
                          h18_or_control=F(t.stack(h_outputs,1)))
    att=l.self_attn.o_proj(z.reshape(a,b,width,m.width).contiguous())
    block=live+att
    mlp=l.mlp(l.post_attention_layernorm(block))
    y=block+mlp
    fields=[(att[:,:,-1].float()-att[:1,:,-1].float()).norm(dim=-1),(mlp[:,:,-1].float()-mlp[:1,:,-1].float()).norm(dim=-1)]
    if cfg["scope"]=="temporal":
        for pos in (p["e"],p["g"]):
            sample=y[:,r,(pos-pivot).cuda()].float()
            fields.append((sample-sample[:1]).norm(dim=-1))
        for x1 in (k,v):
            here=x1.permute(1,3,0,2,4)[r,slot].transpose(0,1).float()
            fields.append((here-here[:1]).flatten(2).norm(dim=-1))
        p["slot_native"][25]=p["actual_kv"]["v25_before"]
    p["trace"][25]=F(t.stack(fields,-1)).swapaxes(0,1)
    p["loc_error"][25]=0.
    p["branches"]=y.cpu()


def step(m,l,i,p,mode,cfg):
    if i==25 and mode!="screen-e":
        return middle(m,l,p,mode,cfg)
    return N.step(m,l,i,p,mode,cfg)


def freeze_rollout(m,p,cfg):
    """Greedy verification from E. Blackout changes one full-vocabulary argmax only."""
    t=m.t
    names=list(cfg["arms"])
    pos=t.arange(len(p["rows"]),device="cuda")
    e=(p["e"]-p["pivot"]).cuda()
    logits=m.logits(p["branches"][names.index("switch")].cuda())
    rules=cfg["arms"]["switch"]
    if rules["suppress"]:
        state=t.tensor([r["state_ids"][0] for r in p["rows"]],device="cuda")
        logits[pos,e,state]=-t.inf
    ids=logits.argmax(-1).cpu().numpy()
    tail=p["ids"].numpy()
    answer=[]
    for j,row in enumerate(p["rows"]):
        ee=int(e[j]); raw=ids[j,ee:]
        reference=tail[j,p["pivot"]+ee+1:]
        wrong=np.flatnonzero(raw[:-1]!=reference)
        b=int(wrong[0]) if len(wrong) else -1
        answer.append(dict(id=row["id"],state=int(raw[0]),code=int(raw[-1]) if b<0 else None,raw_code=int(raw[-1]),
                           first_mismatch=b,actual_token=int(raw[b]) if b>=0 else None,verified=b<0))
    return answer


def finish(m,p,mode,cfg,known):
    if mode=="probe":
        return dict(ids=np.array([r["id"] for r in p["rows"]]),**p["actual_kv"])
    if mode=="rollout":
        return freeze_rollout(m,p,cfg)
    out=N.finish(m,p,mode,cfg,known)
    if mode in ("code","temporal"):
        out.update(p["actual_kv"])
    return out


def panel(m,rows,mode,cfg,known):
    down=reshape_rows(rows,cfg["scope"])
    t0=time.monotonic()
    span=24 if mode=="screen-e" else 26 if mode=="probe" else 40
    size=8*(C.SLAB if cfg["scope"]=="temporal" else C.N.SLAB)
    records=[]
    for left in range(0,len(rows),size):
        group=[m.packet(chunk) for chunk in batched(down[left:left+size],8)]
        total=(len(rows)+size-1)//size
        C.N.progress(mode,0,span,t0,part=left//size+1,parts=total)
        for i in range(span):
            with m.block(i) as l:
                for p in group:
                    step(m,l,i,p,mode,cfg)
            C.N.progress(mode,i+1,span,t0,part=left//size+1,parts=total)
        records.extend(finish(m,p,mode,cfg,known) for p in group)
    if mode=="rollout":
        return [r for x in records for r in x]
    return {k:np.concatenate([r[k] for r in records]) for k in records[0]}
