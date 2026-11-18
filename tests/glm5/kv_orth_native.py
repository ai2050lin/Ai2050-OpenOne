"""Genuine L39 lexical choice, greedy-prefix verification, and one-query BF16 interventions."""

from itertools import batched
import re
import time

import numpy as np

import kv_orth_run as C
from kv_prefix_native import F


def polarity(rows, role):
    return np.array([1 if r[f"code_{role}"]=="Q" else -1 for r in rows],dtype=np.float32)


def output(m,p):
    t=m.t
    start=min(p["starts"])-1
    x=p["x"].cuda()
    scores=m.logits(x[:,start:])
    best=scores.argmax(-1).cpu().numpy()
    native_pq=F(scores[:,-1,m.letters])
    e=p["slots"][:,0].numpy()-1
    return x,e,start,best,native_pq


def observe(m,rows,label):
    by_id={}; cached={}; whitespace={}
    work=rows
    for attempt in range(C.PASSES):
        following=[]
        for _, packets in m.batches(work,label):
            for p in packets:
                slab=p["rows"]
                x,e,start,pred,pq=output(m,p)
                if not attempt:
                    ix={(r["family"],r["distance"],r["map"],r["state"],r["fact"]):j for j,r in enumerate(slab)}
                    donors=[ix[r["family"],r["distance"],r["map"],r["state"],r["state"]] for r in slab]
                    t=m.t
                    native=x[t.arange(len(slab),device="cuda"),t.as_tensor(e,device="cuda")]
                    picked=m.logits(native[donors]).argmax(-1).cpu().tolist()
                    for j,r in enumerate(slab):
                        assert slab[donors[j]]["fact"]==r["state"]
                        cached[r["id"]]=(int(picked[j]),slab[donors[j]]["id"],int(pred[j,e[j]-start]))
                for j,r in enumerate(slab):
                    rid=r["id"]
                    g=r["token"]["slots"][0]
                    first=r["token"]["start"]
                    ids=r["token"]["ids"]
                    donor,donor_id,untreated=cached[rid]
                    pred[j,e[j]-start]=donor
                    native=pred[j,p["starts"][j]-1-start:]
                    conflict=np.flatnonzero(native[:-1]!=ids[first:])
                    stop=int(conflict[0]) if len(conflict) else len(ids)-first
                    nxt=int(native[stop])
                    matched=int(stop)==len(ids)-first
                    if not matched and attempt<C.PASSES-1 and first+stop>g:
                        shown=m.tok.decode([nxt])
                        changed=m.whitespace_proposal(r,first+stop,nxt,shown) if shown.endswith("\n") else None
                        if changed is not None:
                            line,new=changed
                            whitespace.setdefault(rid,[]).append(dict(line=line,token=nxt,offset=first+stop))
                            following.append(dict(r,token=new))
                            continue
                    seen=ids[first:first+stop]+[nxt]
                    words_actual=bool(first+stop>=g and donor==r["word_ids"][r["state"]])
                    by_id[rid]=dict(id=rid,matched=matched,pq=pq[j].tolist() if matched else None,top=nxt if matched else None,
                                    state=native[g-first].item() if stop>=g-first else None,expected_state=r["word_ids"][r["state"]],
                                    requested_state_emitted=words_actual,raw_unpatched_state=untreated,donor=donor_id,
                                    mismatch=None if matched else stop,pre_state_stop=first+stop<g,
                                    observed=nxt,emitted=seen,rendered=m.tok.decode(seen),
                                    whitespace=whitespace.get(rid,[]),resolved=r["token"] if rid in whitespace else None)
        work=following
        if not work:
            break
    return [by_id[r["id"]] for r in rows]


def replay_rejects(m,rows,obs):
    """Continue format-divergent but verified first emissions without forcing drafts."""
    actual={x["id"]:x for x in obs}
    pending=[]
    result={}
    for r in rows:
        x=actual[r["id"]]
        if x["matched"] or not x["requested_state_emitted"]:
            continue
        tok=x["resolved"] or r["token"]
        k=tok["start"]
        seen=x["emitted"][:80]
        draft=seen+tok["ids"][k+len(seen):]
        pending.append(dict(row=r,tok=tok,seen=seen,base=tok["ids"][:k],draft=draft[:80],passes=[]))
        if len(pending)>=C.MAX_REPLAY:
            break
    attempted=len(pending)
    eos=m.model.generation_config.eos_token_id
    ending={eos} if isinstance(eos,int) else set(eos or [])

    def save(x,stop,code=None):
        r=x["row"]
        result[r["id"]]=dict(id=r["id"],stop=stop,top=code,emitted=x["seen"],rendered=m.tok.decode(x["seen"]),passes=x["passes"])

    for attempt in range(12):
        if not pending:
            break
        trials=[dict(x["row"],token=dict(x["tok"],ids=x["base"]+x["draft"])) for x in pending]
        waiting=[]
        lookup={x["row"]["id"]:x for x in pending}
        for _,packets in m.batches(trials,"orth-format-replay"):
            for p in packets:
                _,e,start,pred,_=output(m,p)
                for j,r in enumerate(p["rows"]):
                    x=lookup[r["id"]]
                    raw=pred[j,p["starts"][j]-1-start:]
                    orig=x["seen"]
                    q=x["tok"]["slots"][0]-x["tok"]["start"]
                    # The verified output intervention at E is fixed, irrespective of replay padding.
                    raw[q]=r["word_ids"][r["state"]]
                    if raw[:len(orig)].tolist()!=orig:
                        x["passes"].append(dict(pass_id=attempt,cause="prefix-drift"))
                        save(x,"prefix-drift");continue
                    for i in range(len(orig),len(raw)):
                        v=int(raw[i])
                        ready=re.search(r"(?:^|\n)Code:[ \t]*$",m.tok.decode(orig)) is not None
                        orig.append(v)
                        if ready or v in ending or len(orig)>=80:
                            x["passes"].append(dict(pass_id=attempt,actual=v,complete=len(orig)))
                            save(x,"code" if ready else "eos" if v in ending else "token-budget",v if ready else None);break
                        if i>=len(x["draft"]) or v!=x["draft"][i]:
                            x["draft"]=orig.copy()+x["draft"][i+1:]
                            x["passes"].append(dict(pass_id=attempt,actual=v,position=i));waiting.append(x);break
        pending=waiting
    for x in pending: save(x,"pass-budget")
    return dict(tested=attempted,observations=result)


def site_masks(m,p):
    a=np.zeros((len(p["rows"]),3,p["ids"].shape[1]),dtype=bool)
    for j,(r,pad) in enumerate(zip(p["rows"],p["padding"],strict=True)):
        a[j,0,r["token"]["slots"][0]+pad]=True
        for k,key in enumerate(("fact","book"),1):
            selected=r["sites"][key]
            assert max(selected)<r["token"]["start"]
            a[j,k,np.array(selected)+pad]=True
    return m.t.as_tensor(a,device="cuda")


def screen(m,l,p,x,q,k,v,vis):
    t=m.t
    axes=m.read(q[:,:,-1:],k,vis[:,:,-1:])
    signs=t.as_tensor(np.stack([polarity(p["rows"],r) for r in ("s","f","s")],1),device="cuda")
    inp=axes[:,None]*site_masks(m,p)[:,:,None]
    native=t.einsum("bohl,bhld->bohd",inp,v.repeat_interleave(m.groups,dim=1).float())
    term=m.projections(l,tuple(range(m.heads)),native)
    w=m.model.lm_head.weight.detach()[m.letters].float()
    d=(w[1]-w[0])*m.decoder.norm.weight.detach().float()
    scale=(x[:,-1].float().square().mean(-1)+1e-6).sqrt()
    effect=t.einsum("bohd,d->boh",term,d)*signs[:,:,None]/scale[:,None,None]
    p.setdefault("screen",[]).append(F(t.stack((effect,term.norm(dim=-1),inp.sum(-1)),-1)))


def raw_delta(m,l,head,q,k,v,vis,mask,own=False):
    target=head//m.groups
    vv=v.clone()
    vv[:,target].masked_fill_(mask[...,None],0)
    if own:
        vv[:,target]=m.t.where(mask[...,None],v[:,target],vv[:,target])
        assert m.t.equal(v,vv)
    return m.foreground(l,head,q,k,vv,vis).float()-m.foreground(l,head,q,k,v,vis).float()


def dose(m,l,h,value):
    return m.projections(l,(h,),value[:,None]).norm(dim=-1)[:,0]


def step(m,l,i,p,arms,mode):
    t=m.t
    if i==0 and mode!="screen":
        p["first"]=min(op[0] for arm in arms.values() for op in arm)
        p["physical"]=np.zeros((len(p["rows"]),len(arms),3,3),np.float32)
        p["source_alpha"]=np.zeros((len(p["rows"]),len(arms),3),np.float32)
        p["anchor_data"]={}
    x=p["x"].cuda(); mask=p["mask"].cuda(); angles=tuple(y.cuda() for y in p["cs"])
    q0,k0,v0=m.qkv(l,x,angles); o0=m.attention(l,q0,k0,v0,mask)
    if mode=="screen" and i in C.LAYERS:
        screen(m,l,p,x,q0,k0,v0,mask)
    p["x"]=m.residual(l,x,o0).cpu()
    if mode=="screen" or i<p["first"]:
        return
    a,b=len(arms),len(x)
    vmask=mask[:,:,-1:]
    live=p["branch"].cuda() if "branch" in p else x[None,:,-1].expand(a,-1,-1)
    angle=tuple(y[:,-1:].repeat(a,1,1) for y in angles)
    q,k,v=m.branch_qkv(l,live.reshape(a*b,m.width),k0.repeat(a,1,1,1),v0.repeat(a,1,1,1),angle)
    z=m.attention(l,q,k,v,vmask.repeat(a,1,1,1)).reshape(a,b,m.heads,128)
    q=q.reshape(a,b,*q.shape[1:]);k=k.reshape(a,b,*k.shape[1:]);v=v.reshape(a,b,*v.shape[1:])
    todo=[(j,*op[1:]) for j,arm in enumerate(arms.values()) for op in arm if op[0]==i]
    if todo:
        masks=site_masks(m,p)
    for j,head,src,op,peer in todo:
        col=C.ROLES.index(src); spot=masks[:,col]
        old=z[j,:,head].clone(); d=None
        alpha=m.read(q[j],k[j],vmask)[:,head]
        p["source_alpha"][:,j,col]=F((alpha*spot).sum(-1))
        if op=="restore":
            z[j,:,head]=z[0,:,head]
            assert t.equal(z[j,:,head],z[0,:,head])
        else:
            d=raw_delta(m,l,head,q[j],k[j],v[j],vmask,spot,op=="self")
            raw=dose(m,l,head,d)
            aimed=raw
            if op=="balance":
                assert t.equal(z[0,:,head],old)
                rival=raw_delta(m,l,peer,q[0],k[0],v[0],vmask,spot)
                aimed=t.minimum(raw,dose(m,l,peer,rival))
                scale=t.where(raw>0,aimed/raw,t.zeros_like(raw))
                d=d*scale[:,None]
            z[j,:,head]=(old.float()+d).to(old.dtype)
            p["physical"][:,j,col,:2]=F(t.stack((raw,aimed),-1))
            if op=="self":
                assert t.equal(old,z[j,:,head])
        realized=z[j,:,head].float()-old.float()
        p["physical"][:,j,col,2]=F(dose(m,l,head,realized))
        if d is not None and op in ("block","balance"):
            error=(realized-d).norm(dim=-1)/(1+d.norm(dim=-1))
            assert float(error.max())<.05,(i,head,src,float(error.max()))
        if [i,head,src]==C.ANCHOR and op=="block":
            r=t.arange(b,device="cuda");g=p["slots"][:,0].cuda();group=head//m.groups
            if "h18_source_k" not in p["anchor_data"]:
                p["anchor_data"].update(h18_source_k=F(k[j,r,group,g]),h18_source_v=F(v[j,r,group,g]),
                                        h18_write_change=F(realized),h18_live_q=F(q[j,:,head,0]))
    p["branch"]=m.residual(l,live,z).cpu()


def panel(m,rows,arms,mode,known=None):
    result=[]
    span=39 if mode=="screen" else 40
    n_parts=(len(rows)+C.SLAB-1)//C.SLAB
    for j,part in enumerate(batched(rows,C.SLAB),1):
        clock=time.monotonic(); packets=[m.packet(x) for x in batched(part,8)]
        C.N.progress(mode,0,span,clock,part=j,parts=n_parts)
        for i in range(span):
            with m.block(i) as l:
                for p in packets:
                    step(m,l,i,p,arms,mode)
            C.N.progress(mode,i+1,span,clock,part=j,parts=n_parts)
        for p in packets:
            ids=np.array([r["id"] for r in p["rows"]]); z=dict(ids=ids)
            if mode=="screen":
                z["screen"]=np.stack(p["screen"],axis=1)
            else:
                scores=[m.logits(x.cuda()) for x in p["branch"]]
                pq=np.stack([F(y[:,m.letters]) for y in scores],1)
                top=np.stack([y.argmax(-1).cpu().numpy() for y in scores],1)
                ref=np.array([known[k]["pq"] for k in ids])
                error=abs(ref-pq[:,0]).max(axis=1)
                good=(error<=C.N.SHAM_TOL)&(top[:,0]==[known[k]["top"] for k in ids])
                z.update(pq=pq,top=top,technical=good,numeric_diff=error,physical=p["physical"],alpha=p["source_alpha"],**p["anchor_data"])
            result.append(z)
    return {k:np.concatenate([p[k] for p in result]) for k in result[0]}
