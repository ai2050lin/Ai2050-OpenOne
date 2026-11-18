"""Exact native edits and a causal suffix replay with live Q at every layer."""

import numpy as np

import kv_temporal_run as C
import kv_origin_trace as O


F=O.F


def arm(early=(), code=None, memory=None, writes=()):
    return dict(early=list(early),code=code,memory=memory,writes=list(writes))


def position(head,kind="fact",restore=False):
    return O.source_mode("position",head,zero=(kind,),restore=(kind,) if restore else ())


def scope_plan(rows,scope):
    if scope!="early":
        return rows
    return [dict(r,token=dict(r["token"],ids=r["token"]["ids"][:r["token"]["slots"][0]])) for r in rows]


def init(m,p,mode,cfg):
    t=m.t
    width=p["ids"].shape[1]
    bars=t.tensor(p["padding"])
    slots=p["slots"][:,0]
    p["e"]=slots-1
    p["g"]=slots
    p["spare"]=p["slots"][:,2]
    p["c"]=width-1
    if mode in ("screen-e","early"):
        assert bool((p["e"]==width-1).all())
    p["pivot"]=int(p["e"].min()) if cfg["scope"]=="temporal" else width-1
    p["earliest"]=min([25,*[l for a in cfg["arms"].values() for l,_ in a["early"]]])
    p["origins"]={k:(t.arange(len(bars)),t.tensor([r["origins"][k][0] for r in p["rows"]])+bars) for k in ("fact","other")}
    p["origins"]["gen"]=(t.arange(len(bars)),p["g"])
    p["trace"],p["loc_error"],p["slot_native"]={},{},{}


def native_norms(m,l,q,k,v,mask,loc):
    alpha=m.read(q,k,mask)
    ids,slots=loc
    values=v.repeat_interleave(m.groups,dim=1)[ids,:,slots]
    a=alpha[ids,:,slots]
    terms=values.float()*a[...,None]
    writes=m.projections(l,tuple(range(m.heads)),terms)
    return writes,a


def screen(m,l,i,p,x,q,k,v,mask):
    t=m.t
    co=tuple(z.cuda() for z in p["origins"]["fact"])
    w,a=native_norms(m,l,q[:,:,-1:],k,v,mask[:,:,-1:],co)
    ids=t.tensor([r["state_ids"] for r in p["rows"]],device=x.device)
    bank=m.model.lm_head.weight.detach()
    v_read=(bank[ids[:,0]].float()-bank[ids[:,1]].float())*m.decoder.norm.weight.detach().float()
    rms=(x[:,-1].float().square().mean(-1)+1e-6).sqrt()
    score=t.einsum("bhm,bm->bh",w,v_read)/rms[:,None]
    p["trace"][i]=[F(z) for z in (score,w.norm(dim=-1),a)]


def branch_qkv(m,l,now,k,v,angles,pivot):
    t=m.t
    a,b,s,d=now.shape
    q,kk,vv=m.qkv(l,now.reshape(a*b,s,d),angles)
    if pivot:
        kk=t.cat([k[:,:,:pivot].repeat(a,1,1,1),kk],dim=2)
        vv=t.cat([v[:,:,:pivot].repeat(a,1,1,1),vv],dim=2)
    return q.reshape(a,b,*q.shape[1:]),kk.reshape(a,b,*kk.shape[1:]),vv.reshape(a,b,*vv.shape[1:])


def at(t,x,positions):
    if x.ndim==3:
        return x[t.arange(len(positions),device=x.device),positions]
    return x[:,t.arange(len(positions),device=x.device),positions]


def local_edit(m,l,h,q,k,v,vis,outputs,query,which,positions):
    t=m.t
    b=len(query)
    seq=t.arange(b,device=q.device)
    query_q=q[seq,:,query][:,:,None]
    local=vis[seq,0,query][:,None,None]
    dest=outputs[seq,query,h]
    edited,err=O.position_edit(m,l,h,query_q,k,v,local,dest,which,positions)
    outputs[seq,query,h]=edited
    return err


def unit_scores(m,l,rows,activations):
    w=l.mlp.down_proj.weight.detach().float()
    v=O.lm_vector(m).float()
    t=m.t
    sign=t.tensor([1 if r["code"]=="Q" else -1 for r in rows],device=w.device,dtype=t.float32)
    a,off=activations[0,:,-1].float(),activations[1,:,-1].float()
    dv=a-off
    length=w.norm(dim=0)
    return dict(score=F((dv*(v@w)*sign[:,None]).sum(0)),change=F((dv.abs()*length).sum(0)),intact=F((a.abs()*length).sum(0)))


def step(m,l,i,p,mode,cfg):
    t=m.t
    if i==0:
        init(m,p,mode,cfg)
    x=p["x"].cuda(); cs=tuple(z.cuda() for z in p["cs"]); mask=p["mask"].cuda()
    q0,k0,v0=m.qkv(l,x,cs)
    o0=m.attention(l,q0,k0,v0,mask)
    if mode=="screen-e":
        screen(m,l,i,p,x,q0,k0,v0,mask)
    p["x"]=m.residual(l,x,o0).cpu()
    if mode=="screen-e" or i<p["earliest"]:
        return
    specs=list(cfg["arms"].values())
    a,b=len(specs),len(x)
    pivot=p["pivot"]
    scope=cfg["scope"]
    suffix=x.shape[1]-pivot
    live=p["branches"].cuda() if "branches" in p else x[:,pivot:].unsqueeze(0).expand(a,-1,-1,-1)
    vis=mask[:,:,pivot:]
    ev=(p["e"]-pivot).cuda()
    gv=(p["g"]-pivot).cuda()
    pos=t.arange(b,device=x.device)
    loc={k:tuple(v.cuda() for v in z) for k,z in p["origins"].items()}
    if mode=="scan-c" and i==25:
        w,_=native_norms(m,l,q0[:,:,-1:],k0,v0,mask[:,:,-1:],loc["gen"])
        p["source_match"]=F(w.norm(dim=-1))
    if mode=="scan-c" and i==28:
        p["receiver_match"]=F(m.projections(l,tuple(range(m.heads)),o0[:,-1]).norm(dim=-1))
    angles=tuple(z[:,pivot:].repeat(a,1,1) for z in cs)
    q,k,v=branch_qkv(m,l,live,k0,v0,angles,pivot)
    vis_pack=vis.repeat(a,1,1,1)
    z=m.attention(l,q.flatten(0,1),k.flatten(0,1),v.flatten(0,1),vis_pack).reshape(a,b,suffix,m.heads,128)
    patched_k,patched_v=k,v
    if any(op["memory"] for op in specs):
        assert scope=="temporal"
        after=t.zeros((a,b,suffix),device=x.device,dtype=t.bool)
        patched_k,patched_v=k.clone(),v.clone()
        axis=t.arange(pivot,x.shape[1],device=x.device)[None]
        for j,op in enumerate(specs):
            memory=op["memory"]
            if memory is None:
                continue
            src=p["g"].cuda() if memory!="spare-rescue" else p["spare"].cuda()
            if memory in ("state-zero","state-self"):
                patched_v[j,pos,:,src]=0
                if memory=="state-self":
                    patched_v[j,pos,:,src]=v[j,pos,:,src]
            else:
                patched_k[j,pos,:,src]=k[0,pos,:,src]
                patched_v[j,pos,:,src]=v[0,pos,:,src]
                assert t.equal(patched_k[j,pos,:,src],k[0,pos,:,src]) and t.equal(patched_v[j,pos,:,src],v[0,pos,:,src])
            after[j]=axis>src[:,None]
        raw=m.attention(l,q.flatten(0,1),patched_k.flatten(0,1),patched_v.flatten(0,1),vis_pack)
        z=t.where(after[...,None,None],raw.reshape_as(z),z)
    edits=[]
    for j,op in enumerate(specs):
        kk,vv=patched_k[j],patched_v[j]
        for layer,which in op["early"]:
            if i==layer:
                err=local_edit(m,l,which["head"],q[j],k[j],v[j],vis,z[j],ev,which,loc)
                edits.append(err)
        if i==25 and op["code"]:
            which=op["code"]
            end=t.full((b,),suffix-1,device=x.device,dtype=t.long)
            err=local_edit(m,l,which["head"],q[j],kk,vv,vis,z[j],end,which,loc)
            edits.append(err)
    p["loc_error"][i]=max(edits,default=0.)
    interventions=[(j,*w[1:]) for j,op in enumerate(specs) for w in op["writes"] if w[0]==i]
    for j,k_type,hs in interventions:
        if k_type=="head":
            z[j,:,-1,hs]=z[0,:,-1,hs]
            assert t.equal(z[j,:,-1,hs],z[0,:,-1,hs])
    delta=l.self_attn.o_proj(z.reshape(a,b,suffix,m.width).contiguous())
    r=live+delta
    st=l.post_attention_layernorm(r)
    unit=any(k_type=="unit" for _,k_type,_ in interventions)
    if unit or (mode=="scan-c" and i in (27,37)):
        mlp=l.mlp
        aa=mlp.act_fn(mlp.gate_proj(st))*mlp.up_proj(st)
        if mode=="scan-c":
            p[f"unit{str(i)}"]=unit_scores(m,l,p["rows"],aa)
        for j,k_type,indices in interventions:
            if k_type=="unit":
                aa[j,:,-1,indices]=aa[0,:,-1,indices]
                assert t.equal(aa[j,:,-1,indices],aa[0,:,-1,indices])
        feed=mlp.down_proj(aa)
    else:
        feed=l.mlp(st)
    for j,k_type,_ in interventions:
        if k_type=="mlp":
            feed[j,:,-1]=feed[0,:,-1]
    y=r+feed
    fields=[(delta[:,:,-1]-delta[:1,:,-1]).float().norm(dim=-1),(feed[:,:,-1]-feed[:1,:,-1]).float().norm(dim=-1)]
    if scope=="temporal":
        for pos_at in (ev,gv):
            sample=y[:,pos,pos_at].float()
            fields.append((sample-sample[:1]).norm(dim=-1))
        gs=p["g"].cuda()
        k_here=k.permute(1,3,0,2,4)[pos,gs].transpose(0,1)
        v_here=v.permute(1,3,0,2,4)[pos,gs].transpose(0,1)
        fields.extend([(arr.float()-arr[:1].float()).flatten(2).norm(dim=-1) for arr in (k_here,v_here)])
        if i in (25,28):
            g_id=(18 if i==25 else 20)//m.groups
            p["slot_native"][i]=F(v[:,pos,g_id,gs]).swapaxes(0,1)
    p["trace"][i]=F(t.stack(fields,-1)).swapaxes(0,1)
    p["branches"]=y.cpu()


def finish(m,p,mode,cfg,known):
    if mode=="screen-e":
        return dict(ids=np.array([r["id"] for r in p["rows"]]),scan=np.stack([np.stack(z,1) for z in p["trace"].values()],1))
    t=m.t
    scope=cfg["scope"]
    rows=p["rows"]
    a,b,s,h=p["branches"].shape
    positions=t.arange(b,device="cuda")
    e=(p["e"]-p["pivot"]).cuda()
    options=t.tensor([r["state_ids"] for r in rows],device="cuda")
    tails=p["ids"].numpy()
    top=[]; pq=[]; states=[]; tokens=[]; valid=[]; first=[]; first_generated=[]
    for z in p["branches"]:
        scores=m.logits(z.cuda())
        best=scores.argmax(-1)
        top.append(best[:,-1].cpu().numpy())
        pq.append(F(scores[:,-1,m.letters]))
        if scope!="code":
            states.append(F(scores[positions[:,None],e[:,None],options]))
            tokens.append(best[positions,e].cpu().numpy())
        if scope=="temporal":
            hit=best.cpu().numpy()
            agreed=[]; changed=[]; emitted=[]
            for j,x0 in enumerate(e.tolist()):
                d=np.flatnonzero(hit[j,x0:-1]!=tails[j,p["pivot"]+x0+1:])
                agreed.append(not len(d))
                changed.append(int(d[0]) if len(d) else -1)
                emitted.append(int(hit[j,x0+d[0]]) if len(d) else -1)
            valid.append(agreed); first.append(changed); first_generated.append(emitted)
    top=np.stack(top,1); pq=np.stack(pq,1)
    raw_latent=p["x"].cuda()
    if scope=="early":
        ref=m.logits(raw_latent[:,-1:])[:,0]
        base=ref.argmax(-1).cpu().numpy(); diff=np.abs(F(ref[positions[:,None],options])-np.stack(states,1)[:,0]).max(-1)
        okay=(top[:,0]==base)&(base==options[:,0].cpu().numpy())&(diff<=C.N.SHAM_TOL)
    else:
        ref=np.array([known[r["id"]]["pq"] for r in rows]); expect=np.array([known[r["id"]]["top_token"] for r in rows])
        diff=np.abs(ref-pq[:,0]).max(-1)
        okay=(top[:,0]==expect)&(diff<=C.N.SHAM_TOL)
    out=dict(ids=np.array([r["id"] for r in rows]),pq=pq,top=top,technical=okay,numeric_diff=diff,
             trace=np.stack(list(p["trace"].values()),axis=2),origin_error=np.full(b,max(p["loc_error"].values())))
    if scope!="code":
        out.update(state=np.stack(states,1),state_top=np.stack(tokens,1))
    if scope=="temporal":
        out.update(prefix_valid=np.array(valid).T,first_mismatch=np.array(first).T,
                   first_generated=np.array(first_generated).T,
                   state_value25=p["slot_native"][25],state_value28=p["slot_native"][28])
        if not cfg.get("alternate"):
            out["technical"] &= out["prefix_valid"][:,0]
    if mode=="scan-c":
        out.update(source_match=p["source_match"],receiver_match=p["receiver_match"])
        for i in (27,37):
            for k,z in p[f"unit{i}"].items():
                # Retain per packet summaries with packet counts and aggregate on the study side.
                out[f"_{i}-{k}"]=z
    return out


def run_panel(m,rows,mode,cfg,known):
    data=scope_plan(rows,cfg["scope"])
    flat=[]
    limit=8*(C.SLAB if cfg["scope"]=="temporal" else C.N.SLAB)
    for j in range(0,len(data),limit):
        for _,group in m.batches(data[j:j+limit],mode,cfg):
            flat.extend(finish(m,p,mode,cfg,known) for p in group)
    return {k:(np.stack([x[k] for x in flat]) if k.startswith("_") else np.concatenate([x[k] for x in flat])) for k in flat[0]}
