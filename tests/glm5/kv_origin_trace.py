"""Dynamic-Q, one-query original BF16 interventions and residual accounting."""

import numpy as np

import kv_origin_run as C


A,B=C.A,C.B
F=C.Prev.Native.f32
SOURCE=("fact","book","gen")
POSITIONS=(*SOURCE,"other")


def command(source=None,patch=None):
    return dict(source=source,patch=patch)


def source_mode(kind,h=18,zero=(),restore=()):
    return dict(kind=kind,head=h,zero=tuple(zero),restore=tuple(restore))


def variants():
    return {"sham":command(),"u":command(source_mode("full"))}


def positional(controls):
    d=variants()
    d["self-u"]=command(source_mode("full-self"))
    positions={**{k:(k,) for k in POSITIONS},"prompt":SOURCE[:2],"all":SOURCE}
    for label,names in positions.items():
        d[label]=command(source_mode("position",zero=names))
        if label!="other":
            d["self-"+label]=command(source_mode("position",zero=names,restore=names))
    for k in SOURCE:
        d["restore-"+k]=command(source_mode("position",zero=SOURCE,restore=(k,)))
    selection={k:controls for k in ("u",*SOURCE)} if isinstance(controls,list) else controls
    for j,h in enumerate(selection["u"],1):
        d[f"u-rand{j}"]=command(source_mode("full",h))
    for k in SOURCE:
        for j,h in enumerate(selection[k],1):
            d[f"{k}-rand{j}"]=command(source_mode("position",h,zero=(k,)))
    return d


def pilot_arms():
    d=positional([20,28])
    d["pilot-head"]=command(source_mode("full"),(26,"head","restore",[7]))
    d["pilot-unit"]=command(source_mode("full"),(27,"unit","restore",list(range(C.NUNIT))))
    d["pilot-mlp"]=command(source_mode("full"),(28,"mlp","restore",[]))
    return dict(arms=d,track=True,units=27,heads=26)


def columns(row,key):
    return [row["token"]["slots"][0]] if key=="gen" else row["origins"][key]


def encode_origins(t,p):
    positions={}
    rows,pads=p["rows"],p["padding"]
    device="cuda"
    for key in POSITIONS:
        z=[[(x,i+pad) for i in columns(r,key)] for x,(r,pad) in enumerate(zip(rows,pads,strict=True))]
        pairs=[v for sub in z for v in sub]
        idx=t.tensor(pairs,device=device,dtype=t.long)
        positions[key]=(idx[:,0],idx[:,1])
    return positions


def attention_masses(t,alpha,positions):
    values=[]
    n=alpha.shape[0]
    for key in POSITIONS:
        r,c=positions[key]
        dst=t.zeros(n,device=alpha.device,dtype=t.float32)
        dst.index_add_(0,r,alpha[r,c])
        values.append(dst)
    return t.stack(values,-1)


def position_edit(m,l,h,q,k,v,vis,base,edit,positions):
    t=m.t
    if edit["kind"]=="full":
        return t.zeros_like(base),0.
    if edit["kind"]=="full-self":
        wiped=t.zeros_like(base)
        wiped.copy_(base)
        return wiped,0.
    g=h//m.groups
    vv=v.clone()
    for key in edit["zero"]:
        r,c=positions[key]
        vv[r,g,c]=0
    for key in edit["restore"]:
        r,c=positions[key]
        vv[r,g,c]=v[r,g,c]
    for key in set(edit["zero"])-set(edit["restore"]):
        r,c=positions[key]
        assert not bool(vv[r,g,c].any())
    raw=m.foreground(l,h,q,k,v,vis)
    after=m.foreground(l,h,q,k,vv,vis)
    # Apply the corresponding same-kernel response change to the intact full SDPA head. The cast is BF16.
    effective=(base.float()+after.float()-raw.float()).to(base.dtype)
    scores=(q[:,h].float()*k[:,g].float()).sum(-1)*(128**-.5)
    alpha=scores.masked_fill(~vis[:,0,0],-t.inf).softmax(-1)
    analytic=(alpha[...,None]*(vv[:,g].float()-v[:,g].float())).sum(-2)
    error=float(t.linalg.vector_norm(effective.float()-base.float()-analytic,dim=-1).max())
    return effective,error


def lm_vector(m):
    return (m.model.lm_head.weight[m.letters[1]].detach().double()-m.model.lm_head.weight[m.letters[0]].detach().double())*m.decoder.norm.weight.detach().double()


def split_projection(m,l,o,heads):
    return m.projections(l,tuple(heads),o[...,list(heads),:]).float()


def unit_features(m,layer,p,active):
    t=m.t
    a0,a1=active[0].float(),active[1].float()
    weight=layer.mlp.down_proj.weight.detach().float()
    sign=t.as_tensor([1 if r["code"]=="Q" else -1 for r in p["rows"]],device=active.device,dtype=t.float32)
    read=lm_vector(m).float()@weight
    dv=a0-a1
    wnorm=t.linalg.vector_norm(weight,dim=0)
    p["unit_scan"]=dict(signed_sum=F((sign[:,None]*dv*read).sum(0)),
                         abs_change_sum=F((dv.abs()*wnorm).sum(0)),abs_intact_sum=F((a0.abs()*wnorm).sum(0)),n=len(a0))


def step(m,l,i,p,mode,config):
    t=m.t
    names=config["arms"]
    a=len(names)
    x=p["x"].cuda()
    b=len(x)
    mask=p["mask"].cuda()
    vis=mask[:,:,-1:]
    cs=tuple(z.cuda() for z in p["cs"])
    q,k,v=m.qkv(l,x,cs)
    o=m.attention(l,q,k,v,mask)
    if i==25:
        cur=x[:,-1].unsqueeze(0).expand(a,-1,-1)
        before=o[:,-1].unsqueeze(0).expand(a,-1,-1,-1)
        loc=encode_origins(t,p)
        pots=[]
        edited=before.clone()
        err=[]
        for j,spec in enumerate(names.values()):
            edit=spec["source"]
            h=18 if edit is None else edit["head"]
            if edit:
                edited[j,:,h],delta=position_edit(m,l,h,q[:,:,-1:],k,v,vis,before[j,:,h],edit,loc)
            else:
                delta=0.
            err.append(delta)
            g=h//m.groups
            score=(q[:,h,-1:].float()*k[:,g].float()).sum(-1)*(128**-.5)
            alpha=score.masked_fill(~vis[:,0,0],-t.inf).softmax(-1)
            pots.append(attention_masses(t,alpha,loc))
        p["source_alpha"]=F(t.stack(pots,1))
        p["source_head"]=F(edited[:,:,18]).swapaxes(0,1)
        p["origin_err"]=np.array(err,dtype=np.float32)
        p["layers"]=[]
        p["deltas"]=[]
        if mode=="unit-scan":
            proj=split_projection(m,l,before[0],range(m.heads))
            totals=[proj.norm(dim=-1)]
            all_alpha=m.read(q[:,:,-1:],k,vis)
            all_v=v.repeat_interleave(m.groups,dim=1)
            for key in SOURCE:
                rows,cols=loc[key]
                z=t.zeros((b,m.heads,128),device=x.device,dtype=t.float32)
                z.index_add_(0,rows,all_v[rows,:,cols].float()*all_alpha[rows,:,cols,None])
                totals.append(split_projection(m,l,z,range(m.heads)).norm(dim=-1))
            p["source_scan"]=F(t.stack(totals,1))
    else:
        cur=p["branches"].cuda()
        angles=tuple(z[:,-1:].repeat(a,1,1) for z in cs)
        qq,kk,vv=m.branch_qkv(l,cur.reshape(a*b,m.width),k.repeat(a,1,1,1),v.repeat(a,1,1,1),angles)
        before=m.attention(l,qq,kk,vv,vis.repeat(a,1,1,1)).reshape(a,b,m.heads,128)
        edited=before
    interventions=[(j,spec["patch"]) for j,spec in enumerate(names.values()) if spec["patch"] and spec["patch"][0]==i]
    heads=[(j,op,where) for j,(_,kind,op,where) in interventions if kind=="head"]
    if heads:
        edited=before.clone()
        for j,op,where in heads:
            edited[j,:,where]=before[0,:,where] if op=="restore" else 0
    att=l.self_attn.o_proj(edited.reshape(a,b,m.width))
    for j,(_,kind,op,where) in interventions:
        if kind=="att":
            att[j]=att[0] if op=="restore" else 0
    r=cur+att
    state=l.post_attention_layernorm(r)
    unit_ops=[(j,op,where) for j,(_,kind,op,where) in interventions if kind=="unit"]
    if unit_ops or (mode=="unit-scan" and i==config["units"]):
        mlp=l.mlp
        act=mlp.act_fn(mlp.gate_proj(state))*mlp.up_proj(state)
        if mode=="unit-scan":
            unit_features(m,l,p,act)
        for j,op,where in unit_ops:
            act[j,:,where]=act[0,:,where] if op=="restore" else 0
            if op=="restore":
                assert t.equal(act[j,:,where],act[0,:,where])
        if "units_selected" in config and i==config["units"]:
            pick=config["units_selected"]
            selected=act[...,pick]
            p["unit_activations"]=F(selected).swapaxes(0,1)
            p["unit_write_norm"]=F((selected.float()@mlp.down_proj.weight[:,pick].float().T).norm(dim=-1)).T
        feed=mlp.down_proj(act)
    else:
        feed=l.mlp(state)
    for j,(_,kind,op,where) in interventions:
        if kind=="mlp":
            feed[j]=feed[0] if op=="restore" else 0
    y=r+feed
    real_att=r[:2].float()-cur[:2].float()
    real_mlp=y[:2].float()-r[:2].float()
    p["layers"].append((real_att[0]-real_att[1]).cpu())
    p["layers"].append((real_mlp[0]-real_mlp[1]).cpu())
    p["deltas"].append(F(t.stack(((att.float()-att[:1].float()).norm(dim=-1),
                                   (feed.float()-feed[:1].float()).norm(dim=-1)),dim=-1)).swapaxes(0,1))
    if mode=="unit-scan" and i==config["heads"]:
        proj=split_projection(m,l,before[0],range(m.heads))
        p["head_scan"]=F(proj.norm(dim=-1))
    if "candidate" in config and i==config["candidate"][0]:
        target=config["candidate"][1]
        p["receiver_head"]=F(edited[:,:,target]).swapaxes(0,1)
        rr=qq[:,target].reshape(a,b,128).float()
        p["receiver_qdiff"]=F((rr-rr[:1]).norm(dim=-1)).T
    p["branches"]=y.cpu()
    p["x"]=m.residual(l,x,o).cpu()


def integral(m,p):
    """Exact realized increments; integrate only the continuous RMSNorm readout."""
    # CPU float64 keeps integration and cancellation checks independent of CUDA BF16.
    f=p["branches"][:2].numpy(force=True) if p["branches"].dtype==m.t.float32 else p["branches"][:2].float().numpy()
    y0,y1=f[1].astype(np.float64),f[0].astype(np.float64)
    v=lm_vector(m).cpu().numpy()
    eps=m.decoder.norm.variance_epsilon
    signs=np.array([1 if row["code"]=="Q" else -1 for row in p["rows"]],dtype=np.float64)
    g=np.zeros_like(y0)
    xx,ww=np.polynomial.legendre.leggauss(16)
    for x,w in zip(xx,ww,strict=True):
        mid=y0+(x+1)/2*(y1-y0)
        rms=np.sqrt((mid*mid).mean(-1)+eps)
        g+=(v[None]/rms[:,None]-((mid@v)/(mid.shape[-1]*rms**3))[:,None]*mid)*(w/2)
    g*=signs[:,None]
    by_site=np.stack([(x.numpy().astype(np.float64)*g).sum(-1) for x in p["layers"]],axis=-1)

    def read(y):
        return signs*(y@v)/np.sqrt((y*y).mean(-1)+eps)

    true=read(y1)-read(y0)
    diff=by_site.sum(-1)-true
    assert np.max(abs(diff))<1e-6, np.max(abs(diff))
    return dict(readout_sites=by_site,readout_total=true,readout_closure=diff)


def results(m,p,config,known):
    raw=m.logits(p["branches"].cuda())
    pq=F(raw[...,m.letters]).swapaxes(0,1)
    top=raw.argmax(-1).cpu().numpy().T
    expect=[known[r["id"]] for r in p["rows"]]
    untreated=np.asarray([r["pq"] for r in expect],dtype=np.float32)
    assert np.array_equal(top[:,0],[r["top_token"] for r in expect]),(top[:,0],expect)
    assert np.max(abs(untreated-pq[:,0]))<=C.Prev.SHAM_TOL,(untreated,pq[:,0])
    for i,n in enumerate(config["arms"]):
        if n.startswith("self-"):
            assert np.array_equal(top[:,i],top[:,0]) and np.max(abs(pq[:,i]-pq[:,0]))<=C.Prev.RESTORE_TOL,n
    core=dict(ids=np.array([r["id"] for r in p["rows"]]),pq=pq,top=top,untreated=untreated,
              source_alpha=p["source_alpha"],source_head=p["source_head"],
              actual_output_norm=np.stack(p["deltas"],2),**integral(m,p))
    if "candidate" in config:
        core.update(receiver_qdiff=p["receiver_qdiff"],receiver_head=p["receiver_head"],
                    unit_activations=p["unit_activations"],unit_write_norm=p["unit_write_norm"])
    meta={key:p[key] for key in ("source_scan","head_scan","unit_scan") if key in p}
    meta["origin_err"]=p["origin_err"]
    return core,meta


def panel(m,rows,mode,config,behavior):
    known={x["id"]:x for x in behavior["rows"]}
    complete=[]
    for _,packets in m.batches(rows,mode,config):
        complete.extend(results(m,p,config,known) for p in packets)
    return {k:np.concatenate([r[k] for r,_ in complete]) for k in complete[0][0]},[x for _,x in complete]


def discover(m,p):
    from kv_origin_study import discover as run
    return run(m,p)


def confirm(m,p):
    from kv_origin_study import confirm as run
    return run(m,p)
