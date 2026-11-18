"""Resume the frozen holdout and replay near-tied shams in the full native shape."""

from contextlib import contextmanager
from itertools import batched
import math
import time

import numpy as np

import kv_origin_run as C
from kv_origin_study import final_config
import kv_origin_trace as T


A,B,P=C.A,C.B,C.P


@contextmanager
def original_arms_output(m, full_logits):
    """Keep the standard output checker, supplying the original-shape head output."""
    original=m.logits
    try:
        m.logits=lambda _: full_logits.cuda()
        yield
    finally:
        m.logits=original


def replay(m,rows,config,known):
    t=m.t
    t.cuda.empty_cache()
    p=m.packet(rows)
    start=time.monotonic()
    C.Prev.progress("full-shape-recovery",0,40,start,family=rows[0]["family"])
    modes=list(config["arms"].values())
    a=len(modes)
    p["layers"],p["deltas"]=[],[]
    scope=min(p["starts"])-1
    for i in range(len(m.layers)):
        with m.block(i) as l:
            if i<25:
                m.base_step(l,i,p,"full-baseline",None)
            else:
                full_layer(m,l,i,p,config,modes,scope)
        C.Prev.progress("full-shape-recovery",i+1,40,start,family=rows[0]["family"])
    with original_arms_output(m,p.pop("full_logits")):
        result,meta=T.results(m,p,config,known)
    return result,meta


def full_layer(m,l,i,p,config,modes,scope):
    t=m.t
    x=p["x"].cuda()
    mask=p["mask"].cuda()
    cs=tuple(x.cuda() for x in p["cs"])
    vis=mask[:,:,-1:]
    heads,units,raw_att,raw_mlp,enter,outs=[],[],[],[],[],[]
    q_record=[]
    unit_projected=[]
    alpha=[]
    verror=[]
    final=[]
    candidates=config["candidate"]
    unit_here=i==config["units"]
    track_head=i==candidates[0]
    cut_activations=None
    if i==25:
        loc=T.encode_origins(t,p)
        q0,k0,v0=m.qkv(l,x,cs)
        o0=m.attention(l,q0,k0,v0,mask)
    for j,spec in enumerate(modes):
        z=x if i==25 else x.clone()
        if i==25:
            q,k,v,o=q0,k0,v0,o0.clone()
            edit=spec["source"]
            h=18 if edit is None else edit["head"]
            if edit:
                o[:,-1,h],err=T.position_edit(m,l,h,q[:,:,-1:],k,v,vis,o[:,-1,h],edit,loc)
            else:
                err=0.
            verror.append(err)
            group=h//m.groups
            s=(q[:,h,-1:].float()*k[:,group].float()).sum(-1)*(128**-.5)
            zalpha=s.masked_fill(~vis[:,0,0],-t.inf).softmax(-1)
            alpha.append(T.attention_masses(t,zalpha,loc))
        else:
            z[:,-1]=p["branches"][j].cuda()
            q,k,v=m.qkv(l,z,cs)
            o=m.attention(l,q,k,v,mask)
        op=spec["patch"]
        kind,action,where=op[1:] if op and op[0]==i else (None,None,None)
        if j==0:
            intact_o=o[:,-1].clone()
        if kind=="head":
            o[:,-1,where]=intact_o[:,where] if action=="restore" else 0
        if track_head or i==25:
            h=candidates[1] if track_head else 18
            heads.append(T.F(o[:,-1,h]))
            if track_head:
                q_record.append(T.F(q[:,h,-1]))
        att=l.self_attn.o_proj(o.reshape(*z.shape[:-1],m.width).contiguous())
        if kind=="att":
            att[:,-1]=raw_att[0] if action=="restore" else 0
        r=z+att
        state=l.post_attention_layernorm(r)
        if unit_here:
            mlp=l.mlp
            act=mlp.act_fn(mlp.gate_proj(state))*mlp.up_proj(state)
            if j==0:
                cut_activations=act[:,-1].clone()
            if kind=="unit":
                act[:,-1,where]=cut_activations[:,where] if action=="restore" else 0
            pick=config["units_selected"]
            chosen=act[:,-1,pick]
            units.append(T.F(chosen))
            unit_projected.append(T.F((chosen.float()@mlp.down_proj.weight[:,pick].float().T).norm(dim=-1)))
            feed=mlp.down_proj(act)
        else:
            feed=l.mlp(state)
        if kind=="mlp":
            feed[:,-1]=raw_mlp[0] if action=="restore" else 0
        y=r+feed
        raw_att.append(att[:,-1].clone())
        raw_mlp.append(feed[:,-1].clone())
        if j<2:
            enter.append((r[:,-1].float()-z[:,-1].float(),y[:,-1].float()-r[:,-1].float()))
        outs.append(y[:,-1].clone().cpu())
        if j==0:
            p["x"]=y.cpu()
        if i==39:
            final.append(m.logits(y[:,scope:])[:,-1].clone().cpu())
    if i==25:
        p["source_alpha"]=T.F(t.stack(alpha,1))
        p["source_head"]=np.stack(heads,1)
        p["origin_err"]=np.array(verror,dtype=np.float32)
    if track_head:
        qs=np.stack(q_record,1)
        p["receiver_qdiff"]=np.linalg.norm(qs-qs[:,:1],axis=-1)
        p["receiver_head"]=np.stack(heads,1)
    if unit_here:
        p["unit_activations"]=np.stack(units,1)
        p["unit_write_norm"]=np.stack(unit_projected,1)
    p["layers"].extend((v-w).cpu() for v,w in zip(enter[0],enter[1],strict=True))
    p["deltas"].append(np.stack([T.F((t.stack(xs,1).float()-xs[0][:,None].float()).norm(dim=-1)) for xs in (raw_att,raw_mlp)],axis=-1))
    p["branches"]=t.stack(outs)
    if i==39:
        p["full_logits"]=t.stack(final)


def run():
    import torch
    p=C.verify(); f=A.load("freeze.json"); b=A.load("confirm-behavior.json")
    assert f["plan"]==B.sha("plan.json") and b["freeze"]==B.sha("freeze.json") and b["passed"]
    assert all(B.sha(n)==h for n,h in f["discovery"].items())
    assert B.model_identity("q14")==p["model"]
    modes=final_config(f)
    assert list(modes["arms"])==f["arms"]
    known={o["id"]:o for o in b["rows"]}
    rows=C.resolved(A.load(p["materials"]["confirm"]["name"]),b)
    width=C.Prev.BATCH*C.Prev.SLAB
    total=math.ceil(len(rows)/width)
    clock=time.monotonic()
    fixed=[]
    files={}
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    with P.GpuLease(),torch.inference_mode():
        C.Prev.progress("recovery-load",0,1,clock)
        m=C.Engine()
        C.Prev.progress("recovery-load",1,1,clock)
        try:
            for j,ss in enumerate(batched(rows,width),1):
                atom=f"confirm-{j:03d}.npz"
                if not (C.OUT/atom).exists():
                    answers=[]
                    for _,packs in m.batches(list(ss),"confirm-path",modes):
                        for packet in packs:
                            raw=m.logits(packet["branches"].cuda())[0]
                            tok=raw.argmax(-1).cpu().numpy()
                            pp=T.F(raw[...,m.letters])
                            ids=[r["id"] for r in packet["rows"]]
                            expect=np.array([known[i]["top_token"] for i in ids])
                            q0=np.array([known[i]["pq"] for i in ids])
                            if (tok!=expect).any() or np.max(abs(pp-q0))>C.Prev.SHAM_TOL:
                                repaired,meta=replay(m,packet["rows"],modes,known)
                                fixed.append(dict(family=packet["rows"][0]["family"],ids=ids,
                                                  sham_single=tok.tolist(),sham_full=repaired["top"][:,0].tolist(),
                                                  expected=expect.tolist(),pq_single=pp.tolist(),pq_full=repaired["pq"][:,0].tolist(),
                                                  pq_original=q0.tolist()))
                                A.save("recovery-counterexamples.json",dict(plan=B.sha("plan.json"),families=fixed))
                            else:
                                repaired,meta=T.results(m,packet,modes,known)
                            repaired["source_kernel_error"]=np.broadcast_to(meta["origin_err"],(len(ids),len(modes["arms"])))
                            answers.append(repaired)
                    B.write_np(atom,{k:np.concatenate([b[k] for b in answers]) for k in answers[0]})
                with np.load(C.OUT/atom) as d:
                    assert list(d["ids"])==[r["id"] for r in ss]
                files[atom]=B.sha(atom)
                C.Prev.progress("recovery-shards",j,total,clock)
        finally:
            m.close()
    A.save("recovery-counterexamples.json",dict(plan=B.sha("plan.json"),families=fixed))
    A.save("confirm.json",dict(state="finished",behavior=B.sha("confirm-behavior.json"),freeze=B.sha("freeze.json"),
                           families=len(rows)//8,shards=files,recovery=B.sha("recovery-counterexamples.json")))


if __name__=="__main__":
    C.setup(); run()
