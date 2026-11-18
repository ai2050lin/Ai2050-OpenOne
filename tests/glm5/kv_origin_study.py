"""Discovery-only causal selection and precommitted held-out work order."""

from itertools import batched
import math
import time

import numpy as np

import kv_origin_run as C
import kv_origin_trace as T


A,B=C.A,C.B
ATTS=[(i,"att") for i in C.LAYERS if i>25]
MLPS=[(i,"mlp") for i in C.LAYERS]


def signed(d,rows):
    return (d["pq"][:,:,1].astype(np.float64)-d["pq"][:,:,0])*np.array([1 if r["code"]=="Q" else -1 for r in rows])[:,None]


def module_config():
    arms=T.variants()
    for l,k in (*ATTS,*MLPS):
        arms[f"res-{l}-{k}"]=T.command(T.source_mode("full"),(l,k,"restore",[]))
    return dict(arms=arms)


def units_config(head_layer,mlp_layer,heads):
    arms=T.variants()
    for h in range(heads):
        arms[f"res-{h}"]=T.command(T.source_mode("full"),(head_layer,"head","restore",[h]))
    return dict(arms=arms,heads=head_layer,units=mlp_layer)


def matched_heads(x,selected,rng):
    scores=np.log(np.maximum(x,1e-7)).mean(0)
    order=sorted((h for h in range(len(scores)) if h!=selected),key=lambda h:(abs(scores[h]-scores[selected]),h))
    pool=order[:len(scores)//4]
    return rng.choice(pool,2,replace=False).tolist(),pool


def matched_units(change,intact,targets,rng):
    points=np.log(np.maximum(np.stack([change,intact],axis=-1),1e-8))
    used=set(targets)
    result=[]
    for _ in range(2):
        keep=[]
        for unit in targets:
            dist=((points-points[unit])**2).sum(-1)
            dist[list(used)]=np.inf
            pool=np.argpartition(dist,24)[:24]
            pick=int(rng.choice(sorted(pool)))
            keep.append(pick)
            used.add(pick)
        result.append(keep)
    return result


def final_config(f):
    d=T.positional(f["source_controls"])
    l,h=f["head"]
    mlp=f["mlp"]
    for tag,layer,kind,indices in [("head",l,"head",[h]),("unit",mlp,"unit",f["units"]),
                                   ("att",l,"att",[]),("mlp",mlp,"mlp",[])]:
        d["res-"+tag]=T.command(T.source_mode("full"),(layer,kind,"restore",indices))
        d["block-"+tag]=T.command(None,(layer,kind,"zero",indices))
    for j,(h0,units) in enumerate(zip(f["head_controls"],f["unit_controls"],strict=True),1):
        for typ,layer,val in (("head",l,[h0]),("unit",mlp,units)):
            d[f"res-{typ}-rand{j}"]=T.command(T.source_mode("full"),(layer,typ,"restore",val))
            d[f"block-{typ}-rand{j}"]=T.command(None,(layer,typ,"zero",val))
    return dict(arms=d,candidate=f["head"],heads=l,units=mlp,units_selected=f["units"])


def discover(m,p):
    assert not (C.OUT/"confirm-behavior.json").exists() and not (C.OUT/"freeze.json").exists()
    rows=A.load(p["materials"]["discover"]["name"])
    name="discover-behavior.json"
    if not (C.OUT/name).exists():
        obs,_=m.observe(rows,"discover-behavior")
        A.save(name,C.Prev.behavior(rows,obs))
    b=A.load(name)
    assert b["passed"],b["cells"]
    use=C.resolved(rows,b)
    first=module_config()
    key="discover-modules.npz"
    if not (C.OUT/key).exists():
        data,_=T.panel(m,use,"module-scan",first,b)
        B.write_np(key,data)
    with np.load(C.OUT/key) as d:
        assert list(d["ids"])==[x["id"] for x in use]
        pq=signed(d,use)
    means={a:float((pq[:,j]-pq[:,1]).mean()) for j,a in enumerate(first["arms"]) if a not in ("sham","u")}
    best_att=max(ATTS,key=lambda k:means[f"res-{k[0]}-{k[1]}"])[0]
    best_mlp=max(MLPS,key=lambda k:means[f"res-{k[0]}-{k[1]}"])[0]
    second=units_config(best_att,best_mlp,m.heads)
    k2="discover-units.npz"
    if not (C.OUT/k2).exists():
        data,traces=T.panel(m,use,"unit-scan",second,b)
        source=np.concatenate([z["source_scan"] for z in traces])
        head_norm=np.concatenate([z["head_scan"] for z in traces])
        unit={k:np.stack([z["unit_scan"][k] for z in traces]).sum(0,dtype=np.float64)/len(use)
              for k in ("signed_sum","abs_change_sum","abs_intact_sum")}
        B.write_np(k2,dict(**data,source_scan=source,head_norm=head_norm,**unit))
    with np.load(C.OUT/k2) as d:
        assert list(d["ids"])==[x["id"] for x in use]
        scores=signed(d,use)
        gains=(scores[:,2:]-scores[:,[1]]).mean(0)
        head=int(gains.argmax())
        units=np.argsort(-d["signed_sum"],kind="stable")[:C.NUNIT].tolist()
        rng=np.random.default_rng(C.SEED)
        source_control={}
        pools={}
        for j,name0 in enumerate(("u",*T.SOURCE)):
            source_control[name0],pools[name0]=matched_heads(d["source_scan"][:,j],18,rng)
        heads,pools["receiver"]=matched_heads(d["head_norm"],head,rng)
        random_units=matched_units(d["abs_change_sum"],d["abs_intact_sum"],units,rng)
        feature_details=dict(target_change=float(d["abs_change_sum"][units].sum()),target_intact=float(d["abs_intact_sum"][units].sum()),
             control_change=[float(d["abs_change_sum"][z].sum()) for z in random_units],
             control_intact=[float(d["abs_intact_sum"][z].sum()) for z in random_units])
    f=dict(plan=B.sha("plan.json"),behavior=B.sha(name),time=C.stamp(),discovery={key:B.sha(key),k2:B.sha(k2)},
           head=[best_att,head],mlp=best_mlp,units=units,module_scores=means,head_scores=gains.tolist(),
           source_controls=source_control,head_controls=heads,unit_controls=random_units,head_pools=pools,unit_dose=feature_details,
           positions=list(T.POSITIONS),families=len(use)//8)
    f["arms"]=list(final_config(f)["arms"])
    A.save("freeze.json",f)
    print("Frozen candidates",f["head"],f["mlp"],f["source_controls"],f["head_controls"],flush=True)


def confirm(m,p):
    f=A.load("freeze.json")
    assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-behavior.json")
    assert all(B.sha(name)==h for name,h in f["discovery"].items())
    config=final_config(f)
    assert list(config["arms"])==f["arms"]
    rows=A.load(p["materials"]["confirm"]["name"])
    freeze=B.sha("freeze.json")
    key="confirm-behavior.json"
    if not (C.OUT/key).exists():
        obs,_=m.observe(rows,"confirm-behavior")
        A.save(key,dict(**C.Prev.behavior(rows,obs),freeze=freeze))
    b=A.load(key)
    assert b["freeze"]==freeze
    if not b["passed"]:
        A.save("confirm.json",dict(state="behavior-gate-failed",behavior=B.sha(key),freeze=freeze,shards={}))
        return
    use=C.resolved(rows,b)
    size=C.Prev.SLAB*C.Prev.BATCH
    count=math.ceil(len(use)/size)
    clock=time.monotonic()
    files={}
    C.Prev.progress("confirm-shards",0,count,clock)
    for j,part in enumerate(batched(use,size),1):
        name=f"confirm-{j:03d}.npz"
        if not (C.OUT/name).exists():
            d,z=T.panel(m,list(part),"confirm-path",config,b)
            errors=np.stack([x["origin_err"] for x in z])
            assert len(d["ids"])==len(part)
            B.write_np(name,dict(**d,source_kernel_error=np.repeat(errors,C.Prev.BATCH,axis=0)[:len(part)]))
        with np.load(C.OUT/name) as d:
            assert list(d["ids"])==[x["id"] for x in part]
        files[name]=B.sha(name)
        C.Prev.progress("confirm-shards",j,count,clock)
    A.save("confirm.json",dict(state="finished",behavior=B.sha(key),freeze=freeze,shards=files,families=len(use)//8))
