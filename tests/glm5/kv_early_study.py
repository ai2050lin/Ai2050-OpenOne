"""Frozen early-to-cache route selection and matched temporal confirmation."""

from itertools import batched
import time

import numpy as np

import kv_early_native as T
import kv_early_run as C
from kv_origin_study import matched_heads
from kv_temporal_study import early_specs,restore_writes,technical


A,B=C.A,C.B


def code_arms(f):
    z=T.read(18,"zero")
    d={"sham":T.arm(),"c-block":T.arm(relay=z),"self-c":T.arm(relay=T.read(18,"self"))}
    for j in range(3):
        d["single-"+str(j)]=T.arm(relay=z,writes=restore_writes([j],f["features"]))
    d["joint"]=T.arm(relay=z,writes=restore_writes(range(3),f["features"]))
    for j in range(2):
        d[f"c-random{j+1}"]=T.arm(relay=T.read(f["c_controls"][j],"zero"))
        d[f"joint-random{j+1}"]=T.arm(relay=z,writes=restore_writes(range(3),f["control_features"][j]))
    return dict(scope="code",arms=d)


def temporal_arms(f):
    a=early_specs(f["early"])
    z=T.read(18,"zero")
    full=restore_writes(range(3),f["features"])
    d={"sham":T.arm(),"e-block":T.arm(early=a),"self-e":T.arm(early=early_specs(f["early"],restore=True)),
       "e-other":T.arm(early=early_specs(f["early"],"other")),"relay-k":T.arm(early=a,relay=T.read(18,"k")),
       "relay-v":T.arm(early=a,relay=T.read(18,"v")),"relay-kv":T.arm(early=a,relay=T.read(18,"kv")),
       "relay-reverse":T.arm(relay=T.read(18,"reverse",donor=1)),
       "ec-block":T.arm(early=a,relay=z),"ec-cache":T.arm(early=a,relay=T.read(18,"joint")),
       "ec-end":T.arm(early=a,relay=z,writes=full),"ec-joint":T.arm(early=a,relay=T.read(18,"joint"),writes=full),
       "c-block":T.arm(relay=z)}
    for j in range(2):
        d[f"e-random{j+1}"]=T.arm(early=early_specs(f["early_random"][j]))
        d[f"relay-random{j+1}"]=T.arm(early=a,relay=T.read(f["relay_controls"][j],"kv"))
    return dict(scope="temporal",arms=d)


def frozen_controls(p):
    return {k:p[k] for k in ("features","c_controls","control_features")}


def load_np(file):
    with np.load(C.OUT/file) as z:
        return {k:z[k] for k in z.files}


def store(m,rows,mode,cfg,known,file):
    if not (C.OUT/file).exists():
        z=T.panel(m,rows,mode,cfg,known)
        B.write_np(file,z)
    else:
        z=load_np(file)
    assert list(z["ids"])==[x["id"] for x in rows]
    return z


def observe(m,p,stage):
    file=f"{stage}-behavior.json"
    bind=B.sha("freeze.json" if stage=="confirm" else "plan.json")
    rows=A.load(p["materials"][stage]["file"])
    if not (C.OUT/file).exists():
        scores,_=m.observe(rows,stage+"-natural")
        A.save(file,dict(**C.N.behavior(rows,scores),binding=bind))
    seen=A.load(file)
    assert seen["binding"]==bind
    return rows,seen


def pilot(m,p):
    memo=A.load("pilot-behavior.json")
    rows=C.O.resolved(A.load("material-pilot.json"),memo)[:8]
    known={x["id"]:x for x in memo["rows"]}
    old=C.prior()[1]
    f=dict(features=old["features"],control_features=old["random"],c_controls=old["c_random"],early=[[9,8],[15,8],[22,22]],
           early_random=[[[9,4],[15,3],[22,4]],[[9,12],[15,6],[22,6]]],relay_controls=[15,17])
    probe=T.panel(m,rows,"probe",dict(scope="temporal",arms={"sham":T.arm(),"e-block":T.arm(early=early_specs(f["early"]))}),known)
    cfg=temporal_arms(f)
    z=T.panel(m,rows,"temporal",cfg,known)
    check=technical(z,cfg)
    assert z["technical"].all(),check
    assert np.isfinite(probe["kv_ratios"]).all() and np.isfinite(z["kv_ratios"]).all()
    ix=list(cfg["arms"]).index("relay-kv")
    assert np.array_equal(z["applied_k25"][:,ix],z["k25_before"][:,0]) and np.array_equal(z["applied_v25"][:,ix],z["v25_before"][:,0])
    assert np.array_equal(z["state"][:,1],z["state"][:,ix])
    A.save("pilot-engine.json",dict(passed=True,n=len(rows),checks=check,probe_max=float(probe["kv_ratios"].max()),
                                   timing_max_sham=float(z["numeric_diff"].max()),shapes={k:list(v.shape) for k,v in z.items()}))


def discover(m,p):
    assert not (C.OUT/"confirm-behavior.json").exists() and not (C.OUT/"freeze.json").exists()
    rows,b=observe(m,p,"discover"); assert b["passed"],b["cells"]
    use=C.O.resolved(rows,b); known={x["id"]:x for x in b["rows"]}
    scan_file="discover-field.npz"
    s=store(m,use,"screen-e",dict(scope="early",arms={"sham":T.arm()}),known,scan_file)["scan"]
    positive=np.mean(abs(s[:,:,0]),axis=0)
    dose=s[:,:,1].mean(0)
    by_decision=sorted([(i,int(v.argmax()),float(v.max()),"absolute State projection") for i,v in enumerate(positive)],key=lambda x:-x[2])[:4]
    exclude={x[0] for x in by_decision}
    by_dose=sorted([(i,int(v.argmax()),float(v.max()),"fact projected norm") for i,v in enumerate(dose) if i not in exclude],key=lambda x:-x[2])[:4]
    ranked=[*by_decision,*by_dose]
    arms={"sham":T.arm(),**{f"{l}-{h}":T.arm(early=early_specs([(l,h)])) for l,h,_,_ in ranked}}
    d=store(m,use,"probe",dict(scope="temporal",arms=arms),known,"discover-kv25.npz")
    scores=np.median(d["kv_ratios"][:,1:].mean(-1),axis=0)
    ix=np.argsort(-scores,kind="stable")[:3]
    early=[[ranked[j][0],ranked[j][1]] for j in ix]
    rng=np.random.default_rng(C.SEED)
    r=[[],[]]; pools={}
    for l,h in early:
        group,pool=matched_heads(s[:,l,1],h,rng)
        pools[str(l)]=pool
        for j,hh in enumerate(group): r[j].append([l,hh])
    # Other query heads of the same GQA group share exactly the same native K/V donor.
    receivers=rng.choice([h for h in range(40) if h//m.groups==18//m.groups and h!=18],2,replace=False).tolist()
    f=dict(**frozen_controls(p),early=early,early_random=r,pools=pools,relay_controls=receivers,n=len(use)//8,
           screening=[dict(layer=l,head=h,screen=s0,criterion=k,kv=float(scores[j]),
                           median_K=float(np.median(d['kv_ratios'][:,j+1,0])),median_V=float(np.median(d['kv_ratios'][:,j+1,1])))
                      for j,(l,h,s0,k) in enumerate(ranked)])
    # Discovery replay at the same endpoints assesses actual availability without changing the chosen group.
    cfg=temporal_arms(f)
    e=store(m,use,"temporal",cfg,known,"discover-route.npz"); cert=technical(e,cfg)
    m_id=list(cfg["arms"]).index("e-block")
    f.update(discover_effect=dict(n=int(e["technical"].sum()),raw_controls=cert,
                                  median_group_KV25=np.median(e["kv_ratios"][:,m_id],axis=0).tolist()),time=C.O.stamp(),
             plan=B.sha("plan.json"),behavior=B.sha("discover-behavior.json"),files={k:B.sha(k) for k in (scan_file,"discover-kv25.npz","discover-route.npz")},
             code_names=list(code_arms(f)["arms"]),temporal_names=list(cfg["arms"]))
    A.save("freeze.json",f)
    print("Frozen",early,"matched",r,"L25 shared group controls",receivers,flush=True)


def rollout(m,fit,switched,known,f):
    if switched:
        requests=[(x,token) for x,token in switched[:16]]
        title="early-fact-block"
        spec=T.arm(early=early_specs(f["early"]))
    else:
        # One role/distance view per family, in the predetermined material order.
        requests=[(r,None) for r in fit[::8][:16]]
        title="lexical-only-blackout"
        spec=T.arm(suppress=True)
    cfg=dict(scope="temporal",arms={"sham":T.arm(),"switch":spec})
    active=[]
    for r,token in requests:
        seq=r["token"]["ids"].copy()
        if token is not None: seq[r["token"]["slots"][0]]=token
        active.append(dict(r,token=dict(r["token"],ids=seq)))
    logs={r["id"]:[] for r in active}; final={}
    eos=m.model.generation_config.eos_token_id
    ends={eos} if isinstance(eos,int) else set(eos or [])
    for attempt in range(5):
        if not active: break
        z=T.panel(m,active,"rollout",cfg,known)
        pending=[]
        for r,t in zip(active,z,strict=True):
            slot=r["token"]["slots"][0]
            seq=r["token"]["ids"]
            offset=t["first_mismatch"]
            match=bool(t["verified"]) and m.tok.decode(seq[r["token"]["start"]:]).rstrip().endswith("Code:")
            state=m.tok.decode([t["state"]])
            changes=dict(try_index=attempt,**t,rendered_state=state)
            logs[r["id"]].append(changes)
            if match:
                text=m.tok.decode([*seq[r["token"]["start"]:],t["code"]])
                final[r["id"]]=dict(verified=True,state=state,code=m.tok.decode([t["code"]]),actual_text=text)
            elif offset<0 or t["actual_token"] in ends or attempt==4:
                stop=len(seq) if offset<0 else slot+offset
                prefix=seq[r["token"]["start"]:stop]+([] if offset<0 else [t["actual_token"]])
                final[r["id"]]=dict(verified=False,state=state,first_incompatible=offset,actual_prefix=m.tok.decode(prefix),
                                     stopped_token=t["actual_token"] if t["actual_token"] in ends else None)
            else:
                outcome=seq.copy(); outcome[slot+offset]=t["actual_token"]
                pending.append(dict(r,token=dict(r["token"],ids=outcome)))
        active=pending
    return dict(mode=title,real_early_switches=len(switched),tested=len(requests),observations=final,attempts=logs)


def confirm(m,p):
    f=A.load("freeze.json")
    assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-behavior.json")
    assert all(B.sha(k)==v for k,v in f["files"].items())
    rows,behavior=observe(m,p,"confirm")
    if not behavior["passed"]:
        A.save("confirm.json",dict(state="behavior-gate-failed",behavior=B.sha("confirm-behavior.json"),freeze=B.sha("freeze.json")))
        return
    use=C.O.resolved(rows,behavior); known={x["id"]:x for x in behavior["rows"]}; keep={}; tapes={}; temporal=[]; masks=[]
    for mode,cfg in (("code",code_arms(f)),("temporal",temporal_arms(f))):
        assert list(cfg["arms"])==f[f"{mode}_names"]
        clock=time.monotonic(); keep[mode]={}; tapes[mode]={}; accepted=[]
        groups=list(batched(use,8*(C.SLAB if mode=="temporal" else C.N.SLAB)))
        C.N.progress(f"{mode}-shards",0,len(groups),clock)
        for j,part in enumerate(groups,1):
            key=f"{mode}-{j:03d}.npz"; status=f"{mode}-{j:03d}-acceptance.json"
            if not (C.OUT/key).exists() or not (C.OUT/status).exists():
                x=T.panel(m,list(part),mode,cfg,known)
                checks=technical(x,cfg)
                B.write_np(key,x)
                A.save(status,dict(checks=checks,invalid_ids=x["ids"][~x["technical"]].tolist(),freeze=B.sha("freeze.json")))
            else:
                x=load_np(key)
            assert list(x["ids"])==[r["id"] for r in part]
            accepted.extend(x["technical"].tolist())
            if mode=="temporal": temporal.extend(x["state_top"][:,f["temporal_names"].index("e-block")].tolist())
            keep[mode][key]=B.sha(key); tapes[mode][status]=B.sha(status)
            C.N.progress(f"{mode}-shards",j,len(groups),clock)
        masks.append(accepted)
    fitmask=np.repeat(np.asarray(masks).all(0).reshape(-1,8).all(1),8)
    fit=[r for r,k in zip(use,fitmask,strict=True) if k]
    switched=[(r,t) for r,ok,t in zip(use,fitmask,temporal,strict=True) if ok and t!=r["state_ids"][0]]
    if not (C.OUT/"rollout.json").exists():
        A.save("rollout.json",rollout(m,fit,switched,known,f))
    A.save("confirm.json",dict(state="finished",behavior=B.sha("confirm-behavior.json"),freeze=B.sha("freeze.json"),
                               panels=keep,acceptance=tapes,rollout=B.sha("rollout.json"),n=len(use)//8))
