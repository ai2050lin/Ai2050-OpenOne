"""Discovery, frozen same-layer controls, untouched confirmation, and exact replay."""

from itertools import combinations,batched
import math
import time

import numpy as np

import kv_temporal_run as C
import kv_temporal_native as T
from kv_origin_study import matched_heads,matched_units,signed


A,B=C.A,C.B
SETS={"".join(map(str,z)):z for n in (1,2,3) for z in combinations(range(3),n)}


def restore_writes(part,features):
    specs=list(zip((27,28,37),(("mlp",[]) if features is None else ("unit",features[0]),
                              ("head",[20]) if features is None else ("head",[features[1]]),
                              ("mlp",[]) if features is None else ("unit",features[2])),strict=True))
    return [(specs[i][0],*specs[i][1]) for i in part]


def early_specs(pairs,kind="fact",restore=False):
    return [(l,T.position(h,kind,restore)) for l,h in pairs]


def code_arms(f=None):
    src=T.position(18,"gen")
    d={"sham":T.arm(),"c-block":T.arm(code=src)}
    if f is None:
        return dict(scope="code",arms=d)
    d["self-c"]=T.arm(code=T.position(18,"gen",True))
    for k,v in SETS.items():
        d["coarse-"+k]=T.arm(code=src,writes=restore_writes(v,None))
        d["fine-"+k]=T.arm(code=src,writes=restore_writes(v,f["features"]))
    for j in range(2):
        d[f"c-rand{j+1}"]=T.arm(code=T.position(f["c_random"][j],"gen"))
        if f.get("joint"):
            d[f"fine-rand{j+1}"]=T.arm(code=src,writes=restore_writes(SETS[f["joint"]],f["random"][j]))
    return dict(scope="code",arms=d)


def temporal_arms(f):
    candidate=early_specs(f["early"])
    end=T.position(18,"gen")
    rescue=restore_writes(SETS[f["joint"]],f["features"])
    d={"sham":T.arm(),"e-block":T.arm(early=candidate),"self-e":T.arm(early=early_specs(f["early"],restore=True)),
       "e-rand1":T.arm(early=early_specs(f["early_random"][0])),"e-rand2":T.arm(early=early_specs(f["early_random"][1])),
       "e-other":T.arm(early=early_specs(f["early"],"other")),"g-block":T.arm(memory="state-zero"),"self-g":T.arm(memory="state-self"),
       "g-rescue":T.arm(early=candidate,memory="state-rescue"),"g-spare":T.arm(early=candidate,memory="spare-rescue"),
       "ec-block":T.arm(early=candidate,code=end),"ec-g":T.arm(early=candidate,code=end,memory="state-rescue"),
       "ec-fine":T.arm(early=candidate,code=end,writes=rescue),
       "ec-joint":T.arm(early=candidate,code=end,memory="state-rescue",writes=rescue)}
    return dict(scope="temporal",arms=d)


def n_ok(rows,d,name):
    assert list(d["ids"])==[p["id"] for p in rows]
    keep=np.repeat(d["technical"].reshape(-1,8).all(-1),8)
    assert keep.sum()>=8, name
    return keep


def read_data(name):
    with np.load(C.OUT/name) as z:
        return {k:z[k] for k in z.files}


def stored(m,rows,mode,design,known,name):
    if not (C.OUT/name).exists():
        z=T.run_panel(m,rows,mode,design,known)
        B.write_np(name,z)
        return z
    z=read_data(name)
    assert list(z["ids"])==[x["id"] for x in rows]
    return z


def observe(m,p,stage):
    name=f"{stage}-behavior.json"
    rows=A.load(p["materials"][stage]["name"])
    binding=B.sha("freeze.json" if stage=="confirm" else "plan.json")
    if not (C.OUT/name).exists():
        obs,_=m.observe(rows,stage+"-natural")
        A.save(name,dict(**C.N.behavior(rows,obs),binding=binding))
    seen=A.load(name)
    assert seen["binding"]==binding
    return rows,seen


def technical(z,cfg):
    checks={}
    for j,k in enumerate(cfg["arms"]):
        if not k.startswith("self-"):
            continue
        err=np.max(abs(z["pq"][:,j]-z["pq"][:,0]),axis=-1)
        valid=(err<=C.N.RESTORE_TOL)&(z["top"][:,j]==z["top"][:,0])
        if "state" in z:
            valid &= (np.max(abs(z["state"][:,j]-z["state"][:,0]),axis=-1)<=C.N.RESTORE_TOL)&(z["state_top"][:,j]==z["state_top"][:,0])
        if "first_mismatch" in z:
            valid &= (z["first_mismatch"][:,j]==z["first_mismatch"][:,0])
        checks[k]=(float(err.max()),int((~valid).sum()))
        z["technical"] &= valid
    return checks


def pilot(m,p):
    rows=A.load("material-pilot.json"); seen=A.load("pilot-behavior.json")
    assert seen["passed"]
    rows=C.O.resolved(rows,seen)[:8]
    known={x["id"]:x for x in seen["rows"]}
    f=dict(early=[[20,6],[24,17],[25,18]],early_random=[[[20,10],[24,15],[25,16]],[[20,0],[24,31],[25,20]]],
           joint="012",features=[list(range(C.UNITS)),20,list(range(C.UNITS))])
    tc=temporal_arms(f)
    d=T.run_panel(m,rows,"pilot-runner",tc,known)
    checks=technical(d,tc)
    assert d["technical"].all(),checks
    short=dict(scope="early",arms={"sham":T.arm(),"e-block":T.arm(early=early_specs(f["early"]))})
    z=T.run_panel(m,rows,"early",short,known)
    assert z["technical"].all(),z["numeric_diff"]
    A.save("pilot-runner.json",dict(passed=True,n=len(rows),shapes={k:list(x.shape) for k,x in d.items()},
                    max_sham_error=float(d["numeric_diff"].max()),max_native_v_error=float(d["origin_error"].max()),direct_sentinels=checks))


def discover(m,p):
    assert not (C.OUT/"confirm-behavior.json").exists() and not (C.OUT/"freeze.json").exists()
    rows,b=observe(m,p,"discover"); assert b["passed"],b["cells"]
    use=C.O.resolved(rows,b); known={x["id"]:x for x in b["rows"]}
    rng=np.random.default_rng(C.SEED)
    keys=[]
    def stage(mode,design,key):
        keys.append(key)
        return stored(m,use,mode,design,known,key)
    scan=stage("screen-e",dict(scope="early",arms={"sham":T.arm()}),"discover-fact-field.npz")["scan"]
    means=scan[:,:,0].mean(0)
    pool=sorted([(l,int(vals.argmax()),float(vals.max())) for l,vals in enumerate(means)],key=lambda a:-a[2])[:6]
    short=dict(scope="early",arms={"sham":T.arm(),**{f"e-{l}-{h}":T.arm(early=early_specs([(l,h)])) for l,h,_ in pool}})
    d=stage("early",short,"discover-early-knockouts.npz")
    fit=n_ok(use,d,"early")
    sm=d["state"][fit,:,0]-d["state"][fit,:,1]
    loss=(sm[:,[0]]-sm[:,1:]).mean(0)
    picks=np.argsort(-loss,kind="stable")[:3]
    early=[[pool[j][0],pool[j][1]] for j in picks]
    e_rand=[[],[]]; e_pools={}
    for layer,h in early:
        v,poolx=matched_heads(scan[fit,layer,1],h,rng)
        e_pools[str(layer)]=poolx
        for ii,target in enumerate(v): e_rand[ii].append([layer,target])
    print("Early selected",early,"loss",loss[picks].tolist(),flush=True)
    cs=stage("scan-c",code_arms(),"discover-code-native.npz")
    ck=n_ok(use,cs,"code-readout")
    c_rand,c_pools=matched_heads(cs["source_match"][ck],18,rng)
    recv,recv_pool=matched_heads(cs["receiver_match"][ck],20,rng)
    groups=[]; rand_groups=[[],[]]; unit_dose={}
    for layer in (27,37):
        fit_packets=ck.reshape(-1,8).all(1)
        score,change,intact=(cs[f"_{layer}-{k}"][fit_packets].sum(0,dtype=np.float64)/int(ck.sum()) for k in ("score","change","intact"))
        chosen=np.argsort(-score,kind="stable")[:C.UNITS].tolist()
        controls=matched_units(change,intact,chosen,rng)
        groups.append(chosen)
        for j,x in enumerate(controls): rand_groups[j].append(x)
        unit_dose[str(layer)]=[[float(change[z].sum()),float(intact[z].sum())] for z in [chosen,*controls]]
    f=dict(early=early,early_random=e_rand,early_match=e_pools,early_families=int(fit.sum())//8,
        early_screen=[dict(layer=l,head=h,linear_score=s,exact_loss=float(loss[j])) for j,(l,h,s) in enumerate(pool)],
        features=[groups[0],20,groups[1]],c_random=c_rand,c_match=c_pools,head_controls=recv,receiver_match=recv_pool,
        random=[[rand_groups[j][0],recv[j],rand_groups[j][1]] for j in range(2)],unit_dose=unit_dose)
    cfg=code_arms(f)
    c=stage("code",cfg,"discover-code-factorial.npz"); keep=n_ok(use,c,"factorial")
    mm=signed(c,use)[keep]
    scores={k:float((mm[:,i]-mm[:,1]).mean()) for i,k in enumerate(cfg["arms"]) if k.startswith(("coarse-","fine-"))}
    f["joint"]=max((k for k in SETS if len(k)>1),key=lambda z:scores["fine-"+z])
    f["singleton"]=max((k for k in SETS if len(k)==1),key=lambda z:scores["fine-"+z])
    f.update(plan=B.sha("plan.json"),behavior=B.sha("discover-behavior.json"),files={x:B.sha(x) for x in keys},
        time=C.O.stamp(),scores=scores,code_names=list(code_arms(f)["arms"]),temporal_names=list(temporal_arms(f)["arms"]),n=len(use)//8)
    A.save("freeze.json",f)
    print("Frozen",f["early"],"fine joint",f["joint"],"singleton",f["singleton"],flush=True)


def confirm(m,p):
    f=A.load("freeze.json"); assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-behavior.json")
    assert all(B.sha(k)==v for k,v in f["files"].items())
    rows,b=observe(m,p,"confirm")
    if not b["passed"]:
        A.save("confirm.json",dict(state="behavior-gate-failed",behavior=B.sha("confirm-behavior.json")))
        return
    use=C.O.resolved(rows,b); known={x["id"]:x for x in b["rows"]}
    manifest={}; meta={}; keymaps={}
    for mode,cfg in (("code",code_arms(f)),("temporal",temporal_arms(f))):
        assert list(cfg["arms"])==f[f"{mode}_names"]
        manifest[mode]={}; meta[mode]={}; keymaps[mode]=[]
        sz=8*(C.SLAB if mode=="temporal" else C.N.SLAB)
        total=math.ceil(len(use)/sz); clock=time.monotonic()
        C.N.progress(f"{mode}-shards",0,total,clock)
        for j,group in enumerate(batched(use,sz),1):
            dst=f"{mode}-{j:03d}.npz"; status=f"{mode}-{j:03d}-acceptance.json"
            if not (C.OUT/dst).exists() or not (C.OUT/status).exists():
                z=T.run_panel(m,list(group),mode,cfg,known)
                ck=technical(z,cfg)
                B.write_np(dst,z)
                A.save(status,dict(sentinels=ck,invalid_ids=z["ids"][~z["technical"]].tolist(),freeze=B.sha("freeze.json")))
            else:
                z=read_data(dst)
            assert list(z["ids"])==[r["id"] for r in group]
            manifest[mode][dst]=B.sha(dst); meta[mode][status]=B.sha(status)
            if mode=="temporal":
                jj=f["temporal_names"].index("e-block")
                keymaps[mode].extend((r,int(tok)) for r,tok in zip(group,z["state_top"][:,jj],strict=True) if int(tok)!=r["state_ids"][0])
            C.N.progress(f"{mode}-shards",j,total,clock)
    proposals=[]; others=[]
    for row,token in keymaps["temporal"]:
        slots=row["token"]["slots"]
        possibilities=[row["state_ids"][1],*[row["token"]["ids"][i] for i in slots[1:]]]
        if token in possibilities and len(proposals)<32:
            seq=row["token"]["ids"].copy(); seq[slots[0]]=token
            proposals.append(dict(row,token=dict(row["token"],ids=seq),observed_state=token))
        else:
            others.append(dict(id=row["id"],state=token,recognized=token in possibilities))
    raw=[]
    if proposals:
        cfg=dict(scope="temporal",alternate=True,arms={"sham":T.arm(),"e-block":T.arm(early=early_specs(f["early"]))})
        v=T.run_panel(m,proposals,"temporal",cfg,known)
        B.write_np("observed-state-replay.npz",v)
        for j,r in enumerate(proposals):
            raw.append(dict(id=r["id"],state=r["observed_state"],state_token=v["state_top"][j,1].item(),
                verified=bool(v["prefix_valid"][j,1]),position_of_first_deviation=int(v["first_mismatch"][j,1]),
                code=int(v["top"][j,1]) if v["prefix_valid"][j,1] else None))
    A.save("observed-state-replay.json",dict(attempted=raw,unresolved=others,source="Natural output is scored only with the entire affected prefix verified."))
    A.save("confirm.json",dict(state="finished",n=len(use)//8,behavior=B.sha("confirm-behavior.json"),freeze=B.sha("freeze.json"),panels=manifest,acceptance=meta,
                               replay=B.sha("observed-state-replay.json")))
