"""Freeze addressing forecasts before collecting or intervening on confirmation."""

from datetime import datetime
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

import kv_address_run as C
import kv_address_native as N


SOURCES=tuple("tests/glm5/"+s for s in (
    "kv_address_run.py","kv_address_materials.py","kv_address_native.py","kv_address_study.py",
    "kv_alpha_run.py","kv_route_run.py","own_state_run.py","own_state_intervene.py","condition_chain_run.py",
    "condition_chain_materials.py","source_binding_resources.py"))+(
        "deploy/storage_policy.py","deploy/storage_policy.json","shared/dist_protocol/mechanism_trace.py","tools/power_analysis.py")


def mark():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def plan():
    assert not (C.OUT/"study-plan.json").exists()
    assert C.load("pilot-behavior.json")["passed"]
    tok=C.Base.tok()
    bindings={}
    wordsets={}
    for stage in ("discover","confirm"):
        rows=C.M.material(stage,tok)
        for row in rows:
            C.encoded(tok,row)
        wordsets[stage]={r[k] for r in rows for k in ("state","note","alt")}
        by={r["id"]:r for r in rows}
        for row in rows:
            mirror=by[C.M.flip_id(row,"identity")]
            assert (row["state"],row["note"])==(mirror["note"],mirror["state"])
            assert all(row[k]==mirror[k] for k in ("alt","code","reverse","adopt","map_reverse"))
        name=f"material-{stage}.json"
        C.save(name,rows)
        bindings[stage]=dict(file=name,sha=C.digest(C.OUT/name),families=C.M.SIZE[stage],n=len(rows))
    assert not wordsets["discover"]&wordsets["confirm"]
    model=C.Base.collect_identity()
    sid=C.Base.official_id(model)
    assert sid==C.R.Base.Previous.load(C.OUT.with_name("kv_alpha_20261115_v2")/"plan.json")["provenance"]["model_revision"]
    power=C.Base.check_phase(.085,192,p_d=.10,alpha=.025)
    assert power["pass"]
    C.save("study-plan.json",dict(
        run=C.RUN,time=mark(),materials=bindings,files=model,model=sid,batch=C.BATCH,cap=C.MAX_NEW,
        sources={f:C.digest(C.Base.ROOT/f) for f in SOURCES},
        runtime=C.Base.versions(),provenance=dict(**C.Base.git_identity(),model_revision=sid,
                                                 seed=C.M.SEED,model_path=str(C.Base.MODEL)),
        pilot_sha=C.digest(C.OUT/"pilot-behavior.json"),power=power,
        gate="All eight cells >=90 percent; score all failures separately; analyses use complete eight-view families.",
        contrast="g=log(alpha_State/alpha_Note); State preference g>0; main paired family effect signed and absolute, lex/context/order separate",
        prediction=dict(prototype="Freeze discovery unrotated Q, K_State, K_Note centroid by context/order/code, no target Q/K as inputs.",
                        anchors="Factor means and position-only baselines frozen on discovery.",
                        primary="Heldout sign agreement >=.70, at least .05 over position-only, same folds/families.",
                        split="No confirm behavior, Q/K or output before fit.json; freeze per-trial intervention predictions before edited forwards."),
        directed=dict(arms=[a.__dict__ for a in N.intervention_arms()],
                      acceptance="Predicted gap RMSE <=.03 and preserved current native Q/K/V aside from declared edit. Change in native output and code margin separately.",
                      interventions="Replay same prefix, transplant one query or pair of keys from same-family factor flip; align relative phase; RoPE-only phase exchange.",
                      repair="Replay Phase2940 high-dose trio V, balance only L21H25 K along q to equalize pair logits at fixed summed pair mass. Compare each balance against its own V-free baseline.",
                      cap=2.05,min_matched_families=153,equiv_bound=.25,
                      success="Among failing families report repair count, losses, dose and P/Q content effect. Full coverage gate separate."),
        budget="380 MiB Windows project disk reserve checked; retain QKV anchors and first four views with all layers for each split."))


def verify():
    p=C.load("study-plan.json")
    assert all(C.digest(C.Base.ROOT/f)==h for f,h in p["sources"].items())
    assert all(C.digest(C.OUT/z["file"])==z["sha"] for z in p["materials"].values())
    assert C.Base.versions()==p["runtime"]
    return p


def collect(stage):
    import torch
    p=verify()
    if stage=="confirm":
        f=C.load("fit.json")
        assert f["design"]==C.digest(C.OUT/"study-plan.json")
        assert C.digest(C.OUT/"fit-centroids.npz")==f["centroids"]
    rows=C.load(p["materials"][stage]["file"])
    assert not (C.OUT/f"capture-{stage}.json").exists()
    assert C.Base.collect_identity()==p["files"]

    def work(task):
        obs=C.behavioral(task,rows,stage+"-behavior")
        C.save(f"behavior-{stage}.json",obs)
        if not obs["passed"]:
            return
        arrays={}
        start=time.monotonic()
        groups=C.groups(rows)
        for i,(group,ob) in enumerate(zip(groups,C.groups(obs["rows"]),strict=True),1):
            torch.cuda.empty_cache()
            cap=N.capture(task,group,ob,sample=i==1)
            hidden=cap.pop("hidden",None)
            if hidden is not None:
                np.savez_compressed(C.R.Base.Previous.check_write(C.OUT/f"hidden-{stage}.npz"),bf16=hidden,ids=[r["id"] for r in group[:4]])
            freq=cap.pop("freq")
            for k,v in cap.items():
                arrays.setdefault(k,[]).append(v)
            C.R.Base.Previous.stage_log("q4",stage+"-capture",i,len(groups),start)
        keys={k:np.concatenate(v) for k,v in arrays.items()}
        keys["freq"]=freq
        file=f"capture-{stage}.npz"
        np.savez_compressed(C.R.Base.Previous.check_write(C.OUT/file),**keys)
        C.save(f"capture-{stage}.json",dict(sha=C.digest(C.OUT/file),ids=[r["id"] for r in rows],
                                          behavior_sha=C.digest(C.OUT/f"behavior-{stage}.json"),
                                          design=C.digest(C.OUT/"study-plan.json"),
                                          native_field_sha=C.digest(C.OUT/f"hidden-{stage}.npz")))
    C.with_task(work)


def rotate(x,delta,freq):
    phase=np.asarray(delta,dtype=np.float32)[...,None]*freq
    a,b=np.split(x,2,axis=-1)
    return np.concatenate((a*np.cos(phase)-b*np.sin(phase),b*np.cos(phase)+a*np.sin(phase)),axis=-1)


def strata(row,fields):
    return tuple(row[k] for k in fields)


def fit():
    p=verify()
    assert not (C.OUT/"behavior-confirm.json").exists() and not (C.OUT/"fit.json").exists()
    rows,a=captured("discover")
    good=a["good"]
    raw=rotate(a["qkv"],-a["pos"][:,None,:],a["freq"])
    prototypes=[]
    for order in range(2):
        for context in range(2):
            for code in range(2):
                chosen=np.array([strata(r,("order","context","codebook"))==(order,context,code) for r in rows])&good
                assert chosen.sum()>=16
                prototypes.append(raw[chosen].mean(0))
    baselines={}
    for name,fields in (("order",("order",)),("context",("context",)),("factorial",("context","order","identity","codebook"))):
        z={}
        for key in set(strata(r,fields) for r in rows):
            keep=np.array([strata(r,fields)==key for r in rows])&good
            z["".join(map(str,key))]=a["gap"][keep].mean(0).tolist()
        baselines[name]=dict(fields=fields,means=z)
    np.savez_compressed(C.R.Base.Previous.check_write(C.OUT/"fit-centroids.npz"),
                        qk=np.asarray(prototypes).reshape(2,2,2,3,3,128),freq=a["freq"])
    C.save("fit.json",dict(frozen=mark(),design=C.digest(C.OUT/"study-plan.json"),
                           discovery=C.digest(C.OUT/"capture-discover.npz"),
                           centroids=C.digest(C.OUT/"fit-centroids.npz"),baselines=baselines,
                           n_valid=int(good.sum()),target_inputs=["context","order","codebook",
                                                                  "relative field offsets from emitted format, without Q/K or score"]))


def captured(stage):
    entry=C.load(f"capture-{stage}.json")
    name=f"capture-{stage}.npz"
    assert C.digest(C.OUT/name)==entry["sha"]
    assert C.digest(C.OUT/f"behavior-{stage}.json")==entry["behavior_sha"]
    rows=C.load(f"material-{stage}.json")
    assert [r["id"] for r in rows]==entry["ids"]
    with np.load(C.OUT/name) as f:
        return rows,{k:f[k] for k in f.files}


commands={"plan":plan,"discover":lambda:collect("discover"),"fit":fit,"confirm":lambda:collect("confirm")}
