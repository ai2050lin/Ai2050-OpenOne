"""Freeze case-level Q/K predictions, then perform native query/key and rescue edits."""

from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
import sys
import time

import numpy as np

import kv_address_run as C
import kv_address_native as N
import kv_address_study as S
from own_state_audit import round_bf16


def norm(x):
    return np.linalg.norm(x.astype(np.float64),axis=-1)


def gaps(x):
    return ((x[...,1,:]-x[...,2,:])*x[...,0,:]).sum(-1)*N.SCALE


def predicted(a,b,arm,donor):
    idx=arm.head
    vec=a["qkv"][:,idx].copy()
    pos=a["pos"]
    freq=a["freq"]
    if arm.mode=="q":
        vec[:,0]=round_bf16(S.rotate(b[donor,idx,0],pos[:,0]-pos[donor,0],freq))
    elif arm.mode=="k":
        vec[:,1:]=round_bf16(S.rotate(b[donor,idx,1:],pos[:,1:]-pos[donor,1:],freq))
    else:
        vec[:,1:]=round_bf16(S.rotate(vec[:,1:],pos[:,[2,1]]-pos[:,1:],freq))
    return gaps(vec)


def freeze():
    S.verify()
    assert not (C.OUT/"case-forecasts.json").exists()
    assert not list(C.OUT.glob("edited-*.json"))
    rows,a=S.captured("confirm")
    f=C.load("fit.json")
    assert C.digest(C.OUT/"capture-discover.npz")==f["discovery"] and C.digest(C.OUT/"fit-centroids.npz")==f["centroids"]
    with np.load(C.OUT/"fit-centroids.npz") as d:
        sub=np.stack([d["qk"][r["order"],r["context"],r["codebook"]] for r in rows])
    relative=a["pos"]-a["pos"][:,0,None]
    proto=S.rotate(sub,relative[:,None,:],a["freq"])
    forecast={"prototype":gaps(proto)}
    for name,s in f["baselines"].items():
        forecast[name]=np.array([s["means"]["".join(str(r[k]) for k in s["fields"])] for r in rows])
    good=a["good"]
    assert np.max(np.abs(gaps(a["qkv"])[good]-a["gap"][good]))<.0001
    by={r["id"]:i for i,r in enumerate(rows)}
    family_good={}
    for r,okay in zip(rows,good,strict=True):
        family_good[r["family"]]=family_good.get(r["family"],True) and bool(okay)
    primary=np.asarray([family_good[r["family"]] for r in rows])
    forecast["good_family"]=primary
    for arm in N.intervention_arms():
        dst=np.array([by[C.M.flip_id(r,arm.factor)] for r in rows]) if arm.factor else np.arange(len(rows))
        forecast[arm.name]=predicted(a,a["qkv"],arm,dst)
        forecast[arm.name+"-donor"]=dst
    np.savez_compressed(C.R.Base.Previous.check_write(C.OUT/"case-forecasts.npz"),**forecast)
    C.save("case-forecasts.json",dict(
        time=S.mark(),sha=C.digest(C.OUT/"case-forecasts.npz"),fit=C.digest(C.OUT/"fit.json"),
        diagnostics=C.digest(C.OUT/"capture-confirm.npz"),n=int(primary.sum()),
        sources={str(q.relative_to(C.Base.ROOT)):C.digest(q) for q in [Path(__file__)]},
        predeclared_thresholds=C.load("study-plan.json")["directed"],
        repair_contract="High-dose original vs L21H25 balance. Q/K balance was specified in study-plan. Balanced dose reuses its V-free baseline. Phase2940 gates exactly applied to each head and both directions.",
        identity=dict(factor_donors="Same family exactly one factor changed. Invalid families excluded before edits.",
                      rope="Phase swap at fixed target source slots. No token, role or V edit.")))


def compact_dose(data,cfg):
    target=data["target"].cpu().numpy()
    analytic=data["analytic"].cpu().numpy()
    actual=data["actual"].cpu().numpy()
    want=norm(target)
    ds=norm(actual)
    cos=np.sum(actual*target,-1)/np.maximum(ds*want,1e-12)
    return dict(n=want,w=ds,actual=actual,fail={
        "small":want<cfg["norm_floor"],
        "capped":data["gain"].cpu().numpy()>2.05+2e-6,
        "analytic":norm(analytic-target)>np.maximum(cfg["analytic_absolute_error_floor"],cfg["max_planned_relative_error"]*want),
        "native":norm(actual-target)>np.maximum(cfg["actual_absolute_error_floor"],cfg["max_actual_relative_error"]*want),
        "collapsed":ds<cfg["min_actual_norm_fraction"]*want,
        "direction":cos<cfg["min_actual_cosine"],
    })


def weighted_results(r,base,eligible):
    cfg_path=C.OUT.with_name("kv_alpha_20261115_v2")/"plan.json"
    cfg=C.R.Base.Previous.load(cfg_path)["hypothesis"]
    t=r.t
    r.initialize(donor=True)
    bal=r.replay(N.Arm("balance","balance"),usable=eligible)
    assert float(bal["balance_error"].max())<.02
    r.balance=bal
    results={}
    for a in N.value_arms():
        data=r.replay(a,usable=eligible)
        results[a.name]=dict(**compact_dose(data,cfg),alpha=data["alpha"].cpu().numpy(),
                             score=data["scores"].cpu().numpy(),balance_error=float(data["balance_error"].max()))
    store={}
    for mode in ("original","balanced"):
        dct={}
        for slot,signal in product(("s","n"),("donor","rotate")):
            dct[slot,signal]=results[f"{mode}-{slot}-{signal}"]
        cases={}
        for tag in next(iter(dct.values()))["fail"]:
            cases[tag]=np.logical_or.reduce([x["fail"][tag] for x in dct.values()])
        for signal in ("donor","rotate"):
            x,y=(dct[slot,signal] for slot in "sn")
            cases["paired-"+signal]=norm(x["actual"]-y["actual"])>np.maximum(
                cfg["actual_absolute_error_floor"],cfg["max_actual_paired_relative_error"]*np.maximum(x["n"],y["n"]))
        no_fail=~np.logical_or.reduce(list(cases.values()))
        sel=eligible.cpu().numpy()
        store[mode]=dict(
            admitted=(no_fail.all(1)&sel).tolist(),
            by_head=[[[tag for tag,mask in cases.items() if mask[j,i]] if sel[j] else ["ineligible"] for i in range(3)]
                     for j in range(len(sel))],
            target_min=np.min(np.stack([b["n"] for b in dct.values()]),axis=0).tolist(),
            actual_min=np.min(np.stack([b["w"] for b in dct.values()]),axis=0).tolist(),
            arms={f"{slot}-{signal}":dict(score=b["score"].tolist(),alpha=b["alpha"].tolist(),balance_error=b["balance_error"])
                  for (slot,signal),b in dct.items()})
    return dict(arms=store,baseline_balanced=bal["scores"].cpu().tolist(),baseline_alpha=bal["alpha"].cpu().tolist(),
                balance_error=float(bal["balance_error"].max()),original_scores=base["scores"].cpu().tolist())


def execute():
    import torch
    start=len(list(C.OUT.glob("edited-*.json")))
    p=S.verify()
    rows,a=S.captured("confirm")
    ob=C.load("behavior-confirm.json")["rows"]
    f=C.load("case-forecasts.json")
    assert C.digest(C.OUT/"case-forecasts.npz")==f["sha"]
    assert all(C.digest(C.Base.ROOT/k)==h for k,h in f["sources"].items())
    assert C.digest(C.OUT/"fit.json")==f["fit"]
    assert C.digest(C.OUT/"capture-confirm.npz")==f["diagnostics"]
    with np.load(C.OUT/"case-forecasts.npz") as archive:
        pred={k:archive[k] for k in archive.files}
    marker=C.digest(C.OUT/"case-forecasts.json")
    assert C.Base.collect_identity()==p["files"]

    def work(task):
        t0=time.monotonic()
        jobs=C.groups(rows)
        for i,(batch,behavior) in enumerate(zip(jobs,C.groups(ob),strict=True)):
            name=f"edited-{i:03d}.json"
            ix=np.arange(i*C.BATCH,i*C.BATCH+len(batch))
            if (C.OUT/name).exists():
                assert C.load(name)["frozen"]==marker
                continue
            torch.cuda.empty_cache()
            reader=N.Reader(task,batch,behavior)
            try:
                reader.initialize()
                eligible=torch.as_tensor(pred["good_family"][ix],device="cuda",dtype=torch.bool)
                sham=reader.replay(N.Arm("capture","capture"))
                assert np.array_equal(sham["gap"].cpu().numpy(),a["gap"][ix])
                assert np.array_equal(sham["scores"].cpu().numpy(),a["scores"][ix])
                assert np.array_equal(sham["qkv"].cpu().numpy(),a["qkv"][ix])
                reader.reference=sham
                records={}
                for arm in N.intervention_arms():
                    ix2=pred[arm.name+"-donor"][ix]
                    if arm.mode=="rope":
                        dst,pos=None,None
                    else:
                        dst=torch.as_tensor(a["qkv"][ix2],device="cuda")
                        pos=torch.as_tensor(a["pos"][ix2],device="cuda")
                    result=reader.replay(arm,dst,pos,eligible)
                    h=arm.head
                    records[arm.name]=dict(
                        gap=result["gap"][:,h].cpu().tolist(),alpha=result["alpha"][:,h].cpu().tolist(),
                        head_write=result["actual"][:,h].norm(dim=-1).cpu().tolist(),scores=result["scores"].cpu().tolist())
                paired=weighted_results(reader,sham,eligible)
                C.save(name,dict(frozen=marker,ids=[x["id"] for x in batch],qk=records,weighted=paired))
            finally:
                reader.close()
            C.R.Base.Previous.stage_log("q4","qk-and-dose-edits",i+1,len(jobs),t0,resumed=start)
        C.save("interventions.json",dict(
            frozen=marker,shards={f"edited-{i:03d}.json":C.digest(C.OUT/f"edited-{i:03d}.json") for i in range(len(jobs))},
            time=S.mark(),prefill_and_full_vocab_sham="matched during collection, Q/K and P/Q exactly matched again for each edit batch"))
    C.with_task(work)


if __name__=="__main__":
    C.configure()
    {"freeze":freeze,"run":execute}[sys.argv[1]]()
