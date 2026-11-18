"""Check numerical reconstruction, rounding controls and common-prefix cross-size results."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

import kv_bf_study as S
from kv_address_audit import family_stats
from kv_relay_native import CONDITIONS
from own_state_audit import round_bf16


C=S.C
A=C.A
METHODS=("dynamic","native","once","late-round","late")


def prior():
    base=C.OUT.with_name("kv_relay_20261115_v1")
    ledger=A.R.Base.Previous.load(base/"edited.json")
    stat=Counter()
    examples=[]
    for f,h in sorted(ledger["shards"].items()):
        assert A.digest(base/f)==h
        with np.load(base/f) as z:
            select=z["good"]
            flags=z["trio-q-dose-flags"][:, :,1]
            failing=flags.any(1)
            anal=failing[:,2]&select
            direct=failing[:,5]&select
            stat.update(n=int(select.sum()),analytic=int(anal.sum()),direction=int(direct.sum()),
                        overlap=int((anal&direct).sum()),only_any=int((anal|direct).sum()))
            examples.extend(z["ids"][anal|direct].tolist())
    A.save("phase2942-reassessment.json",dict(counts=stat,examples=examples,prior_ledger=A.digest(base/"edited.json")))


def family_ids(d):
    return np.array([id.split("-")[0] for id in d["ids"]])


def mask_family(data,selected):
    return {k:v[np.isin(family_ids(data),list(selected))] for k,v in data.items()}


def classify(d,m,truth):
    families=defaultdict(list)
    for i,k in enumerate(family_ids(d)):
        families[k].append(i)
    assert all(len(x)==8 for x in families.values())
    good=d["good"]
    yes=S.view_pass(d,m)
    native=S.view_pass(d,"native")
    f=[x for x in families.values() if yes[x].all()]
    sg=np.array([1 if truth[id]["alt_code"]=="Q" else -1 for id in d["ids"]])
    logits=d[m+"-scores"]
    margin=(logits[...,1]-logits[...,0])*sg[:,None]
    position=(margin[:,1]-margin[:,0])-(margin[:,3]-margin[:,2])
    flags=d[m+"-flags"].any(1)&good[:,None,None]
    floor=flags[:,:,0].any(1)
    nloss=(native&~yes)
    return dict(
        views=int(yes.sum()),families=len(f),gain=int((~native&yes&good).sum()),loss=int(nloss.sum()),
        qualified_families=sum(bool(good[v].all()) for v in families.values()),
        flags=flags.sum(0).tolist(),
        dose_floor_blocked_families=sum(bool(good[v].all() and floor[v].any()) for v in families.values()),
        pair=(d[m+"-pair"].any(1)&good[:,None]).sum(0).tolist(),
        margin_effect=family_stats([position[x].mean() for x in f]),
        remaining=[str(x) for x in d["ids"][good&~yes][:12]],
        lost=[str(x) for x in d["ids"][nloss][:12]],
        absolute_score_drift_from_native=float(np.mean(abs(logits[good]-d["native-scores"][good]))))


def addressing(d,truth):
    c=d["good"]
    pairs=[]
    for context in (0,1):
        for order in (0,1):
            dose=c&np.array([truth[id]["context"]==context and truth[id]["order"]==order for id in d["ids"]])
            if not dose.any():
                continue
            weights=d["natural_alpha"][dose]
            odds=np.log(np.maximum(weights[...,0],1e-12)/np.maximum(weights[...,1],1e-12))
            pairs.append(dict(context=context,order=order,n=int(dose.sum()),mean_log_odds=odds.mean(0).tolist(),
                              state_preference=(odds>0).mean(0).tolist()))
    drift=np.log(np.maximum(d["dynamic-alpha"][c],1e-12))-np.log(np.maximum(d["native-alpha"][c],1e-12))
    return dict(preference=pairs,live_q_vs_restored_logalpha_rms=np.sqrt((drift**2).mean((0,1,3))).tolist())


def diagnose(d,layer,lim,truth):
    y=d["good"]
    native=S.view_pass(d,"native")
    once=S.view_pass(d,"once")
    late=S.view_pass(d,"late")
    rounded=S.view_pass(d,"late-round")
    targetVec=[]
    error=[]
    error1=[]
    drop=[]
    vco=d["value_donor"][:,1]-d["value_state"][:,1]
    for cell,(slot,signal) in enumerate(CONDITIONS):
        nxt=vco if signal else np.roll(vco,(13+7*layer)%127+1,axis=-1)
        origin=d[("value_state","value_note")[slot]][:,1]
        power=d["native-gain"][:,cell,1]
        live=d["native-alpha"][:,cell,1,slot]
        natural=d["natural_alpha"][:,1]
        r=np.minimum(natural.max(-1),2*natural.min(-1))
        ideal=nxt*r[:,None]
        step=np.minimum(power,2.05)[:,None]*nxt
        rounded_step=round_bf16(origin+step)-origin
        recovered=rounded_step*live[:,None]
        exp_unrounded=step*live[:,None]
        assert np.max(abs(recovered[y]-d["native-head25-analytic"][y,cell]))<3e-6
        assert np.max(abs(recovered[y]-d["late-round-head25-analytic"][y,cell]))<3e-6
        assert np.max(abs(exp_unrounded[y]-d["once-head25-analytic"][y,cell]))<3e-6
        assert np.max(abs(exp_unrounded[y]-d["late-head25-actual"][y,cell]))<3e-6
        assert np.allclose(np.linalg.norm(ideal.astype(np.float64),axis=-1)[y],d["native-want"][y,cell,1],rtol=2e-6)
        targetVec.append(ideal)
        error.append(np.linalg.norm((recovered-ideal).astype(np.float64),axis=-1))
        error1.append(np.linalg.norm((d["once-head25-actual"][:,cell]-exp_unrounded).astype(np.float64),axis=-1))
        drop.append(((rounded_step==0)&(step!=0)).sum(-1)/np.maximum(1,(step!=0).sum(-1)))
    targets=np.stack(targetVec,1)
    error=np.stack(error,1)
    error1=np.stack(error1,1)
    drop=np.stack(drop,1)
    flags=d["native-flags"][:,:,1]
    mask=y&flags.any((1,2))
    direction=y&flags[:,:,5].any(1)
    input_only=y&late&~rounded
    output_only=y&late&~once
    show=[]
    for i in sorted(np.flatnonzero(mask),key=lambda j:-error[j].max())[:16]:
        j=int(error[i].argmax())
        sin=float(np.dot(targets[i,j],d["native-head25-actual"][i,j])/
                  max(np.linalg.norm(targets[i,j])*np.linalg.norm(d["native-head25-actual"][i,j]),1e-12))
        show.append(dict(id=str(d["ids"][i]),text=truth[d["ids"][i]]["text"],cell=j,
                         cosine=sin,input_error=float(error[i,j]),output_error_after_once=float(error1[i,j]),
                         zero_delta_coordinate_rate=float(drop[i,j]),matched={k:bool(S.view_pass(d,k)[i]) for k in METHODS}))
    return dict(recompute="All raw selected Value perturbations, explicit BF16 reinsertions and FP32 delta controls recomputed.",
                native_target_failure_views=int(mask.sum()),direction_views=int(direction.sum()),
                rounding_reinsert_loses=int(input_only.sum()),head_output_cast_loses=int(output_only.sum()),
                native_direction_rescued_by_once=int((direction&once).sum()),
                native_to_once_rescued=int((y&~native&once).sum()),
                native_to_late_rescued=int((y&~native&late).sum()),
                median_zero_coordinate_rate=float(np.median(drop[y])),
                med_input_error=float(np.median(error[y])),med_output_error_after_once=float(np.median(error1[y])),
                examples=show)


def results():
    p=S.verify()
    prior()
    rows=A.load("material-confirm.json")
    truth={r["id"]:r for r in rows}
    logs={}
    data={}
    for key,layer in (("q4",21),("q14",None)):
        ledger=A.load(f"{key}-confirm.json")
        if ledger["status"]!="finished":
            logs[key]=dict(status=ledger["status"])
            continue
        d=S.load_data(ledger)
        data[key]=d
        freeze=A.load(f"freeze-{key}.json")
        if key=="q4":
            assert ledger["plan"]==S.sha("plan.json")
        else:
            assert ledger["freeze"]==S.sha("freeze-q14.json")
            layer=A.load("q14-selection.json")["receiver_layer"]
        m=freeze["primary"]
        out={k:classify(d,k,truth) for k in METHODS}
        stats=out[m]["margin_effect"]
        out[m]["threshold_153"]=bool(out[m]["families"]>=153) if key=="q4" else None
        out[m]["equivalence_025"]=bool(out[m]["families"]>=153 and stats["ci95"] and stats["ci95"][0]>-.25 and stats["ci95"][1]<.25) if key=="q4" else None
        diag=diagnose(d,layer,p["dose"]["limits"],truth)
        logs[key]=dict(status="finished",frozen_primary=m,arms=out,n=int(d["good"].sum()),rounding=diag,
                       addressing=addressing(d,truth),
                       baseline_state_note_median=np.median(d["natural_alpha"][d["good"]],axis=0).tolist())
    common={}
    if len(data)==2:
        fam=[]
        for d in data.values():
            g=defaultdict(list)
            for a,b in zip(family_ids(d),d["good"],strict=True):
                g[a].append(b)
            fam.append({a for a,b in g.items() if len(b)==8 and all(b)})
        overlap=set.intersection(*fam)
        assert overlap
        for key,d in data.items():
            valid=mask_family(d,overlap)
            common[key]={m:classify(valid,m,truth) for m in METHODS}
            common[key]["addressing"]=addressing(valid,truth)
        common["families"]=len(overlap)
    q14b=A.load("q14-behavior.json")
    subset={}
    if "q14" in data:
        by=defaultdict(list)
        for r in q14b["rows"]:
            by[r["id"].split("-")[0]].append(r["greedy_prefix_and_code"])
        exact={k for k,v in by.items() if len(v)==8 and all(v)}
        if exact:
            filtered=mask_family(data["q14"],exact)
            subset=dict(families=len(exact),arms={k:classify(filtered,k,truth) for k in METHODS})
    obs=A.load("q4-behavior-confirm.json")
    A.save("audit.json",dict(provenance=p["provenance"],plan=S.sha("plan.json"),q4_snapshot=p["model_q4"]["snapshot"],
        q14_snapshot=A.load("q14-plan.json")["model"]["snapshot"],full=logs,paired_common=common,
        q4_behavior=dict(n=len(obs["rows"]),strict=sum(b["joint"] for b in obs["rows"]),cells=obs["cells"]),
        q14_behavior=dict(n=len(q14b["rows"]),conditional=sum(b["conditional"] for b in q14b["rows"]),
                          exact_prefix_code=sum(b["greedy_prefix_and_code"] for b in q14b["rows"]),cells=q14b["cells"],
                          whole_greedy_compatible_families=subset),
        script_sha=A.digest(Path(__file__))))


if __name__=="__main__":
    C.setup()
    if sys.argv[1:]==["history"]:
        prior()
    else:
        results()
