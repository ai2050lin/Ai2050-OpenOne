"""Independent arithmetic, frozen prediction scores, factorial effects and rescue ledger."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kv_address_run as C
import kv_address_study as S
import kv_address_native as N


def family_stats(v):
    x=np.asarray(v,dtype=np.float64)
    if not len(x):
        return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(294012)
    b=x[rng.integers(0,len(x),size=(6000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())


def scores(g,p,keep):
    d=(g-p)[keep]
    return dict(n=int(keep.sum()),sign=(np.sign(g[keep])==np.sign(p[keep])).mean(0).tolist(),
                rmse=np.sqrt((d*d).mean(0)).tolist(),bias=d.mean(0).tolist())


def margin(pair,rows):
    v=np.asarray(pair,dtype=np.float64)
    return (v[...,1]-v[...,0])*np.asarray([1 if r["alt_code"]=="Q" else -1 for r in rows])


def report():
    from transformers import GenerationConfig
    p=S.verify()
    rows,cap=S.captured("confirm")
    plan=C.load("case-forecasts.json")
    assert C.digest(C.OUT/"case-forecasts.npz")==plan["sha"]
    assert C.digest(C.OUT/"fit.json")==plan["fit"]
    assert all(C.digest(C.Base.ROOT/f)==v for f,v in plan["sources"].items())
    done=C.load("interventions.json")
    assert done["frozen"]==C.digest(C.OUT/"case-forecasts.json")
    tok=C.Base.tok()
    eos=GenerationConfig.from_pretrained(C.Base.MODEL,local_files_only=True).eos_token_id
    # Parsing and token anchors are independently rechecked on CPU, without activating a model.
    stub=object.__new__(C.Task)
    stub.tok=tok
    stub.eos=set(eos if isinstance(eos,list) else [eos])
    stub.letters={x:tok.encode(" "+x,add_special_tokens=False)[0] for x in "PQ"}
    for stage in ("discover","confirm"):
        material=C.load(f"material-{stage}.json")
        behavior=C.load(f"behavior-{stage}.json")
        assert behavior["passed"]
        for r,ob in zip(material,behavior["rows"],strict=True):
            ids,opts=C.encoded(tok,r)
            assert stub.parse(r,ids,opts,ob["tokens"])==ob
    with np.load(C.OUT/"case-forecasts.npz") as f:
        pred={x:f[x] for x in f.files}
    keep=pred["good_family"]
    g=cap["gap"]
    assert np.max(np.abs(np.log(cap["alpha"][keep,:,0]/cap["alpha"][keep,:,1])-g[keep]))<2e-6
    assert int(keep.sum())==plan["n"]
    by={r["id"]:j for j,r in enumerate(rows)}
    families=defaultdict(list)
    for j,r in enumerate(rows):
        if keep[j]:
            families[r["family"]].append(j)
    assert all(len(v)==8 for v in families.values())
    natural=[]
    for context in range(2):
        for order in range(2):
            d=keep&np.array([r["context"]==context and r["order"]==order for r in rows])
            natural.append(dict(context=context,order=order,n=int(d.sum()),mean_gap=g[d].mean(0).tolist(),
                                alpha_median=np.median(cap["alpha"][d],axis=0).tolist(),state_rate=(g[d]>0).mean(0).tolist()))
    forecasts={name:scores(g,pred[name],keep) for name in ("prototype","order","context","factorial")}
    forecasts["prototype_pass"]=[a>=.70 and a-b>=.05 for a,b in zip(forecasts["prototype"]["sign"],forecasts["order"]["sign"],strict=True)]
    factor={}
    for label in C.FACTORS:
        by_head=[]
        for h in range(3):
            block=[[g[by[C.M.flip_id(rows[i],label)],h]-g[i,h] for i in v if not rows[i][label]]
                   for v in families.values()]
            by_head.append(dict(signed=family_stats(np.mean(block,axis=1)),
                                absolute=family_stats(np.mean(np.abs(block),axis=1))))
        factor[label]=by_head

    qk=defaultdict(lambda:defaultdict(list))
    weighted=defaultdict(lambda:defaultdict(list))
    balance_error=[]
    balanced_alpha=[]
    altered_baseline=[]
    orig_baseline=[]
    ids_in_order=[]
    for f,hs in sorted(done["shards"].items()):
        assert C.digest(C.OUT/f)==hs
        obj=C.load(f)
        assert obj["frozen"]==done["frozen"]
        ids_in_order.extend(obj["ids"])
        for k,data in obj["qk"].items():
            for v,x in data.items():
                qk[k][v].extend(x)
        data=obj["weighted"]
        for k,d in data["arms"].items():
            for v in ("admitted","by_head","target_min","actual_min"):
                weighted[k][v].extend(d[v])
            for arm,b in d["arms"].items():
                weighted[k][arm].extend(b["score"])
        balance_error.append(data["balance_error"])
        balanced_alpha.extend(data["baseline_alpha"])
        altered_baseline.extend(data["baseline_balanced"])
        orig_baseline.extend(data["original_scores"])
    assert ids_in_order==[r["id"] for r in rows]
    actual={}
    raw_margin=margin(cap["scores"],rows)
    for arm in N.intervention_arms():
        h=arm.head
        rec=qk[arm.name]
        read=np.asarray(rec["gap"])
        alpha=np.asarray(rec["alpha"])
        assert np.max(np.abs(np.log(alpha[keep,0]/alpha[keep,1])-read[keep]))<3e-6
        error=read-pred[arm.name]
        delta=read-g[:,h]
        edited=margin(rec["scores"],rows)-raw_margin
        if arm.factor:
            donor=pred[arm.name+"-donor"]
            difference=g[donor,h]-g[:,h]
            mse0=(difference[keep]**2).mean()
            pred_residual=float(np.sqrt(np.mean((difference[keep]-delta[keep])**2)/mse0))
            toward=float((difference[keep]*delta[keep]>0).mean())
            direction=[v2 for v2 in families.values()]
            signed=family_stats([float(np.mean([delta[i] for i in v if not rows[i][arm.factor]])) for v in direction])
        else:
            toward=pred_residual=signed=None
        rms=float(np.sqrt((error[keep]**2).mean()))
        worst=np.flatnonzero(keep)[np.argsort(np.abs(error[keep]))[-5:][::-1]]
        actual[arm.name]=dict(
            frozen_prediction_rmse=rms,passed=rms<=.03,
            n=int(keep.sum()),write_norm_median=float(np.median(np.asarray(rec["head_write"])[keep])),
            gap_change_abs=float(np.mean(abs(delta[keep]))),code_margin_change_abs=float(np.mean(abs(edited[keep]))),
            gap_flip=int(np.sum((read[keep]>0)!=(g[keep,h]>0))),prediction_maxerr=float(abs(error[keep]).max()),
            donor_direction=toward,donor_relative_error=pred_residual,signed_factor_direction=signed,
            counterexamples=[dict(id=rows[i]["id"],pred=float(pred[arm.name][i]),actual=float(read[i])) for i in worst])

    flags={k:np.array(v["admitted"])&keep for k,v in weighted.items()}
    original,repaired=(flags[k] for k in ("original","balanced"))
    gain=~original&repaired&keep
    loss=original&~repaired&keep
    bdelta=margin(altered_baseline,rows)-margin(orig_baseline,rows)
    repaired_data={}
    for k,v in weighted.items():
        ok=flags[k]
        fam=[items for items in families.values() if ok[items].all()]
        content=[margin(v[f"{j}-donor"],rows)-margin(v[f"{j}-rotate"],rows) for j in "sn"]
        relative=content[0]-content[1]
        contrast=family_stats([relative[items].mean() for items in fam])
        rr=dict(views=int(ok.sum()),families=len(fam),source=contrast,coverage_pass=len(fam)>=153,
                peak_dose_by_head=np.max(np.asarray(v["actual_min"])[ok],axis=0).tolist() if ok.any() else None,
                median_dose_by_head=np.median(np.asarray(v["actual_min"])[ok],axis=0).tolist() if ok.any() else None,
                failures=[dict(Counter(t for i in np.flatnonzero(keep) for t in v["by_head"][i][h])) for h in range(3)])
        rr["equivalence"]=bool(rr["coverage_pass"] and contrast["ci95"][0]>-.25 and contrast["ci95"][1]<.25)
        repaired_data[k]=rr
    also=[v for v in families.values() if repaired[v].all() and not original[v].all()]
    drop=[v for v in families.values() if original[v].all() and not repaired[v].all()]
    bal=np.asarray(balanced_alpha)
    weights=dict(arms=repaired_data,
                 rescued_views=int(gain.sum()),lost_views=int(loss.sum()),
                 rescued_families=len(also),lost_families=len(drop),
                 balance_alpha_max_error=max(balance_error),balance_baseline_margin_abs=float(np.mean(abs(bdelta[keep]))),
                 h25_natural_alpha_median=np.median(cap["alpha"][keep,1],axis=0).tolist(),
                 h25_balanced_alpha_median=np.median(bal[keep,1],axis=0).tolist(),
                 score_and_failure_examples=dict(
                     rescued=[dict(id=rows[i]["id"],original=weighted["original"]["by_head"][i],
                                   balanced=weighted["balanced"]["by_head"][i]) for i in np.flatnonzero(gain)[:12]],
                     still_failed=[dict(id=rows[i]["id"],original=weighted["original"]["by_head"][i],
                                        balanced=weighted["balanced"]["by_head"][i]) for i in np.flatnonzero(keep&~repaired)[:12]],
                     losses=[dict(id=rows[i]["id"],original=weighted["original"]["by_head"][i],
                                  balanced=weighted["balanced"]["by_head"][i]) for i in np.flatnonzero(loss)[:12]]))
    worst_idx=np.argsort(np.max(abs(pred["prototype"]-g),axis=1))[::-1]
    examples=[dict(id=rows[i]["id"],observed=g[i].tolist(),pred=pred["prototype"][i].tolist(),
                   condition=[rows[i][k] for k in C.FACTORS]) for i in worst_idx if keep[i]][:12]
    C.save("audit.json",dict(model=p["model"],n_eligible=int(keep.sum()),families=len(families),raw_natural=natural,
                             factors=factor,natural_predictions=forecasts,interventions=actual,weighted=weights,
                             forecast_counterexamples=examples,design=C.digest(C.OUT/"study-plan.json"),
                             forecast_sha=done["frozen"],analysis_sha=C.digest(Path(__file__)),provenance=p["provenance"]))


if __name__=="__main__":
    C.configure()
    report()
