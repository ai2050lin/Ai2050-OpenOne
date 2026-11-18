"""Audit effective writes and estimate the source contrast from frozen family pairs."""

from collections import Counter, defaultdict
from pathlib import Path
import argparse

import numpy as np
from scipy.stats import spearmanr

import kv_alpha_run as C
from kv_alpha_native import arms, GROUPS, HEADS
from kv_route_audit import audit_behavior
from own_state_audit import round_bf16


R=C.R
SLOTS=("state","note")
MODES=("swap","null")
DOSES=("low","high")


def norm(v):
    return np.linalg.norm(v.astype(np.float64),axis=-1)


def dose_report(cfg,rows,good,obj,archive):
    baseline=archive["sham_alpha"]
    assert baseline.shape==(len(rows),3,2) and np.isfinite(baseline).all()
    assert np.all((baseline>=0)&(baseline<=1)) and (baseline.sum(-1)<=1.000001).all()
    d=archive["donor"]-archive["state"]
    assert d.shape==(len(rows),3,128) and np.isfinite(d).all()
    roll=np.stack([np.roll(d[:,i],(13+7*l)%127+1,axis=-1) for i,(l,_) in enumerate(HEADS)],axis=1)
    assert all(archive[k].shape==d.shape for k in ("state","note","donor"))
    target_n={}
    write_records={}
    checks={}
    for a in arms():
        group,regime,slot,mode=a.name.split("-")
        heads=[HEADS.index(h) for h in a.group]
        si=SLOTS.index(slot)
        v=(d if mode=="swap" else roll)[:,heads]
        stats=np.array(obj["stats"][a.name],dtype=np.float32)
        scored=obj["paths"][a.name]
        assert obj["exact_all_vocab"]
        assert np.array_equal(scored["read_count"],good*len(heads))
        assert np.count_nonzero(scored["restore_count"])==0
        assert np.count_nonzero(np.asarray(scored["scores"])[~good])==0
        ids=np.array([h for h in range(3) if h not in heads],dtype=np.int64)
        w=archive[a.name]
        assert w.shape==d.shape and np.isfinite(w).all()
        assert not np.any(w[:,ids]) and not np.any(stats[:,ids])
        assert not np.any(w[~good]) and not np.any(stats[~good])
        assert np.isfinite(stats).all()
        live=stats[:,heads,si]
        live_target=stats[:,heads,2]
        scale=stats[:,heads,3]
        a0=baseline[:,heads]
        lo,hi=a0.min(-1),a0.max(-1)
        goal=(lo if regime=="low" else np.minimum(hi,2*lo)) if regime in DOSES else live
        assert np.allclose(live_target[good],goal[good],atol=2e-7)
        ideal_scale=goal/np.maximum(live,1e-12) if regime in DOSES else np.ones_like(live)
        assert np.allclose(scale[good],np.minimum(2.05,ideal_scale)[good],atol=2e-6)
        # With no earlier intervention, the Q/K used by both raw methods must equal sham.
        isolated=[j for j,i in enumerate(heads) if len(heads)==1 or i==2]
        assert np.allclose(stats[:,heads,:2][:,isolated],a0[:,isolated],atol=3e-7)
        w=w[:,heads].astype(np.float64)
        expected=v*goal[...,None]
        measured=norm(w)
        want=norm(expected)
        target_n[group,regime,mode]=want
        write_records[group,regime,mode,slot]=w
        if regime not in DOSES:
            # Raw paths' effective linear contribution and actual BF16 SDPA output difference.
            u=archive[slot][:,heads]
            projected=(round_bf16(u+scale[...,None]*v)-u)*live[...,None] if regime=="raw" else expected
            error=norm(w-projected)
            checks[a.name]=dict(error=error,measured=measured,expected=want)
            continue
        u=archive[slot][:,heads]
        fp_linear=(round_bf16(u+scale[...,None]*v)-u)*live[...,None]
        planned=norm(fp_linear-expected)
        error=norm(w-expected)
        cosine=np.sum(w*expected,axis=-1)/np.maximum(measured*want,1e-12)
        checks[a.name]=dict(
            small=(want<cfg["norm_floor"]).any(1),
            capped=(ideal_scale>2.05+2e-6).any(1),
            analytic=(planned>np.maximum(cfg["analytic_absolute_error_floor"],cfg["max_planned_relative_error"]*want)).any(1),
            native=(error>np.maximum(cfg["actual_absolute_error_floor"],cfg["max_actual_relative_error"]*want)).any(1),
            collapsed=(measured<cfg["min_actual_norm_fraction"]*want).any(1),
            direction=(cosine<cfg["min_actual_cosine"]).any(1),
            measured=measured,
            expected=want,
            native_error=error,
            cosine=cosine)
    table=[]
    output={}
    for group in GROUPS:
        output[group]={}
        for regime in DOSES:
            a=[checks[f"{group}-{regime}-{slot}-{mode}"] for slot in SLOTS for mode in MODES]
            labels=("small","capped","analytic","native","collapsed","direction")
            fail={why:np.logical_or.reduce([arm[why] for arm in a]) for why in labels}
            pairs=[]
            for mode in MODES:
                left,right=[write_records[group,regime,mode,slot] for slot in SLOTS]
                goal=target_n[group,regime,mode]
                pairs.append(norm(left-right))
                bad=pairs[-1]>np.maximum(cfg["actual_absolute_error_floor"],cfg["max_actual_paired_relative_error"]*goal)
                fail[f"paired-{mode}"]=bad.any(1)
            admit=good & ~np.logical_or.reduce(list(fail.values()))
            output[group][regime]=dict(
                admit=admit,
                reasons=[[why for why in fail if fail[why][i]] if good[i] else ["behavior"] for i in range(len(rows))],
                actual_shared=np.min(np.stack([x["measured"] for x in a]),axis=(0,2)),
                target_shared=np.min(np.stack([x["expected"] for x in a]),axis=(0,2)),
                actual_gap=np.max(np.stack(pairs),axis=(0,2)),
                max_native_error=np.max(np.stack([x["native_error"] for x in a]),axis=(0,2)))
    for i,row in enumerate(rows):
        alt=int(row["alt_code"]=="Q")
        valid=bool(good[i])
        entry=dict(id=row["id"],family=row["family"],cell=row["voice"],valid=valid,
                   alpha=baseline[i].tolist(),margin={},doses={})
        assert obj["ids"][i]==row["id"]
        for group in GROUPS:
            entry["margin"][group]={}
            for regime in ("raw","low","high","write"):
                record={}
                for slot in SLOTS:
                    rr=[]
                    for mode in MODES:
                        sc=obj["paths"][f"{group}-{regime}-{slot}-{mode}"]["scores"][i]
                        rr.append(sc[alt]-sc[1-alt])
                    record[slot]=rr[0]-rr[1]
                entry["margin"][group][regime]=dict(**record,source=record["state"]-record["note"])
            entry["doses"][group]={
                regime:{k:(v[i] if k=="reasons" else bool(v[i]) if k=="admit" else float(v[i]))
                        for k,v in dct.items()}
                for regime,dct in output[group].items()}
        table.append(entry)
    native_close={
        a:dict(max=float(data["error"].max()),n=int(good.sum()),
               rmse=float(np.mean(data["error"][good]**2)**.5) if good.any() else None)
        for a,data in checks.items() if "error" in data}
    return table,native_close


def stats(pairs,cfg):
    v=np.asarray(pairs,dtype=np.float64)
    if not len(v):
        return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(cfg["analysis_seed"])
    sample=v[rng.integers(0,len(v),size=(cfg["family_bootstrap_draws"],len(v)))].mean(1)
    lo,hi=np.quantile(sample,[.025,.975])
    return dict(n=len(v),mean=float(v.mean()),ci95=[float(lo),float(hi)],
                median=float(np.median(v)),positive=int((v>0).sum()),negative=int((v<0).sum()))


def select(buckets,filter_by):
    return [v for v in buckets.values() if len(v)==4 and all(filter_by(r) for r in v)]


def mean(block,g,regime,readout):
    return float(np.mean([r["margin"][g][regime][readout] for r in block]))


def summary(rows,cfg):
    families=defaultdict(list)
    for r in rows:
        families[r["family"]].append(r)
    valid=select(families,lambda r:r["valid"])
    summaries={}
    for group in GROUPS:
        by_group={}
        raw={k:stats([mean(b,group,"raw",k) for b in valid],cfg) for k in (*SLOTS,"source")}
        predicted={k:stats([mean(b,group,"write",k) for b in valid],cfg) for k in (*SLOTS,"source")}
        pred=[r["margin"][group]["write"]["source"] for r in rows if r["valid"]]
        ref=[r["margin"][group]["raw"]["source"] for r in rows if r["valid"]]
        error=np.asarray(pred)-ref
        rho=float(spearmanr(pred,ref).statistic) if np.std(ref)>0 and np.std(pred)>0 else None
        bias=float(error.mean())
        rmse=float(np.linalg.norm(error)/np.sqrt(len(error)))
        predictive=dict(bias=bias,rmse=rmse,rho=rho,passed=bool(
            abs(bias)<=cfg["raw_predictive_bias_max"] and rmse<=cfg["raw_predictive_rmse_max"]
            and rho is not None and rho>=cfg["raw_prediction_rho_min"]),
            failures=sorted([dict(id=r["id"],gap=r["margin"][group]["write"]["source"]-r["margin"][group]["raw"]["source"])
                             for r in rows if r["valid"]],key=lambda x:-abs(x["gap"]))[:8])
        for regime in DOSES:
            admitted=select(families,lambda r:r["doses"][group][regime]["admit"])
            kept=[r for block in admitted for r in block]
            mat={k:stats([mean(b,group,regime,k) for b in admitted],cfg) for k in (*SLOTS,"source")}
            raw_same=stats([mean(b,group,"raw","source") for b in admitted],cfg)
            pred_same=stats([mean(b,group,"write","source") for b in admitted],cfg)
            sim_bias=stats([mean(b,group,"write","source")-mean(b,group,"raw","source") for b in admitted],cfg)
            removed=stats([mean(b,group,regime,"source")-mean(b,group,"raw","source") for b in admitted],cfg)
            ci=mat["source"]["ci95"]
            enough=len(admitted)>=cfg["minimum_matched_families"]
            equiv=enough and ci is not None and ci[0]>-cfg["equivalence"] and ci[1]<cfg["equivalence"]
            support=bool(enough and raw_same["mean"]<=-.25 and raw_same["ci95"][1]<0
                         and equiv and predictive["passed"])
            pop_result=None
            if "population_prediction" in cfg:
                pop_result=bool(enough and equiv and raw_same["mean"]<=-.25 and raw_same["ci95"][1]<0
                                and pred_same["ci95"][1]<0 and sim_bias["ci95"][0]>-cfg["equivalence"]
                                and sim_bias["ci95"][1]<cfg["equivalence"])
            by_group[regime]=dict(
                families=len(admitted),rows=len(kept),qualifying_rows=sum(r["doses"][group][regime]["admit"] for r in rows),
                sufficient_coverage=enough,equivalent=equiv,matched=mat,raw_on_matched=raw_same,
                prediction_on_matched=pred_same,prediction_bias_on_matched=sim_bias,population_explanatory_check=pop_result,
                shift_matched_minus_raw=removed,combined_explanatory_check=support,
                dose_medians={k:float(np.median([r["doses"][group][regime][k] for r in kept])) if kept else None
                              for k in ("actual_shared","target_shared","actual_gap","max_native_error")},
                reasons=dict(Counter(why for r in rows for why in r["doses"][group][regime]["reasons"])),
                retained_cells=dict(Counter(r["cell"] for r in kept)),
                excluded_examples=[r["id"] for r in rows if r["doses"][group][regime]["reasons"]][:8],
                counterexamples=[dict(id=r["id"],source=r["margin"][group][regime]["source"])
                                 for r in sorted(kept,key=lambda r:-abs(r["margin"][group][regime]["source"]))[:8]])
        raw_by_cell={}
        for cell in sorted({r["cell"] for r in rows}):
            members=[b for b in valid if b[0]["cell"]==cell]
            raw_by_cell[cell]=stats([mean(b,group,"raw","source") for b in members],cfg)
        heads=[HEADS.index(h) for h in GROUPS[group]]
        matrix=np.array([r["alpha"] for r in rows if r["valid"]])
        raw_examples=sorted([dict(id=r["id"],state_note=r["margin"][group]["raw"]["source"],alpha=r["alpha"])
                             for r in rows if r["valid"]],key=lambda x:x["state_note"])
        summaries[group]=dict(raw=raw,predicted=predicted,prediction_check=predictive,
                              doses=by_group,raw_strata=raw_by_cell,raw_note_examples=raw_examples[:8],
                              raw_state_examples=raw_examples[-8:][::-1],
                              alpha_medians=np.median(matrix[:,heads],axis=0).tolist(),
                              note_alpha_gt_state=(matrix[:,heads,1]>matrix[:,heads,0]).sum(0).tolist())
    return summaries


def audit(out, frozen):
    C.configure(out)
    plan=C.validate_plan(frozen)
    exe=R.read("execution.json")
    assert exe["plan_sha"]==R.sha(frozen/"plan.json")
    if out==frozen:
        assert all(R.sha(out/name)==h for name,h in plan["pilots"].items())
    panels=[]
    for stage in exe["panels"]:
        key=stage["panel"]
        rows=R.Base.Previous.load(frozen/f"materials-{key}.json")
        name=f"behavior-{key}.json"
        assert R.sha(out/name)==stage["behavior_sha"]
        behavior=R.read(name)
        audit_behavior("q4",rows,behavior["rows"])
        import json
        assert json.loads(json.dumps(R.gate(rows,behavior["rows"])))==behavior["cells"]
        r=dict(panel=key,status=stage["status"],behavior=behavior["cells"])
        if stage["status"]=="complete":
            seq=[]
            numeric=[]
            jobs=C.batches(rows)
            obs=C.batches(behavior["rows"])
            for shard,batch,b_obs in zip(stage["shards"],jobs,obs,strict=True):
                assert R.sha(out/shard["json"])==shard["sha"]
                assert R.sha(out/shard["npz"])==shard["npz_sha"]
                doc=R.read(shard["json"])
                assert doc["write_sha"]==shard["npz_sha"] and doc["write_file"]==shard["npz"]
                assert all(R.sha(out/f)==hashval for f,hashval in doc["fields"].items())
                good=np.array([o["joint"] for o in b_obs])
                with np.load(out/shard["npz"]) as a:
                    records,checks=dose_report(plan["hypothesis"],batch,good,doc,a)
                seq.extend(records)
                numeric.append(checks)
            assert len(seq)==len(rows)
            r["group"]=summary(seq,plan["hypothesis"])
            r["reconstructed_native_checks"]={a:dict(max=max(z[a]["max"] for z in numeric),
                rmse=float(np.sqrt(np.average([z[a]["rmse"]**2 for z in numeric],weights=[z[a]["n"] for z in numeric]))))
                for a in numeric[0]}
            name=f"audit-rows-{key}.json"
            R.write(name,dict(plan_sha=exe["plan_sha"],provenance=plan["provenance"],rows=seq))
            r["row_sha"]=R.sha(out/name)
        panels.append(r)
        print("CPU audit",key,stage["status"],len(rows),flush=True)
    R.write("audit.json",dict(plan_sha=exe["plan_sha"],node=R.read("node.json"),panels=panels,
                             analysis_sha=R.sha(Path(__file__)),provenance=plan["provenance"]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=C.OUT)
    p.add_argument("--frozen",type=Path,default=C.OUT)
    args=p.parse_args()
    audit(args.out.resolve(),args.frozen.resolve())
