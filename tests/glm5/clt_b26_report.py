"""Audit frozen B26 artifacts and calculate prespecified family-level effects."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np

from clt_b26 import ARMS, CAP, GROUPS, OUT, SEED, D, effects, load_run, save


def paired(values):
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci975=np.quantile(means,[.0125,.9875]).tolist())


def audit():
    pre,seal,rows = load_run("confirm")
    prior = D.read(OUT/"discovery-materials.json")
    assert not ({n for r in prior for n in r["names"]} & {n for r in rows for n in r["names"]})
    reps = [r for r in rows if r["representative"]]
    assert len(reps)==len({r["family"] for r in reps})==128
    assert (OUT/"candidate.json").stat().st_mtime < (OUT/"confirm-natural.npz").stat().st_mtime
    info = {}
    for split, material in (("discovery",prior),("confirm",rows)):
        file = OUT/(split+"-natural.npz")
        rec = D.read(OUT/(split+"-natural.json"))
        assert D.sha(file)==rec["data_sha"]
        with np.load(file) as z:
            scores = rec["scores"]
            audit_scores(material,scores)
            h18 = z["h"]
            prefix = sum(np.array_equal(h18[i],h18[i+1]) for i in range(0,len(material),2))
            assert prefix==rec["prefix_equal"]
            cell = []
            for i in range(16):
                subset = scores[i::16]
                cell.append(dict(cell=i,n=len(subset),strict=sum(s["gold_margin"]>0 for s in subset),
                                 top1=sum(s["top1_correct"] for s in subset),
                                 simultaneous=sum(s["top1_correct"] and s["gold_margin"]>0 for s in subset)))
            truth = np.array([r["target_record"] for r in material])
            assert np.isclose((z["source"][:,:4].argmax(1)==truth).mean(),rec["route_acc"])
            assert float(z["source"][:,8].max())==rec["max_error"]
        info[split] = dict(cell=cell, gates=rec["gates"], route=rec["route_acc"], prefixes=prefix,
                           error=rec["max_error"], elapsed=rec["elapsed"])

    result = dict(provenance=pre["provenance"],candidate_sha=D.sha(OUT/"candidate.json"),natural=info,
                  power=pre["power"],
                  representative_archive_blocks=dict(Counter(r["b"] for r in reps)),
                  representative_title_tokens=dict(Counter(r["content"].split("Archive to use: ",1)[1].splitlines()[0] for r in reps)),
                  audit_scope="file hashes, sample identities, arithmetic, timings and same-input restoration")
    if not (OUT/"confirm-interventions.json").exists():
        result["status"]="Confirmation admission failed; intervention artifact absent."
        save("analysis.json",result)
        return
    intervention = D.read(OUT/"confirm-interventions.json")
    data = intervention["arms"]
    fragments = [D.read(f)["arms"] for f in sorted(OUT.glob("confirm-interventions-??.json"))]
    for a in ARMS:
        assert data[a]==sum([f[a] for f in fragments],[])
        audit_scores(reps,data[a])
    truth = np.array([r["target_record"] for r in reps])
    idx = np.arange(len(reps))
    arrays = {}
    for a,scores in data.items():
        route = np.array([s["route"] for s in scores])
        norm = np.array([s["write_norm"] for s in scores])
        arrays[a] = dict(route=route.argmax(1)==truth,within=route[idx,truth]>route[idx,truth^1],
                         other_line=route.argmax(1)==(truth^1),
                         attention_gap=route[idx,truth]-route[idx,truth^1],
                         log_mass_odds=np.log(np.maximum(route[idx,truth],1e-38))-
                                       np.log(np.maximum(route[idx,truth^1],1e-38)),
                         gold_logit_margin=np.array([s["gold_margin"] for s in scores]),
                         logits=np.array([s["scores"] for s in scores]),
                         answer=np.array([s["top1_correct"] for s in scores]),
                         sdpa=np.array([s["sdpa_error"] for s in scores]),
                         mass=route, write=norm, dose=np.array([s["dose"] for s in scores]))
    base, restore = arrays["base"], arrays["restore"]
    restoration = (np.array_equal(base["logits"],restore["logits"]) and
                   np.array_equal(base["mass"],restore["mass"]) and
                   np.array_equal(base["answer"],restore["answer"]))
    technical = restoration and all(float(x["sdpa"].max())<=CAP for x in arrays.values())
    diagnostics={}
    for name in (*GROUPS,"rope","value"):
        control=arrays[name+"_control"]
        target=arrays[name]
        dose_ratio=float(np.sqrt(np.mean(target["dose"]**2)/max(1e-20,np.mean(control["dose"]**2))))
        matched=.67<=dose_ratio<=1.5
        table = {}
        for a in (name,name+"_control"):
            x=arrays[a]
            table[a]=dict(route=float(x["route"].mean()),within=float(x["within"].mean()),
                          other_line=float(x["other_line"].mean()),answer=float(x["answer"].mean()),
                          gold_gap=float(x["attention_gap"].mean()),gold_write_norm=float(x["write"][idx,truth].mean()),
                          log_mass_odds=float(x["log_mass_odds"].mean()),
                          sdpa_error=float(x["sdpa"].max()),median_dose=float(np.median(x["dose"])),
                          margin_delta=paired(x["gold_logit_margin"]-base["gold_logit_margin"]))
        route_diff=paired(control["route"].astype(float)-target["route"])
        diagnostics[name]=dict(arms=table, dose_ratio=dose_ratio,matched=matched,
             route_drop_vs_control=route_diff,
             attention_gap_drop_vs_control=paired(control["attention_gap"]-target["attention_gap"]),
             log_mass_odds_drop_vs_control=paired(control["log_mass_odds"]-target["log_mass_odds"]),
             logit_margin_drop_vs_control=paired(control["gold_logit_margin"]-target["gold_logit_margin"]),
             primary_pass=bool(name in ("h_row","k_id") and technical and matched and
                               route_diff["mean"]>=.10 and route_diff["ci975"][0]>0))
    with np.load(OUT/"confirm-natural.npz") as z,np.load(OUT/"native-parameters.npz") as weights:
        confirmations={}
        for name,(signal,other,dose) in effects(rows,z,weights).items():
            picked,control=seal["indices"][name],seal["indices"][name+"_control"]
            confirmations[name]=dict(signal=float(signal[picked].sum()),other=float(other[picked].sum()),
                                     control_signal=float(signal[control].sum()),control_other=float(other[control].sum()))
    misses={}
    for name in ("h_row","k_id","rope","value"):
        misses[name]=[r["id"] for i,r in enumerate(reps) if base["route"][i] and not arrays[name]["route"][i]]
    result.update(status="Complete",base=dict(route=float(base["route"].mean()),answer=float(base["answer"].mean()),
                  mean_attention_gap=float(base["attention_gap"].mean()),
                  mean_log_mass_odds=float(base["log_mass_odds"].mean()),
                  mean_write_norm=float(base["write"][idx,truth].mean())),
                  technical=technical, restoration=restoration, diagnostics=diagnostics,
                  confirmation_effects=confirmations, counterexample_ids=misses, elapsed=intervention["elapsed"])
    save("analysis.json",result)
    for a,d in diagnostics.items():
        print(a, d["route_drop_vs_control"], d["logit_margin_drop_vs_control"],"dose",d["dose_ratio"],flush=True)


def audit_scores(rows,scores):
    assert [r["id"] for r in rows]==[s["id"] for s in scores]
    for r,s in zip(rows,scores):
        gold=2*r["b"]+r["r"][r["b"]]
        assert gold==r["target_record"]
        assert s["top1_correct"]==(s["top1"]==r["gold_id"])
        assert s["gold_margin"]==s["scores"][gold]-max(v for i,v in enumerate(s["scores"]) if i!=gold)


if __name__=="__main__":
    audit()
