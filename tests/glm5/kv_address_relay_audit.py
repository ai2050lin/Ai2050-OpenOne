"""Trace three-head scale overflow to the live L21 coefficients."""

from collections import Counter
from pathlib import Path

import numpy as np

import kv_address_run as C
import kv_address_study as S


def run():
    rows,cap=S.captured("confirm")
    done=C.load("interventions.json")
    observed=[]
    nats=[]
    natural_flags=[]
    original_flags=[]
    for j,(name,hs) in enumerate(sorted(done["shards"].items())):
        assert C.digest(C.OUT/name)==hs
        b=C.load(name)["weighted"]
        ix=slice(j*C.BATCH,j*C.BATCH+len(b["original_scores"]))
        a=cap["alpha"][ix]
        target=np.minimum(a.max(-1),2*a.min(-1))
        branch=b["arms"]["original"]
        nats.append(a)
        gain_live=[]
        gain_natural=[]
        for slot in "sn":
            for col in ("donor","rotate"):
                p=np.asarray(branch["arms"][f"{slot}-{col}"]["alpha"],dtype=np.float64)
                idx="sn".index(slot)
                gain_live.append(target/np.maximum(p[...,idx],1e-12))
                gain_natural.append(target/np.maximum(a[...,idx],1e-12))
        observed.append(np.max(gain_live,axis=0))
        natural_flags.append(np.max(gain_natural,axis=0))
        original_flags.extend([[("capped" in s) for s in r] for r in branch["by_head"]])
    real=np.concatenate(observed)
    no_prior=np.concatenate(natural_flags)
    a0=np.concatenate(nats)
    with np.load(C.OUT/"case-forecasts.npz") as file:
        eligible=file["good_family"]
    cap_live=real>2.05+2e-6
    cap_baseline=no_prior>2.05+2e-6
    assert np.array_equal(cap_live[eligible],np.asarray(original_flags)[eligible])
    assert not cap_baseline[eligible].any()
    counts={}
    for c in range(2):
        for o in range(2):
            mask=eligible&np.array([r["context"]==c and r["order"]==o for r in rows])
            counts[f"{c}{o}"]=dict(n=int(mask.sum()),overflow=cap_live[mask].sum(0).tolist())
    # Each positive record confirms the threshold using the existing alpha arrays,
    # while the natural-anchor calculation uses the pre-edit numerator and denominator.
    examples=[
        [dict(id=rows[i]["id"],live_required=float(real[i,h]),natural_required=float(no_prior[i,h]),alpha=a0[i,h].tolist())
         for i in np.argsort(real[:,h])[::-1] if eligible[i] and cap_live[i,h]][:8] for h in range(3)]
    C.save("relay-gain-audit.json",dict(
        n=int(eligible.sum()),head_overflow_live=cap_live[eligible].sum(0).tolist(),
        head_overflow_anchored=cap_baseline[eligible].sum(0).tolist(),by_context_order=counts,examples=examples,
        interpretation="In these final-query fixed-prefix edits, L20H5 Value is the only earlier selected intervention before L21 Q/K. Frozen-target cap flags recompute exactly from live alpha, and none occur with the natural alpha denominator. This identifies sensitivity to the early write but does not localize intermediate weights.",
        source=C.digest(Path(__file__)),interventions=C.digest(C.OUT/"interventions.json")))


if __name__=="__main__":
    C.configure()
    run()
