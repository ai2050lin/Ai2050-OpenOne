"""Independent algebra, native source trace, and actual lexical coverage checks."""

from collections import Counter

import numpy as np

import kv_orth_run as C
from kv_orth_audit import material_checks
from kv_orth_study import load, resolved


A,B=C.A,C.B


def natural_recheck(p,raw,led):
    r=A.load(p["materials"]["confirm"]["file"])
    gate=A.load("confirm-gate.json");y={o["id"]:o for o in raw["rows"]}
    all_={}
    for d in (0,1):
        here=[x for x in r if x["distance"]==d]
        conflict=[x for x in here if x["state"]!=x["fact"]]
        c=Counter()
        for row in conflict:
            o=y[row["id"]]
            c["attempted"]+=1
            c["real_state_changed"]+=int(o["requested_state_emitted"] and o["raw_unpatched_state"]!=o["state"])
            c["nominal_state_emitted"]+=int(o["requested_state_emitted"])
            c["valid_code_prefix"]+=int(o["matched"])
        gaps=[x["token"]["gap"][0] for x in here]
        realized=[]
        for row in here:
            o=y[row["id"]]
            if o["matched"]:
                tok=o["resolved"] or row["token"]
                realized.append(len(tok["ids"])-1-tok["slots"][0])
        all_[str(d)]=dict(**c,nominal_gap=[min(gaps),max(gaps)],median_nominal_gap=float(np.median(gaps)),
                          actual_gap=[min(realized),max(realized)] if realized else None)
    use=resolved(r,raw["rows"],gate)
    if not use:
        return all_,None
    q=np.asarray([y[a["id"]]["pq"] for a in use],dtype=np.float64)
    dq=(q[:,1]-q[:,0]).reshape(-1,2,8)
    role=lambda k:np.asarray([1 if row[k]=="Q" else -1 for row in use]).reshape(-1,2,8)
    sign_f,sign_s=role("code_f"),role("code_s")
    f=(dq*sign_f).mean(-1);s=(dq*sign_s).mean(-1)
    diff=sign_f!=sign_s
    assert (diff.sum(-1)==4).all()
    col=lambda keep:(dq*sign_s*keep).sum(-1)/4
    assert np.array_equal(s-f,col(diff)) and np.array_equal(s+f,col(~diff))
    fact_map=[]; switch=Counter()
    for index in range(0,len(use),16):
        z=use[index:index+16]
        for j in range(8):
            near,far=z[j],z[j+8]
            yr=lambda t:"PQ"[t["token"]["letters"].index(y[t["id"]]["top"])]
            if near["fact"]!=near["state"] and yr(near)!=yr(far):
                switch["fact_to_state" if yr(far)==far["code_s"] else "state_to_fact"]+=1
                fact_map.append(dict(id=[near["id"],far["id"]],outputs=[yr(near),yr(far)],state_code=near["code_s"],
                                     fact_code=near["code_f"],state=near["words"][near["state"]],fact=near["words"][near["fact"]]))
    return all_,dict(complete=len(use)//16,conflict_signed=[float(x) for x in col(diff).mean(0)],
                     congruent_signed=[float(x) for x in col(~diff).mean(0)],distance_flip_count=len(fact_map),
                     directional_flips=dict(switch),examples=fact_map[:8])


def main():
    C.setup();p=C.verify();f=A.load("freeze.json");led=A.load("confirm.json");raw=A.load("confirm-natural.json")
    assert led["freeze"]==raw["binding"]==B.sha("freeze.json") and f["plan"]==B.sha("plan.json")
    assert all(B.sha(k)==v for k,v in led.get("shards",{}).items())
    assert all(B.sha(k)==v for k,v in f["discovery"].items())
    sources=["material-pilot.json",p["materials"]["discover"]["file"],p["materials"]["confirm"]["file"]]
    identity=[material_checks(A.load(name)) for name in sources]
    assert sum(map(len,identity))==len(set().union(*identity))
    natural,factor=natural_recheck(p,raw,led)
    re={}; conflict={}
    accepted=set(led.get("accepted",[]))
    material={x["id"]:x for x in A.load(p["materials"]["confirm"]["file"])}
    miscodes=[]
    for ob in raw["rows"]:
        row=material[ob["id"]]
        if row["fact"]==row["state"] and ob["matched"] and ob["top"]!=row["token"]["letters"][row["code_f"]=="Q"]:
            miscodes.append(dict(id=row["id"],state=row["values"]["State"],code=row["code_f"],actual=ob["rendered"],raw_pq=ob["pq"]))
    arms=("sham","h18","fact","book","parallel","joint","restore-h18","restore-parallel","restore-all")
    for file in sorted(led.get("shards",{})):
        z=load(file)
        h=f["arm_names"].index("h18")
        alpha=z["alpha"][:,h,0].astype(np.float64)
        v=z["h18_source_v"].astype(np.float64)
        actual=z["h18_write_change"].astype(np.float64)
        predicted=-alpha[:,None]*v
        score=(z["h18_source_k"].astype(np.float64)*z["h18_live_q"].astype(np.float64)).sum(-1)/np.sqrt(v.shape[1])
        error=np.linalg.norm(actual-predicted,axis=1)/(1+np.linalg.norm(predicted,axis=1))
        assert np.isfinite(score).all() and error.max()<.05
        re[file]=dict(rows=len(v),max_source_formula_error=float(error.max()),median_h18_unmasked_score=float(np.median(score)),
                      median_h18_attention_mass=float(np.median(alpha)))
        for i,rid in enumerate(z["ids"]):
            row=material[rid]
            if rid not in accepted or row["fact"]==row["state"]:
                continue
            p_id,q_id=row["token"]["letters"]
            for arm in arms:
                top=int(z["top"][i,f["arm_names"].index(arm)])
                letter="P" if top==p_id else "Q" if top==q_id else None
                effect="state" if letter==row["code_s"] else "fact" if letter==row["code_f"] else "other"
                contrast=conflict.setdefault(arm,{}).setdefault(str(row["distance"]),Counter())
                contrast[effect]+=1
                contrast["n"]+=1
    A.save("native-recheck.json",dict(natural=natural,factor=factor,trace=re,material_words=[len(w) for w in identity],
                                      causal_conflict=conflict,agreed_miscodes=miscodes,
                                      binding={k:B.sha(k) for k in ("plan.json","freeze.json","confirm.json")}))
    print("Verified 16-cell design, donor coverage, Walsh identities, and",len(re),"native response shards",flush=True)


if __name__=="__main__":main()
