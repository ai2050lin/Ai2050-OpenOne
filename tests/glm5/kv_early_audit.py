"""Recompute paired held-out effects, relay physics, and actual continuations."""

import numpy as np

import kv_early_run as C
from kv_origin_audit import summarize,fam,leave_lexeme
from kv_temporal_study import signed


A,B=C.A,C.B


def native_role_pairs(rows,obs):
    known={x["id"]:x for x in obs}
    ids={x["id"]:x for x in rows}
    examples=[]; complete=0; true_flip=0
    for r in rows:
        if r["side"]:
            continue
        other=r["family"]+"-1"+str(r["distance"])+str(r["order"])
        s=ids[other]
        u,v=known[r["id"]],known[other]
        if not u["code_correct"] or not v["code_correct"]: continue
        complete+=1
        state0=u["generated_tokens"][r["token"]["slots"][0]-r["token"]["start"]]
        state1=v["generated_tokens"][s["token"]["slots"][0]-s["token"]["start"]]
        if state0==r["state_ids"][0] and state1==s["state_ids"][0] and state0!=state1 and u["top_token"]!=v["top_token"]:
            true_flip+=1
            if len(examples)<8:
                examples.append(dict(ids=[r["id"],other],states=[r["values"]["State"],s["values"]["State"]],
                                     codes=[r["code"],s["code"]],actual=[u["rendered"],v["rendered"]]))
    return dict(attempted=len(rows)//2,both_correct=complete,natural_state_and_code_flips=true_flip,examples=examples)


def split_effects(p,sets,allrows):
    report={}
    for key,x in sets.items():
        z=summarize(fam(x),p["gate"]["confidence"])
        report[key]=dict(**z,passed=bool(z["n"]>=p["gate"]["n"] and z["ci"][0]>0 and z["mean"]>=p["gate"]["margin"]),
                         leave_one_lexeme=leave_lexeme(fam(x),allrows))
    return report


def main():
    p=C.verify(); f=A.load("freeze.json"); ledger=A.load("confirm.json"); behavior=A.load("confirm-behavior.json")
    assert ledger["freeze"]==B.sha("freeze.json")==behavior["binding"] and f["plan"]==B.sha("plan.json")
    assert (C.OUT/"freeze.json").stat().st_mtime_ns<(C.OUT/"confirm-behavior.json").stat().st_mtime_ns
    assert all(B.sha(n)==h for n,h in f["files"].items()) and ledger["behavior"]==B.sha("confirm-behavior.json")
    rr=A.load("material-confirm.json"); known={x["id"]:x for x in behavior["rows"]}
    natural=native_role_pairs(C.O.resolved(rr,dict(rows=behavior["rows"],eligible=list({r["family"] for r in rr}))),behavior["rows"])
    bad=[dict(row=r,result=known[r["id"]]) for r in rr if not known[r["id"]]["code_correct"]]
    if ledger["state"]!="finished":
        A.save("audit.json",dict(state=ledger["state"],natural=natural,failures=bad,cells=behavior["cells"]))
        return
    assert ledger["rollout"]==B.sha("rollout.json")
    use=C.O.resolved(rr,behavior)
    d={k:B.load_data(dict(shards=ledger["panels"][k])) for k in ("code","temporal")}
    assert all(B.sha(n)==h for panel in ledger["acceptance"].values() for n,h in panel.items())
    assert all(list(dd["ids"])==[r["id"] for r in use] for dd in d.values())
    accepted=np.repeat((d["code"]["technical"]&d["temporal"]["technical"]).reshape(-1,8).all(-1),8)
    case_tech={k:[dict(id=r["id"],raw_diff=float(h["numeric_diff"][i]),code=int(h["top"][i,0]),observed=known[r["id"]]["top_token"],
                       first_mismatch=int(h["first_mismatch"][i,0]) if k=="temporal" else None,
                       actual_token=int(h["first_generated"][i,0]) if k=="temporal" else None)
                  for i,r in enumerate(use) if not h["technical"][i]] for k,h in d.items()}
    if not accepted.any():
        A.save("audit.json",dict(state="technical-gate-failed",natural=natural,failures=bad,technical=case_tech,cells=behavior["cells"]))
        return
    rows=[r for r,ok in zip(use,accepted,strict=True) if ok]
    c,t=({k:v[accepted] for k,v in d[x].items()} for x in ("code","temporal"))
    ci={k:i for i,k in enumerate(f["code_names"])}; ti={k:i for i,k in enumerate(f["temporal_names"])}
    cm,tm=signed(c,rows),signed(t,rows)
    sm=t["state"][:,:,0].astype(np.float64)-t["state"][:,:,1]
    x=lambda k:cm[:,ci[k]]
    y=lambda k:tm[:,ti[k]]
    s=lambda k:sm[:,ti[k]]
    w=lambda k:t["kv_ratios"][:,ti[k]].mean(-1).astype(np.float64)
    avg=lambda func,k:(func(k+"1")+func(k+"2"))/2
    sets={"e-state":avg(s,"e-random")-s("e-block"),"kv-physical":w("e-block")-avg(w,"e-random"),
          "e-code":avg(y,"e-random")-y("e-block"),"relay-abs":y("relay-kv")-y("e-block"),
          "relay-specific":y("relay-kv")-avg(y,"relay-random"),"reverse":y("sham")-y("relay-reverse"),
          "c-vblock":avg(x,"c-random")-x("c-block"),"c-joint":x("joint")-avg(x,"joint-random"),
          "ec-joint":y("ec-joint")-y("ec-block"),"ec-over-partial":y("ec-joint")-np.maximum(y("ec-cache"),y("ec-end"))}
    assert len(sets)==len(p["gate"]["labels"])
    stats=split_effects(p,sets,rows)
    raw_physical=w("e-block").mean(); delta=stats["kv-physical"]["mean"]
    share=delta/raw_physical if raw_physical>0 else None
    dpp=stats["kv-physical"]
    dpp["passed"]=bool(dpp["n"]>=p["gate"]["n"] and dpp["ci"][0]>0 and share is not None and share>=p["gate"]["physical_fraction"])
    abs_effects={"e-state":s("sham")-s("e-block"),"e-other-state":s("sham")-s("e-other"),"self-e":s("self-e")-s("e-block"),
                 "e-code":y("sham")-y("e-block"),"res-k":y("relay-k")-y("e-block"),"res-v":y("relay-v")-y("e-block"),
                 "res-kv":y("relay-kv")-y("e-block"),"occlusion":(y("sham")-y("e-block"))-(y("c-block")-y("ec-block")),
                 "c-vblock":x("sham")-x("c-block"),"c-joint":x("joint")-x("c-block"),
                 "ec-block":y("sham")-y("ec-block"),"ec-cache":y("ec-cache")-y("ec-block"),"ec-end":y("ec-end")-y("ec-block"),
                 "ec-joint":sets["ec-joint"],**{"single-"+str(j):x("single-"+str(j))-x("c-block") for j in range(3)}}
    abs_statistics={k:summarize(fam(v)) for k,v in abs_effects.items()}
    mean=lambda k:abs_statistics[k]["mean"]
    frac=lambda v,g:None if mean(g)<=0 else mean(v)/mean(g)
    fractions=dict(c=frac("c-joint","c-vblock"),ec=frac("ec-joint","ec-block"),physical=share)
    bridge=all(stats[k]["passed"] for k in ("kv-physical","e-code","relay-abs","relay-specific","reverse"))
    full=bool(bridge and all(stats[k]["passed"] for k in ("c-vblock","c-joint","ec-joint","ec-over-partial")) and fractions["ec"] is not None and fractions["ec"]>=p["gate"]["joint_fraction"])
    st=np.array([r["state_ids"][0] for r in rows]); gold=np.array([r["token"]["letters"][r["code"]=="Q"] for r in rows])
    summaries={"code":{k:dict(code_wrong=int((c["top"][:,j]!=gold).sum())) for k,j in ci.items()},"temporal":{}}
    for k,j in ti.items():
        correct=t["top"][:,j]==gold; valid=t["prefix_valid"][:,j]
        summaries["temporal"][k]=dict(state_changed=int((t["state_top"][:,j]!=st).sum()),verified=int(valid.sum()),
                                      actual_code_wrong=int((valid&~correct).sum()),conditional_code_wrong=int((~correct).sum()))
    true_fit=t["prefix_valid"][:,[ti[k] for k in ("sham","e-block","relay-kv","relay-reverse","ec-block","ec-joint")]].all(1).reshape(-1,8).all(1)
    assert np.array_equal(t["applied_k25"][:,ti["relay-kv"]],t["k25_before"][:,0]); assert np.array_equal(t["applied_v25"][:,ti["relay-kv"]],t["v25_before"][:,0]);
    assert np.array_equal(t["applied_k25"][:,ti["relay-reverse"]],t["k25_before"][:,1]); assert np.array_equal(t["applied_v25"][:,ti["relay-reverse"]],t["v25_before"][:,1]);
    assert np.array_equal(t["applied_v25"][:,ti["relay-k"]],t["v25_before"][:,1]); assert np.array_equal(t["applied_k25"][:,ti["relay-v"]],t["k25_before"][:,1]);
    causal={k:bool(np.array_equal(t["state"][:,ti[k]],t["state"][:,1])) for k in ("relay-k","relay-v","relay-kv","ec-block","ec-cache","ec-end","ec-joint")}
    assert all(causal.values())
    geom={k:dict(mean_ratios=t["kv_ratios"][:,j].mean(0).tolist(),median_ratios=np.median(t["kv_ratios"][:,j],axis=0).tolist(),
                 median_g_attention=float(np.median(t["g_alpha25"][:,j])),median_head_change=float(np.median(np.linalg.norm(t["h18_or_control"][:,j]-t["h18_or_control"][:,0],axis=-1)))) for k,j in ti.items()}
    start=min(25,*[l for l,_ in f["early"]]); raw=t["trace"][:,1]
    trace=[dict(layer=l,median=np.median(raw[:,j],axis=0).tolist()) for j,l in enumerate(range(start,40))]
    examples={}
    order=lambda scores,mask:sorted(np.flatnonzero(mask),key=lambda i:scores[i])[:6]
    picks={"e_actual_switch":np.flatnonzero(t["state_top"][:,1]!=st)[:8],"relay_hurts":order(sets["relay-abs"],sets["relay-abs"]<0),
           "large_kv_without_code_effect":order(-w("e-block"),abs(abs_effects["e-code"])<=.125),
           "opposite_reverse":order(sets["reverse"],sets["reverse"]<0),
           "natural_c_block_wrong":np.flatnonzero(c["top"][:,ci["c-block"]]!=gold)[:8]}
    for k,ix in picks.items():
        examples[k]=[dict(row=rows[i],state=t["state"][i].tolist(),actual_state=t["state_top"][i].tolist(),
                          pq=t["pq"][i].tolist(),code=t["top"][i].tolist(),verified=t["prefix_valid"][i].tolist(),
                          first_mismatch=t["first_mismatch"][i].tolist(),first_actual=t["first_generated"][i].tolist(),
                          kv25_ratio=t["kv_ratios"][i].tolist()) for i in ix]
    cells={}
    for side in (0,1):
        for dist in (0,1):
            ok=np.array([r["side"]==side and r["distance"]==dist for r in rows]); cells[f"{side}{dist}"]={k:summarize(v[ok].reshape(-1,2).mean(1)) for k,v in sets.items()}
    A.save("audit.json",dict(state="finished",n=len(rows)//8,eligible_original=len(use)//8,natural_common=int(true_fit.sum()),
           natural=natural,rollout=A.load("rollout.json"),technical=case_tech,original_failures=bad,primary=stats,absolute=abs_statistics,
           fractions=fractions,bridge_passed=bridge,whole_passed=full,arms=summaries,geometry=geom,trace=trace,cells=cells,examples=examples,
           causal_checks=causal,accuracy_gates=behavior["cells"],source=B.sha("confirm.json")))
    print("Heldout n",len(rows)//8,"primary",{k:(v["mean"],v["passed"]) for k,v in stats.items()}, "fractions",fractions,flush=True)


if __name__=="__main__":
    C.setup(); main()
