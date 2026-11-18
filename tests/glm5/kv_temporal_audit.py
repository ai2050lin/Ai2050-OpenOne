"""Independent paired-family evaluation of emission, State memory, and code."""

import numpy as np

import kv_temporal_run as C
from kv_origin_audit import summarize, fam, leave_lexeme
from kv_temporal_study import code_arms,temporal_arms,signed


A,B=C.A,C.B


def timing(d,ids,target,st):
    columns={k:i for i,k in enumerate(ids)}
    metrics={}
    for a,j in columns.items():
        x=target[:,j]
        z=dict(conditional_code_correct=int(x.sum()),conditional_code_total=len(x))
        if "prefix_valid" in d:
            valid=d["prefix_valid"][:,j]
            z.update(natural_valid=int(valid.sum()),natural_wrong=int((valid&~x).sum()),prefix_diverged=int((~valid).sum()),
                     state_wrong=int((d["state_top"][:,j]!=st).sum()))
        metrics[a]=z
    return metrics


def filtered_data(ledger,kind):
    assert all(B.sha(k)==v for k,v in ledger["acceptance"][kind].items())
    return B.load_data(dict(shards=ledger["panels"][kind]))


def main():
    p=C.verify(); f=A.load("freeze.json"); d0=A.load("confirm.json"); base=A.load("confirm-behavior.json")
    assert (C.OUT/"freeze.json").stat().st_mtime_ns<(C.OUT/"confirm-behavior.json").stat().st_mtime_ns
    assert f["plan"]==B.sha("plan.json") and d0["behavior"]==B.sha("confirm-behavior.json")
    rows=A.load(p["materials"]["confirm"]["name"])
    quality=[dict(row=r,observed=b) for r,b in zip(rows,base["rows"],strict=True) if not b["code_correct"]]
    if d0["state"]!="finished":
        A.save("audit.json",dict(state=d0["state"],behavior=base["cells"],failures=quality))
        return
    assert d0["freeze"]==B.sha("freeze.json")==base["binding"]
    assert all(B.sha(k)==v for k,v in f["files"].items())
    assert d0["replay"]==B.sha("observed-state-replay.json")
    use=C.O.resolved(rows,base)
    dd={k:filtered_data(d0,k) for k in ("code","temporal")}
    ids=np.array([r["id"] for r in use]); assert all(np.array_equal(z["ids"],ids) for z in dd.values())
    both=dd["code"]["technical"]&dd["temporal"]["technical"]
    good=np.repeat(both.reshape(-1,8).all(-1),8)
    known={x["id"]:x for x in base["rows"]}
    exclusions={k:[dict(id=r["id"],heldout_pq=base_obs["pq"],sham_pq=d["pq"][i,0].tolist(),
                         code_token=int(d["top"][i,0]),numeric_diff=float(d["numeric_diff"][i]))
                   for i,(r,base_obs) in enumerate(zip(use,[known[r["id"]] for r in use],strict=True))
                   if not d["technical"][i]] for k,d in dd.items()}
    rr=[r for r,ok in zip(use,good,strict=True) if ok]
    replay=A.load("observed-state-replay.json")
    if not rr:
        A.save("audit.json",dict(state="technical-gate-failed",behavior=base["cells"],failures=quality,technical=exclusions))
        return
    c,t=({k:x[good] for k,x in dd[stage].items()} for stage in ("code","temporal"))
    assert f["code_names"]==list(code_arms(f)["arms"]) and f["temporal_names"]==list(temporal_arms(f)["arms"])
    gold=np.array([r["token"]["letters"][r["code"]=="Q"] for r in rr]); st=np.array([r["state_ids"][0] for r in rr])
    cm,tm=signed(c,rr),signed(t,rr)
    sm=t["state"][:,:,0].astype(np.float64)-t["state"][:,:,1]
    ci={x:j for j,x in enumerate(f["code_names"])}
    ti={x:j for j,x in enumerate(f["temporal_names"])}
    x=lambda k:cm[:,ci[k]]
    y=lambda k:tm[:,ti[k]]
    s=lambda k:sm[:,ti[k]]
    mid=lambda fn,base:(fn(base+"1")+fn(base+"2"))/2
    joint="fine-"+f["joint"]
    single="fine-"+f["singleton"]
    ep={"e-specific":mid(s,"e-rand")-s("e-block"),"e-restore":s("self-e")-s("e-block"),
        "e-code":mid(y,"e-rand")-y("e-block"),"g-rescue":y("g-rescue")-y("e-block"),"g-specific":y("g-rescue")-y("g-spare"),
        "c-specific":mid(x,"c-rand")-x("c-block"),"c-joint-random":x(joint)-mid(x,"fine-rand"),
        "c-joint-single":x(joint)-x(single),"ec-joint":y("ec-joint")-y("ec-block")}
    primary={}
    tech=len(rr)//8
    for key,z in ep.items():
        family=fam(z); q=summarize(family,p["primary"]["confidence"])
        primary[key]=dict(**q,passed=(q["mean"]>=p["primary"]["floor"] and q["ci"][0]>0 and tech>=p["primary"]["n"]),
                          delete_one_word=leave_lexeme(family,rr))
    stage={"e-loss":s("sham")-s("e-block"),"e-other":s("sham")-s("e-other"),"ec-loss":y("sham")-y("ec-block"),
           "e-code-loss":y("sham")-y("e-block"),"g-block":y("sham")-y("g-block"),"c-loss":x("sham")-x("c-block"),
           "ec-g":y("ec-g")-y("ec-block"),"ec-fine":y("ec-fine")-y("ec-block"),"ec-joint":ep["ec-joint"],
           "ec-joint-extra":y("ec-joint")-np.maximum(y("ec-g"),y("ec-fine")),"c-joint-absolute":x(joint)-x("c-block")}
    stage={k:summarize(fam(v)) for k,v in stage.items()}
    frac=lambda v,z:v/z if z>0 else None
    fractions=dict(c_joint=frac(stage["c-joint-absolute"]["mean"],stage["c-loss"]["mean"]),
                   ec_joint=frac(stage["ec-joint"]["mean"],stage["ec-loss"]["mean"]))
    c_factor={k:summarize(fam(x(k)-x("c-block"))) for k in f["code_names"] if k.startswith(("fine-","coarse-"))}
    summary={"temporal":timing(t,ti,t["top"]==gold[:,None],st),"code":timing(c,ci,c["top"]==gold[:,None],st)}
    agree=t["prefix_valid"]
    common=agree[:,[ti[k] for k in ("sham","e-block","g-rescue","g-spare","ec-block","ec-joint")]].all(1).reshape(-1,8).all(1)
    natural={k:summarize(fam(ep[k])[common],p["primary"]["confidence"]) for k in ("e-code","g-rescue","g-specific","ec-joint")} if common.any() else {}
    effects={}
    for m in (0,1):
        for n in (0,1):
            w=np.array([r["side"]==m and r["distance"]==n for r in rr])
            effects[f"{m}{n}"]={k:summarize(v[w].reshape(-1,2).mean(1)) for k,v in ep.items()}
    offset=min(25,*[k[0] for k in f["early"]])
    xx=t["trace"]
    trace=[dict(layer=l,e_resid=float(np.median(xx[:,ti["e-block"],j,2])),g_resid=float(np.median(xx[:,ti["e-block"],j,3])),
                g_key=float(np.median(xx[:,ti["e-block"],j,4])),g_value=float(np.median(xx[:,ti["e-block"],j,5])),
                e_disruption_at_code=float(np.median(xx[:,ti["e-block"],j,0])),mlp_at_code=float(np.median(xx[:,ti["e-block"],j,1])))
           for j,l in enumerate(range(offset,40))]
    cases={}
    choose={"early_emits_different_state":list(np.flatnonzero(t["state_top"][:,ti["e-block"]]!=st))[:10],
            "propagation_without_code_change":sorted(np.flatnonzero((abs(y("sham")-y("e-block"))<=.1)&(t["top"][:,ti["e-block"]]==gold)),key=lambda i:-np.linalg.norm(t["state_value25"][i,ti["e-block"]]-t["state_value25"][i,0]))[:6],
            "native_state_memory_rescue_harms":sorted(np.flatnonzero(ep["g-rescue"]<0),key=lambda i:ep["g-rescue"][i])[:6],
            "joint_below_single":sorted(np.flatnonzero(ep["c-joint-single"]<0),key=lambda i:ep["c-joint-single"][i])[:6],
            "natural_error_after_code_block":list(np.flatnonzero(c["top"][:,ci["c-block"]]!=gold))[:10]}
    for label,ix in choose.items():
        cases[label]=[dict(row=rr[i],state={k:t["state"][i,j].tolist() for k,j in ti.items()},state_top=t["state_top"][i].tolist(),
                          verified_prefix=t["prefix_valid"][i].tolist(),first_mismatch=t["first_mismatch"][i].tolist(),
                          first_generated=t["first_generated"][i].tolist(),code_top_temporal=t["top"][i].tolist(),
                          code_top=c["top"][i].tolist(),pq_temporal=t["pq"][i].tolist(),pq=c["pq"][i].tolist()) for i in ix]
    result=dict(state="finished",n=tech,behavior=base["cells"],failures=quality,technical=exclusions,primary=primary,absolute=stage,
                coverage=dict(natural_common=int(common.sum()),behavior_families=len(use)//8,no_op_token_recovered=tech),
                fractions=fractions,code_factorial=c_factor,arms=summary,trace=trace,condition=effects,examples=cases,
                natural_endpoints=natural,replay=replay,technical_max={k:float(z["numeric_diff"].max()) for k,z in (("code",c),("temporal",t))},
                origin_max={k:float(z["origin_error"].max()) for k,z in (("code",c),("temporal",t))},confirmed=B.sha("confirm.json"))
    A.save("audit.json",result)
    print("Paired estimates",{k:(v["mean"],v["passed"]) for k,v in primary.items()},"families",tech,"joint fractions",fractions,flush=True)


if __name__=="__main__":
    C.setup(); main()
