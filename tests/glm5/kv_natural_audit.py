"""Independent CPU effect accounting on the frozen Phase 2945 panel."""

from collections import defaultdict
import sys

import numpy as np
from scipy.stats import norm

import kv_natural_run as C
from kv_natural_trace import arms, select_controls


A, B = C.A,C.B


def stats(v, confidence=.95, seed=294500):
    x=np.asarray(v,dtype=np.float64)
    rng=np.random.default_rng(seed)
    values=x[rng.integers(0,len(x),size=(24000,len(x)))].mean(1)
    q=(1-confidence)/2
    return dict(n=len(x),mean=float(x.mean()),ci=np.quantile(values,[q,1-q]).tolist(),
                dz=float(x.mean()/x.std(ddof=1)) if x.std(ddof=1)>0 else None)


def by_family(x):
    return x.reshape(-1,8,*x.shape[1:]).mean(1)


def leave_one_word_out(rows,x):
    f=[]
    for r in rows[::8]:
        f.append(set(r["values"].values())|{r["rejected"]})
    words=set().union(*f)
    vals={w:float(x[[w not in y for y in f]].mean()) for w in sorted(words) if not all(w in y for y in f)}
    return [min(vals.values()),max(vals.values())]


def summarize():
    p=C.verify()
    f=A.load("freeze.json")
    assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-behavior.json")
    assert f["norms"]==B.sha("discover-head-norms.npz")
    if (C.OUT/"confirm-behavior.json").exists():
        assert (C.OUT/"freeze.json").stat().st_mtime_ns < (C.OUT/"confirm-behavior.json").stat().st_mtime_ns
    with np.load(C.OUT/"discover-head-norms.npz") as z:
        assert select_controls({25:z["layer25"],26:z["layer26"]},5)==f["control"]
    seen=A.load("confirm-behavior.json")
    ledger=A.load("confirm.json")
    assert ledger["behavior"]==B.sha("confirm-behavior.json")
    all_rows=A.load(p["materials"]["confirm"]["name"])
    assert C.behavior(all_rows,seen["rows"])["cells"]==seen["cells"]
    natural_fails=[]
    for r,o in zip(all_rows,seen["rows"],strict=True):
        if not o["code_correct"]:
            natural_fails.append({"row":r,"observed":o})
    if ledger["state"]!="finished":
        A.save("audit.json",dict(state=ledger["state"],behavior=seen["cells"],natural_fails=natural_fails))
        return
    assert ledger["freeze"]==B.sha("freeze.json")
    indices={x:i for i,x in enumerate(f["arm_order"])}
    assert list(indices)==list(arms(f["control"]))
    d=B.load_data(ledger)
    rows=[r for r in all_rows if r["family"] in set(seen["eligible"])]
    assert list(d["ids"])==[r["id"] for r in rows]
    assert len(d["ids"])==len(rows)==8*ledger["families"]
    assert all(len({r["family"] for r in rows[j:j+8]})==1 for j in range(0,len(rows),8))
    pq=d["pq"].astype(np.float64)
    margin=(pq[:,:,1]-pq[:,:,0])*np.asarray([1 if r["code"]=="Q" else -1 for r in rows])[:,None]
    token_gold=np.asarray([r["token"]["letters"][r["code"]=="Q"] for r in rows])
    right=d["top"]==token_gold[:,None]
    assert right[:,0].all()
    arm=lambda n:margin[:,indices[n]]
    drop=lambda n:arm("sham")-arm(n)
    losses={k:drop(k) for k in ("u","r4","r34","pair","u-pair")}
    rnd=lambda k:(arm(k+"-rand1")+arm(k+"-rand2"))/2
    endpoints={k: rnd(k)-arm(k) for k in ("u","r4","r34","pair")}
    endpoints["rescue-specific"]=arm("rescue")-rnd("rescue")
    endpoints["rescue-absolute"]=arm("rescue")-arm("u")
    n_family=ledger["families"]
    primary={}
    for k,x in endpoints.items():
        fam=by_family(x)
        z=stats(fam,p["primary"]["confidence"])
        primary[k]={**z,"passed":bool(n_family>=80 and z["mean"]>=p["primary"]["min_mean_logit"] and z["ci"][0]>0),
                    "leave_one_word_out":leave_one_word_out(rows,fam)}
    behavior={}
    for name,i in indices.items():
        wins=right[:,i]
        behavior[name]=dict(right=int(wins.sum()),n=len(wins),flips=int((~wins).sum()),non_code=int((~np.isin(d["top"][:,i],rows[0]["token"]["letters"])).sum()),
                            complete=int(wins.reshape(-1,8).all(1).sum()),margin_loss=stats(by_family(drop(name))))
    physical={}
    for layer in C.TARGET:
        raw=d[f"{layer}-head_raw"].astype(np.float64)
        changed=d[f"{layer}-head_edit"].astype(np.float64)
        assert np.isfinite(raw).all() and np.isfinite(changed).all()
        base,act,write={},{},{}
        def brief(x):
            return np.quantile(x,[.10,.5,.90],axis=0).tolist()
        for name,i in indices.items():
            act[name]={k:brief(d[f"{layer}-{k}"][:,i]) for k in ("query_change","actual_projection_delta","actual_residual_delta","actual_block_delta")}
            write[name] = dict(before_with_live_q=brief(np.linalg.norm(raw[:,i]-raw[:,0],axis=-1)),
                               applied_vs_sham=brief(np.linalg.norm(changed[:,i]-changed[:,0],axis=-1)))
        for k in ("alpha","prompt_alpha","slot_projected","head_projected"):
            base[k]={str(c)+str(dist):brief(d[f"{layer}-{k}"][np.array([r["compete"]==c and r["distance"]==dist for r in rows]),0])
                      for c in (0,1) for dist in (0,1)}
        assert np.all(d[f"{layer}-alpha"]>=0) and np.all(d[f"{layer}-alpha"].sum(-1)<=1.00001)
        physical[str(layer)]={"sham":base,"actual_deltas":act,"head_write_change":write}
    for name,layer,hs in (("u",25,[0]),("r4",26,[0]),("r34",26,[1]),("pair",26,[0,1])):
        assert not np.any(d[f"{layer}-head_edit"][:,indices[name],hs])
        assert np.any(d[f"{layer}-actual_projection_delta"][:,indices[name]]>0)
        assert np.array_equal(d["top"][:,indices["self-"+name]],d["top"][:,0])
    u,r=indices["u"],indices["rescue"]
    assert np.array_equal(d["26-query_change"][:,u],d["26-query_change"][:,r])
    assert np.array_equal(d["26-head_edit"][:,r],d["26-intact_head"][:,r])
    patch_error=np.linalg.norm(d["26-head_edit"][:,r]-d["26-head_edit"][:,0],axis=-1)
    assert np.array_equal(d["25-head_edit"][:,u],d["25-head_edit"][:,r])
    by_condition={}
    for compete in (0,1):
        for distance in (0,1):
            sel=np.array([r["compete"]==compete and r["distance"]==distance for r in rows])
            by_condition[str(compete)+str(distance)]=dict(
                n=int(sel.sum()),family_n=n_family, endpoints={k:stats(x[sel].reshape(-1,2).mean(1)) for k,x in endpoints.items()},
                flips={n:int((~right[sel,i]).sum()) for n,i in indices.items()})
    candidates={
        "upstream_flip":np.flatnonzero(~right[:,u]).tolist(),
        "upstream_block_improves_margin":sorted(range(len(rows)),key=lambda j:losses["u"][j])[:8],
        "pair_flip":np.flatnonzero(~right[:,indices["pair"]]).tolist(),
        "recipients_restored_code_still_wrong":np.flatnonzero(~right[:,r]).tolist(),
        "rescue_worsens_margin":sorted(range(len(rows)),key=lambda j:endpoints["rescue-absolute"][j])[:8],
    }
    qchange=d["26-query_change"][:,u].mean(1)
    invariant=np.flatnonzero(right[:,u] & (abs(losses["u"])<=.1))
    candidates["query_moves_code_and_margin_stable"]=sorted(invariant,key=lambda j:-qchange[j])[:8]
    cases={}
    for kind,js in candidates.items():
        cases[kind]=[dict(row=rows[j],arms={k:dict(top=int(d["top"][j,i]),pq=d["pq"][j,i].tolist()) for k,i in indices.items()},
                         source_projection=float(d["25-actual_projection_delta"][j,u]),
                         upstream_query_displacement=d["26-query_change"][j,u].tolist(),
                         signed_source_loss=float(losses["u"][j]),signed_rescue=float(endpoints["rescue-absolute"][j])) for j in js[:12]]
    alpha=p["primary"]["alpha"]
    power=float(norm.sf(norm.isf(alpha/2)-.4*np.sqrt(n_family))+norm.cdf(-norm.isf(alpha/2)-.4*np.sqrt(n_family)))
    chain=all(primary[k]["passed"] for k in ("u","pair","rescue-specific","rescue-absolute"))
    numeric=dict(sham_vs_open=float(np.max(abs(d["untreated"]-d["pq"][:,0]))),
                 rescue_recipient_max_delta_from_live_sham=float(np.max(patch_error)),
                 self_restore_vs_sham={n:float(np.max(abs(d["pq"][:,indices["self-"+n]]-d["pq"][:,0]))) for n in ("u","r4","r34","pair")})
    effect=dict(absolute_source=stats(by_family(losses["u"])), absolute_pair=stats(by_family(losses["pair"])),
                occlusion=stats(by_family(losses["pair"]-(arm("u")-arm("u-pair")))), mean_rnd_rescue=stats(by_family(rnd("rescue")-arm("u"))))
    A.save("audit.json",dict(state="finished", plan=B.sha("plan.json"), freeze=B.sha("freeze.json"),
                           eligible_families=n_family, baseline=seen["cells"], n_natural_failures=len(natural_fails),
                           natural_fails=natural_fails, primary=primary, chain_passed=chain, behavior=behavior,
                           physical=physical, numeric=numeric, cells=by_condition, counterexamples=cases,
                           effects=effect, independent_power_dz_040=power, ledger=B.sha("confirm.json")))
    print("endpoint results",primary,flush=True)


if __name__=="__main__":
    C.setup()
    summarize()
