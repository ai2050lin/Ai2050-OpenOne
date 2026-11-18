"""Reconstruct factorial behavior and frozen native interventions from saved outputs."""

from collections import Counter
from itertools import product
import re

import numpy as np

import kv_orth_run as C
from kv_origin_audit import summarize
from kv_orth_study import behavior_gate, config, load, physics, resolved, walsh


A,B=C.A,C.B


def payload(r):
    return {w:c for w,c in (x.split(" -> ") for x in r["prompt"].split("\nCodebook:\n",1)[1].split("\n\n",1)[0].splitlines())}


def material_checks(rows):
    w=set()
    for k in range(0,len(rows),16):
        a=rows[k:k+16]
        assert len({r["family"] for r in a})==1
        group={(r["distance"],r["map"],r["state"],r["fact"]):r for r in a}
        assert list(group)==list(product((0,1),repeat=4))
        for (d,m,s,f),r in group.items():
            w.update([*r["words"],*r["values"].values()])
            book=payload(r)
            assert book[r["words"][0]]=="PQ"[m] and book[r["words"][1]]=="PQ"[1-m]
            assert r["code_f"]=="PQ"[f^m] and r["code_s"]=="PQ"[s^m]
            assert r["prompt"]==group[d,m,1-s,f]["prompt"]
            assert r["values"]["State"]==r["words"][s] and r["token"]["ids"][r["token"]["slots"][0]]==r["word_ids"][s]
            assert set(r["sites"]["fact"]).isdisjoint(r["sites"]["book"]) and max([*r["sites"]["fact"],*r["sites"]["book"]])<r["token"]["start"]
    return w


def counts(rows,raw):
    base={}; by_distance={}; cases={}; donors={r["id"]:r for r in rows}
    for r,o in zip(rows,raw,strict=True):
        j=f"{r['distance']}{r['map']}{r['state']}{r['fact']}"
        donor=donors[o["donor"]]
        assert donor["family"]==r["family"] and donor["distance"]==r["distance"] and donor["map"]==r["map"] and donor["fact"]==r["state"]
        a=base.setdefault(j,Counter())
        a["total"]+=1
        labels=r["token"]["letters"]
        valid=o["matched"] and o["requested_state_emitted"]
        if o["requested_state_emitted"]:
            a["state"]+=1
        if not valid:
            a["prefix_unresolved"]+=1
            if "divergent_format" not in cases:
                cases["divergent_format"]=dict(row=r,observed=o)
            continue
        full=re.findall(r"^State:[ \t]*([^\r\n]+)",o["rendered"],re.M)
        assert [v.strip() for v in full]==[r["values"]["State"]]
        answer="PQ"[labels.index(o["top"])] if o["top"] in labels else "other"
        a[answer]+=1
        if r["fact"]==r["state"]:
            a["agree_correct" if answer==r["code_f"] else "agree_incorrect"]+=1
        else:
            d=by_distance.setdefault(str(r["distance"]),Counter())
            k="follows_emitted" if answer==r["code_s"] else "follows_fact" if answer==r["code_f"] else "other"
            d[k]+=1;a[k]+=1
            cases.setdefault(k,dict(row=r,observed=o))
    return dict(cells={k:dict(a) for k,a in base.items()},distance={k:dict(a) for k,a in by_distance.items()},examples=cases)


def natural_coeff(rows,obs,gn):
    if not gn["eligible"]:
        return None
    use=resolved(rows,obs,gn)
    ix={o["id"]:o for o in obs}
    x=np.array([ix[r["id"]]["pq"] for r in use],dtype=np.float64)
    ff,ss=(walsh(x[:,1]-x[:,0],use,label) for label in ("f","s"))
    dd=(ss[:,1]-ff[:,1])-(ss[:,0]-ff[:,0])
    return dict(state=ss,fact=ff,dist=dd)


def gate_result(y,threshold,signless=False,delivered=True):
    result=summarize(y,threshold["confidence"])
    x=result["mean"];lo,hi=result["ci"]
    pos=lo>0 and x>=threshold["margin"]
    neg=signless and hi<0 and x<=-threshold["margin"]
    result["passed"]=bool(result["n"]>=threshold["minimum_n"] and delivered and (pos or neg))
    return result


def natural_summary(rows,raw,gn,stats,alt):
    n=counts(rows,raw)
    n["whole_families"]=len(gn["eligible"])
    n["alternate_format"] = dict(tested=alt["tested"],stops=dict(Counter(x["stop"] for x in alt["observations"].values())))
    alt_examples={}
    words={r["id"]:r for r in rows}
    for key,obs in alt["observations"].items():
        r=words[key]
        names=re.findall(r"^State:[ \t]*([^\r\n]+)",obs["rendered"],re.M)
        letters=r["token"]["letters"]
        accepted=(obs["top"] in letters and [x.strip() for x in names]==[r["values"]["State"]])
        if obs["stop"]=="code" and accepted:
            code="PQ"[letters.index(obs["top"])]
            label="resolved_with_state_code" if code==r["code_s"] else "resolved_with_fact_code" if code==r["code_f"] else "resolved_other"
        else:
            label=obs["stop"]
        alt_examples.setdefault(label,dict(row=r,observed=obs))
    n["alternate_format"]["examples"]=alt_examples
    coef=natural_coeff(rows,raw,gn)
    if coef is None:
        return n,{}
    n["coefficients_by_distance"]={r:[summarize(coef[r][:,i]) for i in (0,1)] for r in ("fact","state")}
    n["letter_balance"]= {k:{"P":v.get("P",0),"Q":v.get("Q",0)} for k,v in n["cells"].items()}
    return n,{"state-coef":gate_result(coef["state"].mean(1),stats),"fact-coef":gate_result(coef["fact"].mean(1),stats),
              "distance-balance":gate_result(coef["dist"],stats,signless=True)}


def matched_causal(rows,z,cfg,stats):
    names=list(cfg);k=names.index
    x=z["pq"][:,:,1].astype(np.float64)-z["pq"][:,:,0]
    m=x.copy()
    gold=np.array([r["token"]["letters"][r["code_s"]=="Q"] for r in rows])
    p=np.array([r["token"]["letters"][1] for r in rows])
    old=np.where(z["top"][:,0]==p,1,-1)
    confirm=native_stats={};phys={};good={}
    for name,key in zip(("h18","fact","book"),C.ROLES,strict=True):
        col=C.ROLES.index(key)
        eps=[];effects=[]
        for i in (1,2):
            tar,ran=k(f"{name}-t{i}"),k(f"{name}-r{i}")
            t=z["physical"][:,tar,col];r=z["physical"][:,ran,col]
            assert np.allclose(t[:,1],r[:,1],atol=2e-5,rtol=2e-5)
            err=abs(t[:,2]-r[:,2])/np.maximum(t[:,1],.02)
            eps.append(float(np.quantile(err,.95)))
            phys[f"{name}-{i}"]=dict(p95=eps[-1],target=np.median(t,axis=0).tolist(),random=np.median(r,axis=0).tolist())
            effects.append(x[:,ran]-x[:,tar])
        good[name]=max(eps)<=.03
        delta=np.mean(effects,axis=0)
        signal={tag:walsh(delta,rows,tag) for tag in ("f","s")}
        native_stats[name]={tag:[summarize(sig[:,j]) for j in (0,1)] for tag,sig in signal.items()}
        tag="f" if name=="fact" else "s"
        confirm[f"{name}-{'fact' if tag=='f' else 'state'}"]=gate_result(signal[tag].mean(1),stats,delivered=good[name])
    family=lambda a:a.reshape(-1,16).mean(1)
    for name,a,b in (("joint-code","sham","joint"),("h18-rescue","restore-h18","joint"),("parallel-rescue","restore-parallel","joint")):
        confirm[name]=gate_result(family((x[:,k(a)]-x[:,k(b)])*old),stats)
    arms={}
    for a in names:
        i=k(a)
        sw=z["top"][:,i]!=z["top"][:,0]
        arms[a]=dict(retained=int((~sw).sum()),changed=int(sw.sum()),original_model_margin=summarize(family(x[:,i]*old)),
               beta_f=[summarize(walsh(x[:,i],rows,"f")[:,j]) for j in (0,1)],beta_s=[summarize(walsh(x[:,i],rows,"s")[:,j]) for j in (0,1)],
               ends_on_state=int((z["top"][:,i]==gold).sum()))
    cuts={"joint-real-code-flip":np.flatnonzero(z["top"][:,k("joint")]!=z["top"][:,0])[:7],
          "H18-restoration-harms":np.argsort((x[:,k("restore-h18")]-x[:,k("joint")])*old,kind="stable")[:7],
          "parallel-restoration-harms":np.argsort((x[:,k("restore-parallel")]-x[:,k("joint")])*old,kind="stable")[:7]}
    ex={e:[dict(row=rows[i],pq=z["pq"][i].tolist(),top=z["top"][i].tolist(),
                alpha=z["alpha"][i].tolist(),dose=z["physical"][i].tolist()) for i in ind0] for e,ind0 in cuts.items()}
    return dict(primary=confirm,cross_coefficients=native_stats,phys=phys,paired_delivery=good,arms=arms,examples=ex,n=len(rows)//16)


def audit(p):
    f,led,raw,gate=(A.load(k) for k in ("freeze.json","confirm.json","confirm-natural.json","confirm-gate.json"))
    assert f["plan"]==B.sha("plan.json") and all(B.sha(k)==v for k,v in f["discovery"].items())
    assert (C.OUT/"freeze.json").stat().st_mtime_ns<(C.OUT/"confirm-natural.json").stat().st_mtime_ns
    assert led["freeze"]==raw["binding"]==B.sha("freeze.json") and led["behavior"]==B.sha("confirm-natural.json")
    assert led["gate"]==B.sha("confirm-gate.json") and led["alternate_format"]==B.sha("confirm-alternate-format.json")
    names=["material-pilot.json",p["materials"]["discover"]["file"],p["materials"]["confirm"]["file"]]
    mat=[A.load(name) for name in names]
    sets=[material_checks(v) for v in mat]
    assert all(not sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
    rows=mat[-1];obs=raw["rows"]
    assert behavior_gate(rows,obs)==gate
    summary,prim=natural_summary(rows,obs,gate,p["statistics"],A.load("confirm-alternate-format.json"))
    out=dict(state=led["state"],natural=summary,primary=prim,split_words=[len(s) for s in sets],
             hash={k:B.sha(k) for k in ("plan.json","freeze.json","confirm.json","confirm-natural.json","confirm-gate.json","confirm-alternate-format.json")})
    if led["state"]=="finished":
        use={r["id"]:r for r in resolved(rows,obs,gate)}
        cfg=config(f)
        packs=[];negative=[]
        for name,h in sorted(led["shards"].items()):
            assert B.sha(name)==h
            z=load(name)
            assert physics(z,cfg)==led["technical"][name]
            for i in np.flatnonzero(~z["technical"]):
                negative.append(dict(id=z["ids"][i],pq=z["pq"][i,0].tolist(),top=int(z["top"][i,0]),drift=float(z["numeric_diff"][i])))
            mask=np.repeat(z["technical"].reshape(-1,16).all(1),16)
            packs.append({k:z[k][mask] for k in ("ids","pq","top","physical","alpha")})
        if packs:
            all_={k:np.concatenate([z[k] for z in packs]) for k in packs[0]}
            assert all_["ids"].tolist()==led["accepted"]
            if len(all_["ids"]):
                m=matched_causal([use[k] for k in all_["ids"]],all_,cfg,p["statistics"])
                out["primary"].update(m.pop("primary"));out["causal"]=m
        out["technical"]={"shards":led["technical"],"failures":negative,"arm_names":list(cfg)}
    A.save("audit.json",out)
    print("Accepted",out.get("causal",{}).get("n"),{k:(v["mean"],v["passed"]) for k,v in out["primary"].items()},flush=True)
