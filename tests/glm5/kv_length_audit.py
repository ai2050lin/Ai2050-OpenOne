"""Recompute actual lexical behavior and the prebound path endpoints from exact shards."""

from collections import Counter
from itertools import product
import re

import numpy as np

import kv_length_run as C
from kv_length_natural import gate
from kv_length_read import TYPES
from kv_length_study import active,arms,load,physical
from kv_origin_audit import summarize
from kv_orth_study import walsh


A,B=C.A,C.B


def material_check(rows,bank):
    words=set()
    for lo in range(0,len(rows),16):
        a=rows[lo:lo+16]
        assert len({x["family"] for x in a})==1
        grid={(r["distance"],r["map"],r["state"],r["fact"]):r for r in a}
        assert list(grid)==list(product((0,1),repeat=4))
        for (d,m,s,f),r in grid.items():
            words.update([*r["values"].values(),*r["words"]])
            assert r["code_f"]=="PQ"[f^m] and r["code_s"]=="PQ"[s^m]
            assert grid[1-d,m,s,f]["prompt"]==r["prompt"]==grid[d,m,1-s,f]["prompt"]
            if not d:
                z=r["token"]["ids"];b=C.receipt_anchor(r)+1
                assert z[:b]+bank["extension"]+z[b:]==grid[1,m,s,f]["token"]["ids"]
    return words


def letters(r,t):
    opts=r["token"]["letters"]
    return "PQ"[opts.index(int(t))] if t in opts else "other"


def counts(rows,seen,g,bank):
    all_={"0":Counter(),"1":Counter()};cell={};cases={};distance={}
    verified={x["id"]:x for x in seen}
    lookup={r["id"]:r for r in rows}
    fresh=[r for r in rows if r["id"] in set(g["individual_failures"])][:8]
    complete=set(g["eligible"])
    for r,o in zip(rows,seen,strict=True):
        d=str(r["distance"]);key=r["id"].rsplit("-",1)[-1]
        c=cell.setdefault(key,Counter());c["attempted"]+=1
        donor=lookup[o["donor"]]
        assert [donor[k] for k in ("family","distance","map","fact")]==[r["family"],r["distance"],r["map"],r["state"]]
        if r["fact"]!=r["state"]:
            all_[d]["attempted"]+=1
            all_[d]["state_chosen"]+=int(o["state_emitted"])
            all_[d]["actual_state_switched"]+=int(o["state_emitted"] and o["state"]!=o["untreated"])
        c["matched"]+=int(o["matched"])
        if not o["matched"]:continue
        rendered=o["rendered"];parsed=lambda label:re.findall(rf"^{label}:[ \t]*([^\r\n]+)",rendered,re.M)
        assert [x.strip() for x in parsed("State")]==[r["values"]["State"]]
        assert [x.strip() for x in parsed("Receipt")]==[(C.SHORT,C.LONG)[r["distance"]]]
        k=o["adapted"] or r["token"]
        c["forced_branch_differs_from_unedited"]+=int(o["intended_boundary"]!=o["untreated_boundary"])
        distance.setdefault(d,[]).append(len(k["ids"])-1-k["slots"][0])
        label=letters(r,o["top"]);c[label]+=1
        if r["fact"]!=r["state"]:
            follow="state" if label==r["code_s"] else "fact" if label==r["code_f"] else "other"
            c[f"follows_{follow}"]+=1;all_[d][f"follows_{follow}"]+=1
            cases.setdefault(d+"-"+follow,dict(row=r,observed=o))
    paired=Counter();details={}
    for start in range(0,len(rows),16):
        if rows[start]["family"] not in complete:continue
        for i in range(8):
            l,h=rows[start+i],rows[start+i+8]
            if l["fact"]==l["state"]:continue
            a,b=verified[l["id"]],verified[h["id"]]
            x,y=letters(l,a["top"]),letters(h,b["top"])
            state=l["code_s"]
            key=("state" if x==state else "fact")+"_to_"+("state" if y==state else "fact")
            paired[key]+=1
            if x!=y:details.setdefault(key,dict(row=l,short=a,long=b))
    return dict(cells={k:dict(v) for k,v in cell.items()},raw_conflict={k:dict(v) for k,v in all_.items()},
                exact_gap={k:[min(v),max(v)] for k,v in distance.items()},length_transitions=dict(paired),
                records={**cases,**details},invalid=[dict(row=r,observed=verified[r["id"]]) for r in fresh])


def stat(x,p,physical=True):
    s=summarize(x,p["confidence"])
    s["passed"]=bool(s["n"]>=p["n"] and physical and s["mean"]>=p["margin"] and s["ci"][0]>0)
    return s


def nature_metrics(rows,y,g,b,p):
    behavior=counts(rows,y,g,b)
    behavior.update(exact_pairs=g["exact_pairs"],complete=len(g["eligible"]),pair_failures=g["pair_failures"])
    chosen=active(rows,y,g)
    if not chosen:return behavior,{}
    obs={r["id"]:r for r in y}
    margin=np.array([obs[r["id"]]["pq"][1]-obs[r["id"]]["pq"][0] for r in chosen],dtype=np.float64)
    f=walsh(margin,chosen,"f");s=walsh(margin,chosen,"s")
    conflict=np.array([r["fact"]!=r["state"] for r in chosen]).reshape(-1,2,8)
    truth=np.array([1 if r["code_s"]=="Q" else -1 for r in chosen]).reshape(-1,2,8)
    same=(margin.reshape(-1,2,8)*truth*conflict).sum(-1)/4
    assert np.array_equal(s-f,same)
    behavior["coefs"]={k:[summarize(x[:,d]) for d in (0,1)] for k,x in (("state",s),("fact",f),("conflict",same))}
    length=same[:,1]-same[:,0]
    return behavior,{"distance-preference":stat(length,p)}


def tested_reads(rows,z,cfg,p):
    names=list(cfg);j=names.index
    x=z["pq"][:,:,1].astype(np.float64)-z["pq"][:,:,0]
    factors=lambda a,f:walsh(a,rows,f)
    primary={};dosages={};coefs={};results={}
    for key,src in zip(TYPES,(0,0,1,2),strict=True):
        diff=[];eps=[]
        for k in (1,2):
            t,r=j(f"{key}-t{k}"),j(f"{key}-r{k}")
            target=z["dose"][:,t,src];rand=z["dose"][:,r,src]
            assert np.allclose(target[:,1],rand[:,1],rtol=1e-5,atol=1e-5)
            err=abs(target[:,2]-rand[:,2])/np.maximum(target[:,1],.02)
            eps.append(float(np.quantile(err,.95)))
            dosages[f"{key}-{k}"]=dict(p95=eps[-1],target=np.median(target,axis=0).tolist(),control=np.median(rand,axis=0).tolist())
            diff.append(x[:,r]-x[:,t])
        by=np.mean(diff,0)
        f,s=factors(by,"f"),factors(by,"s")
        coefs[key]={n:[summarize(arr[:,i]) for i in (0,1)] for n,arr in (("fact",f),("state",s))}
        k={"address":"address-state","value":"value-state","fact":"fact-preference","book":"book-preference"}[key]
        primary[k]=stat((f if key=="fact" else s).mean(1),p,max(eps)<=p["p95"])
    q=np.array([r["token"]["letters"][1] for r in rows])
    sign=np.where(z["top"][:,0]==q,1.,-1.)
    mean_group=lambda y:y.reshape(-1,16).mean(-1)
    for k,h,l in (("joint-margin","sham","joint"),("h18-rescue","restore-h18","joint"),("parallel-rescue","restore-parallel","joint")):
        primary[k]=stat(mean_group((x[:,j(h)]-x[:,j(l)])*sign),p)
    for k in ("sham","address","value","key","query","kv","restore-k","restore-v","restore-kv","fact","book","parallel","joint","restore-h18","restore-parallel","restore-all"):
        index=j(k); c={"0":Counter(),"1":Counter()}
        for i,r in enumerate(rows):
            if r["fact"]==r["state"]:continue
            letter=letters(r,z["top"][i,index]);key="state" if letter==r["code_s"] else "fact" if letter==r["code_f"] else "other"
            c[str(r["distance"])][key]+=1
        loss=(x[:,0]-x[:,index])*sign
        results[k]=dict(flip=int((z["top"][:,0]!=z["top"][:,index]).sum()),loss=summarize(mean_group(loss)),
            conflict={d:dict(v) for d,v in c.items()},f=[summarize(a) for a in factors(x[:,index],"f").T],
            s=[summarize(a) for a in factors(x[:,index],"s").T])
    geom={k:np.median(z["geom"][:,j(k)],0).tolist() for k in ("address","value","key","query","kv","restore-k","restore-v","restore-kv")}
    key_abs={k:float(np.quantile(abs(z["geom"][:,j(k),3]),.95)) for k in ("key","query","kv")}
    selectors={"joint_changes_actual_code":np.flatnonzero(z["top"][:,j("joint")]!=z["top"][:,0])[:6],
               "key_only_recovers":np.flatnonzero((z["top"][:,j("kv")]!=z["top"][:,0])&(z["top"][:,j("restore-k")]==z["top"][:,0]))[:6],
               "value_only_recovers":np.flatnonzero((z["top"][:,j("kv")]!=z["top"][:,0])&(z["top"][:,j("restore-v")]==z["top"][:,0]))[:6],
               "h18_restore_lowers_original_margin":np.flatnonzero((x[:,j("restore-h18")]-x[:,j("joint")])*sign<0)[:6],
               "address_and_value_give_different_code":np.flatnonzero(z["top"][:,j("address")]!=z["top"][:,j("value")])[:6]}
    examples={k:[dict(row=rows[i],pq=z["pq"][i].tolist(),top=z["top"][i].tolist(),geo=z["geom"][i].tolist()) for i in ix] for k,ix in selectors.items()}
    return dict(primary=primary,n=len(rows)//16,dose=dosages,cross=coefs,arms=results,geometry=geom,score95=key_abs,examples=examples)


def audit(p):
    f,z,g,raw,bank=(A.load(k) for k in ("freeze.json","confirm.json","confirm-gate.json","confirm-natural.json","neutral-donors.json"))
    assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-natural.json")
    assert z["freeze"]==raw["binding"]==B.sha("freeze.json") and z["gate"]==B.sha("confirm-gate.json")
    assert z["natural"]==B.sha("confirm-natural.json") and f["donors"]==B.sha("neutral-donors.json")
    assert all(B.sha(k)==v for k,v in f["source"].items())
    assert (C.OUT/"freeze.json").stat().st_mtime_ns < (C.OUT/"confirm-natural.json").stat().st_mtime_ns
    material=[A.load(k) for k in ("material-pilot.json",p["panels"]["discover"]["file"],p["panels"]["confirm"]["file"])]
    used=[material_check(x,bank) for x in material]
    assert len(set.union(*used))==sum(map(len,used))
    rows=material[-1];seen=raw["rows"]
    assert gate(rows,seen,bank)==g
    nat,primary=nature_metrics(rows,seen,g,bank,p["gate"])
    ref={k:B.sha(k) for k in ("plan.json","freeze.json","confirm.json","confirm-gate.json","confirm-natural.json","neutral-donors.json")}
    info=dict(state=z["state"],natural=nat,primary=primary,word_count=[len(x) for x in used],binding=ref)
    if z["state"]=="finished":
        cfg=arms(f);parts=[]
        for k,h in sorted(z["files"].items()):
            assert B.sha(k)==h
            v=load(k);assert physical(v,cfg)==z["checks"][k]
            m=np.repeat(v["technical"].reshape(-1,16).all(-1),16)
            parts.append({s:v[s][m] for s in ("ids","pq","top","dose","geom")})
        if parts:
            all_={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
            assert all_["ids"].tolist()==z["accepted"]
            if len(all_["ids"]):
                lut={r["id"]:r for r in active(rows,seen,g)}
                result=tested_reads([lut[k] for k in all_["ids"]],all_,cfg,p["gate"])
                primary.update(result.pop("primary"));info["reads"]=result
        info["technical"]=z["checks"]
    A.save("audit.json",info)
    print("Admitted",info.get("reads",{}).get("n"),{k:(v["mean"],v["passed"]) for k,v in primary.items()},flush=True)
