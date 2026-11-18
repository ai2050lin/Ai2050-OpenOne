"""Independent actual-token insertion check and Q/K/V support proof."""

from collections import Counter

import numpy as np

import kv_length_run as C
from kv_length_study import load
from kv_origin_audit import summarize


A,B=C.A,C.B


def lexical(rows,obs,bank,gate):
    seen={r["id"]:r for r in obs}
    known=set(gate["eligible"])
    gaps={"0":[],"1":[]};good=0
    strict_conflict={"0":Counter(),"1":Counter()}
    fail=[]
    switches=Counter()
    for j in range(0,len(rows),16):
        part=rows[j:j+16]
        for k in range(8):
            a,b=part[k],part[k+8]
            x,y=seen[a["id"]],seen[b["id"]]
            assert a["prompt"]==b["prompt"]
            pair=[]
            for r,o in ((a,x),(b,y)):
                if not o["matched"]:continue
                d=str(r["distance"]);t=o["adapted"] or r["token"];start=t["start"]
                assert o["emitted"][:-1]==t["ids"][start:]
                assert o["emitted"][-1]==o["top"]
                assert t["ids"][o["pivot"]+1]==bank["sources"][int(d)]["next"]
                assert t["ids"][t["slots"][0]]==o["state"]==r["word_ids"][r["state"]]
                gaps[d].append(len(t["ids"])-1-t["slots"][0]);pair.append((t["ids"],o["pivot"]+1))
                if r["fact"]==r["state"] or r["family"] not in known:continue
                opt=r["token"]["letters"]
                word="state" if o["top"]==opt[r["code_s"]=="Q"] else "fact" if o["top"]==opt[r["code_f"]=="Q"] else "other"
                strict_conflict[d][word]+=1
                switches[d]+=int(o["state"]!=o["untreated"])
            if len(pair)==2:
                (s,t),(l,k1)=pair
                updated=l[:k1]+l[k1+len(bank["extension"]):]
                if updated==s and t==k1 and l[k1:k1+len(bank["extension"])]==bank["extension"]:good+=1
                else:fail.append([a["id"],b["id"]])
    assert good==gate["exact_pairs"]
    return dict(exact_pairs=good,matched_conflict={k:dict(v) for k,v in strict_conflict.items()},
                genuine_switches=dict(switches),gap={k:[min(v),max(v)] if v else None for k,v in gaps.items()},
                token_difference_examples=fail[:8])


def source_check(z,names):
    idx=names.index
    yes={}
    for name in ("address","value","key","query","kv","restore-k","restore-v","restore-kv"):
        v=z["geom"][:,idx(name)].astype(np.float64)
        a,b,s,t,q,k,x=v.T
        if name=="address":assert (b==0).all() and (v[:,4:]==0).all() and np.array_equal(s,t)
        if name=="value":assert np.array_equal(a,b) and np.array_equal(s,t) and (v[:,4:6]==0).all()
        if name=="key":assert (q==0).all() and (x==0).all()
        if name=="query":assert (k==0).all() and (x==0).all()
        if name=="restore-kv":assert np.array_equal(a,b) and np.array_equal(s,t) and (v[:,4:]==0).all()
        yes[name]=dict(max_abs_score_after=float(abs(t).max()),median_weight_before=float(np.median(a)),median_weight_after=float(np.median(b)))
    err={}
    for key in ("address","value"):
        v,u,w=z[key].astype(np.float64).transpose(1,0,2)
        a=z["geom"][:,idx(key),0:1].astype(np.float64)
        theoretical=a*(u-v) if key=="address" else -a*v
        err[key]=float((np.linalg.norm(w-theoretical,axis=-1)/(1+np.linalg.norm(theoretical,axis=-1))).max())
        assert err[key]<.05
    return dict(paths=yes,error=err,n=len(z["ids"]))


def paired_geometry(panels,names,accepted):
    if not accepted:
        return {}
    h=names.index("address")
    geometry=[];values=[];ids=[]
    for z in panels:
        good=np.array([k in accepted for k in z["ids"]],dtype=bool)
        geometry.append(z["geom"][good,h]);values.append(z["address"][good,0]);ids.extend(z["ids"][good].tolist())
    assert len(ids)==len(accepted)
    z=np.concatenate(geometry).astype(np.float64).reshape(-1,2,8,7)
    v=np.concatenate(values).astype(np.float64).reshape(len(z),2,8,128)
    a,s=z[...,0],z[...,2]
    # a/(1-a) = exp(score_G)/sum(exp(score_others)). This is a physical identity, with measured a in float32.
    measurable=(a>1e-7)&(a<1-1e-7)
    clips=np.clip(a,1e-7,1-1e-7)
    pool=s-np.log(clips/(1-clips))
    m=measurable[:,0]&measurable[:,1]
    result={}
    for key,arr,sel in (("alpha",a,np.ones(m.shape,bool)),("score",s,np.ones(m.shape,bool)),("other_log_partition",pool,m)):
        short,long=arr[:,0][sel],arr[:,1][sel]
        result[key]=dict(short=float(np.median(short)),long=float(np.median(long)),paired_change=float(np.median(long-short)),n=int(sel.sum()))
    dv=np.linalg.norm(v[:,1]-v[:,0],axis=-1)/np.maximum(np.linalg.norm(v[:,0],axis=-1),1e-3)
    result["state_value_preserved"]=dict(exact=int((dv==0).sum()),n=int(dv.size),max_relative=float(dv.max()),median_relative=float(np.median(dv)))
    return result


def distance_response(panels,names,accepted,rows):
    if not accepted:return {}
    bank={x["id"]:x for x in rows}
    ids=[];outputs=[]
    for z in panels:
        mask=np.array([k in accepted for k in z["ids"]],dtype=bool)
        ids.extend(z["ids"][mask]);outputs.append(z["pq"][mask])
    p=np.concatenate(outputs).astype(np.float64)
    sign=np.array([1 if bank[x]["code_s"]=="Q" else -1 for x in ids],dtype=np.float64)
    conflict=np.array([bank[x]["fact"]!=bank[x]["state"] for x in ids])
    score=(p[:,:,1]-p[:,:,0])*sign[:,None]*conflict[:,None]
    by=score.reshape(-1,2,8,len(names)).sum(2)/4
    delta=by[:,1]-by[:,0]
    return {k:dict(long_short=summarize(delta[:,i]),
                difference_from_sham=summarize(delta[:,i]-delta[:,0])) for k in (
                    "sham","address","value","key","query","parallel","joint","restore-h18","restore-parallel")
            for i in [names.index(k)]}


def main():
    C.setup();p=C.verify();f=A.load("freeze.json");led=A.load("confirm.json")
    a=A.load("confirm-natural.json");b=A.load("neutral-donors.json");g=A.load("confirm-gate.json")
    assert led["freeze"]==a["binding"]==B.sha("freeze.json") and f["donors"]==B.sha("neutral-donors.json")
    assert all(B.sha(k)==v for k,v in led.get("files",{}).items())
    rows=A.load(p["panels"]["confirm"]["file"])
    r=lexical(rows,a["rows"],b,g)
    panels={k:load(k) for k in sorted(led.get("files",{}))}
    proof={k:source_check(z,f["arm_names"]) for k,z in panels.items()}
    observed={x["id"]:x for x in a["rows"]}
    for name,z in panels.items():
        if "rounding_error" in z:
            e=(z["rounding_error"]<.05).all(axis=(1,2))
            expected=(z["drift"]<=C.N.SHAM_TOL)&(z["top"][:,0]==[observed[k]["top"] for k in z["ids"]])&e
            assert np.array_equal(expected,z["technical"]),name
    accepted=set(led.get("accepted",[]))
    geometry=paired_geometry(panels.values(),f["arm_names"],accepted)
    response=distance_response(panels.values(),f["arm_names"],accepted,rows)
    bindings=("plan.json","freeze.json","confirm.json","confirm-natural.json","neutral-donors.json","rounding-exclusions.json")
    A.save("native-recheck.json",dict(lexical=r,reads=proof,paired_geometry=geometry,distance_response=response,
                                      binding={k:B.sha(k) for k in bindings if (C.OUT/k).exists()}))
    print("Exact actual token pairs",r["exact_pairs"],"independently verified read shards",len(proof),flush=True)


if __name__=="__main__":main()
