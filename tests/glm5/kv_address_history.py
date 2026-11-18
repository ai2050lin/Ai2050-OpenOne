"""Reopen Phase 2940's raw dose arrays and assign failures by head and condition."""

from collections import Counter
import numpy as np

import kv_address_run as C
from own_state_audit import round_bf16


def norm(x):
    return np.linalg.norm(x.astype(np.float64),axis=-1)


def audit():
    cfg_dir=C.OUT.with_name("kv_alpha_20261115_v2")
    get=lambda s:C.R.Base.Previous.load(cfg_dir/s)
    cfg=get("plan.json")["hypothesis"]
    prev=get("audit-rows-polarity.json")["rows"]
    batches=get("execution.json")["panels"][0]["shards"]
    ids=[]
    table={dose:[] for dose in ("low","high")}
    for files in batches:
        assert C.digest(cfg_dir/files["json"])==files["sha"]
        assert C.digest(cfg_dir/files["npz"])==files["npz_sha"]
        blob=get(files["json"])
        ids.extend(blob["ids"])
        with np.load(cfg_dir/files["npz"]) as a:
            d=a["donor"]-a["state"]
            rotated=np.stack([np.roll(d[:,h],(13+7*l)%127+1,axis=-1) for h,(l,_) in enumerate(C.STOCK_HEADS)],axis=1)
            alpha=a["sham_alpha"]
            lo,hi=alpha.min(-1),alpha.max(-1)
            for dose in table:
                target=lo if dose=="low" else np.minimum(hi,2*lo)
                flags={}
                cap_sham=np.zeros(target.shape,bool)
                seen={}
                for mode in ("swap","null"):
                    dv=d if mode=="swap" else rotated
                    desired=dv*target[...,None]
                    n=norm(desired)
                    for si,slot in enumerate(("state","note")):
                        key=f"trio-{dose}-{slot}-{mode}"
                        s=np.asarray(blob["stats"][key],dtype=np.float32)
                        al=s[...,si]
                        g=s[...,3]
                        old=a[slot]
                        res=a[key]
                        calc=(round_bf16(old+g[...,None]*dv)-old)*al[...,None]
                        cos=np.sum(desired*res,-1)/np.maximum(n*norm(res),1e-12)
                        cap_sham |= target/np.maximum(alpha[...,si],1e-12)>2.05+2e-6
                        causes={
                            "small":n<cfg["norm_floor"],
                            "capped":target/np.maximum(al,1e-12)>2.05+2e-6,
                            "analytic":norm(calc-desired)>np.maximum(cfg["analytic_absolute_error_floor"],cfg["max_planned_relative_error"]*n),
                            "native":norm(res-desired)>np.maximum(cfg["actual_absolute_error_floor"],cfg["max_actual_relative_error"]*n),
                            "collapsed":norm(res)<cfg["min_actual_norm_fraction"]*n,
                            "direction":cos<cfg["min_actual_cosine"]
                        }
                        for tag,bad in causes.items():
                            flags[tag]=bad|flags.get(tag,False)
                        seen[slot,mode]=res
                    flags["paired-"+mode]=norm(seen["state",mode]-seen["note",mode])>np.maximum(
                        cfg["actual_absolute_error_floor"],cfg["max_actual_paired_relative_error"]*n)
                for j,k in enumerate(blob["ids"]):
                    table[dose].append(dict(id=k,heads=[sorted(key for key,m in flags.items() if m[j,h])
                                                       for h in range(3)],capped_from_sham=cap_sham[j].tolist(),
                                            sham_alpha=alpha[j].tolist()))
    assert ids==[r["id"] for r in prev]
    out={}
    for dose,rows in table.items():
        for x,y in zip(rows,prev,strict=True):
            assert set(t for h in x["heads"] for t in h)==set(y["doses"]["trio"][dose]["reasons"])
        failing=[r for r in rows if any(r["heads"])]
        out[dose]=dict(failed=len(failing),
                       by_head=[dict(Counter(x for r in failing for x in r["heads"][h])) for h in range(3)],
                       only_head=[sum(bool(r["heads"][h]) and sum(map(bool,r["heads"]))==1 for r in failing) for h in range(3)],
                       cap_from_sham_count=[sum(r["capped_from_sham"][h] for r in rows) for h in range(3)],rows=rows)
    C.save("phase2940-failure-by-head.json",dict(doses=out,plan_sha=C.digest(cfg_dir/"plan.json"),
                                               archive_audit_sha=C.digest(cfg_dir/"audit.json")))


if __name__=="__main__":
    C.configure()
    audit()
