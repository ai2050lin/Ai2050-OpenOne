"""Hold the original high-dose target fixed while balancing only L21H25."""

from collections import Counter
from itertools import product
from pathlib import Path
import sys
import time

import numpy as np

import kv_address_intervene as I
import kv_address_run as C
import kv_address_native as N
import kv_address_study as S


def freeze():
    assert not (C.OUT/"fixed-target-plan.json").exists()
    assert not (C.OUT/"interventions.json").exists()
    assert not list(C.OUT.glob("fixed-target-*.json"))
    C.save("fixed-target-plan.json",dict(
        time=S.mark(),forecast=C.digest(C.OUT/"case-forecasts.json"),source=C.digest(Path(__file__)),
        design="Before aggregate edits are available. Same original per-row per-head high target, same cap and directions; only the native L21H25 State/Note keys are balanced per arm.",
        prediction="Some original trio failures recover without decreasing target amplitude. Changing L21H25 can leave L21H19 cap violations. Compare to original frozen arm on the same families.",
        thresholds="All Phase2940 dose gates, >=153 complete families, P/Q interval ±0.25. Preserve all losses and K-only baseline shifts."))


def execute():
    import torch
    plan=C.load("fixed-target-plan.json")
    assert C.digest(Path(__file__))==plan["source"]
    assert C.digest(C.OUT/"case-forecasts.json")==plan["forecast"]
    p=S.verify()
    rows,cap=S.captured("confirm")
    with np.load(C.OUT/"case-forecasts.npz") as data:
        include=data["good_family"]
    behavior=C.load("behavior-confirm.json")["rows"]
    old=C.load("interventions.json")
    cfg=C.R.Base.Previous.load(C.OUT.with_name("kv_alpha_20261115_v2")/"plan.json")["hypothesis"]
    assert C.Base.collect_identity()==p["files"]
    manifest={}

    def run(task):
        start=time.monotonic()
        jobs=C.groups(rows)
        for j,(batch,obs) in enumerate(zip(jobs,C.groups(behavior),strict=True)):
            name=f"fixed-dose-{j:03d}.json"
            main=f"edited-{j:03d}.json"
            assert C.digest(C.OUT/main)==old["shards"][main]
            source=C.load(main)["weighted"]
            if not (C.OUT/name).exists():
                torch.cuda.empty_cache()
                r=N.Reader(task,batch,obs)
                eligible=torch.as_tensor(include[j*C.BATCH:j*C.BATCH+len(batch)],device="cuda",dtype=torch.bool)
                try:
                    r.initialize()
                    sham=r.replay(N.Arm("capture","capture"))
                    assert np.array_equal(sham["gap"].cpu().numpy(),cap["gap"][j*C.BATCH:j*C.BATCH+len(batch)])
                    r.reference=r.balance=sham
                    r.initialize(donor=True)
                    d={}
                    for a in (x for x in N.value_arms() if x.repair):
                        output=r.replay(a,usable=eligible)
                        d["sn"[a.slot],"donor" if a.signal else "rotate"]=dict(**I.compact_dose(output,cfg),score=output["scores"].cpu().tolist())
                    fail={tag:np.logical_or.reduce([v["fail"][tag] for v in d.values()])
                          for tag in next(iter(d.values()))["fail"]}
                    for signal in ("donor","rotate"):
                        s,n=(d[x,signal] for x in "sn")
                        fail["paired-"+signal]=I.norm(s["actual"]-n["actual"])>np.maximum(
                            cfg["actual_absolute_error_floor"],cfg["max_actual_paired_relative_error"]*s["n"])
                    target=np.min(np.stack([v["n"] for v in d.values()]),axis=0)
                    assert np.allclose(target,np.asarray(source["arms"]["original"]["target_min"]),atol=1e-6)
                    keep=eligible.cpu().numpy()
                    admitted=keep&~np.logical_or.reduce(list(fail.values())).any(1)
                    rr=[[[tag for tag,bad in fail.items() if bad[i,h]] for h in range(3)]
                        if keep[i] else [["ineligible"]]*3 for i in range(len(batch))]
                    C.save(name,dict(ids=[x["id"] for x in batch],include=keep.tolist(),admitted=admitted.tolist(),fail=rr,
                                     target=target.tolist(),original=source["arms"]["original"]["admitted"],
                                     scores={f"{slot}-{s}":v["score"] for (slot,s),v in d.items()},
                                     prereg=C.digest(C.OUT/"fixed-target-plan.json"),source_shard=old["shards"][main]))
                finally:
                    r.close()
            manifest[name]=C.digest(C.OUT/name)
            C.R.Base.Previous.stage_log("q4","same-target-key-balance",j+1,len(jobs),start)
    C.with_task(run)
    C.save("fixed-dose-index.json",dict(shards=manifest,freeze=C.digest(C.OUT/"fixed-target-plan.json")))


def summarize():
    from kv_address_audit import family_stats, margin
    p=C.load("fixed-dose-index.json")
    assert p["freeze"]==C.digest(C.OUT/"fixed-target-plan.json")
    rows=C.load("material-confirm.json")
    by={}
    lookup={r["id"]:r for r in rows}
    examples={k:[] for k in ("rescued","lost","failed")}
    count=Counter()
    fail=[Counter() for _ in range(3)]
    for name,h in sorted(p["shards"].items()):
        assert h==C.digest(C.OUT/name)
        b=C.load(name)
        assert b["prereg"]==p["freeze"]
        trials=[lookup[i] for i in b["ids"]]
        margin_at=lambda a:margin(b["scores"][a],trials)
        delta=margin_at("s-donor")-margin_at("s-rotate")-margin_at("n-donor")+margin_at("n-rotate")
        for i,r in enumerate(trials):
            if not b["include"][i]:
                continue
            o,n=b["original"][i],b["admitted"][i]
            by.setdefault(r["family"],[]).append((o,n,float(delta[i])))
            status=("same" if o else "rescued") if n else ("lost" if o else "failed")
            count[status]+=1
            for h,why in enumerate(b["fail"][i]):
                fail[h].update(why)
            if status in examples and len(examples[status])<12:
                examples[status].append(dict(id=r["id"],failed=b["fail"][i],margin_diff=float(delta[i])))
    assert all(len(v)==8 for v in by.values())
    old=[v for v in by.values() if all(x[0] for x in v)]
    new=[v for v in by.values() if all(x[1] for x in v)]
    mean=family_stats([np.mean([x[2] for x in v]) for v in new])
    C.save("fixed-dose-audit.json",dict(n_families=len(by),original_families=len(old),repaired_families=len(new),
                                       rescued_families=sum(not all(x[0] for x in v) for v in new),
                                       lost_families=sum(not all(x[1] for x in v) for v in old),
                                       views=dict(count),source=mean,fail_heads=[dict(c) for c in fail],
                                       examples=examples,coverage_pass=len(new)>=153,
                                       design=p["freeze"],index_sha=C.digest(C.OUT/"fixed-dose-index.json")))


if __name__=="__main__":
    C.configure()
    {"freeze":freeze,"run":execute,"audit":summarize}[sys.argv[1]]()
