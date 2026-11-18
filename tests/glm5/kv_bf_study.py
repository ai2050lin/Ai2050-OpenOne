"""Frozen comparison, shared dose accounting, and the serial 4B driver."""

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

import kv_bf_run as C
import kv_bf_q4 as Q
import kv_relay_study as RelayStudy


A=C.A
R=A.R
SOURCES=tuple(f"tests/glm5/{n}.py" for n in ("kv_bf_run","kv_bf_q4","kv_bf_study","kv_relay_native","kv_relay_study",
    "kv_address_run","kv_address_native","kv_address_materials","kv_alpha_run","condition_chain_run","own_state_run",
    "kv_route_run","source_binding_resources"))+("deploy/storage_policy.py","deploy/storage_policy.json","shared/dist_protocol/mechanism_trace.py")


def sha(name):
    return A.digest(C.OUT/name)


def write_np(name,record):
    np.savez_compressed(R.Base.Previous.check_write(C.OUT/name),**record)
    return sha(name)


def model_identity(model):
    folder=C.ROOT/"models/hf"/R.Base.MODELS[model]
    paths={p.name:p for p in sorted(folder.iterdir()) if p.suffix in (".safetensors",".json",".txt",".model") and p.is_file()}
    import hashlib
    import json
    snap={p:dict(sha256=A.digest(paths[p]),size=paths[p].stat().st_size) for p in paths}
    common={k:v for k,v in snap.items() if k.endswith((".safetensors",".model"))
            or k.startswith(("vocab","tokenizer")) or k in ("config.json","generation_config.json","merges.txt","special_tokens_map.json")}
    return dict(files=snap,snapshot=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest())


def arms_summary(records,results,cfg):
    good=records["good"]
    for name,arms in results.items():
        assert len(arms)==4
        d=[x["dose"] for x in arms]
        checks=[RelayStudy.dose_checks(x,cfg,good) for x in d]
        flags=np.stack([v[1] for v in checks],1)
        pair=[]
        for i,j in ((0,2),(1,3)):
            diff=np.linalg.norm((d[i]["actual"]-d[j]["actual"]).astype(np.float64),axis=-1)
            n=np.maximum(checks[i][2],checks[j][2])
            pair.append(diff>np.maximum(cfg["actual_absolute_error_floor"],cfg["max_actual_paired_relative_error"]*n))
        pair=np.stack(pair,1)
        records.update({
            name+"-scores":np.stack([x["scores"] for x in arms],1),
            name+"-alpha":np.stack([x["alpha"] for x in arms],1),
            name+"-flags":flags,name+"-pair":pair,
            name+"-gain":np.stack([x["gain"] for x in d],1),
            name+"-want":np.stack([x[2] for x in checks],1),
            name+"-head25-analytic":np.stack([x["analytic"][:,1] for x in d],1),
            name+"-head25-actual":np.stack([x["actual"][:,1] for x in d],1),
            name+"-effect-norm":np.stack([x[3] for x in checks],1)
        })
    if "native" in results:
        # All precision arms use the same target and natural Q, down to the float values.
        for name in Q.METHODS:
            if name not in results:
                continue
            for suf in ("-want","-gain","-alpha"):
                assert np.array_equal(records[name+suf],records["native"+suf]),(name,suf)
    return records


def engine_pilot(task):
    rows=A.load("pilot-material.json")[:16]
    obs=A.load("pilot-behavior.json")["rows"][:16]
    good=np.array([r["joint"] for r in obs])
    cfg=prior_rules()
    rec=arms_summary(*Q.run_batch(task,rows,obs,good),cfg)
    A.save("engine-pilot.json",dict(n=int(good.sum()),keys=sorted(rec),methods=Q.METHODS,
        invariants="common requested target, common gain, unchanged attention probabilities"))


def prior_rules():
    return R.Base.Previous.load(C.OUT.with_name("kv_alpha_20261115_v2")/"plan.json")["hypothesis"]


def selected(rows,n):
    families=list(dict.fromkeys(r["family"] for r in rows))
    return [families[j] for j in np.linspace(0,len(families)-1,n,dtype=int)]


def plan():
    assert not (C.OUT/"plan.json").exists()
    assert A.load("pilot-behavior.json")["passed"]
    assert A.load("engine-pilot.json")["n"]>0
    tok=A.Base.tok()
    prev=C.OUT.with_name("kv_relay_20261115_v1")
    ancient=set()
    for f in ("material-discover.json","material-confirm.json"):
        old=R.Base.Previous.load(prev/f)
        for row in old:
            ancient.update((row["state"],row["note"],row["alt"]))
    materials={}
    for stage in ("discover","confirm"):
        rows=C.material(stage,tok)
        for row in rows:
            A.encoded(tok,row)
        words={r[k] for r in rows for k in ("state","note","alt")}
        if stage=="confirm":
            assert not words&ancient,sorted(words&ancient)
        name=f"material-{stage}.json"
        A.save(name,rows)
        materials[stage]=dict(name=name,sha=sha(name),n=len(rows),families=len(rows)//8,
                              compare_families=selected(rows,16 if stage=="discover" else 48))
    four=model_identity("q4")
    p=R.Base.Previous.load(prev/"plan.json")
    assert p["snapshot"]==four["snapshot"]
    pw=A.Base.check_phase(.085,192,p_d=.10,alpha=.025)
    A.save("plan.json",dict(run=C.RUN,stamp=C.stamp(),materials=materials,model_q4=four,power=pw,
        sources={k:A.digest(C.ROOT/k) for k in SOURCES},runtime=A.Base.versions(),
        provenance=dict(**A.Base.git_identity(),model_path=str(A.Base.MODEL),model_revision=four["snapshot"],seed=C.SEED),
        hypothesis="After Q recovery most residual H25 failures arise from rounding V+delta and the selected head's output BF16 boundary.",
        conditions=[["State","rotate"],["State","donor"],["Note","rotate"],["Note","donor"]],
        behavioral="4B greedy actual prefix and P/Q+State+Note+EOS. >=90% each cell, use whole eight-view families.",
        modes={"native":"Unchanged BF16 input and aggregation.",
               "once":"Keep delta in FP32 through attention aggregation, cast the modified head output to BF16 immediately before W_O.",
               "late-round":"Explicitly reintroduce BF16 input rounding, then add the selected FP32 head contribution through W_O before output cast.",
               "late":"Keep only the selected delta in FP32 through the original W_O, cast result before residual. Other routes BF16.",
               "dynamic":"Native BF16 with the two live queries, for model-specific cap reference."},
        discovery_selector="From once and late choose more complete discovery families, tie goes to once. No confirmation responses before selection.",
        comparisons={"q4":"Native controls and dose match at whole-family 153 gate. Source and two receivers fixed from Phase 2942.",
                     "q14":"Same text fixed-prefix panel of 48 precommitted families with identical generated-source lexical values where possible. Own discovery scan, adjacent layers and distinct KV groups. 14B reference does not use the 153 cutoff.",
                     "scope":"Natural behavior versus conditional code scores reported separately. No aligning head numbers across sizes."},
        dose=dict(cap=2.05,minimum_families=153,equivalence=.25,limits=prior_rules()),
        calibration={k:sha(k) for k in ("pilot-material.json","pilot-behavior.json","engine-pilot.json")}))


def verify():
    p=A.load("plan.json")
    assert all(sha(x["name"])==x["sha"] for x in p["materials"].values())
    assert all(A.digest(C.ROOT/k)==h for k,h in p["sources"].items())
    assert all(sha(n)==h for n,h in p["calibration"].items())
    assert p["runtime"]==A.Base.versions()
    return p


def good_families(rows,observed):
    ok=defaultdict(list)
    for r,o in zip(rows,observed,strict=True):
        ok[r["family"]].append(o["joint"])
    assert all(len(x)==8 for x in ok.values())
    return {k for k,v in ok.items() if all(v)}


def execute(stage):
    import torch
    p=verify()
    assert model_identity("q4")==p["model_q4"]
    if stage=="confirm":
        z=A.load("freeze-q4.json")
        assert z["plan"]==sha("plan.json") and z["discovery"]==sha("q4-discover.json")
    rows=A.load(p["materials"][stage]["name"])
    key=f"q4-behavior-{stage}.json"
    report=f"q4-{stage}.json"
    def work(task):
        if (C.OUT/key).exists():
            observed=A.load(key)
        else:
            observed=A.behavioral(task,rows,stage+"-behavior")
            A.save(key,observed)
        if not observed["passed"]:
            A.save(report,dict(reason="behavior-gate-failed",shards={},cells=observed["cells"],plan=sha("plan.json")))
            return
        admitted=good_families(rows,observed["rows"])
        start=time.monotonic()
        files={}
        jobs=A.groups(rows)
        for j,(batch,b) in enumerate(zip(jobs,A.groups(observed["rows"]),strict=True)):
            name=f"q4-{stage}-{j:03d}.npz"
            good=np.array([r["family"] in admitted for r in batch])
            if not good.any():
                continue
            if not (C.OUT/name).exists():
                torch.cuda.empty_cache()
                meta,runs=Q.run_batch(task,batch,b,good)
                result=arms_summary(meta,runs,p["dose"]["limits"])
                files[name]=write_np(name,result)
            else:
                with np.load(C.OUT/name) as d:
                    assert np.array_equal(d["ids"],[r["id"] for r in batch])
                files[name]=sha(name)
            R.Base.Previous.stage_log("q4",stage+"-numerics",j+1,len(jobs),start)
        A.save(report,dict(plan=sha("plan.json"),behavior=sha(key),shards=files,eligible_families=len(admitted),status="finished"))
    torch.set_float32_matmul_precision("highest")
    A.with_task(work)


def view_pass(z,method):
    return z["good"]&~z[method+"-flags"].any((1,2,3))&~z[method+"-pair"].any((1,2))


def load_data(ledger):
    files={}
    for name,digest in sorted(ledger["shards"].items()):
        assert sha(name)==digest
        with np.load(C.OUT/name) as z:
            for k in z.files:
                files.setdefault(k,[]).append(z[k])
    return {k:np.concatenate(v) for k,v in files.items()}


def frozen_method(data):
    family=defaultdict(list)
    for i,id in enumerate(data["ids"].tolist()):
        family[id.split("-")[0]].append(i)
    assert all(len(x)==8 for x in family.values())
    choices={}
    for k in ("native",*Q.METHODS[1:]):
        good=view_pass(data,k)
        choices[k]=dict(families=sum(good[v].all() for v in family.values()),views=int(good.sum()))
    choose=max(("once","late"),key=lambda k:(choices[k]["families"],k=="once"))
    return choose,choices


def freeze():
    p=verify()
    assert not (C.OUT/"q4-behavior-confirm.json").exists()
    d=A.load("q4-discover.json")
    assert d["plan"]==sha("plan.json") and d["status"]=="finished"
    name,tally=frozen_method(load_data(d))
    A.save("freeze-q4.json",dict(plan=sha("plan.json"),discovery=sha("q4-discover.json"),
                                primary=name,choices=tally,stamp=C.stamp(),models="q4 only"))


commands={"engine-pilot":lambda:A.with_task(engine_pilot),"plan":plan,"discover":lambda:execute("discover"),
          "freeze":freeze,"confirm":lambda:execute("confirm")}
