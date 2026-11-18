"""Recompute the prior crossing ledger and the frozen source/time/subset contrasts."""
from collections import defaultdict
import sys
import json
import numpy as np
from scipy.stats import binomtest
from transformers import AutoTokenizer,GenerationConfig
import kv_route_run as R
from kv_route_native import SUBSETS,discover
from own_state_audit import official_snapshot,f32,round_bf16


def margin(scores,row):
    i=int(row["alt_code"]=="Q")
    return scores[i]-scores[1-i]


def previous():
    old=R.M.OLD
    load=R.Base.Previous.load
    protocol=load(old/"confirmation.json")
    shards=next(s["shards"] for s in protocol["suites"] if s["suite"]=="gap")
    rows={r["id"]:r for r in load(old/"materials-gap.json")}
    found=[]
    for shard in shards:
        path=old/shard["name"]
        assert R.sha(path)==shard["sha"]
        src=load(path)
        for i,key in enumerate(src["ids"]):
            r=rows[key]
            x,y,both=(margin(src["paths"][k]["scores"][i],r) for k in
                      ("natural","values-selected","values-all_swap"))
            output=src["free"]["values-selected"][i]
            flip=(output["code"]==r["alt_code"] and output["eos"]
                  and output["state"]==r["state"] and output["aux"]==r["aux"])
            found.append(dict(id=key,original=x,triad=y,whole=both,lift=y-x,flip=flip))
    assert len(found)==384 and sum(x["flip"] for x in found)==19
    return dict(n=384,flips=19,rises=sum(r["lift"]>0 for r in found),
                sign_disagreements=sum(r["flip"]!=(r["triad"]>0) for r in found),
                medians={key:float(np.median([r[key] for r in found])) for key in ("original","lift","triad","whole")},rows=found)


def test(rows,d,threshold):
    family=defaultdict(list)
    cells=defaultdict(list)
    for r,v in zip(rows,d,strict=True):
        family[r["family"]].append(v)
        cells[(r["update"],r["swap"],r.get("second_keeps"),r.get("voice"))].append(v)
    v=np.array([np.mean(x) for x in family.values()])
    assert len(v)==96 and all(len(x)==4 for x in family.values())
    pos=int((v>0).sum()); neg=int((v<0).sum())
    p=float(binomtest(pos,pos+neg,alternative="greater").pvalue) if pos+neg else 1.
    means=[dict(cell=k,mean=float(np.mean(x)),n=len(x)) for k,x in sorted(cells.items())]
    lo,hi=np.quantile(np.random.default_rng(2939).choice(v,size=(4000,len(v)),replace=True).mean(1),[.025,.975])
    i=int(np.argmin(d))
    return dict(mean=float(v.mean()),ci95=[float(lo),float(hi)],pos=pos,neg=neg,p=p,
                passed=bool(v.mean()>=threshold and p<=R.P and all(c["mean"]>=0 for c in means)),
                threshold=threshold,cells=means,worst=dict(id=rows[i]["id"],contrast=float(d[i])))


def audit_behavior(key,rows,obs,diagnostic=True):
    path=R.Base.ROOT/"models/hf"/R.Base.MODELS[key]
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    config=GenerationConfig.from_pretrained(path,local_files_only=True).eos_token_id
    eos=set(config if isinstance(config,list) else [config])
    letters={c:tok.encode(" "+c,add_special_tokens=False)[0] for c in ("P","Q")}
    for row,rec in zip(rows,obs,strict=True):
        assert row["id"]==rec["id"]
        assert R.Base.encode(tok,row)==(rec["input_ids"],rec["value_ids"])
        assert (row["state"],row["alt"])==((row["new"],row["old"]) if row["update"] else (row["old"],row["new"]))
        assert rec["text"]==tok.decode(rec["tokens"],skip_special_tokens=True)
        expected=R.parse(row,tok,rec["tokens"],eos,rec["value_ids"],letters)
        assert all(rec[k]==v for k,v in expected.items())
        if rec["joint"]:
            assert rec["note_index"]+1<rec["answer_index"]-1
    return tok,eos,letters


def native_sample(suite):
    with np.load(R.OUT/f"native-{suite}.npz") as d:
        shapes={k:d[k].shape for k in ("original","donor","note","hidden")}
        assert shapes["hidden"][1:]==(38,4,2560)
        assert all(shapes[k]==(36,4,8,128) for k in ("original","donor","note"))
        k,v,n=(f32(d[x]) for x in ("original","donor","note"))
        delta=v-k
        shift=np.stack([np.roll(x,(13+7*l)%127+1,axis=-1) for l,x in enumerate(delta)])
        a=np.linalg.norm(delta,axis=-1)
        for target,change in ((k,shift),(n,delta),(n,shift)):
            actual=np.linalg.norm(round_bf16(target+change)-target,axis=-1)
            assert np.max(abs(actual-a)/np.maximum(a,1e-8))<=.03
        assert np.isfinite(f32(d["hidden"])).all()
    return shapes


def suite_case(suite,entry,rows,behavior,tools,arms):
    tok,eos,letters=tools
    inputs={r["id"]:r for r in rows}
    natural=dict(zip(inputs,behavior,strict=True))
    jobs=[]
    for item in entry["shards"]:
        assert R.sha(R.OUT/item["name"])==item["sha"]
        jobs.append(R.read(item["name"]))
    assert [k for j in jobs for k in j["ids"]]==list(inputs)
    assert all(j["exact_code_logits"] and max(j["dose"].values())<=.03 for j in jobs)
    sampled=native_sample(suite)
    values={}
    heads=R.HEADS
    for spec in arms:
        name=spec["name"]
        expected=3 if spec["kind"].startswith("recover") else (len(spec["group"]) if spec["group"] is not None else 1152) if spec["kind"] else 0
        scores=[]
        for j in jobs:
            arm=j["paths"][name]
            for k,ab,n,count in zip(j["ids"],arm["scores"],arm["read_count"],arm["restore_count"],strict=True):
                good=natural[k]["joint"]
                assert n==expected*good
                assert count==(len(spec["group"]) if spec["kind"].startswith("recover") else 0)*good
                assert good or ab==[0.,0.]
                scores.append(margin(ab,inputs[k]))
        values[name]=np.array(scores)
    assert np.array_equal(values["recover-012"],values["natural"])
    measures={}
    base=values["natural"]
    x=values["swap-012"]-values["null-012"]
    dose=lambda loc,time: values[f"swap-{loc}-{time}"]-values[f"null-{loc}-{time}"]
    for label,d,t in (
        ("source",x-dose("note","answer"),.25),("clock",x-dose("state","early"),.25),
        ("content",x,.50),("remaining",values["remaining"]-base,2.),
        ("whole_content",values["whole"]-values["whole_null"],.50)):
        measures[label]=test(rows,d,t)
    subsets={}
    for s in SUBSETS:
        change=test(rows,values[f"swap-{s}"]-values[f"null-{s}"],.20)
        remove=test(rows,values[f"zero-{s}"]-values[f"zero_null-{s}"],.25)
        restore=test(rows,values[f"recover_null-{s}"]-values[f"recover-{s}"],.25)
        gain=float(np.mean(values["zero-012"]-values[f"recover-{s}"]))
        subsets[s]=dict(swap=change,remove=remove,restore=restore,gain=gain,
                        controlled_pair=bool(remove["passed"] and restore["passed"] and gain>=.25))
    available=[j for j in jobs if "unavailable" not in j["free"]]
    free={}
    for arm in available[0]["free"] if available else ():
        outcomes=[]
        for j in available:
            outcomes.extend(zip(j["ids"],j["free"][arm],strict=True))
        flips=[]
        for k,r in outcomes:
            row=inputs[k]
            expected=R.parse(row,tok,r["tokens"],eos,natural[k]["value_ids"],letters)
            assert expected==r
            flips.append(r["eos"] and r["code"]==row["alt_code"] and r["state"]==row["state"] and r["notes"]==row["notes"])
        free[arm]=dict(n=len(outcomes),flips=sum(flips),examples=[dict(id=k,text=r["text"]) for (k,r),ok in zip(outcomes,flips) if not ok][:5])
    att={}
    for l,h in heads:
        for t in ("early","answer"):
            name=f"{l}.{h}-{t}"
            a=np.asarray([row for j in jobs for row in j["attention"][name]])
            att[name]=dict(state_median=float(np.median(a[:,0])),note_median=float(np.median(a[:,1])))
    interaction=values["whole"]-values["remaining"]-values["swap-012"]+base
    return dict(suite=suite,status="complete",n=len(rows),native_sample=sampled,measures=measures,subsets=subsets,
                field_median_margins={k:float(np.median(values[k])) for k in ("natural","swap-012","remaining","whole")},
                interaction_mean=float(interaction.mean()),free=free,attention=att,
                equal_dose_error=max(max(j["dose"].values()) for j in jobs),exact_batches=len(jobs))


def main():
    R.setup()
    R.check()
    doc=R.read("confirmation.json")
    chosen=R.read("selection.json")
    assert doc["selection_sha"]==R.sha(R.OUT/"selection.json")
    outcome=[]
    behavior={}
    for stage in doc["ledger"]:
        suite=stage["suite"]
        rows=R.read(f"materials-{suite}.json")
        d=R.read(f"behavior-{suite}.json")
        a=audit_behavior("q4",rows,d["rows"])
        cells=json.loads(json.dumps(R.gate(rows,d["rows"])))
        assert cells==d["cells"]
        behavior[suite]=cells
        if suite=="discovery" and stage["status"]=="complete":
            assert all(R.sha(R.OUT/x["name"])==x["sha"] for x in stage["shards"])
            reconstructed=discover([R.read(x["name"]) for x in stage["shards"]],rows)
            assert json.loads(json.dumps(reconstructed))=={k:chosen[k] for k in reconstructed}
        if suite!="discovery":
            outcome.append(suite_case(suite,stage,rows,d["rows"],a,doc["arms"]) if stage["status"]=="complete"
                           else dict(suite=suite,status=stage["status"]))
        print("CPU audit",suite,len(rows),flush=True)
    refs={}
    for key in ("q14","glm"):
        if (R.OUT/f"reference-{key}.json").exists():
            ref=R.read(f"reference-{key}.json")
            audit_behavior(key,R.read("materials-discovery.json")[:4],ref["rows"])
            refs[key]=ref["cells"]
    snapshot=official_snapshot("q4",R.read("q4-model.json")["files"])
    R.write("audit.json",dict(run=R.RUN,snapshot=snapshot,retrospective=previous(),selection=chosen,
                             behavior=behavior,holdout=outcome,references=refs,alpha=R.P,
                             qualification="Same-node CUDA + CPU audit, independent GPU rerun not done"))


if __name__=="__main__":
    main()
