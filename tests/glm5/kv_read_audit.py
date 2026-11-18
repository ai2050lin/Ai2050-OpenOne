"""Independent CPU checks and frozen comparisons for the KV reading study."""
from collections import defaultdict
import re
import time

import numpy as np
from scipy.stats import binomtest
from transformers import AutoTokenizer

import kv_read_run as R
from kv_read_experiment import FIELDS, TIMES, arm_list, mean_margin
from own_state_audit import f32, round_bf16, official_snapshot


def truth(row):
    u=row["update"]
    state=(row["old"],row["new"])[u]
    other=(row["new"],row["old"])[u]
    codes=("Q","P") if row["swap"] else ("P","Q")
    assert row["state"]==state and row["alt"]==other
    assert row["code"]==codes[u] and row["alt_code"]==codes[1-u]
    assert f"{row['old']} -> {codes[0]}\n{row['new']} -> {codes[1]}" in row["text"]
    if row["suite"]=="multi":
        text=row["text"]
        first=re.search(r"First update: replace the current label with ([a-z]+)\.",text)[1]
        second=re.search(r"Second update: (.*)\.",text)[1]
        keep=second.casefold().startswith("keep")
        final=first if keep else second.split()[-1]
        assert final==state and row["second_keeps"]==keep


def check_tokens(row,out,tok,eos,letters):
    tokens=out["tokens"]
    assert tokens and not any(k in eos for k in tokens[:-1])
    actual=tok.decode(tokens,skip_special_tokens=True)
    assert actual==out["text"] and out["eos"]==(tokens[-1] in eos)
    lines=actual.strip().splitlines()
    actual_form=False
    state,code,aux=None,None,None
    if len(lines)==(3 if row["suite"]=="gap" else 2):
        s=lines[0].strip()
        n,_,letter=lines[-1].partition(":")
        condition=True
        if row["suite"]=="gap":
            h,_,word=lines[1].partition(":")
            condition=h.strip()=="Note" and re.fullmatch("[a-z]+",word.strip()) is not None
        actual_form=(re.fullmatch("[a-z]+",s) is not None and n.strip()=="Code"
                     and letter.strip() in ("P","Q") and condition)
        if actual_form:
            state=s
            code=letter.strip()
            aux=word.strip() if row["suite"]=="gap" else None
    assert (state,code,aux)==(out["state"],out["code"],out["aux"])
    site=out["code_index"]
    if site is not None:
        assert site>=3 and tokens[site] in letters.values()
        assert tok.decode(tokens[:site],skip_special_tokens=True).strip().endswith("Code:")
    assert out["joint"]==bool(out["site"] and out["eos"] and state==row["state"] and code==row["code"] and aux==row["aux"])


def check_behavior(rows,out,tok,eos,letters):
    for row,obs in zip(rows,out,strict=True):
        truth(row)
        ids,values=R.Base.encode(tok,row)
        assert obs["id"]==row["id"] and obs["input_ids"]==ids and obs["value_ids"]==values
        assert obs["site"]==bool(obs["code_index"] is not None and obs["tokens"][0] in values.values())
        check_tokens(row,obs,tok,eos,letters)


def paired(rows,differences,threshold):
    families=defaultdict(list)
    factors=defaultdict(list)
    for row,d in zip(rows,differences,strict=True):
        families[row["family"]].append(d)
        factors[(row["update"],row["swap"],row["second_keeps"])].append(d)
    assert len(families)==96 and all(len(v)==4 for v in families.values())
    family=np.asarray([np.mean(v) for v in families.values()])
    win,loss=int((family>0).sum()),int((family<0).sum())
    p=float(binomtest(win,win+loss,p=.5,alternative="greater").pvalue) if win+loss else 1.
    draws=np.random.default_rng(20261115).choice(family,size=(4000,len(family)),replace=True).mean(1)
    cells=[dict(cell=list(k),n=len(v),mean=float(np.mean(v))) for k,v in sorted(factors.items())]
    estimate=float(family.mean())
    return dict(mean=estimate,ci95=np.quantile(draws,[.025,.975]).tolist(),n=96,positive=win,negative=loss,
                ties=96-win-loss,p=p,threshold=threshold,cells=cells,
                passed=(estimate>=threshold and p<=R.P and all(x["mean"]>=0 for x in cells)))


def sample_native(suite):
    record=R.OUT/f"native-{suite}.npz"
    with np.load(record) as blob:
        assert len(blob["ids"])==4
        h=f32(blob["field"])
        assert h.shape[1:]==(38,4,2560) and np.isfinite(h).all()
        measurements={}
        for field in FIELDS:
            bits=blob[field]
            assert bits.shape==(2,36,4,8,128)
            native,donor=f32(bits)
            assert np.isfinite(native).all() and np.isfinite(donor).all()
            worst=0.
            for change in (donor,np.zeros_like(donor)):
                difference=change-native
                rolled=np.stack([np.roll(d,shift=(11+17*l)%127+1,axis=-1) for l,d in enumerate(difference)])
                actual=round_bf16(native+rolled)-native
                a=np.linalg.norm(difference,axis=-1)
                b=np.linalg.norm(actual,axis=-1)
                worst=max(worst,float((np.abs(a-b)/np.maximum(a,1e-12)).max()))
            assert worst<=.03
            measurements[field]=worst
    return dict(sha=R.sha(record),field_shape=list(h.shape),maximum_sample_norm_error=measurements)


def discovery_audit(selection):
    if selection["status"]!="frozen":
        return dict(status=selection["status"])
    assert R.sha(R.OUT/"discovery-results.json")==selection["discovery_sha"]
    doc=R.read("discovery-results.json")
    rows=R.read("materials-discovery.json")
    assert doc["ids"]==[r["id"] for r in rows]
    base=mean_margin(doc["baseline"],rows)
    discovered={}
    for field in FIELDS:
        data=[a for a in doc["arms"] if a["field"]==field]
        assert len(data)==113
        assert all(abs(a["delta"]-(mean_margin(a["raw"],rows)-base))<1e-8 for a in data)
        layers=sorted([a for a in data if a["name"]=="layer"],key=lambda z:(-z["delta"],z["points"][0][0]))
        chosen=[a["points"][0][0] for a in layers[:2]]
        heads=sorted([a for a in data if a["name"]=="head"],key=lambda z:(-z["delta"],*z["points"][0][:2]))
        points=[]
        for h in heads[:3]:
            l,q,_=h["points"][0]
            relevant=[a for a in data if a["name"]=="time" and a["points"][0][:2]==[l,q]]
            best=min(relevant,key=lambda z:(-z["delta"],TIMES.index(z["points"][0][2])))
            points.append(best["points"][0])
        assert selection["choices"][field]==dict(layers=chosen,points=points)
        discovered[field]=dict(layers=[dict(layer=x["points"][0][0],delta=x["delta"]) for x in layers],
                               broad_clock={a["name"]:a["delta"] for a in data if a["name"].startswith("all")},
                               points=points,heads=[dict(point=x["points"][0],delta=x["delta"]) for x in heads])
    return dict(status="recomputed discovery selection",by_field=discovered,native=sample_native("discovery"))


def audit_suite(suite,entry,rows,behavior,tokenizer,eos,letters,selection):
    serial=[]
    for s in entry["shards"]:
        assert R.sha(R.OUT/s["name"])==s["sha"]
        serial.append(R.read(s["name"]))
    ids=[i for p in serial for i in p["ids"]]
    assert ids==[r["id"] for r in rows]
    assert all(p["replay_exact"] and len(p["prefill_hashes"])==36 and
               max(e for x in p["norm"].values() for e in x.values())<=.03 for p in serial)
    assert [a for p in serial for a in p["sites"]]==[r["code_index"] for r in behavior]
    measurements={}
    natural_ok=np.asarray([o["joint"] for o in behavior],dtype=bool)
    truth_idx=np.asarray([int(r["code"]=="Q") for r in rows])
    assert len(truth_idx)==384
    format_stats={}
    for arm in serial[0]["paths"]:
        values=np.concatenate([np.asarray(p["paths"][arm]["scores"],dtype=float) for p in serial])
        assert values.shape==(384,2) and np.isfinite(values).all()
        assert np.all(values[~natural_ok]==0)
        top=[i for p in serial for i in p["paths"][arm]["top"]]
        diff=values[np.arange(384),1-truth_idx]-values[np.arange(384),truth_idx]
        measurements[arm]=diff
        format_stats[arm]=dict(mean_margin=float(diff.mean()),
                               exact_alternate_top=sum(int(k==letters[r["alt_code"]]) for k,r in zip(top,rows,strict=True)))
    free={}
    for arm in serial[0]["free"]:
        output=[o for p in serial for o in p["free"][arm]]
        for obs,row,base in zip(output,rows,behavior,strict=True):
            check_tokens(row,obs,tokenizer,eos,letters)
            assert obs["tokens"][:2]==base["tokens"][:2]
        free[arm]=dict(alternate=sum(o["eos"] and o["code"]==r["alt_code"] and o["state"]==r["state"] and o["aux"]==r["aux"]
                                     for r,o,b in zip(rows,output,behavior,strict=True) if b["joint"]),
                       unchanged=sum(o["tokens"]==b["tokens"] for o,b in zip(output,behavior,strict=True)),
                       correct_note=sum(o["aux"]==r["aux"] for o,r in zip(output,rows,strict=True)),
                       failure=[dict(id=r["id"],actual=o["text"],expected=r["alt_code"]) for r,o,b in zip(rows,output,behavior,strict=True)
                                if b["joint"] and not (o["eos"] and o["code"]==r["alt_code"] and o["state"]==r["state"] and o["aux"]==r["aux"])][:10])
    assert free["natural"]["unchanged"]==384

    def contrast(field,a,b,threshold):
        x=measurements[field+"-"+a]
        y=measurements[field+"-"+b]
        difference=np.where(natural_ok,x-y,0.)
        summary=paired(rows,difference,threshold)
        ix=int(np.argmin(difference))
        summary["least_effect"]=dict(id=rows[ix]["id"],difference=float(difference[ix]),
                                     edited=float(x[ix]),control=float(y[ix]))
        return summary

    findings={}
    for field in FIELDS:
        arm=(lambda a,b,threshold:contrast(field,a,b,threshold))
        broad=arm("all_swap","all_scramble",.75)
        sparse=arm("selected","selected_scramble",.50)
        timing=arm("selected","other_time",.25)
        removal=arm("selected_zero","selected_zero_null",.50)
        rescue=arm("all_zero","restore",.50)
        rescue_specific=arm("restore_matched","restore",.25)
        individual=[]
        for i,point in enumerate(selection[field]["points"]):
            own=arm(f"head{i}",f"head{i}_null",.25)
            at=arm(f"head{i}",f"head{i}_other",.25)
            individual.append(dict(point=point,specific=own,clock=at,
                                   intervention_grade=own["passed"],timing_grade=own["passed"] and at["passed"]))
        findings[field]=dict(broad=broad,sparse=sparse,timing=timing,removal=removal,rescue=rescue,
                             rescue_specific=rescue_specific,individual=individual,
                             local_and_timed=sparse["passed"] and timing["passed"],
                             complete_rescue=removal["passed"] and rescue["passed"] and rescue_specific["passed"])
    return dict(suite=suite,status="complete",n=384,natural_good=int(natural_ok.sum()),
                structural=dict(code_indices={str(i):sum(o["code_index"]==i for o in behavior)
                                              for i in sorted({o["code_index"] for o in behavior if o["code_index"] is not None})},
                                exact_batches=len(serial),sampled_native=sample_native(suite)),
                norms={f:max(max(p["norm"][f].values()) for p in serial) for f in FIELDS},findings=findings,
                fixed=format_stats,free=free)


def main():
    R.setup()
    R.plan()
    from transformers import GenerationConfig
    q4=R.Base.ROOT/"models/hf"/R.Base.MODELS["q4"]
    tok=AutoTokenizer.from_pretrained(q4,local_files_only=True)
    eos_value=GenerationConfig.from_pretrained(q4,local_files_only=True).eos_token_id
    eos=set(eos_value if isinstance(eos_value,list) else [eos_value])
    letters={x:tok.encode(" "+x,add_special_tokens=False)[0] for x in ("P","Q")}
    stats=[]
    done=[]
    refs={}
    plan=R.read("confirmation.json")
    selection=R.read("selection.json")
    from json import dumps, loads
    if selection["status"]=="frozen":
        assert plan["arms"]==loads(dumps([x.__dict__ for x in arm_list(selection["choices"])]))
        assert plan["selection_sha"]==R.sha(R.OUT/"selection.json")
    found=discovery_audit(selection)
    snapshots={}
    for key,folder in R.Base.MODELS.items():
        p=R.OUT/f"{key}-model.json"
        if p.exists():
            ident=R.read(p.name)
            snapshots[key]=official_snapshot(key,ident["files"])
    for suite in R.Material.SUITES:
        if not (R.OUT/f"q4-{suite}-behavior.json").exists():
            continue
        rows=R.read(f"materials-{suite}.json")
        document=R.read(f"q4-{suite}-behavior.json")
        behavior=document["rows"]
        start=time.monotonic()
        check_behavior(rows,behavior,tok,eos,letters)
        assert document["cells"]==R.gate(rows,behavior)
        stats.append(dict(suite=suite,cells=document["cells"],passed=document["passed"],
                          failures=[dict(id=r["id"],expected=[r["state"],r["aux"],r["code"]],raw=o["text"])
                                    for r,o in zip(rows,behavior,strict=True) if not o["joint"]]))
        if suite!="discovery":
            e=next(s for s in plan["suites"] if s["suite"]==suite)
            done.append(audit_suite(suite,e,rows,behavior,tok,eos,letters,selection["choices"])
                        if e["status"]=="complete" else dict(suite=suite,status=e["status"]))
        print("CPU audit",suite,len(rows),f"{time.monotonic()-start:.1f}s",flush=True)
    for key in ("q14","glm"):
        if (R.OUT/f"{key}-reference.json").exists():
            path=R.Base.ROOT/"models/hf"/R.Base.MODELS[key]
            tk=AutoTokenizer.from_pretrained(path,local_files_only=True)
            e=GenerationConfig.from_pretrained(path,local_files_only=True).eos_token_id
            ls={x:tk.encode(" "+x,add_special_tokens=False)[0] for x in ("P","Q")}
            docs=R.read(f"{key}-reference.json")["suites"]
            for suite,x in docs.items():
                short=[r for r in R.read(f"materials-{suite}.json") if r["index"]<2]
                check_behavior(short,x["rows"],tk,set(e if isinstance(e,list) else [e]),ls)
                assert x["cells"]==R.gate(short,x["rows"])
            refs[key]={k:dict(cells=v["cells"],examples=[r["text"] for r in v["rows"][:4]]) for k,v in docs.items()}
    R.write("audit.json",dict(run_id=R.RUN,snapshots=snapshots,selection=selection,discovery=found,behavior=stats,confirmation=done,
                             reference=refs,alpha=R.P,qualification="CPU audit and same-node replay, no independent GPU reproduction"))


if __name__=="__main__":
    main()
