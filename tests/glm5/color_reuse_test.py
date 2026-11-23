"""Original Qwen3-14B BF16 color naming to comparison transfer. See frozen criteria."""
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile
import numpy as np

ROOT = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) if Path(__file__).name=="test.py" else Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tests/glm5"),str(ROOT)]
from color_reuse_material import (COLORS, ENTRY, GROUPS, LAYERS, MODEL, OUT, SEED, SIZES,
                                  digest, prepare, read, save, sha, sources)
from deploy.storage_policy import check_write
from source_binding_runtime import Runtime


class Native(Runtime):
    """Hook only the selected token of the genuine gated MLP's down projection."""
    def block(self,b):
        layer=super().block(b)
        self.block_num=b
        self.offset=0
        def hook(mod,args,out):
            n=len(out)
            part=self.batch[self.offset:self.offset+n]
            self.offset+=n
            pos=self.t.tensor([out.shape[1]-len(r['tokens'])+r['pos'] for r in part],device=out.device)
            idx=self.t.arange(n,device=out.device)
            native=args[0][idx,pos]
            if b in LAYERS:
                for i,r in enumerate(part):
                    if r.get("capture") or r.get("corpus"):
                        self.captures[r["id"],b]=native[i].view(self.t.uint16).cpu().numpy().copy()
            for i,r in enumerate(part):
                arm=r["arm"]
                if arm=="base" or r.get("block")!=b:
                    continue
                a=self.t.as_tensor(r["units"],device=out.device)
                other=self.t.as_tensor(r["ctrl"],device=out.device)
                x=native[i].float()
                target=mod.weight[:,a].float() @ x[a]
                control=mod.weight[:,other].float() @ x[other]
                if not float(control.norm()): raise ValueError("zero matched control")
                control*=target.norm()/control.norm()
                original=out[i,pos[i]].clone()
                subtract=control if arm=="ctrl" else target
                altered=(original.float()-subtract).to(original.dtype)
                r["edit_norm"]=float((original.float()-altered.float()).norm())
                if arm=="restore":
                    out[i,pos[i]]=altered
                    if not self.t.equal(out[i,pos[i]],altered): raise ValueError("cut write failed")
                    out[i,pos[i]]=original
                    if not self.t.equal(out[i,pos[i]],original): raise ValueError("restore not exact")
                else:
                    out[i,pos[i]]=altered
            return out
        layer.mlp.down_proj.register_forward_hook(hook)
        return layer


def val(bits):
    return (bits.astype(np.uint32)<<16).view(np.float32)


def score_items(model, variants, stream, state, stage, work, batch_size=128):
    import torch
    from source_binding_resources import resources
    answers={}
    jobs=[]
    for r in variants:
        base=dict(r,sub=(),tokens=r["tokens"])
        jobs.append(base)
        # Teacher-forced suffixes belong only to explicitly multi-token color answers.
        for prefix in sorted({tuple(ids[:i]) for ids in r["choices"] for i in range(1,len(ids))}):
            jobs.append(dict(base,sub=prefix,tokens=base['tokens']+list(prefix),capture=False))
    fields={}
    model.captures={}
    for lo in range(0,len(jobs),batch_size):
        batch=jobs[lo:lo+batch_size]
        model.batch=batch
        def callback(b,layer,x,y,part,length,offset=0,**kwargs):
            h=x if b==-1 else y
            for i,row in enumerate(part):
                if row.get("capture"):
                    key=row["id"]
                    if key not in fields:
                        fields[key]=np.empty((model.cfg.num_hidden_layers+1,model.dim),np.uint16)
                    fields[key][b+1]=h[i,length-len(row["tokens"])+row["pos"]].view(torch.uint16).cpu().numpy()
            if offset==0 and b in (-1,19,39):
                elapsed=time.monotonic()-state["start"]
                frac=lo+len(batch)*(b+1)/model.cfg.num_hidden_layers
                print(json.dumps(dict(stage=stage,done=round(frac,1),total=len(jobs),elapsed=round(elapsed,1),
                    stage_eta=round((time.monotonic()-stage_time)*(len(jobs)-frac)/frac,1) if frac else None,
                    **resources())),flush=True)
        if lo==0: stage_time=time.monotonic()
        check_write(work,32*2**20,operation="颜色复用采集")
        _,logits=model.forward(batch,callback)
        tops=logits.argmax(-1).tolist()
        logp=logits.log_softmax(-1)
        for i,row in enumerate(batch):
            key=(row["id"],row["arm"],row.get("block"),row.get("budget"))
            data=answers.setdefault(key,dict(parts={},row=row))
            prefix=row["sub"]
            relevant={seq[len(prefix)] for seq in row["choices"] if len(seq)>len(prefix) and tuple(seq[:len(prefix)])==prefix}
            data["parts"][prefix]={k:float(logp[i,k]) for k in relevant}
            if not prefix:
                data.update(top=tops[i],norm=row.get("edit_norm",0.))
        del logits,logp
        for key in list(answers):
            data=answers[key];row=data["row"]
            if any(tuple(ids[:k]) not in data["parts"] for ids in row["choices"] for k in range(len(ids))):
                continue
            scores=[sum(data["parts"][tuple(ids[:k])][token] for k,token in enumerate(ids)) for ids in row["choices"]]
            pick=row["answers"][int(np.argmax(scores))] if scores else None
            rec=dict(id=row["id"],stage=row["stage"],task=row["task"],item=row["item"],color=row["color"],
                category=row["category"],ref=row.get("ref"),refcolor=row.get("refcolor"),arm=row["arm"],
                block=row.get("block"),budget=row.get("budget"),edit_norm=data["norm"],logp=scores,
                answers=row["answers"],gold=row["gold"],pred=pick,top=data["top"],top_text=model.tok.decode([data["top"]]),
                correct=pick==row["gold"] if scores else None,
                top_first_correct=data["top"]==row["choices"][row["answers"].index(row["gold"])][0] if scores else None,
                cooccur=row.get("corpus_pair_count"))
            stream.write(json.dumps(rec,ensure_ascii=False)+"\n")
            state["records"].append(rec)
            del answers[key]
    if answers: raise ValueError("missing teacher-forced suffix scores")
    if model.captures or fields:
        payload={}
        if fields:
            ids=sorted(fields);payload["state_ids"]=np.asarray(ids);payload["hs_bf16"]=np.stack([fields[k] for k in ids])
        if model.captures:
            ids=sorted({key[0] for key in model.captures});payload["mlp_ids"]=np.array(ids)
            for b in LAYERS:
                payload["mlp_b"+str(b)]=np.stack([model.captures[i,b] for i in ids])
        path=work/f"native-{len(state['native']):04d}.npz"
        np.savez_compressed(path,**payload)
        state["native"].append(path.name)
    return model.captures


def ranked(rows,captures):
    ranking={}
    for b in LAYERS:
        x=np.stack([val(captures[r["id"],b]) for r in rows])
        c=np.array([r["color"] for r in rows])
        d=(x[c=="red"].mean(0)-x[c=="yellow"].mean(0))/(x.std(0)+1e-3)
        order=np.argsort(-abs(d),kind="stable")
        # Random independent units outside the maximum target set, same block.
        perm=np.random.default_rng(SEED+b).permutation(order[max(SIZES):])
        for size in SIZES:
            ranking[b,size]=dict(block=b,budget=size,units=order[:size].tolist(),ctrl=perm[:size].tolist(),
                                sign=np.sign(d[order[:size]]).tolist())
    return ranking


def arms(row,candidate,which):
    for arm in which:
        yield dict(row,arm=arm,**{k:candidate[k] for k in ("block","budget","units","ctrl")})


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def naming_effect(pairs):
    d=np.asarray([np.subtract(a["logp"],b["logp"]) for a,b in pairs])
    return dict(red_to_yellow=float(np.mean(d[:,1]-d[:,0])),yellow_over_blue=float(np.mean(d[:,1]-d[:,2])))


def summary(records,selected,calibration,gate,corpus,field,runtime):
    from collections import defaultdict
    natural=[r for r in records if r["arm"]=="base"]
    conf=[r for r in natural if r["stage"]=="confirm"]
    byarm={(r["id"],r["arm"]):r for r in records if r["stage"]=="confirm"}
    def rate(rs):
        return dict(n=len(rs),correct=sum(r["correct"] for r in rs),top_first=sum(r["top_first_correct"] for r in rs))
    naming={c:rate([r for r in conf if r["task"]=="name" and r["color"]==c]) for c in COLORS}
    cells={a+"/"+b:rate([r for r in conf if r["task"]=="compare" and r["color"]==a and r["refcolor"]==b]) for a in COLORS for b in COLORS}
    def margin(r):
        return r["logp"][0]-r["logp"][1]
    def collected(rs):
        d=defaultdict(list)
        for r in rs:
            cut,ctrl=(byarm[r["id"],a] for a in ("cut","ctrl"))
            d[r["item"]].append(margin(cut)-margin(ctrl))
        return {k:mean(v) for k,v in sorted(d.items())}
    def interval(byitem):
        v=np.asarray(list(byitem.values()))
        if not len(v): return dict(n=0,mean=None,ci95=[])
        rng=np.random.default_rng(SEED); boot=v[rng.integers(len(v),size=(5000,len(v)))].mean(1)
        return dict(n=len(v),mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    compares=[r for r in conf if r["task"]=="compare"]
    red=[r for r in compares if r["color"]=="red"]
    vectors={color:collected([r for r in red if r["refcolor"]==color]) for color in COLORS}
    effects={c:interval(v) for c,v in vectors.items()}
    effects["yellow_minus_blue"]=interval({k:v-vectors["blue"][k] for k,v in vectors["yellow"].items()})
    labels={r["item"]:r["correct"] for r in natural if r["stage"] in ("confirm","reference") and r["task"]=="name"}
    known=[r for r in compares if labels[r["item"]] and labels[r["ref"]]]
    category={c:rate([r for r in compares if r["category"]==c]) for c in sorted({r["category"] for r in compares})}
    same_color=[r for r in compares if r["gold"]=="same"]
    def lexical(r):
        return any(c in r["item"].lower() for c in COLORS)
    clean=[r for r in compares if not lexical(r)]
    subgroup={}
    for name, subset in (("recognized_red_pairs",[r for r in known if r["color"]=="red"]),
            ("no_color_substring",[r for r in red if not lexical(r)]),
            ("heldout_mineral_and_signal",[r for r in red if r["category"] in ("mineral","signal")])):
        subgroup[name]={c:interval(collected([r for r in subset if r["refcolor"]==c])) for c in COLORS}
    base_diff=rate(compares)["n"]-len(same_color)
    restored=sum(byarm[r["id"],"restore"]["logp"]==r["logp"] and byarm[r["id"],"restore"]["top"]==r["top"] for r in conf)
    matched=[]
    for r in conf:
        tar,ctrl=[byarm[r["id"],a]["edit_norm"] for a in ("cut","ctrl")]
        matched.append(tar>0 and abs(tar-ctrl)<=.05*tar)
    food=[r for r in conf if r["task"]=="food"]
    foodshift=mean([abs(margin(byarm[r["id"],"cut"])-margin(byarm[r["id"],"ctrl"])) for r in food])
    food_norm=rate(food)
    tails=defaultdict(list)
    for r in natural:
        if r["stage"]=="tail": tails[r["category"]+"/"+r["task"]].append(r)
    shifts=defaultdict(lambda:defaultdict(int))
    cell_shifts=defaultdict(list)
    for r in compares:
        key=r["color"]+"/"+r["refcolor"]; cut=byarm[r["id"],"cut"]
        if r["correct"] and not cut["correct"]: shifts[key]["new_error"]+=1
        if not r["correct"] and cut["correct"]: shifts[key]["fixed_error"]+=1
        cell_shifts[key].append(margin(cut)-margin(byarm[r["id"],"ctrl"]))
    behavior=all(v["correct"]>=.8*v["n"] for v in [*naming.values(),*cells.values()])
    select=effects["red"]["mean"]<=-.1 and effects["red"]["ci95"][1]<0 and effects["yellow"]["mean"]>=.1 and effects["yellow"]["ci95"][0]>0 and effects["yellow_minus_blue"]["mean"]>=.1 and effects["yellow_minus_blue"]["ci95"][0]>0
    checks=dict(calibration=gate,behavior=behavior,selective_transfer=select,exact_restore=restored==len(conf),
                matched=mean(matched)>=.95,food=foodshift<=.2)
    return dict(run_id=runtime,stage="color_reuse",checks=checks,supported=all(checks.values()),naming=naming,comparison=cells,
        color_delta=effects,cell_shift={k:mean(v) for k,v in cell_shifts.items()},new_fixed_errors=dict(shifts),
        candidate=dict(block=selected["block"],units=selected["budget"]),calibration=calibration,category=category,
        naming_known_pairs=rate(known),comparison_total=rate(compares),no_color_substring=rate(clean),subgroup=subgroup,
        food=food_norm,food_shift=foodshift,
        restored=[restored,len(conf)],matched_fraction=mean(matched),baselines=dict(always_different=[base_diff,len(compares)],
            word_cooccurrence=[sum(("same" if r["cooccur"] else "different")==r["gold"] for r in compares),len(compares)]),
        tails={k:rate(v) for k,v in tails.items()},corpus=corpus,natural_component=field,
        unique_questions=len({r["id"] for r in natural}),forwards_scored=len(records),
        limitations=["English only; 12 confirm first items per color; gross color ambiguity, explicit lexical clues and subtype overlap.",
                     "Native cut is limited to one MLP and last item token. A null cannot distinguish alternative sites from later rereading.",
                     "Tail frequency is from one English corpus. Full-sequence preferred answer and first-token top1 have separate meanings."])


def execute():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    work=Path(os.environ["AI2050_TEST_WORK_DIR"]); qp=Path(os.environ["AI2050_TEST_QUESTIONS"])
    plan,pre,review=[read(qp.parent/n) for n in ("protocol.json","preflight.json","review.json")]
    if pre["plan_sha256"]!=digest(plan) or not pre.get("preflight_id") or not review.get("server_run_id") or review["archive_state"]!="pending" or Path(review["scratch"]).resolve()!=work.resolve() or sha(qp)!=plan["dataset"]["questions_file_sha256"] or sha(Path(__file__))!=plan["entrypoint_sha256"] or digest(sources())!=plan["acquisition"]["code_sha256"]:
        raise ValueError("archive identity mismatch")
    state=dict(start=time.monotonic(),records=[],native=[])
    print(json.dumps(dict(step="verify BF16 snapshot",run=review["server_run_id"],done=0)),flush=True)
    if snapshot(MODEL)!=plan["acquisition"]["model_snapshot"]: raise ValueError("weight identity mismatch")
    pack=read(qp); allrows=pack["rows"]
    if digest([r["tokens"] for r in allrows])!=plan["acquisition"]["inputs_sha256"]: raise ValueError("token manifest mismatch")
    with zipfile.ZipFile(work/"research-scripts.zip","x",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sources():z.write(ROOT/path,arcname=path)
    with GpuLease(),torch.inference_mode(),(work/"case_results.jsonl").open("x",encoding="utf8",buffering=1) as stream:
        model=Native("qwen14")
        discover=[dict(r,arm="base",capture=True) for r in allrows if r["stage"]=="discovery"]
        captured=score_items(model,discover,stream,state,"discovery",work)
        rank=ranked(discover,captured)
        cal=[r for r in allrows if r["stage"]=="calibration"]
        jobs=[dict(r,arm="base",capture=True) for r in cal]
        for cand in rank.values():
            for r in cal: jobs.extend(arms(r,cand,("cut","ctrl")))
        score_items(model,jobs,stream,state,"calibration",work)
        rr={(r["id"],r["arm"],r["block"],r["budget"]):r for r in state["records"] if r["stage"]=="calibration"}
        cal_stats={}
        for b,k in rank:
            pairs=[(rr[r["id"],"cut",b,k],rr[r["id"],"ctrl",b,k]) for r in cal if r["color"]=="red"]
            baseline=[(rr[r["id"],"cut",b,k],rr[r["id"],"base",None,None]) for r in cal if r["color"]=="red"]
            cal_stats[b,k]={**naming_effect(pairs),"absolute":naming_effect(baseline)["red_to_yellow"]}
        choice=max(rank,key=lambda k:(cal_stats[k]["red_to_yellow"],-k[1],-k[0]))
        sel=rank[choice]; cs=cal_stats[choice]
        gate=cs["absolute"]>=.25 and cs["red_to_yellow"]>=.15 and cs["yellow_over_blue"]>=.10
        freeze=dict(selected=sel,calibration={f"{b}/{k}":v for (b,k),v in cal_stats.items()},gate=gate,
                    interpretation="If calibration fails, held-out transfer is diagnostic only.")
        save(work/"candidate.json",freeze)
        print(json.dumps(dict(step="candidate frozen before confirmation",block=choice[0],size=choice[1],gate=gate)),flush=True)
        for stage in ("confirm","reference","tail","corpus"):
            jobs=[]
            for row in allrows:
                if row["stage"]!=stage: continue
                jobs.append(dict(row,arm="base",capture=row["task"]=="name",corpus=stage=="corpus"))
                if stage=="confirm": jobs.extend(arms(row,sel,("cut","ctrl","restore")))
            data=score_items(model,jobs,stream,state,stage,work)
            if stage=="corpus":
                field={}
                for c in (*COLORS,"turquoise","ochre"):
                    for group in ("colorword","background"):
                        vals=[np.mean(val(data[r["id"],sel["block"]])[sel["units"]]*sel["sign"]) for r in jobs if r["color"]==c and r["category"]==group]
                        field[c+"/"+group]=dict(n=len(vals),mean_signed_activation=mean(vals))
    result=summary(state["records"],sel,cs,gate,pack["corpus"],field,review["server_run_id"])
    save(work/"conclusion.json",result)
    entries=[dict(path=n,role="native_field",retention="important") for n in state["native"]]
    entries.extend([dict(path="research-scripts.zip",role="script",retention="core"),
                    dict(path="candidate.json",role="evidence",retention="important")])
    save(work/"artifacts.json",dict(schema="research_test_files.v1",files=entries))
    print(json.dumps(dict(step="finished",checks=result["checks"],cases=result["forwards_scored"])),flush=True)


if __name__=="__main__":
    if sys.argv[1:]==["prepare"]:prepare()
    elif sys.argv[1:]==["launch"]:
        from clt_role_reread_desktop import launch
        raise SystemExit(launch("study",ENTRY,OUT))
    elif not sys.argv[1:]:execute()
    else:raise SystemExit("use prepare or launch")
