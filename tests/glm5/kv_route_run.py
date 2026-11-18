"""Freeze and replay an authored state-to-code study on native BF16 attention."""
from collections import defaultdict
from datetime import datetime
import importlib.metadata
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

from own_state_intervene import Field,state_digest
import own_state_run as Base
import kv_route_materials as M

OUT=Base.ROOT/"tests/glm5/result/kv_route_20261115_v1"
TEMP=Base.ROOT/"tests/glm5_temp/kv_route_20261115_v1"
RUN="KV-ROUTE-20261115-v1"
BATCH=32
CAP=19
P=.05/96
HEADS=((21,19),(21,25),(20,5))


def setup():
    Base.OUT,Base.TEMP,Base.RUN=OUT,TEMP,RUN
    Base.configure()


def read(name):
    return Base.Previous.load(OUT/name)


def sha(path):
    return Base.digest(path)


def write(name,data):
    assert Base.Previous.OUT==OUT
    return Base.Previous.write(name,data)


def freeze():
    assert not (OUT/"plan.json").exists()
    material={}
    for suite in M.SUITES:
        key=f"materials-{suite}.json"
        rows=M.generate(suite)
        write(key,rows)
        material[key]={"sha":sha(OUT/key),"n":len(rows)}
    sources=["kv_route_run.py","kv_route_native.py","kv_route_materials.py","own_state_run.py","own_state_intervene.py",
             "condition_chain_run.py","condition_chain_reference_stream.py"]
    write("plan.json",dict(run=RUN,date=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"),
                           material=material,sources={s:sha(Path(__file__).with_name(s)) for s in sources},
                           runtime={k:importlib.metadata.version(k) for k in ("torch","transformers","numpy","scipy","safetensors")},
                           batch=BATCH,cap=CAP,primary_checkpoint="original fp BF16 Qwen3-4B",
                           references="Qwen3-14B and GLM4 behavior only, one discovery family (four prompts), serial",
                           source_license="Original fictional records and previously exposed Phase2938 discovery",
                           previous={f:sha(M.OLD/f) for f in ("materials-gap.json","confirmation.json")},
                           heads=HEADS,discovery="16 exposed families, 64 views plus retrospective complete 384-view margin ledger",
                           holdout="96 families per suite x four views, frozen before discovery, order long/multi/role",
                           gate="all predeclared cells >=90% strict output plus single-token state, generated distractor and code sites",
                           policy="No post-hoc subset. Invalid natural rows zero-effect. Replay exact at all measured answer logits and complete original tokens.",
                           clocks=["one query immediately after first generated Note word","one query predicting code"],
                           source=["generated first state token","generated first Note word, already present at early query"],
                           contrasts=dict(alpha=P,min_source=.25,min_time=.25,min_content=.50,min_complement=2.,
                                           min_remove=.25,min_restore=.25,min_individual=.20,max_norm_error=.03,
                                           primary="family mean logit-margin difference; exact one-sided sign test; all factor cells nonnegative"),
                           bounds="Timing/source full factorial at the same raw BF16 vector dose per head and exactly one read per eligible head. Native broad/complement factorial intentionally varies number of heads; not a matched-dose comparison.",
                           rescue="All three blocked at state/answer via V=0; restore every nonempty subset to original, compare norm-matched rotations. Single and all pair swap/zero also tested.",
                           adaptive="Choose strongest individual head and negative-control contrast from exposed discovery only; arm list fixed, promotion threshold unchanged.",
                           budget="400 MiB workspace preflight; sample 4 views per suite across all layers and natural generation"))


def check():
    doc=read("plan.json")
    for name,d in doc["material"].items():
        assert sha(OUT/name)==d["sha"],name
    for name,d in doc["sources"].items():
        assert sha(Path(__file__).with_name(name))==d,name
    for k,x in doc["runtime"].items():
        assert importlib.metadata.version(k)==x
    return doc


def parse(row,tok,tokens,eos,values,letters):
    stop=next((i for i,t in enumerate(tokens) if t in eos),None)
    if stop is not None:
        tokens=tokens[:stop+1]
    text=tok.decode(tokens,skip_special_tokens=True)
    match=re.fullmatch(r"[ \t]*([a-z]+)[ \t]*\r?\n[ \t]*Note[ \t]*:[ \t]*([a-z]+(?: +[a-z]+)*)[ \t]*\r?\n[ \t]*Code[ \t]*:[ \t]*([PQ])\s*",text)
    state,notes,code=(match[1],match[2].split(),match[3]) if match else (None,[],None)
    note_present=None
    code_index=None
    for j,t in enumerate(tokens):
        prefix=tok.decode(tokens[:j],skip_special_tokens=True).rstrip()
        if prefix.endswith("Note:") and tok.decode([t],skip_special_tokens=True).strip()==row["notes"][0]:
            note_present=j
        if prefix.endswith("Code:") and t in letters.values():
            code_index=j
    eligible=(bool(tokens) and tokens[0] in values.values() and note_present is not None
              and code_index is not None and note_present+1<code_index-1)
    return dict(tokens=tokens,text=text,state=state,notes=notes,code=code,eos=stop is not None,
                note_index=note_present,answer_index=code_index,
                joint=bool(eligible and stop is not None and state==row["state"] and notes==row["notes"] and code==row["code"]))


def gate(rows,observations):
    by=defaultdict(list)
    for r,obs in zip(rows,observations,strict=True):
        by[(r["update"],r["swap"],r.get("second_keeps"),r.get("voice"))].append(obs["joint"])
    return [dict(cell=k,n=len(v),correct=sum(v),passed=sum(v)/len(v)>=.90) for k,v in sorted(by.items())]


class Task(Base.Greedy):
    def __init__(self,engine):
        super().__init__(engine)
        self.letters={x:self.tok.encode(" "+x,add_special_tokens=False) for x in ("P","Q")}
        assert all(len(x)==1 for x in self.letters.values())
        self.letters={k:v[0] for k,v in self.letters.items()}

    def natural(self,rows,retain=False,sample=(),cap=CAP,step_log=None):
        import torch
        enc,inp=self.inputs(rows)
        field=Field(self,list(sample))
        try:
            first=self.first(inp)
            h=[first.logits[:,-1].argmax(-1)]
            pending_cache=self.clone(first.past_key_values) if retain else None
            c,mask,log=self.step(first.past_key_values,inp.attention_mask,h[0])
            c1=self.clone(c) if retain else None
            m1=mask
            steps=[log]
            done=self.ended(h[0])
            h.append(self.next_token(log,done))
            done=done|self.ended(h[-1])
            if step_log:
                step_log(2)
            while len(h)<cap and not bool(done.all()):
                c,mask,log=self.step(c,mask,h[-1])
                steps.append(log)
                h.append(self.next_token(log,done))
                done=done|self.ended(h[-1])
                if step_log:
                    step_log(len(h))
            raw=torch.stack(h,dim=1).cpu().tolist()
        finally:
            field.stop()
        obs=[dict(id=r["id"],input_ids=x,value_ids=m,**parse(r,self.tok,y,self.eos,m,self.letters))
             for r,(x,m),y in zip(rows,enc,raw,strict=True)]
        if not retain:
            return obs
        init_hash=self.full_hash(pending_cache)
        alt=torch.as_tensor([m[r["alt"]] if o["joint"] else o["tokens"][0] for r,(_,m),o in zip(rows,enc,obs,strict=True)],device="cuda")
        donor,_,_=self.step(self.clone(pending_cache),inp.attention_mask,alt)
        v_alt=[layer.values[:,:,-1].clone() for layer in donor.layers]
        assert self.full_hash(pending_cache)==init_hash
        source=inp.input_ids.shape[1]
        note=torch.as_tensor([source+(o["note_index"] if o["joint"] else 0) for o in obs],device="cuda")
        row_idx=torch.arange(len(rows),device="cuda")
        v_note=[layer.values[row_idx,:,note].clone() for layer in c.layers]
        return dict(rows=rows,obs=obs,history=h,steps=steps,c1=c1,mask=m1,source=source,decoy=note,
                    v_note=v_note,v_alt=v_alt,prompt_hash=init_hash,field=field.saved,
                    after_state=state_digest(steps[0]))


def load_model(key):
    identity=Base.Previous.identity(key)
    engine=Base.load_model(key)
    target=OUT/f"{key}-model.json"
    if target.exists():
        assert read(target.name)["files"]==identity
    else:
        write(target.name,dict(files=identity,digest=Base.Previous.digest(identity),device_map=engine.placement,
                               protocol=sha(OUT/"plan.json")))
    return engine


def behavior(g,suite):
    rows=read(f"materials-{suite}.json")
    jobs=list(Base.groups(rows,BATCH))
    start=time.monotonic()
    out=[]
    for i,group in enumerate(jobs,1):
        out.extend(g.natural(group))
        Base.Previous.stage_log("q4",suite+"-behavior",i,len(jobs),start)
    cells=gate(rows,out)
    write(f"behavior-{suite}.json",dict(rows=out,cells=cells,passed=all(x["passed"] for x in cells)))
    return rows,out,all(x["passed"] for x in cells)


def reference(key):
    import torch
    torch.set_num_threads(4)
    check()
    rows=read("materials-discovery.json")[:4]
    with Base.Previous.GpuLease(),torch.inference_mode():
        model=load_model(key)
        try:
            start=time.monotonic()
            def step(j):
                Base.Previous.stage_log(key,"reference-generation",j,14,start)
            observations=Task(model).natural(rows,cap=14,step_log=step)
            write(f"reference-{key}.json",dict(rows=observations,cells=gate(rows,observations)))
        finally:
            model.close()


if __name__=="__main__":
    setup()
    if sys.argv[1]=="freeze":
        freeze()
    elif sys.argv[1]=="reference":
        reference(sys.argv[2])
