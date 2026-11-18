"""Ground truth and same-generation checkpoints for a K/V reading experiment."""
from collections import defaultdict
from datetime import datetime
import importlib.metadata
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import own_state_run as Base
from own_state_intervene import Field,state_digest
import kv_read_materials_v2 as Material

OUT=Base.ROOT/"tests/glm5/result/kv_read_20261115_v2"
TEMP=Base.ROOT/"tests/glm5_temp/kv_read_20261115_v2"
RUN="KV-READ-20261115-v2"
CAP=16
BATCH=32
P=.05/90
BEHAVIOR=.90


def setup():
    Base.OUT,Base.TEMP,Base.RUN=OUT,TEMP,RUN
    Base.configure()


def read(name):
    return Base.Previous.load(OUT/name)


def write(name,value):
    assert Base.Previous.OUT==OUT
    return Base.Previous.write(name,value)


def sha(path):
    return Base.digest(path)


def freeze():
    assert not (OUT/"plan.json").exists()
    prompts={}
    for suite in Material.SUITES:
        raw=Material.generate(suite)
        name=f"materials-{suite}.json"
        write(name,raw)
        prompts[name]=dict(sha256=sha(OUT/name),views=len(raw))
    files=["kv_read_run.py","kv_read_materials.py","kv_read_materials_v2.py","kv_read_experiment.py",
           "own_state_run.py","own_state_intervene.py","condition_chain_run.py","condition_chain_reference_stream.py"]
    write("plan.json",dict(run_id=RUN,source_license="Original fictional informational records",dtype="BF16",
                           frozen_at=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"),
                           sources={p:sha(Path(__file__).with_name(p)) for p in files},
                           runtime={p:importlib.metadata.version(p) for p in ("torch","transformers","scipy","numpy","safetensors")},
                           materials=prompts,batch=BATCH,capped_generation=CAP,
                           discovery_source=dict(path=str(Material.PRIOR.relative_to(Base.ROOT)),sha256=sha(Material.PRIOR),exposed=True),
                           selection=dict(layers_per_side=2,heads_per_side=3,
                                          score="mean positive change of alternative-code margin, top of fixed discovery suites",
                                          max_broad_to_local="scan every layer per side, then all 32 query heads in each of the top two layers; three times on the top three heads"),
                           behavior=dict(each_cell=BEHAVIOR,split="all four replace x codebook cells, plus both second-update modes in multi",
                                         eos=True,last_code="single leading-space Q/P token after native Code: boundary",
                                         no_row_selection="Natural failure rows get zero contrast. If a whole suite fails, stop affected suite."),
                           inference=dict(alpha=P,family_level="exact one-sided sign test on family-mean logit-margin contrasts",
                                          decision_units="up to 90 scalar contrasts; four views per family",
                                          broad_threshold=.75,sparse_threshold=.50,individual_threshold=.25,
                                          time_threshold=.25,removal_threshold=.50,rescue_threshold=.50,matched_rescue_threshold=.25),
                           method=dict(source="after processing natural first state token, one slot",
                                       fixed_prefix="natural non-answer tokens through Code:, never feed the natural answer to judge candidate code",
                                       free_scope="separate greedy free continuation after natural state for broad Key, broad Value, matched Value, selected Value",
                                       clocks=["first query after state","interior queries before answer","answer-predictor query"],
                                       geometry="Qwen3-4B 32Q heads / 8 K/V groups, head h reads group floor(h/4)",
                                       removal="zero content for selected query-head reads; Key zero leaves source available and is not masking",
                                       recovery="from broad-zero stock, restore original selected head/time inputs; compare equal-size off-target restoration",
                                       norm_control="cyclic permutation of coordinate delta; <=3% after BF16 rounding",
                                       behavior_reference="q14 and GLM4 serial pilot, first two families per holdout, no mechanism comparison")))


def plan():
    p=read("plan.json")
    for file,value in p["sources"].items():
        assert sha(Path(__file__).with_name(file))==value,file
    for file,value in p["materials"].items():
        assert sha(OUT/file)==value["sha256"],file
    for pkg,version in p["runtime"].items():
        assert importlib.metadata.version(pkg)==version
    return p


def parsed(row,tok,tokens,eos,value_ids,answer_ids):
    end=next((i for i,k in enumerate(tokens) if k in eos),None)
    ids=tokens if end is None else tokens[:end+1]
    raw=tok.decode(ids,skip_special_tokens=True)
    if row["suite"]=="gap":
        expr=r"[ \t]*([a-z]+)[ \t]*\r?\n[ \t]*Note[ \t]*:[ \t]*([a-z]+)[ \t]*\r?\n[ \t]*Code[ \t]*:[ \t]*([PQ])\s*"
    else:
        expr=r"[ \t]*([a-z]+)[ \t]*\r?\n[ \t]*Code[ \t]*:[ \t]*([PQ])\s*"
    match=re.fullmatch(expr,raw)
    state=match[1] if match else None
    code=match.groups()[-1] if match else None
    aux=match[2] if match and row["suite"]=="gap" else None
    code_index=next((i for i,k in enumerate(ids) if i>=3 and k in answer_ids.values() and
                     tok.decode(ids[:i],skip_special_tokens=True).strip().endswith("Code:")),None)
    viable=bool(code_index is not None and ids[0] in value_ids.values())
    return dict(tokens=ids,text=raw,state=state,code=code,aux=aux,eos=end is not None,code_index=code_index,
                site=viable,joint=bool(viable and end is not None and state==row["state"]
                                        and code==row["code"] and aux==row["aux"]))


class Task(Base.Greedy):
    def __init__(self,engine):
        super().__init__(engine)
        encoded={x:self.tok.encode(" "+x,add_special_tokens=False) for x in ("P","Q")}
        assert all(len(seq)==1 for seq in encoded.values())
        self.answer_ids={x:v[0] for x,v in encoded.items()}

    def natural(self,rows,checkpoints=False,field_indices=()):
        import torch
        encoded,inp=self.inputs(rows)
        field=Field(self,list(field_indices))
        try:
            initial=self.first(inp)
            first=initial.logits[:,-1].argmax(-1)
            frozen=self.clone(initial.past_key_values) if checkpoints else None
            full,mask,logits=self.step(initial.past_key_values,inp.attention_mask,first)
            start=self.clone(full) if checkpoints else None
            history=[first]
            all_logits=[logits.detach()]
            done=self.ended(first)
            nxt=self.next_token(logits,done)
            history.append(nxt)
            done=done|self.ended(nxt)
            while len(history)<CAP and not bool(done.all()):
                full,_,after=self.step(full,torch.cat((mask,torch.ones_like(mask[:,:1]).expand(-1,len(history)-2)),dim=1),history[-1])
                all_logits.append(after.detach())
                nxt=self.next_token(after,done)
                history.append(nxt)
                done=done|self.ended(nxt)
            generated=torch.stack(history,dim=1).cpu().tolist()
        finally:
            field.stop()
        records=[dict(id=r["id"],input_ids=x,value_ids=m,
                      **parsed(r,self.tok,tokens,self.eos,m,self.answer_ids))
                 for r,(x,m),tokens in zip(rows,encoded,generated,strict=True)]
        if not checkpoints:
            return records
        original_hash=state_digest(logits)
        pre_donor_hash=self.full_hash(frozen)
        alt=torch.as_tensor([m[r["alt"]] if obs["joint"] else obs["tokens"][0]
                             for r,(_,m),obs in zip(rows,encoded,records,strict=True)],device="cuda")
        donor,_,_=self.step(self.clone(frozen),inp.attention_mask,alt)
        donors={p:[getattr(layer,p)[:,:,-1].clone() for layer in donor.layers] for p in ("keys","values")}
        assert self.full_hash(frozen)==pre_donor_hash
        return dict(rows=rows,records=records,mask=mask,cache=start,source=inp.input_ids.shape[1],
                    donor=donors,history=history,logits=all_logits,hashes=pre_donor_hash,site_hash=original_hash,field=field.saved)


def gate(rows,records):
    cells=defaultdict(list)
    for r,o in zip(rows,records,strict=True):
        cells[(r["update"],r["swap"],r["second_keeps"])].append(o["joint"])
    return [dict(cell=list(cell),n=len(values),correct=sum(values),passed=sum(values)/len(values)>=BEHAVIOR)
            for cell,values in sorted(cells.items())]


def open_model(key):
    start=time.monotonic()
    identity=Base.Previous.identity(key)
    engine=Base.load_model(key)
    assert Base.OUT==OUT
    destination=OUT/f"{key}-model.json"
    if destination.exists():
        assert read(destination.name)["files"]==identity
    else:
        write(destination.name,dict(files=identity,model_manifest_sha=Base.Previous.digest(identity),
                                    precision="BF16",placement=engine.placement,
                                    plan=sha(OUT/"plan.json"),seconds_to_open=time.monotonic()-start))
    return engine


def measure(g,suite,key,reference=False):
    rows=read(f"materials-{suite}.json")
    if reference:
        rows=[r for r in rows if r["index"]<2]
    jobs=list(Base.groups(rows,BATCH))
    start=time.monotonic()
    results=[]
    Base.Previous.stage_log(key,f"{suite}-behavior",0,len(jobs),start)
    for i,batch in enumerate(jobs,1):
        results.extend(g.natural(batch))
        Base.Previous.stage_log(key,f"{suite}-behavior",i,len(jobs),start)
    thresholds=gate(rows,results)
    write(f"{key}-{suite}-behavior.json",dict(rows=results,cells=thresholds,
                                               passed=all(x["passed"] for x in thresholds)))
    return rows,results,all(x["passed"] for x in thresholds)


def reference(key):
    import torch
    plan()
    assert (OUT/"selection.json").exists()
    torch.set_num_threads(4)
    with Base.Previous.GpuLease(),torch.inference_mode():
        engine=open_model(key)
        try:
            g=Task(engine)
            raw=[]
            rows=[]
            for suite in Material.SUITES[1:]:
                candidates=[r for r in read(f"materials-{suite}.json") if r["index"]<2]
                rows.extend(candidates)
            start=time.monotonic()
            Base.Previous.stage_log(key,"reference",0,1,start)
            raw=g.natural(rows)
            output={}
            for suite in Material.SUITES[1:]:
                take=[(r,a) for r,a in zip(rows,raw,strict=True) if r["suite"]==suite]
                output[suite]=dict(cells=gate([t[0] for t in take],[t[1] for t in take]),
                                   rows=[t[1] for t in take])
            write(f"{key}-reference.json",dict(suites=output))
            Base.Previous.stage_log(key,"reference",1,1,start)
        finally:
            engine.close()


if __name__=="__main__":
    setup()
    if sys.argv[1]=="freeze":
        freeze()
    elif sys.argv[1]=="reference":
        reference(sys.argv[2])
