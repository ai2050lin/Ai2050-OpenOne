"""Greedy behavior and same-generation KV counterfactuals on CUDA."""
from collections import defaultdict
from datetime import datetime
import importlib.metadata
from pathlib import Path
import json
import os
import re
import sys
import time
from zoneinfo import ZoneInfo

import condition_chain_run as Previous
import own_state_materials as Materials

ROOT = Previous.ROOT
OUT = ROOT / "tests/glm5/result/own_state_20261115_v1"
TEMP = ROOT / "tests/glm5_temp/own_state_20261115_v1"
RUN = "OWN-STATE-20261115-v1"
MAX_NEW = 11
GPU_BATCH = 32
MODELS = {"q4":"qwen3-4b", "q14":"Qwen3-14B", "glm":"glm4-9b-chat-hf"}
GATES = dict(behavior=.90, min_token_effect=.25, min_v_minus_control=.10, max_noise_flip=.25, alpha=.05/4,
             same_cache_exact=1., max_norm_relative_error=.03, max_control_format_loss=.10)


def configure():
    Previous.OUT, Previous.TMP, Previous.RUN = OUT, TEMP, RUN
    Previous.BUDGET = 400 * 2**20
    Previous.setup()
    os.environ.update(CUDA_CACHE_PATH=str(TEMP), PYTHONUTF8="1")


def read(name):
    return Previous.load(OUT / name)


def write(name, obj):
    Previous.write(name, obj)


def digest(path):
    return Previous.file_sha(path)


def freeze():
    assert not (OUT/"plan.json").exists()
    manifests={}
    for split in Materials.SPLITS:
        rows=Materials.all_rows(split)
        assert len({r["text"] for r in rows})==len(rows)
        name=f"materials-{split}.json"
        write(name,rows)
        manifests[name]=dict(sha=digest(OUT/name),n=len(rows))
    files=[Path(__file__),Path(Materials.__file__),Path(Previous.__file__),
           Path(__file__).with_name("condition_chain_reference_stream.py")]
    write("plan.json",dict(run_id=RUN,frozen_at=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"),
                            source_license="Original fictional dispatch and permit field records",
                            model_paths=MODELS,precision="BF16",dtype_intent="unquantized",batch=GPU_BATCH,
                            material=manifests,parts=Materials.SPLITS,thresholds=GATES,max_new=MAX_NEW,
                            sources={p.name:digest(p) for p in files},
                            versions={p:importlib.metadata.version(p) for p in ("torch","transformers","numpy","safetensors","accelerate","scipy")},
                            stages=["q4 pilot per-cell gate","q4 confirm for pilot-passed task","same-cache token and V-only crossed control","independent CPU audit"],
                            checkpoints="fixed first generated single value token; before it no generated state has entered KV; after it patch that position only",
                            behavior_gate="all 8 cells of each task separately, >=90% true state+code+strict structure+EOS+available causal site",
                            selection="all rows of each admitted task, including natural failure rows. Missing natural cut has zero success for all causal arms.",
                            reference_scope="q14 and GLM4: first 4 pilot families per task, behavioral feasibility only, no causal evidence",
                            metric="alternative code valid, at least mapping-specific directional effect, family is unit, joint across two tasks uses Bonferroni",
                            intervention="swap generated state token with alternate using prompt KV; transplant BF16 V at generated site from alternate-token forward under same KV; hold original K and g2; matched cyclic coordinate permutation of donor V update.",
                            retention="All generated tokens and failures, final cases and target V BF16 slices, prompt KV per-layer hashes and first replay logits hashes. Full states sampled at recorded boundaries; prompt KV tensors released per batch.",
                            max_seconds_between_progress=300))


def check_plan():
    plan=read("plan.json")
    for name,item in plan["material"].items():
        assert digest(OUT/name)==item["sha"],name
    for name,sha in plan["sources"].items():
        assert digest(Path(__file__).with_name(name))==sha,name
    for package,version in plan["versions"].items():
        assert importlib.metadata.version(package)==version
    return plan


def load_model(key):
    if key=="q14":
        from condition_chain_reference_stream import StreamEngine
        configure()
        return StreamEngine(key)
    return Previous.Engine(key)


def prompt(tokenizer,text):
    return tokenizer.apply_chat_template([dict(role="user",content=text)],tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False) + "State:"


def encode(tokenizer,row):
    message=prompt(tokenizer,row["text"])
    ids=tokenizer.encode(message,add_special_tokens=False)
    possible={}
    for value in (row["old"],row["new"]):
        candidate=tokenizer.encode(message+" "+value,add_special_tokens=False)
        assert len(candidate)==len(ids)+1 and candidate[:-1]==ids, (row["id"],value)
        possible[value]=candidate[-1]
    return ids,possible


def answer_pattern(text):
    pattern=r"\s*([a-z]+)[ \t]*\r?\n[ \t]*(?:Code)[ \t]*:[ \t]*([PQ])[ \t\r\n]*"
    match=re.fullmatch(pattern,text,flags=re.I)
    if match:
        return match[1].lower(),match[2].upper()
    return None,None


def score(tokens,row,tokenizer,eos_ids,choices):
    stop=next((i for i,t in enumerate(tokens) if t in eos_ids),None)
    retained=tokens if stop is None else tokens[:stop+1]
    text=tokenizer.decode(retained,skip_special_tokens=True)
    state,code=answer_pattern(text)
    # At g1 the state is one native token. At g2 no final code can already be fixed.
    at_site=bool(len(retained)>=3 and retained[0] in choices.values()
                 and re.fullmatch(r"\s*[a-z]+",tokenizer.decode([retained[0]],skip_special_tokens=False),flags=re.I)
                 and not re.search(r"\b[QP]\b",tokenizer.decode(retained[:2],skip_special_tokens=True)))
    return dict(tokens=retained,text=text,state=state,code=code,eos=stop is not None,site=at_site,
                state_correct=state==row["state"],code_correct=code==row["code"],
                joint=state==row["state"] and code==row["code"] and stop is not None and at_site,
                flips=code==row["alt_code"] and stop is not None and state is not None)


def groups(rows,n):
    for offset in range(0,len(rows),n):
        yield rows[offset:offset+n]


def marks(rows,records):
    cells=defaultdict(list)
    for row,out in zip(rows,records,strict=True):
        cells[(row["task"],tuple(row["factors"]))].append(out["natural"])
    return [dict(task=t,cell=list(k),n=len(arr),joint=sum(r["joint"] for r in arr),
                 state=sum(r["state_correct"] for r in arr),code=sum(r["code_correct"] for r in arr),
                 eos=sum(r["eos"] for r in arr),sites=sum(r["site"] for r in arr),
                 passed=sum(r["joint"] for r in arr)/len(arr)>=GATES["behavior"])
            for (t,k),arr in sorted(cells.items())]


def approved(cells):
    return [task for task in Materials.TASKS
            if len([s for s in cells if s["task"]==task and s["passed"]])==len(Materials.FACTORS)]


def torch_eos(model,tok):
    got=model.generation_config.eos_token_id
    if got is None:
        got=tok.eos_token_id
    return set(got if isinstance(got,list) else [got])


class Greedy:
    def __init__(self,engine):
        import torch
        self.torch=torch
        self.engine=engine
        self.model=engine.model
        self.tok=engine.tok
        self.eos=torch_eos(self.model,self.tok)
        self.pad=self.tok.pad_token_id

    def first(self, inputs):
        pos=(inputs.attention_mask.cumsum(-1)-1).masked_fill(inputs.attention_mask==0,1)
        return self.model(**inputs,position_ids=pos,use_cache=True,logits_to_keep=1)

    def step(self,cache,mask,pending):
        import torch
        newmask=torch.cat((mask,torch.ones_like(mask[:,:1])),dim=1)
        pos=newmask.sum(-1,keepdim=True)-1
        ans=self.model(input_ids=pending[:,None],position_ids=pos,attention_mask=newmask,
                       past_key_values=cache,use_cache=True,logits_to_keep=1)
        return ans.past_key_values,newmask,ans.logits[:, -1]

    def next_token(self,logits,done):
        token=logits.argmax(-1)
        return token.masked_fill(done,self.pad)

    def ended(self,token):
        return self.torch.isin(token,self.torch.tensor(sorted(self.eos),device=token.device))

    def advance(self,cache,mask,history,done,cap=MAX_NEW):
        torch=self.torch
        while len(history)<cap and not bool(done.all()):
            cache,mask,logits=self.step(cache,mask,history[-1])
            selected=self.next_token(logits,done)
            done=done|self.ended(selected)
            history.append(selected)
        return torch.stack(history,dim=1).cpu().tolist()

    def clone(self,cache):
        from transformers import DynamicCache
        return DynamicCache(ddp_cache_data=[(layer.keys.clone(),layer.values.clone()) for layer in cache.layers],
                            config=self.model.config)

    def full_hash(self,cache):
        from b36_switch14_engine import tensor_hash
        return [(tensor_hash(l.keys),tensor_hash(l.values)) for l in cache.layers]

    def inputs(self,rows):
        encoded=[encode(self.tok,row) for row in rows]
        padded=self.tok.pad(dict(input_ids=[pair[0] for pair in encoded]),padding=True,return_tensors="pt").to("cuda")
        return encoded,padded

    def behavior(self,rows,verify_hf=False):
        encoded,inputs=self.inputs(rows)
        output=self.first(inputs)
        first=output.logits[:,-1].argmax(-1)
        tokens=self.advance(output.past_key_values,inputs.attention_mask,[first],self.ended(first))
        records=self.records(rows,encoded,tokens)
        if verify_hf:
            generated=self.model.generate(**inputs,max_new_tokens=MAX_NEW,do_sample=False,
                                          pad_token_id=self.pad,use_cache=True)
            seq=generated[:,inputs.input_ids.shape[1]:].cpu().tolist()
            reference=self.records(rows,encoded,seq)
            assert [r["natural"]["tokens"] for r in records]==[r["natural"]["tokens"] for r in reference]
        return records

    def records(self,rows,encoded,tokens):
        return [dict(id=r["id"],input_ids=x,native_values=m,natural=score(y,r,self.tok,self.eos,m))
                for r,(x,m),y in zip(rows,encoded,tokens,strict=True)]


def behavior(key):
    import torch
    check_plan()
    assert not (OUT/f"{key}-pilot.json").exists()
    torch.set_num_threads(4)
    with Previous.GpuLease(),torch.inference_mode():
        start=time.monotonic()
        model_identity=Previous.identity(key)
        engine=load_model(key)
        try:
            generation=Greedy(engine)
            write(f"{key}-model.json",dict(files=model_identity,model_manifest_sha=Previous.digest(model_identity),
                                         placement=engine.placement,eos=sorted(generation.eos),plan_sha=digest(OUT/"plan.json")))
            eligible=Materials.TASKS
            for split in ("pilot","confirm"):
                if key!="q4" and split!="pilot":
                    continue
                rows=[r for r in read(f"materials-{split}.json") if r["task"] in eligible
                      and (key=="q4" or r["index"]<4)]
                if not rows:
                    write(f"{key}-{split}.json",dict(records=[],cells=[],admitted=[],reason="pilot gate stopped confirmation"))
                    continue
                output=[]
                batches=list(groups(rows,GPU_BATCH if key=="q4" else 64))
                stage=f"{key}-{split}-behavior"
                begin=time.monotonic()
                Previous.stage_log(key,stage,0,len(batches),begin)
                for i,batch in enumerate(batches,1):
                    output.extend(generation.behavior(batch,verify_hf=(split=="pilot" and i==1 and key=="q4")))
                    Previous.stage_log(key,stage,i,len(batches),begin)
                summary=marks(rows,output)
                eligible=approved(summary)
                write(f"{key}-{split}.json",dict(records=output,cells=summary,admitted=eligible,
                                               model_manifest_sha=Previous.digest(model_identity),
                                               first_pilot_batch_hf_token_match=key=="q4" and split=="pilot"))
        finally:
            engine.close()
        Previous.stage_log(key,"behavior-complete",1,1,start)


if __name__=="__main__":
    configure()
    if sys.argv[1]=="freeze":
        freeze()
    elif sys.argv[1]=="behavior":
        behavior(sys.argv[2])
