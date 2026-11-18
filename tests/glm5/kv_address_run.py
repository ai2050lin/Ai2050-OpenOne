"""Natural behavior and answer-prefix replay for Q/K addressing."""

from collections import defaultdict
from datetime import datetime
from itertools import batched
import json
import os
from pathlib import Path
import re
import sys
import time

import kv_alpha_run as Base
import kv_address_materials as M


R=Base.R
OUT=Base.ROOT/"tests/glm5/result/kv_address_20261115_v1"
TEMP=Base.ROOT/"tests/glm5_temp/kv_address_20261115_v1"
BATCH=16
MAX_NEW=18
STOCK_HEADS=R.HEADS
FACTORS=("context","order","identity")
RUN="KV-ADDRESS-20261115-v1"


def configure():
    R.OUT,R.TEMP,R.RUN=OUT,TEMP,RUN
    R.Base.Previous.check_write(OUT,380*2**20,operation=RUN)
    R.Base.Previous.check_write(TEMP,380*2**20,operation=RUN)
    R.setup()
    os.environ["MPLCONFIGDIR"]=str(TEMP)


def digest(p):
    return R.sha(p)


def load(key):
    return R.read(key)


def save(key,obj):
    R.write(key,obj)


def encoded(tok,row):
    prompt=tok.apply_chat_template([dict(role="user",content=row["text"])],tokenize=False,
                                   add_generation_prompt=True,enable_thinking=False)+row["first"]+":"
    ids=tok.encode(prompt,add_special_tokens=False)
    options={}
    for word in (row["state"],row["note"],row["alt"]):
        tokens=tok.encode(" "+word,add_special_tokens=False)
        assert len(tokens)==1 and tok.encode(prompt+" "+word,add_special_tokens=False)==ids+tokens
        options[word]=tokens[0]
    return ids,options


class Task(R.Base.Greedy):
    def __init__(self,engine):
        super().__init__(engine)
        self.letters={x:self.tok.encode(" "+x,add_special_tokens=False)[0] for x in "PQ"}

    def inputs(self,rows):
        enc=[encoded(self.tok,row) for row in rows]
        batch=self.tok.pad(dict(input_ids=[x for x,_ in enc]),padding=True,return_tensors="pt").to("cuda")
        return enc,batch

    def parse(self,row,ids,options,tokens):
        eos=next((j for j,t in enumerate(tokens) if t in self.eos),None)
        if eos is not None:
            tokens=tokens[:eos+1]
        tok=self.tok
        txt=tok.decode(tokens,skip_special_tokens=True)
        second="State" if row["order"] else "Note"
        m=re.fullmatch(r"[ \t]*([a-z]+)[ \t]*\n[ \t]*"+second+r":[ \t]*([a-z]+)[ \t]*\n[ \t]*Code:[ \t]*([PQ])\s*",txt)
        indexes={}
        for i,x in enumerate(tokens):
            prefix=tok.decode(tokens[:i],skip_special_tokens=True).rstrip()
            if i==0 and x==options[row[row["first"].lower()]]:
                indexes[row["first"].lower()]=i
            if prefix.endswith(second+":") and x==options[row[second.lower()]]:
                indexes[second.lower()]=i
            if prefix.endswith("Code:") and x in self.letters.values():
                indexes["code"]=i
        wanted=(row[row["first"].lower()],row[second.lower()],row["code"])
        valid=bool(m and m.groups()==wanted and eos is not None and len(indexes)==3 and indexes["code"]-1>max(indexes["state"],indexes["note"]))
        return dict(id=row["id"],input_ids=ids,options=options,tokens=tokens,text=txt,index=indexes,joint=valid)

    def natural(self,rows):
        enc,inp=self.inputs(rows)
        first=self.first(inp)
        token=first.logits[:,-1].argmax(-1)
        raws=self.advance(first.past_key_values,inp.attention_mask,[token],self.ended(token),cap=MAX_NEW)
        return [self.parse(r,x,y,z) for r,(x,y),z in zip(rows,enc,raws,strict=True)]


def groups(rows):
    return [list(g) for g in batched(rows,BATCH)]


def behavioral(task,rows,stage):
    jobs=groups(rows)
    disk=[]
    start=time.monotonic()
    for i,b in enumerate(jobs,1):
        disk.extend(task.natural(b))
        R.Base.Previous.stage_log("q4",stage,i,len(jobs),start)
    cells=defaultdict(list)
    for r,o in zip(rows,disk,strict=True):
        cells[tuple(r[k] for k in FACTORS)].append(o["joint"])
    trials=[dict(cell=k,n=len(v),correct=sum(v),passed=sum(v)/len(v)>=.90) for k,v in sorted(cells.items())]
    return dict(rows=disk,cells=trials,passed=all(c["passed"] for c in trials))


def with_task(fn):
    import torch
    torch.set_num_threads(4)
    with R.Base.Previous.GpuLease(),torch.inference_mode():
        start=time.monotonic()
        R.Base.Previous.stage_log("q4","load",0,1,start)
        engine=R.Base.load_model("q4")
        try:
            R.Base.Previous.stage_log("q4","load",1,1,start)
            fn(Task(engine))
        finally:
            engine.close()


def pilot(task):
    rows=M.material("pilot",task.tok)
    save("pilot-materials.json",rows)
    save("pilot-behavior.json",behavioral(task,rows,"pilot"))


if __name__=="__main__":
    configure()
    if sys.argv[1]=="pilot":
        with_task(pilot)
    else:
        from kv_address_study import commands
        commands[sys.argv[1]]()
