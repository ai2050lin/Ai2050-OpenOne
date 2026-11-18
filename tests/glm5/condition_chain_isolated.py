"""Localize behavior failures by supplying controlled starting states at each stage."""
from itertools import product
from pathlib import Path
import sys
import time

import condition_chain_chat as C
import condition_chain_materials as M
import condition_chain_run as R

R.RUN = "CONDITION-CHAIN-ISOLATED-20261115-v4"
R.OUT = R.OUT.with_name("condition_chain_isolated_20261115_v4")
R.TMP = R.TMP.with_name("condition_chain_isolated_20261115_v4")
R.BUDGET = 95 * 2**20
STAGES = ("select","update","maintain","report","composed")
COUNTS = {"pilot":8,"words":96,"frames":96}
NAMES = {
    "pilot": "cabin copper harbor jasper",
    "words": "alder bay cypress falcon ginkgo honey ivy junco kiwi lilac mint poppy raven spruce teal violet",
    "frames": "aspen beaver cedar dogwood fox heron iris kelpie lupine moose newt oriole prawn quail thyme zinnia",
}


def material(split, domain, i, stage, flip):
    rng = M.rng("isolated-v4",split,domain,i)
    words = NAMES[split].split()
    pairs = [(a,b) for a in range(len(words)) for b in range(len(words)) if a!=b]
    M.rng("isolated-v4-pair-order",split,domain).shuffle(pairs)
    first,second = pairs[i]
    third = next(j for j in ((second+k)%len(words) for k in range(1,len(words))) if j!=first)
    fourth = rng.choice([j for j in range(len(words)) if j not in (first,second,third)])
    a,b,c,d = (words[j] for j in (first,second,third,fourth))
    state = (a,b) if stage=="select" else (a,c)
    if stage=="composed":
        state=(a,b)
    v=state[flip]
    noun, key = ("crate label","crate") if domain=="stock" else ("meeting room","team")
    choices = [state[0],state[1]]
    positive,negative = "Use the replacement.","Keep the starting value."
    command = positive if flip else negative
    comb = ["North","South"][flip]
    other = choices[1-flip]
    messages = {
        "select": f"North {key} has {a}. South {key} has {b}. The active {key} is {comb}. Use the active {key}'s {noun}.",
        "update": f"The starting {noun} is {a}. The replacement is {c}. {command}",
        "maintain": f"The final {noun} is {v}. A practice example says {other}. Ignore the practice example.",
        "report": f"The final {noun} is {v}. The code for {a} is silver. The code for {c} is purple.",
        "composed": f"North {key} starts with {a}. South starts with {b}. Use the {comb} {key}. "
                    f"The two available replacement names are {c} and {d}, but no replacement is authorized. "
                    f"The example value {other} is irrelevant. For the answer, {a} means silver and {b} means purple.",
    }
    if split=="frames":
        messages.update({
            "select": f"For {noun}, the entry {a} belongs to North and {b} belongs to South. Choose the one belonging to {comb}.",
            "update": f"For {noun}, you may switch from {a} to {c}. {'Switch to the replacement.' if flip else 'Do not switch.'}",
            "maintain": f"Do not use the demonstration value {other}. The actual saved {noun} is {v}.",
            "report": f"Use purple when the value is {c}, silver when it is {a}. The actual final {noun} is {v}.",
            "composed": f"The {noun} for South is {b}, while North has {a}. Look up the one for {comb}. "
                        f"Entries {c} and {d} are unused backups. The illustration {other} is not a record entry. "
                        f"Answer in the code that uses purple for {b} and silver for {a}.",
        })
    if stage in ("report","composed"):
        choices=["silver","purple"]
        if rng.randrange(2):
            for color,new in zip(("silver","purple"),("amber","violet")):
                messages[stage]=messages[stage].replace(color,new)
            choices=["amber","violet"]
    order=rng.randrange(2)
    correct=choices[flip]
    if order:
        choices=choices[::-1]
    question="Which code should be reported?" if stage in ("report","composed") else f"Which {noun} should be used?"
    family=M.key("isolated-v4",split,domain,i)
    return dict(id=family+"-"+stage+str(flip),family=family,split=split,task=domain,stage=stage,index=i,
                factors=[flip],panel="candidate",pieces=[messages[stage]],
                ending=f"{question}\nA. {choices[0]}\nB. {choices[1]}", options=choices, answer=choices.index(correct),
                correct_option=correct,initial=a,replacement=c,selected=comb,truth=[flip])


def render(row):
    lead="Read the record and answer its question. Pick either A or B.\n\n"+row["pieces"][0]+"\nCheckpoint:"
    return lead+"\n\n"+row["ending"],[len(lead)]


def freeze():
    import importlib.metadata
    assert not (R.OUT/"plan.json").exists()
    panels={}
    for split,n in COUNTS.items():
        rows=[material(split,domain,i,stage,flip) for domain,stage in product(("stock","calendar"),STAGES)
              for i,flip in product(range(n),(0,1))]
        name=f"materials-{split}.json"
        R.write(name,rows)
        panels[name]=dict(n=len(rows),sha256=R.file_sha(R.OUT/name))
    frozen=[Path(p) for p in (__file__,M.__file__,C.__file__,R.__file__)]
    R.write("plan.json",dict(run_id=R.RUN,precision="BF16",model=M.MODELS["q4"],materials=panels,
                            thresholds=dict(cell_accuracy=.85,publication="all flip x domain x stage cells on words and frames, plus audit"),
                            scope="Qwen3-4B. Supplied states by task. Composed row always keeps the selected initial value; no replace test here.",
                            natural="Generate only in pilot. First option, exact bare letter, and other forms remain separate.",
                            sources={f.name:R.file_sha(f) for f in frozen},
                            runtime={p:importlib.metadata.version(p) for p in ("torch","accelerate","numpy","transformers","safetensors")}))


def run():
    import torch
    import re
    from collections import defaultdict
    torch.set_num_threads(4)
    plan=R.protocol()
    with R.GpuLease(),torch.inference_mode():
        st=time.monotonic()
        files=R.identity("q4")
        engine=R.Engine("q4")
        try:
            R.write("q4-model.json",dict(files=files,snapshot=R.digest(files),letters=engine.letters,placement=engine.placement,
                                         plan_sha=R.file_sha(R.OUT/"plan.json")))
            for split in COUNTS:
                stem=f"q4-{split}"
                assert not (R.OUT/(stem+".json")).exists()
                rows=R.load(R.OUT/f"materials-{split}.json")
                groups=list(R.blocks(rows,32 if split=="pilot" else 64))
                results=[]
                counts=defaultdict(list)
                begin=time.monotonic()
                R.stage_log("q4",stem,0,len(groups),begin)
                for i,batch in enumerate(groups,1):
                    rec,_=engine.batch(batch,"candidate",False,[],generate=split=="pilot")
                    for r,out in zip(batch,rec,strict=True):
                        answer=out["natural_answer"]
                        if answer is not None:
                            first=re.match(r"^\s*[*_]*(?:option\s+)?([AB])(?=$|\W)",answer)
                            out["natural_initial_correct"]=bool(first and first[1]=="AB"[r["answer"]])
                        counts[(r["task"],r["stage"],*r["factors"])].append(out["correct"])
                    results.extend(rec)
                    R.stage_log("q4",stem,i,len(groups),begin)
                gate=[dict(cell=k,correct=sum(v),n=len(v),passed=sum(v)/len(v)>=plan["thresholds"]["cell_accuracy"])
                      for k,v in counts.items()]
                R.write(stem+".json",dict(results=results,gate=gate))
        finally:
            engine.close()
        R.stage_log("q4","complete",1,1,st)


if __name__=="__main__":
    C.render=render
    R.prompts=render
    R.encode=C.encode
    R.setup()
    if sys.argv[1]=="freeze":
        freeze()
    elif sys.argv[1]=="run":
        run()
