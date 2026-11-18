"""Fresh chat-format semantic controls for three frozen Qwen3-4B archive indices."""
from itertools import product
import sys
import time
from pathlib import Path

import condition_chain_chat as C
import condition_chain_run as R
import condition_chain_materials as M

R.RUN = "CONDITION-CHAIN-CANDIDATES-20261115-v3"
R.OUT = R.OUT.with_name("condition_chain_candidates_20261115_v3")
R.TMP = R.TMP.with_name("condition_chain_candidates_20261115_v3")
R.BUDGET = 115 * 2**20
SCOPES = {
    "pilot": ("gauge","locker","motor"),
    "discover": ("gauge","locker","motor"),
    "words": ("valve","coat rack","hearing aid"),
    "frames": ("refrigerator","camp pitch","circuit board"),
}


def candidate(split, task, index, factors):
    target, keyword = factors
    rng = M.rng("semantic-chat-v3",split,task,index)
    name = rng.choice(C.TOKENS[split].split()).capitalize()
    a = rng.randrange(31,70)
    difference = rng.randrange(10,19)
    b = a + (difference if target else -difference)
    topic, positive, negative = {
        "boundary": ("threshold", "The reading exceeded the bound.", "The reading was below the bound."),
        "payment": ("payment", "A fee is due for this visit.", "This visit is free."),
        "assessment": ("testing", "The verdict was based on a measurement.", "The verdict was based on a guess."),
    }[task]
    pressure, item, device = SCOPES[split]
    if task=="boundary":
        line={
            "pilot":f"For {name}'s {pressure}, the safe maximum is {a} bars. Its current reading is {b} bars.",
            "discover":f"For {name}'s {pressure}, the safe maximum is {a} bars. Its current reading is {b} bars.",
            "words":f"At station {name} the maximum permitted pressure on the {pressure} is {a} bars. The dial reads {b} bars.",
            "frames":f"The temperature of the {pressure} at {name} was {b} degrees, compared with its upper bound of {a} degrees.",
        }[split]
    elif task=="payment":
        used = "West" if target else "East"
        line = f"{name} uses the {used} {item}. Use of the East {item} is free. Use of the West {item} costs five dollars."
        if split=="frames":
            line = f"Only the West {item} costs the customer anything. The {item} chosen by {name} was the {used} one. The East {item} is free."
    else:
        choices = {
            "pilot": ("guessing by eye", "checking instrument readings"),
            "discover": ("guessing by eye", "checking instrument readings"),
            "words": ("following a hunch", "checking a sound level meter reading"),
            "frames": ("an unverified estimate", "a documented meter reading"),
        }[split]
        line = f"The technician judged the {device} at {name} by {choices[target]}, with no other evidence."
        if split=="frames":
            line = f"The judgement on the condition of the {device} at {name} rests solely on {choices[target]}."
    gloss = f"Glossary heading for a different exercise: {topic if keyword else 'reference'}."
    order = rng.randrange(2)
    options = (positive,negative) if order else (negative,positive)
    ending = "Choose the conclusion that follows from the actual record.\nA. "+options[0]+"\nB. "+options[1]
    family=M.key("semantic-chat-v3",split,task,index)
    return dict(id=family+"".join(map(str,factors)),family=family,index=index,task=task,split=split,panel="candidate",
                factors=factors,truth=[target],pieces=[gloss+"\n"+line],ending=ending,options=options,
                evidence=dict(a=a,b=b),answer=options.index(positive if target else negative))


def render(row):
    intro="Read the factual record and pick the true conclusion. Return only A or B.\n\n"
    first=intro+row["pieces"][0]+"\nCheckpoint:"
    return first+"\n\n"+row["ending"],[len(first)]


def freeze():
    assert not (R.OUT/"plan.json").exists()
    source = [__file__, R.__file__, C.__file__, M.__file__, str(Path(R.__file__).with_name("condition_chain_analysis.py"))]
    panels = {}
    for split,n in C.SPLITS.items():
        rows=[candidate(split,task,i,arm) for task in M.CANDIDATES for i in range(n) for arm in product((0,1),repeat=2)]
        name=f"materials-{split}-candidate.json"
        R.write(name,rows)
        panels[name]=dict(n=len(rows),sha256=R.file_sha(R.OUT/name))
    import importlib.metadata
    R.write("plan.json",dict(run_id=R.RUN,precision="BF16",thresholds=R.THRESHOLDS,candidates=M.CANDIDATES,materials=panels,
                            purpose="Fresh controlled v3 material after v1 comprehension-gate failure",
                            license="Original fictional examples",admission="all four fact x glossary cells each >=.85 on pilot and corresponding split",
                            primary="unchanged v1 G direction, effect size, 12 Bonferroni comparisons and supervised label prediction above lexical/embedding baseline",
                            site="last stable token of Checkpoint before the question",
                            sources={Path(p).name:R.file_sha(Path(p)) for p in source},
                            runtime={p:importlib.metadata.version(p) for p in ("torch","accelerate","numpy","transformers","scipy","safetensors","scikit-learn")}))


def run():
    import torch
    import numpy as np
    torch.set_num_threads(4)
    plan=R.protocol()
    assert not (R.OUT/"q4-model.json").exists()
    with R.GpuLease(),torch.inference_mode():
        start=time.monotonic()
        info=R.identity("q4")
        engine=R.Engine("q4")
        try:
            R.write("q4-model.json",dict(files=info,snapshot=R.digest(info),letters=engine.letters,placement=engine.placement,
                                         plan_sha=R.file_sha(R.OUT/"plan.json")))
            pilot_failed=set()
            for split in C.SPLITS:
                stage=f"q4-{split}-candidate"
                rows=R.load(R.OUT/f"materials-{split}-candidate.json")
                batches=list(R.blocks(rows,64))
                results=[]
                begin=time.monotonic()
                R.stage_log("q4",stage,0,len(batches),begin)
                for i,group in enumerate(batches,1):
                    rec,_=engine.batch(group,"candidate",False,[],generate=split=="pilot")
                    results.extend(rec)
                    R.stage_log("q4",stage,i,len(batches),begin)
                gates=R.cell_pass(rows,results,plan["thresholds"]["behavior"])
                R.write(stage+".json",dict(gate=gates,results=results,seconds=time.monotonic()-begin))
                failed={g["cell"][0] for g in gates if not g["pass_gate"]}
                if split=="pilot":
                    pilot_failed=failed
                    continue
                keep=set(M.CANDIDATES)-pilot_failed-failed
                capture=list(R.blocks([r for r in rows if r["task"] in keep],64))
                base={r["id"]:r for r in results}
                field=stage+"-native"
                R.stage_log("q4",field,0,len(capture),begin)
                for i,group in enumerate(capture,1):
                    rec,native=engine.batch(group,"candidate",True,[])
                    assert all(r["option_logits"]==base[r["id"]]["option_logits"] for r in rec)
                    if split!="discover":
                        native.pop("controls")
                    np.savez(R.check_write(R.OUT/f"{field}-{i:03d}.npz"), ids=[r["id"] for r in rec],**native)
                    R.stage_log("q4",field,i,len(capture),begin)
                R.write(field+".json",dict(eligible=sorted(keep),batches=len(capture)))
        finally:
            engine.close()
        R.stage_log("q4","complete",1,1,start)


if __name__=="__main__":
    R.setup()
    C.render=render
    R.encode=C.encode
    R.prompts=render
    if sys.argv[1]=="freeze":
        freeze()
    elif sys.argv[1]=="run":
        run()
    elif sys.argv[1]=="analyze":
        from condition_chain_analysis import main
        main()
    elif sys.argv[1]=="intervene":
        R.intervene()
