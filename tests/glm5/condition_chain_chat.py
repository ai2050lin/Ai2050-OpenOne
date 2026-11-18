"""Follow-up with new families after v1 base-completion behavior failures."""
from itertools import product
import sys
import time
from pathlib import Path

import condition_chain_run as R
import condition_chain_materials as M

R.RUN = "CONDITION-CHAIN-CHAT-20261115-v2"
R.OUT = R.OUT.with_name("condition_chain_chat_20261115_v2")
R.TMP = R.TMP.with_name("condition_chain_chat_20261115_v2")
R.BUDGET = 300 * 2**20
SPLITS = {"pilot":8, "discover":32, "words":96, "frames":96}
TOKENS = {
    "pilot": "almond bamboo canvas denim falcon goblet ivory kettle",
    "discover": "anvil badger cactus dolphin easel fossil gecko hammer lemon marble orchid parrot quill ribbon saddle wagon",
    "words": "acacia basket castle dragon eagle feather guitar harbor ink jungle ladder mango nozzle otter pebble tinsel",
    "frames": "anchor beetle compass drum elm flute garden helmet iceberg koala lantern melon notebook opal pistol whistle",
}
DOMAINS = {
    "inventory": ("Warehouse", "rack", "stored label", "spare label"),
    "calendar": ("Calendar", "team", "booked room", "backup room"),
}


def material(split, task, index, factors):
    select, update, distract, report = factors
    randomizer = M.rng("chat-v2", split, task, index)
    labels = randomizer.sample(TOKENS[split].split(),4)
    codes = randomizer.sample(["red","blue","green","black"],4)
    permutation = randomizer.sample(range(4),4)
    shifted = permutation[report:] + permutation[:report]
    title, key, initial, replacement = DOMAINS[task]
    state = 2*select
    end = state+update
    example = [0,3][distract]
    entry = ("East","West")[select]
    if split == "frames":
        pieces = [
            f"{title} note: the active {key} is {entry}. Its main value is listed among these: "
            f"East has {labels[0]}, and West has {labels[2]}. Start with the active {key}'s main value.",
            f"East can switch to {labels[1]}, and West can switch to {labels[3]}. "
            f"{'Leave the current value as it is.' if not update else f'Switch to the active {key}’s alternate.'}",
            f"There is also a practice example mentioning {labels[example]}. The practice example never changes the current value.",
        ]
    else:
        pieces = [
            f"{title} record\nEast {initial}: {labels[0]}\nWest {initial}: {labels[2]}\n"
            f"Active {key}: {entry}. The current value starts as its {initial}.",
            f"East {replacement}: {labels[1]}\nWest {replacement}: {labels[3]}\n"
            f"Update: {'keep the current value' if not update else f'use the active {key}’s {replacement} instead'}.",
            f"Practice example (ignore): {labels[example]}. The actual value stays as set by the update.",
        ]
    if report:
        answer_part = "Use this color key to convert the resulting value into a color.\n" + "\n".join(
            f"{label} -> {code}" for label,code in zip(labels,codes))
        options = [codes[j] for j in shifted]
    else:
        answer_part = "Use the resulting value as written in the record."
        options = [labels[j] for j in shifted]
    ending = answer_part + "\n" + "\n".join(f"{letter}. {option}" for letter,option in zip("ABCD",options))
    family = M.key("chat-v2",split,task,index)
    return dict(id=family+"".join(map(str,factors)),family=family,task=task,index=index,split=split,
                panel="chain",factors=factors,pieces=pieces,ending=ending,truth=[state,end,end],
                option_order=shifted,answer=shifted.index(end),values=labels,codes=codes,diagnostic_index=example)


def render(row):
    lead = "Find the final value in this short record. Follow the selection and update instructions in order. Return only A, B, C, or D.\n\n"
    marks = []
    for piece in row["pieces"]:
        lead += piece + "\nCheckpoint:"
        marks.append(len(lead))
        lead += "\n\n"
    return lead+row["ending"],marks


def encode(tokenizer,row):
    message,cuts = render(row)
    prompt = tokenizer.apply_chat_template([{"role":"user","content":message}],tokenize=False,
                                           add_generation_prompt=True,enable_thinking=False) + "Answer:"
    start = prompt.index(message)
    encoded = tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    ids,offsets=encoded["input_ids"],encoded["offset_mapping"]
    sites=[]
    for cut in cuts:
        boundary = start+cut
        for i,(lo,hi) in reversed(list(enumerate(offsets))):
            if boundary-len("Checkpoint:")<hi<=boundary and tokenizer.encode(prompt[:hi],add_special_tokens=False)==ids[:i+1]:
                sites.append(i)
                break
        else:
            raise ValueError(f"Readout is unstable under tokenization: {row['id']}")
    return ids,sites+[len(ids)-1]


def freeze():
    source = [__file__, R.__file__, M.__file__, str(Path(R.__file__).with_name("condition_chain_analysis.py"))]
    panels={}
    for split,n in SPLITS.items():
        arms=product((0,1),repeat=4)
        options=list(arms)
        rows=[material(split,task,i,arm) for task in DOMAINS for i in range(n) for arm in options]
        name=f"materials-{split}-chain.json"
        R.write(name,rows)
        panels[name]=dict(n=len(rows),sha256=R.file_sha(R.OUT/name))
    import importlib.metadata
    R.write("plan.json",dict(run_id=R.RUN,models=M.MODELS,precision="BF16",license="Original fictional examples",
                            purpose="New independently seeded families after v1 discovery of behavior-interface failure",
                            thresholds=R.THRESHOLDS,materials=panels,
                            sources={Path(p).name:R.file_sha(Path(p)) for p in source},
                            runtime={p:importlib.metadata.version(p) for p in ("torch","accelerate","numpy","transformers","scipy","safetensors")},
                            capture="Qwen3-4B only, B24 fixed and 2 families/task/split all original coordinates if behavior admitted",
                            admission="Each task must pass all 16 factorial cells on pilot and current batch split; test behavior on all rows",
                            reference_models="Qwen3-14B and GLM4 on preselected indices 0 through 31 for each heldout split",
                            checked_interfaces="Full answer is separately sampled by natural generation on pilot only"))


def run(key):
    import torch
    import numpy as np
    plan=R.protocol()
    torch.set_num_threads(4)
    R.encode=encode
    with R.GpuLease(),torch.inference_mode():
        start=time.monotonic()
        info=R.identity(key)
        engine=R.Engine(key)
        try:
            R.write(f"{key}-model.json",dict(files=info,snapshot=R.digest(info),letters=engine.letters,placement=engine.placement,
                                            plan_sha=R.file_sha(R.OUT/"plan.json")))
            blocked=set()
            for split in SPLITS:
                if split=="discover" and key!="q4":
                    continue
                stem=f"{key}-{split}-chain"
                if (R.OUT/(stem+".json")).exists():
                    raise FileExistsError(stem)
                rows=R.load(R.OUT/f"materials-{split}-chain.json")
                if key!="q4" and split!="pilot":
                    rows=[row for row in rows if row["index"]<32]
                batches=list(R.blocks(rows,64 if key=="q4" else 64))
                results=[]
                begin=time.monotonic()
                R.stage_log(key,stem,0,len(batches),begin)
                for idx,batch in enumerate(batches,1):
                    raw,_=engine.batch(batch,"chain",False,[],generate=split=="pilot")
                    results.extend(raw)
                    R.stage_log(key,stem,idx,len(batches),begin)
                gates=R.cell_pass(rows,results,plan["thresholds"]["behavior"])
                R.write(stem+".json",dict(gate=gates,results=results,seconds=time.monotonic()-begin))
                failed={g["cell"][0] for g in gates if not g["pass_gate"]}
                if split=="pilot":
                    blocked=failed
                    continue
                if key!="q4":
                    continue
                keep=set(DOMAINS)-failed-blocked
                native_batches=list(R.blocks([row for row in rows if row["task"] in keep],64))
                lookup={r["id"]:r for r in results}
                stage=stem+"-native"
                R.stage_log(key,stage,0,len(native_batches),begin)
                for idx,batch in enumerate(native_batches,1):
                    family_sample=[i for i,r in enumerate(batch) if r["index"]<2]
                    raw,native=engine.batch(batch,"chain",True,family_sample)
                    assert all(r["option_logits"]==lookup[r["id"]]["option_logits"] for r in raw)
                    native.pop("prefix0")
                    np.savez(R.check_write(R.OUT/f"{stage}-{idx:03d}.npz"), **native,
                             ids=[r["id"] for r in raw],field_indices=family_sample)
                    R.stage_log(key,stage,idx,len(native_batches),begin)
                R.write(stage+".json",dict(eligible=sorted(keep),batches=len(native_batches)))
        finally:
            engine.close()
        R.stage_log(key,"complete",1,1,start)


if __name__=="__main__":
    R.setup()
    if sys.argv[1]=="freeze":
        assert not (R.OUT/"plan.json").exists()
        freeze()
    elif sys.argv[1]=="run":
        run(sys.argv[2])
