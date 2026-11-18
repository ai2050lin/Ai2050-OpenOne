"""Prospective v2 after the symbolic APPLY pilot failed; all families are new."""
import hashlib
from itertools import permutations
from pathlib import Path
import random
import sys

import own_state_run as R
import own_state_materials as Base

R.OUT=R.OUT.with_name("own_state_explicit_20261115_v2")
R.TEMP=R.TEMP.with_name("own_state_explicit_20261115_v2")
R.RUN="OWN-STATE-EXPLICIT-20261115-v2"
SPLITS=Base.SPLITS
TASKS=Base.TASKS
FACTORS=Base.FACTORS
VOCAB={
    ("pilot","dispatch"):"dog cat fish bird horse wolf bear tiger",
    ("pilot","permit"):"allowed denied exempt restricted approved revoked accepted rejected",
    ("confirm","dispatch"):"port london paris station cairo japan china texas terminal berlin dock chicago canada india harbor brazil",
    ("confirm","permit"):"alpha beta gamma delta senior junior civil military student adult teacher doctor owner worker volunteer resident",
}


def rows(split,task):
    names=list(permutations(VOCAB[split,task].split(),2))
    rng=random.Random("state explicit v2 "+split+task)
    rng.shuffle(names)
    result=[]
    for i,(old,new) in enumerate(names[:SPLITS[split]]):
        example=rng.choice((old,new))
        family=hashlib.sha256(f"explicit-v2/{split}/{task}/{old}/{new}".encode()).hexdigest()[:16]
        for u,d,m in FACTORS:
            selected=(old,new)[u]
            alt=(new,old)[u]
            codebook=dict(zip((old,new),("P","Q") if not m else ("Q","P")))
            desc="parcel routing record" if task=="dispatch" else "building access record"
            intro=(f"Read one {desc}. Resolve the live value according to the Action line. "
                   "Then encode the live value with this record's Codebook. The training example supplies no action or code.\n"
                   "Write exactly two lines: State: <live value> and Code: <code from the current codebook>. "
                   "Write no explanation or extra punctuation.\n\n"
                   f"Original value: {old}\nReplacement value: {new}\n")
            action=("Action: Replace the original with the replacement value.\n" if u else
                    "Action: Keep the original; do not use the replacement value.\n")
            noise=f"Unrelated training example value: {example}\n"
            order=action+noise if d else noise+action
            table=f"Codebook:\n{old} -> {codebook[old]}\n{new} -> {codebook[new]}\n"
            note="\nNow write the two lines for this record. State is the value after following the Action line."
            result.append(dict(id=family+"-"+"".join(str(k) for k in (u,d,m)),family=family,index=i,task=task,
                               split=split,factors=[u,d,m],old=old,new=new,example=example,state=selected,
                               alt=alt,code=codebook[selected],alt_code=codebook[alt],text=intro+order+table+note))
    return result


def all_rows(split):
    return [r for task in TASKS for r in rows(split,task)]


def select():
    R.Materials=sys.modules[__name__]


if __name__=="__main__":
    select()
    R.configure()
    action=sys.argv[1]
    if action=="freeze":
        R.freeze()
        R.write("v2-amendment.json",dict(generator_sha=R.digest(Path(__file__)),
                                      previous_protocol_sha=R.digest(R.OUT.with_name("own_state_20261115_v1")/"plan.json"),
                                      rationale="First version symbolic APPLY failed free generation. Explicit action instructions, new families; identical behavior cutoffs."))
    elif action=="behavior":
        R.behavior(sys.argv[2])
