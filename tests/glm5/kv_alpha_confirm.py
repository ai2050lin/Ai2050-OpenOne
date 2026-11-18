"""Run the frozen weighted-write protocol on fresh Phase 2939-style polarity families."""

from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
import random
import sys

import kv_alpha_run as C
from kv_alpha_materials import N, SEED


PANELS=("polarity",)
WORDS={"polarity":"cinnamon ginger basil honey chocolate coffee tea vinegar sausage bacon cracker pretzel cookie biscuit brownie muffin pancake waffle cereal popcorn sauce salsa mayonnaise pickle shrimp salmon tuna pudding donut latte espresso"}
PILOT={"polarity":"dog cat horse cow pig goat sheep hen rooster duck goose"}
NOTES=("blue","red","green","yellow","black","white","orange","brown")


def make(panel,split,words):
    pilot=split=="pilot"
    rng=random.Random(f"{SEED}/fresh-polarity/{split}")
    root_pairs=list(combinations(sorted(words),2))
    rng.shuffle(root_pairs)
    rows=[]
    for i,pair in enumerate(root_pairs[:12 if pilot else N]):
        old,new=pair if rng.getrandbits(1) else pair[::-1]
        negative_first=i%2==0
        note=NOTES[(i//2+i%2)%8]
        family=sha256(f"weighted-value-fresh-polarity/{'/'.join(pair)}".encode()).hexdigest()[:18]
        for take_new,swap in product((0,1),repeat=2):
            state,alt=(new,old) if take_new else (old,new)
            mapping=dict(zip((old,new),("Q","P") if swap else ("P","Q")))
            pos=f"The list includes {state}."
            neg=f"The list does not include {alt}."
            report="\n".join((neg,pos) if negative_first else (pos,neg))
            text=(
                "Read the two sentences and find the entry that is included. Ignore the excluded entry. "
                "First write the included entry itself as State. Then copy the separate reference word into Note. "
                "Finally use the Codebook to write the code corresponding to your State.\n\n"
                "Separate example of the response format:\nState: silver\nNote: black\nCode: Q\n\n"
                f"Sentences for this question:\n{report}\n"
                f"Reference word: {note}\nCodebook:\n{old} -> {mapping[old]}\n{new} -> {mapping[new]}\n\n"
                "Give the actual response now, exactly in this format:\n"
                "State: <included entry>\nNote: <reference word>\nCode: <code>\n"
                "Write only those three lines. P and Q are reserved for the Code line."
            )
            rows.append(dict(id=f"{family}-{take_new}{swap}",family=family,index=i,suite=panel,split=split,
                             old=old,new=new,notes=[note],update=take_new,swap=swap,state=state,alt=alt,
                             code=mapping[state],alt_code=mapping[alt],voice="negative_first" if negative_first else "positive_first",
                             second_keeps=None,text=text))
    assert len(rows)==len({r["text"] for r in rows})==(48 if pilot else 4*N)
    return rows


def attach():
    C.M=sys.modules[__name__]
    C.RUN="KV-ALPHA-20261115-v2"
    C.OUT=C.ROOT/"tests/glm5/result/kv_alpha_20261115_v2"
    C.TMP=C.ROOT/"tests/glm5_temp/kv_alpha_20261115_v2"
    overlay="tests/glm5/"+Path(__file__).name
    if overlay not in C.SOURCES:
        C.SOURCES=(*C.SOURCES,overlay,"tests/glm5/kv_alpha_audit.py")


def freeze():
    one=C.OUT.with_name("kv_alpha_20261115_v1")
    previous=[]
    for f in ("materials-polarity.json","materials-update.json"):
        previous += C.R.Base.Previous.load(one/f)
    previous += [r for rr in C.R.Base.Previous.load(one/"pilot-materials.json")["panels"].values() for r in rr]
    new=make("polarity","confirm",C.vocabulary(C.tok())["polarity"])
    pairs=lambda rr:{tuple(sorted((r["old"],r["new"]))) for r in rr}
    assert not pairs(new)&pairs(previous)
    C.freeze()
    plan=C.R.read("plan.json")
    plan["hypothesis"]["population_prediction"] = (
        "Separate preregistered population endpoint. On the same matched cohort, the 95% family bootstrap CIs of "
        "the unrounded alpha-only simulator raw bias (predicted minus native) and the native matched residual "
        "must each lie within ±.25 logits, with >=153 eligible families. Require native raw reversal "
        "<=-.25, upper CI below zero, and simulator raw mean upper CI below zero. "
        "The original individual-case RMSE, bias and rank test remains a distinct stricter endpoint."
    )
    C.R.write("plan.json",plan)


if __name__=="__main__":
    attach()
    if sys.argv[1]=="audit":
        from kv_alpha_audit import audit
        out=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else C.OUT
        frozen=Path(sys.argv[3]).resolve() if len(sys.argv)>3 else C.OUT
        audit(out,frozen)
    elif sys.argv[1]=="freeze":
        C.configure()
        freeze()
    else:
        C.main()
