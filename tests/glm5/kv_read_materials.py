"""New fictional non-arithmetic update, read, and report families."""
from itertools import permutations, product
import hashlib
import random

SUITES=("discovery","frames","multi","gap")
SEEDS={
    "discovery":"orange purple brown yellow gray green black white pink silver gold blue red copper bronze violet",
    "frames":"forest island coast river lake mountain desert garden harbor beach valley hill canyon cave bay reef",
    "multi":"doctor teacher nurse driver pilot clerk guide baker farmer ranger lawyer writer guard coach runner dancer",
    "gap":"winter summer spring autumn morning evening noon night dawn midnight east west south north today tomorrow",
}


def generate(suite):
    words=SEEDS[suite].split()
    pairs=list(permutations(words,2))
    rng=random.Random(f"ai2050-nov15-KV-read-v1/{suite}")
    rng.shuffle(pairs)
    n=8 if suite=="discovery" else 96
    result=[]
    for i,(old,new) in enumerate(pairs[:n]):
        family=hashlib.sha256(f"kv-read-v1/{suite}/{old}/{new}".encode()).hexdigest()[:18]
        aux=rng.choice(["blue","green","white","red"])
        for update,swap in product((0,1),repeat=2):
            answer=(old,new)[update]
            other=(new,old)[update]
            codes=dict(zip((old,new),("P","Q") if not swap else ("Q","P")))
            intro="Resolve a fictional registry's live label. Then report the live label and its code."
            if suite=="frames":
                story=(f"A curator is indexing a label pair, {old} and {new}. The card in force before today used {old}. "
                       + (f"Today's bulletin supersedes that label with {new}." if update else
                          f"Today's bulletin leaves {old} in force; {new} is a discarded proposal."))
            elif suite=="multi":
                second_keeps = i % 2 == 0
                intermediate = answer if second_keeps else other
                second = ("keep the label established by the first update." if second_keeps else
                          f"replace the current label with {answer}.")
                story=(f"The staff roster begins with label {old}. The second allowed label is {new}.\n"
                       f"First update: replace the current label with {intermediate}.\n"
                       f"Second update: {second}\n"
                       "Apply both updates in order and report the label after the second update.")
            elif suite=="gap":
                story=(f"The live notice initially used the label {old}. The record's proposed label is {new}.\n"
                       + (f"Replace the original with {new}." if update else f"Keep the original {old}.")
                       +f"\nAn unrelated reference card has color {aux}. The color does not choose the live label or its code.")
            else:
                story=(f"Original registry label: {old}.\nReplacement label: {new}.\n"
                       + ("Use the replacement label." if update else "Keep the original label."))
            codebook=f"Codebook:\n{old} -> {codes[old]}\n{new} -> {codes[new]}"
            output=("Write exactly three lines: State: <live label>, Note: <reference card color>, Code: <code of the live label>. "
                    "The Note line is independent of the live label." if suite=="gap" else
                    "Write exactly two lines: State: <live label> and Code: <code of the live label>.")
            text=f"{intro}\n\n{story}\n\n{codebook}\n\n{output} Do not add explanations or punctuation."
            result.append(dict(id=f"{family}-{update}{swap}",family=family,suite=suite,index=i,update=update,swap=swap,
                               old=old,new=new,state=answer,alt=other,code=codes[answer],alt_code=codes[other],
                               aux=aux if suite=="gap" else None,second_keeps=(i%2==0) if suite=="multi" else None,text=text))
    assert len(set(r["text"] for r in result))==len(result)
    return result
