"""Authored new holdouts for source-, time-, and content-matched native reads."""
from itertools import permutations, product
import hashlib
import json
from pathlib import Path
import random

SUITES=("discovery","long","multi","role")
TERMS={
    "long":"oak maple pine cedar wheat ash palm bamboo apple pear plum peach grape rice lemon lime",
    "multi":"office kitchen bedroom basement garage lobby library cafe classroom studio dorm plaza corner tower cabin bridge",
    "role":"poet singer actor dancer tailor artist barber painter banker miner baker chef writer merchant captain student",
}
OLD=Path(__file__).resolve().parents[2]/"tests/glm5/result/kv_read_20261115_v2"


def generate(suite):
    if suite=="discovery":
        rows=json.loads((OLD/"materials-gap.json").read_text(encoding="utf-8"))
        return [dict(r,suite="discovery",notes=[r["aux"]],voice=None) for r in rows if r["index"]<16]
    vocab=TERMS[suite].split()
    rng=random.Random(f"20261115-native-kv-route-v1/{suite}")
    ordered=list(permutations(vocab,2))
    rng.shuffle(ordered)
    rows=[]
    for i,(old,new) in enumerate(ordered[:96]):
        note=rng.sample(["red","blue","green","white","black","silver","brown","gold"],3 if suite=="long" else 1)
        family=hashlib.sha256(f"native-kv-route-v1/{suite}/{old}/{new}".encode()).hexdigest()[:18]
        for update,swap in product((0,1),repeat=2):
            state=(old,new)[update]
            alt=(new,old)[update]
            codes=dict(zip((old,new),("Q","P") if swap else ("P","Q")))
            keeper=None
            voice=None
            if suite=="long":
                instruction="Replace the original with the proposal." if update else "Keep the original and ignore the proposal."
                story=(f"A museum record has original classification {old} and proposed classification {new}.\n"
                       f"Decision: {instruction}\nResolve the current classification.")
            elif suite=="multi":
                keeper=i%2==0
                first=state if keeper else alt
                second=("Keep the current value from the first update." if keeper else f"Replace the current value with {state}.")
                story=(f"The assigned meeting location starts at {old}. The two available locations are {old} and {new}.\n"
                       f"First update: replace the current label with {first}.\n"
                       f"Second update: {second}\nApply the updates in order and report only the resulting location.")
            else:
                voice="passive" if i%2 else "active"
                sentence=(f"The {state} was given a permit by the {alt}." if voice=="passive" else
                          f"The {alt} gave a permit to the {state}.")
                story=f"Find the person who received the permit, stated in this sentence:\n{sentence}"
            note_in=" ".join(note)
            prompt=(f"{story}\nCodebook:\n{old} -> {codes[old]}\n{new} -> {codes[new]}\n"
                    f"Unrelated reference words: {note_in}\n"
                    "Output exactly three lines without extra punctuation. State: <requested value>. "
                    "On the next line, Note: <copy the unrelated reference words in the given order, space separated>. "
                    "On the final line, Code: <code of the requested value>. Do not use reference words to select the state or the code.")
            rows.append(dict(id=f"{family}-{update}{swap}",family=family,suite=suite,index=i,old=old,new=new,notes=note,
                             update=update,swap=swap,state=state,alt=alt,code=codes[state],alt_code=codes[alt],
                             second_keeps=keeper,voice=voice,text=prompt))
    assert len({r["text"] for r in rows})==len(rows)
    return rows
