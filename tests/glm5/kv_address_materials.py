"""Cross lexical assignment, generated field order, and the instruction context."""

from hashlib import sha256
from itertools import combinations, product
import random


SEED=29402026
SIZE={"pilot":12,"discover":72,"confirm":192}
WORDS={
    "pilot":"bread butter milk honey apple pear lemon lime sugar cheese rice beef chicken ham soup salt flour pasta bacon salad cake",
    "discover":"desk mirror towel chair lamp bottle basket glove jacket coat shirt hat skirt belt sock brush bench rug curtain pillow box blanket shoe spoon plate clock phone pen pencil ruler",
    "confirm":"river lake pond sea ocean harbor valley hill cliff mountain cave canyon forest grove meadow island beach coast shore creek stream waterfall bridge tunnel marsh field plain desert prairie jungle swamp bay reef lagoon dune spring trail path road farm garden park plaza",
}


def material(stage,tok):
    words=[w for w in WORDS[stage].split() if len(tok.encode(" "+w,add_special_tokens=False))==1]
    pairs=list(combinations(words,2))
    rng=random.Random(f"address/{SEED}/{stage}")
    rng.shuffle(pairs)
    out=[]
    for a,b in pairs[:SIZE[stage]]:
        other=rng.choice([w for w in words if w not in (a,b)])
        family=sha256(f"address/{stage}/{a}/{b}".encode()).hexdigest()[:16]
        codebook=rng.randrange(2)
        reverse,adopt,map_reverse=[rng.randrange(2) for _ in range(3)]
        for context,order,identity in product((0,1),repeat=3):
            state,note=(b,a) if identity else (a,b)
            correct,alt=("Q","P") if codebook else ("P","Q")
            orig,proposal=(other,state) if adopt else (state,other)
            if context==0:
                positive=f"The inventory includes {state}."
                negative=f"The inventory does not include {other}."
                facts="\n".join((negative,positive) if reverse else (positive,negative))
                ask="Read the two sentences. The entry that is included is the State."
            else:
                decision="Adopt the proposal." if adopt else "Retain the original."
                facts=f"Original: {orig}.\nProposal: {proposal}.\nDecision: {decision}"
                ask="Apply the decision to the register. The live entry after the decision is the State."
            if map_reverse:
                code=f"{other} -> {alt}\n{state} -> {correct}"
            else:
                code=f"{state} -> {correct}\n{other} -> {alt}"
            first,second=("Note","State") if order else ("State","Note")
            layout=f"{first}: <{'reference word' if order else 'live entry'}>\n{second}: <{'live entry' if order else 'reference word'}>\nCode: <P or Q>"
            example="Note: black\nState: silver" if order else "State: silver\nNote: black"
            text=(f"{ask} The Note is an independent reference word. The Code is the Codebook entry for the State.\n\n"
                  f"Separate format example:\n{example}\nCode: Q\n\n"
                  f"Record for this question:\n{facts}\nReference word: {note}\nCodebook:\n{code}\n\n"
                  f"Write the actual answer in exactly this order, without extra text:\n{layout}\n"
                  "Write the actual live word on the State line. Write the reference word on the Note line. "
                  "Use the letters P and Q only on the Code line.")
            out.append(dict(id=f"{family}-{context}{order}{identity}",family=family,stage=stage,
                            context=context,order=order,identity=identity,codebook=codebook,
                            reverse=reverse,adopt=adopt,map_reverse=map_reverse,
                            state=state,note=note,alt=other,code=correct,alt_code=alt,first=first,text=text))
    assert len(out)==8*SIZE[stage] and len({r["text"] for r in out})==len(out)
    return out


def flip_id(r,factor):
    c,o,i=(r[name]^(name==factor) for name in ("context","order","identity"))
    return f"{r['family']}-{c}{o}{i}"
