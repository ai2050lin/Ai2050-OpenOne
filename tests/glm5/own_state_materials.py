"""Frozen, authored non-arithmetic field revision tasks."""
import hashlib
import itertools
import random

SPLITS = {"pilot": 12, "confirm": 96}
TASKS = ("dispatch", "permit")
FACTORS = tuple(itertools.product((0, 1), repeat=3))
TERMS = {
    ("pilot", "dispatch"): "blue red green white black pink gold brown",
    ("pilot", "permit"): "open closed active paused ready blocked valid pending",
    ("confirm", "dispatch"): "north south west east river lake hill forest coast valley garden tower field beach park island",
    ("confirm", "permit"): "draft final basic full guest member public private temporary permanent limited standard bronze silver copper orange",
}


def build(split, task, n=None):
    vocab = TERMS[split, task].split()
    combos = list(itertools.permutations(vocab, 2))
    rng = random.Random(f"20261115/own-state/{split}/{task}")
    rng.shuffle(combos)
    names = ("courier route", "delivery destination") if task == "dispatch" else ("visitor permit", "access category")
    rows = []
    for i, (old, new) in enumerate(combos[:n or SPLITS[split]]):
        family = hashlib.sha256(f"{split}/{task}/{old}/{new}".encode()).hexdigest()[:16]
        # The unrelated training example uses one of the two candidate values.
        example = rng.choice((old, new))
        for update, example_after, swap in FACTORS:
            current = (old, new)[update]
            alternate = (new, old)[update]
            codes = dict(zip((old, new), ("P", "Q") if not swap else ("Q", "P")))
            intro = (
                f"A live {names[0]} has an original {names[1]} and a proposed revision. "
                f"Use the proposed revision only when the decision says APPLY. When it says KEEP, use the original. "
                "A training example is unrelated to the live record. Report the live value and its code.\n\n"
                f"Original: {old}\nProposed revision: {new}\n"
            )
            action = "Decision: " + ("APPLY\n" if update else "KEEP\n")
            decoy = f"Training example: {example}\n"
            body = action + decoy if example_after else decoy + action
            book = f"Codebook: {old} = {codes[old]}; {new} = {codes[new]}.\n"
            form = (
                f"\nUse exactly two lines. On the first line write State: followed by the live {names[1]}. "
                "On the second line write Code: followed by the matching code. Stop after the code."
            )
            rows.append(dict(id=family+"-"+str(update)+str(example_after)+str(swap),
                             family=family,index=i,split=split,task=task,
                             factors=[update,example_after,swap],old=old,new=new,example=example,
                             state=current,alt=alternate,code=codes[current],alt_code=codes[alternate],
                             text=intro+body+book+form))
    return rows


def all_rows(split):
    return [row for task in TASKS for row in build(split, task)]
