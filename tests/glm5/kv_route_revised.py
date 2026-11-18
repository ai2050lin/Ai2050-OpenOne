"""Behavior calibration on disjoint authored vocabulary and revised locked materials."""

from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
import random
import sys
import time

from kv_route_materials import generate as old_generate, OLD, SUITES
import kv_route_run as R
import kv_route_native as Native
import kv_route_audit as Audit


WORDS = {
    "long": "stone metal paper clay wool glass cotton wood brick sand steel rubber plastic iron copper silver gold zinc",
    "multi": "park beach hotel station port village market airport store factory school bank museum farm hospital church stadium garden",
    "role": "doctor teacher lawyer nurse officer pilot driver judge guard clerk farmer worker soldier manager owner reporter engineer",
}
PILOT = {
    "long": "salt sugar cheese flour bread milk butter jam",
    "multi": "cave lake river house street valley island forest",
    "role": "agent coach broker hunter scout waiter priest monk",
}
PILOT2 = {
    "long": PILOT["long"],
    "multi": "pond tower cabin lane court lab base hall",
    "role": "cook poet artist author dean tutor spy miner",
}
CHOICES = "black white red blue green yellow brown orange".split()
ORIGINAL_ARMS = Native.specs


def arms():
    single = R.HEADS[:1]
    return ORIGINAL_ARMS() + [
        Native.Arm(f"single-{kind}-{source}-{clock}", kind, single, source, clock)
        for (source, clock), kind in product(
            (("state", "early"), ("note", "early"), ("note", "answer")), ("swap", "null"))
    ]


def prompts(suite, split):
    pilot = split.startswith("pilot")
    words = sorted(set((PILOT2 if split=="pilot2" else PILOT if pilot else WORDS)[suite].split()))
    rng = random.Random(f"kv-route-v2/{suite}/{split}")
    pairs = list(combinations(words, 2))
    rng.shuffle(pairs)
    for i, pair in enumerate(pairs[:8 if pilot else 96]):
        old, new = pair if rng.randrange(2) else pair[::-1]
        notes = rng.sample(CHOICES, 3 if suite == "long" else 1)
        family = sha256(f"kv-route-v2/{suite}/{split}/{'/'.join(pair)}".encode()).hexdigest()[:18]
        for update, swap in product((0, 1), repeat=2):
            state, alt = (new, old) if update else (old, new)
            codes = dict(zip((old, new), ("Q", "P") if swap else ("P", "Q")))
            keeps = voice = None
            if suite == "long":
                action = "Replace the original with the replacement value." if update else "Keep the original; do not use the replacement value."
                facts = f"Read a notice registry.\nOriginal value: {old}\nReplacement value: {new}\nAction: {action}"
                goal = "Resolve the live value after the actual update instruction."
            elif suite == "multi":
                keeps = i % 2 == 0
                first = state if keeps else alt
                second = "Keep the current label from Step 1." if keeps else f"Replace the current label with {state}."
                facts = (f"Read a staff roster with a two-step update. At the start, its label is {old}. "
                         f"The two available labels are {old} and {new}.\n"
                         f"Step 1: replace the current label with {first}.\nStep 2: {second}")
                goal = ("Apply both steps in order to resolve the live value. "
                        "If Step 2 says keep, use the Step 1 word. If Step 2 says replace, use the Step 2 word.")
            else:
                voice = "passive" if i % 2 else "active"
                sentence = (f"The {alt} handed a parcel to the {state}." if voice == "active" else
                            f"The {state} was handed a parcel by the {alt}.")
                facts = f"Read this report of a parcel handoff.\n{sentence}"
                goal = ("Find the occupation of the person who received the parcel. "
                        "In an active sentence take the occupation after 'to the'. "
                        "In a passive sentence take the occupation immediately before 'was handed'.")
            note_text = " ".join(notes)
            note_example = "white red blue" if suite == "long" else "blue"
            text = (
                f"{goal} Encode that live value using this record's Codebook.\n"
                "For illustration only, if the live value were lemon and its code were Q, the format would be:\n"
                f"State: lemon\nNote: {note_example}\nCode: Q\n\n"
                f"{facts}\nReference words: {note_text}\n"
                f"Codebook:\n{old} -> {codes[old]}\n{new} -> {codes[new]}\n\n"
                "Now write the actual result for this record in three lines. "
                f"On the State line write the live word itself ({old} or {new}). "
                "Copy the reference words in their given order on the Note line. "
                "On the Code line write the code of the live word. P and Q are reserved for the Code line. "
                "Follow this exact format:\nState: <live word>\nNote: <reference words>\nCode: <code>")
            yield dict(id=f"{family}-{update}{swap}", family=family, suite=suite, index=i,
                       old=old, new=new, notes=notes, update=update, swap=swap, state=state, alt=alt,
                       code=codes[state], alt_code=codes[alt], second_keeps=keeps, voice=voice, text=text)


def generate(suite):
    if suite == "discovery":
        return old_generate(suite)
    rows = list(prompts(suite, "confirm"))
    assert len(rows) == len({r["text"] for r in rows}) == 384
    return rows


def pilot(split):
    import torch

    torch.set_num_threads(4)
    panel = {s: list(prompts(s, split)) for s in SUITES[1:]}
    R.write(f"{split}-source.json", dict(source=R.sha(Path(__file__)), materials=panel))
    with R.Base.Previous.GpuLease(), torch.inference_mode():
        model = R.Base.load_model("q4")
        try:
            g = R.Task(model)
            result = {}
            start = time.monotonic()
            for i, (suite, rows) in enumerate(panel.items(), 1):
                obs = g.natural(rows)
                result[suite] = dict(cells=R.gate(rows, obs), rows=obs)
                R.write(f"{split}-behavior.json", result)
                R.Base.Previous.stage_log("q4", "behavior-only-pilot", i, len(panel), start)
        finally:
            model.close()


def main():
    R.M = sys.modules[__name__]
    R.OUT = R.Base.ROOT / "tests/glm5/result/kv_route_20261115_v2"
    R.TEMP = R.Base.ROOT / "tests/glm5_temp/kv_route_20261115_v2"
    R.RUN = "KV-ROUTE-20261115-v2"
    R.setup()
    step = sys.argv[1]
    if step == "pilot2":
        pilot(step)
    elif step == "freeze":
        R.freeze()
        plan = R.read("plan.json")
        plan["sources"][Path(__file__).name] = R.sha(Path(__file__))
        plan["references"] = "Serial Qwen3-14B and GLM4 behavior reference artifacts live in v1; not rerun here."
        plan["pilot_artifacts"] = {f"{split}-{kind}.json":R.sha(R.OUT / f"{split}-{kind}.json")
                                   for split in ("pilot","pilot2") for kind in ("source","behavior")}
        plan["head19_matched"] = "Same three primary contrasts as the trio, using the head19-only arms and identical sign tests/thresholds."
        plan["factorial_descriptive"] = "Report native-minus-rotation contrast at all four source/time cells and source×time interaction for both groups."
        plan["failure_handling"] = "Suite gates do not pool cells. For eligible suites, incorrect rows are retained at zero scores for each arm."
        R.write("plan.json", plan)
    elif step == "run":
        plan = R.check()
        assert all(R.sha(R.OUT / name)==h for name,h in plan["pilot_artifacts"].items())
        Native.specs = arms
        Native.execute()
    elif step == "audit":
        Audit.main()


if __name__ == "__main__":
    main()
