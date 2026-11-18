"""Isolate signed-clause selection as a second language-function holdout."""

from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
import random
import sys
import time

from kv_route_materials import OLD, generate as original
from kv_route_revised import arms
from kv_route_contrast import source_time
import kv_route_run as R
import kv_route_native as Native
import kv_route_audit as Audit


SUITES = ("discovery", "polarity")
PILOT = "music drama dance chess tennis golf rugby yoga"
CONFIRM = "carrot onion potato tomato bean cherry pumpkin banana berry mango olive garlic cucumber pepper broccoli cabbage lettuce squash"


def materials(pilot):
    split = "pilot" if pilot else "holdout"
    rng = random.Random("kv-route-v3/polarity/"+split)
    pairs = list(combinations(sorted((PILOT if pilot else CONFIRM).split()),2))
    rng.shuffle(pairs)
    rows = []
    for i,pair in enumerate(pairs[:8 if pilot else 96]):
        old,new = pair if rng.randrange(2) else pair[::-1]
        negative_first = i%2==0
        family = sha256(f"kv-route-v3/polarity/{split}/{'/'.join(pair)}".encode()).hexdigest()[:18]
        note = ["blue","red","green","yellow","black","white","orange","brown"][i%8]
        for take_new,swap in product((0,1), repeat=2):
            state,alt = (new,old) if take_new else (old,new)
            mapping = dict(zip((old,new),("Q","P") if swap else ("P","Q")))
            pos,neg = f"The list includes {state}.", f"The list does not include {alt}."
            report = "\n".join((neg,pos) if negative_first else (pos,neg))
            text = (
                "Read the two sentences and find the entry that is included. Ignore the excluded entry. "
                "First write the included entry itself as State. Then copy the separate reference word into Note. "
                "Finally use the Codebook to write the code corresponding to your State.\n\n"
                "Separate example of the response format:\nState: silver\nNote: black\nCode: Q\n\n"
                f"Sentences for this question:\n{report}\n"
                f"Reference word: {note}\nCodebook:\n{old} -> {mapping[old]}\n{new} -> {mapping[new]}\n\n"
                "Give the actual response now, exactly in this format:\n"
                "State: <included entry>\nNote: <reference word>\nCode: <code>\n"
                "Write only those three lines. P and Q are reserved for the Code line.")
            rows.append(dict(id=f"{family}-{take_new}{swap}",family=family,index=i,suite="polarity",
                             old=old,new=new,notes=[note],update=take_new,swap=swap,state=state,alt=alt,
                             code=mapping[state],alt_code=mapping[alt],voice="negative_first" if negative_first else "positive_first",
                             second_keeps=None,text=text))
    assert len(rows)==len({r["text"] for r in rows})==(32 if pilot else 384)
    return rows


def generate(suite):
    return original(suite) if suite=="discovery" else materials(False)


def configure():
    R.M = sys.modules[__name__]
    R.OUT = R.Base.ROOT / "tests/glm5/result/kv_route_20261115_v3"
    R.TEMP = R.Base.ROOT / "tests/glm5_temp/kv_route_20261115_v3"
    R.RUN = "KV-ROUTE-20261115-v3"
    R.setup()


def pilot():
    import torch

    torch.set_num_threads(4)
    rows = materials(True)
    R.write("pilot-materials.json",rows)
    start = time.monotonic()
    with R.Base.Previous.GpuLease(), torch.inference_mode():
        R.Base.Previous.stage_log("q4","polarity-pilot",0,1,start)
        model = R.Base.load_model("q4")
        try:
            obs = R.Task(model).natural(rows)
            R.write("pilot-behavior.json",dict(rows=obs,cells=R.gate(rows,obs)))
            R.Base.Previous.stage_log("q4","polarity-pilot",1,1,start)
        finally:
            model.close()


def freeze():
    from transformers import AutoTokenizer

    pilot = R.read("pilot-behavior.json")
    assert sum(i["correct"] for i in pilot["cells"])>=30
    tok = AutoTokenizer.from_pretrained(R.Base.ROOT/"models/hf"/R.Base.MODELS["q4"],local_files_only=True)
    for row in materials(False):
        R.Base.encode(tok,row)
    R.freeze()
    plan = R.read("plan.json")
    files = ("kv_route_revised.py","kv_route_audit.py","kv_route_contrast.py",Path(__file__).name)
    plan["sources"].update({x:R.sha(Path(__file__).with_name(x)) for x in files})
    plan["holdout"] = "96 new unordered word-pair families in signed-clause task x4 views; first-clause polarity balanced."
    plan["pilot_artifacts"] = {x:R.sha(R.OUT/x) for x in ("pilot-materials.json","pilot-behavior.json")}
    plan["selection_scope"] = "same already exposed 16 families; new polarity suite follows separate role behavior failure"
    plan["references"] = "Uses prior serial Qwen3-14B and GLM4 runtime reference scope from v1."
    plan["contrasts"]["other"] = "All v2 preregistered individual and trio arms, 2x2 raw means and source-time interaction."
    R.write("plan.json",plan)


def run():
    import torch

    plan = R.check()
    assert all(R.sha(R.OUT/name)==h for name,h in plan["pilot_artifacts"].items())
    execute = Native.one

    def flush_cache(*args):
        torch.cuda.empty_cache()
        return execute(*args)

    Native.specs = arms
    Native.one = flush_cache
    Native.execute()


def audit():
    Audit.main()
    doc = R.read("audit.json")
    stages = {s["suite"]:s for s in R.read("confirmation.json")["ledger"]}
    suites = [source_time(s["suite"],stages[s["suite"]],R.read(f"materials-{s['suite']}.json"),s)
              for s in doc["holdout"] if s["status"]=="complete"]
    R.write("matched-contrast.json",dict(suites=suites,plan_sha=R.sha(R.OUT/"plan.json"),audit_sha=R.sha(R.OUT/"audit.json")))


if __name__=="__main__":
    configure()
    {"pilot":pilot,"freeze":freeze,"run":run,"audit":audit}[sys.argv[1]]()
