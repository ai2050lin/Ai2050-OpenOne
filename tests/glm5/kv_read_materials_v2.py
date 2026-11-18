"""Use already exposed calibration examples for screening and new holdouts for validation."""
from pathlib import Path
import json
import kv_read_materials as Original

SUITES=Original.SUITES
PRIOR=Path(__file__).resolve().parents[2]/"tests/glm5/result/own_state_explicit_20261115_v2/materials-confirm.json"


def generate(suite):
    if suite=="discovery":
        data=json.loads(PRIOR.read_text(encoding="utf-8"))
        return [dict(r,suite="discovery",update=r["factors"][0],swap=r["factors"][2],
                     aux=None,second_keeps=None)
                for r in data if r["index"]<4 and r["factors"][1]==r["index"]%2]
    result=[]
    for row in Original.generate(suite):
        old,new=row["old"],row["new"]
        if suite=="frames":
            instruction=(f"Replace the prior label with the proposed label." if row["update"] else
                         "Keep the prior label; do not use the proposed label.")
            facts=(f"A curator is processing a museum catalog revision.\n"
                   f"Label currently in force: {old}.\nProposed catalog label: {new}.\n"
                   f"Catalog decision: {instruction}")
        elif suite=="multi":
            first=row["state"] if row["second_keeps"] else row["alt"]
            second=("Keep the current label established by Step 1; do not reset it." if row["second_keeps"] else
                    f"Replace the current label with {row['state']}.")
            facts=(f"Read a staff roster with a two-step update. At the start, its label is {old}. "
                   f"The two available labels are {old} and {new}.\n"
                   f"First update: replace the current label with {first}.\n"
                   f"Second update: {second}\nApply both steps in order.")
        elif suite=="gap":
            operation=("Replace the original with the replacement value." if row["update"] else
                       "Keep the original; do not use the replacement value.")
            facts=(f"Read a notice registry.\nOriginal value: {old}\nReplacement value: {new}\n"
                   f"Action: {operation}\nReference card color: {row['aux']}.\n"
                   "The unrelated reference card color goes only in the Note line.")
        mapping=[("Q","P"),("P","Q")][not row["swap"]]
        codebook=f"Codebook:\n{old} -> {mapping[0]}\n{new} -> {mapping[1]}"
        components="State: <live value>\nCode: <code from the current codebook>"
        if suite=="gap":
            components="State: <live value>\nNote: <reference card color>\nCode: <code from the current codebook>"
        text=("Resolve the live value after the actual update instructions. Encode that live value using this record's Codebook."
              f"\n\n{facts}\n{codebook}\n\nWrite these fields on separate lines:\n{components}\n"
              "Write only the requested fields, in that order, without explanation or punctuation after the code. "
              "State is the live value after following the update instructions.")
        tag=row["id"].replace("-","v2-",1)
        result.append(dict(row,text=text,id=tag,family=row["family"]+"v2"))
    return result
