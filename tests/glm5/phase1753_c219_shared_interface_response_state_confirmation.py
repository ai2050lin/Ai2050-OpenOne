#!/usr/bin/env python3
"""C219: shared-interface, new-lexicon confirmation of C216 response states."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1739_c205_response_ecology_common as common
import phase1750_c216_multi_family_conditional_response_state as c216
import phase1751_c217_reworded_response_state_validation as c217
import phase1571_c098_observation_first_graph_campaign as graph_base

core = common.core
OUT = common.RESULT / "phase1753_c219_shared_interface_response_state_confirmation"
PHASE, CAMPAIGN = 1753, "C219"
BATCH = 6
UNITS = [
    ("Mira", "beacon", "Jonas", "folder", "secured", "sorted", "badger", "mammal", "animal", "chisel", "tool"),
    ("Soren", "ledger", "Keira", "bottle", "checked", "filled", "daisy", "flower", "plant", "drum", "instrument"),
    ("Yara", "sensor", "Lucan", "banner", "mounted", "carried", "scooter", "vehicle", "machine", "spoon", "utensil"),
    ("Emil", "cabinet", "Zara", "viola", "locked", "played", "granite", "rock", "material", "heron", "bird"),
    ("Noor", "parcel", "Silas", "curtain", "tagged", "opened", "tulip", "flower", "plant", "wrench", "tool"),
    ("Tarin", "tablet", "Maeve", "helmet", "charged", "stored", "lizard", "reptile", "animal", "flute", "instrument"),
    ("Vera", "archive", "Ronan", "candle", "indexed", "lit", "maple", "tree", "plant", "rake", "tool"),
    ("Inez", "module", "Arlo", "canvas", "tested", "painted", "quartz", "mineral", "material", "swan", "bird"),
]


def options(correct: str, wrong: str, order: int):
    return ((f"(A) {correct} (B) {wrong}", 0) if order == 1 else (f"(A) {wrong} (B) {correct}", 1))


def wrap(record1: str, record2: str, request: str, choice: str) -> str:
    return f"Record one: {record1}. Record two: {record2}. Request: {request}. Candidates: {choice}. Return only A or B."


def add(rows: list[dict], case_id: str, arm: str, unit: int, a: int, b: int, order: int, records: tuple[str, str], request: str, roles: dict, correct: str, wrong: str):
    choice, gold = options(correct, wrong, order)
    first, second = records if b == 0 else records[::-1]
    rows.append({"case_id": case_id, "arm": arm, "unit": unit, "partition": "confirmation" if unit < 4 else "fresh", "factor_a": a, "factor_b": b, "order": order, "gold_position": gold, "prompt": wrap(first, second, request, choice), "role_values": roles})


def material() -> list[dict]:
    rows = []
    for unit, values in enumerate(UNITS):
        agent, obj, distractor, other_obj, verb, other_verb, node, middle, parent, other_node, other_parent = values
        for a, b, order in itertools.product((0, 1), (0, 1), (1, -1)):
            target = f"{agent} {verb} the {obj}" if a == 0 else f"the {obj} was {verb} by {agent}"
            add(rows, f"c219-agent-{unit:02d}-{a}{b}-{order:+d}", "agent_surface", unit, a, b, order, (target, f"{distractor} {other_verb} the {other_obj}"), f"person linked to {obj}", {"primary": agent, "secondary": distractor, "relation": verb, "context": obj, "query": obj}, agent, distractor)

            target = f"{node} belongs directly to {parent}" if a == 0 else f"{node} belongs to {middle}, and {middle} belongs to {parent}"
            if b:
                target += f"; {node} also belongs directly to {parent}"
            add(rows, f"c219-path-{unit:02d}-{a}{b}-{order:+d}", "type_path", unit, a, b, order, (target, f"{other_node} belongs to {other_parent}"), f"broad class linked to {node}", {"primary": node, "secondary": middle if a else other_node, "relation": "belongs", "context": parent, "query": node}, parent, other_parent)

            target = f"{agent} owns the {obj}" if a == 0 else f"the {obj} belongs to {agent}"
            relation = "owns" if a == 0 else "belongs"
            add(rows, f"c219-possess-{unit:02d}-{a}{b}-{order:+d}", "possession_surface", unit, a, b, order, (target, f"{distractor} owns the {other_obj}"), f"owner linked to {obj}", {"primary": agent, "secondary": distractor, "relation": relation, "context": obj, "query": obj}, agent, distractor)

            target = f"{agent} is taller than {distractor}" if a == 0 else f"{distractor} is shorter than {agent}"
            relation = "taller" if a == 0 else "shorter"
            add(rows, f"c219-compare-{unit:02d}-{a}{b}-{order:+d}", "comparison_surface", unit, a, b, order, (target, f"the {obj} is beside the {other_obj}"), "person with greater height", {"primary": agent, "secondary": distractor, "relation": relation, "context": distractor, "query": "height"}, agent, distractor)

            target = f'"{node}" means "{parent}"' if a == 0 else f'"{node}" translates as "{parent}"'
            relation = "means" if a == 0 else "translates"
            add(rows, f"c219-translate-{unit:02d}-{a}{b}-{order:+d}", "translation_surface", unit, a, b, order, (target, f'"{other_node}" means "{other_parent}"'), f"value linked to {node}", {"primary": node, "secondary": other_node, "relation": relation, "context": parent, "query": node}, parent, other_parent)
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidates): raise RuntimeError(candidates)
    compiled = []
    for item in rows:
        ids = core.chat_ids(tokenizer, "Use the two records and request. Return exactly A or B.", item["prompt"])
        positions = {}
        for role, value in item["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans: raise RuntimeError((item["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**item, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def contract() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load((common.RESULT / "phase1752_c218_cross_surface_response_state_atlas/audit/independent_final_audit.json"))
    rows = material(); compiled = compile_rows(graph_base.tokenizer(), rows)
    checks = {"authorization": parent["all_checks_passed"], "cases": len(rows) == 320, "hidden": sum(item["order"] == 1 for item in rows) == 160, "arms": {item["arm"] for item in rows} == set(c216.ARMS), "new_units": len(UNITS) == 8, "candidate_balance": sum(item["gold_position"] == 0 for item in rows) == 160, "width": max(len(item["prompt_ids"]) for item in compiled) <= 96}
    if not all(checks.values()): raise RuntimeError({"checks": checks, "max_width": max(len(item["prompt_ids"]) for item in compiled)})
    OUT.mkdir(parents=True); core.write_rows(OUT / "material/cases.jsonl", rows); core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "shared_interface_response_state_confirmation_frozen", "model": "Qwen3-4B BF16 CUDA nonquantized", "source_templates": "unchanged C216 discovery templates", "shared_interface": "identical Record one / Record two / Request / Candidates wrapper across all arms", "new_lexical_units": 8, "partitions": {"confirmation": [0,1,2,3], "fresh": [4,5,6,7]}, "cases": 320, "hidden_rows": 160, "behavior_floor": 0.65, "classification_gate": {"each_partition_accuracy_min": 0.60, "each_partition_arms_at_or_above_half_min": 3}, "claim_boundary": "shared wrapper reduces but does not remove lexical, syntactic, role-span or operation-specific cues", "forbidden": ["attention", "MLP", "weights", "PCA", "template refitting", "post-reveal arm removal"], "producer_sha256": core.sha(Path(__file__)), "authorization": "single_Qwen3_run_all_arms_then_frozen_template_reveal"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks, "max_width": max(len(item["prompt_ids"]) for item in compiled)}, indent=2))


@torch.inference_mode()
def run() -> None:
    rows = core.rows(OUT / "compiled/qwen3.jsonl"); hidden_rows = [item for item in rows if item["order"] == 1]; (OUT / "raw").mkdir(parents=True, exist_ok=True)
    logits = np.zeros((320,2), np.float32); fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(160,4,96,2560)); roles = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(160,4,6,2560)); model=None
    try:
        model, tokenizer, device, placement = common.load_bf16("qwen3"); quant=common.quantization_audit(model); pad=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0,320,BATCH):
            batch=rows[start:start+BATCH]; _,scores,_=common.baseline_full(model,batch,pad,device); logits[start:start+len(batch)]=scores
        index=[]
        for start in range(0,160,BATCH):
            batch=hidden_rows[start:start+BATCH]; batch_fields,_,lengths=common.baseline_full(model,batch,pad,device); fields[start:start+len(batch)]=batch_fields; roles[start:start+len(batch)]=common.role_means(batch_fields.astype(np.float32),batch).astype(np.float16)
            for local,item in enumerate(batch): index.append({"hidden_index":start+local,"case_id":item["case_id"],"arm":item["arm"],"unit":item["unit"],"partition":item["partition"],"factor_a":item["factor_a"],"factor_b":item["factor_b"],"length":int(lengths[local])})
            fields.flush(); roles.flush(); print(f"[C219] hidden {start+len(batch)}/160",flush=True)
        behavior=[]
        for i,item in enumerate(rows):
            prediction=int(logits[i,1]>logits[i,0]); behavior.append({"case_id":item["case_id"],"arm":item["arm"],"unit":item["unit"],"partition":item["partition"],"order":item["order"],"gold_position":item["gold_position"],"prediction":prediction,"correct":prediction==item["gold_position"]})
        core.write_rows(OUT / "raw/behavior_index.jsonl",behavior); core.write_rows(OUT / "raw/hidden_index.jsonl",index)
        checks={"behavior":len(behavior)==320,"hidden":len(index)==160,"full_shape":list(fields.shape)==[160,4,96,2560],"role_shape":list(roles.shape)==[160,4,6,2560],"finite":bool(np.isfinite(logits).all()) and bool(np.isfinite(fields).all()) and bool(np.isfinite(roles).all()),"bf16":quant["has_bf16_parameters"],"unquantized":not quant["has_quantized_modules"]}; core.save(OUT / "analysis/run.json",{"checks":checks,"runtime":placement}); core.save(OUT / "audit/internal_run_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"checks":checks},indent=2))
    finally:
        fields.flush(); roles.flush(); del fields,roles; common.release(model); gc.collect()


def sig(states,key,arm,unit):
    h00=np.asarray(states[key[(arm,unit,0,0)],1:4],np.float32); h10=np.asarray(states[key[(arm,unit,1,0)],1:4],np.float32); h01=np.asarray(states[key[(arm,unit,0,1)],1:4],np.float32); h11=np.asarray(states[key[(arm,unit,1,1)],1:4],np.float32)
    value=np.concatenate([(h10-h00).reshape(-1),(h01-h00).reshape(-1),(h11-h10-h01+h00).reshape(-1)]).astype(np.float32); return value/max(float(np.sqrt(np.mean(np.square(value,dtype=np.float64)))),1e-12)


def analyze() -> None:
    protocol=core.load(OUT / "protocol/preregistration.json"); behavior=core.rows(OUT / "raw/behavior_index.jsonl"); new_index=core.rows(OUT / "raw/hidden_index.jsonl"); new_states=np.load(OUT / "raw/role_states.float16.npy",mmap_mode="r"); old_index=core.rows(c216.OUT / "raw/hidden_index.jsonl"); old_states=np.load(c216.OUT / "raw/role_states.float16.npy",mmap_mode="r")
    old_key={(x["arm"],x["unit"],x["factor_a"],x["factor_b"]):x["hidden_index"] for x in old_index}; new_key={(x["arm"],x["unit"],x["factor_a"],x["factor_b"]):x["hidden_index"] for x in new_index}; templates={arm:np.mean(np.stack([sig(old_states,old_key,arm,u) for u in range(4)]),axis=0) for arm in c216.ARMS}
    rows=[]
    for arm in c216.ARMS:
        for unit in range(8):
            value=sig(new_states,new_key,arm,unit); distances={candidate:float(np.sqrt(np.mean(np.square(value,template,dtype=np.float64)))) for candidate,template in []}
            distances={candidate:float(np.sqrt(np.mean(np.square(value-template,dtype=np.float64)))) for candidate,template in templates.items()}; prediction=min(c216.ARMS,key=lambda candidate:distances[candidate]); rows.append({"partition":"confirmation" if unit<4 else "fresh","arm":arm,"unit":unit,"prediction":prediction,"correct":prediction==arm,"own_distance":distances[arm],"nearest_wrong_distance":min(v for k,v in distances.items() if k!=arm)})
    behavior_summary={part:{arm:float(np.mean([x["correct"] for x in behavior if x["partition"]==part and x["arm"]==arm])) for arm in c216.ARMS} for part in ("confirmation","fresh")}; classification={}
    for part in ("confirmation","fresh"):
        selected=[x for x in rows if x["partition"]==part]; classification[part]={"support":len(selected),"accuracy":float(np.mean([x["correct"] for x in selected])),"by_arm_accuracy":{arm:float(np.mean([x["correct"] for x in selected if x["arm"]==arm])) for arm in c216.ARMS}}
    gate=all(min(behavior_summary[p].values())>=protocol["behavior_floor"] and classification[p]["accuracy"]>=protocol["classification_gate"]["each_partition_accuracy_min"] and sum(v>=.5 for v in classification[p]["by_arm_accuracy"].values())>=protocol["classification_gate"]["each_partition_arms_at_or_above_half_min"] for p in ("confirmation","fresh"))
    report={"phase":PHASE,"campaign":CAMPAIGN,"status":"shared_interface_response_state_adjudicated","behavior_by_partition_arm":behavior_summary,"frozen_template_classification":classification,"shared_interface_gate_passed":gate,"interpretation":"All arms share the same outer record/request/candidate interface and use new words. A pass supports operation-conditioned response organization beyond the old wrapper, but does not prove a context-free semantic code or causal sufficiency.","next_authorization":"C220_response_state_minimality_and_collision_negative_controls" if gate else "retain_only_arm_specific_rewording_candidates"}; core.save(OUT / "analysis/shared_interface_confirmation.json",report); core.write_rows(OUT / "analysis/classification_rows.jsonl",rows)
    checks={"behavior":len(behavior)==320,"rows":len(rows)==40,"partitions":all(classification[p]["support"]==20 for p in classification),"five_arms":all(set(classification[p]["by_arm_accuracy"])==set(c216.ARMS) for p in classification),"finite":bool(np.isfinite([x[k] for x in rows for k in ("own_distance","nearest_wrong_distance")]).all())}; core.save(OUT / "audit/internal_analysis_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"checks":checks,"report":report},indent=2))


def close() -> None:
    protocol=core.load(OUT / "protocol/preregistration.json"); report=core.load(OUT / "analysis/shared_interface_confirmation.json"); checks={"contract":core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],"run":core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"],"analysis":core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed","checks":checks,"all_checks_passed":all(checks.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT / "analysis/final.json",final); print(json.dumps(final,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("contract","run","analyze","close")); args=parser.parse_args(); {"contract":contract,"run":run,"analyze":analyze,"close":close}[args.command]()


if __name__=="__main__": main()
