#!/usr/bin/env python3
"""C221: third-material confirmation and exact response-field prediction test."""
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
import phase1754_c220_response_state_minimality_collision_controls as c220
import phase1571_c098_observation_first_graph_campaign as graph_base

core = common.core
OUT = common.RESULT / "phase1755_c221_independent_response_state_prediction"
PHASE, CAMPAIGN = 1755, "C221"
BATCH = 6
UNITS = [
    ("Amina", "crate", "Felix", "lantern", "packed", "polished", "otter", "mammal", "animal", "hammer", "tool"),
    ("Bram", "journal", "Liora", "vase", "edited", "washed", "crocus", "flower", "plant", "harp", "instrument"),
    ("Celia", "router", "Niko", "flag", "installed", "folded", "tram", "vehicle", "machine", "ladle", "utensil"),
    ("Darin", "locker", "Oona", "cello", "sealed", "tuned", "basalt", "rock", "material", "falcon", "bird"),
    ("Elara", "packet", "Pavel", "screen", "labeled", "raised", "iris", "flower", "plant", "pliers", "tool"),
    ("Farah", "console", "Quinn", "shield", "updated", "stored", "gecko", "reptile", "animal", "oboe", "instrument"),
    ("Galen", "catalog", "Rhea", "torch", "sorted", "ignited", "cedar", "tree", "plant", "saw", "tool"),
    ("Hana", "device", "Tomas", "mural", "verified", "painted", "mica", "mineral", "material", "robin", "bird"),
]


def choices(correct: str, wrong: str, order: int):
    return ((f"A={correct}; B={wrong}", 0) if order == 1 else (f"A={wrong}; B={correct}", 1))


def wrap(fact1: str, fact2: str, question: str, options: str) -> str:
    return f"First: {fact1}. Second: {fact2}. Ask: {question}. Choices: {options}. Output A or B."


def add(rows: list[dict], case_id: str, arm: str, unit: int, a: int, b: int, order: int, facts: tuple[str, str], question: str, roles: dict, correct: str, wrong: str) -> None:
    option_text, gold = choices(correct, wrong, order)
    fact1, fact2 = facts if b == 0 else facts[::-1]
    rows.append({
        "case_id": case_id,
        "arm": arm,
        "unit": unit,
        "partition": "confirmation" if unit < 4 else "fresh",
        "factor_a": a,
        "factor_b": b,
        "order": order,
        "gold_position": gold,
        "prompt": wrap(fact1, fact2, question, option_text),
        "role_values": roles,
    })


def material() -> list[dict]:
    rows = []
    for unit, values in enumerate(UNITS):
        agent, obj, distractor, other_obj, verb, other_verb, node, middle, parent, other_node, other_parent = values
        for a, b, order in itertools.product((0, 1), (0, 1), (1, -1)):
            target = f"{agent} {verb} the {obj}" if a == 0 else f"the {obj} was {verb} by {agent}"
            add(rows, f"c221-agent-{unit:02d}-{a}{b}-{order:+d}", "agent_surface", unit, a, b, order, (target, f"{distractor} {other_verb} the {other_obj}"), f"which person is associated with {obj}", {"primary": agent, "secondary": distractor, "relation": verb, "context": obj, "query": obj}, agent, distractor)

            target = f"every {node} is a {parent}" if a == 0 else f"every {node} is a {middle}, and every {middle} is a {parent}"
            if b:
                target += f"; every {node} is also stated to be a {parent}"
            add(rows, f"c221-path-{unit:02d}-{a}{b}-{order:+d}", "type_path", unit, a, b, order, (target, f"every {other_node} is a {other_parent}"), f"which broad category contains {node}", {"primary": node, "secondary": middle if a else other_node, "relation": "is a", "context": parent, "query": node}, parent, other_parent)

            target = f"{agent} possesses the {obj}" if a == 0 else f"the {obj} is owned by {agent}"
            relation = "possesses" if a == 0 else "owned"
            add(rows, f"c221-possess-{unit:02d}-{a}{b}-{order:+d}", "possession_surface", unit, a, b, order, (target, f"{distractor} possesses the {other_obj}"), f"who is the owner of {obj}", {"primary": agent, "secondary": distractor, "relation": relation, "context": obj, "query": obj}, agent, distractor)

            target = f"{agent} has greater height than {distractor}" if a == 0 else f"{distractor} has less height than {agent}"
            relation = "greater" if a == 0 else "less"
            add(rows, f"c221-compare-{unit:02d}-{a}{b}-{order:+d}", "comparison_surface", unit, a, b, order, (target, f"the {obj} is near the {other_obj}"), "which person has greater height", {"primary": agent, "secondary": distractor, "relation": relation, "context": distractor, "query": "height"}, agent, distractor)

            target = f'"{node}" is defined as "{parent}"' if a == 0 else f'"{node}" corresponds to "{parent}"'
            relation = "defined" if a == 0 else "corresponds"
            add(rows, f"c221-translate-{unit:02d}-{a}{b}-{order:+d}", "translation_surface", unit, a, b, order, (target, f'"{other_node}" is defined as "{other_parent}"'), f"what value is assigned to {node}", {"primary": node, "secondary": other_node, "relation": relation, "context": parent, "query": node}, parent, other_parent)
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidate_ids):
        raise RuntimeError(candidate_ids)
    compiled = []
    for item in rows:
        ids = core.chat_ids(tokenizer, "Use the facts to answer the question. Return exactly A or B.", item["prompt"])
        positions = {}
        for role, value in item["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((item["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**item, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": positions})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(c220.OUT / "audit/independent_final_audit.json")
    selected = core.load(c220.OUT / "analysis/minimality_collision_controls.json")["selected_subset"]
    rows = material()
    compiled = compile_rows(graph_base.tokenizer(), rows)
    checks = {
        "authorization": parent["all_checks_passed"],
        "parent_gate": parent["authorization"] == "C221_independent_material_response_state_prediction_then_targeted_causal_test",
        "selected_subset_frozen": selected["name"] == "q24_q25_relation_boundary",
        "cases": len(rows) == 320,
        "hidden": sum(row["order"] == 1 for row in rows) == 160,
        "arms": {row["arm"] for row in rows} == set(c216.ARMS),
        "candidate_balance": sum(row["gold_position"] == 0 for row in rows) == 160,
        "width": max(len(row["prompt_ids"]) for row in compiled) <= common.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)})
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "third_material_response_state_and_exact_field_prediction_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "arms": list(c216.ARMS),
        "cases": 320,
        "hidden_rows": 160,
        "partitions": {"confirmation": [0, 1, 2, 3], "fresh": [4, 5, 6, 7]},
        "selected_subset": selected,
        "templates": "unchanged raw and RMS-normalized C216 discovery means",
        "behavior_floor": 0.65,
        "classification_gate": {"each_partition_accuracy_min": 0.60, "arms_at_or_above_half_min": 3},
        "exact_prediction_gate": {"fresh_median_nrmse_max": 0.75, "fresh_median_weighted_sign_min": 0.75, "fresh_arms_passing_min": 3},
        "causal_authorization": "Only if behavior, classification and exact signed-field prediction all pass; classification alone does not license patching.",
        "claim_boundary": "The raw template predicts an entire selected signed response field without target scaling. Passing is still task-family conditioned and does not imply a universal semantic operator.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "target-derived scaling", "template refitting", "causal patching before exact prediction qualification"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "one_Qwen3_run_then_joint_reveal",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


@torch.inference_mode()
def run() -> None:
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    hidden_rows = [row for row in rows if row["order"] == 1]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    logits = np.zeros((len(rows), 2), np.float32)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), 4, common.WIDTH, common.DIM))
    roles = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), 4, len(common.ROLES), common.DIM))
    model = None
    try:
        model, tokenizer, device, placement = common.load_bf16("qwen3")
        quant = common.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            _, scores, _ = common.baseline_full(model, batch, pad, device)
            logits[start:start + len(batch)] = scores
        index = []
        for start in range(0, len(hidden_rows), BATCH):
            batch = hidden_rows[start:start + BATCH]
            batch_fields, _, lengths = common.baseline_full(model, batch, pad, device)
            fields[start:start + len(batch)] = batch_fields
            roles[start:start + len(batch)] = common.role_means(batch_fields.astype(np.float32), batch).astype(np.float16)
            for local, item in enumerate(batch):
                index.append({"hidden_index": start + local, "case_id": item["case_id"], "arm": item["arm"], "unit": item["unit"], "partition": item["partition"], "factor_a": item["factor_a"], "factor_b": item["factor_b"], "length": int(lengths[local])})
            fields.flush(); roles.flush()
            print(f"[C221] hidden {start + len(batch)}/{len(hidden_rows)}", flush=True)
        behavior = []
        for i, item in enumerate(rows):
            prediction = int(logits[i, 1] > logits[i, 0])
            behavior.append({"case_id": item["case_id"], "arm": item["arm"], "unit": item["unit"], "partition": item["partition"], "order": item["order"], "gold_position": item["gold_position"], "prediction": prediction, "correct": prediction == item["gold_position"]})
        core.write_rows(OUT / "raw/behavior_index.jsonl", behavior)
        core.write_rows(OUT / "raw/hidden_index.jsonl", index)
        checks = {
            "behavior": len(behavior) == 320,
            "hidden": len(index) == 160,
            "full_shape": list(fields.shape) == [160, 4, 96, 2560],
            "role_shape": list(roles.shape) == [160, 4, 6, 2560],
            "finite": bool(np.isfinite(logits).all()) and bool(np.isfinite(fields).all()) and bool(np.isfinite(roles).all()),
            "bf16": quant["has_bf16_parameters"],
            "unquantized": not quant["has_quantized_modules"],
        }
        core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        fields.flush(); roles.flush()
        del fields, roles
        common.release(model)
        gc.collect()


def summarize_classification(templates: dict, cubes: dict, units: range) -> dict:
    rows = []
    for arm in c216.ARMS:
        for unit in units:
            value = c220.normalize(cubes[(arm, unit)])
            distances = {candidate: float(np.sqrt(np.mean(np.square(value - template, dtype=np.float64)))) for candidate, template in templates.items()}
            prediction = min(c216.ARMS, key=lambda candidate: distances[candidate])
            rows.append({"arm": arm, "unit": unit, "prediction": prediction, "correct": prediction == arm})
    return {"support": len(rows), "accuracy": float(np.mean([row["correct"] for row in rows])), "by_arm_accuracy": {arm: float(np.mean([row["correct"] for row in rows if row["arm"] == arm])) for arm in c216.ARMS}}


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    subset = protocol["selected_subset"]
    old_states, old_key = c220.load_source(c216.OUT)
    new_states, new_key = c220.load_source(OUT)
    old_cubes = {(arm, unit): c220.response_cube(old_states, old_key, arm, unit, subset) for arm in c216.ARMS for unit in range(4)}
    new_cubes = {(arm, unit): c220.response_cube(new_states, new_key, arm, unit, subset) for arm in c216.ARMS for unit in range(8)}
    normalized_templates = {arm: np.mean(np.stack([c220.normalize(old_cubes[(arm, unit)]) for unit in range(4)]), axis=0) for arm in c216.ARMS}
    raw_templates = {arm: np.mean(np.stack([old_cubes[(arm, unit)] for unit in range(4)]), axis=0) for arm in c216.ARMS}
    classification = {"confirmation": summarize_classification(normalized_templates, new_cubes, range(4)), "fresh": summarize_classification(normalized_templates, new_cubes, range(4, 8))}
    behavior = core.rows(OUT / "raw/behavior_index.jsonl")
    behavior_summary = {part: {arm: float(np.mean([row["correct"] for row in behavior if row["partition"] == part and row["arm"] == arm])) for arm in c216.ARMS} for part in ("confirmation", "fresh")}
    prediction_rows = []
    for arm in c216.ARMS:
        for unit in range(8):
            actual = new_cubes[(arm, unit)]
            prediction = raw_templates[arm]
            prediction_rows.append({"arm": arm, "unit": unit, "partition": "confirmation" if unit < 4 else "fresh", "nrmse": common.nrmse(prediction, actual), "weighted_sign": common.weighted_sign(prediction, actual)})
    exact = {}
    for part in ("confirmation", "fresh"):
        exact[part] = {}
        for arm in c216.ARMS:
            selected = [row for row in prediction_rows if row["partition"] == part and row["arm"] == arm]
            exact[part][arm] = {"support": len(selected), "median_nrmse": float(np.median([row["nrmse"] for row in selected])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in selected]))}
    behavior_pass = all(min(behavior_summary[part].values()) >= protocol["behavior_floor"] for part in behavior_summary)
    classification_pass = all(classification[part]["accuracy"] >= protocol["classification_gate"]["each_partition_accuracy_min"] and sum(value >= 0.5 for value in classification[part]["by_arm_accuracy"].values()) >= protocol["classification_gate"]["arms_at_or_above_half_min"] for part in classification)
    exact_arm_pass = {arm: exact["fresh"][arm]["median_nrmse"] <= protocol["exact_prediction_gate"]["fresh_median_nrmse_max"] and exact["fresh"][arm]["median_weighted_sign"] >= protocol["exact_prediction_gate"]["fresh_median_weighted_sign_min"] for arm in c216.ARMS}
    exact_pass = sum(exact_arm_pass.values()) >= protocol["exact_prediction_gate"]["fresh_arms_passing_min"]
    causal_eligible = behavior_pass and classification_pass and exact_pass
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "third_material_response_state_and_exact_field_prediction_adjudicated",
        "behavior": {"by_partition_arm": behavior_summary, "passed": behavior_pass},
        "classification": classification,
        "classification_passed": classification_pass,
        "exact_signed_field_prediction": exact,
        "exact_arm_pass": exact_arm_pass,
        "exact_prediction_passed": exact_pass,
        "causal_eligible": causal_eligible,
        "interpretation": "Nearest-template classification and exact raw-field prediction are separate claims. Only the latter can nominate a signed field for a targeted causal edit without using target-derived scaling.",
        "next_authorization": "C222_targeted_signed_field_deletion_rescue" if causal_eligible else "C222_amplitude_conditioning_observation_without_causal_claim",
    }
    core.save(OUT / "analysis/independent_prediction.json", report)
    core.write_rows(OUT / "analysis/exact_prediction_rows.jsonl", prediction_rows)
    checks = {"behavior": len(behavior) == 320, "classification": all(classification[part]["support"] == 20 for part in classification), "prediction_rows": len(prediction_rows) == 40, "five_arms": set(exact["fresh"]) == set(c216.ARMS), "finite": bool(np.isfinite([row[key] for row in prediction_rows for key in ("nrmse", "weighted_sign")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "report": report}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/independent_prediction.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
