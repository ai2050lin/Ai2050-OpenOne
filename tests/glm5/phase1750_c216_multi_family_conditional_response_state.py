#!/usr/bin/env python3
"""C216: broad five-family observation of conditional response-state tensors."""
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
import phase1746_c212_true_factorial_composition as c212
import phase1571_c098_observation_first_graph_campaign as graph_base

core = common.core
OUT = common.RESULT / "phase1750_c216_multi_family_conditional_response_state"
PHASE, CAMPAIGN = 1750, "C216"
BATCH = 6
ARMS = ("agent_surface", "type_path", "possession_surface", "comparison_surface", "translation_surface")


def options(correct: str, wrong: str, order: int):
    return ((f"(A) {correct} (B) {wrong}", 0) if order == 1 else (f"(A) {wrong} (B) {correct}", 1))


def row(case_id: str, arm: str, unit: int, a: int, b: int, order: int, prompt: str, role_values: dict, correct: str, wrong: str) -> dict:
    choice, gold = options(correct, wrong, order)
    return {
        "case_id": case_id,
        "arm": arm,
        "unit": unit,
        "partition": c212.partition(unit),
        "factor_a": a,
        "factor_b": b,
        "order": order,
        "gold_position": gold,
        "prompt": f"{prompt} {choice}. Reply with only A or B.",
        "role_values": role_values,
    }


def material() -> list[dict]:
    rows = []
    for unit, values in enumerate(c212.UNITS):
        agent, obj, distractor, other_obj, verb, other_verb, node, middle, parent, other_node, other_parent = values
        for a, b, order in itertools.product((0, 1), (0, 1), (1, -1)):
            target = f"{agent} {verb} the {obj}" if a == 0 else f"the {obj} was {verb} by {agent}"
            noise = f"{distractor} {other_verb} the {other_obj}"
            statement = f"{target}, while {noise}." if b == 0 else f"While {noise}, {target}."
            rows.append(row(f"c216-agent-{unit:02d}-{a}{b}-{order:+d}", "agent_surface", unit, a, b, order, f"Read the statement. {statement} Who {verb} the {obj}?", {"primary": agent, "secondary": distractor, "relation": verb, "context": obj, "query": obj}, agent, distractor))

            if a == 0:
                facts = f"A {node} is a kind of {parent}. A {middle} is a kind of {other_parent}."
            else:
                facts = f"A {node} is a kind of {middle}. A {middle} is a kind of {parent}."
            if b:
                facts += f" The taxonomy also states directly that every {node} is a {parent}."
            facts += f" A {other_node} is a kind of {other_parent}."
            rows.append(row(f"c216-path-{unit:02d}-{a}{b}-{order:+d}", "type_path", unit, a, b, order, f"Read the taxonomy. {facts} Which category contains the {node}?", {"primary": node, "secondary": middle, "relation": "kind of", "context": parent, "query": node}, parent, other_parent))

            target = f"{agent} owns the {obj}" if a == 0 else f"the {obj} belongs to {agent}"
            noise = f"{distractor} owns the {other_obj}"
            statement = f"{target}, while {noise}." if b == 0 else f"While {noise}, {target}."
            relation = "owns" if a == 0 else "belongs"
            rows.append(row(f"c216-possess-{unit:02d}-{a}{b}-{order:+d}", "possession_surface", unit, a, b, order, f"Read the ownership record. {statement} Who owns the {obj}?", {"primary": agent, "secondary": distractor, "relation": relation, "context": obj, "query": obj}, agent, distractor))

            target = f"{agent} is taller than {distractor}" if a == 0 else f"{distractor} is shorter than {agent}"
            noise = f"The {obj} is beside the {other_obj}"
            statement = f"{target}. {noise}." if b == 0 else f"{noise}. {target}."
            relation = "taller" if a == 0 else "shorter"
            rows.append(row(f"c216-compare-{unit:02d}-{a}{b}-{order:+d}", "comparison_surface", unit, a, b, order, f"Read the comparison. {statement} Who is taller?", {"primary": agent, "secondary": distractor, "relation": relation, "context": distractor, "query": "taller"}, agent, distractor))

            target = f'In the code language, "{node}" means "{parent}"' if a == 0 else f'The codebook translates "{node}" as "{parent}"'
            noise = f'"{other_node}" means "{other_parent}"'
            statement = f"{target}; {noise}." if b == 0 else f"{noise}; {target}."
            relation = "means" if a == 0 else "translates"
            rows.append(row(f"c216-translate-{unit:02d}-{a}{b}-{order:+d}", "translation_surface", unit, a, b, order, f'Read the codebook. {statement} What does "{node}" denote?', {"primary": node, "secondary": other_node, "relation": relation, "context": parent, "query": node}, parent, other_parent))
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidate_ids):
        raise RuntimeError(candidate_ids)
    compiled = []
    for item in rows:
        ids = core.chat_ids(tokenizer, "Answer from the supplied statement. Reply exactly A or B.", item["prompt"])
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
    parent = core.load(common.C215 / "audit/independent_final_audit.json")
    rows = material()
    compiled = compile_rows(graph_base.tokenizer(), rows)
    checks = {
        "authorization": parent["all_checks_passed"],
        "cases": len(rows) == 480,
        "arms": {item["arm"] for item in rows} == set(ARMS),
        "arm_balance": all(sum(item["arm"] == arm for item in rows) == 96 for arm in ARMS),
        "partition_balance": {part: sum(item["partition"] == part for item in rows) for part in ("discovery", "confirmation", "fresh")} == {"discovery": 160, "confirmation": 160, "fresh": 160},
        "candidate_balance": sum(item["gold_position"] == 0 for item in rows) == 240,
        "width": max(len(item["prompt_ids"]) for item in compiled) <= common.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_width": max(len(item["prompt_ids"]) for item in compiled)})
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "five_family_conditional_response_state_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "arms": list(ARMS),
        "cases": 480,
        "hidden_rows": 240,
        "saved": "all tokens x all 2560 coordinates at embedding/q23/q24/q25 plus semantic role means",
        "partitions": {"discovery": [0, 1, 2, 3], "confirmation": [4, 5, 6, 7], "fresh": [8, 9, 10, 11]},
        "behavior_floor": 0.65,
        "composition_gates": {"fresh_nrmse_max": 0.75, "fresh_weighted_sign_min": 0.75},
        "response_state_gate": {"fresh_five_way_accuracy_min": 0.60, "fresh_each_arm_support": 4},
        "response_signature": "RMS-normalized concatenation of factor-A, factor-B and interaction responses over q23/q24/q25 x six roles x 2560 coordinates",
        "claim_boundary": "arm identity may include prompt-template and task-interface structure; it is not an abstract semantic code or causal circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-reveal arm removal", "project-level stop after one arm failure"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_all_five_arms_on_Qwen3_then_reveal_together",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(item["prompt_ids"]) for item in compiled)}, indent=2))


@torch.inference_mode()
def run() -> None:
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    hidden_rows = [item for item in rows if item["order"] == 1]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    logits = np.zeros((len(rows), 2), np.float32)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), 4, common.WIDTH, common.DIM))
    role_states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), 4, len(common.ROLES), common.DIM))
    lengths = np.zeros(len(hidden_rows), np.int32)
    model = None
    try:
        model, tokenizer, device, placement = common.load_bf16("qwen3")
        quant = common.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            _, scores, _ = common.baseline_full(model, batch, pad, device)
            logits[start:start + len(batch)] = scores
        hidden_index = []
        for start in range(0, len(hidden_rows), BATCH):
            batch = hidden_rows[start:start + BATCH]
            batch_fields, _, batch_lengths = common.baseline_full(model, batch, pad, device)
            fields[start:start + len(batch)] = batch_fields
            role_states[start:start + len(batch)] = common.role_means(batch_fields.astype(np.float32), batch).astype(np.float16)
            lengths[start:start + len(batch)] = batch_lengths
            for local, item in enumerate(batch):
                hidden_index.append({"hidden_index": start + local, "case_id": item["case_id"], "arm": item["arm"], "unit": item["unit"], "partition": item["partition"], "factor_a": item["factor_a"], "factor_b": item["factor_b"], "length": int(batch_lengths[local])})
            fields.flush(); role_states.flush()
            print(f"[C216] hidden {start + len(batch)}/{len(hidden_rows)}", flush=True)
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        behavior = []
        for i, item in enumerate(rows):
            prediction = int(logits[i, 1] > logits[i, 0])
            behavior.append({"case_id": item["case_id"], "arm": item["arm"], "unit": item["unit"], "partition": item["partition"], "factor_a": item["factor_a"], "factor_b": item["factor_b"], "order": item["order"], "gold_position": item["gold_position"], "prediction": prediction, "correct": prediction == item["gold_position"]})
        core.write_rows(OUT / "raw/behavior_index.jsonl", behavior)
        core.write_rows(OUT / "raw/hidden_index.jsonl", hidden_index)
        checks = {"behavior_rows": len(behavior) == 480, "hidden_rows": len(hidden_index) == 240, "field_shape": list(fields.shape) == [240, 4, 96, 2560], "role_shape": list(role_states.shape) == [240, 4, 6, 2560], "finite": bool(np.isfinite(logits).all()) and bool(np.isfinite(fields).all()) and bool(np.isfinite(role_states).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        fields.flush(); role_states.flush()
        del fields, role_states
        common.release(model)
        gc.collect()


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    return {"nrmse": common.nrmse(prediction, target), "weighted_sign": common.weighted_sign(prediction, target)}


def normalized_signature(h00: np.ndarray, h10: np.ndarray, h01: np.ndarray, h11: np.ndarray) -> np.ndarray:
    signature = np.concatenate([(h10 - h00).reshape(-1), (h01 - h00).reshape(-1), (h11 - h10 - h01 + h00).reshape(-1)]).astype(np.float32)
    return signature / max(float(np.sqrt(np.mean(np.square(signature, dtype=np.float64)))), 1e-12)


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.rows(OUT / "raw/behavior_index.jsonl")
    index_rows = core.rows(OUT / "raw/hidden_index.jsonl")
    states = np.load(OUT / "raw/role_states.float16.npy", mmap_mode="r")
    by_key = {(item["arm"], item["unit"], item["factor_a"], item["factor_b"]): item["hidden_index"] for item in index_rows}
    behavior_by_arm = {arm: {part: float(np.mean([item["correct"] for item in behavior if item["arm"] == arm and item["partition"] == part])) for part in ("discovery", "confirmation", "fresh")} for arm in ARMS}
    global_accuracy = float(np.mean([item["correct"] for item in behavior]))

    unit_rows, signatures = [], {}
    for arm in ARMS:
        for unit in range(12):
            h00 = np.asarray(states[by_key[(arm, unit, 0, 0)], 1:4], np.float32)
            h10 = np.asarray(states[by_key[(arm, unit, 1, 0)], 1:4], np.float32)
            h01 = np.asarray(states[by_key[(arm, unit, 0, 1)], 1:4], np.float32)
            h11 = np.asarray(states[by_key[(arm, unit, 1, 1)], 1:4], np.float32)
            additive = (h10 - h00) + (h01 - h00)
            combined = h11 - h00
            item = {"arm": arm, "unit": unit, "partition": c212.partition(unit), **metrics(additive, combined)}
            item["interaction_to_combined_rms"] = float(np.sqrt(np.square(h11 - h10 - h01 + h00, dtype=np.float64).sum() / max(np.square(combined, dtype=np.float64).sum(), 1e-30)))
            unit_rows.append(item)
            signatures[(arm, unit)] = normalized_signature(h00, h10, h01, h11)

    composition = {}
    for arm in ARMS:
        composition[arm] = {}
        for part in ("discovery", "confirmation", "fresh"):
            selected = [item for item in unit_rows if item["arm"] == arm and item["partition"] == part]
            composition[arm][part] = {"support": len(selected), "median_nrmse": float(np.median([item["nrmse"] for item in selected])), "median_weighted_sign": float(np.median([item["weighted_sign"] for item in selected])), "median_interaction_to_combined_rms": float(np.median([item["interaction_to_combined_rms"] for item in selected]))}
        fresh = composition[arm]["fresh"]
        fresh["passed"] = fresh["median_nrmse"] <= protocol["composition_gates"]["fresh_nrmse_max"] and fresh["median_weighted_sign"] >= protocol["composition_gates"]["fresh_weighted_sign_min"]

    templates = {arm: np.mean(np.stack([signatures[(arm, unit)] for unit in range(4)]), axis=0) for arm in ARMS}
    classification = {}
    rows = []
    for part, units in (("confirmation", range(4, 8)), ("fresh", range(8, 12))):
        correct = 0
        by_arm_correct = {arm: 0 for arm in ARMS}
        for arm in ARMS:
            for unit in units:
                signature = signatures[(arm, unit)]
                distances = {candidate: float(np.sqrt(np.mean(np.square(signature - template, dtype=np.float64)))) for candidate, template in templates.items()}
                prediction = min(ARMS, key=lambda candidate: distances[candidate])
                hit = prediction == arm
                correct += int(hit); by_arm_correct[arm] += int(hit)
                rows.append({"partition": part, "arm": arm, "unit": unit, "prediction": prediction, "correct": hit, "own_distance": distances[arm], "nearest_wrong_distance": min(value for candidate, value in distances.items() if candidate != arm)})
        classification[part] = {"support": len(ARMS) * 4, "accuracy": correct / (len(ARMS) * 4), "by_arm_accuracy": {arm: by_arm_correct[arm] / 4 for arm in ARMS}}
    state_gate = classification["fresh"]["accuracy"] >= protocol["response_state_gate"]["fresh_five_way_accuracy_min"]
    behavior_eligible_arms = [arm for arm in ARMS if min(behavior_by_arm[arm].values()) >= protocol["behavior_floor"]]
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "five_family_conditional_response_state_observed",
        "behavior": {"global_accuracy": global_accuracy, "by_arm_partition": behavior_by_arm, "eligible_arms": behavior_eligible_arms},
        "composition": composition,
        "response_state_classification": classification,
        "response_state_gate_passed": state_gate,
        "composition_passing_arms": [arm for arm in ARMS if composition[arm]["fresh"]["passed"]],
        "interpretation": "A response signature that identifies an arm may encode prompt and task interface as well as relation structure. Per-arm composition remains separately adjudicated, and no arm failure stops observation of the others.",
        "next_authorization": "C217_holdout_surface_and_language_rewording_of_any_C216_response_state_candidate" if state_gate else "C217_redefine_response_state_with_event_order_and_full_token_locality_without_fixed_coordinate_formula",
    }
    core.save(OUT / "analysis/conditional_response_state.json", report)
    core.write_rows(OUT / "analysis/classification_rows.jsonl", rows)
    checks = {"behavior_accounting": len(behavior) == 480, "unit_accounting": len(unit_rows) == 60, "five_arms": set(composition) == set(ARMS), "classification_accounting": len(rows) == 40, "finite": bool(np.isfinite([item[key] for item in unit_rows for key in ("nrmse", "weighted_sign", "interaction_to_combined_rms")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "report": report}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/conditional_response_state.json")
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
