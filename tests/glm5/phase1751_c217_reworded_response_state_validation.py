#!/usr/bin/env python3
"""C217: prospectively test C216 response-state templates under full rewording."""
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
import phase1750_c216_multi_family_conditional_response_state as c216
import phase1571_c098_observation_first_graph_campaign as graph_base

core = common.core
OUT = common.RESULT / "phase1751_c217_reworded_response_state_validation"
PHASE, CAMPAIGN = 1751, "C217"
BATCH = 6


def options(correct: str, wrong: str, order: int):
    return ((f"(A) {correct} (B) {wrong}", 0) if order == 1 else (f"(A) {wrong} (B) {correct}", 1))


def make(case_id: str, arm: str, unit: int, a: int, b: int, order: int, text: str, role_values: dict, correct: str, wrong: str) -> dict:
    choice, gold = options(correct, wrong, order)
    return {"case_id": case_id, "arm": arm, "unit": unit, "partition": "external_fresh", "factor_a": a, "factor_b": b, "order": order, "gold_position": gold, "prompt": f"{text} Choose the answer: {choice}. Return only A or B.", "role_values": role_values}


def material() -> list[dict]:
    rows = []
    for unit in range(8, 12):
        agent, obj, distractor, other_obj, verb, other_verb, node, middle, parent, other_node, other_parent = c212.UNITS[unit]
        for a, b, order in itertools.product((0, 1), (0, 1), (1, -1)):
            target = f"The report says {agent} {verb} the {obj}" if a == 0 else f"The report says the {obj} was {verb} by {agent}"
            noise = f"A separate note says {distractor} {other_verb} the {other_obj}"
            text = f"{target}. {noise}. Identify who dealt with the {obj}." if b == 0 else f"{noise}. {target}. Identify who dealt with the {obj}."
            rows.append(make(f"c217-agent-{unit:02d}-{a}{b}-{order:+d}", "agent_surface", unit, a, b, order, text, {"primary": agent, "secondary": distractor, "relation": verb, "context": obj, "query": obj}, agent, distractor))

            target = f"The catalog assigns {node} directly to the class {parent}" if a == 0 else f"The catalog places {node} under {middle}, and places {middle} under {parent}"
            shortcut = f" An index also links {node} straight to {parent}." if b else ""
            text = f"{target}.{shortcut} {middle} is under {other_parent}. {other_node} is under {other_parent}. Select the broader class for {node}."
            rows.append(make(f"c217-path-{unit:02d}-{a}{b}-{order:+d}", "type_path", unit, a, b, order, text, {"primary": node, "secondary": middle, "relation": "under", "context": parent, "query": node}, parent, other_parent))

            target = f"{agent} has custody of the {obj}" if a == 0 else f"the {obj} is the property of {agent}"
            noise = f"{distractor} has custody of the {other_obj}"
            text = f"Ownership memo: {target}. {noise}. Name the owner of the {obj}." if b == 0 else f"Ownership memo: {noise}. {target}. Name the owner of the {obj}."
            relation = "custody" if a == 0 else "property"
            rows.append(make(f"c217-possess-{unit:02d}-{a}{b}-{order:+d}", "possession_surface", unit, a, b, order, text, {"primary": agent, "secondary": distractor, "relation": relation, "context": obj, "query": obj}, agent, distractor))

            target = f"{agent} exceeds {distractor} in height" if a == 0 else f"{distractor} does not reach {agent} in height"
            noise = f"The {obj} remains next to the {other_obj}"
            text = f"Comparison record: {target}. {noise}. Select the taller person." if b == 0 else f"Comparison record: {noise}. {target}. Select the taller person."
            relation = "exceeds" if a == 0 else "height"
            rows.append(make(f"c217-compare-{unit:02d}-{a}{b}-{order:+d}", "comparison_surface", unit, a, b, order, text, {"primary": agent, "secondary": distractor, "relation": relation, "context": distractor, "query": "taller"}, agent, distractor))

            target = f'The glossary pairs "{node}" with "{parent}"' if a == 0 else f'The glossary renders "{node}" as "{parent}"'
            noise = f'It pairs "{other_node}" with "{other_parent}"'
            text = f"{target}. {noise}. Select the glossary value for {node}." if b == 0 else f"{noise}. {target}. Select the glossary value for {node}."
            relation = "pairs" if a == 0 else "renders"
            rows.append(make(f"c217-translate-{unit:02d}-{a}{b}-{order:+d}", "translation_surface", unit, a, b, order, text, {"primary": node, "secondary": other_node, "relation": relation, "context": parent, "query": node}, parent, other_parent))
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidates):
        raise RuntimeError(candidates)
    compiled = []
    for item in rows:
        ids = core.chat_ids(tokenizer, "Use only the supplied record. Return exactly A or B.", item["prompt"])
        positions = {}
        for role, value in item["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((item["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**item, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(c216.OUT / "audit/independent_final_audit.json")
    parent_report = core.load(c216.OUT / "analysis/final.json")["headline"]
    rows = material()
    compiled = compile_rows(graph_base.tokenizer(), rows)
    checks = {"authorization": parent["all_checks_passed"] and parent_report["response_state_gate_passed"], "cases": len(rows) == 160, "hidden_rows": sum(item["order"] == 1 for item in rows) == 80, "five_arms": {item["arm"] for item in rows} == set(c216.ARMS), "candidate_balance": sum(item["gold_position"] == 0 for item in rows) == 80, "width": max(len(item["prompt_ids"]) for item in compiled) <= common.WIDTH}
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_width": max(len(item["prompt_ids"]) for item in compiled)})
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "reworded_response_state_validation_frozen", "model": "Qwen3-4B BF16 CUDA nonquantized", "source_templates": "C216 discovery units 0-3, frozen before C217 execution", "external_units": [8, 9, 10, 11], "arms": list(c216.ARMS), "cases": 160, "hidden_rows": 80, "saved": "all tokens x embedding/q23/q24/q25 x 2560 plus roles", "behavior_floor": 0.65, "classification_gate": {"overall_accuracy_min": 0.60, "arms_at_or_above_half_min": 3}, "composition_gates": {"nrmse_max": 0.75, "weighted_sign_min": 0.75}, "claim_boundary": "rewording changes much of the interface but is still controlled English; classification can retain task-family structure and is not semantic ontology", "forbidden": ["attention", "MLP", "weights", "PCA", "template refitting", "dropping failed arms"], "producer_sha256": core.sha(Path(__file__)), "authorization": "run_all_reworded_arms_once_then_compare_to_frozen_C216_templates"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(item["prompt_ids"]) for item in compiled)}, indent=2))


@torch.inference_mode()
def run() -> None:
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    hidden_rows = [item for item in rows if item["order"] == 1]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    logits = np.zeros((len(rows), 2), np.float32)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(80, 4, 96, 2560))
    roles = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(80, 4, 6, 2560))
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
            batch_fields, _, lengths = common.baseline_full(model, batch, pad, device)
            fields[start:start + len(batch)] = batch_fields
            roles[start:start + len(batch)] = common.role_means(batch_fields.astype(np.float32), batch).astype(np.float16)
            for local, item in enumerate(batch):
                hidden_index.append({"hidden_index": start + local, "case_id": item["case_id"], "arm": item["arm"], "unit": item["unit"], "factor_a": item["factor_a"], "factor_b": item["factor_b"], "length": int(lengths[local])})
            fields.flush(); roles.flush()
            print(f"[C217] hidden {start + len(batch)}/80", flush=True)
        behavior = []
        for i, item in enumerate(rows):
            prediction = int(logits[i, 1] > logits[i, 0])
            behavior.append({"case_id": item["case_id"], "arm": item["arm"], "unit": item["unit"], "order": item["order"], "gold_position": item["gold_position"], "prediction": prediction, "correct": prediction == item["gold_position"]})
        core.write_rows(OUT / "raw/behavior_index.jsonl", behavior)
        core.write_rows(OUT / "raw/hidden_index.jsonl", hidden_index)
        checks = {"behavior": len(behavior) == 160, "hidden": len(hidden_index) == 80, "full_shape": list(fields.shape) == [80, 4, 96, 2560], "role_shape": list(roles.shape) == [80, 4, 6, 2560], "finite": bool(np.isfinite(logits).all()) and bool(np.isfinite(fields).all()) and bool(np.isfinite(roles).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        fields.flush(); roles.flush(); del fields, roles
        common.release(model); gc.collect()


def signature(states: np.ndarray, by_key: dict, arm: str, unit: int) -> tuple[np.ndarray, dict]:
    h00 = np.asarray(states[by_key[(arm, unit, 0, 0)], 1:4], np.float32)
    h10 = np.asarray(states[by_key[(arm, unit, 1, 0)], 1:4], np.float32)
    h01 = np.asarray(states[by_key[(arm, unit, 0, 1)], 1:4], np.float32)
    h11 = np.asarray(states[by_key[(arm, unit, 1, 1)], 1:4], np.float32)
    raw = np.concatenate([(h10 - h00).reshape(-1), (h01 - h00).reshape(-1), (h11 - h10 - h01 + h00).reshape(-1)]).astype(np.float32)
    raw /= max(float(np.sqrt(np.mean(np.square(raw, dtype=np.float64)))), 1e-12)
    additive = (h10 - h00) + (h01 - h00); combined = h11 - h00
    metrics = {"nrmse": common.nrmse(additive, combined), "weighted_sign": common.weighted_sign(additive, combined), "interaction_to_combined_rms": float(np.sqrt(np.square(h11 - h10 - h01 + h00, dtype=np.float64).sum() / max(np.square(combined, dtype=np.float64).sum(), 1e-30)))}
    return raw, metrics


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.rows(OUT / "raw/behavior_index.jsonl")
    new_index = core.rows(OUT / "raw/hidden_index.jsonl")
    new_states = np.load(OUT / "raw/role_states.float16.npy", mmap_mode="r")
    old_index = core.rows(c216.OUT / "raw/hidden_index.jsonl")
    old_states = np.load(c216.OUT / "raw/role_states.float16.npy", mmap_mode="r")
    old_key = {(item["arm"], item["unit"], item["factor_a"], item["factor_b"]): item["hidden_index"] for item in old_index}
    new_key = {(item["arm"], item["unit"], item["factor_a"], item["factor_b"]): item["hidden_index"] for item in new_index}
    templates = {arm: np.mean(np.stack([signature(old_states, old_key, arm, unit)[0] for unit in range(4)]), axis=0) for arm in c216.ARMS}
    rows, composition = [], {arm: [] for arm in c216.ARMS}
    for arm in c216.ARMS:
        for unit in range(8, 12):
            vector, item_metrics = signature(new_states, new_key, arm, unit)
            distances = {candidate: float(np.sqrt(np.mean(np.square(vector - template, dtype=np.float64)))) for candidate, template in templates.items()}
            prediction = min(c216.ARMS, key=lambda candidate: distances[candidate])
            rows.append({"arm": arm, "unit": unit, "prediction": prediction, "correct": prediction == arm, "own_distance": distances[arm], "nearest_wrong_distance": min(value for candidate, value in distances.items() if candidate != arm)})
            composition[arm].append(item_metrics)
    behavior_by_arm = {arm: float(np.mean([item["correct"] for item in behavior if item["arm"] == arm])) for arm in c216.ARMS}
    class_by_arm = {arm: float(np.mean([item["correct"] for item in rows if item["arm"] == arm])) for arm in c216.ARMS}
    accuracy = float(np.mean([item["correct"] for item in rows]))
    composition_summary = {arm: {"support": 4, "median_nrmse": float(np.median([item["nrmse"] for item in values])), "median_weighted_sign": float(np.median([item["weighted_sign"] for item in values])), "median_interaction_to_combined_rms": float(np.median([item["interaction_to_combined_rms"] for item in values]))} for arm, values in composition.items()}
    for arm, values in composition_summary.items():
        values["passed"] = values["median_nrmse"] <= protocol["composition_gates"]["nrmse_max"] and values["median_weighted_sign"] >= protocol["composition_gates"]["weighted_sign_min"]
    gate = min(behavior_by_arm.values()) >= protocol["behavior_floor"] and accuracy >= protocol["classification_gate"]["overall_accuracy_min"] and sum(value >= 0.5 for value in class_by_arm.values()) >= protocol["classification_gate"]["arms_at_or_above_half_min"]
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "reworded_response_state_adjudicated", "behavior": {"global_accuracy": float(np.mean([item["correct"] for item in behavior])), "by_arm": behavior_by_arm}, "frozen_template_classification": {"support": 20, "accuracy": accuracy, "by_arm_accuracy": class_by_arm}, "composition": composition_summary, "rewording_gate_passed": gate, "interpretation": "C216 discovery templates were frozen and not refit. A pass shows response-state task-family information surviving these rewordings, but does not isolate semantic identity from the remaining controlled task structure.", "next_authorization": "C218_cross_task_semantic_collision_and_same_task_surface_diversity" if gate else "retain_C216_as_template_specific_candidate_only"}
    core.save(OUT / "analysis/reworded_validation.json", report)
    core.write_rows(OUT / "analysis/classification_rows.jsonl", rows)
    checks = {"behavior": len(behavior) == 160, "classification": len(rows) == 20, "five_arms": set(composition_summary) == set(c216.ARMS), "finite": bool(np.isfinite([item[key] for item in rows for key in ("own_distance", "nearest_wrong_distance")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "report": report}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/reworded_validation.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "run", "analyze", "close")); args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
