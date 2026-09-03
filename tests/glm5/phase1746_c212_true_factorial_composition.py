#!/usr/bin/env python3
"""C212: new-material factorial tests of surface and path combination."""
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
import phase1571_c098_observation_first_graph_campaign as graph_base

core = common.core
OUT = common.C212
PHASE, CAMPAIGN = 1746, "C212"
BATCH = 8

UNITS = [
    ("Nolan", "lantern", "Tessa", "map", "carried", "folded", "sparrow", "bird", "animal", "mallet", "tool"),
    ("Elena", "journal", "Damon", "scarf", "opened", "packed", "salmon", "fish", "animal", "kettle", "appliance"),
    ("Caleb", "compass", "Iris", "flag", "moved", "raised", "cactus", "plant", "organism", "trumpet", "instrument"),
    ("Priya", "token", "Owen", "crown", "polished", "wore", "bicycle", "vehicle", "machine", "fork", "utensil"),
    ("Marco", "box", "Nina", "piano", "packed", "tuned", "ruby", "mineral", "material", "falcon", "bird"),
    ("Amara", "lamp", "Felix", "case", "lit", "sealed", "soprano", "singer", "artist", "cedar", "tree"),
    ("Theo", "notebook", "Uma", "mirror", "copied", "cleaned", "tablet", "device", "machine", "rose", "plant"),
    ("Lina", "gauge", "Galen", "diagram", "tested", "drew", "cello", "instrument", "artifact", "otter", "animal"),
    ("Rhea", "parcel", "Basil", "window", "mailed", "washed", "orchid", "flower", "plant", "hammer", "tool"),
    ("Kian", "vase", "Marta", "basket", "painted", "wove", "trout", "fish", "animal", "violin", "instrument"),
    ("Asha", "engine", "Pavel", "fence", "inspected", "built", "robin", "bird", "animal", "ladle", "utensil"),
    ("Dara", "bridge", "Ivo", "letter", "designed", "translated", "opal", "mineral", "material", "willow", "tree"),
]


def partition(unit: int) -> str:
    return "discovery" if unit < 4 else ("confirmation" if unit < 8 else "fresh")


def options(correct: str, wrong: str, order: int):
    return ((f"(A) {correct} (B) {wrong}", 0) if order == 1 else (f"(A) {wrong} (B) {correct}", 1))


def material() -> list[dict]:
    rows = []
    for unit, values in enumerate(UNITS):
        agent, obj, distractor, other_obj, verb, other_verb, node, middle, parent, other_node, other_parent = values
        for voice, clause_order, order in itertools.product((0, 1), (0, 1), (1, -1)):
            target_clause = f"{agent} {verb} the {obj}" if voice == 0 else f"the {obj} was {verb} by {agent}"
            distractor_clause = f"{distractor} {other_verb} the {other_obj}"
            statement = f"{target_clause}, while {distractor_clause}." if clause_order == 0 else f"While {distractor_clause}, {target_clause}."
            choice, gold = options(agent, distractor, order)
            rows.append({"case_id": f"c212-surface-{unit:02d}-{voice}{clause_order}-{order:+d}", "arm": "surface_factorial", "unit": unit, "partition": partition(unit), "factor_a": voice, "factor_b": clause_order, "order": order, "gold_position": gold, "prompt": f"Read the statement. {statement} Who {verb} the {obj}? {choice}. Reply with only A or B.", "role_values": {"primary": agent, "secondary": distractor, "relation": verb, "context": obj, "query": obj}})
        for indirect_path, shortcut, order in itertools.product((0, 1), (0, 1), (1, -1)):
            if indirect_path:
                facts = f"A {node} is a kind of {middle}. A {middle} is a kind of {parent}. A {other_node} is a kind of {other_parent}."
            else:
                facts = f"A {node} is a kind of {parent}. A {middle} is a kind of {other_parent}. A {other_node} is also a kind of {other_parent}."
            if shortcut:
                facts += f" The hierarchy directly confirms that every {node} is a {parent}."
            choice, gold = options(parent, other_parent, order)
            rows.append({"case_id": f"c212-path-{unit:02d}-{indirect_path}{shortcut}-{order:+d}", "arm": "path_factorial", "unit": unit, "partition": partition(unit), "factor_a": indirect_path, "factor_b": shortcut, "order": order, "gold_position": gold, "prompt": f"Read the taxonomy. {facts} Which category contains the {node}? {choice}. Reply with only A or B.", "role_values": {"primary": node, "secondary": middle, "relation": "kind of", "context": parent, "query": node}})
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidate_ids):
        raise RuntimeError(candidate_ids)
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, "Answer from the supplied statement. Reply exactly A or B.", row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": positions})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C211 / "audit/independent_final_audit.json")
    rows = material()
    compiled = compile_rows(graph_base.tokenizer(), rows)
    checks = {
        "authorization": parent["all_checks_passed"],
        "cases": len(rows) == 192,
        "arms": {row["arm"] for row in rows} == {"surface_factorial", "path_factorial"},
        "partitions": {part: sum(row["partition"] == part for row in rows) for part in ("discovery", "confirmation", "fresh")} == {"discovery": 64, "confirmation": 64, "fresh": 64},
        "candidate_balance": float(np.mean([row["gold_position"] == 0 for row in rows])) == 0.5,
        "factor_balance": all(sum(row["factor_a"] == value and row["factor_b"] == other for row in rows if row["arm"] == arm) == 24 for arm in ("surface_factorial", "path_factorial") for value in (0, 1) for other in (0, 1)),
        "width": max(len(row["prompt_ids"]) for row in compiled) <= common.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "new_factorial_composition_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "arms": {
            "surface_factorial": "active/passive voice x target/distractor clause order, answer held fixed",
            "path_factorial": "direct versus two-hop taxonomy path x explicit direct shortcut, answer held fixed",
        },
        "cases": 192,
        "hidden_rows": "order=+1 only: 96 rows",
        "behavior_gates": {"global_min": 0.80, "arm_partition_min": 0.65},
        "factorial_prediction": "combined delta predicted without fitting as atomic-A delta plus atomic-B delta from the same lexical unit",
        "hidden_gates": {"fresh_nrmse_max": 0.75, "fresh_weighted_sign_min": 0.75, "fresh_interaction_to_combined_max": 0.75, "both_arms_required": True},
        "claim_boundary": "controlled factorial surface and graph-path programs; a pass is not a general language composition algebra",
        "forbidden": ["attention", "MLP", "weights", "PCA", "fitting combined cells", "post-reveal threshold changes"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_Qwen_behavior_then_hidden_factorial_reveal",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_tokens": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


@torch.inference_mode()
def run() -> None:
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    hidden_rows = [row for row in rows if row["order"] == 1]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    behavior_logits = np.zeros((len(rows), 2), np.float32)
    states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), 4, len(common.ROLES), common.DIM))
    model = None
    try:
        model, tokenizer, device, placement = common.load_bf16("qwen3")
        quant = common.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            _, logits, _ = common.baseline_full(model, batch, pad, device)
            behavior_logits[start:start + len(batch)] = logits
        hidden_index = []
        for start in range(0, len(hidden_rows), BATCH):
            batch = hidden_rows[start:start + BATCH]
            fields, _, _ = common.baseline_full(model, batch, pad, device)
            states[start:start + len(batch)] = common.role_means(fields.astype(np.float32), batch).astype(np.float16)
            for local, row in enumerate(batch):
                hidden_index.append({"hidden_index": start + local, "case_id": row["case_id"], "arm": row["arm"], "unit": row["unit"], "partition": row["partition"], "factor_a": row["factor_a"], "factor_b": row["factor_b"]})
            states.flush()
            print(f"[C212] hidden {start + len(batch)}/96", flush=True)
        np.save(OUT / "raw/behavior_logits.float32.npy", behavior_logits)
        behavior_index = []
        for i, row in enumerate(rows):
            prediction = int(behavior_logits[i, 1] > behavior_logits[i, 0])
            behavior_index.append({"case_id": row["case_id"], "arm": row["arm"], "unit": row["unit"], "partition": row["partition"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "margin": float(behavior_logits[i, row["gold_position"]] - behavior_logits[i, 1 - row["gold_position"]])})
        core.write_rows(OUT / "raw/behavior_index.jsonl", behavior_index)
        core.write_rows(OUT / "raw/hidden_index.jsonl", hidden_index)
        checks = {"behavior": len(behavior_index) == 192, "hidden": len(hidden_index) == 96, "shape": list(states.shape) == [96, 4, 6, common.DIM], "finite": bool(np.isfinite(behavior_logits).all()) and bool(np.isfinite(states).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        states.flush()
        del states
        common.release(model)
        gc.collect()


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.rows(OUT / "raw/behavior_index.jsonl")
    states = np.load(OUT / "raw/role_states.float16.npy", mmap_mode="r")
    hidden_index = core.rows(OUT / "raw/hidden_index.jsonl")
    global_accuracy = float(np.mean([row["correct"] for row in behavior]))
    by_arm_partition = {arm: {part: float(np.mean([row["correct"] for row in behavior if row["arm"] == arm and row["partition"] == part])) for part in ("discovery", "confirmation", "fresh")} for arm in ("surface_factorial", "path_factorial")}
    behavior_passed = global_accuracy >= protocol["behavior_gates"]["global_min"] and min(value for arm in by_arm_partition.values() for value in arm.values()) >= protocol["behavior_gates"]["arm_partition_min"]
    by_key = {(row["arm"], row["unit"], row["factor_a"], row["factor_b"]): row["hidden_index"] for row in hidden_index}
    arm_rows = {}
    for arm in ("surface_factorial", "path_factorial"):
        unit_rows = []
        for unit in range(12):
            h00 = np.asarray(states[by_key[(arm, unit, 0, 0)], 1:4], np.float32)
            h10 = np.asarray(states[by_key[(arm, unit, 1, 0)], 1:4], np.float32)
            h01 = np.asarray(states[by_key[(arm, unit, 0, 1)], 1:4], np.float32)
            h11 = np.asarray(states[by_key[(arm, unit, 1, 1)], 1:4], np.float32)
            atomic_prediction = (h10 - h00) + (h01 - h00)
            combined = h11 - h00
            interaction = h11 - h10 - h01 + h00
            unit_rows.append({"unit": unit, "partition": partition(unit), "nrmse": common.nrmse(atomic_prediction, combined), "weighted_sign": common.weighted_sign(atomic_prediction, combined), "interaction_to_combined_rms": float(np.sqrt(np.square(interaction, dtype=np.float64).sum() / max(np.square(combined, dtype=np.float64).sum(), 1e-30)))})
        arm_rows[arm] = unit_rows
    summaries = {}
    gates = protocol["hidden_gates"]
    for arm, rows in arm_rows.items():
        summaries[arm] = {}
        for split in ("discovery", "confirmation", "fresh"):
            selected = [row for row in rows if row["partition"] == split]
            summaries[arm][split] = {"support": len(selected), "median_nrmse": float(np.median([row["nrmse"] for row in selected])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in selected])), "median_interaction_to_combined_rms": float(np.median([row["interaction_to_combined_rms"] for row in selected]))}
        fresh = summaries[arm]["fresh"]
        fresh["passed"] = fresh["median_nrmse"] <= gates["fresh_nrmse_max"] and fresh["median_weighted_sign"] >= gates["fresh_weighted_sign_min"] and fresh["median_interaction_to_combined_rms"] <= gates["fresh_interaction_to_combined_max"]
    passed = behavior_passed and all(value["fresh"]["passed"] for value in summaries.values())
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "factorial_composition_adjudicated", "behavior": {"global_accuracy": global_accuracy, "by_arm_partition": by_arm_partition, "passed": behavior_passed}, "arm_summaries": summaries, "unit_rows": arm_rows, "factorial_composition_gate_passed": passed, "interpretation": "The combined cell is held out from fitting, but these are two specific factorial constructions. Passing would establish local additive predictability for them, not a universal language algebra.", "next_authorization": "C213_qualified_deletion_rescue_if_C208_or_C212_passes_else_typed_not_tested"}
    core.save(OUT / "analysis/factorial_composition.json", report)
    checks = {"behavior_accounting": len(behavior) == 192, "two_arms": set(summaries) == {"surface_factorial", "path_factorial"}, "units": all(len(rows) == 12 for rows in arm_rows.values()), "finite": bool(np.isfinite([row[key] for rows in arm_rows.values() for row in rows for key in ("nrmse", "weighted_sign", "interaction_to_combined_rms")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"behavior": report["behavior"], "summaries": summaries, "passed": passed, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/factorial_composition.json")
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

