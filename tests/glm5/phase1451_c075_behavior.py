#!/usr/bin/env python3
"""Phase1451: behavior qualification for the C075 six-relation atlas."""
from __future__ import annotations

import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1451, "C075"
CONTRACT = TESTS / "result/phase1450_c075_full_field_atlas_contract"
OUT = TESTS / "result/phase1451_c075_behavior"
BATCH = 24


def accuracy(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows)


def balanced_accuracy(rows: list[dict]) -> float:
    positive = [row for row in rows if row["truth"]]
    negative = [row for row in rows if not row["truth"]]
    return (accuracy(positive) + accuracy(negative)) / 2.0


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1451 exists")
    contract_final = core.load(CONTRACT / "analysis/final.json")
    contract_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if contract_final["authorization"] != "run_phase1451_c075_behavior" or not contract_audit["all_checks_passed"]:
        raise RuntimeError("Phase1450 did not authorize behavior")
    source = core.rows(CONTRACT / "material/active_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    composition = core.rows(CONTRACT / "material/composition_sets.jsonl")
    set_keys = tuple(f"{surface}_{cell}" for surface in protocol["surfaces"] for cell in protocol["cells"])
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        predictions, first = [], None
        for start in range(0, len(compiled), BATCH):
            values = runner.forward(model, compiled[start:start + BATCH], pad, device, supports)
            if first is None:
                first = values
            predictions.extend(values)
        repeat = runner.forward(model, compiled[:BATCH], pad, device, supports)
        numeric_repeat = max(abs(a["scores"][index] - b["scores"][index]) for a, b in zip(first, repeat) for index in range(2))
        behavior = [{**row, **pred, "correct": pred["prediction"] == row["gold_position"]} for row, pred in zip(source, predictions)]
        by_case = {row["case_id"]: row for row in behavior}
        eligible = [row for row in composition if all(by_case[row[key]]["correct"] for key in set_keys)]
        gate = protocol["behavior"]
        family_relation_surface = {}
        for family in core.load(CONTRACT / "material/frozen_concept_graph.json")["families"]:
            family_relation_surface[family] = {}
            for relation in protocol["relations"]:
                family_relation_surface[family][relation] = {}
                for surface in protocol["surfaces"]:
                    rows = [row for row in behavior if row["record_object"] == family and row["record_relation"] == relation and row["surface"] == surface]
                    metrics = {
                        "count": len(rows), "accuracy": accuracy(rows), "balanced_accuracy": balanced_accuracy(rows),
                        "partition": {name: accuracy([row for row in rows if row["partition"] == name]) for name in protocol["partitions"]},
                        "truth": {str(value).lower(): accuracy([row for row in rows if row["truth"] == value]) for value in (True, False)},
                        "cell": {cell: accuracy([row for row in rows if row["cell"] == cell]) for cell in protocol["cells"]},
                    }
                    checks = {
                        "accuracy": metrics["accuracy"] >= gate["family_relation_surface_accuracy_min"],
                        "balanced_accuracy": metrics["balanced_accuracy"] >= gate["family_relation_surface_balanced_accuracy_min"],
                        "partition": min(metrics["partition"].values()) >= gate["partition_min"],
                        "truth": min(metrics["truth"].values()) >= gate["truth_min"],
                        "cell": min(metrics["cell"].values()) >= gate["cell_min"],
                    }
                    family_relation_surface[family][relation][surface] = {"metrics": metrics, "checks": checks, "qualified": all(checks.values())}
        relation_results, selected = {}, []
        for relation in protocol["relations"]:
            qualified_families = [family for family in family_relation_surface if all(family_relation_surface[family][relation][surface]["qualified"] for surface in protocol["surfaces"])]
            relation_eligible = [row for row in eligible if row["record_relation"] == relation]
            partition_counts = {name: sum(row["partition"] == name for row in relation_eligible) for name in protocol["partitions"]}
            checks = {
                "family_breadth": len(qualified_families) >= protocol["material"]["minimum_families"],
                "set_all": len(relation_eligible) / 36 >= gate["relation_set_all_min"],
                "fixed_capture_shape": all(count == 12 for count in partition_counts.values()),
            }
            qualified = all(checks.values())
            relation_results[relation] = {"qualified_families": qualified_families, "eligible_count": len(relation_eligible), "eligible_partition_counts": partition_counts, "checks": checks, "qualified": qualified}
            if qualified:
                selected.extend(sorted(relation_eligible, key=lambda row: row["set_id"]))
        qualified_relations = [relation for relation, result in relation_results.items() if result["qualified"]]
        surface_global = {surface: {"accuracy": accuracy([row for row in behavior if row["surface"] == surface]), "balanced_accuracy": balanced_accuracy([row for row in behavior if row["surface"] == surface])} for surface in protocol["surfaces"]}
        breadth = {
            "all_relations": len(qualified_relations) == len(protocol["relations"]),
            "all_composition_sets": len(selected) == len(composition),
            "all_surfaces": all(value["balanced_accuracy"] >= protocol["zero_model_gate"]["required_model_balanced_accuracy_min"] for value in surface_global.values()),
            "beats_incomplete_zero_model": min(value["balanced_accuracy"] for value in surface_global.values()) > protocol["zero_model_gate"]["maximum_incomplete_balanced_accuracy"],
            "numeric_repeat": numeric_repeat <= gate["same_batch_repeat_max_abs_diff"],
            "finite": all(math.isfinite(value) for row in behavior for value in row["scores"]),
            "bf16": quant["has_bf16_parameters"], "not_quantized": not quant["has_quantized_modules"],
        }
        behavior_qualified = all(breadth.values())
        core.write_rows(OUT / "raw/active_behavior.jsonl", behavior)
        core.write_rows(OUT / "material/eligible_composition_sets.jsonl", selected)
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "global_accuracy": accuracy(behavior), "global_balanced_accuracy": balanced_accuracy(behavior),
            "surface_global": surface_global, "family_relation_surface": family_relation_surface,
            "relation_results": relation_results, "qualified_relations": qualified_relations,
            "selected_count": len(selected), "selected_partition_counts": {name: sum(row["partition"] == name for row in selected) for name in protocol["partitions"]},
            "breadth_checks": breadth, "behavior_qualified": behavior_qualified, "numeric_repeat_max_abs_diff": numeric_repeat,
            "hidden_state_accessed": False, "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/behavior_summary.json", summary)
        authorization = "run_phase1452_c075_discovery_full_field_capture" if behavior_qualified else "close_c075_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "behavior_qualified": behavior_qualified, "qualified_relations": qualified_relations, "authorization": authorization})
        print(json.dumps({key: value for key, value in summary.items() if key not in ("family_relation_surface", "relation_results")}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
