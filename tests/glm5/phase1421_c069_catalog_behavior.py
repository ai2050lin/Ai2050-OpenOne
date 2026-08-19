#!/usr/bin/env python3
"""Phase1421: catalog-scoped behavior qualification for C069."""
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

PHASE, CAMPAIGN = 1421, "C069"
CONTRACT = TESTS / "result/phase1420_c069_catalog_four_role_contract"
OUT = TESTS / "result/phase1421_c069_catalog_behavior"
BATCH = 24
SET_KEYS = (
    "true_recipient", "false_recipient", "true_surface", "false_surface",
    "true_member", "false_member", "g_true", "h_true", "g_false_h",
    "h_false_g", "cross_gh", "cross_hg",
)


def accuracy(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows)


def balanced_accuracy(rows: list[dict]) -> float:
    positive = [row for row in rows if row["truth"]]
    negative = [row for row in rows if not row["truth"]]
    return (accuracy(positive) + accuracy(negative)) / 2.0


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1421 exists")
    contract_final = core.load(CONTRACT / "analysis/final.json")
    contract_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if contract_final["authorization"] != "run_phase1421_c069_catalog_behavior" or not contract_audit["all_checks_passed"]:
        raise RuntimeError("contract missing")

    source = core.rows(CONTRACT / "material/active_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    composition = core.rows(CONTRACT / "material/composition_sets.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        predictions = []
        first = None
        for start in range(0, len(compiled), BATCH):
            values = runner.forward(model, compiled[start:start + BATCH], pad, device, supports)
            if first is None:
                first = values
            predictions.extend(values)
        repeat = runner.forward(model, compiled[:BATCH], pad, device, supports)
        numeric_repeat = max(
            abs(a["scores"][index] - b["scores"][index])
            for a, b in zip(first, repeat) for index in range(2)
        )
        behavior = [
            {**row, **pred, "correct": pred["prediction"] == row["gold_position"]}
            for row, pred in zip(source, predictions)
        ]
        by_case = {row["case_id"]: row for row in behavior}
        eligible = [row for row in composition if all(by_case[row[key]]["correct"] for key in SET_KEYS)]
        gate = protocol["behavior"]
        family_results = {}
        selected = []
        for family in protocol["material"]["families"]:
            all_family = [row for row in behavior if row["record_family"] == family]
            catalog = [row for row in all_family if row["surface"] == "catalog"]
            ordinary = [row for row in all_family if row["surface"] == "ordinary"]
            family_eligible = [row for row in eligible if row["family"] == family]
            partition_counts = {
                name: sum(row["partition"] == name for row in family_eligible)
                for name in protocol["material"]["partitions"]
            }
            metrics = {
                "catalog_count": len(catalog),
                "catalog_accuracy": accuracy(catalog),
                "catalog_balanced_accuracy": balanced_accuracy(catalog),
                "catalog_partition": {
                    name: accuracy([row for row in catalog if row["partition"] == name])
                    for name in protocol["material"]["partitions"]
                },
                "catalog_truth": {
                    str(value).lower(): accuracy([row for row in catalog if row["truth"] == value])
                    for value in (True, False)
                },
                "catalog_cell": {
                    cell: accuracy([row for row in catalog if row["cell"] == cell])
                    for cell in sorted({row["cell"] for row in catalog})
                },
                "ordinary_accuracy_descriptive": accuracy(ordinary),
                "eligible_count": len(family_eligible),
                "eligible_partition_counts": partition_counts,
            }
            checks = {
                "catalog_accuracy": metrics["catalog_accuracy"] >= gate["catalog_accuracy_min"],
                "catalog_balanced_accuracy": metrics["catalog_balanced_accuracy"] >= gate["catalog_balanced_accuracy_min"],
                "catalog_partition": min(metrics["catalog_partition"].values()) >= gate["catalog_partition_min"],
                "catalog_truth": min(metrics["catalog_truth"].values()) >= gate["catalog_truth_min"],
                "catalog_cell": min(metrics["catalog_cell"].values()) >= gate["catalog_cell_min"],
                "required_set_all": len(family_eligible) / 12 >= gate["required_set_all_min"],
                "factorial": all(
                    count == protocol["material"]["selected_per_family_partition"]
                    for count in partition_counts.values()
                ),
            }
            qualified = all(checks.values())
            family_results[family] = {"metrics": metrics, "checks": checks, "qualified": qualified}
            if qualified:
                selected.extend(sorted(family_eligible, key=lambda row: row["set_id"]))

        qualified_families = [family for family, result in family_results.items() if result["qualified"]]
        breadth = {
            "minimum_family_breadth": len(qualified_families) >= protocol["material"]["minimum_qualified_families"],
            "numeric_repeat": numeric_repeat <= gate["same_shape_repeat_max_abs_diff"],
            "finite": all(math.isfinite(value) for row in behavior for value in row["scores"]),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        behavior_qualified = all(breadth.values())
        core.write_rows(OUT / "raw/active_behavior.jsonl", behavior)
        core.write_rows(OUT / "material/eligible_composition_sets.jsonl", selected)
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "global_accuracy": accuracy(behavior),
            "global_balanced_accuracy": balanced_accuracy(behavior),
            "catalog_accuracy": accuracy([row for row in behavior if row["surface"] == "catalog"]),
            "catalog_balanced_accuracy": balanced_accuracy([row for row in behavior if row["surface"] == "catalog"]),
            "ordinary_accuracy_descriptive": accuracy([row for row in behavior if row["surface"] == "ordinary"]),
            "family_results": family_results,
            "qualified_families": qualified_families,
            "selected_count": len(selected),
            "selected_partition_counts": {
                name: sum(row["partition"] == name for row in selected)
                for name in protocol["material"]["partitions"]
            },
            "breadth_checks": breadth,
            "behavior_qualified": behavior_qualified,
            "numeric_same_shape_max_abs_diff": numeric_repeat,
            "hidden_state_accessed": False,
            "runtime": {
                "placement": placement,
                "quantization": quant,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        core.save(OUT / "analysis/behavior_summary.json", summary)
        authorization = "run_phase1422_c069_quartet_camera" if behavior_qualified else "close_c069_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "behavior_qualified": behavior_qualified,
            "qualified_families": qualified_families,
            "authorization": authorization,
        })
        print(json.dumps({key: value for key, value in summary.items() if key != "family_results"}, indent=2))
        print(json.dumps(family_results, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
