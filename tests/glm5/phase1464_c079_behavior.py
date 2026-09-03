#!/usr/bin/env python3
"""Phase1464: aggregate behavior qualification for C079."""
from __future__ import annotations

import inspect
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
import phase1457_c077_behavior as metric
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

CONTRACT = TESTS / "result/phase1463_c079_aggregate_observation_contract"
OUT = TESTS / "result/phase1464_c079_behavior"
BATCH = 24


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1464 exists")
    contract = core.load(CONTRACT / "analysis/final.json")
    audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if contract["authorization"] != "run_phase1464_c079_behavior" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1463 did not authorize behavior")
    source = core.rows(CONTRACT / "material/active_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    composition = core.rows(CONTRACT / "material/composition_sets.jsonl")
    keys = tuple(f"{surface}_{cell}" for surface in protocol["surfaces"] for cell in protocol["cells"])
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
        repeated = runner.forward(model, compiled[:BATCH], pad, device, supports)
        repeat_error = max(abs(left["scores"][i] - right["scores"][i]) for left, right in zip(first, repeated) for i in range(2))
        rows = [{**row, **pred, "correct": pred["prediction"] == row["gold_position"]} for row, pred in zip(source, predictions)]
        by_case = {row["case_id"]: row for row in rows}
        eligible = [row for row in composition if all(by_case[row[key]]["correct"] for key in keys)]
        gate = protocol["behavior"]
        surface = {}
        for surface_id in protocol["surfaces"]:
            values = [row for row in rows if row["surface"] == surface_id]
            surface[surface_id] = {
                "accuracy": metric.accuracy(values),
                "balanced_accuracy": metric.balanced_accuracy(values),
                "partition": {name: metric.balanced_accuracy([row for row in values if row["partition"] == name]) for name in protocol["partitions"]},
                "truth": {str(truth).lower(): metric.accuracy([row for row in values if row["truth"] == truth]) for truth in (True, False)},
            }
        relation_surface = {}
        for relation in protocol["relations"]:
            relation_surface[relation] = {}
            for surface_id in protocol["surfaces"]:
                values = [row for row in rows if row["record_relation_id"] == relation and row["surface"] == surface_id]
                relation_surface[relation][surface_id] = {"count": len(values), "accuracy": metric.accuracy(values), "balanced_accuracy": metric.balanced_accuracy(values)}
        split_counts = {name: sum(row["partition"] == name for row in eligible) for name in protocol["partitions"]}
        relation_counts = Counter(row["record_relation_id"] for row in eligible)
        checks = {
            "surface": min(value["balanced_accuracy"] for value in surface.values()) >= gate["global_surface_balanced_accuracy_min"],
            "surface_partition": min(score for value in surface.values() for score in value["partition"].values()) >= gate["surface_partition_balanced_accuracy_min"],
            "surface_truth": min(score for value in surface.values() for score in value["truth"].values()) >= gate["surface_truth_accuracy_min"],
            "relation_surface": min(value["balanced_accuracy"] for surfaces in relation_surface.values() for value in surfaces.values()) >= gate["relation_surface_balanced_accuracy_min"],
            "eligible_total": len(eligible) >= gate["eligible_set_total_min"],
            "eligible_splits": min(split_counts.values()) >= gate["eligible_set_split_min"],
            "eligible_relations": len(relation_counts) == len(protocol["relations"]) and min(relation_counts.values()) >= gate["eligible_set_relation_min"],
            "repeat": repeat_error <= gate["same_batch_repeat_max_abs_diff"],
            "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        qualified = all(checks.values())
        core.write_rows(OUT / "raw/active_behavior.jsonl", rows)
        core.write_rows(OUT / "material/eligible_composition_sets.jsonl", eligible)
        summary = {
            "phase": 1464,
            "campaign": "C079",
            "global_accuracy": metric.accuracy(rows),
            "global_balanced_accuracy": metric.balanced_accuracy(rows),
            "surface": surface,
            "relation_surface": relation_surface,
            "eligible_count": len(eligible),
            "eligible_partition_counts": split_counts,
            "eligible_relation_counts": dict(relation_counts),
            "error_surface_truth_counts": {surface_id: {str(truth).lower(): sum(not row["correct"] for row in rows if row["surface"] == surface_id and row["truth"] == truth) for truth in (True, False)} for surface_id in protocol["surfaces"]},
            "checks": checks,
            "behavior_qualified": qualified,
            "numeric_repeat_max_abs_diff": repeat_error,
            "hidden_state_accessed": False,
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/behavior_summary.json", summary)
        authorization = "run_phase1465_c079_discovery_full_field_capture" if qualified else "close_c079_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {"phase": 1464, "campaign": "C079", "behavior_qualified": qualified, "eligible_count": len(eligible), "authorization": authorization})
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
