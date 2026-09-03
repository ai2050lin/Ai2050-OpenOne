#!/usr/bin/env python3
"""Phase1470: behavior qualification for the C080 explicit-label branch."""
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
import phase1469_c080_balanced_interaction_contract as contract_module
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

CONTRACT = TESTS / "result/phase1469_c080_balanced_interaction_contract"
OUT = TESTS / "result/phase1470_c080_explicit_behavior"
BATCH = 32


def subset_metrics(rows: list[dict], protocol: dict) -> dict:
    surfaces = protocol["branches"]["explicit"]["surfaces"]
    result = {
        "surface": {},
        "nuisance_surface": {},
        "equal_label_surface": {},
        "unequal_pair_surface": {},
    }
    for surface in surfaces:
        values = [row for row in rows if row["surface"] == surface]
        result["surface"][surface] = {
            "accuracy": metric.accuracy(values),
            "balanced_accuracy": metric.balanced_accuracy(values),
            "partition": {name: metric.balanced_accuracy([row for row in values if row["partition"] == name]) for name in protocol["partitions"]},
            "truth": {str(truth).lower(): metric.accuracy([row for row in values if row["truth"] == truth]) for truth in (True, False)},
        }
        for nuisance in protocol["nuisance_cells"]:
            current = [row for row in values if row["nuisance_cell"] == nuisance]
            result["nuisance_surface"][f"{surface}__{nuisance}"] = metric.balanced_accuracy(current)
        for relation_id in contract_module.IDS:
            current = [row for row in values if row["record_relation_id"] == relation_id and row["query_relation_id"] == relation_id]
            result["equal_label_surface"][f"{relation_id}__{surface}"] = metric.accuracy(current)
        for pair in contract_module.PAIR_IDS:
            left, right = pair.split("__")
            current = [row for row in values if {row["record_relation_id"], row["query_relation_id"]} == {left, right}]
            result["unequal_pair_surface"][f"{pair}__{surface}"] = metric.accuracy(current)
    return result


def evaluate(rows: list[dict], sets: list[dict], protocol: dict, repeat_error: float, quant: dict) -> tuple[dict, list[dict]]:
    by_case = {row["case_id"]: row for row in rows}
    reference_keys = [key for key in sets[0] if key.startswith(tuple(protocol["branches"]["explicit"]["surfaces"]))]
    eligible = [row for row in sets if all(by_case[row[key]]["correct"] for key in reference_keys)]
    metrics = subset_metrics(rows, protocol)
    gate = protocol["branches"]["explicit"]["behavior"]
    split_counts = Counter(row["partition"] for row in eligible)
    pair_counts = Counter(row["pair_id"] for row in eligible)
    checks = {
        "global": metric.balanced_accuracy(rows) >= gate["global_surface_balanced_accuracy_min"],
        "surface_partition": min(score for value in metrics["surface"].values() for score in value["partition"].values()) >= gate["surface_partition_balanced_accuracy_min"],
        "surface_truth": min(score for value in metrics["surface"].values() for score in value["truth"].values()) >= gate["surface_truth_accuracy_min"],
        "nuisance_surface": min(metrics["nuisance_surface"].values()) >= gate["nuisance_surface_balanced_accuracy_min"],
        "equal_label_surface": min(metrics["equal_label_surface"].values()) >= gate["equal_label_surface_accuracy_min"],
        "unequal_pair_surface": min(metrics["unequal_pair_surface"].values()) >= gate["unequal_pair_surface_accuracy_min"],
        "eligible_total": len(eligible) >= gate["eligible_set_total_min"],
        "eligible_splits": len(split_counts) == 3 and min(split_counts.values()) >= gate["eligible_set_split_min"],
        "eligible_pairs": len(pair_counts) == 15 and min(pair_counts.values()) >= gate["eligible_set_pair_min"],
        "repeat": repeat_error <= gate["same_batch_repeat_max_abs_diff"],
        "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
        "bf16": quant["has_bf16_parameters"],
        "not_quantized": not quant["has_quantized_modules"],
        "hidden_not_accessed": True,
    }
    summary = {
        "phase": 1470,
        "campaign": "C080",
        "branch": "explicit",
        "global_accuracy": metric.accuracy(rows),
        "global_balanced_accuracy": metric.balanced_accuracy(rows),
        **metrics,
        "eligible_count": len(eligible),
        "eligible_partition_counts": dict(split_counts),
        "eligible_pair_counts": dict(pair_counts),
        "error_counts": {
            "truth": {str(value).lower(): sum(not row["correct"] for row in rows if row["truth"] == value) for value in (True, False)},
            "surface": {surface: sum(not row["correct"] for row in rows if row["surface"] == surface) for surface in protocol["branches"]["explicit"]["surfaces"]},
        },
        "numeric_repeat_max_abs_diff": repeat_error,
        "checks": checks,
        "behavior_qualified": all(checks.values()),
        "hidden_state_accessed": False,
    }
    return summary, eligible


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1470 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1470_c080_explicit_behavior" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1469 did not authorize Phase1470")
    source = core.rows(CONTRACT / "material/explicit_active_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_explicit.jsonl")
    sets = core.rows(CONTRACT / "material/explicit_interaction_sets.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        predictions, first = [], None
        for start in range(0, len(compiled), BATCH):
            block = runner.forward(model, compiled[start:start + BATCH], pad, device, supports)
            if first is None:
                first = block
            predictions.extend(block)
        repeated = runner.forward(model, compiled[:BATCH], pad, device, supports)
        repeat_error = max(abs(left["scores"][index] - right["scores"][index]) for left, right in zip(first, repeated) for index in range(2))
        rows = [{**row, **prediction, "correct": prediction["prediction"] == row["gold_position"]} for row, prediction in zip(source, predictions)]
        summary, eligible = evaluate(rows, sets, protocol, repeat_error, quant)
        summary["runtime"] = {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        core.write_rows(OUT / "raw/explicit_behavior.jsonl", rows)
        core.write_rows(OUT / "material/eligible_interaction_sets.jsonl", eligible)
        core.save(OUT / "analysis/behavior_summary.json", summary)
        authorization = "run_phase1471_c080_explicit_discovery_capture" if summary["behavior_qualified"] else "close_c080_explicit_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {"phase": 1470, "campaign": "C080", "behavior_qualified": summary["behavior_qualified"], "eligible_count": len(eligible), "authorization": authorization})
        print(json.dumps({key: value for key, value in summary.items() if key not in ("equal_label_surface", "unequal_pair_surface", "runtime")}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
