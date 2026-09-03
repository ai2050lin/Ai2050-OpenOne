#!/usr/bin/env python3
"""Phase1461: behavior qualification for C078 colon-label observation."""
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
import phase1457_c077_behavior as prior
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1461, "C078"
CONTRACT = TESTS / "result/phase1460_c078_colon_label_contract"
OUT = TESTS / "result/phase1461_c078_behavior"
BATCH = 24


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1461 exists")
    contract = core.load(CONTRACT / "analysis/final.json")
    audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if contract["authorization"] != "run_phase1461_c078_behavior" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1460 did not authorize behavior")
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
        repeated = runner.forward(model, compiled[:BATCH], pad, device, supports)
        repeat_error = max(abs(left["scores"][i] - right["scores"][i]) for left, right in zip(first, repeated) for i in range(2))
        behavior = [{**row, **pred, "correct": pred["prediction"] == row["gold_position"]} for row, pred in zip(source, predictions)]
        by_case = {row["case_id"]: row for row in behavior}
        eligible = [row for row in composition if all(by_case[row[key]]["correct"] for key in set_keys)]
        gate = protocol["behavior"]
        graph = core.load(CONTRACT / "material/frozen_concept_graph.json")
        detail = {}
        for family in graph["families"]:
            detail[family] = {}
            for relation_id in protocol["relations"]:
                detail[family][relation_id] = {}
                for surface in protocol["surfaces"]:
                    rows = [row for row in behavior if row["record_object"] == family and row["record_relation_id"] == relation_id and row["surface"] == surface]
                    metrics = {
                        "count": len(rows),
                        "accuracy": prior.accuracy(rows),
                        "balanced_accuracy": prior.balanced_accuracy(rows),
                        "partition": {name: prior.accuracy([row for row in rows if row["partition"] == name]) for name in protocol["partitions"]},
                        "truth": {str(value).lower(): prior.accuracy([row for row in rows if row["truth"] == value]) for value in (True, False)},
                        "cell": {cell: prior.accuracy([row for row in rows if row["cell"] == cell]) for cell in protocol["cells"]},
                    }
                    checks = {
                        "accuracy": metrics["accuracy"] >= gate["family_relation_surface_accuracy_min"],
                        "balanced_accuracy": metrics["balanced_accuracy"] >= gate["family_relation_surface_balanced_accuracy_min"],
                        "partition": min(metrics["partition"].values()) >= gate["partition_min"],
                        "truth": min(metrics["truth"].values()) >= gate["truth_min"],
                        "cell": min(metrics["cell"].values()) >= gate["cell_min"],
                    }
                    detail[family][relation_id][surface] = {"metrics": metrics, "checks": checks, "qualified": all(checks.values())}
        surface_global = {surface: {"accuracy": prior.accuracy([row for row in behavior if row["surface"] == surface]), "balanced_accuracy": prior.balanced_accuracy([row for row in behavior if row["surface"] == surface])} for surface in protocol["surfaces"]}
        split_counts = {name: sum(row["partition"] == name for row in eligible) for name in protocol["partitions"]}
        relation_counts = Counter(row["record_relation_id"] for row in eligible)
        checks = {
            "all_family_relation_surface": all(detail[family][relation][surface]["qualified"] for family in detail for relation in detail[family] for surface in detail[family][relation]),
            "all_surfaces": all(value["balanced_accuracy"] >= gate["global_surface_balanced_accuracy_min"] for value in surface_global.values()),
            "eligible_total": len(eligible) >= gate["eligible_set_total_min"],
            "eligible_splits": min(split_counts.values()) >= gate["eligible_set_split_min"],
            "eligible_relations": min(relation_counts.values(), default=0) >= gate["eligible_set_relation_min"] and len(relation_counts) == len(protocol["relations"]),
            "repeat": repeat_error <= gate["same_batch_repeat_max_abs_diff"],
            "finite": all(math.isfinite(value) for row in behavior for value in row["scores"]),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        qualified = all(checks.values())
        core.write_rows(OUT / "raw/active_behavior.jsonl", behavior)
        core.write_rows(OUT / "material/eligible_composition_sets.jsonl", eligible)
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "global_accuracy": prior.accuracy(behavior),
            "global_balanced_accuracy": prior.balanced_accuracy(behavior),
            "surface_global": surface_global,
            "family_relation_surface": detail,
            "eligible_count": len(eligible),
            "eligible_partition_counts": split_counts,
            "eligible_relation_counts": dict(relation_counts),
            "error_surface_truth_counts": {surface: {str(truth).lower(): sum(not row["correct"] for row in behavior if row["surface"] == surface and row["truth"] == truth) for truth in (True, False)} for surface in protocol["surfaces"]},
            "checks": checks,
            "behavior_qualified": qualified,
            "numeric_repeat_max_abs_diff": repeat_error,
            "hidden_state_accessed": False,
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/behavior_summary.json", summary)
        authorization = "run_phase1462_c078_discovery_full_field_capture" if qualified else "close_c078_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "behavior_qualified": qualified, "eligible_count": len(eligible), "authorization": authorization})
        print(json.dumps({key: value for key, value in summary.items() if key != "family_relation_surface"}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
