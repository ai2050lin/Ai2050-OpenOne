#!/usr/bin/env python3
"""Phase1413: behavior qualification for C067 paired composition."""
from __future__ import annotations

import inspect
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1413, "C067"
CONTRACT = TESTS / "result/phase1412_c067_paired_state_composition_contract"
OUT = TESTS / "result/phase1413_c067_behavior"
BATCH = 24
SET_KEYS = ("recipient", "surface_same", "member_same", "g_true", "h_true", "natural_fg", "natural_gf", "natural_gh", "natural_hg")


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1413 exists")
    contract_final = core.load(CONTRACT / "analysis/final.json")
    contract_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if contract_final["authorization"] != "run_phase1413_c067_behavior" or not contract_audit["all_checks_passed"]:
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
            if start == 0:
                first = values
            predictions.extend(values)
        repeat = runner.forward(model, compiled[:BATCH], pad, device, supports)
        numeric_repeat = max(abs(a["scores"][j] - b["scores"][j]) for a, b in zip(first, repeat) for j in range(2))
        behavior = [{**row, **pred, "correct": pred["prediction"] == row["gold_position"]} for row, pred in zip(source, predictions)]
        by_case = {r["case_id"]: r for r in behavior}
        eligible = [r for r in composition if all(by_case[r[key]]["correct"] for key in SET_KEYS)]
        gate = protocol["behavior"]
        family_results = {}
        selected = []
        for family in protocol["material"]["families"]:
            rows = [r for r in behavior if r["record_family"] == family]
            pair_groups = defaultdict(list)
            for row in rows:
                pair_groups[(row["pair"], row["index"], row["surface"], row["record_family"])].append(row["correct"])
            family_eligible = [r for r in eligible if r["family"] == family]
            partition_counts = {p: sum(r["partition"] == p for r in family_eligible) for p in protocol["material"]["partitions"]}
            metrics = {
                "count": len(rows),
                "accuracy": accuracy(rows),
                "partition": {p: accuracy([r for r in rows if r["partition"] == p]) for p in protocol["material"]["partitions"]},
                "surface": {s: accuracy([r for r in rows if r["surface"] == s]) for s in protocol["material"]["surfaces"]},
                "truth": {str(v).lower(): accuracy([r for r in rows if r["truth"] == v]) for v in (True, False)},
                "pair_all_fraction": sum(all(v) for v in pair_groups.values()) / len(pair_groups),
                "eligible_count": len(family_eligible),
                "eligible_partition_counts": partition_counts,
            }
            checks = {
                "accuracy": metrics["accuracy"] >= gate["family_active_accuracy_min"],
                "partition": min(metrics["partition"].values()) >= gate["family_partition_min"],
                "surface": min(metrics["surface"].values()) >= gate["family_surface_min"],
                "truth": min(metrics["truth"].values()) >= gate["family_truth_min"],
                "pair_all": metrics["pair_all_fraction"] >= gate["family_pair_all_min"],
                "factorial": all(count == protocol["material"]["selected_per_family_partition"] for count in partition_counts.values()),
            }
            qualified = all(checks.values())
            family_results[family] = {"metrics": metrics, "checks": checks, "qualified": qualified}
            if qualified:
                selected.extend(sorted(family_eligible, key=lambda r: r["set_id"]))
        qualified_families = [f for f, result in family_results.items() if result["qualified"]]
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
            "family_results": family_results,
            "qualified_families": qualified_families,
            "selected_count": len(selected),
            "selected_partition_counts": {p: sum(r["partition"] == p for r in selected) for p in protocol["material"]["partitions"]},
            "breadth_checks": breadth,
            "behavior_qualified": behavior_qualified,
            "numeric_same_shape_max_abs_diff": numeric_repeat,
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/behavior_summary.json", summary)
        authorization = "run_phase1414_c067_dual_write_camera" if behavior_qualified else "close_c067_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "behavior_qualified": behavior_qualified, "qualified_families": qualified_families, "authorization": authorization})
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
