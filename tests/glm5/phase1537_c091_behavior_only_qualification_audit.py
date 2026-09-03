#!/usr/bin/env python3
"""Independent audit for Phase1537."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1537_c091_behavior_only_qualification"
CONTRACT = TESTS / "result/phase1536_c091_human_validated_chinese_relation_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def recall(rows, label):
    subset = [row for row in rows if row["gold_label"] == label]
    return sum(row["correct"] for row in subset) / len(subset)


def ba(rows):
    return (recall(rows, "是") + recall(rows, "否")) / 2


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/behavior_summary.json")
    rows = core.rows(OUT / "raw/behavior_logits.jsonl")
    three_way = core.rows(OUT / "analysis/three_way_pair_selection.jsonl")
    checks = {
        "behavior_hash": core.sha(OUT / "raw/behavior_logits.jsonl") == report["files"]["behavior_logits"]["sha256"],
        "three_way_hash": core.sha(OUT / "analysis/three_way_pair_selection.jsonl") == report["files"]["three_way"]["sha256"],
        "coverage": len(rows) == 540 and len(three_way) == 180,
        "finite": all(all(math.isfinite(value) for value in row["candidate_logits"]) for row in rows),
        "global_ba": abs(ba(rows) - report["global"]["balanced_accuracy"]) < 1e-12,
        "family_ba": all(abs(ba([row for row in rows if row["query_family"] == family and row["partition"] == "response_discovery"]) - report["family"][family]["discovery_balanced_accuracy"]) < 1e-12 for family in protocol["families"]),
        "repeat": report["repeat_logit_max_abs"] <= protocol["numeric_gate_before_hidden_use"]["repeat_logit_max_abs"],
        "no_hidden": report["checks"]["hidden_not_requested"],
        "bf16": report["checks"]["bf16"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1538_c091_behavior_gate_adjudication",
    }
    result = {
        "phase": 1537,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
