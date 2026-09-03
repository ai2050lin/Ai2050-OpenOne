#!/usr/bin/env python3
"""Phase1458: close C077 after the frozen behavior gate failed."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1458, "C077"
CONTRACT = TESTS / "result/phase1456_c077_labeled_relation_contract"
BEHAVIOR = TESTS / "result/phase1457_c077_behavior"
OUT = TESTS / "result/phase1458_c077_behavior_gate_closure"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1458 exists")
    ca = core.load(CONTRACT / "audit/independent_final_audit.json")
    ba = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    bf = core.load(BEHAVIOR / "analysis/final.json")
    summary = core.load(BEHAVIOR / "analysis/behavior_summary.json")
    rows = core.rows(BEHAVIOR / "raw/active_behavior.jsonl")
    compiled = []
    for path in (
        TESTS / "phase1456_c077_labeled_relation_contract.py",
        TESTS / "phase1456_c077_labeled_relation_contract_audit.py",
        TESTS / "phase1457_c077_behavior.py",
        TESTS / "phase1457_c077_behavior_audit.py",
    ):
        py_compile.compile(str(path), doraise=True)
        compiled.append(path.name)
    errors = [row for row in rows if not row["correct"]]
    by_surface_truth = {}
    for surface in sorted({row["surface"] for row in rows}):
        by_surface_truth[surface] = {}
        for truth in (True, False):
            subset = [row for row in rows if row["surface"] == surface and row["truth"] == truth]
            by_surface_truth[surface][str(truth).lower()] = {
                "count": len(subset),
                "correct": sum(row["correct"] for row in subset),
                "accuracy": sum(row["correct"] for row in subset) / len(subset),
            }
    checks = {
        "contract_audit": ca["all_checks_passed"],
        "behavior_audit": ba["all_checks_passed"],
        "behavior_failed": not summary["behavior_qualified"],
        "close_authorized": bf["authorization"] == "close_c077_at_behavior_gate",
        "surface_split": summary["surface_global"]["a_labeled"]["balanced_accuracy"] > 0.99 and summary["surface_global"]["b_labeled"]["balanced_accuracy"] < 0.70,
        "b_true_specific": by_surface_truth["b_labeled"]["true"]["accuracy"] < 0.40 and by_surface_truth["b_labeled"]["false"]["accuracy"] == 1.0,
        "numeric_healthy": summary["checks"]["repeat"] and summary["checks"]["finite"] and summary["checks"]["bf16"] and summary["checks"]["not_quantized"],
        "hidden_not_accessed": summary["hidden_state_accessed"] is False,
        "no_capture": not (TESTS / "result/phase1458_c077_discovery_full_field_capture").exists(),
        "scripts_compile": len(compiled) == 4,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed_at_behavior_gate_after_surface_specific_true_label_failure",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "error_count": len(errors),
        "error_surface_counts": Counter(row["surface"] for row in errors),
        "surface_truth": by_surface_truth,
        "retained": [
            "surface A behavior was nearly perfect",
            "surface B failures were concentrated on equal-label truth cases",
            "BF16 CUDA execution and repeatability were healthy",
        ],
        "rejected": [
            "C077 behavior qualification",
            "any C077 hidden-state conclusion",
            "relation labels absent from Qwen3",
            "surface A may be retained post hoc inside C077",
        ],
        "authorization": "preregister_c078_colon_label_observation_campaign",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
