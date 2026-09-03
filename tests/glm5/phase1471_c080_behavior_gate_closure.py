#!/usr/bin/env python3
"""Phase1471: close C080 after the explicit behavior gate failure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
CONTRACT = TESTS / "result/phase1469_c080_balanced_interaction_contract"
BEHAVIOR = TESTS / "result/phase1470_c080_explicit_behavior"
OUT = TESTS / "result/phase1471_c080_behavior_gate_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1471 exists")
    contract = core.load(CONTRACT / "analysis/final.json")
    contract_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    summary = core.load(BEHAVIOR / "analysis/behavior_summary.json")
    rows = core.rows(BEHAVIOR / "raw/explicit_behavior.jsonl")
    for path in (
        TESTS / "phase1469_c080_balanced_interaction_contract.py",
        TESTS / "phase1469_c080_balanced_interaction_contract_audit.py",
        TESTS / "phase1470_c080_explicit_behavior.py",
        TESTS / "phase1470_c080_explicit_behavior_audit.py",
    ):
        py_compile.compile(str(path), doraise=True)
    errors = [row for row in rows if not row["correct"]]
    checks = {
        "audits": contract_audit["all_checks_passed"] and behavior_audit["all_checks_passed"],
        "authorization_chain": contract["authorization"] == "run_phase1470_c080_explicit_behavior" and behavior["authorization"] == "close_c080_explicit_at_behavior_gate",
        "behavior_failed": not summary["behavior_qualified"] and not summary["checks"]["global"],
        "surface_split": summary["surface"]["a_explicit"]["balanced_accuracy"] > 0.99 and summary["surface"]["b_explicit"]["balanced_accuracy"] < 0.70,
        "positive_only_errors": len(errors) == 607 and all(row["truth"] for row in errors),
        "all_negatives_correct": summary["error_counts"]["truth"]["false"] == 0,
        "no_eligible_sets": summary["eligible_count"] == 0,
        "numeric_healthy": summary["checks"]["repeat"] and summary["checks"]["finite"] and summary["checks"]["bf16"] and summary["checks"]["not_quantized"],
        "hidden_denied": summary["hidden_state_accessed"] is False,
        "scripts_compile": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": 1471,
        "campaign": "C080",
        "status": "closed_at_explicit_behavior_gate_with_surface_specific_positive_failure",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "retained": [
            "complete preregistered 6x6 equality-interaction material and zero-model contract",
            "near-perfect behavior on explicit surface A",
            "numeric health and deterministic logits",
            "surface-specific positive-class failure localized to explicit surface B",
        ],
        "not_tested": [
            "any C080 Hidden State",
            "the balanced equality interaction field",
            "the label-withdrawal natural-verb branch",
            "causal use, neurons, attention, MLP, parameters, or cross-model invariance",
        ],
        "adjudication": "the current dual-surface execution contract failed; the equality-interaction hypothesis was not tested",
        "rescue_limit": "one fresh-material rescue using only the two historically qualified C079 interface families; rescue failure closes the explicit-label interaction route",
        "authorization": "preregister_c081_historically_validated_interface_rescue_on_fresh_material",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
