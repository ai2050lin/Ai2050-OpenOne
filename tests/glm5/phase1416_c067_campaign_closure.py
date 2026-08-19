#!/usr/bin/env python3
"""Phase1416: close C067 and separate graded effects from failed composition."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1416_c067_campaign_closure"
P1412 = TESTS / "result/phase1412_c067_paired_state_composition_contract"
P1413 = TESTS / "result/phase1413_c067_behavior"
P1414 = TESTS / "result/phase1414_c067_dual_write_camera"
P1415 = TESTS / "result/phase1415_c067_paired_composition"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1416 exists")
    protocol = core.load(P1412 / "protocol/preregistration.json")
    behavior = core.load(P1413 / "analysis/behavior_summary.json")
    camera = core.load(P1414 / "analysis/camera_summary.json")
    composition = core.load(P1415 / "analysis/composition_summary.json")
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1412, P1413, P1414, P1415)]
    scripts = []
    for phase, stem in (
        (1412, "c067_paired_state_composition_contract"),
        (1413, "c067_behavior"),
        (1414, "c067_dual_write_camera"),
        (1415, "c067_paired_composition"),
    ):
        scripts.extend([TESTS / f"phase{phase}_{stem}.py", TESTS / f"phase{phase}_{stem}_audit.py"])
    compiled = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            compiled = False
    split = composition["split_metrics"]
    checks = {
        "audits": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": compiled,
        "behavior_six_families": len(behavior["qualified_families"]) == 6,
        "behavior_72_sets": behavior["selected_count"] == 72,
        "camera_exact": camera["camera_qualified"] and camera["qwen_output_max_abs_diff"] == 0.0,
        "holdout_48": composition["holdout_set_count"] == 48,
        "state16_catalog": protocol["mechanism"]["state_index"] == 16 and protocol["mechanism"]["surface"] == "catalog",
        "record_graded_both": all(split[name]["checks"]["record_damage"] and split[name]["checks"]["record_win"] for name in ("confirmation", "lockbox")),
        "matched_order_both": all(split[name]["checks"]["matched_positive"] and split[name]["checks"]["matched_over_mismatched"] for name in ("confirmation", "lockbox")),
        "query_redirect_failed_both": all(not split[name]["checks"]["query_redirect"] for name in ("confirmation", "lockbox")),
        "mismatch_negative_failed_both": all(not split[name]["checks"]["mismatched_negative"] for name in ("confirmation", "lockbox")),
        "composition_not_confirmed": not composition["composition_confirmed"] and composition["qualified_families"] == [],
    }
    result = {
        "phase": 1416,
        "campaign": "C067",
        "status": "closed_after_failed_discrete_pair_composition_with_graded_ordering",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "retained": {
            "behavior": "six fresh families and 72 balanced composition sets qualified",
            "camera": "dual-role self write is exact on Qwen logits and 256 known-truth systems",
            "graded_effects": {
                "record_damage_both_splits": True,
                "matched_positive_both_splits": True,
                "matched_over_mismatched_both_splits": True,
            },
        },
        "rejected": {
            "strong_hypothesis": "catalog state16 record_family/query_family singleton full states form a sufficient discrete family-equality comparator under the frozen write semantics",
            "failed_gates": {
                "confirmation": [key for key, value in split["confirmation"]["checks"].items() if not value],
                "lockbox": [key for key, value in split["lockbox"]["checks"].items() if not value],
            },
            "qualified_families": composition["qualified_families"],
        },
        "untested": [
            "distributed tuples that also include record_target and query_target",
            "other states, surfaces, models, languages, and open text",
            "minimal sufficient or necessary event bundles",
            "attention, MLP, parameters, gradients, or learned probes",
        ],
        "claim_boundary": {
            "allowed": "Qwen-specific controlled-registry graded state16 role-pair response plus failure of the frozen sufficient-comparator hypothesis",
            "forbidden": [
                "relation manifold discovered",
                "semantic comparator localized",
                "relative encoding refuted",
                "minimal or unique mechanism",
                "cross-model law",
            ],
        },
        "next_question": {
            "campaign": "C068",
            "object": "catalog state16 four-role tuple: record_target, record_family, query_target, query_family",
            "reason": "C067 changed margins but left all arms strongly positive, indicating omitted distributed recipient state rather than a confirmed two-state comparator",
            "constraints": ["no layer search", "no attention/MLP", "new material", "behavior first", "matched and mismatched natural-true donors"],
        },
        "authorization": "preregister_c068_distributed_four_role_composition",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
