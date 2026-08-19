#!/usr/bin/env python3
"""Phase1419: close C068 at the preregistered behavior gate."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1419_c068_behavior_gate_closure"
P1417 = TESTS / "result/phase1417_c068_four_role_composition_contract"
P1418 = TESTS / "result/phase1418_c068_behavior"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1419 exists")
    protocol = core.load(P1417 / "protocol/preregistration.json")
    behavior = core.load(P1418 / "analysis/behavior_summary.json")
    behavior_final = core.load(P1418 / "analysis/final.json")
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1417, P1418)]
    scripts = []
    for phase, stem in ((1417, "c068_four_role_composition_contract"), (1418, "c068_behavior")):
        scripts.extend([TESTS / f"phase{phase}_{stem}.py", TESTS / f"phase{phase}_{stem}_audit.py"])
    compiled = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            compiled = False
    catalog_by_family = {family: result["metrics"]["surface"]["catalog"] for family, result in behavior["family_results"].items()}
    statement_by_family = {family: result["metrics"]["surface"]["statement"] for family, result in behavior["family_results"].items()}
    checks = {
        "audits": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": compiled,
        "behavior_failed": not behavior["behavior_qualified"] and behavior_final["authorization"] == "close_c068_at_behavior_gate",
        "only_two_families": behavior["qualified_families"] == ["organ", "month"],
        "breadth_failed": not behavior["breadth_checks"]["minimum_family_breadth"],
        "catalog_scoped_signal": min(catalog_by_family.values()) >= 0.99,
        "statement_scope_failure": min(statement_by_family.values()) < 0.80,
        "hidden_denied": not (TESTS / "result/phase1419_c068_four_role_camera").exists() and not (TESTS / "result/phase1420_c068_bidirectional_quartet_composition").exists(),
        "state16_quartet_untested": protocol["camera"]["state_index"] == 16 and len(protocol["camera"]["roles"]) == 4,
    }
    result = {
        "phase": 1419,
        "campaign": "C068",
        "status": "closed_at_behavior_gate_before_hidden_state",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "retained": {
            "qualified_families": behavior["qualified_families"],
            "catalog_accuracy_by_family": catalog_by_family,
            "eligible_set_count_by_family": {family: result["metrics"]["eligible_count"] for family, result in behavior["family_results"].items()},
            "numeric_repeat_max_abs_diff": behavior["numeric_same_shape_max_abs_diff"],
        },
        "rejected": {
            "object": "three-surface, eight-cell, minimum-four-family behavior qualification for C068",
            "reason": "only organ and month passed every frozen family gate",
        },
        "untested": [
            "state16 four-role identity camera",
            "true-to-mismatch quartet intervention",
            "false-to-match quartet rescue",
            "graded interaction and discrete quartet sufficiency",
        ],
        "design_diagnosis": "mechanism was catalog-only but family qualification conjoined unused statement performance; this scope mismatch must be repaired only in a new campaign",
        "claim_boundary": {
            "allowed": "C068 behavior interface failed its frozen cross-surface breadth gate while catalog behavior remained strong",
            "forbidden": ["quartet mechanism failed", "target identity is irrelevant", "relative encoding refuted", "hidden state negative result"],
        },
        "next_question": {
            "campaign": "C069",
            "object": "new-material catalog-scoped state16 bidirectional four-role composition",
            "contract_change": "qualify the mechanism surface and each required donor arm directly; ordinary is a donor control and statement is outside the mechanism gate",
            "unchanged": ["Qwen3 BF16", "state16", "four roles", "bidirectional nine arms", "graded/discrete ledgers", "confirmation and lockbox", "no attention/MLP/parameter/gradient/dimension reduction"],
        },
        "authorization": "preregister_c069_catalog_scoped_four_role_composition",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
