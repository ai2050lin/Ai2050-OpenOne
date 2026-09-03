#!/usr/bin/env python3
"""Phase1551: close C094 and adjudicate the C091-C094 macro-stage."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1550_c094_discovery_behavior_qualification"
OUT = RESULT / "phase1551_c094_and_output_orthogonalization_final_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1551 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    current = core.load(PARENT / "analysis/discovery_behavior_summary.json")
    if parent["authorization"] != "run_phase1551_c094_discovery_behavior_adjudication" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1550 authorization missing")
    audited_phases = {
        1542: "phase1542_c091_final_adjudication",
        1543: "phase1543_c092_truth_output_code_factorial_contract",
        1544: "phase1544_c092_behavior_only_qualification",
        1545: "phase1545_c092_behavior_gate_adjudication_and_closure",
        1546: "phase1546_c093_symmetric_code_interface_breadth_contract",
        1547: "phase1547_c093_discovery_behavior_breadth_screen",
        1548: "phase1548_c093_interface_adjudication_and_closure",
        1549: "phase1549_c094_demonstrated_codebook_contract",
        1550: "phase1550_c094_discovery_behavior_qualification",
    }
    audits = {phase: core.load(RESULT / name / "audit/independent_final_audit.json") for phase, name in audited_phases.items()}
    if not all(value["all_checks_passed"] for value in audits.values()):
        raise RuntimeError("macro-stage audit failure")
    native_pass = current["codebooks"]["native"]["qualified"]
    reversed_pass = current["codebooks"]["reversed"]["qualified"]
    if native_pass or not reversed_pass:
        raise RuntimeError("unexpected C094 branch")
    recommendation = {
        "next_campaign": "C095",
        "automatic_model_run_authorized": False,
        "required_object_change": "replace prompt-level arbitrary code reversal with either a separately calibrated known-truth compiler or a code-free pre-output semantic response object",
        "reason": "three frozen campaigns show that output-interface behavior is not stable enough to identify T versus A without prompt-specific confounding",
        "allowed_preparation": [
            "reuse existing C091 and C092-C094 logits for code-bias accounting",
            "calibrate a compiler in a known-truth synthetic model before returning to Qwen3",
            "pre-register a code-free response target that does not require arbitrary emitted labels",
        ],
    }
    report = {
        "phase": 1551,
        "campaign": "C094 and C091-C094 macro-stage",
        "status": "macro_stage_complete_with_k267_preserved_and_output_orthogonalization_unresolved",
        "audits": {str(phase): f"{value['passed']}/{value['total']}" for phase, value in audits.items()},
        "c094": {"native_qualified": native_pass, "reversed_qualified": reversed_pass, "hidden_states_accessed": False},
        "macro_answer": {
            "established": "K267 is a prospectively repeated, behavior-qualified late-boundary descriptive response field in C091",
            "failed_identification": "C092-C094 did not produce an interface where semantic truth and emitted answer identity could both be behavior-qualified and orthogonally varied",
            "interpretation": "K267 remains valid but cannot be upgraded to pure whole-part truth coding",
        },
        "core_puzzle_update": "none_after_K267",
        "theory": {
            "name": "条件化输出场闭合理论",
            "update": "late-boundary state is a mixture candidate whose truth, codebook, emitted-token, and task-termination components require independent qualification",
            "formula": "H=S+y*T+c*C+(y*c)*A+epsilon",
            "identified_terms": [],
            "new_mathematics_gate": "closed",
        },
        "hard_limits": [
            "single Qwen3-4B model",
            "controlled Chinese whole-part task",
            "prompt-level compiler behavior unstable",
            "no intervention, cross-model replication, or neuron identity",
        ],
        "recommendation": recommendation,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/macro_stage_final_adjudication.json", report)
    core.save(OUT / "protocol/next_stage_requirements.json", recommendation)
    core.save(OUT / "analysis/final.json", {"phase": 1551, "campaign": "C094", "status": report["status"], "authorization": "no_automatic_model_run_until_C095_object_contract"})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
