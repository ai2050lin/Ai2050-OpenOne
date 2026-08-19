#!/usr/bin/env python3
"""Campaign-level closure audit for C059 phases 1375-1379."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE_DIRS = {
    1375: TESTS / "result/phase1375_c059_independent_relaunch_contract",
    1376: TESTS / "result/phase1376_c059_qwen_behavior_qualification",
    1377: TESTS / "result/phase1377_c059_response_field_camera",
    1378: TESTS / "result/phase1378_c059_dose_distance_observation",
    1379: TESTS / "result/phase1379_c059_coordinate_group_evaluation",
}
SCRIPTS = [
    TESTS / "phase1375_c059_independent_relaunch_contract.py",
    TESTS / "phase1375_c059_independent_relaunch_contract_audit.py",
    TESTS / "phase1376_c059_qwen_behavior_qualification.py",
    TESTS / "phase1376_c059_qwen_behavior_qualification_audit.py",
    TESTS / "phase1377_c059_response_field_camera.py",
    TESTS / "phase1377_c059_response_field_camera_audit.py",
    TESTS / "phase1378_c059_dose_distance_observation.py",
    TESTS / "phase1378_c059_dose_distance_observation_audit.py",
    TESTS / "phase1379_c059_coordinate_group_evaluation.py",
    TESTS / "phase1379_c059_coordinate_group_evaluation_audit.py",
    TESTS / "phase1379_c059_coordinate_group_split_postanalysis.py",
]


def main() -> None:
    audits = {phase: core.load(path / "audit/independent_final_audit.json")
              for phase, path in PHASE_DIRS.items()}
    finals = {phase: core.load(path / "analysis/final.json")
              for phase, path in PHASE_DIRS.items()}
    for script in SCRIPTS + [Path(__file__)]:
        py_compile.compile(str(script), doraise=True)
    forbidden_access_patterns = (".self_attn", ".mlp", "output_attentions=True", "named_parameters(",
                                 "torch.pca", "PCA(", "UMAP(", "TSNE(")
    forbidden_hits = {script.name: [pattern for pattern in forbidden_access_patterns
                                    if pattern in script.read_text(encoding="utf-8")]
                      for script in SCRIPTS}
    authorization_chain = [
        finals[1375]["authorization"], finals[1376]["authorization"], finals[1377]["authorization"],
        finals[1378]["authorization"], finals[1379]["authorization"],
    ]
    expected_chain = [
        "run_phase1376_c059_behavior_qualification",
        "run_phase1377_c059_instrument_calibration",
        "run_phase1378_c059_dose_distance_observation",
        "run_phase1379_c059_coordinate_group_evaluation",
        "close_c059_after_all_frozen_eligible_branches",
    ]
    checks = {
        "all_phase_audits_pass": all(a["all_checks_passed"] for a in audits.values()),
        "authorization_chain": authorization_chain == expected_chain,
        "behavior_before_hidden": finals[1376]["behavior_qualified"] and finals[1377]["camera_qualified"],
        "mediation_not_eligible": finals[1378]["mediation_eligible"] is False,
        "mediation_not_run": not (TESTS / "result/phase1380_c059_early_mediation").exists(),
        "coordinate_branch_completed": finals[1379]["any_sufficiency_route"] is True,
        "forbidden_access_absent": not any(forbidden_hits.values()),
        "scripts_compile": True,
    }
    artifact = {
        "phase": 1379, "campaign": "C059", "kind": "campaign_closure_audit",
        "authorization_chain": authorization_chain, "forbidden_access_hits": forbidden_hits,
        "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "campaign_status": "closed_after_all_frozen_eligible_branches" if all(checks.values()) else "closure_audit_failed",
    }
    target = PHASE_DIRS[1379] / "audit/campaign_closure_audit.json"
    core.save(target, artifact)
    print(json.dumps(artifact, indent=2))
    if not artifact["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
