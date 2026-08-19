#!/usr/bin/env python3
"""Phase1386: deterministic closure of the completed C060 campaign."""
from __future__ import annotations

import json
import py_compile
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1386_c060_campaign_closure"
PHASES = {
    1380: TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract",
    1381: TESTS / "result/phase1381_c060_qwen_behavior_qualification",
    1382: TESTS / "result/phase1382_c060_response_coalition_camera",
    1383: TESTS / "result/phase1383_c060_refined_dose_observation",
    1384: TESTS / "result/phase1384_c060_fixed_dynamic_coalitions",
    1385: TESTS / "result/phase1385_c060_early_mediation",
}
SCRIPTS = [
    TESTS / f"phase{phase}_c060_{name}.py"
    for phase, name in (
        (1380, "conditional_coalition_campaign_contract"),
        (1380, "conditional_coalition_campaign_contract_audit"),
        (1381, "qwen_behavior_qualification"),
        (1381, "qwen_behavior_qualification_audit"),
        (1382, "response_coalition_camera"),
        (1382, "response_coalition_camera_audit"),
        (1383, "refined_dose_observation"),
        (1383, "refined_dose_observation_audit"),
        (1384, "fixed_dynamic_coalitions"),
        (1384, "fixed_dynamic_coalitions_audit"),
        (1385, "early_mediation"),
        (1385, "early_mediation_audit"),
    )
]


def main() -> None:
    target = OUT / "analysis/final.json"
    if target.exists():
        raise RuntimeError("Phase1386 closure already exists")
    audits = {phase: core.load(path / "audit/independent_final_audit.json") for phase, path in PHASES.items()}
    finals = {phase: core.load(path / "analysis/final.json") for phase, path in PHASES.items()}
    expected_auth = {
        1380: "run_phase1381_c060_behavior_qualification",
        1381: "run_phase1382_c060_instrument_calibration",
        1382: "run_phase1383_c060_refined_dose_observation",
        1383: "run_phase1384_c060_fixed_dynamic_coalitions",
        1384: "run_phase1385_c060_early_mediation",
        1385: "run_phase1386_c060_campaign_closure",
    }
    for script in SCRIPTS + [Path(__file__)]:
        py_compile.compile(str(script), doraise=True)
    runtime_scripts = [script for script in SCRIPTS if "audit" not in script.stem and "contract" not in script.stem]
    forbidden_patterns = (
        r"output_attentions\s*=\s*True",
        r"\.self_attn\b",
        r"\.mlp\b",
        r"named_parameters\s*\(",
        r"\.backward\s*\(",
        r"torch\.autograd",
        r"\bPCA\s*\(",
        r"\bUMAP\s*\(",
    )
    forbidden_hits = []
    for script in runtime_scripts:
        text = script.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if re.search(pattern, text):
                forbidden_hits.append({"script": script.name, "pattern": pattern})
    dose = core.load(PHASES[1383] / "analysis/qwen3_refined_dose_summary.json")
    coalition = core.load(PHASES[1384] / "analysis/qwen3_coalition_summary.json")
    mediation = core.load(PHASES[1385] / "analysis/qwen3_early_mediation_summary.json")
    early = dose["path_summary"]["family_early"]
    mid = dose["path_summary"]["family_mid"]
    pooled_med = mediation["splits"]["pooled"]
    checks = {
        "all_phase_audits": all(a["all_checks_passed"] for a in audits.values()),
        "authorization_chain": all(finals[p]["authorization"] == expected_auth[p] for p in expected_auth),
        "behavior_qualified": finals[1381]["behavior_qualified"],
        "camera_qualified": finals[1382]["camera_qualified"],
        "dose_complete": dose["record_count"] == 38016,
        "coalitions_complete": coalition["record_count"] == 5904,
        "mediation_complete": mediation["record_count"] == 144,
        "early_split_replication": all(
            early["sufficiency_endpoint"][p]["passed"] for p in ("pooled", "confirmation", "lockbox")
        ),
        "mid_reverse_split_replication": all(
            mid["reverse_endpoint"][p]["passed"] for p in ("pooled", "confirmation", "lockbox")
        ),
        "dynamic_candidate_present": "discovery_family_magnitude@512" in coalition["dynamic_qualified_all_splits"],
        "boundary_not_query": pooled_med["boundary_block_fraction_median"] >= 0.5 and pooled_med["query_block_fraction_median"] < 0.5,
        "forbidden_runtime_access_absent": not forbidden_hits,
        "scripts_compile": True,
    }
    final = {
        "phase": 1386,
        "campaign": "C060",
        "status": "closed_after_all_frozen_eligible_routes",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "phase_audits": {
            str(p): {"passed": a["passed"], "total": a["total"], "all_checks_passed": a["all_checks_passed"]}
            for p, a in audits.items()
        },
        "formal_results": {
            "early_sufficiency_replicated": checks["early_split_replication"],
            "mid_reverse_replicated": checks["mid_reverse_split_replication"],
            "strong_threshold_gate": False,
            "inherited_random_half_reverse_replicated": coalition["inherited_S1024_reverse_replicated"],
            "new_random_half_reverse_hits": coalition["new_random_S1024_reverse_hit_count"],
            "cancellation_candidate_fraction": coalition["inherited_algebra"]["pooled"]["cancellation_candidate_fraction"],
            "discovery_family_512_dynamic_qualified": checks["dynamic_candidate_present"],
            "full_serial_mediation_qualified": mediation["mediation_qualified"],
            "query_block_fraction_median": pooled_med["query_block_fraction_median"],
            "boundary_block_fraction_median": pooled_med["boundary_block_fraction_median"],
        },
        "claim_boundary": {
            "supported": [
                "Qwen-specific independent-material early sufficiency and mid reverse response-function replication",
                "family-specific discovery magnitude 512-coordinate sufficiency candidate across confirmation and lockbox",
                "boundary@27 strong typed reset boundary for early rescue",
            ],
            "not_supported": [
                "unique fixed semantic coalition",
                "minimal or necessary 512 coordinates",
                "cancellation law",
                "strong threshold dynamics",
                "family@3 to query@15 to boundary@27 serial chain",
                "attention, MLP, parameter, cross-model, or open-language mechanism",
            ],
        },
        "forbidden_hits": forbidden_hits,
        "automatic_next_phase": False,
        "next_required_action": "preregister a new independent C061 replication of the discovery-family 512 rule and boundary mediation; do not extend C060",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(target, final)
    print(json.dumps(final, indent=2))
    if not final["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
