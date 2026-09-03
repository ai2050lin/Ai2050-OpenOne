#!/usr/bin/env python3
"""Phase1481: close the complete C080-C083 major stage."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1481_c080_c083_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


STAGES = {
    1469: "phase1469_c080_balanced_interaction_contract",
    1470: "phase1470_c080_explicit_behavior",
    1471: "phase1471_c080_behavior_gate_closure",
    1472: "phase1472_c081_validated_interface_contract",
    1473: "phase1473_c081_behavior",
    1474: "phase1474_c081_route_closure",
    1475: "phase1475_c082_coordinate_atlas_contract",
    1476: "phase1476_c082_coordinate_atlas",
    1477: "phase1477_c082_atlas_synthesis",
    1478: "phase1478_c082_campaign_closure",
    1479: "phase1479_c083_fresh_validation_contract",
    1480: "phase1480_c083_behavior",
}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1481 exists")
    paths = {phase: RESULT / name for phase, name in STAGES.items()}
    audits = {phase: core.load(path / "audit/independent_final_audit.json") for phase, path in paths.items()}
    finals = {phase: core.load(path / "analysis/final.json") for phase, path in paths.items()}
    c080 = core.load(paths[1470] / "analysis/behavior_summary.json")
    c081 = core.load(paths[1473] / "analysis/behavior_summary.json")
    c083 = core.load(paths[1480] / "analysis/behavior_summary.json")
    atlas = core.load(paths[1476] / "analysis/atlas_metadata.json")
    manifest = core.load(paths[1477] / "frozen/future_prediction_manifest.json")
    scripts = []
    for phase in range(1469, 1481):
        for path in sorted(TESTS.glob(f"phase{phase}_*.py")):
            py_compile.compile(str(path), doraise=True)
            scripts.append(path.name)
    checks = {
        "audits": all(value["all_checks_passed"] for value in audits.values()),
        "behavior_stops": not c080["behavior_qualified"] and not c081["behavior_qualified"] and not c083["behavior_qualified"],
        "no_new_hidden": not c080["hidden_state_accessed"] and not c081["hidden_state_accessed"] and not c083["hidden_state_accessed"],
        "c082_complete": finals[1478]["status"] == "closed_with_retrospective_lexical_to_common_boundary_convergence_candidate",
        "atlas_hash": atlas["files"]["mean_effect.float32.npy"]["sha256"] == core.sha(paths[1476] / "atlas/mean_effect.float32.npy"),
        "prediction_freeze": manifest["freeze_sha256"] == core.digest({key: value for key, value in manifest.items() if key != "freeze_sha256"}),
        "c083_untested": finals[1480]["authorization"] == "close_c083_at_behavior_gate",
        "scripts_compile": len(scripts) == 24,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": 1481,
        "campaign": "C080-C083",
        "status": "major_stage_closed_with_one_retrospective_structure_candidate_and_unresolved_fresh_validation",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "answers": {
            "balanced_equality_interaction": "not tested because C080 and its one-shot C081 rescue failed preregistered cross-surface positive behavior gates",
            "coordinate_atlas": "C079 relation-specific token differences retrospectively converge to a shared distributed state35 boundary response with a 17-coordinate recurrent top-1-percent scaffold",
            "fresh_validation": "not tested because C083 narrowly failed its preregistered behavior and breadth gates",
        },
        "latest_mechanism_candidate": "identity-specific lexical carriers are conditionally transformed into a reused late decision field; a small coordinate scaffold recurs but most energy remains distributed",
        "theory_boundary": "consistent with RDC/conditional output-field closure, but not a closed language theory and not evidence for natural semantics or causal use",
        "mathematics": "existing causal order, vector difference, cosine, and coordinate energy are sufficient; no new mathematical theory is licensed",
        "next_legal_stage": {
            "requirement": "prospectively obtain a behavior-qualified fresh object with margin, without further prompt-family search on the closed explicit-label route",
            "then": "test the unchanged P082 manifest; only a double-holdout pass may authorize weak full-vector causal work",
            "automatic_model_run": False,
        },
        "authorization": "no_automatic_continuation_until_a_new_behavior_qualified_object_or_project_level_gate_policy_is_preregistered",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
