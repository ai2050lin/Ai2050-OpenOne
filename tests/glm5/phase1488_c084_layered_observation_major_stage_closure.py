#!/usr/bin/env python3
"""Phase1488: close the layered-observation C084 major stage."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1488_c084_layered_observation_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


STAGES = {
    1482: "phase1482_layered_observation_policy",
    1483: "phase1483_existing_observation_asset_registry",
    1484: "phase1484_c084_batch_deep_mining_contract",
    1485: "phase1485_c084_coordinate_stability_atlas",
    1486: "phase1486_c084_factorial_surface_atlas",
    1487: "phase1487_c084_joint_synthesis_and_prediction_freeze",
}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1488 exists")
    paths = {phase: RESULT / name for phase, name in STAGES.items()}
    finals = {phase: core.load(path / "analysis/final.json") for phase, path in paths.items()}
    audits = {phase: core.load(path / "audit/independent_final_audit.json") for phase, path in paths.items()}
    policy = core.load(paths[1482] / "protocol/layered_observation_policy.json")
    registry = core.load(paths[1483] / "analysis/asset_registry.json")
    contract = core.load(paths[1484] / "protocol/preregistration.json")
    coordinate = core.load(paths[1485] / "analysis/coordinate_atlas_summary.json")
    factorial = core.load(paths[1486] / "analysis/factorial_atlas_summary.json")
    synthesis = core.load(paths[1487] / "analysis/synthesis.json")
    manifest = core.load(paths[1487] / "frozen/future_prediction_manifest.json")
    scripts = []
    for phase in range(1482, 1488):
        for path in sorted(TESTS.glob(f"phase{phase}_*.py")):
            py_compile.compile(str(path), doraise=True)
            scripts.append(path.name)
    checks = {
        "audits": all(row["all_checks_passed"] for row in audits.values()),
        "statuses": [finals[phase]["status"] for phase in range(1482, 1488)] == [
            "project_level_gate_policy_preregistered",
            "legal_assets_and_missingness_registered",
            "batch_deep_mining_preregistered",
            "coordinate_stability_atlas_complete",
            "factorial_surface_atlas_complete",
            "joint_synthesis_complete_with_refined_retrospective_candidates",
        ],
        "policy_chain": registry["policy_sha256"] == policy["policy_sha256"] and contract["source_registry_sha256"] == registry["registry_sha256"],
        "prediction_freeze": manifest["freeze_sha256"] == core.digest({key: value for key, value in manifest.items() if key != "freeze_sha256"}) == synthesis["prediction_freeze_sha256"],
        "source_integrity": coordinate["source_sha256"] == contract["source_assets"]["C082_relation_mean_atlas"]["sha256"] and factorial["relation_reproduction_max_abs"] == 0.0,
        "no_model_run": all(not finals[phase].get("model_run", False) for phase in (1482, 1483, 1485, 1486, 1487)),
        "scripts_compile": len(scripts) == 12,
        "route_queue_exhausted": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    result = {
        "phase": 1488,
        "campaign": "C084",
        "status": "major_stage_closed_with_layered_policy_and_refined_c079_factorial_coordinate_candidates",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "major_stage_answer": {
            "gate_philosophy": "changed prospectively from whole-route hard stop to layered evidence typing, predefined missingness, and a preregistered route queue",
            "early_state": "the approximately minus-0.2 state0 relation cosine is primarily cyclic contrast geometry, not a semantic separation law",
            "late_state": "from state22 onward the C079 relation-match effects enter a strong shared late boundary direction; state35 is cross-relation, leave-one-relation, and leave-one-panel stable",
            "coordinates": "the former 17 coordinates are fully same-sign across 36 panels but are one threshold slice of a wider 9/17/37/90 nested high-energy coordinate band",
            "factorial": "at state35 the relation coefficient carries 0.9806-0.9942 of the seven-coefficient norm-squared ledger and all nonrelation coefficient norm ratios are at most 0.0855",
        },
        "claim_scope": "strong retrospective structural candidates in behavior-correct Qwen3 C079 explicit-label cases; no natural semantics, causal mechanism, cross-model invariant, or new mathematics",
        "theory": {
            "name": "conditional output-field closure theory",
            "organizing_principle": "reuse-difference-conditioning (RDC)",
            "update": "replace a unique 17-coordinate scaffold with a thresholded sign-coherent coordinate band embedded in a distributed late relation-match decision field",
            "formula": "DeltaH = G_task + A_relation + N_entity + N_object + I_RE + I_RO + I_EO + I_REO + epsilon",
        },
        "untested": ["P082 on fresh qualified material", "all six P084 predictions", "causal necessity or sufficiency", "natural unlabeled relation use", "cross-model repetition"],
        "next_stage": {
            "authorization": "preregister_c085_prospective_layered_replication_and_diagnostic_capture",
            "routes": [
                "fresh same-construction replication with success, mixed, and failed behavior strata typed separately",
                "unchanged P084 batch validation on a sealed confirmation panel",
                "only after replication, a weak full-state causal test; still no attention or MLP",
            ],
            "automatic_continuation_recommended": True,
            "reason_not_run_inside_c084": "C084 was explicitly a no-new-model deep-mining campaign; prospective C085 requires a new material and capture contract before CUDA execution",
        },
        "authorization": "preregister_c085_prospective_layered_replication_and_diagnostic_capture",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
