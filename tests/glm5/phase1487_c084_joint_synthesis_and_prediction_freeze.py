#!/usr/bin/env python3
"""Phase1487: synthesize C084 and freeze refined future predictions."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
COORD = RESULT / "phase1485_c084_coordinate_stability_atlas"
FACTOR = RESULT / "phase1486_c084_factorial_surface_atlas"
OUT = RESULT / "phase1487_c084_joint_synthesis_and_prediction_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def compact(values: list[float]) -> dict:
    return {"minimum": float(min(values)), "median": float(np.median(values)), "mean": float(np.mean(values)), "maximum": float(max(values))}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1487 exists")
    coord_final = core.load(COORD / "analysis/final.json")
    coord_audit = core.load(COORD / "audit/independent_final_audit.json")
    coord = core.load(COORD / "analysis/coordinate_atlas_summary.json")
    factor_final = core.load(FACTOR / "analysis/final.json")
    factor_audit = core.load(FACTOR / "audit/independent_final_audit.json")
    factor = core.load(FACTOR / "analysis/factorial_atlas_summary.json")
    coordinate_rows = core.rows(COORD / "analysis/layer_coordinate_metrics.jsonl")
    factorial_rows = core.rows(FACTOR / "analysis/layer_factorial_metrics.jsonl")
    surface_rows = core.rows(FACTOR / "analysis/cross_surface_metrics.jsonl")
    checks = {
        "parents": coord_final["status"] == "coordinate_stability_atlas_complete" and factor_final["status"] == "factorial_surface_atlas_complete",
        "audits": coord_audit["all_checks_passed"] and factor_audit["all_checks_passed"],
        "outputs": all(coord["output_checks"].values()) and all(factor["output_checks"].values()),
        "relation_reproduced": factor["relation_reproduction_max_abs"] == 0.0,
        "no_model_run": not coord_final["model_run"] and not factor_final["model_run"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    boundary_rows = [row for row in coordinate_rows if row["role"] == "boundary"]
    stable = [row for row in boundary_rows if row["relation_pairwise_cosine"]["minimum"] >= 0.90 and row["leave_one_relation_common_cosine"]["minimum"] >= 0.95]
    first_stable_state = min(row["state"] for row in stable)
    state35_factor = [row for row in factorial_rows if row["state"] == 35 and row["role"] == "boundary"]
    relation_energy = [row["effects"]["relation"]["factorial_coefficient_energy_fraction"] for row in state35_factor]
    nuisance_names = ["entity", "object", "relation_entity", "relation_object", "entity_object", "relation_entity_object"]
    max_nuisance = [max(row["effects"][name]["factorial_coefficient_norm_ratio_to_relation"] for name in nuisance_names) for row in state35_factor]
    relation_surface = [row for row in surface_rows if row["effect"] == "relation" and row["state"] == 35 and row["role"] == "boundary"]
    early = coord["state0_cyclic_geometry"]["global|query_label"]
    boundary = coord["boundary_state35"]
    findings = {
        "F084_1_early_geometry_correction": {
            "status": "established_for_C079_contrast_design",
            "result": "state0 mean anti-cosine is explained primarily by the six-way near-zero-centroid cyclic contrast geometry, not identified semantic repulsion",
            "pairwise_mean": early["pairwise_cosine"]["mean"],
            "simplex_reference": early["simplex_reference"],
            "centroid_norm_ratio": early["centroid_norm_over_mean_vector_norm"],
        },
        "F084_2_late_common_direction": {
            "status": "strong_retrospective_single_task_candidate",
            "first_state_meeting_frozen_descriptive_stability": first_stable_state,
            "state35_pairwise": boundary["relation_pairwise_cosine"],
            "state35_leave_one_relation": boundary["leave_one_relation_common_cosine"],
            "state35_leave_one_panel": coord["boundary_panel_loo_cosine"],
        },
        "F084_3_thresholded_coordinate_band": {
            "status": "strong_retrospective_coordinate-band_candidate",
            "threshold_intersection_counts": {str(row["fraction"]): row["intersection_count"] for row in boundary["thresholds"]},
            "threshold_union_counts": {str(row["fraction"]): row["union_count"] for row in boundary["thresholds"]},
            "former17_relation_unanimous": coord["frozen17"]["relation_unanimous_nonzero_count"],
            "former17_all_panel_unanimous": coord["frozen17"]["all_36_panels_unanimous_nonzero_count"],
            "correction": "the 17 coordinates are a one-percent threshold slice of a wider nested high-energy band, not a unique coordinate mechanism",
        },
        "F084_4_factorial_specificity": {
            "status": "strong_retrospective_success-domain_candidate",
            "relation_coefficient_energy_fraction": compact(relation_energy),
            "maximum_nonrelation_coefficient_norm_ratio": compact(max_nuisance),
            "relation_cross_surface_cosine": compact([row["cosine"] for row in relation_surface]),
            "caveat": "coefficient energy is an orthogonal condition-cube ledger, not output variance explained or causal attribution",
        },
    }
    predictions = {
        "phase": 1487,
        "campaign": "C084",
        "source_scope": "fresh material using the same explicit-label 2x2x2 construction, with prospectively typed behavior-success and behavior-failure strata",
        "predictions": [
            {"id": "P084-1", "object": "state0 cyclic contrast control", "gate": "query-label pairwise mean cosine in [-0.23,-0.17] and centroid-norm ratio <=0.05", "claim": "design-geometry control only, not semantics"},
            {"id": "P084-2", "object": "late boundary common relation-match direction", "gate": "state35 minimum pairwise relation cosine >=0.90, minimum leave-one-relation cosine >=0.95, and minimum leave-one-panel cosine >=0.95"},
            {"id": "P084-3", "object": "nested sign-coherent coordinate band", "gate": "state35 all-relation intersections at 0.5/1/2/5 percent contain at least 6/12/25/60 coordinates and every intersected coordinate is same-sign across relation means"},
            {"id": "P084-4", "object": "factorial relation specificity", "gate": "state35 relation coefficient energy fraction >=0.95 for every relation and every nonrelation coefficient norm ratio <=0.15"},
            {"id": "P084-5", "object": "surface robustness", "gate": "state35 relation-effect cross-surface cosine >=0.90 in every relation-split panel"},
            {"id": "P084-6", "object": "late emergence band", "gate": "first boundary state jointly reaching pairwise cosine >=0.90 and leave-one-relation cosine >=0.95 lies in states 20 through 26"},
        ],
        "evidence_rule": "behavior-qualified strata test mechanism candidates; prospectively captured mixed/failed strata test diagnostic divergence only; all current observations are discovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    predictions["freeze_sha256"] = core.digest(predictions)
    synthesis = {
        "phase": 1487,
        "campaign": "C084",
        "checks": checks,
        "findings": findings,
        "adjudication": {
            "retain": "a late, distributed, sign-coherent relation-match decision response is reused across six lexical relation labels in the C079 success domain",
            "downgrade": [
                "state0 anti-cosine is contrast-design geometry rather than evidence of semantic family separation",
                "the former 17-coordinate scaffold is renamed a thresholded slice of a shared coordinate band",
                "the common late field remains confounded with explicit labels, task control, and yes/no output competition",
            ],
            "reject": ["semantic neurons", "coordinate causal anchors", "natural-language relation manifold", "cross-model invariant", "new mathematics"],
        },
        "theory_update": {
            "name": "conditional output-field closure theory",
            "organizing_principle": "reuse-difference-conditioning (RDC)",
            "formula": "DeltaH = G_task + A_relation + N_entity + N_object + I_RE + I_RO + I_EO + I_REO + epsilon",
            "status": "descriptive decomposition only; no orthogonality in neural coordinates and no causal closure claimed",
        },
        "prediction_freeze_sha256": predictions["freeze_sha256"],
        "authorization": "run_phase1488_c084_layered_observation_major_stage_closure",
    }
    core.save(OUT / "frozen/future_prediction_manifest.json", predictions)
    core.save(OUT / "analysis/synthesis.json", synthesis)
    core.save(OUT / "analysis/final.json", {
        "phase": 1487,
        "campaign": "C084",
        "status": "joint_synthesis_complete_with_refined_retrospective_candidates",
        "prediction_freeze_sha256": predictions["freeze_sha256"],
        "model_run": False,
        "authorization": synthesis["authorization"],
    })
    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    main()
