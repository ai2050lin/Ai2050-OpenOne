#!/usr/bin/env python3
"""Phase1587: close C102 with typed evidence and a narrow next authorization."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    audit_paths = [
        "audit/independent_contract_audit.json",
        "audit/independent_c101_discovery_audit.json",
        "audit/independent_full_field_capture_audit.json",
        "audit/independent_coefficient_audit.json",
        "audit/independent_response_discovery_audit.json",
        "audit/independent_confirmation_audit.json",
        "audit/independent_staged_barcode_final_audit.json",
        "audit/independent_coordinate_intervention_audit.json",
        "audit/independent_heatmap_export_audit.json",
        "audit/independent_client_integration_audit.json",
    ]
    audits = [core.load(OUT / path) for path in audit_paths]
    capture = core.load(OUT / "analysis/qwen_full_field_capture_summary.json")
    selection = core.load(OUT / "protocol/response_discovery_selection.json")
    confirmation = core.rows(OUT / "analysis/confirmation_barcode_results.jsonl")
    lockbox = core.rows(OUT / "analysis/lockbox_barcode_results.jsonl")
    formation = [row for row in core.rows(OUT / "analysis/formation_trajectory_validation.jsonl") if row["partition"] == "lockbox"]
    intervention = core.load(OUT / "analysis/coordinate_coalition_intervention_final.json")
    intervention_rows = core.rows(OUT / "analysis/coordinate_coalition_intervention_summary.jsonl")
    final = {
        "phase": 1587,
        "campaign": "C102",
        "status": "major_stage_complete_with_distributed_barcode_replication_and_semantic_causal_boundary",
        "checks": {
            "all_stage_audits": all(audit["all_checks_passed"] for audit in audits),
            "materials": capture["valid_tokens"] == 179416,
            "numeric": all(capture["checks"].values()),
            "barcode": len(confirmation) == len(lockbox) == 8 and all(row["primary"]["beats_q99"] for row in [*confirmation, *lockbox]),
            "intervention": intervention["important_count"] == 0,
            "heatmap": core.load(OUT / "analysis/client_integration.json")["all_checks_passed"],
        },
        "materials": {"units": 72, "cases": 1152, "families": 8, "partitions": 3, "factorial_cells_per_unit": 16, "human_naturalness": "M_HUMAN_NATURALNESS"},
        "field": {"shape": capture["shape"], "valid_tokens": capture["valid_tokens"], "bytes": capture["bytes"], "archive": "exact BF16 bits", "activation_coordinates_not_parameters": True},
        "behavior": capture["behavior"],
        "barcode": {
            "three_stage_passed": 8,
            "total": 8,
            "selected_k_counts": {str(k): sum(row["k"] == k for row in selection["selection"].values()) for k in sorted({row["k"] for row in selection["selection"].values()})},
            "confirmation_cosine_range": [min(row["primary"]["observed_cosine"] for row in confirmation), max(row["primary"]["observed_cosine"] for row in confirmation)],
            "lockbox_cosine_range": [min(row["primary"]["observed_cosine"] for row in lockbox), max(row["primary"]["observed_cosine"] for row in lockbox)],
            "confirmation_primary_specific": sum(row["primary_specific_over_controls"] for row in confirmation),
            "lockbox_primary_specific": sum(row["primary_specific_over_controls"] for row in lockbox),
            "lockbox_sign_agreement_range": [min(row["primary"]["sign_agreement"] for row in lockbox), max(row["primary"]["sign_agreement"] for row in lockbox)],
            "formation_norm_cosine_range": [min(row["source_fresh_norm_cosine"] for row in formation), max(row["source_fresh_norm_cosine"] for row in formation)],
            "fresh_peak_states": sorted({row["fresh_peak_state"] for row in formation}),
        },
        "intervention": {
            "pairs": intervention["pairs"],
            "families_passing_both_partitions": intervention["important_count"],
            "total_families": 8,
            "positive_correct_cells": sum(row["correct_positive"] for row in intervention_rows),
            "controlled_cells": sum(row["correct_beats_informative_controls"] for row in intervention_rows),
            "total_family_partition_cells": len(intervention_rows),
        },
        "adjudication": {
            "retained": [
                "all-coordinate preservation plus controlled factorial effects is a valid observation-first method",
                "activation-coordinate effects and finite-difference layer trajectories are the correct objects; raw activation sign co-occurrence is not collaboration",
                "eight distributed late-boundary task-response barcodes repeated prospectively on lexically independent materials",
                "formation norm trajectories repeated and peaked at state35",
            ],
            "corrected": [
                "activation coordinates are not individual model parameters",
                "the result does not establish sparse semantic neurons: three families selected K=1024 and five required K=2560",
                "high predictive similarity is not semantic causal sufficiency: controlled intervention closed for 0/8 families",
                "chance behavior and standard/reversed code asymmetry prohibit a natural semantic understanding claim",
                "the repeated late trajectory may be a shared output-computation skeleton rather than relation-family identity",
                "no manifold, fiber, group, functor or new-mathematics claim is licensed",
            ],
        },
        "puzzles": {
            "K273": "Eight-family distributed late task-response activation barcodes repeat across new lexical materials, partitions and full-coordinate permutation nulls.",
            "K274": "Predictive barcode identity separates from semantic controllability: frozen natural-donor coordinate coalitions fail controlled two-partition closure in 0/8 families.",
        },
        "theory_update": "RDC retains a conditional response-field view, but must type late relation contrasts as task/output response mixtures until code-invariant earlier-role structure and controlled causal use are both shown.",
        "next_authorization": {
            "campaign": "C103",
            "scope": "existing-data-only observation of code-residualized primary effects across every registered role and state",
            "no_new_model_run": True,
            "stop": "freeze any candidate only after descriptive role-state atlas; use fresh data later for validation",
        },
        "files": {"heatmap": "frontend/public/vis_data/research_kernel/c102_coordinate_barcode_heatmap.json", "raw": "tests/glm5/result/phase1581_c102_typed_relation_coordinate_campaign/raw/qwen3_all_token_state_coordinate_field.uint16.npy"},
        "authorization": "append_phase1587_c102_memo_then_run_c103_existing_data_observation",
    }
    final["passed"] = sum(final["checks"].values())
    final["total"] = len(final["checks"])
    final["all_checks_passed"] = all(final["checks"].values())
    if not final["all_checks_passed"] or not all(math.isfinite(value) for value in [*final["barcode"]["confirmation_cosine_range"], *final["barcode"]["lockbox_cosine_range"]]):
        raise RuntimeError(final)
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
