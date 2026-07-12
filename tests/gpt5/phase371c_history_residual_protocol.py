#!/usr/bin/env python3
"""Freeze exact-vector history residual gates for t1 provisional routes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
MAPPING = PHASE371 / "phase371c_discovery_mapping/phase371c_discovery_mapping_summary.json"
OUT = PHASE371 / "phase371c_history_residual_protocol.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "47.21.0",
        "phase_id": "Phase371C-History",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_whether_t1_route_difference_is_a_history_sufficient_direction_for_t2_adjacent_output_difference",
        "authorization_basis": {
            "provisional_model_candidate_count": mapping["denominator"]["provisional_model_candidate_count"],
            "full_candidate_count_before_history": mapping["results"]["full_candidate_language_path_count"],
            "mapping_hash": sha256_file(MAPPING),
        },
        "time_contract": {
            "past": 0,
            "current": 1,
            "future": 2,
            "t0_candidates_history_eligible": False,
            "t1_candidates_history_eligible": True,
            "t2_candidates_history_eligible": False,
        },
        "basic_geometry": {
            "current_error": "norm(y_future-proj_span(x_current)(y_future))/norm(y_future)",
            "past_error": "norm(y_future-proj_span(x_past)(y_future))/norm(y_future)",
            "history_error": "norm(y_future-proj_span(x_current,x_past)(y_future))/norm(y_future)",
            "history_gain": "current_error-history_error",
            "fitted_regression_or_coordinate_rotation": False,
        },
        "lexical_pair_gate": {
            "current_error_must_be_strictly_less_than_past_error": True,
            "history_gain_max": 0.01,
            "A_B_and_C_D_both_required": True,
            "zero_future_difference_passes": False,
        },
        "replication_gate": {
            "minimum_independent_groups_per_model_mechanism_route": 8,
            "heterogeneous_level2_requires_glm4": True,
            "same_canonical_depth_role_route_required": True,
        },
        "claim_boundary": {
            "history_projection_is_a_linear_local_diagnostic_not_state_sufficiency_proof": True,
            "causal_same_graph_intervention_replay_completed": False,
            "history_pass_can_open_calibration": False,
            "language_path_claim": False,
        },
        "authorization": {
            "implement_and_hash_exact_history_evaluator": True,
            "open_calibration": False,
            "open_physical": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
