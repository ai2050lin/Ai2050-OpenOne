#!/usr/bin/env python3
"""End-to-end integrity and claim-scope audit for Phase1008."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1008_global_response_atlas_protocol import (
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
)


MODELS = ("qwen3", "glm4", "deepseek7b")
REFINED_MODELS = ("qwen3", "glm4")


def assert_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name}: non-finite values")


def range_for(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values = [float(row[field]) for row in rows]
    return [min(values), max(values)]


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    protocol_digest = protocol["preregistration_digest"]
    global_summary = read_json(OUT_ROOT / "final" / "summary.json")
    if global_summary["protocol_digest"] != protocol_digest:
        raise RuntimeError("global summary protocol drift")
    protocol_audit: dict[str, Any] = {}
    scan_audit: dict[str, Any] = {}
    behavior_audit: dict[str, Any] = {}
    for model_name in MODELS:
        model_protocol = read_json(
            OUT_ROOT / "protocol" / model_name / "summary.json"
        )
        cases = read_jsonl(
            OUT_ROOT / "protocol" / model_name / "cases.jsonl"
        )
        units = read_jsonl(
            OUT_ROOT / "protocol" / model_name / "units.jsonl"
        )
        case_by_id = {row["record_id"]: row for row in cases}
        if len(cases) != 336 or len(units) != 48:
            raise RuntimeError(f"{model_name}: protocol cardinality drift")
        bq_same_answer = 0
        b_changes_answer = 0
        q_changes_answer = 0
        for unit in units:
            unit_id = unit["unit_id"]
            base = case_by_id[f"{unit_id}.base"]
            b = case_by_id[f"{unit_id}.B"]
            q = case_by_id[f"{unit_id}.Q"]
            bq = case_by_id[f"{unit_id}.BQ"]
            bq_same_answer += bq["gold"] == base["gold"]
            b_changes_answer += b["gold"] != base["gold"]
            q_changes_answer += q["gold"] != base["gold"]
        if (bq_same_answer, b_changes_answer, q_changes_answer) != (
            48, 48, 48
        ):
            raise RuntimeError(f"{model_name}: factorial answer drift")
        protocol_audit[model_name] = {
            "case_count": len(cases),
            "unit_count": len(units),
            "split_unit_counts": model_protocol["split_unit_counts"],
            "bq_same_answer_count": bq_same_answer,
            "b_changes_answer_count": b_changes_answer,
            "q_changes_answer_count": q_changes_answer,
            "single_token_name_count": model_protocol[
                "tokenizer_audit"
            ]["single_token_name_count"],
            "single_token_code_word_count": model_protocol[
                "tokenizer_audit"
            ]["single_token_code_word_count"],
        }

        behavior = read_json(
            OUT_ROOT / "behavior" / model_name / "summary.json"
        )
        if behavior["protocol_digest"] != protocol_digest:
            raise RuntimeError(f"{model_name}: behavior digest drift")
        behavior_audit[model_name] = {
            "overall_semantic_case_rate": behavior[
                "overall_semantic_case_rate"
            ],
            "overall_rollout_case_rate": behavior[
                "overall_rollout_case_rate"
            ],
            "operation_summary": behavior["operation_summary"],
        }

        scan = read_json(OUT_ROOT / "scan" / model_name / "summary.json")
        if scan["protocol_digest"] != protocol_digest:
            raise RuntimeError(f"{model_name}: scan digest drift")
        response = np.load(
            OUT_ROOT / "scan" / model_name / "response_scalars.npz"
        )
        direction = np.load(
            OUT_ROOT / "scan" / model_name / "direction_consistency.npz"
        )
        assert_finite(
            f"{model_name}/raw_magnitude", response["raw_magnitude"]
        )
        assert_finite(
            f"{model_name}/normalized_magnitude",
            response["normalized_magnitude"],
        )
        valid_direction = direction["direction_count"] >= 2
        assert_finite(
            f"{model_name}/valid_direction_consistency",
            direction["direction_consistency"][valid_direction],
        )
        identity_index = list(scan["operations"]).index("I")
        identity_maximum = float(np.max(
            response["raw_magnitude"][:, identity_index, :]
        ))
        if identity_maximum != 0.0:
            raise RuntimeError(f"{model_name}: identity floor nonzero")
        if response["raw_magnitude"].size != scan[
            "scalar_measurement_count"
        ]:
            raise RuntimeError(f"{model_name}: scalar count drift")
        scan_audit[model_name] = {
            "event_count": scan["event_count"],
            "scalar_measurement_count": scan[
                "scalar_measurement_count"
            ],
            "identity_maximum": identity_maximum,
            "raw_hidden_tensors_persisted": scan[
                "raw_hidden_tensors_persisted"
            ],
            "valid_direction_cell_count": int(valid_direction.sum()),
        }

    refinement_protocol = read_json(
        OUT_ROOT / "refinement" / "protocol.json"
    )
    refinement_digest = refinement_protocol["preregistration_digest"]
    if refinement_protocol["protocol_revision"] != 4:
        raise RuntimeError("refinement protocol is not revision 4")
    refinement_summary = read_json(
        OUT_ROOT / "refinement_final" / "summary.json"
    )
    if refinement_summary["protocol_digest"] != refinement_digest:
        raise RuntimeError("refinement final digest drift")
    refinement_audit: dict[str, Any] = {}
    for model_name in REFINED_MODELS:
        source = OUT_ROOT / "refinement_scan" / model_name
        scan_summary = read_json(source / "summary.json")
        if scan_summary["refinement_protocol_digest"] != refinement_digest:
            raise RuntimeError(f"{model_name}: stale refinement data")
        if not scan_summary["weight_reconstruction_audit"]["all_pass"]:
            raise RuntimeError(f"{model_name}: reconstruction failed")
        if not scan_summary["dual_weight_rank_audit"]["all_pass"]:
            raise RuntimeError(f"{model_name}: rank audit failed")
        head = np.load(source / "head_observations.npz")
        neuron = np.load(source / "neuron_observations.npz")
        assert_finite(
            f"{model_name}/head_write", head["write_magnitude"]
        )
        assert_finite(
            f"{model_name}/neuron_write", neuron["write_magnitude"]
        )
        if head["write_magnitude"].size != scan_summary[
            "head_observation_count"
        ]:
            raise RuntimeError(f"{model_name}: head count drift")
        if neuron["write_magnitude"].size != scan_summary[
            "neuron_observation_count"
        ]:
            raise RuntimeError(f"{model_name}: neuron count drift")
        final_model = read_json(
            OUT_ROOT / "refinement_final" / model_name / "summary.json"
        )
        head_rows = read_jsonl(
            OUT_ROOT / "refinement_final" / model_name
            / "head_candidates.jsonl"
        )
        neuron_rows = read_jsonl(
            OUT_ROOT / "refinement_final" / model_name
            / "neuron_candidates.jsonl"
        )
        population_rows = read_jsonl(
            OUT_ROOT / "refinement_final" / model_name
            / "population_summaries.jsonl"
        )
        overlap_rows = read_jsonl(
            OUT_ROOT / "refinement_final" / model_name
            / "operation_overlaps.jsonl"
        )
        if len(head_rows) != final_model["head_candidate_count"]:
            raise RuntimeError(f"{model_name}: head candidate drift")
        if len(neuron_rows) != final_model["neuron_candidate_count"]:
            raise RuntimeError(f"{model_name}: neuron candidate drift")
        head_unique = {
            (
                row["stage"],
                row["role"],
                row["layer"],
                row["head_index"],
            )
            for row in head_rows
        }
        neuron_unique = {
            (
                row["stage"],
                row["role"],
                row["layer"],
                row["neuron_index"],
            )
            for row in neuron_rows
        }
        head_population = [
            row for row in population_rows
            if row["component"] == "attention_head"
        ]
        neuron_population = [
            row for row in population_rows
            if row["component"] == "mlp_neuron"
        ]
        head_overlap = [
            row["jaccard"] for row in overlap_rows
            if row["component"] == "attention_head"
        ]
        neuron_overlap = [
            row["jaccard"] for row in overlap_rows
            if row["component"] == "mlp_neuron"
        ]
        refinement_audit[model_name] = {
            "head_observation_count": scan_summary[
                "head_observation_count"
            ],
            "neuron_observation_count": scan_summary[
                "neuron_observation_count"
            ],
            "head_candidate_rows": len(head_rows),
            "head_unique_physical_components": len(head_unique),
            "neuron_candidate_rows": len(neuron_rows),
            "neuron_unique_physical_components": len(neuron_unique),
            "candidate_counts_by_operation": final_model[
                "candidate_counts_by_operation"
            ],
            "head_participation_fraction_range": range_for(
                head_population, "participation_fraction_median"
            ),
            "neuron_participation_fraction_range": range_for(
                neuron_population, "participation_fraction_median"
            ),
            "neuron_top_1pct_mass_range": range_for(
                neuron_population, "top_1pct_mass_median"
            ),
            "mean_head_operation_jaccard": float(
                np.mean(head_overlap)
            ),
            "mean_neuron_operation_jaccard": float(
                np.mean(neuron_overlap)
            ),
            "maximum_runtime_reconstruction_error": scan_summary[
                "weight_reconstruction_audit"
            ]["maximum_runtime_relative_error"],
            "minimum_dual_weight_head_jaccard": scan_summary[
                "dual_weight_rank_audit"
            ]["attention_minimum_jaccard"],
            "minimum_dual_weight_neuron_jaccard": scan_summary[
                "dual_weight_rank_audit"
            ]["mlp_minimum_jaccard"],
            "raw_head_vectors_persisted": scan_summary[
                "raw_head_vectors_persisted"
            ],
            "raw_neuron_activations_persisted": scan_summary[
                "raw_neuron_activations_persisted"
            ],
        }

    causal_audit: dict[str, Any] = {}
    for model_name in REFINED_MODELS:
        result = read_json(
            OUT_ROOT / "causal_sample" / model_name / "summary.json"
        )
        if result["selection_used_confirmation_data"]:
            raise RuntimeError(f"{model_name}: causal selection leakage")
        if not result["no_op_audit_pass"]:
            raise RuntimeError(f"{model_name}: causal no-op failed")
        causal_audit[model_name] = {
            "unit_operation_count": result["unit_operation_count"],
            "operation_summaries": result["operation_summaries"],
            "localized_directional_contribution_count": sum(
                row["localized_directional_contribution"]
                for row in result["operation_summaries"]
            ),
        }

    total_refinement_observations = sum(
        refinement_audit[model]["head_observation_count"]
        + refinement_audit[model]["neuron_observation_count"]
        for model in REFINED_MODELS
    )
    output = {
        "schema_version": "phase1008_end_to_end_audit.v1",
        "phase": PHASE,
        "all_integrity_gates_pass": True,
        "protocol_digest": protocol_digest,
        "refinement_protocol_revision": 4,
        "refinement_protocol_digest": refinement_digest,
        "protocol": protocol_audit,
        "behavior": behavior_audit,
        "global_scan": scan_audit,
        "global_atlas": {
            "total_scalar_internal_measurements": global_summary[
                "total_scalar_internal_measurements"
            ],
            "refinement_eligible_by_model": {
                model: global_summary["model_summaries"][model][
                    "refinement_eligible_count"
                ]
                for model in MODELS
            },
            "qwen_glm_core_cross_motif_count": global_summary[
                "qwen_glm_core_cross_motif_count"
            ],
            "co_response_edge_count": global_summary[
                "co_response_edge_count"
            ],
            "edge_semantics": global_summary["edge_semantics"],
        },
        "refinement": refinement_audit,
        "total_refinement_scalar_observations": (
            total_refinement_observations
        ),
        "causal_sample": causal_audit,
        "claim_audit": {
            "stable_repeated_internal_structure_observed": True,
            "distributed_reusable_decision_field_supported": True,
            "cross_model_physical_coordinate_alignment_supported": False,
            "qwen_local_head_causal_contribution_supported": False,
            "glm_local_head_causal_contribution_supported": True,
            "complete_transport_path_supported": False,
            "mechanism_closure_supported": False,
            "formula_fitted": False,
        },
        "hard_limits": [
            "single narrow synthetic two-entity binding family",
            "only four templates and 48 worlds per model",
            "semantic qualification varies strongly by template",
            "DeepSeek7B rollout behavior is too weak for fine localization",
            "B and Q share the same changed answer in this factorial design",
            "co-response edges are not transport edges",
            "MLP scalar write magnitudes omit write direction and cancellation",
            "one layer and one token role were sampled causally",
            "8bit runtime introduces model-dependent reconstruction error",
        ],
        "automatic_next_action": (
            "phase1009_expand_pattern_families_and_dynamic_role_atlas;"
            "preserve_local_causal_sampling_as_annotation_only"
        ),
    }
    write_json(OUT_ROOT / "audit" / "summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
