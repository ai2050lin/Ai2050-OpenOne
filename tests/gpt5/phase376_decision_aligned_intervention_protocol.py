#!/usr/bin/env python3
"""Freeze direct decision-aligned activation-swap interventions for Phase376."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PHASE375 = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
OUT = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
INPUTS = {
    "decision_alignment": OUT / "phase376_decision_time_alignment_summary.json",
    "phase375_negative_diagnostic": PHASE375
    / "phase375_discovery/phase375_negative_result_diagnostic.json",
    "collector_cases": PHASE371
    / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl",
    "behavior_qwen3": PHASE371
    / "phase371c_behavior_qualification/private/models/qwen3/phase371c_behavior_rows.jsonl",
    "behavior_glm4": PHASE371
    / "phase371c_behavior_qualification/private/models/glm4/phase371c_behavior_rows.jsonl",
    "behavior_deepseek7b": PHASE371
    / "phase371c_behavior_qualification/private/models/deepseek7b/phase371c_behavior_rows.jsonl",
    "condition_key": PHASE371
    / "phase371c_discovery_mapping/private/phase371c_discovery_condition_key.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "schema_version": "49.1.0",
        "phase_id": "Phase376-InterventionProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "test_direct_content_transfer_at_semantically_aligned_answer_decisions_"
            "without_a_predictive_prefilter"
        ),
        "rationale": {
            "phase375_all_absolute_future_projection_gates_failed": True,
            "phase371_fixed_offsets_crossmodel_semantically_aligned": False,
            "same_coordinate_projection_is_primary_causal_readout": False,
            "direct_natural_boundary_intervention_required": True,
        },
        "frozen_scope": {
            "models": ["qwen3", "glm4", "deepseek7b"],
            "execution_order": ["qwen3", "glm4", "deepseek7b"],
            "mechanisms": ["relation_binding", "entity_recency"],
            "discovery_parallel_groups_per_model": 22,
            "conditions_per_group": 4,
            "case_count": 264,
            "relative_depths": ["early", "middle", "late"],
            "calibration_opened": False,
            "physical_opened": False,
            "new_cases_added": False,
        },
        "semantic_alignment": {
            "decision_event": "context_immediately_before_the_token_completing_target_alias",
            "answer_entry_event": "formatted_prompt_before_any_generated_token",
            "decision_context_tokens": "prompt_plus_natural_generated_prefix_before_decision",
            "target_token": "observed_natural_token_at_decision_step",
            "fixed_token_offset_used": False,
        },
        "transfer_pairs": {
            "primary_treatment": ["A_to_C", "C_to_A"],
            "matched_direct_route_control": ["B_to_D", "D_to_B"],
            "primary_group_gate_requires_both_treatment_directions": True,
        },
        "natural_templates": {
            "residual_current": {
                "component": "residual_output",
                "roles": ["current"],
            },
            "residual_source_query_current": {
                "component": "residual_output",
                "roles": ["source", "query", "current"],
            },
            "attention_mlp_current": {
                "component": "attention_mlp_output",
                "roles": ["current"],
            },
        },
        "batched_conditions": {
            "correct": "decision_event_correct_depth_correct_roles",
            "wrong_depth": "decision_event_cyclic_depth_correct_roles",
            "wrong_role": "decision_event_correct_depth_cyclic_donor_roles",
            "wrong_time": "answer_entry_event_correct_depth_correct_roles",
        },
        "primary_readout": {
            "margin": "donor_target_token_logit_minus_recipient_target_token_logit",
            "transfer_gain": "patched_margin_minus_recipient_baseline_margin",
            "winner_transfer": "donor_target_token_becomes_argmax",
            "full_generation_behavior_claimed": False,
            "trained_probe_used": False,
        },
        "frozen_numeric_gates": {
            "minimum_correct_transfer_gain": 0.10,
            "minimum_gain_over_wrong_depth": 0.05,
            "minimum_gain_over_wrong_role": 0.05,
            "minimum_gain_over_wrong_time_when_distinct": 0.05,
            "minimum_independent_groups_per_model_mechanism_template": 8,
            "both_treatment_directions_required": True,
        },
        "cross_model_gate": {
            "canonical_identity": ["mechanism", "relative_depth", "template"],
            "heterogeneous_level2_requires_glm4": True,
            "level3_requires_all_models": True,
            "same_absolute_layer_required": False,
        },
        "evidence_boundary": {
            "positive_transfer_without_winner_flip": "causal_content_carrier_candidate",
            "winner_flip": "single_token_local_sufficiency_candidate",
            "full_language_mechanism": False,
            "natural_necessity_tested": False,
            "single_neuron_causality_tested": False,
        },
        "stop_rules": {
            "no_model_candidate": "close_current_natural_templates",
            "single_model_only": "record_model_specific_intervention_effect",
            "level2_without_winner_flip": "do_not_open_calibration",
            "level2_with_replicated_winner_flip": "authorize_separate_calibration_protocol",
        },
        "input_hashes": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in INPUTS.items()
        },
        "authorization": {
            "run_all_preregistered_discovery_interventions": True,
            "predictive_candidate_required": False,
            "open_calibration": False,
            "open_physical": False,
            "single_neuron_scan": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "phase376_intervention_protocol.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
