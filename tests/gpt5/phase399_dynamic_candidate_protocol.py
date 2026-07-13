#!/usr/bin/env python3
"""Pre-register Phase399 aggregate dynamic-chain discovery and validation gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"


def main() -> None:
    protocol = {
        "schema_version": "73.6.0",
        "phase_id": "Phase399-DynamicCandidateProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "select_a_computation_ordered_aggregate_event_chain_before_calibration_is_opened",
        "event_classes": {
            "source_state": {
                "prefixes": [
                    "state:source_entity_a:layer_output",
                    "state:source_entity_b:layer_output",
                    "state:source_value_a:layer_output",
                    "state:source_value_b:layer_output",
                    "state:clause_end_0:layer_output",
                    "state:clause_end_1:layer_output",
                ],
                "required_for_chain": False,
            },
            "source_to_query_route": {
                "prefixes": [
                    "route:source_entity_a->query_end:attention_write",
                    "route:source_entity_b->query_end:attention_write",
                    "route:source_value_a->query_end:attention_write",
                    "route:source_value_b->query_end:attention_write",
                    "route:source_structure->query_end:attention_write",
                ],
                "required_for_chain": True,
            },
            "query_integration": {
                "prefixes": [
                    "state:query_end:attention_output",
                    "state:query_end:mlp_output",
                    "state:query_end:layer_output",
                ],
                "required_for_chain": True,
            },
            "terminal_integration": {
                "prefixes": [
                    "route:source_entity_a->first_answer:attention_write",
                    "route:source_entity_b->first_answer:attention_write",
                    "route:source_value_a->first_answer:attention_write",
                    "route:source_value_b->first_answer:attention_write",
                    "route:source_structure->first_answer:attention_write",
                    "route:query_entity->first_answer:attention_write",
                    "route:query_context->first_answer:attention_write",
                    "state:first_answer:attention_output",
                    "state:first_answer:mlp_output",
                    "state:first_answer:layer_output",
                ],
                "required_for_chain": True,
            },
            "completion_continuation": {
                "prefixes": [
                    "state:target_completion:layer_output",
                    "state:post_target:layer_output",
                ],
                "required_for_chain": False,
            },
        },
        "per_group_layer_gate": {
            "roq_min_axis_normalized_norm_min": 0.01,
            "roq_cross_axis_cosine_min": 0.75,
            "roq_to_competing_interaction_min": 1.25,
        },
        "discovery_cell_gate": {
            "group_pass_rate_min": 0.80,
            "median_roq_min_axis_normalized_norm_min": 0.015,
            "median_roq_cross_axis_cosine_min": 0.85,
            "median_roq_to_competing_interaction_min": 1.50,
        },
        "calibration_and_physical_gate": {
            "group_pass_rate_min": 0.80,
            "median_roq_min_axis_normalized_norm_min": 0.012,
            "median_roq_cross_axis_cosine_min": 0.80,
            "median_roq_to_competing_interaction_min": 1.25,
        },
        "chain_gate": {
            "required_classes": [
                "source_to_query_route",
                "query_integration",
                "terminal_integration",
            ],
            "ordered_peak_layers": "source_to_query_route <= query_integration <= terminal_integration",
            "crossmodel_surface_gate": "all_three_models_pass_the_same_required_role_classes",
        },
        "search_denominator": {
            "effects_used": ["RO", "RQ", "OQ", "ROQ"],
            "all_layers_searched": True,
            "all_predeclared_event_ids_searched": True,
            "head_ids_searched": False,
            "channel_ids_searched": False,
            "neuron_ids_searched": False,
        },
        "authorization": {
            "run_discovery_analysis_after_instrument_gate": True,
            "open_calibration_before_candidate_freeze": False,
            "open_physical_before_calibration": False,
            "run_causal_before_physical": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "discovery_chain_is_causal": False,
            "ordered_observational_events_are_a_binding_algorithm": False,
        },
    }
    path = OUT / "phase399_dynamic_candidate_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
