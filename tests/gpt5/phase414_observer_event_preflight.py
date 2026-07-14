#!/usr/bin/env python3
"""Audit natural replay identity, observer-indexed traces, and event panels.

Phase413 separated native terminal probabilities from intermediate diagnostic
readouts.  This model-free stage adds the next necessary correction: replaying
the unchanged, complete natural state from any checkpoint must recover the
same terminal kernel.  Such replay validates instrumentation; it cannot form a
layer-wise candidate-narrowing curve.

The stage also generalizes candidate panels from equal-length sequences to
pairwise-disjoint prefix events, freezes semantic event alignment across
different tokenizations, and qualifies the supplied 96-item directory as a
mixed evidence catalog rather than a mechanism-completion denominator.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase409_dynamic_response_protocol import MODELS  # noqa: E402
from phase411_finite_operation_preflight import (  # noqa: E402
    canonical_json,
    read_json,
    write_json,
    write_jsonl,
)


OUT = ROOT / "tests/gpt5/result/phase414_observer_event_preflight"
P413 = ROOT / "tests/gpt5/result/phase413_prediction_kernel_preflight"
SCHEMA_VERSION = "88.0.0"
PHASE_ID = "Phase414-ObserverIndexedEventPreflight"

FiniteState = tuple[int, int, int, int, int]
Layer = Callable[[FiniteState], FiniteState]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any, length: int = 24) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def softmax(logits: Iterable[int | float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in logits)
    offset = max(values)
    exponentials = tuple(math.exp(value - offset) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def probability_signature(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in values)


def initial_state(content: int, role: int, history: int) -> FiniteState:
    return (
        content + 1,
        2 * content + 1,
        role,
        history,
        content - role + history,
    )


def layer_1(state: FiniteState) -> FiniteState:
    u, v, role, history, cache = state
    return u + role, v + history, role, history, cache + u


def layer_2(state: FiniteState) -> FiniteState:
    u, v, role, history, cache = state
    return u + v + cache, v + role - history, role, history, cache + v


def layer_3(state: FiniteState) -> FiniteState:
    u, v, role, history, cache = state
    return 2 * u - v + history, v + cache + role, role, history, cache + role + history


def layer_4(state: FiniteState) -> FiniteState:
    u, v, role, history, cache = state
    return u + cache, v + u + role, role, history, cache + v + history


LAYERS: tuple[Layer, ...] = (layer_1, layer_2, layer_3, layer_4)


def terminal_logits(state: FiniteState) -> tuple[int, int, int]:
    u, v, role, history, cache = state
    return u + role, v + history, u - v + cache


def continue_state(state: FiniteState, completed_layer_count: int) -> FiniteState:
    result = state
    for layer in LAYERS[completed_layer_count:]:
        result = layer(result)
    return result


def natural_states(content: int, role: int, history: int) -> tuple[FiniteState, ...]:
    rows = [initial_state(content, role, history)]
    for layer in LAYERS:
        rows.append(layer(rows[-1]))
    return tuple(rows)


def natural_replay_audit(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[FiniteState, ...]]]:
    cases: dict[str, tuple[FiniteState, ...]] = {}
    rows: list[dict[str, Any]] = []
    for content in range(3):
        for role in range(2):
            for history in range(2):
                case_id = f"content_{content}__role_{role}__history_{history}"
                checkpoints = natural_states(content, role, history)
                cases[case_id] = checkpoints
                direct_terminal = checkpoints[-1]
                direct_logits = terminal_logits(direct_terminal)
                direct_kernel = probability_signature(softmax(direct_logits))
                for completed_layer_count, checkpoint in enumerate(checkpoints):
                    replay_terminal = continue_state(checkpoint, completed_layer_count)
                    replay_logits = terminal_logits(replay_terminal)
                    replay_kernel = probability_signature(softmax(replay_logits))
                    u, v, _, _, _ = checkpoint
                    incomplete_checkpoint: FiniteState = (u, v, 0, 0, 0)
                    incomplete_terminal = continue_state(
                        incomplete_checkpoint, completed_layer_count
                    )
                    incomplete_logits = terminal_logits(incomplete_terminal)
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE_ID,
                            "case_id": case_id,
                            "content_id": content,
                            "role_id": role,
                            "history_id": history,
                            "completed_layer_count": completed_layer_count,
                            "checkpoint_state": list(checkpoint),
                            "direct_terminal_logits": list(direct_logits),
                            "replay_terminal_logits": list(replay_logits),
                            "direct_terminal_kernel": list(direct_kernel),
                            "replay_terminal_kernel": list(replay_kernel),
                            "complete_natural_replay_exact": replay_logits
                            == direct_logits,
                            "incomplete_local_state_terminal_logits": list(
                                incomplete_logits
                            ),
                            "incomplete_local_state_replay_exact": incomplete_logits
                            == direct_logits,
                            "synthetic": True,
                            "model_derived": False,
                        }
                    )
    case_count = len(cases)
    checkpoint_count = len(LAYERS) + 1
    natural_failures = sum(not row["complete_natural_replay_exact"] for row in rows)
    incomplete_failures = sum(
        not row["incomplete_local_state_replay_exact"] for row in rows
    )
    per_case_kernels = {
        case_id: {
            tuple(row["replay_terminal_kernel"])
            for row in rows
            if row["case_id"] == case_id
        }
        for case_id in cases
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-NaturalCompleteStateReplayIdentityAudit",
        "created_at": created_at,
        "valid": bool(
            natural_failures == 0
            and all(len(kernels) == 1 for kernels in per_case_kernels.values())
            and incomplete_failures > 0
        ),
        "synthetic_case_count": case_count,
        "checkpoint_count_per_case": checkpoint_count,
        "complete_natural_replay_cell_count": len(rows),
        "complete_natural_replay_exact_count": len(rows) - natural_failures,
        "complete_natural_replay_failure_count": natural_failures,
        "case_with_layerwise_terminal_kernel_variation_count": sum(
            len(kernels) > 1 for kernels in per_case_kernels.values()
        ),
        "incomplete_local_state_replay_cell_count": len(rows),
        "incomplete_local_state_replay_failure_count": incomplete_failures,
        "natural_complete_replay_is_layerwise_candidate_curve": False,
        "valid_uses": [
            "instrument_fidelity",
            "checkpoint_completeness",
            "unchanged_downstream_replay",
            "counterfactual_replay_baseline_after_authorization",
        ],
        "claim_boundary": (
            "finite_deterministic_identity_and_incomplete_state_counterexample_not_real_model_replay_evidence"
        ),
    }
    return summary, rows, cases


def trajectory_ontology(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    objects = [
        {
            "trajectory_id": "generation_time_terminal_kernel",
            "varying_axis": "generated_prefix",
            "native_probability": True,
            "observer_required": False,
            "intervention_required": False,
            "evidence_class": "native_behavioral_probability",
            "authorized_now": False,
        },
        {
            "trajectory_id": "layer_observer_readability",
            "varying_axis": "layer_or_event",
            "native_probability": False,
            "observer_required": True,
            "intervention_required": False,
            "evidence_class": "observer_indexed_readability",
            "authorized_now": False,
        },
        {
            "trajectory_id": "natural_physical_event_state",
            "varying_axis": "layer_component_position_time",
            "native_probability": False,
            "observer_required": False,
            "intervention_required": False,
            "evidence_class": "descriptive_physical_state",
            "authorized_now": False,
        },
        {
            "trajectory_id": "counterfactual_terminal_kernel_effect",
            "varying_axis": "intervened_event",
            "native_probability": True,
            "observer_required": False,
            "intervention_required": True,
            "evidence_class": "causal_terminal_effect",
            "authorized_now": False,
        },
        {
            "trajectory_id": "natural_complete_state_replay_identity",
            "varying_axis": "checkpoint",
            "native_probability": True,
            "observer_required": False,
            "intervention_required": False,
            "evidence_class": "instrument_identity_not_progress_trajectory",
            "authorized_now": False,
        },
    ]
    rows = [
        {"schema_version": SCHEMA_VERSION, "phase_id": PHASE_ID, **row}
        for row in objects
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-TrajectoryOntology",
        "created_at": created_at,
        "valid": True,
        "registered_trajectory_object_count": len(rows),
        "native_generation_time_probability_trajectory_count": sum(
            row["trajectory_id"] == "generation_time_terminal_kernel"
            for row in rows
        ),
        "observer_indexed_layer_trajectory_count": sum(
            row["observer_required"] for row in rows
        ),
        "natural_descriptive_physical_trajectory_count": sum(
            row["evidence_class"] == "descriptive_physical_state" for row in rows
        ),
        "causal_effect_trajectory_count": sum(
            row["intervention_required"] for row in rows
        ),
        "instrument_identity_object_count": sum(
            row["evidence_class"] == "instrument_identity_not_progress_trajectory"
            for row in rows
        ),
        "unqualified_generic_intermediate_candidate_trajectory_count": 0,
        "mixing_objects_without_type_label_authorized": False,
        "claim_boundary": "measurement_ontology_only_no_model_trajectory_collected",
    }
    return summary, rows


def observer_readability_audit(
    created_at: str, cases: dict[str, tuple[FiniteState, ...]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observer_rows: list[dict[str, Any]] = []
    observer_functions = {
        "raw_coordinate_observer": lambda state: (state[0], state[1], state[0] - state[1]),
        "typed_role_history_observer": lambda state: terminal_logits(state),
    }
    for case_id, checkpoints in cases.items():
        direct_kernel = probability_signature(softmax(terminal_logits(checkpoints[-1])))
        for completed_layer_count, checkpoint in enumerate(checkpoints):
            readings = {
                observer_id: probability_signature(softmax(function(checkpoint)))
                for observer_id, function in observer_functions.items()
            }
            for observer_id, reading in readings.items():
                observer_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "case_id": case_id,
                        "completed_layer_count": completed_layer_count,
                        "observer_id": observer_id,
                        "observer_reading": list(reading),
                        "natural_terminal_kernel": list(direct_kernel),
                        "matches_terminal_kernel": reading == direct_kernel,
                        "observer_indexed": True,
                        "native_intermediate_probability": False,
                        "synthetic": True,
                    }
                )
    observer_ids = tuple(observer_functions)
    varying_case_observer_count = 0
    for case_id in cases:
        for observer_id in observer_ids:
            signatures = {
                tuple(row["observer_reading"])
                for row in observer_rows
                if row["case_id"] == case_id
                and row["observer_id"] == observer_id
            }
            varying_case_observer_count += len(signatures) > 1
    disagreement_cells = 0
    for case_id, checkpoints in cases.items():
        for completed_layer_count in range(len(checkpoints)):
            readings = {
                tuple(row["observer_reading"])
                for row in observer_rows
                if row["case_id"] == case_id
                and row["completed_layer_count"] == completed_layer_count
            }
            disagreement_cells += len(readings) > 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-ObserverIndexedReadabilityAudit",
        "created_at": created_at,
        "valid": bool(
            varying_case_observer_count > 0 and disagreement_cells > 0
        ),
        "synthetic_case_count": len(cases),
        "observer_count": len(observer_ids),
        "checkpoint_count_per_case": len(LAYERS) + 1,
        "observer_cell_count": len(observer_rows),
        "case_observer_trajectory_count": len(cases) * len(observer_ids),
        "varying_case_observer_trajectory_count": varying_case_observer_count,
        "same_state_observer_disagreement_cell_count": disagreement_cells,
        "observer_index_required": True,
        "native_intermediate_probability_count": 0,
        "claim_boundary": (
            "synthetic_observer_dependence_demonstration_not_model_readout_qualification"
        ),
    }
    return summary, observer_rows


def is_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return len(left) <= len(right) and left == right[: len(left)]


def panel_is_prefix_free(sequences: list[tuple[str, ...]]) -> bool:
    return all(
        not is_prefix(left, right)
        for index, left in enumerate(sequences)
        for other_index, right in enumerate(sequences)
        if index != other_index
    )


TOKEN_KERNEL: dict[tuple[str, ...], dict[str, Fraction]] = {
    (): {"A": Fraction(3, 10), "B": Fraction(4, 10), "EOS": Fraction(1, 10), "OTHER": Fraction(2, 10)},
    ("A",): {"EOS": Fraction(8, 10), "OTHER": Fraction(2, 10)},
    ("B",): {"C": Fraction(5, 10), "A": Fraction(25, 100), "OTHER": Fraction(25, 100)},
    ("B", "C"): {"EOS": Fraction(6, 10), "OTHER": Fraction(4, 10)},
    ("B", "A"): {"EOS": Fraction(5, 10), "OTHER": Fraction(5, 10)},
}


def sequence_probability(sequence: tuple[str, ...]) -> Fraction:
    probability = Fraction(1)
    prefix: tuple[str, ...] = ()
    for token in sequence:
        probability *= TOKEN_KERNEL[prefix][token]
        prefix = (*prefix, token)
    return probability


def variable_length_event_panel(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    events = [
        (
            "valid_short",
            "semantic_valid_answer",
            ("A", "EOS"),
            {"semantic_valid": True, "grammar_valid": True, "format_valid": True, "stop_valid": True},
        ),
        (
            "valid_long",
            "semantic_valid_explanation",
            ("B", "C", "EOS"),
            {"semantic_valid": True, "grammar_valid": True, "format_valid": True, "stop_valid": True},
        ),
        (
            "format_wrong",
            "format_invalid_response",
            ("B", "A", "EOS"),
            {"semantic_valid": True, "grammar_valid": True, "format_valid": False, "stop_valid": True},
        ),
        (
            "premature_stop",
            "premature_stop_event",
            ("EOS",),
            {"semantic_valid": False, "grammar_valid": True, "format_valid": False, "stop_valid": False},
        ),
    ]
    sequences = [event[2] for event in events]
    prefix_free = panel_is_prefix_free(sequences)
    rows = []
    raw_probabilities = []
    for event_id, semantic_event_id, tokens, axes in events:
        probability = sequence_probability(tokens)
        raw_probabilities.append(probability)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "panel_id": "phase414_variable_length_prefix_event_panel",
                "event_id": event_id,
                "semantic_event_id": semantic_event_id,
                "abstract_tokens": list(tokens),
                "length": len(tokens),
                "ends_with_eos": tokens[-1] == "EOS",
                "axis_validity": axes,
                "raw_event_probability_fraction": fraction_text(probability),
                "raw_event_probability": float(probability),
                "synthetic": True,
                "model_token_ids_registered": False,
            }
        )
    panel_mass = sum(raw_probabilities, Fraction(0))
    outside_mass = Fraction(1) - panel_mass
    for row, raw_probability in zip(rows, raw_probabilities, strict=True):
        row["conditional_panel_probability_fraction"] = fraction_text(
            raw_probability / panel_mass
        )
        row["conditional_panel_probability"] = float(raw_probability / panel_mass)

    invalid_sequences = [("B",), ("B", "C", "EOS")]
    invalid_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "panel_id": "phase414_invalid_prefix_conflict_control",
            "abstract_tokens": list(sequence),
            "ends_with_eos": sequence[-1] == "EOS",
            "registered": False,
            "blocking_reason": (
                "nonterminal_prefix_overlaps_a_longer_registered_event"
                if sequence == ("B",)
                else "overlapped_by_nonterminal_prefix_event"
            ),
        }
        for sequence in invalid_sequences
    ]
    invalid_rejected = not panel_is_prefix_free(invalid_sequences) and all(
        not row["registered"] for row in invalid_rows
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-VariableLengthCandidateEventContract",
        "created_at": created_at,
        "valid": bool(
            prefix_free
            and len({len(sequence) for sequence in sequences}) > 1
            and all(row["ends_with_eos"] for row in rows)
            and panel_mass <= 1
            and invalid_rejected
        ),
        "registered_event_count": len(rows),
        "event_length_set": sorted({len(sequence) for sequence in sequences}),
        "equal_length_required": False,
        "pairwise_prefix_free": prefix_free,
        "all_events_eos_closed": all(row["ends_with_eos"] for row in rows),
        "panel_mass_fraction": fraction_text(panel_mass),
        "panel_mass": float(panel_mass),
        "outside_mass_fraction": fraction_text(outside_mass),
        "outside_mass": float(outside_mass),
        "invalid_prefix_conflict_panel_count": 1,
        "invalid_prefix_conflict_panel_rejected_count": int(invalid_rejected),
        "model_token_ids_registered": False,
        "general_rule": (
            "events_may_have_different_lengths_when_prefix_free_eos_closed_or_proven_disjoint_by_a_finite_automaton"
        ),
        "claim_boundary": "synthetic_event_probability_contract_not_real_model_candidate_coverage",
    }
    return summary, rows, invalid_rows


def cross_tokenizer_semantic_alignment(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registries = {
        "synthetic_tokenizer_a": {
            "tokenizer_hash": "sha256:synthetic_a",
            "events": {
                "answer_red": [101, 2],
                "grammar_plural": [102, 103, 2],
                "response_complete": [2],
            },
        },
        "synthetic_tokenizer_b": {
            "tokenizer_hash": "sha256:synthetic_b",
            "events": {
                "answer_red": [41, 42, 99],
                "grammar_plural": [43, 44, 45, 99],
                "response_complete": [99],
            },
        },
    }
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "tokenizer_id": tokenizer_id,
            "tokenizer_hash": registry["tokenizer_hash"],
            "semantic_event_id": semantic_event_id,
            "token_ids": token_ids,
            "synthetic": True,
            "model_derived": False,
        }
        for tokenizer_id, registry in registries.items()
        for semantic_event_id, token_ids in registry["events"].items()
    ]
    event_ids = tuple(registries["synthetic_tokenizer_a"]["events"])
    aligned = sum(
        event_id in registries["synthetic_tokenizer_b"]["events"]
        for event_id in event_ids
    )
    token_matches = sum(
        registries["synthetic_tokenizer_a"]["events"][event_id]
        == registries["synthetic_tokenizer_b"]["events"][event_id]
        for event_id in event_ids
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-CrossTokenizerSemanticEventAlignment",
        "created_at": created_at,
        "valid": bool(aligned == len(event_ids) and token_matches < aligned),
        "tokenizer_count": len(registries),
        "semantic_event_count": len(event_ids),
        "cross_tokenizer_semantic_event_alignment_count": aligned,
        "cross_tokenizer_identical_token_id_sequence_count": token_matches,
        "comparison_unit": "registered_semantic_event_not_token_id_sequence",
        "claim_boundary": "synthetic_tokenization_counterexample_not_cross_model_behavioral_equivalence",
    }
    return summary, rows


def observer_qualification_contract(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    methods = [
        {
            "observer_id": "raw_logit_lens",
            "observer_kind": "diagnostic",
            "capacity_class": "fixed_linear",
            "qualified": False,
            "failure_reason": "not_native_and_not_independently_calibrated",
        },
        {
            "observer_id": "final_norm_logit_lens",
            "observer_kind": "diagnostic",
            "capacity_class": "fixed_normalized_linear",
            "qualified": False,
            "failure_reason": "normalization_does_not_make_the_readout_native",
        },
        {
            "observer_id": "low_capacity_calibrated_observer",
            "observer_kind": "learned_observational_decoder",
            "capacity_class": "frozen_before_training",
            "qualified": False,
            "failure_reason": "no_model_discovery_calibration_or_holdout_run",
        },
    ]
    required_splits = [
        "discovery",
        "calibration",
        "behavioral_holdout",
        "lexical_holdout",
        "surface_holdout",
        "role_holdout",
        "interface_holdout",
        "history_holdout",
    ]
    controls = [
        "shuffled_terminal_targets",
        "wrong_layer",
        "random_coordinate_transport",
        "high_capacity_memorization_comparator",
        "outside_panel_mass_error",
        "channel_permutation_transport",
    ]
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            **method,
            "required_splits": required_splits,
            "required_controls": controls,
            "may_be_called_native_intermediate_probability": False,
        }
        for method in methods
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-ObserverQualificationContract",
        "created_at": created_at,
        "valid": True,
        "observer_method_count": len(rows),
        "diagnostic_observer_count": sum(
            row["observer_kind"] == "diagnostic" for row in rows
        ),
        "learned_observer_pending_count": sum(
            row["observer_kind"] == "learned_observational_decoder"
            for row in rows
        ),
        "qualified_observer_count": sum(row["qualified"] for row in rows),
        "required_split_count": len(required_splits),
        "required_control_count": len(controls),
        "low_calibration_error_proves_natural_model_use": False,
        "claim_boundary": "observer_qualification_design_only_no_model_fit",
    }
    return summary, rows


CATALOG: dict[str, list[tuple[str, str]]] = {
    "A_terminal_prediction_future": [
        ("terminal_probability_architecture_scope", "architecture_or_mathematical_boundary"),
        ("greedy_not_sequence_map", "architecture_or_mathematical_boundary"),
        ("recursive_future_kernel", "architecture_or_mathematical_boundary"),
        ("one_step_not_full_future", "finite_or_protocol_result"),
        ("terminal_not_internal_path", "finite_or_protocol_result"),
        ("candidate_mass_need_not_monotonic", "finite_or_protocol_result"),
        ("semantic_boundary_stop_separated", "prior_scoped_empirical_result"),
        ("terminal_interface_not_conserved_scalar", "finite_or_protocol_result"),
    ],
    "B_intermediate_readout_observer": [
        ("no_native_local_probability_head", "architecture_or_mathematical_boundary"),
        ("raw_lens_diagnostic", "finite_or_protocol_result"),
        ("normalized_lens_diagnostic", "finite_or_protocol_result"),
        ("calibrated_lens_learned", "finite_or_protocol_result"),
        ("natural_complete_replay_identity", "finite_or_protocol_result"),
        ("local_single_position_state_incomplete", "prior_scoped_empirical_result"),
        ("readout_requires_split_calibration_transport", "finite_or_protocol_result"),
        ("intermediate_trace_observer_indexed", "finite_or_protocol_result"),
    ],
    "C_coordinates_reparameterization": [
        ("mlp_channel_permutation", "finite_or_protocol_result"),
        ("fixed_neuron_id_not_universal_identity", "finite_or_protocol_result"),
        ("readout_coordinate_transport", "finite_or_protocol_result"),
        ("permutation_not_arbitrary_residual_rotation", "finite_or_protocol_result"),
        ("compare_functional_roles_cross_model", "finite_or_protocol_result"),
        ("functional_sets_may_have_multiple_coordinates", "working_hypothesis"),
        ("terminal_behavior_not_unique_implementation", "finite_or_protocol_result"),
        ("identity_requires_allowed_reparameterization_stability", "finite_or_protocol_result"),
    ],
    "D_candidate_panels": [
        ("candidate_sequences_define_events", "finite_or_protocol_result"),
        ("disjointness_for_probability_sum", "architecture_or_mathematical_boundary"),
        ("equal_length_sufficient_not_necessary", "finite_or_protocol_result"),
        ("prefix_free_or_terminal_events", "finite_or_protocol_result"),
        ("raw_panel_mass_required", "finite_or_protocol_result"),
        ("outside_panel_mass_required", "finite_or_protocol_result"),
        ("validity_axes_nonexclusive", "finite_or_protocol_result"),
        ("cross_model_semantic_event_alignment", "finite_or_protocol_result"),
    ],
    "E_execution_physical_ledger": [
        ("batch_size_numeric_effect", "prior_scoped_empirical_result"),
        ("dtype_kernel_numeric_effect", "prior_scoped_empirical_result"),
        ("cache_hook_contract_required", "prior_scoped_empirical_result"),
        ("component_ledger_replay", "prior_scoped_empirical_result"),
        ("residual_content_carriage", "prior_scoped_empirical_result"),
        ("source_to_query_write_events", "prior_scoped_empirical_result"),
        ("query_component_integration", "prior_scoped_empirical_result"),
        ("late_terminal_content_causal_effect", "prior_scoped_empirical_result"),
    ],
    "F_static_route_negative_results": [
        ("single_neuron_not_closed", "prior_scoped_empirical_result"),
        ("single_head_not_cross_model_path", "prior_scoped_empirical_result"),
        ("fixed_semantic_direction_not_closed", "prior_scoped_empirical_result"),
        ("static_relation_vector_not_sufficient", "prior_scoped_empirical_result"),
        ("query_point_not_sufficient_binding", "prior_scoped_empirical_result"),
        ("peak_layer_chain_not_computation_order", "prior_scoped_empirical_result"),
        ("wide_event_interval_not_answer_predictor", "prior_scoped_empirical_result"),
        ("single_layer_multiparent_not_specific", "prior_scoped_empirical_result"),
    ],
    "G_typed_state_observer": [
        ("world_state_requires_query_role", "finite_or_protocol_result"),
        ("state_transform_transports_query", "finite_or_protocol_result"),
        ("state_transform_relabels_response", "finite_or_protocol_result"),
        ("fixed_observer_false_failure", "finite_or_protocol_result"),
        ("typed_covariance_external_contract", "finite_or_protocol_result"),
        ("single_role_conditional_partition", "finite_or_protocol_result"),
        ("role_movement_transports_partitions", "finite_or_protocol_result"),
        ("role_bundle_not_model_recovered", "open_gap"),
    ],
    "H_state_operations_algebra": [
        ("finite_operations_mostly_permutations", "finite_or_protocol_result"),
        ("closure_external_by_definition", "finite_or_protocol_result"),
        ("partition_operation_congruence", "finite_or_protocol_result"),
        ("observation_sufficiency_distinct_from_congruence", "finite_or_protocol_result"),
        ("joint_injective_observation_singletons", "finite_or_protocol_result"),
        ("no_global_nontrivial_quotient", "finite_or_protocol_result"),
        ("irreversible_state_universe_missing", "open_gap"),
        ("cross_family_bridge_missing", "open_gap"),
    ],
    "I_knowledge_network": [
        ("knowledge_entity_class_attribute_relation_binding", "working_hypothesis"),
        ("content_and_binding_states_distinct", "working_hypothesis"),
        ("query_role_selects_projection", "finite_or_protocol_result"),
        ("single_query_not_full_world", "finite_or_protocol_result"),
        ("joint_signature_external_measurement", "finite_or_protocol_result"),
        ("parameter_and_context_knowledge_distinct", "working_hypothesis"),
        ("object_attribute_binding_operator_missing", "open_gap"),
        ("knowledge_causal_path_missing", "open_gap"),
    ],
    "J_reasoning": [
        ("reasoning_requires_state_transition", "working_hypothesis"),
        ("one_step_equivalence_insufficient", "finite_or_protocol_result"),
        ("reasoning_fact_rule_constraint_intermediate", "working_hypothesis"),
        ("direct_terminal_vs_instruction_execution", "finite_or_protocol_result"),
        ("irreversible_reasoning_universe_missing", "open_gap"),
        ("cross_content_operator_missing", "open_gap"),
        ("unseen_composition_prediction_missing", "open_gap"),
        ("reasoning_causal_path_missing", "open_gap"),
    ],
    "K_grammar_format_punctuation": [
        ("grammar_as_legal_continuation_set", "working_hypothesis"),
        ("grammar_state_interface_distinct", "finite_or_protocol_result"),
        ("punctuation_boundary_role", "working_hypothesis"),
        ("format_not_semantic_ontology", "finite_or_protocol_result"),
        ("semantic_boundary_stop_distinct", "prior_scoped_empirical_result"),
        ("stop_irreversible_operation_candidate", "working_hypothesis"),
        ("cross_model_grammar_state_missing", "open_gap"),
        ("grammar_causal_path_missing", "open_gap"),
    ],
    "L_theory_atlas_closure": [
        ("observation_edge_not_prediction_edge", "finite_or_protocol_result"),
        ("prediction_edge_not_compute_edge", "finite_or_protocol_result"),
        ("compute_edge_not_functional_causal_edge", "finite_or_protocol_result"),
        ("terminal_equivalence_not_internal_identity", "finite_or_protocol_result"),
        ("functional_to_physical_map_missing", "open_gap"),
        ("small_model_extrapolation_limited", "finite_or_protocol_result"),
        ("complete_language_paths_zero", "open_gap"),
        ("single_neuron_and_strict_closure_zero", "open_gap"),
    ],
}


def catalog_qualification(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for category, items in CATALOG.items():
        category_code = category[0]
        for index, (claim_slug, status) in enumerate(items, start=1):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "catalog_item_id": f"{category_code}{index:02d}",
                    "category": category,
                    "claim_slug": claim_slug,
                    "evidence_status": status,
                    "strict_model_mechanism_closed": False,
                    "counts_as_global_progress_unit": False,
                }
            )
    counts = Counter(row["evidence_status"] for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-NinetySixItemCatalogQualification",
        "created_at": created_at,
        "valid": bool(len(rows) == 96 and len(CATALOG) == 12),
        "catalog_item_count": len(rows),
        "category_count": len(CATALOG),
        "items_per_category": sorted(
            Counter(row["category"] for row in rows).values()
        ),
        "evidence_status_counts": dict(sorted(counts.items())),
        "strict_model_mechanism_closed_item_count": sum(
            row["strict_model_mechanism_closed"] for row in rows
        ),
        "global_progress_unit_count": sum(
            row["counts_as_global_progress_unit"] for row in rows
        ),
        "stable_homogeneous_knowledge_directory": False,
        "catalog_as_completion_percentage_denominator_valid": False,
        "reason": (
            "catalog_mixes_architecture_boundaries_finite_protocol_prior_scoped_empirical_results_working_hypotheses_and_open_gaps"
        ),
        "claim_boundary": "evidence_ledger_not_language_mechanism_completion_score",
    }
    return summary, rows


def supplied_claim_audit(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    claims = [
        ("natural_complete_state_replay_is_terminal_identity", "supported_by_finite_or_definitional_analysis"),
        ("natural_replay_is_not_layer_candidate_narrowing", "supported_by_finite_or_definitional_analysis"),
        ("generation_observer_and_causal_traces_must_be_separate", "supported_by_current_methodology"),
        ("variable_length_events_can_be_valid_when_disjoint", "supported_by_finite_or_definitional_analysis"),
        ("outside_panel_mass_must_be_reported", "supported_by_current_methodology"),
        ("cross_model_alignment_should_use_semantic_events", "supported_by_current_methodology"),
        ("channel_permutation_has_limited_scope", "supported_by_current_methodology"),
        ("learned_observer_can_overfit_terminal_targets", "supported_by_current_methodology"),
        ("external_review_and_collector_gates_remain_closed", "supported_by_current_artifacts"),
        ("small_model_results_require_scope_limits", "supported_by_current_methodology"),
        ("event_fingerprint_is_next_measurement_contract", "methodologically_valid_proposal"),
        ("dynamic_mode_network_is_best_current_working_hypothesis", "methodologically_valid_proposal"),
        ("six_level_global_physical_atlas_is_confirmed", "requires_evidence_grade_correction"),
        ("content_query_and_terminal_reuse_are_all_confirmed", "requires_evidence_grade_correction"),
        ("proposed_structural_invariant_is_established", "requires_evidence_grade_correction"),
        ("ninety_six_items_are_uniform_stable_knowledge", "incorrect_as_stated"),
        ("project_management_progress_is_twenty_two_to_twenty_four_percent", "incorrect_as_stated"),
        ("phase414_model_collection_can_start_now", "incorrect_as_stated"),
    ]
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "claim_id": claim_id,
            "status": status,
        }
        for claim_id, status in claims
    ]
    counts = Counter(row["status"] for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-SuppliedPhase413Reaudit",
        "created_at": created_at,
        "valid": True,
        "claim_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "overall_assessment": (
            "deep_replay_correction_is_valid_but_physical_catalog_and_progress_claims_overstate_current_evidence"
        ),
        "claim_boundary": "audit_against_phase413_artifacts_and_phase414_finite_contracts",
    }
    return summary, rows


def main() -> None:
    created_at = now()
    phase413 = read_json(P413 / "phase413_stage_summary.json")
    replay, replay_rows, finite_cases = natural_replay_audit(created_at)
    ontology, ontology_rows = trajectory_ontology(created_at)
    readability, readability_rows = observer_readability_audit(
        created_at, finite_cases
    )
    panel, panel_rows, invalid_panel_rows = variable_length_event_panel(created_at)
    tokenizers, tokenizer_rows = cross_tokenizer_semantic_alignment(created_at)
    observer_contract, observer_rows = observer_qualification_contract(created_at)
    catalog, catalog_rows = catalog_qualification(created_at)
    claims, claim_rows = supplied_claim_audit(created_at)

    machine_preflight = all(
        artifact["valid"]
        for artifact in (
            replay,
            ontology,
            readability,
            panel,
            tokenizers,
            observer_contract,
            catalog,
            claims,
        )
    )
    completed_reviewers = phase413["results"]["completed_external_reviewer_count"]
    required_reviewers = phase413["denominators"][
        "required_independent_reviewer_count"
    ]
    collector_cases = phase413["results"][
        "sealed_model_collector_equivalence_case_count"
    ]
    collector_denominator = phase413["denominators"][
        "future_sealed_model_collector_case_count"
    ]
    model_authorized = bool(
        machine_preflight
        and completed_reviewers == required_reviewers
        and collector_cases == collector_denominator
        and observer_contract["qualified_observer_count"] > 0
    )
    qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase414-ExecutionQualification",
        "created_at": created_at,
        "machine_preflight_pass": machine_preflight,
        "natural_replay_identity_contract_pass": replay["valid"],
        "trajectory_ontology_pass": ontology["valid"],
        "observer_index_contract_pass": readability["valid"],
        "variable_length_event_contract_pass": panel["valid"],
        "cross_tokenizer_semantic_alignment_contract_pass": tokenizers["valid"],
        "catalog_qualification_pass": catalog["valid"],
        "independent_human_rule_review_completed": completed_reviewers
        == required_reviewers,
        "sealed_model_collector_equivalence_completed": collector_cases
        == collector_denominator,
        "qualified_observer_available": observer_contract[
            "qualified_observer_count"
        ]
        > 0,
        "model_qualification_authorized": model_authorized,
        "formal_model_discovery_authorized": False,
        "descriptive_physical_mapping_authorized": False,
        "causal_intervention_authorized": False,
        "neuron_scan_authorized": False,
    }
    stage = {
        "schema_version": "88.1.0",
        "phase_id": "Phase414-ObserverIndexedEventPreflightStage",
        "created_at": created_at,
        "objective": (
            "classify_natural_complete_state_replay_as_an_identity_gate_and_freeze_observer_indexed_layer_traces_variable_length_candidate_events_and_mixed_catalog_evidence_before_model_collection"
        ),
        "assessment": {
            "supplied_phase413_reaudit_direction_correct": True,
            "natural_complete_state_replay_is_terminal_identity": True,
            "natural_complete_state_replay_is_layerwise_candidate_curve": False,
            "generation_time_terminal_kernel_is_native_probability_trajectory": True,
            "layer_readability_is_observer_indexed": True,
            "counterfactual_terminal_effect_is_causal": True,
            "equal_length_candidate_events_required": False,
            "cross_model_token_id_alignment_required": False,
            "ninety_six_item_catalog_is_homogeneous_stable_knowledge": False,
            "single_project_management_percentage_valid": False,
            "machine_preflight_pass": machine_preflight,
            "model_weight_loaded": False,
            "cuda_execution_performed": False,
            "behavioral_case_collected": False,
            "physical_trace_collected": False,
            "causal_intervention_performed": False,
            "neuron_scan_performed": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "supplied_claim_count": claims["claim_count"],
            "catalog_item_count": catalog["catalog_item_count"],
            "catalog_category_count": catalog["category_count"],
            "trajectory_object_count": ontology[
                "registered_trajectory_object_count"
            ],
            "natural_replay_case_count": replay["synthetic_case_count"],
            "natural_replay_cell_count": replay[
                "complete_natural_replay_cell_count"
            ],
            "observer_readability_cell_count": readability["observer_cell_count"],
            "case_observer_trajectory_count": readability[
                "case_observer_trajectory_count"
            ],
            "candidate_event_count": panel["registered_event_count"],
            "invalid_prefix_panel_count": panel[
                "invalid_prefix_conflict_panel_count"
            ],
            "cross_tokenizer_semantic_event_count": tokenizers[
                "semantic_event_count"
            ],
            "observer_method_count": observer_contract["observer_method_count"],
            "required_independent_reviewer_count": required_reviewers,
            "future_sealed_model_collector_case_count": collector_denominator,
            "model_case_count_consumed": 0,
            "physical_case_count_consumed": 0,
        },
        "results": {
            "complete_natural_replay_exact_count": replay[
                "complete_natural_replay_exact_count"
            ],
            "complete_natural_replay_failure_count": replay[
                "complete_natural_replay_failure_count"
            ],
            "case_with_layerwise_terminal_kernel_variation_count": replay[
                "case_with_layerwise_terminal_kernel_variation_count"
            ],
            "incomplete_local_state_replay_failure_count": replay[
                "incomplete_local_state_replay_failure_count"
            ],
            "varying_case_observer_trajectory_count": readability[
                "varying_case_observer_trajectory_count"
            ],
            "same_state_observer_disagreement_cell_count": readability[
                "same_state_observer_disagreement_cell_count"
            ],
            "valid_variable_length_event_panel_count": int(panel["valid"]),
            "invalid_prefix_panel_rejected_count": panel[
                "invalid_prefix_conflict_panel_rejected_count"
            ],
            "candidate_panel_mass": panel["panel_mass"],
            "candidate_outside_mass": panel["outside_mass"],
            "cross_tokenizer_semantic_event_alignment_count": tokenizers[
                "cross_tokenizer_semantic_event_alignment_count"
            ],
            "cross_tokenizer_identical_token_id_sequence_count": tokenizers[
                "cross_tokenizer_identical_token_id_sequence_count"
            ],
            "qualified_observer_count": observer_contract[
                "qualified_observer_count"
            ],
            "strict_model_mechanism_closed_catalog_item_count": catalog[
                "strict_model_mechanism_closed_item_count"
            ],
            "global_progress_catalog_item_count": catalog[
                "global_progress_unit_count"
            ],
            "completed_external_reviewer_count": completed_reviewers,
            "sealed_model_collector_equivalence_case_count": collector_cases,
            "new_behavioral_result_count": 0,
            "new_physical_path_count": 0,
            "new_causal_path_count": 0,
            "new_neuron_path_count": 0,
        },
        "hard_limits": [
            "all_positive_results_are_finite_protocol_or_measurement_ontology_results",
            "natural_replay_identity_is_definitional_and_not_real_model_layer_progress",
            "observer_readability_rows_are_synthetic_and_do_not_qualify_a_model_observer",
            "variable_length_event_probabilities_use_an_abstract_token_tree",
            "semantic_event_alignment_uses_synthetic_tokenizers",
            "the_ninety_six_item_catalog_mixes_multiple_evidence_classes",
            "prior_physical_atlas_claims_keep_their_original_phase_specific_evidence_grades",
            "no_new_model_behavior_physical_causal_or_neuron_evidence_was_added",
            "two_external_human_reviews_are_still_absent",
            "sealed_real_model_collector_equivalence_remains_zero_of_165",
            "qualified_intermediate_observer_remains_zero_of_three",
            "small_models_may_use_coarse_redundant_or_model_specific_structures",
            "single_global_progress_percentage_is_invalid",
        ],
        "authorization": {
            "publish_observer_event_preflight": True,
            "publish_natural_replay_as_layer_candidate_curve": False,
            "publish_synthetic_observer_traces_as_model_readability": False,
            "publish_ninety_six_catalog_as_completion_progress": False,
            "run_qwen3_model_qualification_next": model_authorized,
            "run_glm4_model_qualification_next": False,
            "run_ds7b_model_qualification_next": False,
            "run_formal_discovery_next": False,
            "run_descriptive_physical_mapping_next": False,
            "run_causal_intervention_next": False,
            "run_neuron_scan_next": False,
        },
        "next_stage": {
            "phase_id": "Phase410A-ExternalReviewAndCollectorGate",
            "same_qualification_stage": True,
            "automatic_execution_now": False,
            "blocking_requirements": [
                "two_distinct_external_reviewers_complete_all_65_blinded_items",
                "reference_and_incremental_collectors_match_all_165_sealed_real_model_cases",
                "real_tokenizer_specific_events_are_registered_and_semantically_aligned",
                "at_least_one_low_capacity_observer_passes_frozen_cross_surface_cross_lexical_role_interface_and_history_holdouts",
            ],
            "first_model_measurements_after_gate": [
                "generation_time_native_terminal_kernel",
                "natural_complete_state_replay_identity_only",
                "observer_indexed_readability_with_explicit_observer_id",
            ],
            "model_order_after_gate": list(MODELS),
        },
        "single_global_progress_percentage_valid": False,
    }

    write_json(OUT / "phase414_supplied_claim_audit.json", claims)
    write_jsonl(
        OUT / "protocol/private/phase414_supplied_claim_rows.jsonl", claim_rows
    )
    write_json(OUT / "phase414_natural_replay_identity_audit.json", replay)
    write_jsonl(
        OUT / "protocol/private/phase414_natural_replay_rows.jsonl", replay_rows
    )
    write_json(OUT / "phase414_trajectory_ontology.json", ontology)
    write_jsonl(
        OUT / "protocol/private/phase414_trajectory_object_rows.jsonl",
        ontology_rows,
    )
    write_json(OUT / "phase414_observer_readability_audit.json", readability)
    write_jsonl(
        OUT / "protocol/private/phase414_observer_readability_rows.jsonl",
        readability_rows,
    )
    write_json(OUT / "phase414_variable_length_event_contract.json", panel)
    write_jsonl(
        OUT / "protocol/private/phase414_candidate_event_rows.jsonl", panel_rows
    )
    write_jsonl(
        OUT / "protocol/private/phase414_invalid_event_panel_rows.jsonl",
        invalid_panel_rows,
    )
    write_json(
        OUT / "phase414_cross_tokenizer_semantic_alignment.json", tokenizers
    )
    write_jsonl(
        OUT / "protocol/private/phase414_cross_tokenizer_event_rows.jsonl",
        tokenizer_rows,
    )
    write_json(
        OUT / "phase414_observer_qualification_contract.json", observer_contract
    )
    write_jsonl(
        OUT / "protocol/private/phase414_observer_method_rows.jsonl", observer_rows
    )
    write_json(OUT / "phase414_catalog_qualification.json", catalog)
    write_jsonl(
        OUT / "protocol/private/phase414_catalog_rows.jsonl", catalog_rows
    )
    write_json(OUT / "phase414_execution_qualification.json", qualification)
    write_json(OUT / "phase414_stage_summary.json", stage)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
