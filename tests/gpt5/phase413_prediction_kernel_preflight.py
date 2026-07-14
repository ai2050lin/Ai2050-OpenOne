#!/usr/bin/env python3
"""Qualify prediction-kernel and candidate-trajectory measurements.

The model natively produces a next-token distribution only at its terminal
readout.  A residual or component state at an earlier event does not by itself
carry a normalized future-token distribution.  This preflight therefore does
four model-free things before any CUDA collection is allowed:

1. separates terminal kernels, diagnostic lenses, learned probes, complete-
   state continuation, and counterfactual interventions;
2. gives finite counterexamples showing that terminal behavior does not
   identify an internal candidate-narrowing trajectory;
3. records an exact channel-permutation symmetry showing that fixed neuron
   coordinates are not identified by the final output alone; and
4. freezes a disjoint, equal-horizon, multi-axis candidate-panel contract.

All positive results are protocol or finite-construction results.  The script
does not load a model and establishes no behavioral, physical, causal, or
neuron-level language mechanism.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase409_dynamic_response_protocol import MODELS  # noqa: E402
from phase411_finite_operation_preflight import (  # noqa: E402
    canonical_json,
    read_json,
    write_json,
    write_jsonl,
)


OUT = ROOT / "tests/gpt5/result/phase413_prediction_kernel_preflight"
P412 = ROOT / "tests/gpt5/result/phase412_typed_quotient_preflight"
SCHEMA_VERSION = "87.0.0"
PHASE_ID = "Phase413-PredictionKernelMeasurementPreflight"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any, length: int = 24) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def as_float_distribution(values: Iterable[Fraction]) -> list[float]:
    return [float(value) for value in values]


def distribution_signature(values: Iterable[Fraction]) -> tuple[str, ...]:
    return tuple(f"{value.numerator}/{value.denominator}" for value in values)


def entropy(values: Iterable[Fraction]) -> float:
    return -sum(float(value) * math.log(float(value)) for value in values if value)


def effective_count(values: Iterable[Fraction]) -> float:
    return math.exp(entropy(values))


def js_divergence(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> float:
    middle = tuple((a + b) / 2 for a, b in zip(left, right, strict=True))

    def kl(source: tuple[Fraction, ...], target: tuple[Fraction, ...]) -> float:
        return sum(
            float(a) * math.log(float(a / b))
            for a, b in zip(source, target, strict=True)
            if a
        )

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def candidate_panel_contract(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    axes = (
        "semantic_valid",
        "grammar_valid",
        "logic_valid",
        "format_valid",
        "stop_valid",
    )
    candidates = [
        {
            "candidate_id": "valid_answer_then_eos",
            "abstract_tokens": ["ANSWER", "EOS"],
            "axis_validity": {axis: True for axis in axes},
        },
        {
            "candidate_id": "semantic_and_logic_error_then_eos",
            "abstract_tokens": ["WRONG_FACT", "EOS"],
            "axis_validity": {
                "semantic_valid": False,
                "grammar_valid": True,
                "logic_valid": False,
                "format_valid": True,
                "stop_valid": True,
            },
        },
        {
            "candidate_id": "grammar_and_format_error_then_eos",
            "abstract_tokens": ["BAD_AGREEMENT", "EOS"],
            "axis_validity": {
                "semantic_valid": True,
                "grammar_valid": False,
                "logic_valid": True,
                "format_valid": False,
                "stop_valid": True,
            },
        },
        {
            "candidate_id": "premature_eos_absorbing",
            "abstract_tokens": ["EOS", "EOS"],
            "axis_validity": {
                "semantic_valid": False,
                "grammar_valid": True,
                "logic_valid": True,
                "format_valid": False,
                "stop_valid": False,
            },
        },
    ]
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "panel_id": "phase413_synthetic_equal_horizon_panel",
            "sequence_event_id": digest(row["abstract_tokens"]),
            "horizon_tokens": 2,
            "eos_absorbing": True,
            "abstract_tokens": row["abstract_tokens"],
            "candidate_id": row["candidate_id"],
            "axis_validity": row["axis_validity"],
            "fully_valid": all(row["axis_validity"].values()),
            "model_token_ids_registered": False,
            "protocol_only": True,
        }
        for row in candidates
    ]
    unique_sequences = len({tuple(row["abstract_tokens"]) for row in rows})
    equal_horizon = len({len(row["abstract_tokens"]) for row in rows}) == 1
    complete_axes = all(set(row["axis_validity"]) == set(axes) for row in rows)
    panel_hash = digest(
        [
            {
                "candidate_id": row["candidate_id"],
                "abstract_tokens": row["abstract_tokens"],
                "axis_validity": row["axis_validity"],
            }
            for row in rows
        ],
        length=32,
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase413-CandidatePanelContract",
        "created_at": created_at,
        "valid": bool(
            unique_sequences == len(rows) and equal_horizon and complete_axes
        ),
        "panel_id": "phase413_synthetic_equal_horizon_panel",
        "panel_hash": panel_hash,
        "candidate_count": len(rows),
        "horizon_tokens": 2,
        "axis_count": len(axes),
        "axes": list(axes),
        "events_pairwise_disjoint": unique_sequences == len(rows),
        "all_candidates_equal_horizon": equal_horizon,
        "eos_absorbing": True,
        "axes_are_independent_labels_not_exclusive_buckets": True,
        "model_token_ids_registered": False,
        "real_model_panel_exhaustive": False,
        "required_real_model_fields": [
            "tokenizer_hash",
            "token_ids",
            "surface",
            "split",
            "raw_sequence_probability",
            "raw_panel_mass",
            "outside_panel_mass",
            "conditional_panel_probability",
            "right_censored",
        ],
        "metric_rules": {
            "raw_panel_mass": "sum_of_pairwise_disjoint_equal_horizon_sequence_probabilities",
            "outside_panel_mass": "one_minus_raw_panel_mass_only_after_disjointness_audit",
            "conditional_panel_probability": "raw_sequence_probability_divided_by_raw_panel_mass",
            "effective_candidate_count": "exp_entropy_of_declared_distribution_with_distribution_scope_recorded",
            "valid_mass": "sum_raw_probability_by_each_validity_axis_and_fully_valid_intersection",
            "js_divergence": "only_same_frozen_panel_same_coordinate_system_and_same_qualified_readout",
        },
        "claim_boundary": (
            "synthetic_contract_only_not_a_registered_model_token_panel_or_language_ontology"
        ),
    }
    return contract, rows


def trajectory_audit(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    initial = (
        Fraction(4, 10),
        Fraction(3, 10),
        Fraction(2, 10),
        Fraction(1, 10),
    )
    terminal = (
        Fraction(7, 10),
        Fraction(1, 10),
        Fraction(1, 10),
        Fraction(1, 10),
    )
    paths: dict[str, tuple[tuple[Fraction, ...], ...]] = {
        "monotonic_narrowing": (
            initial,
            (Fraction(5, 10), Fraction(25, 100), Fraction(15, 100), Fraction(1, 10)),
            (Fraction(6, 10), Fraction(2, 10), Fraction(1, 10), Fraction(1, 10)),
            terminal,
        ),
        "expand_then_narrow": (
            initial,
            (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
            (Fraction(5, 10), Fraction(2, 10), Fraction(2, 10), Fraction(1, 10)),
            terminal,
        ),
        "oscillatory_reallocation": (
            initial,
            (Fraction(6, 10), Fraction(1, 10), Fraction(2, 10), Fraction(1, 10)),
            (Fraction(45, 100), Fraction(25, 100), Fraction(1, 10), Fraction(2, 10)),
            terminal,
        ),
        "late_collapse_after_invalid_rerouting": (
            initial,
            (Fraction(4, 10), Fraction(1, 10), Fraction(3, 10), Fraction(2, 10)),
            (Fraction(4, 10), Fraction(2, 10), Fraction(1, 10), Fraction(3, 10)),
            terminal,
        ),
    }
    rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for path_id, states in paths.items():
        valid_masses = [float(state[0]) for state in states]
        entropies = [entropy(state) for state in states]
        step_js = [
            js_divergence(states[index - 1], state) if index else 0.0
            for index, state in enumerate(states)
        ]
        for index, state in enumerate(states):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "path_id": path_id,
                    "event_index": index,
                    "event_id": ("prefix" if index == 0 else "terminal" if index == len(states) - 1 else f"internal_{index}"),
                    "distribution_fraction": list(distribution_signature(state)),
                    "distribution": as_float_distribution(state),
                    "distribution_sum": float(sum(state)),
                    "fully_valid_mass": valid_masses[index],
                    "entropy_nats": round(entropies[index], 12),
                    "effective_candidate_count": round(effective_count(state), 12),
                    "js_from_previous_event": round(step_js[index], 12),
                    "synthetic": True,
                    "model_derived": False,
                }
            )
        path_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "path_id": path_id,
                "event_count": len(states),
                "initial_signature": list(distribution_signature(states[0])),
                "terminal_signature": list(distribution_signature(states[-1])),
                "internal_signature": [
                    list(distribution_signature(state)) for state in states[1:-1]
                ],
                "valid_mass_monotonic_nondecreasing": all(
                    left <= right
                    for left, right in zip(valid_masses, valid_masses[1:])
                ),
                "entropy_monotonic_nonincreasing": all(
                    left >= right - 1e-12
                    for left, right in zip(entropies, entropies[1:])
                ),
                "contains_entropy_expansion": any(
                    right > left + 1e-12
                    for left, right in zip(entropies, entropies[1:])
                ),
                "contains_valid_mass_reversal": any(
                    right < left - 1e-12
                    for left, right in zip(valid_masses, valid_masses[1:])
                ),
            }
        )

    pair_rows = []
    for left, right in combinations(path_rows, 2):
        pair_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "left_path_id": left["path_id"],
                "right_path_id": right["path_id"],
                "same_initial_distribution": left["initial_signature"]
                == right["initial_signature"],
                "same_terminal_distribution": left["terminal_signature"]
                == right["terminal_signature"],
                "different_internal_trajectory": left["internal_signature"]
                != right["internal_signature"],
            }
        )
    terminal_signatures = {tuple(row["terminal_signature"]) for row in path_rows}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase413-TerminalTrajectoryNonidentifiabilityAudit",
        "created_at": created_at,
        "valid": bool(
            len(terminal_signatures) == 1
            and all(row["different_internal_trajectory"] for row in pair_rows)
        ),
        "candidate_count": 4,
        "synthetic_path_count": len(path_rows),
        "event_count_per_path": 4,
        "path_pair_count": len(pair_rows),
        "same_terminal_distribution_path_count": sum(
            row["terminal_signature"] == path_rows[0]["terminal_signature"]
            for row in path_rows
        ),
        "same_endpoint_different_internal_pair_count": sum(
            row["same_initial_distribution"]
            and row["same_terminal_distribution"]
            and row["different_internal_trajectory"]
            for row in pair_rows
        ),
        "valid_mass_nonmonotonic_path_count": sum(
            not row["valid_mass_monotonic_nondecreasing"] for row in path_rows
        ),
        "entropy_expansion_path_count": sum(
            row["contains_entropy_expansion"] for row in path_rows
        ),
        "terminal_distribution_identifies_internal_trajectory": False,
        "model_case_count": 0,
        "claim_boundary": (
            "finite_constructive_nonidentifiability_counterexample_not_a_claim_about_which_path_a_model_uses"
        ),
    }
    return summary, [*path_rows, *pair_rows], rows


def future_equivalence_audit(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_step = (Fraction(1, 2), Fraction(1, 2))
    continuations = {
        "state_a": {
            "x": (Fraction(9, 10), Fraction(1, 10)),
            "y": (Fraction(1, 2), Fraction(1, 2)),
        },
        "state_b": {
            "x": (Fraction(1, 10), Fraction(9, 10)),
            "y": (Fraction(1, 2), Fraction(1, 2)),
        },
    }
    rows = []
    horizon_two: dict[str, tuple[Fraction, ...]] = {}
    for state_id, next_kernels in continuations.items():
        joint = (
            first_step[0] * next_kernels["x"][0],
            first_step[0] * next_kernels["x"][1],
            first_step[1] * next_kernels["y"][0],
            first_step[1] * next_kernels["y"][1],
        )
        horizon_two[state_id] = joint
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "state_id": state_id,
                "horizon_one_distribution_fraction": list(
                    distribution_signature(first_step)
                ),
                "after_x_kernel_fraction": list(
                    distribution_signature(next_kernels["x"])
                ),
                "after_y_kernel_fraction": list(
                    distribution_signature(next_kernels["y"])
                ),
                "horizon_two_joint_fraction": list(distribution_signature(joint)),
                "horizon_two_joint": as_float_distribution(joint),
            }
        )
    one_step_equal = rows[0]["horizon_one_distribution_fraction"] == rows[1][
        "horizon_one_distribution_fraction"
    ]
    future_equal = horizon_two["state_a"] == horizon_two["state_b"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase413-FuturePredictiveEquivalenceAudit",
        "created_at": created_at,
        "valid": bool(one_step_equal and not future_equal),
        "state_pair_count": 1,
        "one_step_equal_state_pair_count": int(one_step_equal),
        "horizon_two_equal_state_pair_count": int(future_equal),
        "one_step_equal_but_future_different_pair_count": int(
            one_step_equal and not future_equal
        ),
        "horizon_two_js_divergence": round(
            js_divergence(horizon_two["state_a"], horizon_two["state_b"]), 12
        ),
        "one_step_equality_implies_full_future_equivalence": False,
        "full_future_distribution_rule": (
            "autoregressive_product_of_terminal_one_step_kernels_at_every_reachable_extended_prefix"
        ),
        "claim_boundary": "finite_counterexample_not_model_predictive_state_discovery",
    }
    return summary, rows


def matrix_vector(
    matrix: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix)


def channel_permutation_audit(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # h is a post-activation two-channel MLP state and B is its down projection.
    # P swaps channels.  B' = B P^{-1} preserves B h exactly.
    down = (
        (Fraction(1), Fraction(2)),
        (Fraction(3), Fraction(1)),
    )
    permutation = (
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0)),
    )
    down_permuted = tuple(
        tuple(row[index] for index in (1, 0)) for row in down
    )
    states = {
        "case_0": (Fraction(2), Fraction(1)),
        "case_1": (Fraction(1), Fraction(3)),
        "case_2": (Fraction(4), Fraction(1)),
    }
    rows = []
    for case_id, hidden in states.items():
        hidden_permuted = matrix_vector(permutation, hidden)
        natural_output = matrix_vector(down, hidden)
        reparameterized_output = matrix_vector(down_permuted, hidden_permuted)
        fixed_coordinate_probe = hidden_permuted
        transported_probe = matrix_vector(permutation, hidden_permuted)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "case_id": case_id,
                "natural_hidden_fraction": list(distribution_signature(hidden)),
                "permuted_hidden_fraction": list(
                    distribution_signature(hidden_permuted)
                ),
                "natural_output_fraction": list(
                    distribution_signature(natural_output)
                ),
                "reparameterized_output_fraction": list(
                    distribution_signature(reparameterized_output)
                ),
                "fixed_coordinate_probe_fraction": list(
                    distribution_signature(fixed_coordinate_probe)
                ),
                "transported_probe_fraction": list(
                    distribution_signature(transported_probe)
                ),
                "native_output_invariant": natural_output == reparameterized_output,
                "fixed_coordinate_probe_invariant": hidden
                == fixed_coordinate_probe,
                "transported_probe_invariant": hidden == transported_probe,
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase413-ChannelPermutationReadoutAudit",
        "created_at": created_at,
        "valid": bool(
            all(row["native_output_invariant"] for row in rows)
            and all(not row["fixed_coordinate_probe_invariant"] for row in rows)
            and all(row["transported_probe_invariant"] for row in rows)
        ),
        "case_count": len(rows),
        "native_output_invariant_case_count": sum(
            row["native_output_invariant"] for row in rows
        ),
        "fixed_coordinate_probe_invariant_case_count": sum(
            row["fixed_coordinate_probe_invariant"] for row in rows
        ),
        "fixed_coordinate_probe_failure_count": sum(
            not row["fixed_coordinate_probe_invariant"] for row in rows
        ),
        "transported_probe_invariant_case_count": sum(
            row["transported_probe_invariant"] for row in rows
        ),
        "identity_conclusion": (
            "final_output_alone_does_not_identify_mlp_channel_numbering_or_a_fixed_channel_probe"
        ),
        "formula": "o=B_h=(B_P_inverse)(P_h)",
        "claim_boundary": (
            "exact_post_activation_channel_permutation_example_not_arbitrary_residual_basis_symmetry_or_model_feature_discovery"
        ),
    }
    return summary, rows


def readout_qualification(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modes = [
        {
            "readout_id": "terminal_native_next_token_softmax",
            "location": "terminal",
            "kind": "native_probability",
            "probability_semantics": "qualified_method",
            "direct_layer_local_readout": False,
            "model_execution_authorized": False,
            "requirements": ["terminal_logits", "declared_tokenizer", "declared_precision"],
        },
        {
            "readout_id": "terminal_recursive_sequence_kernel",
            "location": "terminal_recursive",
            "kind": "native_probability",
            "probability_semantics": "qualified_method",
            "direct_layer_local_readout": False,
            "model_execution_authorized": False,
            "requirements": ["one_step_kernel_at_each_extended_prefix", "fixed_horizon_or_stop_rule"],
        },
        {
            "readout_id": "raw_intermediate_unembedding_logit_lens",
            "location": "intermediate_residual",
            "kind": "diagnostic_lens",
            "probability_semantics": "not_qualified",
            "direct_layer_local_readout": True,
            "model_execution_authorized": False,
            "requirements": ["basis_and_normalization_disclosed", "do_not_call_model_belief"],
        },
        {
            "readout_id": "final_norm_then_intermediate_unembedding",
            "location": "intermediate_residual",
            "kind": "diagnostic_lens",
            "probability_semantics": "not_qualified",
            "direct_layer_local_readout": True,
            "model_execution_authorized": False,
            "requirements": ["architecture_correct_normalization", "independent_calibration"],
        },
        {
            "readout_id": "tuned_lens_or_subregression_decoder",
            "location": "intermediate_state",
            "kind": "learned_observational_decoder",
            "probability_semantics": "pending",
            "direct_layer_local_readout": True,
            "model_execution_authorized": False,
            "requirements": [
                "frozen_discovery_calibration_holdout_splits",
                "target_leakage_audit",
                "lexical_surface_holdout",
                "calibration_and_outside_panel_mass",
                "channel_permutation_or_transport_audit",
            ],
        },
        {
            "readout_id": "complete_state_native_downstream_replay",
            "location": "intermediate_complete_computational_state",
            "kind": "native_continuation_not_local_lens",
            "probability_semantics": "qualified_in_principle_for_unchanged_complete_state",
            "direct_layer_local_readout": False,
            "model_execution_authorized": False,
            "requirements": [
                "all_residual_positions_and_kv_cache",
                "unchanged_remaining_native_computation",
                "collector_equivalence_gate",
            ],
        },
        {
            "readout_id": "counterfactual_state_downstream_replay",
            "location": "intermediate_counterfactual_complete_state",
            "kind": "causal_intervention",
            "probability_semantics": "causal_only_after_intervention_qualification",
            "direct_layer_local_readout": False,
            "model_execution_authorized": False,
            "requirements": [
                "causal_authorization",
                "valid_computational_edge",
                "same_norm_and_random_controls",
                "side_effect_and_rollout_audit",
            ],
        },
    ]
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            **mode,
        }
        for mode in modes
    ]
    direct_local = [row for row in rows if row["direct_layer_local_readout"]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase413-ReadoutQualification",
        "created_at": created_at,
        "valid": True,
        "readout_mode_count": len(rows),
        "native_terminal_method_count": sum(
            row["kind"] == "native_probability" for row in rows
        ),
        "direct_layer_local_readout_mode_count": len(direct_local),
        "qualified_direct_layer_local_probability_readout_count": sum(
            row["probability_semantics"] == "qualified_method"
            for row in direct_local
        ),
        "diagnostic_direct_layer_local_readout_count": sum(
            row["kind"] == "diagnostic_lens" for row in direct_local
        ),
        "learned_direct_layer_local_readout_pending_count": sum(
            row["kind"] == "learned_observational_decoder" for row in direct_local
        ),
        "complete_state_native_continuation_mode_count": sum(
            row["kind"] == "native_continuation_not_local_lens" for row in rows
        ),
        "causal_intervention_mode_count": sum(
            row["kind"] == "causal_intervention" for row in rows
        ),
        "model_executed_readout_count": 0,
        "intermediate_mu_te_native_without_decoder": False,
        "formula_boundary": {
            "native_terminal": "p_theta(v_given_x)=softmax(z_L(x))_v",
            "native_future": "P_theta(y_1_to_H_given_x)=product_t_p_theta(y_t_given_x_y_less_t)",
            "intermediate": "mu_t_e_requires_declared_decoder_D_e_and_cannot_be_assumed_native",
        },
        "claim_boundary": "measurement_qualification_registry_not_model_evidence",
    }
    return summary, rows


def claim_audit(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    claims = [
        ("phase412_role_transport_correction", "supported_by_current_evidence", "Phase412 finite audit explains 72 of 72 fixed-observer failures by query-role transport."),
        ("role_indexed_bundle_is_external_not_model_state", "supported_by_current_evidence", "The one bundle is an external finite contract and carries no model evidence."),
        ("single_global_progress_percentage_is_invalid", "supported_by_current_evidence", "Protocol, model, physical, causal, and neuron denominators cannot be added."),
        ("autoregression_has_common_terminal_prediction_interface", "supported_by_current_evidence", "Every forward prediction terminates in the model next-token readout."),
        ("every_prefix_has_one_unique_correct_next_token", "incorrect_as_stated", "Natural continuations can be multimodal and the data distribution is not a single label."),
        ("population_cross_entropy_has_unique_true_conditional_optimum", "requires_qualification", "This is an almost-everywhere population statement under a specified data distribution, not a claim that training recovers it."),
        ("fixed_model_prefix_distribution_is_deterministic", "supported_by_current_evidence", "With weights, tokenizer, precision, and execution fixed, the terminal distribution is a deterministic computation."),
        ("future_distribution_is_recursive_product_of_one_step_kernels", "supported_by_current_evidence", "The full future law is induced at all reachable extended prefixes."),
        ("minimal_predictive_state_unique_up_to_renaming", "requires_qualification", "It requires exact future equivalence, a fully specified process, and a proven minimal quotient."),
        ("terminal_prediction_kernel_is_a_conserved_global_invariant", "requires_qualification", "It is a common output interface, not a conserved scalar or proven internal invariant."),
        ("candidate_range_must_shrink_monotonically", "incorrect_as_stated", "Finite paths can expand or reallocate mass before the same terminal decision."),
        ("dynamic_subgraph_comparison_is_a_useful_search_direction", "methodologically_valid_proposal", "Graph-level comparison can be tested after edge and readout qualification."),
        ("every_intermediate_event_has_native_normalized_future_distribution", "incorrect_as_stated", "The architecture has no native local probability head at every event."),
        ("raw_logit_lens_is_the_model_intermediate_belief", "incorrect_as_stated", "It is a coordinate- and normalization-dependent diagnostic readout."),
        ("observational_similarity_edge_is_actual_computation_edge", "incorrect_as_stated", "A computational edge requires graph legality and intervention or equivalent causal evidence."),
        ("neurons_are_reusable_basis_functions", "methodologically_valid_proposal", "This is a hypothesis to test, not a result established by current data."),
        ("freeze_disjoint_equal_horizon_multiaxis_candidate_panels", "methodologically_valid_proposal", "This makes panel mass and validity metrics auditable."),
        ("run_three_cuda_models_before_external_gates", "incorrect_as_stated", "Human review is 0/2 and sealed collector equivalence is 0/165."),
    ]
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "claim_id": claim_id,
            "status": status,
            "reason": reason,
        }
        for claim_id, status, reason in claims
    ]
    counts = Counter(row["status"] for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase413-SourceClaimAudit",
        "created_at": created_at,
        "valid": True,
        "claim_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "overall_assessment": (
            "direction_partly_correct_but_intermediate_probability_and_progress_claims_require_strict_correction"
        ),
        "claim_boundary": "audit_of_supplied_text_against_current_artifacts_and_finite_counterexamples",
    }
    return summary, rows


def main() -> None:
    created_at = now()
    phase412 = read_json(P412 / "phase412_stage_summary.json")
    panel, candidate_rows = candidate_panel_contract(created_at)
    trajectories, trajectory_rows, trajectory_step_rows = trajectory_audit(created_at)
    future, future_rows = future_equivalence_audit(created_at)
    channel, channel_rows = channel_permutation_audit(created_at)
    readouts, readout_rows = readout_qualification(created_at)
    claims, claim_rows = claim_audit(created_at)

    machine_preflight = all(
        artifact["valid"]
        for artifact in (panel, trajectories, future, channel, readouts, claims)
    )
    completed_reviewers = phase412["results"]["completed_external_reviewer_count"]
    required_reviewers = phase412["denominators"][
        "required_independent_reviewer_count"
    ]
    collector_cases = phase412["results"][
        "sealed_model_collector_equivalence_case_count"
    ]
    collector_denominator = phase412["denominators"][
        "future_sealed_model_collector_case_count"
    ]
    model_authorized = bool(
        machine_preflight
        and completed_reviewers == required_reviewers
        and collector_cases == collector_denominator
    )
    qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase413-ExecutionQualification",
        "created_at": created_at,
        "machine_preflight_pass": machine_preflight,
        "candidate_panel_contract_pass": panel["valid"],
        "terminal_nonidentifiability_counterexample_pass": trajectories["valid"],
        "future_equivalence_counterexample_pass": future["valid"],
        "channel_permutation_counterexample_pass": channel["valid"],
        "readout_registry_pass": readouts["valid"],
        "independent_human_rule_review_completed": completed_reviewers
        == required_reviewers,
        "sealed_model_collector_equivalence_completed": collector_cases
        == collector_denominator,
        "model_qualification_authorized": model_authorized,
        "formal_model_discovery_authorized": False,
        "descriptive_physical_mapping_authorized": False,
        "causal_intervention_authorized": False,
        "neuron_scan_authorized": False,
    }
    stage = {
        "schema_version": "87.1.0",
        "phase_id": "Phase413-PredictionKernelMeasurementPreflightStage",
        "created_at": created_at,
        "objective": (
            "separate_native_terminal_prediction_from_unqualified_intermediate_readouts_and_prove_what_terminal_behavior_cannot_identify_before_candidate_trajectory_collection"
        ),
        "assessment": {
            "supplied_phase412_audit_direction_correct": True,
            "supplied_global_invariant_argument_partly_correct": True,
            "supplied_dynamic_candidate_graph_direction_plausible": True,
            "intermediate_native_probability_assumption_valid": False,
            "terminal_prediction_kernel_is_conserved_scalar_invariant": False,
            "terminal_distribution_identifies_internal_trajectory": False,
            "one_step_equality_implies_full_future_equivalence": False,
            "raw_logit_lens_is_model_belief": False,
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
            "source_claim_count": claims["claim_count"],
            "readout_mode_count": readouts["readout_mode_count"],
            "direct_layer_local_readout_mode_count": readouts[
                "direct_layer_local_readout_mode_count"
            ],
            "synthetic_path_count": trajectories["synthetic_path_count"],
            "trajectory_event_count": len(trajectory_step_rows),
            "synthetic_path_pair_count": trajectories["path_pair_count"],
            "future_state_pair_count": future["state_pair_count"],
            "channel_permutation_case_count": channel["case_count"],
            "candidate_panel_case_count": panel["candidate_count"],
            "required_independent_reviewer_count": required_reviewers,
            "future_sealed_model_collector_case_count": collector_denominator,
            "model_case_count_consumed": 0,
            "physical_case_count_consumed": 0,
        },
        "results": {
            "supported_claim_count": claims["status_counts"].get(
                "supported_by_current_evidence", 0
            ),
            "methodological_proposal_count": claims["status_counts"].get(
                "methodologically_valid_proposal", 0
            ),
            "qualification_required_claim_count": claims["status_counts"].get(
                "requires_qualification", 0
            ),
            "incorrect_as_stated_claim_count": claims["status_counts"].get(
                "incorrect_as_stated", 0
            ),
            "native_terminal_method_count": readouts[
                "native_terminal_method_count"
            ],
            "qualified_direct_layer_local_probability_readout_count": readouts[
                "qualified_direct_layer_local_probability_readout_count"
            ],
            "diagnostic_direct_layer_local_readout_count": readouts[
                "diagnostic_direct_layer_local_readout_count"
            ],
            "same_terminal_distribution_path_count": trajectories[
                "same_terminal_distribution_path_count"
            ],
            "same_endpoint_different_internal_pair_count": trajectories[
                "same_endpoint_different_internal_pair_count"
            ],
            "valid_mass_nonmonotonic_path_count": trajectories[
                "valid_mass_nonmonotonic_path_count"
            ],
            "entropy_expansion_path_count": trajectories[
                "entropy_expansion_path_count"
            ],
            "one_step_equal_but_future_different_pair_count": future[
                "one_step_equal_but_future_different_pair_count"
            ],
            "native_output_invariant_channel_case_count": channel[
                "native_output_invariant_case_count"
            ],
            "fixed_coordinate_probe_failure_count": channel[
                "fixed_coordinate_probe_failure_count"
            ],
            "transported_probe_invariant_case_count": channel[
                "transported_probe_invariant_case_count"
            ],
            "completed_external_reviewer_count": completed_reviewers,
            "sealed_model_collector_equivalence_case_count": collector_cases,
            "new_behavioral_result_count": 0,
            "new_physical_path_count": 0,
            "new_causal_path_count": 0,
            "new_neuron_path_count": 0,
        },
        "hard_limits": [
            "all_positive_results_are_external_protocol_or_finite_counterexamples",
            "no_model_was_loaded_and_no_candidate_trajectory_was_measured",
            "the_terminal_kernel_is_a_common_output_interface_not_a_conserved_internal_scalar",
            "a_direct_layer_local_probability_is_not_natively_defined_without_a_declared_decoder",
            "raw_and_normalized_logit_lenses_remain_diagnostic_until_independent_qualification",
            "terminally_identical_computations_can_have_distinct_internal_trajectories",
            "one_step_distribution_equality_does_not_establish_full_future_equivalence",
            "channel_permutation_example_does_not_prove_arbitrary_transformer_basis_symmetry",
            "the_candidate_panel_uses_abstract_symbols_and_is_not_a_model_token_registry",
            "two_external_human_reviews_are_still_absent",
            "sealed_real_model_collector_equivalence_remains_zero_of_165",
            "single_global_progress_percentage_is_invalid",
        ],
        "authorization": {
            "publish_measurement_preflight": True,
            "call_terminal_kernel_a_global_conserved_invariant": False,
            "publish_raw_logit_lens_as_model_belief": False,
            "publish_synthetic_paths_as_model_physical_paths": False,
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
                "two_distinct_external_reviewers_complete_all_65_items_with_confidence_and_reasons",
                "reference_and_incremental_collectors_match_all_165_sealed_real_model_cases",
                "real_candidate_panels_register_token_ids_equal_horizon_axes_and_outside_mass_before_collection",
                "intermediate_readout_is_declared_as_diagnostic_learned_or_complete_state_native_continuation",
            ],
            "first_model_measurement_after_gate": (
                "terminal_native_kernel_baseline_and_complete_state_downstream_replay_not_raw_lens_probability"
            ),
            "model_order_after_gate": list(MODELS),
        },
        "single_global_progress_percentage_valid": False,
    }

    write_json(OUT / "phase413_source_claim_audit.json", claims)
    write_jsonl(OUT / "protocol/private/phase413_source_claim_rows.jsonl", claim_rows)
    write_json(OUT / "phase413_candidate_panel_contract.json", panel)
    write_jsonl(
        OUT / "protocol/private/phase413_candidate_panel_rows.jsonl", candidate_rows
    )
    write_json(OUT / "phase413_terminal_nonidentifiability_audit.json", trajectories)
    write_jsonl(
        OUT / "protocol/private/phase413_trajectory_rows.jsonl", trajectory_rows
    )
    write_jsonl(
        OUT / "protocol/private/phase413_trajectory_step_rows.jsonl",
        trajectory_step_rows,
    )
    write_json(OUT / "phase413_future_equivalence_audit.json", future)
    write_jsonl(
        OUT / "protocol/private/phase413_future_equivalence_rows.jsonl",
        future_rows,
    )
    write_json(OUT / "phase413_channel_permutation_audit.json", channel)
    write_jsonl(
        OUT / "protocol/private/phase413_channel_permutation_rows.jsonl",
        channel_rows,
    )
    write_json(OUT / "phase413_readout_qualification.json", readouts)
    write_jsonl(
        OUT / "protocol/private/phase413_readout_registry.jsonl", readout_rows
    )
    write_json(OUT / "phase413_execution_qualification.json", qualification)
    write_json(OUT / "phase413_stage_summary.json", stage)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
