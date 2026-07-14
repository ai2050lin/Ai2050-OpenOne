#!/usr/bin/env python3
"""Audit typed observer covariance and nontrivial finite quotients.

Phase411 tested every registered state operation against a fixed observer.  A
global entity renaming can also rename the queried entity, so a fixed-observer
failure is not by itself a failure of the role-conditioned observation.  This
stage makes the observer action explicit, enumerates every partition of each
finite state universe, and separates four questions:

1. Is a partition nontrivial?
2. Is it congruent under every registered state operation?
3. Is it sufficient for a frozen observer and its registered futures?
4. Does it remain sufficient when the observer role transforms with the state?

The audit concerns external finite protocol objects only.  It does not load a
model or establish a model state, physical path, causal mechanism, or neuron.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_protocol import STATE_IDS  # noqa: E402
from phase409_dynamic_response_protocol import FAMILIES, MODELS  # noqa: E402
from phase411_finite_operation_preflight import (  # noqa: E402
    OUT as PHASE411_OUT,
    apply_operation,
    canonical_json,
    observation_partitions,
    operation_registry,
    operation_signature,
    read_json,
    write_json,
    write_jsonl,
)


OUT = ROOT / "tests/gpt5/result/phase412_typed_quotient_preflight"
SCHEMA_VERSION = "86.0.0"
PHASE_ID = "Phase412-TypedObserverQuotientPreflight"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def canonical_partition(blocks: Iterable[Iterable[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            (tuple(sorted(block)) for block in blocks),
            key=lambda block: block[0],
        )
    )


def set_partitions(items: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], ...]]:
    """Yield each set partition once in canonical order."""

    if not items:
        yield ()
        return
    first = items[0]
    for tail in set_partitions(items[1:]):
        yield canonical_partition(((first,), *tail))
        for index in range(len(tail)):
            updated = list(tail)
            updated[index] = tuple(sorted((*updated[index], first)))
            yield canonical_partition(updated)


def all_partitions(items: tuple[str, ...]) -> list[tuple[tuple[str, ...], ...]]:
    return sorted(
        set(set_partitions(items)),
        key=lambda partition: (len(partition), partition),
    )


def induced_partition(
    states: tuple[str, ...], signatures: dict[str, tuple[str, ...]]
) -> tuple[tuple[str, ...], ...]:
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for state_id in states:
        groups[signatures[state_id]].append(state_id)
    return canonical_partition(groups.values())


def block_index(partition: tuple[tuple[str, ...], ...]) -> dict[str, int]:
    return {
        state_id: index
        for index, block in enumerate(partition)
        for state_id in block
    }


def nontrivial(partition: tuple[tuple[str, ...], ...], state_count: int) -> bool:
    return 1 < len(partition) < state_count


def partition_congruent(
    family: str,
    partition: tuple[tuple[str, ...], ...],
    operations: Iterable[dict[str, Any]],
) -> bool:
    index = block_index(partition)
    for operation in operations:
        for block in partition:
            target_blocks = {
                index[apply_operation(family, operation, state_id)]
                for state_id in block
            }
            if len(target_blocks) != 1:
                return False
    return True


def observation_sufficient(
    partition: tuple[tuple[str, ...], ...],
    signatures: dict[str, tuple[str, ...]],
) -> bool:
    return all(
        len({signatures[state_id] for state_id in block}) == 1
        for block in partition
    )


def future_observation_sufficient(
    family: str,
    partition: tuple[tuple[str, ...], ...],
    operations: Iterable[dict[str, Any]],
    source_observer_id: str,
    observers: dict[str, dict[str, tuple[str, ...]]],
    *,
    transform_observer: bool,
) -> bool:
    for operation in operations:
        target_observer_id = (
            observer_transform(family, source_observer_id, operation)
            if transform_observer
            else source_observer_id
        )
        target_signatures = observers[target_observer_id]
        for block in partition:
            outcomes = {
                target_signatures[apply_operation(family, operation, state_id)]
                for state_id in block
            }
            if len(outcomes) != 1:
                return False
    return True


def observer_transform(
    family: str, observer_id: str, operation: dict[str, Any]
) -> str:
    """Transport a query role with the registered state relabeling.

    For knowledge states, the transformed state at new entity i reads from old
    entity entity_permutation[i].  Therefore an old query role j moves to the
    inverse-permutation location i.
    """

    prefix = "single_entity_value:entity_"
    if family != "knowledge_binding" or not observer_id.startswith(prefix):
        return observer_id
    source_entity = int(observer_id.removeprefix(prefix))
    target_entity = tuple(operation["entity_permutation"]).index(source_entity)
    return f"{prefix}{target_entity}"


def relation_violation_count(
    family: str,
    operation: dict[str, Any],
    source_signatures: dict[str, tuple[str, ...]],
    target_signatures: dict[str, tuple[str, ...]],
) -> int:
    violations = 0
    for state_a, state_b in combinations(STATE_IDS[family], 2):
        source_equal = source_signatures[state_a] == source_signatures[state_b]
        target_equal = target_signatures[
            apply_operation(family, operation, state_a)
        ] == target_signatures[apply_operation(family, operation, state_b)]
        if source_equal != target_equal:
            violations += 1
    return violations


def response_class_map(
    family: str,
    operation: dict[str, Any],
    source_signatures: dict[str, tuple[str, ...]],
    target_signatures: dict[str, tuple[str, ...]],
) -> tuple[bool, list[dict[str, Any]]]:
    mapped: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for state_id in STATE_IDS[family]:
        mapped[source_signatures[state_id]].add(
            target_signatures[apply_operation(family, operation, state_id)]
        )
    rows = [
        {
            "source_signature": list(source),
            "target_signatures": [list(target) for target in sorted(targets)],
        }
        for source, targets in sorted(mapped.items())
    ]
    return all(len(targets) == 1 for targets in mapped.values()), rows


def observer_covariance_audit(
    created_at: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    registry = operation_registry()
    covariance_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    role_moved_count = 0

    for family in FAMILIES:
        observers = dict(observation_partitions(family))
        operations = registry[family]
        id_by_signature = {
            operation_signature(family, operation): operation
            for operation in operations
        }
        for observer_id, source_signatures in observers.items():
            for operation in operations:
                target_observer_id = observer_transform(
                    family, observer_id, operation
                )
                target_signatures = observers[target_observer_id]
                fixed_violations = relation_violation_count(
                    family,
                    operation,
                    source_signatures,
                    source_signatures,
                )
                typed_violations = relation_violation_count(
                    family,
                    operation,
                    source_signatures,
                    target_signatures,
                )
                fixed_map_valid, fixed_map = response_class_map(
                    family,
                    operation,
                    source_signatures,
                    source_signatures,
                )
                typed_map_valid, typed_map = response_class_map(
                    family,
                    operation,
                    source_signatures,
                    target_signatures,
                )
                role_moved = target_observer_id != observer_id
                role_moved_count += role_moved
                covariance_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "family_id": family,
                        "observer_id": observer_id,
                        "operation_id": operation["operation_id"],
                        "target_observer_id": target_observer_id,
                        "joint_observer": observer_id
                        == "joint_all_registered_queries",
                        "role_moved": role_moved,
                        "fixed_observer_relation_violation_count": fixed_violations,
                        "fixed_observer_stable": fixed_violations == 0,
                        "fixed_response_class_map_valid": fixed_map_valid,
                        "fixed_response_class_map": fixed_map,
                        "typed_relation_violation_count": typed_violations,
                        "typed_observer_covariant": typed_violations == 0,
                        "typed_response_class_map_valid": typed_map_valid,
                        "typed_response_class_map": typed_map,
                    }
                )

        for first in operations:
            for second in operations:
                composite_signature = tuple(
                    apply_operation(
                        family,
                        second,
                        apply_operation(family, first, state_id),
                    )
                    for state_id in STATE_IDS[family]
                )
                composite = id_by_signature[composite_signature]
                for observer_id in observers:
                    sequential = observer_transform(
                        family,
                        observer_transform(family, observer_id, first),
                        second,
                    )
                    direct = observer_transform(
                        family, observer_id, composite
                    )
                    action_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE_ID,
                            "family_id": family,
                            "observer_id": observer_id,
                            "first_operation_id": first["operation_id"],
                            "second_operation_id": second["operation_id"],
                            "composite_operation_id": composite["operation_id"],
                            "sequential_target_observer_id": sequential,
                            "direct_target_observer_id": direct,
                            "valid": sequential == direct,
                        }
                    )

    query_rows = [row for row in covariance_rows if not row["joint_observer"]]
    fixed_unstable = [row for row in query_rows if not row["fixed_observer_stable"]]
    typed_unstable = [row for row in query_rows if not row["typed_observer_covariant"]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase412-TypedObserverCovarianceAudit",
        "created_at": created_at,
        "valid": not typed_unstable
        and all(row["typed_response_class_map_valid"] for row in query_rows)
        and all(row["valid"] for row in action_rows),
        "registered_query_observer_operation_cell_count": len(query_rows),
        "synthetic_joint_observer_operation_cell_count": len(covariance_rows)
        - len(query_rows),
        "fixed_observer_unstable_cell_count": len(fixed_unstable),
        "fixed_response_class_map_failure_cell_count": sum(
            not row["fixed_response_class_map_valid"] for row in query_rows
        ),
        "role_moved_cell_count": sum(row["role_moved"] for row in query_rows),
        "fixed_instability_explained_by_role_transport_count": sum(
            (not row["fixed_observer_stable"])
            and row["role_moved"]
            and row["typed_observer_covariant"]
            for row in query_rows
        ),
        "typed_observer_unstable_cell_count": len(typed_unstable),
        "typed_response_class_map_failure_cell_count": sum(
            not row["typed_response_class_map_valid"] for row in query_rows
        ),
        "observer_action_composition_case_count": len(action_rows),
        "observer_action_composition_failure_count": sum(
            not row["valid"] for row in action_rows
        ),
        "phase411_fixed_observer_counterexample_count_reproduced": len(
            fixed_unstable
        ),
        "phase411_counterexample_interpretation": (
            "fixed_query_roles_do_not_commute_with_entity_relabeling;_the_"
            "counterexamples_do_not_by_themselves_refute_role_conditioned_states"
        ),
        "model_observer_covariance_tested": False,
        "claim_boundary": (
            "external_finite_state_observer_response_covariance_not_model_behavior"
        ),
    }
    return summary, covariance_rows, action_rows, registry


def quotient_audit(
    created_at: str,
    registry: dict[str, list[dict[str, Any]]],
    covariance_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    partition_rows: list[dict[str, Any]] = []
    observer_rows: list[dict[str, Any]] = []
    induced_rows: list[dict[str, Any]] = []

    covariance_by_cell = {
        (row["family_id"], row["observer_id"], row["operation_id"]): row
        for row in covariance_rows
    }

    for family in FAMILIES:
        states = tuple(STATE_IDS[family])
        operations = registry[family]
        observers = dict(observation_partitions(family))
        joint = observers["joint_all_registered_queries"]
        partitions = all_partitions(states)
        for partition in partitions:
            is_nontrivial = nontrivial(partition, len(states))
            full_congruent = partition_congruent(family, partition, operations)
            joint_sufficient = observation_sufficient(partition, joint)
            joint_future_sufficient = future_observation_sufficient(
                family,
                partition,
                operations,
                "joint_all_registered_queries",
                observers,
                transform_observer=False,
            )
            partition_key = digest([family, partition])
            partition_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "partition_id": f"{family}__{partition_key}",
                    "family_id": family,
                    "state_count": len(states),
                    "block_count": len(partition),
                    "blocks": [list(block) for block in partition],
                    "nontrivial": is_nontrivial,
                    "full_operation_congruent": full_congruent,
                    "joint_observation_sufficient": joint_sufficient,
                    "joint_future_sufficient": joint_future_sufficient,
                    "global_nontrivial_qualifying": is_nontrivial
                    and full_congruent
                    and joint_sufficient
                    and joint_future_sufficient,
                }
            )
            for observer_id, signatures in observers.items():
                if observer_id == "joint_all_registered_queries":
                    continue
                stabilizer = [
                    operation
                    for operation in operations
                    if observer_transform(family, observer_id, operation)
                    == observer_id
                ]
                zero_sufficient = observation_sufficient(partition, signatures)
                fixed_future = future_observation_sufficient(
                    family,
                    partition,
                    operations,
                    observer_id,
                    observers,
                    transform_observer=False,
                )
                covariant_future = future_observation_sufficient(
                    family,
                    partition,
                    operations,
                    observer_id,
                    observers,
                    transform_observer=True,
                )
                stabilizer_congruent = partition_congruent(
                    family, partition, stabilizer
                )
                stabilizer_future = future_observation_sufficient(
                    family,
                    partition,
                    stabilizer,
                    observer_id,
                    observers,
                    transform_observer=False,
                )
                observer_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "partition_id": f"{family}__{partition_key}",
                        "family_id": family,
                        "observer_id": observer_id,
                        "nontrivial": is_nontrivial,
                        "zero_horizon_observation_sufficient": zero_sufficient,
                        "full_operation_congruent": full_congruent,
                        "fixed_observer_future_sufficient": fixed_future,
                        "role_covariant_future_sufficient": covariant_future,
                        "observer_stabilizer_operation_count": len(stabilizer),
                        "observer_stabilizer_congruent": stabilizer_congruent,
                        "observer_stabilizer_future_sufficient": stabilizer_future,
                        "fixed_observer_nontrivial_qualifying": is_nontrivial
                        and zero_sufficient
                        and full_congruent
                        and fixed_future,
                        "role_stabilizer_nontrivial_qualifying": is_nontrivial
                        and zero_sufficient
                        and stabilizer_congruent
                        and stabilizer_future,
                    }
                )

        for observer_id, signatures in observers.items():
            if observer_id == "joint_all_registered_queries":
                continue
            partition = induced_partition(states, signatures)
            stabilizer = [
                operation
                for operation in operations
                if observer_transform(family, observer_id, operation) == observer_id
            ]
            all_covariant = all(
                covariance_by_cell[(family, observer_id, operation["operation_id"])][
                    "typed_observer_covariant"
                ]
                for operation in operations
            )
            all_maps_valid = all(
                covariance_by_cell[(family, observer_id, operation["operation_id"])][
                    "typed_response_class_map_valid"
                ]
                for operation in operations
            )
            induced_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "family_id": family,
                    "observer_id": observer_id,
                    "partition_id": f"{family}__{digest([family, partition])}",
                    "blocks": [list(block) for block in partition],
                    "state_count": len(states),
                    "block_count": len(partition),
                    "class_sizes": sorted(len(block) for block in partition),
                    "nontrivial": nontrivial(partition, len(states)),
                    "observer_stabilizer_operation_count": len(stabilizer),
                    "stabilizer_congruent": partition_congruent(
                        family, partition, stabilizer
                    ),
                    "all_registered_operations_role_covariant": all_covariant,
                    "all_response_class_maps_valid": all_maps_valid,
                    "external_role_conditioned_quotient": nontrivial(
                        partition, len(states)
                    )
                    and partition_congruent(family, partition, stabilizer)
                    and all_covariant
                    and all_maps_valid,
                    "model_derived": False,
                }
            )

    bundle_observers = {
        "single_entity_value:entity_0",
        "single_entity_value:entity_1",
        "single_entity_value:entity_2",
    }
    bundle_fibers = [
        row
        for row in induced_rows
        if row["family_id"] == "knowledge_binding"
        and row["observer_id"] in bundle_observers
    ]
    bundle_closed = all(
        observer_transform("knowledge_binding", observer_id, operation)
        in bundle_observers
        for observer_id in bundle_observers
        for operation in registry["knowledge_binding"]
    )
    bundles = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "bundle_id": "knowledge_single_entity_role_indexed_partition_bundle",
            "family_id": "knowledge_binding",
            "observer_ids": sorted(bundle_observers),
            "fiber_count": len(bundle_fibers),
            "blocks_per_fiber": sorted(
                {row["block_count"] for row in bundle_fibers}
            ),
            "class_sizes_per_fiber": sorted(
                {tuple(row["class_sizes"]) for row in bundle_fibers}
            ),
            "closed_under_observer_transport": bundle_closed,
            "all_fibers_external_role_conditioned_quotients": all(
                row["external_role_conditioned_quotient"]
                for row in bundle_fibers
            ),
            "exact_for_all_finite_registered_operation_words": bundle_closed
            and all(
                row["all_registered_operations_role_covariant"]
                for row in bundle_fibers
            ),
            "global_state_quotient": False,
            "model_derived": False,
            "claim_boundary": (
                "external_role_indexed_observation_bundle_not_global_or_model_state_quotient"
            ),
        }
    ]

    nontrivial_rows = [row for row in partition_rows if row["nontrivial"]]
    query_induced = [
        row for row in induced_rows if row["external_role_conditioned_quotient"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase412-NontrivialFiniteQuotientAudit",
        "created_at": created_at,
        "valid": True,
        "partition_count": len(partition_rows),
        "partition_count_by_family": dict(
            sorted(Counter(row["family_id"] for row in partition_rows).items())
        ),
        "nontrivial_partition_count": len(nontrivial_rows),
        "full_operation_congruent_nontrivial_partition_count": sum(
            row["full_operation_congruent"] for row in nontrivial_rows
        ),
        "joint_observation_sufficient_nontrivial_partition_count": sum(
            row["joint_observation_sufficient"] for row in nontrivial_rows
        ),
        "global_nontrivial_qualifying_partition_count": sum(
            row["global_nontrivial_qualifying"] for row in nontrivial_rows
        ),
        "observer_partition_evaluation_count": len(observer_rows),
        "fixed_observer_nontrivial_qualifying_partition_count": sum(
            row["fixed_observer_nontrivial_qualifying"] for row in observer_rows
        ),
        "role_stabilizer_nontrivial_qualifying_partition_count": sum(
            row["role_stabilizer_nontrivial_qualifying"] for row in observer_rows
        ),
        "induced_query_partition_count": len(induced_rows),
        "external_role_conditioned_quotient_count": len(query_induced),
        "external_role_indexed_partition_bundle_count": sum(
            row["all_fibers_external_role_conditioned_quotients"]
            and row["closed_under_observer_transport"]
            for row in bundles
        ),
        "model_derived_nontrivial_predictive_quotient_count": 0,
        "finite_partition_space_exhausted": True,
        "future_word_argument": (
            "one_step_covariance_plus_registered_operation_closure_plus_"
            "observer_action_composition_implies_all_finite_registered_words"
        ),
        "claim_boundary": (
            "exhaustive_external_finite_partition_audit_not_model_predictive_state_discovery"
        ),
    }
    return summary, partition_rows, observer_rows, [*induced_rows, *bundles]


def irreversible_operation_readiness(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proposals = [
        (
            "add_fact",
            "knowledge_binding_state",
            "knowledge_binding_state",
            "current_bijection_only_universe_has_no_partial_or_duplicate_assignment_state",
            "partial_multi_fact_knowledge_universe",
        ),
        (
            "delete_fact",
            "knowledge_binding_state",
            "partial_knowledge_state",
            "current_universe_has_no_unknown_or_missing_value_state",
            "explicit_unknown_and_missing_fact_states",
        ),
        (
            "override_current",
            "history_augmented_state",
            "knowledge_binding_state",
            "current_history_solver_is_not_registered_as_a_unary_state_operation",
            "typed_current_prior_history_state_and_reviewed_override_rule",
        ),
        (
            "merge_conflict",
            "conflicting_knowledge_state",
            "resolved_knowledge_state",
            "current_universe_cannot_represent_simultaneous_conflicting_values",
            "conflict_state_and_resolution_policy",
        ),
        (
            "infer_relation",
            "relation_graph_state",
            "relation_graph_state",
            "holder_only_rule_universe_has_no_relation_graph_or_proof_state",
            "finite_relation_graph_and_proof_contract",
        ),
        (
            "abstract_classification",
            "entity_property_state",
            "class_state",
            "current_universes_have_no_reviewed_type_ontology",
            "finite_entity_property_classification_contract",
        ),
        (
            "stop_or_complete",
            "generation_state",
            "terminal_generation_state",
            "grammar_feature_state_is_not_a_generation_or_terminal_state",
            "typed_generation_history_sentence_boundary_and_stop_state",
        ),
    ]
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "operation_id": operation_id,
            "source_state_type": source_type,
            "target_state_type": target_type,
            "registered_executable": False,
            "closed_on_current_finite_universe": False,
            "external_semantic_review_completed": False,
            "blocking_reason": reason,
            "required_contract_extension": extension,
        }
        for operation_id, source_type, target_type, reason, extension in proposals
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase412-IrreversibleOperationReadinessAudit",
        "created_at": created_at,
        "valid": True,
        "proposed_irreversible_operation_count": len(rows),
        "registered_executable_irreversible_operation_count": sum(
            row["registered_executable"] for row in rows
        ),
        "current_universe_closed_irreversible_operation_count": sum(
            row["closed_on_current_finite_universe"] for row in rows
        ),
        "externally_reviewed_irreversible_operation_count": sum(
            row["external_semantic_review_completed"] for row in rows
        ),
        "registration_authorized": False,
        "reason": (
            "registering_these_operations_now_would_silently_invent_missing_state_types"
        ),
        "claim_boundary": "readiness_and_missing_universe_audit_only",
    }
    return summary, rows


def typed_composition_readiness(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proposals = [
        (
            "retrieve_to_binding",
            "knowledge_state",
            "binding_state",
            "no_separate_reviewed_knowledge_and_binding_state_registries",
        ),
        (
            "binding_to_reasoning",
            "binding_state",
            "reasoning_state",
            "no_reviewed_bridge_map_between_knowledge_binding_and_rule_universes",
        ),
        (
            "reasoning_to_grammar",
            "reasoning_output_state",
            "grammar_constraint_state",
            "no_reviewed_semantic_to_grammar_interface_state",
        ),
        (
            "grammar_to_expression",
            "grammar_constraint_state",
            "terminal_generation_state",
            "no_registered_generation_or_terminal_state_universe",
        ),
    ]
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "bridge_id": bridge_id,
            "source_state_type": source_type,
            "target_state_type": target_type,
            "registered_executable": False,
            "semantic_review_completed": False,
            "blocking_reason": reason,
        }
        for bridge_id, source_type, target_type, reason in proposals
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase412-TypedCompositionReadinessAudit",
        "created_at": created_at,
        "valid": True,
        "registered_intra_family_endomorphism_count": sum(
            len(operations) for operations in operation_registry().values()
        ),
        "proposed_cross_family_bridge_count": len(rows),
        "registered_executable_cross_family_bridge_count": sum(
            row["registered_executable"] for row in rows
        ),
        "valid_cross_family_composition_count": 0,
        "cross_family_operation_system_established": False,
        "claim_boundary": "typed_bridge_readiness_audit_not_cross_family_algebra",
    }
    return summary, rows


def main() -> None:
    created_at = now()
    phase411_stage = read_json(PHASE411_OUT / "phase411_stage_summary.json")
    phase411_review = read_json(
        PHASE411_OUT / "phase411_external_review_v2_status.json"
    )
    phase411_qualification = read_json(
        PHASE411_OUT / "phase411_qualification.json"
    )

    covariance, covariance_rows, action_rows, registry = observer_covariance_audit(
        created_at
    )
    quotient, partition_rows, observer_rows, induced_rows = quotient_audit(
        created_at, registry, covariance_rows
    )
    irreversible, irreversible_rows = irreversible_operation_readiness(created_at)
    typed_composition, bridge_rows = typed_composition_readiness(created_at)

    machine_preflight = bool(
        phase411_stage["assessment"]["machine_preflight_pass"]
        and covariance["valid"]
        and quotient["valid"]
        and irreversible["valid"]
        and typed_composition["valid"]
    )
    external_review = phase411_review[
        "independent_human_rule_review_completed"
    ]
    collector_equivalence = phase411_qualification[
        "sealed_model_collector_equivalence_completed"
    ]
    model_qualification_authorized = bool(
        machine_preflight and external_review and collector_equivalence
    )

    qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase412-Qualification",
        "created_at": created_at,
        "machine_preflight_pass": machine_preflight,
        "typed_observer_covariance_pass": covariance["valid"],
        "finite_partition_space_exhausted": quotient[
            "finite_partition_space_exhausted"
        ],
        "global_external_nontrivial_quotient_found": quotient[
            "global_nontrivial_qualifying_partition_count"
        ]
        > 0,
        "external_role_indexed_partition_bundle_found": quotient[
            "external_role_indexed_partition_bundle_count"
        ]
        > 0,
        "model_derived_nontrivial_predictive_quotient_found": False,
        "irreversible_operation_registration_authorized": irreversible[
            "registration_authorized"
        ],
        "cross_family_operation_system_established": typed_composition[
            "cross_family_operation_system_established"
        ],
        "independent_human_rule_review_completed": external_review,
        "sealed_model_collector_equivalence_completed": collector_equivalence,
        "model_qualification_authorized": model_qualification_authorized,
        "formal_model_discovery_authorized": False,
        "descriptive_physical_mapping_authorized": False,
        "causal_intervention_authorized": False,
        "neuron_scan_authorized": False,
    }

    stage = {
        "schema_version": "86.1.0",
        "phase_id": "Phase412-TypedObserverQuotientPreflightStage",
        "created_at": created_at,
        "objective": (
            "separate_fixed_observer_failures_from_typed_role_covariance_and_"
            "exhaust_all_external_finite_nontrivial_quotients_before_model_execution"
        ),
        "assessment": {
            "phase411_direction_correct": True,
            "phase411_fixed_observer_interpretation_requires_correction": True,
            "phase411_fixed_observer_counterexamples_reproduced": covariance[
                "phase411_fixed_observer_counterexample_count_reproduced"
            ],
            "fixed_counterexamples_refute_all_role_conditioned_states": False,
            "typed_observer_covariance_pass": covariance["valid"],
            "global_external_nontrivial_quotient_found": quotient[
                "global_nontrivial_qualifying_partition_count"
            ]
            > 0,
            "external_role_indexed_partition_bundle_found": quotient[
                "external_role_indexed_partition_bundle_count"
            ]
            > 0,
            "external_bundle_is_model_state_evidence": False,
            "irreversible_operation_system_established": False,
            "cross_family_operation_system_established": False,
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
            "registered_query_observer_operation_cell_count": covariance[
                "registered_query_observer_operation_cell_count"
            ],
            "observer_action_composition_case_count": covariance[
                "observer_action_composition_case_count"
            ],
            "finite_partition_count": quotient["partition_count"],
            "nontrivial_partition_count": quotient[
                "nontrivial_partition_count"
            ],
            "observer_partition_evaluation_count": quotient[
                "observer_partition_evaluation_count"
            ],
            "proposed_irreversible_operation_count": irreversible[
                "proposed_irreversible_operation_count"
            ],
            "proposed_cross_family_bridge_count": typed_composition[
                "proposed_cross_family_bridge_count"
            ],
            "required_independent_reviewer_count": phase411_review[
                "required_reviewer_count"
            ],
            "review_scenario_count_per_reviewer": phase411_review[
                "scenario_count_per_reviewer"
            ],
            "future_sealed_model_collector_case_count": 165,
            "model_case_count_consumed": 0,
            "physical_case_count_consumed": 0,
        },
        "results": {
            "fixed_observer_unstable_cell_count": covariance[
                "fixed_observer_unstable_cell_count"
            ],
            "role_moved_cell_count": covariance["role_moved_cell_count"],
            "fixed_instability_explained_by_role_transport_count": covariance[
                "fixed_instability_explained_by_role_transport_count"
            ],
            "typed_observer_unstable_cell_count": covariance[
                "typed_observer_unstable_cell_count"
            ],
            "typed_response_class_map_failure_cell_count": covariance[
                "typed_response_class_map_failure_cell_count"
            ],
            "observer_action_composition_failure_count": covariance[
                "observer_action_composition_failure_count"
            ],
            "full_operation_congruent_nontrivial_partition_count": quotient[
                "full_operation_congruent_nontrivial_partition_count"
            ],
            "joint_observation_sufficient_nontrivial_partition_count": quotient[
                "joint_observation_sufficient_nontrivial_partition_count"
            ],
            "global_nontrivial_qualifying_partition_count": quotient[
                "global_nontrivial_qualifying_partition_count"
            ],
            "fixed_observer_nontrivial_qualifying_partition_count": quotient[
                "fixed_observer_nontrivial_qualifying_partition_count"
            ],
            "role_stabilizer_nontrivial_qualifying_partition_count": quotient[
                "role_stabilizer_nontrivial_qualifying_partition_count"
            ],
            "external_role_conditioned_quotient_count": quotient[
                "external_role_conditioned_quotient_count"
            ],
            "external_role_indexed_partition_bundle_count": quotient[
                "external_role_indexed_partition_bundle_count"
            ],
            "model_derived_nontrivial_predictive_quotient_count": 0,
            "registered_executable_irreversible_operation_count": irreversible[
                "registered_executable_irreversible_operation_count"
            ],
            "registered_executable_cross_family_bridge_count": typed_composition[
                "registered_executable_cross_family_bridge_count"
            ],
            "completed_external_reviewer_count": sum(
                row["structurally_valid"]
                for row in phase411_review["review_results"]
            ),
            "sealed_model_collector_equivalence_case_count": 0,
            "new_behavioral_result_count": 0,
            "new_physical_path_count": 0,
            "new_causal_path_count": 0,
            "new_neuron_path_count": 0,
        },
        "hard_limits": [
            "typed_covariance_is_proven_only_for_external_finite_protocol_objects",
            "the_72_fixed_observer_failures_are_role_transport_mismatches_not_model_state_failures",
            "the_role_indexed_partition_bundle_is_not_a_global_or_model_derived_state_quotient",
            "joint_registered_observation_is_injective_and_forces_singleton_sufficiency",
            "current_finite_universes_cannot_represent_the_proposed_irreversible_operations",
            "no_cross_family_bridge_map_is_registered_or_semantically_reviewed",
            "two_external_human_reviews_are_still_absent",
            "sealed_real_model_collector_equivalence_remains_zero_of_165",
            "no_model_behavior_physical_causal_or_neuron_evidence_was_added",
            "small_models_may_use_coarse_or_model_specific_internal_structures",
            "single_global_progress_percentage_is_invalid",
        ],
        "authorization": {
            "publish_protocol_preflight": True,
            "register_role_indexed_external_partition_bundle": True,
            "register_global_predictive_state_quotient": False,
            "register_irreversible_operation_system": False,
            "register_cross_family_operation_system": False,
            "run_qwen3_model_qualification_next": model_qualification_authorized,
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
                "reviewer_disagreements_or_registry_conflicts_receive_independent_adjudication_or_contract_redesign",
                "reference_and_incremental_collectors_match_tokens_raw_scores_processed_scores_six_axes_events_stop_and_censoring_on_165_sealed_model_cases",
                "irreversible_state_universes_and_typed_cross_family_bridges_receive_external_semantic_review_before_registration",
            ],
            "model_order_after_gate": list(MODELS),
        },
        "single_global_progress_percentage_valid": False,
    }

    write_json(OUT / "phase412_typed_observer_covariance_audit.json", covariance)
    write_jsonl(
        OUT / "protocol/private/phase412_observer_operation_covariance.jsonl",
        covariance_rows,
    )
    write_jsonl(
        OUT / "protocol/private/phase412_observer_action_composition.jsonl",
        action_rows,
    )
    write_json(OUT / "phase412_nontrivial_quotient_audit.json", quotient)
    write_jsonl(
        OUT / "protocol/private/phase412_partition_catalog.jsonl", partition_rows
    )
    write_jsonl(
        OUT / "protocol/private/phase412_partition_observer_evaluations.jsonl",
        observer_rows,
    )
    write_jsonl(
        OUT / "protocol/private/phase412_induced_quotients_and_bundles.jsonl",
        induced_rows,
    )
    write_json(
        OUT / "phase412_irreversible_operation_readiness.json", irreversible
    )
    write_jsonl(
        OUT / "protocol/private/phase412_irreversible_operation_proposals.jsonl",
        irreversible_rows,
    )
    write_json(
        OUT / "phase412_typed_composition_readiness.json", typed_composition
    )
    write_jsonl(
        OUT / "protocol/private/phase412_cross_family_bridge_proposals.jsonl",
        bridge_rows,
    )
    write_json(OUT / "phase412_qualification.json", qualification)
    write_json(OUT / "phase412_stage_summary.json", stage)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
