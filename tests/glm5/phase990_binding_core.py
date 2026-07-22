#!/usr/bin/env python3
"""CPU-only sealed definitions for the Phase 990 binding protocol.

This module must not import torch or load model weights.  It defines the
versioned vocabulary, acyclic scientific schema, deterministic JSON format,
and no-overwrite publication helpers used by the Phase 990 builder and audit.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


PHASE = 990
SCHEMA_VERSION = 1
EXPERIMENT = "delayed_two_hop_binding_protocol_freeze"

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
OUT = GLM5 / "result" / "phase990_delayed_binding_protocol"

DEFINITIONS_PATH = OUT / "definitions_schema_v1.json"
DATASET_PATH = OUT / "dataset.json"
DATASET_AUDIT_PATH = OUT / "dataset_audit.json"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
INDEPENDENT_AUDIT_PATH = OUT / "protocol_audit.json"

MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
MODEL_PATHS = {
    "qwen3": "models/hf/qwen3-4b",
    "glm4": "models/hf/glm4-9b-chat-hf",
    "deepseek7b": "models/hf/deepseek-r1-distill-qwen-7b",
}

SPLIT_COUNTS = {
    "discovery": 96,
    "confirmation": 96,
    "adversarial": 64,
    "sealed_holdout": 64,
}
SPLIT_ORDER = tuple(SPLIT_COUNTS)
SEMANTIC_TRANSFORMS = (
    "original",
    "value_swap",
    "binding_swap",
    "relation_swap",
)
PARAPHRASE_IDS = ("standard", "paraphrase")
FACT_ORDER_IDS = ("order_a", "order_b")
HORIZON_IDS = ("near", "far")
VARIANTS = tuple(
    f"{semantic}__{paraphrase}__{order}__{horizon}"
    for semantic in SEMANTIC_TRANSFORMS
    for paraphrase in PARAPHRASE_IDS
    for order in FACT_ORDER_IDS
    for horizon in HORIZON_IDS
)

OWNER_RELATION = "paired_with"
ATTRIBUTE_RELATIONS = ("inner", "outer")
PEOPLE = (
    "Ava", "Borin", "Celia", "Darin",
    "Elara", "Faron", "Gina", "Hector",
    "Iris", "Jonas", "Kira", "Loren",
    "Maya", "Nolan", "Orla", "Pavel",
)
OBJECTS = (
    "lantern", "violin", "compass", "kettle",
    "telescope", "satchel", "goblet", "notebook",
    "hammer", "flute", "basket", "mirror",
    "helmet", "camera", "pillow", "vase",
)
VALUES = ("red", "blue", "green", "black")

EXPECTED_WORLD_COUNT = sum(SPLIT_COUNTS.values())
EXPECTED_ITEM_COUNT = EXPECTED_WORLD_COUNT * len(VARIANTS)

SCRIPT_PATHS = {
    "core": "tests/glm5/phase990_binding_core.py",
    "dataset": "tests/glm5/phase990_binding_dataset.py",
    "protocol": "tests/glm5/phase990_protocol_freeze.py",
    "audit": "tests/glm5/phase990_binding_audit.py",
}

CLOSURE_CERTIFICATES = (
    "replay_integrity",
    "edge_qualified",
    "instance_graph_validated",
    "task_predictive",
    "composition_replicated",
    "cross_model_replicated",
)

COMMON_RECORD_FIELDS = (
    "phase",
    "schema_version",
    "experiment",
    "record_type",
    "record_id",
    "protocol_sha256",
    "source_record_ids",
    "decision",
    "reason_codes",
    "created_at_utc",
)

DECISIONS = (
    "qualified",
    "rejected",
    "empty_within_frozen_scope",
    "nonunique",
    "undetermined",
    "blocked_upstream",
    "not_tested",
    "invalid",
    "inconclusive",
)

LEGAL_NULL_FIELDS = (
    "scope_sha256",
    "prerequisite_status_refs",
    "frozen_search_space_id",
    "planned_budget",
    "executed_budget",
    "screened_count",
    "qualified_count",
    "search_completed_to_protocol",
    "primary_reason",
    "upstream_status_ref",
    "epistemic_scope",
)

CLOSURE_REQUIRED_FIELDS = (
    "closure_id",
    "closure_type",
    "subject_type",
    "subject_id",
    "gate_sha256",
    "prerequisite_closure_ids",
    "prerequisite_closure_types",
    "model_scope",
    "split_scope",
    "thresholds",
    "budget_id",
    "evidence_refs",
    "counts_by_semantic_world_and_stratum",
    "decision",
    "reason_codes",
)

CLOSURE_EVIDENCE_FIELDS = {
    "replay_integrity": (
        "model_artifact_identity", "tokenizer_identity", "input_token_ids",
        "rng_and_precision", "hook_noop_control", "reference_recompute_error",
    ),
    "edge_qualified": (
        "target_damage", "correct_state_restoration",
        "wrong_donor_specificity", "downstream_mediation",
        "non_target_preservation", "unseen_sample_replication",
    ),
    "instance_graph_validated": (
        "qualified_edge_refs", "joint_intervention_checks",
        "registered_natural_rollout_predictions", "alternative_graph_status",
    ),
    "task_predictive": (
        "next_distribution", "update_consistency", "H_step_rollout",
        "intervention_response", "unseen_generalization",
        "full_state_and_simple_baselines",
    ),
    "composition_replicated": (
        "frozen_equivalence_rule", "confirmation_assignments",
        "holdout_assignments", "failure_boundary",
    ),
    "cross_model_replicated": (
        "role_alignment", "relative_depth_alignment", "component_alignment",
        "event_order_alignment", "intervention_response_alignment",
        "per_model_certificate_refs", "per_model_denominators",
    ),
}

CLOSURE_PREREQUISITE_TYPES = {
    "replay_integrity": (),
    "edge_qualified": ("replay_integrity",),
    "instance_graph_validated": ("edge_qualified",),
    "task_predictive": ("instance_graph_validated",),
    "composition_replicated": ("instance_graph_validated",),
    # Cross-model replication is an orthogonal axis: it names which already
    # defined certificate type is being replicated, rather than requiring the
    # separate composition extension to have passed.
    "cross_model_replicated": (),
}

CLOSURE_REFERENCE_RESOLUTION_CONTRACT = {
    "required_before_any_qualified_closure_is_accepted": True,
    "prerequisite_ids_must_resolve": True,
    "prerequisite_records_must_be_qualified": True,
    "prerequisite_types_must_match_declared_order": True,
    "per_model_refs_must_resolve": True,
    "per_model_refs_must_match_target_closure_type": True,
    "all_refs_must_share_protocol_gate_and_comparable_scope": True,
    "current_cpu_schema_performs_registry_resolution": False,
    "future_gpu_admission_blocked_until_resolver_exists": True,
}

INSTANCE_REFERENCE_RESOLUTION_CONTRACT = {
    "required_before_any_qualified_thread_is_accepted": True,
    "referenced_graph_id_must_resolve": True,
    "referenced_graph_must_be_qualified": True,
    "thread_window_must_equal_resolved_graph_window": True,
    "thread_events_must_belong_to_resolved_graph": True,
    "all_refs_must_share_protocol_and_instance_scope": True,
    "current_cpu_schema_performs_registry_resolution": False,
    "future_gpu_admission_blocked_until_resolver_exists": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_utc_timestamp(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} timestamp is not ISO-8601: {value}") from exc
    require(parsed.tzinfo is not None, f"{label} timestamp must be timezone-aware")
    require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        f"{label} timestamp must be UTC",
    )
    require(
        value.endswith("+00:00"),
        f"{label} timestamp must use canonical +00:00 suffix",
    )
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def strict_json_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label} must contain one JSON object")
    _assert_finite(value, label)
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    require(not path.is_symlink(), f"{label} cannot be a symlink: {path}")
    return strict_json_from_bytes(path.read_bytes(), label)


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{label} contains non-finite float")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")


def sealed_document(
    payload: Mapping[str, Any],
    hash_field: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    require(hash_field not in payload, f"{hash_field} already present in payload")
    require("created_at_utc" not in payload, "created_at_utc already present in payload")
    timestamp = validate_utc_timestamp(
        created_at_utc or utc_now(), "sealed document"
    )
    body = {**dict(payload), "created_at_utc": timestamp}
    return {**body, hash_field: sha256_json(body)}


def verify_self_hash(
    document: Mapping[str, Any],
    hash_field: str,
    label: str,
) -> None:
    require(isinstance(document, Mapping), f"{label} must be an object")
    supplied = document.get(hash_field)
    require(
        isinstance(supplied, str)
        and len(supplied) == 64
        and all(character in "0123456789abcdef" for character in supplied),
        f"{label} {hash_field} is invalid",
    )
    validate_utc_timestamp(document.get("created_at_utc"), label)
    body = {
        key: value for key, value in document.items()
        if key != hash_field
    }
    require(supplied == sha256_json(body), f"{label} self-hash mismatch")


def verify_exact_document(
    candidate: Mapping[str, Any],
    expected: Mapping[str, Any],
    hash_field: str,
    label: str,
) -> None:
    """Reject a validly rehashed candidate that differs from reconstruction."""
    verify_self_hash(candidate, hash_field, label)
    verify_self_hash(expected, hash_field, f"{label} reconstruction")
    require(
        dict(candidate) == dict(expected),
        f"{label} differs from complete reconstruction",
    )


def install_exact(path: Path, payload: bytes) -> bool:
    """Publish immutable bytes without overwriting an existing artifact.

    Returns True when this call installs the file and False when an identical
    file already exists.  A different existing file is always rejected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"sealed path cannot be a symlink: {path}")
    if path.exists():
        require(path.is_file(), f"sealed path is not a file: {path}")
        require(path.read_bytes() == payload, f"sealed artifact differs: {path}")
        return False

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".seal.tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            installed = True
        except FileExistsError:
            installed = False
        require(path.is_file(), f"sealed publication failed: {path}")
        require(not path.is_symlink(), f"sealed publication became symlink: {path}")
        require(path.read_bytes() == payload, f"sealed publication changed: {path}")
        return installed
    finally:
        temporary.unlink(missing_ok=True)


def _validate_source_registry(paths: Mapping[str, str]) -> None:
    require(paths, "source file registry is empty")
    for name, relative in paths.items():
        require(isinstance(name, str) and name, "invalid source seal name")
        require(isinstance(relative, str) and relative, "invalid source path")
        candidate = Path(relative)
        require(not candidate.is_absolute(), f"absolute source path: {relative}")
        require(".." not in candidate.parts, f"escaping source path: {relative}")


def _source_file_measure(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    require(path.is_file(), f"missing sealed source: {relative}")
    require(not path.is_symlink(), f"sealed source cannot be symlink: {relative}")
    with path.open("rb") as handle:
        payload = handle.read()
    return {
        "path": relative.replace("\\", "/"),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def file_seals(paths: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    _validate_source_registry(paths)
    result: dict[str, dict[str, Any]] = {}
    for name, relative in sorted(paths.items()):
        result[name] = _source_file_measure(relative)
    return result


def verify_file_seals(
    seals: Any,
    paths: Mapping[str, str],
    label: str,
) -> None:
    _validate_source_registry(paths)
    require(isinstance(seals, dict), f"{label} file seals must be an object")
    require(set(seals) == set(paths), f"{label} file seal registry changed")
    for name, relative in sorted(paths.items()):
        seal = seals[name]
        measured = _source_file_measure(relative)
        require(
            isinstance(seal, dict)
            and set(seal) == {"path", "bytes", "sha256"}
            and seal["path"] == measured["path"]
            and isinstance(seal["bytes"], int)
            and not isinstance(seal["bytes"], bool)
            and seal["bytes"] == measured["bytes"]
            and seal["sha256"] == measured["sha256"],
            f"{label} file seal mismatch: {name}",
        )


def opaque_id(namespace: str, *parts: Any) -> str:
    material = {"namespace": namespace, "parts": list(parts)}
    return sha256_json(material)[:32]


def definitions_payload() -> dict[str, Any]:
    nodes = [
        {
            "name": "language_factor_candidate",
            "symbol": "L",
            "depends_on": [],
            "required_fields": [
                "factor_id", "controlled_change", "variable_roles",
                "held_fixed", "positive_cases", "negative_controls",
                "external_targets",
            ],
            "forbidden_as_definition": [
                "activation", "attention_weight", "internal_operation_name",
                "model_layer", "mechanism_family",
            ],
        },
        {
            "name": "observed_event",
            "symbol": "e_obs",
            "depends_on": ["language_factor_candidate"],
            "physical_key": [
                "instance_id", "generation_step", "token_position",
                "layer", "component", "physical_group_id",
                "observable_id", "control_condition_id", "contrast_id",
            ],
            "allowed_evidence": [
                "activation_difference", "candidate_margin",
                "attention_weight", "probe_readout",
            ],
            "causal_claim_allowed": False,
        },
        {
            "name": "causal_event",
            "symbol": "e_causal",
            "depends_on": ["observed_event"],
            "required_evidence": [
                "target_damage", "correct_natural_state_restoration",
                "wrong_donor_specificity", "downstream_mediation",
                "non_target_preservation", "unseen_repeat",
            ],
            "recipient_field": "recipient_physical_keys",
            "joint_blocks_allowed": True,
            "allowed_decisions": ["qualified", "rejected", "undetermined"],
        },
        {
            "name": "effective_graph",
            "symbol": "G_eff_hat",
            "depends_on": ["causal_event"],
            "conditioning_fields": [
                "input", "anchor_generation_step", "generation_window",
                "target",
                "intervention_family", "granularity",
                "operational_thresholds", "search_budget",
            ],
            "generation_window_fields": ["start_step", "end_step"],
            "instance_required_fields": [
                "anchor_generation_step", "generation_window",
                "event_generation_steps",
            ],
            "edge_kinds": [
                "physical_legal", "observational_association",
                "intervention_total_effect", "mediated_direct_candidate",
                "joint_hyperedge",
            ],
            "allowed_decisions": [
                "qualified", "empty_within_frozen_scope", "nonunique",
                "inconclusive", "blocked_upstream",
            ],
            "full_physical_graph_is_success": False,
        },
        {
            "name": "instance_thread",
            "symbol": "tau",
            "depends_on": ["effective_graph"],
            "definition": "cross-generation-step temporal subgraph of one qualified effective graph",
            "window_rule": (
                "every thread event must lie inside the referenced graph "
                "generation_window"
            ),
            "required_external_anchors": [
                "trigger", "read", "complete", "exit"
            ],
            "instance_required_fields": [
                "referenced_graph_id", "referenced_generation_window",
                "event_generation_steps", "external_anchors",
            ],
            "link_kinds": [
                "history_or_logical_kv_available", "history_reread",
                "prefix_reconstruction", "generated_token_feedback",
                "mixed", "undetermined",
            ],
            "allowed_decisions": [
                "qualified", "empty_within_frozen_scope",
                "undetermined", "blocked_upstream",
            ],
        },
        {
            "name": "mechanism_family",
            "symbol": "M",
            "depends_on": ["instance_thread"],
            "frozen_before_validation": [
                "response_distance", "role_preservation",
                "event_order_match", "graph_match",
                "multiple_realization_rule", "no_match_result",
                "failure_boundary",
            ],
            "allowed_decisions": [
                "qualified", "rejected", "nonunique", "undetermined",
                "blocked_upstream",
            ],
            "claim_scope": "indistinguishable_under_registered_interventions",
        },
        {
            "name": "task_relative_sufficient_state",
            "symbol": "Z_task",
            "depends_on": ["mechanism_family"],
            "must_freeze": [
                "model_and_protocol", "targets", "future_horizon",
                "allowed_interventions", "query_target_family_Q",
                "representation_family_R", "update_family_U",
                "rng_strategy",
                "error_tolerance", "online_information_boundary",
                "complexity_cost", "simple_baselines",
            ],
            "required_tests": [
                "next_distribution", "update_commutation",
                "multi_step_rollout", "intervention_response",
                "unseen_generalization", "deletion_minimality",
                "merge_minimality", "smaller_than_full_state",
                "beats_simple_baseline",
            ],
            "allowed_decisions": [
                "qualified", "rejected", "undetermined",
                "empty_within_frozen_scope", "blocked_upstream",
            ],
            "qualification_labels": [
                "empirically_minimal_within_family",
                "sufficient_not_minimal",
            ],
            "full_state_can_pass_compression": False,
        },
        {
            "name": "closure_decision",
            "symbol": "D_closure",
            "depends_on": ["task_relative_sufficient_state"],
            "certificates": list(CLOSURE_CERTIFICATES),
            "reporting_dependency_only": True,
            "rule": "certificates remain separate; failures cannot be averaged away",
            "forbidden_equivalences": [
                "EOS_equals_mechanism_closure",
                "thread_exit_equals_mechanism_closure",
                "replay_equals_causal_closure",
                "brain_mapping_equals_model_causal_closure",
            ],
        },
    ]
    outcome_taxonomy = {
        "no_event": {
            "causes": [
                "no_observational_candidate",
                "all_observational_candidates_failed_causal_gate",
            ],
            "direct_conditions": {
                "search_completed_to_protocol": True,
                "qualified_causal_event_count": 0,
            },
        },
        "no_graph": {
            "causes": [
                "qualified_events_do_not_form_registered_graph",
            ],
            "direct_conditions": {
                "qualified_causal_event_count_min": 1,
                "graph_search_completed": True,
                "qualified_graph_count": 0,
            },
            "not_equivalent_to": ["nonunique", "inconclusive"],
        },
        "no_thread": {
            "causes": [
                "qualified_graph_has_no_registered_cross_step_subgraph",
            ],
            "direct_conditions": {
                "qualified_graph_count_min": 1,
                "thread_search_completed": True,
                "qualified_thread_count": 0,
            },
            "not_equivalent_to": ["undetermined"],
        },
        "no_compression": {
            "causes": [
                "not_strictly_smaller_than_full_state",
                "task_sufficiency_failed",
                "did_not_beat_simple_baselines",
                "prediction_failed",
                "update_failed",
                "rollout_failed",
                "intervention_preservation_failed",
                "generalization_failed",
            ],
            "direct_conditions": {
                "baselines_valid": True,
                "compression_search_completed": True,
                "qualified_Z_count": 0,
            },
            "not_equivalent_to": ["sufficient_not_minimal"],
        },
    }
    closure_subjects = {
        "replay_integrity": ["run", "replay_capsule"],
        "edge_qualified": ["causal_edge"],
        "instance_graph_validated": ["effective_graph"],
        "task_predictive": ["task_relative_sufficient_state"],
        "composition_replicated": ["mechanism_family", "composition_claim"],
        "cross_model_replicated": ["abstract_cross_model_claim"],
    }
    closure_contracts = {
        closure_type: {
            "closure_type": closure_type,
            "required_fields": list(CLOSURE_REQUIRED_FIELDS),
            "required_evidence": list(CLOSURE_EVIDENCE_FIELDS[closure_type]),
            "prerequisite_closure_types": list(
                CLOSURE_PREREQUISITE_TYPES[closure_type]
            ),
            "allowed_subject_types": closure_subjects[closure_type],
            "pooled_denominator": False,
            "orthogonal_extension": closure_type in {
                "composition_replicated", "cross_model_replicated",
            },
            **(
                {
                    "additional_required_fields": [
                        "target_closure_type", "per_model_certificate_refs",
                        "per_model_denominators",
                    ]
                }
                if closure_type == "cross_model_replicated"
                else {"additional_required_fields": []}
            ),
        }
        for closure_type in CLOSURE_CERTIFICATES
    }
    payload = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "cpu_only_definitions_schema",
        "definition_order": [node["name"] for node in nodes],
        "nodes": nodes,
        "common_record_contract": {
            "required_fields": list(COMMON_RECORD_FIELDS),
            "decision_enum": list(DECISIONS),
        },
        "outcome_taxonomy": outcome_taxonomy,
        "legal_null_contract": {
            "required_fields": list(LEGAL_NULL_FIELDS),
            "decision": "empty_within_frozen_scope",
            "epistemic_scope": "within_frozen_protocol_only",
            "interrupted_or_oom_decision": "inconclusive",
            "upstream_blocking": {
                "decision": "blocked_upstream",
                "requires": ["upstream_status_ref"],
                "is_direct_null_result": False,
            },
        },
        "closure_contracts": closure_contracts,
        "closure_reference_resolution_contract": dict(
            CLOSURE_REFERENCE_RESOLUTION_CONTRACT
        ),
        "instance_reference_resolution_contract": dict(
            INSTANCE_REFERENCE_RESOLUTION_CONTRACT
        ),
        "symbol_namespace": {
            "L": "language factor candidate",
            "e_obs": "observed physical event candidate",
            "e_causal": "causally qualified event",
            "G_eff_hat": "target-relative recovered effective graph",
            "tau": "instance cross-step thread",
            "M": "mechanism family",
            "Z_task": "task-relative sufficient state",
            "D_closure": "vector of separate closure certificates",
        },
        "runtime_boundary": {
            "cpu_only": True,
            "model_weights_loaded": False,
            "cuda_used": False,
            "gpu_experiment_authorized": False,
            "brain_data_loaded": False,
        },
    }
    audit_definitions_payload(payload)
    return payload


def audit_definitions_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    expected_top_level = {
        "phase", "schema_version", "experiment", "role",
        "definition_order", "nodes", "common_record_contract",
        "outcome_taxonomy", "legal_null_contract", "closure_contracts",
        "closure_reference_resolution_contract",
        "instance_reference_resolution_contract", "symbol_namespace",
        "runtime_boundary",
    }
    if set(payload) != expected_top_level:
        errors.append("top-level definitions schema changed")
    if payload.get("phase") != PHASE:
        errors.append("phase changed")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema version changed")
    if payload.get("experiment") != EXPERIMENT:
        errors.append("experiment changed")
    if payload.get("role") != "cpu_only_definitions_schema":
        errors.append("role changed")

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        errors.append("nodes must be a list")
        nodes = []
    names = [
        node.get("name") for node in nodes
        if isinstance(node, dict)
    ]
    if names != payload.get("definition_order"):
        errors.append("definition order does not match nodes")
    if len(names) != len(set(names)):
        errors.append("definition names are not unique")
    expected_names = [
        "language_factor_candidate", "observed_event", "causal_event",
        "effective_graph", "instance_thread", "mechanism_family",
        "task_relative_sufficient_state", "closure_decision",
    ]
    if names != expected_names:
        errors.append("definition registry changed")

    index = {name: offset for offset, name in enumerate(names)}
    for node in nodes:
        if not isinstance(node, dict):
            errors.append("node is not an object")
            continue
        name = node.get("name")
        for dependency in node.get("depends_on", []):
            if dependency not in index:
                errors.append(f"{name}: unknown dependency {dependency}")
            elif index[dependency] >= index.get(name, -1):
                errors.append(f"{name}: cyclic or forward dependency {dependency}")

    namespace = payload.get("symbol_namespace")
    expected_symbols = {
        "L": "language factor candidate",
        "e_obs": "observed physical event candidate",
        "e_causal": "causally qualified event",
        "G_eff_hat": "target-relative recovered effective graph",
        "tau": "instance cross-step thread",
        "M": "mechanism family",
        "Z_task": "task-relative sufficient state",
        "D_closure": "vector of separate closure certificates",
    }
    if namespace != expected_symbols:
        errors.append("symbol namespace invalid")
    if payload.get("closure_reference_resolution_contract") != (
        CLOSURE_REFERENCE_RESOLUTION_CONTRACT
    ):
        errors.append("closure reference resolution contract changed")
    if payload.get("instance_reference_resolution_contract") != (
        INSTANCE_REFERENCE_RESOLUTION_CONTRACT
    ):
        errors.append("instance reference resolution contract changed")

    by_name = {
        str(node.get("name")): node
        for node in nodes if isinstance(node, dict)
    }
    language_node = by_name.get("language_factor_candidate", {})
    if set(language_node.get("forbidden_as_definition", [])) != {
        "activation", "attention_weight", "internal_operation_name",
        "model_layer", "mechanism_family",
    }:
        errors.append("language-factor internal-state prohibition changed")
    observed_node = by_name.get("observed_event", {})
    if observed_node.get("causal_claim_allowed") is not False:
        errors.append("observed event was allowed to make a causal claim")
    if set(observed_node.get("physical_key", [])) != {
        "instance_id", "generation_step", "token_position", "layer",
        "component", "physical_group_id", "observable_id",
        "control_condition_id", "contrast_id",
    }:
        errors.append("observed-event physical key changed")
    graph_node = by_name.get("effective_graph", {})
    if set(graph_node.get("conditioning_fields", [])) != {
        "input", "anchor_generation_step", "generation_window", "target",
        "intervention_family", "granularity", "operational_thresholds",
        "search_budget",
    }:
        errors.append("effective-graph conditioning fields changed")
    if graph_node.get("generation_window_fields") != [
        "start_step", "end_step"
    ]:
        errors.append("effective-graph generation window is incomplete")
    thread_node = by_name.get("instance_thread", {})
    if "generation_window" not in str(thread_node.get("window_rule", "")):
        errors.append("thread-to-graph generation-window rule missing")
    expected_object_decisions = {
        "causal_event": ["qualified", "rejected", "undetermined"],
        "effective_graph": [
            "qualified", "empty_within_frozen_scope", "nonunique",
            "inconclusive", "blocked_upstream",
        ],
        "instance_thread": [
            "qualified", "empty_within_frozen_scope", "undetermined",
            "blocked_upstream",
        ],
        "mechanism_family": [
            "qualified", "rejected", "nonunique", "undetermined",
            "blocked_upstream",
        ],
        "task_relative_sufficient_state": [
            "qualified", "rejected", "undetermined",
            "empty_within_frozen_scope", "blocked_upstream",
        ],
    }
    for node_name, decisions in expected_object_decisions.items():
        if by_name.get(node_name, {}).get("allowed_decisions") != decisions:
            errors.append(f"{node_name} decision enum diverges from common contract")
    closure_node = by_name.get("closure_decision", {})
    if closure_node.get("certificates") != list(CLOSURE_CERTIFICATES):
        errors.append("closure certificate registry changed")
    if closure_node.get("reporting_dependency_only") is not True:
        errors.append("closure reporting dependency semantics changed")

    record_contract = payload.get("common_record_contract")
    if record_contract != {
        "required_fields": list(COMMON_RECORD_FIELDS),
        "decision_enum": list(DECISIONS),
    }:
        errors.append("common scientific record contract changed")

    outcomes = payload.get("outcome_taxonomy")
    if not isinstance(outcomes, dict) or set(outcomes) != {
        "no_event", "no_graph", "no_thread", "no_compression",
    }:
        errors.append("failure outcome taxonomy changed")
    elif "upstream" in canonical_json(outcomes):
        errors.append("upstream blocking leaked into direct null outcomes")
    else:
        expected_conditions = {
            "no_event": {
                "search_completed_to_protocol": True,
                "qualified_causal_event_count": 0,
            },
            "no_graph": {
                "qualified_causal_event_count_min": 1,
                "graph_search_completed": True,
                "qualified_graph_count": 0,
            },
            "no_thread": {
                "qualified_graph_count_min": 1,
                "thread_search_completed": True,
                "qualified_thread_count": 0,
            },
            "no_compression": {
                "baselines_valid": True,
                "compression_search_completed": True,
                "qualified_Z_count": 0,
            },
        }
        for name, conditions in expected_conditions.items():
            if outcomes.get(name, {}).get("direct_conditions") != conditions:
                errors.append(f"{name} direct conditions changed")

    legal_null = payload.get("legal_null_contract")
    expected_legal_null = {
        "required_fields": list(LEGAL_NULL_FIELDS),
        "decision": "empty_within_frozen_scope",
        "epistemic_scope": "within_frozen_protocol_only",
        "interrupted_or_oom_decision": "inconclusive",
        "upstream_blocking": {
            "decision": "blocked_upstream",
            "requires": ["upstream_status_ref"],
            "is_direct_null_result": False,
        },
    }
    if legal_null != expected_legal_null:
        errors.append("legal-null record contract changed")

    contracts = payload.get("closure_contracts")
    if not isinstance(contracts, dict) or set(contracts) != set(
        CLOSURE_CERTIFICATES
    ):
        errors.append("closure contracts missing or duplicated")
    else:
        for closure_type, contract in contracts.items():
            if contract.get("closure_type") != closure_type:
                errors.append(f"{closure_type} identity changed")
            if contract.get("required_fields") != list(
                CLOSURE_REQUIRED_FIELDS
            ):
                errors.append(f"{closure_type} required fields changed")
            if contract.get("required_evidence") != list(
                CLOSURE_EVIDENCE_FIELDS[closure_type]
            ):
                errors.append(f"{closure_type} evidence gate changed")
            if contract.get("prerequisite_closure_types") != list(
                CLOSURE_PREREQUISITE_TYPES[closure_type]
            ):
                errors.append(f"{closure_type} prerequisite gate changed")
            if contract.get("pooled_denominator") is not False:
                errors.append(f"{closure_type} pooled denominators enabled")
        cross_model = contracts.get("cross_model_replicated", {})
        if cross_model.get("additional_required_fields") != [
            "target_closure_type", "per_model_certificate_refs",
            "per_model_denominators"
        ]:
            errors.append("cross-model per-model evidence contract changed")
        for extension in ("composition_replicated", "cross_model_replicated"):
            if contracts.get(extension, {}).get("orthogonal_extension") is not True:
                errors.append(f"{extension} lost orthogonal status")

    runtime = payload.get("runtime_boundary")
    expected_runtime = {
        "cpu_only": True,
        "model_weights_loaded": False,
        "cuda_used": False,
        "gpu_experiment_authorized": False,
        "brain_data_loaded": False,
    }
    if runtime != expected_runtime:
        errors.append("runtime boundary changed")

    report = {
        "passed": not errors,
        "errors": errors,
        "node_count": len(nodes),
        "acyclic": not any("cyclic" in error for error in errors),
        "legal_failure_states": sorted(outcomes) if isinstance(outcomes, dict) else [],
        "payload_sha256": sha256_json(dict(payload)),
    }
    require(report["passed"], f"definitions schema invalid: {errors}")
    return report


def audit_scientific_record(record: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    missing = [field for field in COMMON_RECORD_FIELDS if field not in record]
    if missing:
        errors.append(f"missing common fields: {missing}")
    if record.get("phase") != PHASE:
        errors.append("phase mismatch")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema version mismatch")
    if record.get("experiment") != EXPERIMENT:
        errors.append("experiment mismatch")
    if record.get("decision") not in DECISIONS:
        errors.append("decision is outside frozen enum")
    source_ids = record.get("source_record_ids")
    if not isinstance(source_ids, list):
        errors.append("source_record_ids must be a list")
    elif (
        any(not isinstance(value, str) or not value for value in source_ids)
        or len(source_ids) != len(set(source_ids))
    ):
        errors.append("source_record_ids must contain unique nonempty strings")
    reason_codes = record.get("reason_codes")
    if not isinstance(reason_codes, list):
        errors.append("reason_codes must be a list")
    elif (
        any(not isinstance(value, str) or not value for value in reason_codes)
        or len(reason_codes) != len(set(reason_codes))
    ):
        errors.append("reason_codes must contain unique nonempty strings")
    if not isinstance(record.get("record_type"), str) or not record.get(
        "record_type"
    ):
        errors.append("record_type is invalid")
    if not isinstance(record.get("record_id"), str) or not record.get("record_id"):
        errors.append("record_id is invalid")
    protocol_sha = record.get("protocol_sha256")
    if (
        not isinstance(protocol_sha, str)
        or len(protocol_sha) != 64
        or any(character not in "0123456789abcdef"
               for character in protocol_sha)
    ):
        errors.append("protocol_sha256 is invalid")
    try:
        validate_utc_timestamp(record.get("created_at_utc"), "scientific record")
    except RuntimeError as exc:
        errors.append(str(exc))
    return {"passed": not errors, "errors": errors}


def _valid_generation_step(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _audit_generation_window(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[int, int] | None:
    if not isinstance(value, dict) or set(value) != {"start_step", "end_step"}:
        errors.append(f"{label} must contain start_step/end_step")
        return None
    start = value.get("start_step")
    end = value.get("end_step")
    if not _valid_generation_step(start) or not _valid_generation_step(end):
        errors.append(f"{label} steps must be nonnegative integers")
        return None
    if start > end:
        errors.append(f"{label} start exceeds end")
        return None
    return int(start), int(end)


def audit_effective_graph_instance(record: Mapping[str, Any]) -> dict[str, Any]:
    report = audit_scientific_record(record)
    errors = list(report["errors"])
    if record.get("record_type") != "effective_graph":
        errors.append("effective graph record_type changed")
    allowed = next(
        node["allowed_decisions"] for node in definitions_payload()["nodes"]
        if node["name"] == "effective_graph"
    )
    if record.get("decision") not in allowed:
        errors.append("effective graph decision is invalid")
    window = _audit_generation_window(
        record.get("generation_window"), "generation_window", errors
    )
    anchor = record.get("anchor_generation_step")
    steps = record.get("event_generation_steps")
    if not _valid_generation_step(anchor):
        errors.append("anchor_generation_step is invalid")
    if (
        not isinstance(steps, list) or not steps
        or any(not _valid_generation_step(step) for step in steps)
    ):
        errors.append("event_generation_steps are invalid")
    if window is not None:
        start, end = window
        if _valid_generation_step(anchor) and not start <= anchor <= end:
            errors.append("anchor lies outside generation_window")
        if isinstance(steps, list) and any(
            _valid_generation_step(step) and not start <= step <= end
            for step in steps
        ):
            errors.append("graph event lies outside generation_window")
    return {"passed": not errors, "errors": errors}


def audit_instance_thread_record(record: Mapping[str, Any]) -> dict[str, Any]:
    report = audit_scientific_record(record)
    errors = list(report["errors"])
    if record.get("record_type") != "instance_thread":
        errors.append("instance thread record_type changed")
    allowed = next(
        node["allowed_decisions"] for node in definitions_payload()["nodes"]
        if node["name"] == "instance_thread"
    )
    if record.get("decision") not in allowed:
        errors.append("instance thread decision is invalid")
    if not isinstance(record.get("referenced_graph_id"), str) or not record.get(
        "referenced_graph_id"
    ):
        errors.append("referenced_graph_id is invalid")
    window = _audit_generation_window(
        record.get("referenced_generation_window"),
        "referenced_generation_window",
        errors,
    )
    steps = record.get("event_generation_steps")
    if (
        not isinstance(steps, list) or not steps
        or any(not _valid_generation_step(step) for step in steps)
    ):
        errors.append("thread event_generation_steps are invalid")
    elif record.get("decision") == "qualified" and len(set(steps)) < 2:
        errors.append("qualified thread must cross generation steps")
    anchors = record.get("external_anchors")
    anchor_names = ("trigger", "read", "complete", "exit")
    if (
        not isinstance(anchors, dict) or set(anchors) != set(anchor_names)
        or any(not _valid_generation_step(anchors.get(name))
               for name in anchor_names)
    ):
        errors.append("thread external_anchors are invalid")
    elif not all(
        anchors[left] <= anchors[right]
        for left, right in zip(
            anchor_names[:-1], anchor_names[1:], strict=True
        )
    ):
        errors.append("thread external anchors are out of order")
    if window is not None:
        start, end = window
        inspected_steps = [
            step for step in steps if _valid_generation_step(step)
        ] if isinstance(steps, list) else []
        if isinstance(anchors, dict):
            inspected_steps.extend(
                value for value in anchors.values()
                if _valid_generation_step(value)
            )
        if any(not start <= step <= end for step in inspected_steps):
            errors.append("thread event or anchor lies outside graph window")
    return {"passed": not errors, "errors": errors}


def audit_legal_null_record(record: Mapping[str, Any]) -> dict[str, Any]:
    report = audit_scientific_record(record)
    errors = list(report["errors"])
    record_type = record.get("record_type")
    taxonomy = definitions_payload()["outcome_taxonomy"]
    if not isinstance(record_type, str) or record_type not in taxonomy:
        errors.append("record_type is not a legal direct null outcome")
    missing = [field for field in LEGAL_NULL_FIELDS if field not in record]
    if missing:
        errors.append(f"missing legal-null fields: {missing}")
    if record.get("decision") != "empty_within_frozen_scope":
        errors.append("direct null decision changed")
    if record.get("epistemic_scope") != "within_frozen_protocol_only":
        errors.append("null result escaped its frozen epistemic scope")
    if record.get("upstream_status_ref") is not None:
        errors.append("direct null result cannot be an upstream propagation")
    if record.get("search_completed_to_protocol") is not True:
        errors.append("search was not completed to protocol")
    if record.get("planned_budget") != record.get("executed_budget"):
        errors.append("executed budget differs from planned budget")
    for count_field in ("screened_count", "qualified_count"):
        count = record.get(count_field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            errors.append(f"{count_field} must be a nonnegative integer")
    if (
        isinstance(record.get("screened_count"), int)
        and isinstance(record.get("qualified_count"), int)
        and record["qualified_count"] > record["screened_count"]
    ):
        errors.append("qualified_count exceeds screened_count")
    if record.get("qualified_count") != 0:
        errors.append("direct null requires qualified_count zero")
    reason_codes = record.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or record.get("primary_reason") not in reason_codes
    ):
        errors.append("primary_reason must appear in reason_codes")
    scope_sha = record.get("scope_sha256")
    if (
        not isinstance(scope_sha, str)
        or len(scope_sha) != 64
        or any(character not in "0123456789abcdef" for character in scope_sha)
    ):
        errors.append("scope_sha256 is invalid")
    if not isinstance(record.get("frozen_search_space_id"), str) or not record.get(
        "frozen_search_space_id"
    ):
        errors.append("frozen_search_space_id is invalid")
    prerequisite_refs = record.get("prerequisite_status_refs")
    if (
        not isinstance(prerequisite_refs, list)
        or any(not isinstance(value, str) or not value
               for value in prerequisite_refs)
        or len(prerequisite_refs) != len(set(prerequisite_refs))
    ):
        errors.append("prerequisite_status_refs are invalid")
    if (
        isinstance(record_type, str)
        and record_type in taxonomy
        and record.get("primary_reason") not in taxonomy[record_type]["causes"]
    ):
        errors.append("primary reason is outside the frozen taxonomy")
    if record_type == "no_event" and record.get(
        "qualified_causal_event_count"
    ) != 0:
        errors.append("no_event requires zero qualified causal events")
    if record_type == "no_graph" and not (
        isinstance(record.get("qualified_causal_event_count"), int)
        and record["qualified_causal_event_count"] > 0
        and record.get("graph_search_completed") is True
        and record.get("qualified_graph_count") == 0
    ):
        errors.append("no_graph direct prerequisites failed")
    if record_type == "no_thread" and not (
        isinstance(record.get("qualified_graph_count"), int)
        and record["qualified_graph_count"] > 0
        and record.get("thread_search_completed") is True
        and record.get("qualified_thread_count") == 0
    ):
        errors.append("no_thread direct prerequisites failed")
    if record_type == "no_compression" and not (
        record.get("baselines_valid") is True
        and record.get("compression_search_completed") is True
        and record.get("qualified_Z_count") == 0
    ):
        errors.append("no_compression direct prerequisites failed")
    return {"passed": not errors, "errors": errors}


def audit_blocked_upstream_record(record: Mapping[str, Any]) -> dict[str, Any]:
    report = audit_scientific_record(record)
    errors = list(report["errors"])
    if record.get("decision") != "blocked_upstream":
        errors.append("upstream record decision is not blocked_upstream")
    if not isinstance(record.get("upstream_status_ref"), str) or not record.get(
        "upstream_status_ref"
    ):
        errors.append("upstream_status_ref is required")
    return {"passed": not errors, "errors": errors}


def audit_closure_certificate(record: Mapping[str, Any]) -> dict[str, Any]:
    report = audit_scientific_record(record)
    errors = list(report["errors"])
    closure_type = record.get("closure_type")
    contracts = definitions_payload()["closure_contracts"]
    if not isinstance(closure_type, str) or closure_type not in contracts:
        errors.append("unknown closure type")
        return {"passed": False, "errors": errors}
    contract = contracts[closure_type]
    required = [
        *CLOSURE_REQUIRED_FIELDS,
        *contract["additional_required_fields"],
    ]
    missing = [field for field in required if field not in record]
    if missing:
        errors.append(f"missing closure fields: {missing}")
    if record.get("record_type") != "closure_certificate":
        errors.append("closure record_type changed")
    if record.get("closure_id") != record.get("record_id"):
        errors.append("closure_id must equal record_id")
    if record.get("subject_type") not in contract["allowed_subject_types"]:
        errors.append("closure subject type is not allowed")
    evidence_refs = record.get("evidence_refs")
    if not isinstance(evidence_refs, dict) or set(evidence_refs) != set(
        contract["required_evidence"]
    ):
        errors.append("closure evidence gate is incomplete")
    elif any(value is None or value == "" or value == [] or value == {}
             for value in evidence_refs.values()):
        errors.append("closure evidence refs contain empty evidence")
    gate_sha = record.get("gate_sha256")
    if (
        not isinstance(gate_sha, str)
        or len(gate_sha) != 64
        or any(character not in "0123456789abcdef" for character in gate_sha)
    ):
        errors.append("closure gate_sha256 is invalid")
    prerequisites = record.get("prerequisite_closure_ids")
    if not isinstance(prerequisites, list):
        errors.append("closure prerequisite refs must be a list")
    elif (
        len(prerequisites) != len(contract["prerequisite_closure_types"])
        or any(not isinstance(value, str) or not value for value in prerequisites)
        or len(prerequisites) != len(set(prerequisites))
    ):
        errors.append("closure prerequisite refs are incomplete")
    if record.get("prerequisite_closure_types") != contract[
        "prerequisite_closure_types"
    ]:
        errors.append("closure prerequisite types changed")
    for scope_field in ("model_scope", "split_scope"):
        scope = record.get(scope_field)
        if (
            not isinstance(scope, list) or not scope
            or any(not isinstance(value, str) or not value for value in scope)
            or len(scope) != len(set(scope))
        ):
            errors.append(f"{scope_field} is invalid")
    if not isinstance(record.get("thresholds"), dict) or not record.get(
        "thresholds"
    ):
        errors.append("closure thresholds are empty")
    if not isinstance(record.get("budget_id"), str) or not record.get("budget_id"):
        errors.append("closure budget_id is invalid")
    counts = record.get("counts_by_semantic_world_and_stratum")
    if not isinstance(counts, dict) or not counts:
        errors.append("closure counts are empty")
    if record.get("pooled_denominator", False) is not False:
        errors.append("closure denominators cannot be pooled")
    if closure_type == "cross_model_replicated":
        target_type = record.get("target_closure_type")
        if (
            not isinstance(target_type, str)
            or target_type not in set(CLOSURE_CERTIFICATES) - {
                "cross_model_replicated"
            }
        ):
            errors.append("cross-model target closure type is invalid")
        certificate_refs = record.get("per_model_certificate_refs")
        if (
            not isinstance(certificate_refs, dict)
            or set(certificate_refs) != set(MODEL_ORDER)
            or any(not isinstance(reference, str) or not reference
                   for reference in certificate_refs.values())
            or len(set(certificate_refs.values())) != len(MODEL_ORDER)
        ):
            errors.append("cross-model certificate refs must be distinct per model")
        denominators = record.get("per_model_denominators")
        if (
            not isinstance(denominators, dict)
            or set(denominators) != set(MODEL_ORDER)
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value <= 0 for value in denominators.values())
        ):
            errors.append("cross-model denominators must remain per model")
    return {"passed": not errors, "errors": errors}


def definitions_document(created_at_utc: str | None = None) -> dict[str, Any]:
    return sealed_document(
        definitions_payload(),
        "definitions_sha256",
        created_at_utc=created_at_utc,
    )


def protocol_static_contract() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "cpu_only_preregistration_no_generation_admission",
        "scientific_target": (
            "delayed two-hop object-relation-value contextual binding and retrieval"
        ),
        "explicit_non_targets": [
            "complete_language_theory", "brain_transformer_identity",
            "thinking_switch", "EOS_mechanism", "open_ended_story_mechanism",
        ],
        "model_order": list(MODEL_ORDER),
        "model_paths": dict(MODEL_PATHS),
        "execution_policy": {
            "one_model_resident_at_a_time": True,
            "order_is_mandatory": True,
            "release_and_gpu_baseline_between_models": True,
            "independent_cpu_audit_after_all_models": True,
            "current_gpu_execution_authorized": False,
        },
        "dataset_plan": {
            "independent_unit": "semantic_world_id",
            "split_counts": dict(SPLIT_COUNTS),
            "variants": list(VARIANTS),
            "variants_are_paired_not_independent": True,
            "abstract_semantic_states_expected": (
                EXPECTED_WORLD_COUNT * len(SEMANTIC_TRANSFORMS)
            ),
            "abstract_semantic_states_globally_unique": True,
            "registered_marginal_and_all_owner_edge_grids_exact": True,
            "query_selected_lexical_conjunction_exact": False,
            "lexical_generalization_is_not_tested": True,
            "holdout_semantics": "preregistered_immutable_not_blind",
            "holdout_may_not_tune_rules_or_thresholds": True,
            "holdout_visibility_limitation": (
                "local deterministic generator and records are inspectable"
            ),
            "gpu_holdout_access_gate": {
                "status": "not_satisfied_gpu_blocker",
                "required_before_first_model_evaluation_or_result_scoring_access": [
                    "candidate_set_sha256", "equivalence_rule_sha256",
                    "thresholds_sha256", "freeze_timestamp",
                ],
                "required_on_first_model_evaluation_access": [
                    "holdout_first_model_access_timestamp"
                ],
                "runner_must_reject_early_model_or_scoring_access": True,
            },
            "expanded_confirmation_worlds_after_three_model_sealed_pass": 128,
            "expanded_confirmation_status": "future_not_generated_not_authorized",
            "expanded_confirmation_generator_contract": {
                "status": "not_frozen_gpu_blocker",
                "current_generator_has_seed_parameter": False,
                "current_seed_key_is_identity_label_only": True,
                "same_locked_generator_claim_authorized": False,
                "must_freeze_before_any_model_result_review": [
                    "generator_source_sha256",
                    "seed_domain_and_exclusion_rule_sha256",
                    "extension_split_assignment_sha256",
                    "primary_candidate_set_sha256",
                    "primary_thresholds_sha256",
                    "observable_semantic_and_prompt_exclusion_proof_sha256",
                ],
                "extension_candidate_set_must_equal_primary": True,
                "extension_thresholds_must_equal_primary": True,
                "must_prove_zero_overlap_with_current_320": [
                    "abstract_semantic_state",
                    "observable_semantic_state",
                    "normalized_prompt",
                ],
            },
        },
        "answer_contract": {
            "natural_reference_template": "The retrieved marker is {value}.",
            "teacher_forced_prefix": "The retrieved marker is",
            "candidate_continuation_template": " {value}",
            "teacher_forced_context_joiner": "\n",
            "input_assembly": (
                "tokenize(prompt + joiner + prefix + continuation) and "
                "require full_ids == context_ids + continuation_ids"
            ),
            "target_not_first_natural_reference_token": True,
            "target_is_first_scored_continuation_after_teacher_prefix": True,
            "strict_final_marker_required": False,
            "all_three_foils_scored": True,
        },
        "tokenizer_preflight_contract": {
            "model_order": list(MODEL_ORDER),
            "all_dataset_records_checked": True,
            "all_four_context_candidate_boundaries_checked": True,
            "candidate_continuations_equal_length": True,
            "candidate_continuations_prefer_single_token": True,
            "unexpected_special_tokens_forbidden": True,
            "value_and_binding_pair_token_multiset_equal": True,
            "near_far_token_length_equal": True,
            "order_pair_token_multiset_equal": True,
            "relation_phrases_equal_length": True,
            "model_weights_forbidden": True,
            "cuda_forbidden": True,
        },
        "prior_dataset_overlap_contract": {
            "phases": [979, 981, 983],
            "normalized_prompt_overlap_required": 0,
            "source_files_must_be_hashed": True,
        },
        "behavior_gates_percent": {
            "overall_each_major_condition_min": 90,
            "each_preregistered_stratum_min": 85,
            "counterfactual_pair_sensitivity_min": 85,
            "semantic_protocol_eos_budget_scored_separately": True,
            "teacher_forcing_does_not_replace_natural_generation": True,
        },
        "registered_shortcut_baseline_contract": {
            "selected_owner_name_object_relation_majority_percent_by_semantic": 50,
            "selected_owner_name_object_relation_majority_percent_pooled": 37.5,
            "chance_percent": 25,
            "must_report_with_behavior_results": True,
            "behavior_above_baseline_is_not_second_hop_mechanism_proof": True,
            "mechanism_claim_requires_registered_value_and_relation_interventions": True,
        },
        "search_contract": {
            "initial_position_groups": [
                "fact_source", "distractor", "query",
                "fixed_answer_or_current_generation",
            ],
            "initial_relative_layer_blocks": 6,
            "initial_component": "residual_state",
            "discovery_common_worlds_max": 16,
            "frozen_candidate_blocks_max": 4,
            "confirmation_worlds": 32,
            "component_refinement_parent_blocks_max_per_model": 2,
            "adversarial_lifecycle_worlds": 16,
            "branch_points_max": 2,
            "branches_per_point_max": 3,
            "recursive_branching": False,
        },
        "confirmation_world_boolean_gates": {
            "denominator": 32,
            "necessity_min": 26,
            "correct_restoration_min": 26,
            "specified_mediation_min": 26,
            "non_target_preservation_min": 26,
            "each_wrong_donor_restoration_max": 3,
            "distance_and_paraphrase_direction_required": True,
            "adversarial_and_sealed_repeat_required": True,
        },
        "cross_step_certificates": [
            "trigger", "write", "persistence_or_reconstruction",
            "read", "update", "output", "branch",
            "rollout", "controls", "holdout",
        ],
        "legal_failure_states": [
            "no_event", "no_graph", "no_thread", "no_compression",
            "blocked_upstream", "inconclusive", "model_specific_only",
            "scientific_no_go",
        ],
        "resource_contract": {
            "result_quota_gib_min": 40,
            "result_quota_gib_max": 60,
            "minimum_free_disk_gib": 80,
            "store_full_dense_trajectory_per_anchor": False,
            "store_replay_capsule": True,
            "kv_saved_once_per_token": True,
            "source_conditioned_values_recomputed_and_reduced": True,
            "activation_streamed_to_cpu": True,
            "batch_size_policy": "one_or_small",
        },
        "brain_prior_contract": {
            "brain_data_first_round": False,
            "brain_score_selects_model_components": False,
            "context_spans": ["near", "far_length_matched"],
            "expectation_contrasts_secondary_only": [
                "high_expected", "low_expected_legal", "semantic_conflict",
            ],
            "brain_evidence": "observational",
            "not_an_admission_criterion": True,
        },
        "runtime_boundary": {
            "cpu_only_protocol_freeze": True,
            "tokenizers_may_load_on_cpu": True,
            "model_weights_may_load": False,
            "cuda_may_run": False,
            "formal_gpu_scripts_may_be_created": False,
            "formal_generation_admission": False,
        },
    }


def self_test() -> dict[str, Any]:
    first = definitions_payload()
    second = definitions_payload()
    deterministic = first == second

    def schema_rejects(candidate: Mapping[str, Any]) -> bool:
        try:
            audit_definitions_payload(candidate)
        except RuntimeError:
            return True
        return False

    cycle_rejected = False
    cycle_probe = json.loads(canonical_json(first))
    cycle_probe["nodes"][0]["depends_on"] = ["closure_decision"]
    cycle_rejected = schema_rejects(cycle_probe)

    runtime_rejected = False
    runtime_probe = json.loads(canonical_json(first))
    runtime_probe["runtime_boundary"]["gpu_experiment_authorized"] = True
    runtime_rejected = schema_rejects(runtime_probe)

    window_probe = json.loads(canonical_json(first))
    graph_probe = next(
        node for node in window_probe["nodes"]
        if node["name"] == "effective_graph"
    )
    graph_probe["conditioning_fields"].remove("generation_window")

    envelope_probe = json.loads(canonical_json(first))
    envelope_probe["common_record_contract"]["required_fields"].remove(
        "reason_codes"
    )

    closure_probe = json.loads(canonical_json(first))
    del closure_probe["closure_contracts"]["edge_qualified"]

    upstream_probe = json.loads(canonical_json(first))
    upstream_probe["outcome_taxonomy"]["no_graph"]["causes"].append(
        "upstream_no_event"
    )

    pooled_probe = json.loads(canonical_json(first))
    pooled_probe["closure_contracts"]["cross_model_replicated"][
        "pooled_denominator"
    ] = True

    document = definitions_document("2000-01-01T00:00:00+00:00")
    verify_self_hash(document, "definitions_sha256", "synthetic definitions")
    rehash_attack = dict(document)
    rehash_attack["runtime_boundary"] = dict(rehash_attack["runtime_boundary"])
    rehash_attack["runtime_boundary"]["gpu_experiment_authorized"] = True
    attack_body = {
        key: value for key, value in rehash_attack.items()
        if key != "definitions_sha256"
    }
    rehash_attack["definitions_sha256"] = sha256_json(attack_body)
    self_hash_accepts_rehash = True
    try:
        verify_self_hash(
            rehash_attack, "definitions_sha256", "synthetic rehash attack"
        )
    except RuntimeError:
        self_hash_accepts_rehash = False
    exact_rebuild_rejects_rehash = False
    try:
        verify_exact_document(
            rehash_attack,
            document,
            "definitions_sha256",
            "synthetic rehash attack",
        )
    except RuntimeError:
        exact_rebuild_rejects_rehash = True

    bad_timestamp = dict(document)
    bad_timestamp["created_at_utc"] = "not-a-time"
    timestamp_rejected = False
    try:
        verify_self_hash(bad_timestamp, "definitions_sha256", "bad timestamp")
    except RuntimeError:
        timestamp_rejected = True

    duplicate_json_rejected = False
    try:
        strict_json_from_bytes(b'{"x":1,"x":2}', "duplicate probe")
    except RuntimeError:
        duplicate_json_rejected = True
    nan_json_rejected = False
    try:
        strict_json_from_bytes(b'{"x":NaN}', "NaN probe")
    except RuntimeError:
        nan_json_rejected = True

    null_record = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "record_type": "no_event",
        "record_id": "null-test",
        "protocol_sha256": "0" * 64,
        "source_record_ids": [],
        "decision": "empty_within_frozen_scope",
        "reason_codes": ["no_observational_candidate"],
        "created_at_utc": "2000-01-01T00:00:00+00:00",
        "scope_sha256": "1" * 64,
        "prerequisite_status_refs": [],
        "frozen_search_space_id": "space-test",
        "planned_budget": {"candidates": 8},
        "executed_budget": {"candidates": 8},
        "screened_count": 8,
        "qualified_count": 0,
        "search_completed_to_protocol": True,
        "primary_reason": "no_observational_candidate",
        "upstream_status_ref": None,
        "epistemic_scope": "within_frozen_protocol_only",
        "qualified_causal_event_count": 0,
    }
    valid_null_accepted = audit_legal_null_record(null_record)["passed"]
    upstream_as_null = dict(null_record)
    upstream_as_null["upstream_status_ref"] = "prior-no-event"
    upstream_as_direct_null_rejected = not audit_legal_null_record(
        upstream_as_null
    )["passed"]

    instance_common = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "protocol_sha256": "2" * 64,
        "source_record_ids": ["source-1"],
        "reason_codes": ["synthetic_positive_probe"],
        "created_at_utc": "2000-01-01T00:00:00+00:00",
    }
    graph_record = {
        **instance_common,
        "record_type": "effective_graph",
        "record_id": "graph-test",
        "decision": "qualified",
        "anchor_generation_step": 1,
        "generation_window": {"start_step": 0, "end_step": 2},
        "event_generation_steps": [0, 1, 2],
    }
    valid_graph_instance_accepted = audit_effective_graph_instance(
        graph_record
    )["passed"]
    bad_graph_record = json.loads(canonical_json(graph_record))
    bad_graph_record["event_generation_steps"].append(3)
    out_of_window_graph_rejected = not audit_effective_graph_instance(
        bad_graph_record
    )["passed"]

    thread_record = {
        **instance_common,
        "record_type": "instance_thread",
        "record_id": "thread-test",
        "decision": "qualified",
        "referenced_graph_id": "graph-test",
        "referenced_generation_window": {"start_step": 0, "end_step": 3},
        "event_generation_steps": [0, 1, 3],
        "external_anchors": {
            "trigger": 0, "read": 1, "complete": 2, "exit": 3,
        },
    }
    valid_thread_instance_accepted = audit_instance_thread_record(
        thread_record
    )["passed"]
    bad_thread_record = json.loads(canonical_json(thread_record))
    bad_thread_record["external_anchors"]["read"] = 4
    out_of_window_thread_rejected = not audit_instance_thread_record(
        bad_thread_record
    )["passed"]

    checks = {
        "definitions_deterministic": deterministic,
        "cycle_rejected": cycle_rejected,
        "runtime_boundary_tamper_rejected": runtime_rejected,
        "missing_generation_window_rejected": schema_rejects(window_probe),
        "missing_common_record_field_rejected": schema_rejects(envelope_probe),
        "missing_closure_contract_rejected": schema_rejects(closure_probe),
        "upstream_direct_null_cause_rejected": schema_rejects(upstream_probe),
        "pooled_cross_model_denominator_rejected": schema_rejects(pooled_probe),
        "self_hash_alone_accepts_rehash_attack": self_hash_accepts_rehash,
        "exact_rebuild_rejects_rehash_attack": exact_rebuild_rejects_rehash,
        "invalid_timestamp_rejected": timestamp_rejected,
        "duplicate_json_key_rejected": duplicate_json_rejected,
        "nan_json_rejected": nan_json_rejected,
        "opaque_id_is_structurally_unambiguous": (
            opaque_id("probe", "a|b", "c")
            != opaque_id("probe", "a", "b|c")
            and opaque_id("probe", 1) != opaque_id("probe", "1")
        ),
        "valid_direct_null_accepted": valid_null_accepted,
        "upstream_propagation_rejected_as_direct_null": (
            upstream_as_direct_null_rejected
        ),
        "valid_effective_graph_instance_accepted": (
            valid_graph_instance_accepted
        ),
        "out_of_window_graph_instance_rejected": (
            out_of_window_graph_rejected
        ),
        "valid_cross_step_thread_instance_accepted": (
            valid_thread_instance_accepted
        ),
        "out_of_window_thread_instance_rejected": (
            out_of_window_thread_rejected
        ),
        "full_state_forbidden_as_compression_success": any(
            node.get("name") == "task_relative_sufficient_state"
            and node.get("full_state_can_pass_compression") is False
            for node in first["nodes"]
        ),
        "full_physical_graph_forbidden_as_success": any(
            node.get("name") == "effective_graph"
            and node.get("full_physical_graph_is_success") is False
            for node in first["nodes"]
        ),
    }
    require(all(checks.values()), f"Phase990 core self-test failed: {checks}")
    return {"passed": True, "checks": checks}


if __name__ == "__main__":
    print(canonical_json(self_test()))
