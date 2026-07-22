#!/usr/bin/env python3
"""Fail-closed Phase991 registry and graph/thread reference resolver.

Phase990 deliberately froze only record shapes.  It did not establish that a
reference names a real, qualified record with the same protocol, gate, scope,
and instance.  This CPU-only module supplies that missing semantic resolution
layer.  It does not import torch/transformers, load model weights, or authorize
GPU execution.
"""
from __future__ import annotations

from copy import deepcopy
import json
import sys
from typing import Any, Iterable, Mapping

import phase990_binding_core as p990


PHASE = 991
SCHEMA_VERSION = 1
EXPERIMENT = "delayed_two_hop_reference_resolution"
ROLE = "cpu_only_fail_closed_reference_registry"
REGISTRY_HASH_FIELD = "registry_sha256"

REGISTRY_FIELDS = {
    "phase", "schema_version", "experiment", "role",
    "phase990_protocol_sha256", "gates", "evidence",
    "effective_graphs", "instance_threads", "closures",
    "created_at_utc", REGISTRY_HASH_FIELD,
}
GATE_FIELDS = {
    "gate_sha256", "protocol_sha256", "model_scope", "split_scope",
    "budget_id", "thresholds_sha256",
}
EVIDENCE_FIELDS = {
    "evidence_id", "evidence_type", "protocol_sha256", "gate_sha256",
    "model_scope", "split_scope", "budget_id", "subject_id",
    "instance_id", "decision", "payload", "payload_sha256",
}
GRAPH_FIELDS = {
    *p990.COMMON_RECORD_FIELDS,
    "gate_sha256", "scope_sha256", "instance_id", "model_scope",
    "split_scope", "budget_id", "anchor_generation_step",
    "generation_window", "event_generation_steps", "qualified_events",
    "input_sha256", "target_id", "intervention_family", "granularity",
    "operational_thresholds", "search_budget",
}
THREAD_FIELDS = {
    *p990.COMMON_RECORD_FIELDS,
    "gate_sha256", "scope_sha256", "instance_id", "model_scope",
    "split_scope", "budget_id", "referenced_graph_id",
    "referenced_generation_window", "event_generation_steps",
    "referenced_event_ids", "external_anchors", "link_kind",
}
CLOSURE_BASE_FIELDS = {
    *p990.COMMON_RECORD_FIELDS,
    *p990.CLOSURE_REQUIRED_FIELDS,
    "pooled_denominator",
}
HEX = frozenset("0123456789abcdef")
ALLOWED_LINK_KINDS = {
    "history_or_logical_kv_available", "history_reread",
    "prefix_reconstruction", "generated_token_feedback", "mixed",
    "undetermined",
}


def _require_exact_fields(
    value: Any,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    p990.require(isinstance(value, Mapping), f"{label} must be an object")
    actual = set(value)
    p990.require(
        actual == expected,
        f"{label} fields differ: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}",
    )
    return value


def _require_sha(value: Any, label: str) -> str:
    p990.require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _require_identifier(value: Any, label: str) -> str:
    p990.require(
        isinstance(value, str) and bool(value.strip()),
        f"{label} must be a nonempty string",
    )
    return value


def _require_scope(
    value: Any,
    universe: Iterable[str],
    label: str,
) -> list[str]:
    allowed = set(universe)
    p990.require(isinstance(value, list) and value, f"{label} must be a list")
    p990.require(
        all(isinstance(item, str) and item in allowed for item in value),
        f"{label} contains an unknown value",
    )
    p990.require(len(value) == len(set(value)), f"{label} contains duplicates")
    return list(value)


def _require_string_list(value: Any, label: str) -> list[str]:
    p990.require(isinstance(value, list), f"{label} must be a list")
    p990.require(
        all(isinstance(item, str) and bool(item) for item in value),
        f"{label} contains an invalid identifier",
    )
    p990.require(len(value) == len(set(value)), f"{label} contains duplicates")
    return list(value)


def _scope_sha(record: Mapping[str, Any]) -> str:
    return p990.sha256_json({
        "protocol_sha256": record["protocol_sha256"],
        "gate_sha256": record["gate_sha256"],
        "instance_id": record["instance_id"],
        "model_scope": record["model_scope"],
        "split_scope": record["split_scope"],
        "budget_id": record["budget_id"],
    })


def _seal_registry_payload(
    payload: Mapping[str, Any],
    created_at_utc: str = "2000-01-01T00:00:00+00:00",
) -> dict[str, Any]:
    return p990.sealed_document(
        payload,
        REGISTRY_HASH_FIELD,
        created_at_utc=created_at_utc,
    )


def _reseal_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    timestamp = str(registry["created_at_utc"])
    payload = {
        key: deepcopy(value)
        for key, value in registry.items()
        if key not in {"created_at_utc", REGISTRY_HASH_FIELD}
    }
    return _seal_registry_payload(payload, timestamp)


def _index_unique(
    rows: Any,
    id_field: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    p990.require(isinstance(rows, list), f"{label} registry must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for offset, row in enumerate(rows):
        p990.require(isinstance(row, Mapping), f"{label}[{offset}] is not an object")
        identifier = _require_identifier(row.get(id_field), f"{label}[{offset}].{id_field}")
        p990.require(identifier not in result, f"duplicate {label} id: {identifier}")
        result[identifier] = row
    return result


def _validate_gate(
    gate: Mapping[str, Any],
    protocol_sha256: str,
) -> None:
    _require_exact_fields(gate, GATE_FIELDS, "gate")
    _require_sha(gate["gate_sha256"], "gate.gate_sha256")
    p990.require(
        gate["protocol_sha256"] == protocol_sha256,
        "gate protocol differs from registry protocol",
    )
    _require_scope(gate["model_scope"], p990.MODEL_ORDER, "gate.model_scope")
    _require_scope(gate["split_scope"], p990.SPLIT_ORDER, "gate.split_scope")
    _require_identifier(gate["budget_id"], "gate.budget_id")
    _require_sha(gate["thresholds_sha256"], "gate.thresholds_sha256")
    body = {key: value for key, value in gate.items() if key != "gate_sha256"}
    p990.require(
        gate["gate_sha256"] == p990.sha256_json(body),
        "gate content hash mismatch",
    )


def _validate_evidence_shape(
    evidence: Mapping[str, Any],
    protocol_sha256: str,
    gates: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_exact_fields(evidence, EVIDENCE_FIELDS, "evidence")
    _require_identifier(evidence["evidence_id"], "evidence.evidence_id")
    _require_identifier(evidence["evidence_type"], "evidence.evidence_type")
    p990.require(
        evidence["protocol_sha256"] == protocol_sha256,
        "evidence protocol differs from registry protocol",
    )
    gate_id = _require_sha(evidence["gate_sha256"], "evidence.gate_sha256")
    p990.require(gate_id in gates, "evidence gate does not resolve")
    model_scope = _require_scope(
        evidence["model_scope"], p990.MODEL_ORDER, "evidence.model_scope"
    )
    split_scope = _require_scope(
        evidence["split_scope"], p990.SPLIT_ORDER, "evidence.split_scope"
    )
    gate = gates[gate_id]
    p990.require(
        set(model_scope) <= set(gate["model_scope"]),
        "evidence model scope escapes its gate",
    )
    p990.require(
        set(split_scope) <= set(gate["split_scope"]),
        "evidence split scope escapes its gate",
    )
    p990.require(
        evidence["budget_id"] == gate["budget_id"],
        "evidence budget differs from gate",
    )
    _require_identifier(evidence["subject_id"], "evidence.subject_id")
    _require_identifier(evidence["instance_id"], "evidence.instance_id")
    p990.require(
        evidence["decision"] in {"qualified", "rejected", "inconclusive"},
        "evidence decision is invalid",
    )
    _require_sha(evidence["payload_sha256"], "evidence.payload_sha256")
    p990.require(
        isinstance(evidence["payload"], dict) and bool(evidence["payload"]),
        "evidence payload is empty",
    )
    p990.require(
        p990.sha256_json(evidence["payload"]) == evidence["payload_sha256"],
        "evidence payload hash mismatch",
    )


def _validate_graph_shape(
    graph: Mapping[str, Any],
    protocol_sha256: str,
    gates: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_exact_fields(graph, GRAPH_FIELDS, "effective graph")
    report = p990.audit_effective_graph_instance(graph)
    p990.require(report["passed"], f"effective graph shape failed: {report['errors']}")
    p990.require(
        graph["protocol_sha256"] == protocol_sha256,
        "effective graph protocol differs from registry protocol",
    )
    gate_id = _require_sha(graph["gate_sha256"], "effective graph gate")
    p990.require(gate_id in gates, "effective graph gate does not resolve")
    models = _require_scope(
        graph["model_scope"], p990.MODEL_ORDER, "effective graph model_scope"
    )
    p990.require(len(models) == 1, "an effective graph must belong to one model")
    splits = _require_scope(
        graph["split_scope"], p990.SPLIT_ORDER, "effective graph split_scope"
    )
    gate = gates[gate_id]
    p990.require(set(models) <= set(gate["model_scope"]), "graph model escapes gate")
    p990.require(set(splits) <= set(gate["split_scope"]), "graph split escapes gate")
    p990.require(graph["budget_id"] == gate["budget_id"], "graph budget differs")
    _require_identifier(graph["instance_id"], "effective graph instance_id")
    _require_sha(graph["scope_sha256"], "effective graph scope_sha256")
    p990.require(graph["scope_sha256"] == _scope_sha(graph), "graph scope hash mismatch")
    _require_sha(graph["input_sha256"], "effective graph input_sha256")
    for field in ("target_id", "intervention_family", "granularity"):
        _require_identifier(graph[field], f"effective graph {field}")
    p990.require(
        isinstance(graph["operational_thresholds"], dict)
        and bool(graph["operational_thresholds"]),
        "effective graph thresholds are empty",
    )
    p990.require(
        p990.sha256_json(graph["operational_thresholds"])
        == gate["thresholds_sha256"],
        "effective graph thresholds differ from gate",
    )
    p990.require(
        isinstance(graph["search_budget"], dict) and bool(graph["search_budget"]),
        "effective graph search budget is empty",
    )
    events = graph["qualified_events"]
    p990.require(isinstance(events, dict) and events, "graph event registry is empty")
    for event_id, step in events.items():
        _require_identifier(event_id, "qualified event id")
        p990.require(
            isinstance(step, int) and not isinstance(step, bool) and step >= 0,
            "qualified event step is invalid",
        )
    graph_steps = graph["event_generation_steps"]
    p990.require(
        graph_steps == sorted(set(events.values())),
        "graph event_generation_steps do not exactly summarize qualified_events",
    )


def _validate_thread_shape(
    thread: Mapping[str, Any],
    protocol_sha256: str,
    gates: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_exact_fields(thread, THREAD_FIELDS, "instance thread")
    report = p990.audit_instance_thread_record(thread)
    p990.require(report["passed"], f"instance thread shape failed: {report['errors']}")
    p990.require(
        thread["protocol_sha256"] == protocol_sha256,
        "thread protocol differs from registry protocol",
    )
    gate_id = _require_sha(thread["gate_sha256"], "thread gate")
    p990.require(gate_id in gates, "thread gate does not resolve")
    models = _require_scope(
        thread["model_scope"], p990.MODEL_ORDER, "thread model_scope"
    )
    p990.require(len(models) == 1, "an instance thread must belong to one model")
    splits = _require_scope(
        thread["split_scope"], p990.SPLIT_ORDER, "thread split_scope"
    )
    gate = gates[gate_id]
    p990.require(set(models) <= set(gate["model_scope"]), "thread model escapes gate")
    p990.require(set(splits) <= set(gate["split_scope"]), "thread split escapes gate")
    p990.require(thread["budget_id"] == gate["budget_id"], "thread budget differs")
    _require_identifier(thread["instance_id"], "thread instance_id")
    _require_sha(thread["scope_sha256"], "thread scope_sha256")
    p990.require(thread["scope_sha256"] == _scope_sha(thread), "thread scope hash mismatch")
    _require_string_list(thread["referenced_event_ids"], "referenced_event_ids")
    p990.require(thread["link_kind"] in ALLOWED_LINK_KINDS, "thread link kind is invalid")


def _closure_fields(closure_type: Any) -> set[str]:
    fields = set(CLOSURE_BASE_FIELDS)
    if closure_type == "cross_model_replicated":
        fields.update({
            "target_closure_type", "per_model_certificate_refs",
            "per_model_denominators",
        })
    return fields


def _registered_denominator(record: Mapping[str, Any]) -> int:
    counts = record["counts_by_semantic_world_and_stratum"]
    p990.require(isinstance(counts, dict), "closure counts must be an object")
    value = counts.get("denominator")
    p990.require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        "closure counts require a positive denominator",
    )
    return int(value)


def _validate_closure_shape(
    closure: Mapping[str, Any],
    protocol_sha256: str,
    gates: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_exact_fields(
        closure,
        _closure_fields(closure.get("closure_type")),
        "closure certificate",
    )
    report = p990.audit_closure_certificate(closure)
    p990.require(report["passed"], f"closure shape failed: {report['errors']}")
    p990.require(
        closure["protocol_sha256"] == protocol_sha256,
        "closure protocol differs from registry protocol",
    )
    p990.require(
        closure["decision"] in {"qualified", "rejected", "inconclusive"},
        "closure decision cannot be resolved",
    )
    gate_id = _require_sha(closure["gate_sha256"], "closure gate")
    p990.require(gate_id in gates, "closure gate does not resolve")
    models = _require_scope(
        closure["model_scope"], p990.MODEL_ORDER, "closure model_scope"
    )
    splits = _require_scope(
        closure["split_scope"], p990.SPLIT_ORDER, "closure split_scope"
    )
    gate = gates[gate_id]
    p990.require(set(models) <= set(gate["model_scope"]), "closure model escapes gate")
    p990.require(set(splits) <= set(gate["split_scope"]), "closure split escapes gate")
    p990.require(closure["budget_id"] == gate["budget_id"], "closure budget differs")
    p990.require(
        p990.sha256_json(closure["thresholds"]) == gate["thresholds_sha256"],
        "closure thresholds differ from gate",
    )
    p990.require(closure["pooled_denominator"] is False, "pooled denominator enabled")
    _registered_denominator(closure)


def _assert_sources_resolve(
    record: Mapping[str, Any],
    all_records: Mapping[str, Mapping[str, Any]],
) -> None:
    for source_id in _require_string_list(
        record["source_record_ids"], f"{record['record_id']} source_record_ids"
    ):
        p990.require(source_id != record["record_id"], "record directly cites itself")
        p990.require(source_id in all_records, f"source record does not resolve: {source_id}")


def _resolve_threads(
    threads: Mapping[str, Mapping[str, Any]],
    graphs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    qualified: list[str] = []
    for thread_id, thread in threads.items():
        graph_id = str(thread["referenced_graph_id"])
        p990.require(graph_id in graphs, f"thread graph does not resolve: {graph_id}")
        graph = graphs[graph_id]
        p990.require(graph["decision"] == "qualified", "thread graph is not qualified")
        for field in (
            "protocol_sha256", "gate_sha256", "scope_sha256", "instance_id",
            "model_scope", "split_scope", "budget_id",
        ):
            p990.require(thread[field] == graph[field], f"thread/graph {field} mismatch")
        p990.require(
            thread["referenced_generation_window"] == graph["generation_window"],
            "thread window differs from resolved graph",
        )
        event_ids = thread["referenced_event_ids"]
        graph_events = graph["qualified_events"]
        p990.require(
            all(event_id in graph_events for event_id in event_ids),
            "thread cites an event outside the resolved graph",
        )
        expected_steps = [graph_events[event_id] for event_id in event_ids]
        p990.require(
            thread["event_generation_steps"] == expected_steps,
            "thread event steps differ from resolved graph events",
        )
        p990.require(
            graph_id in thread["source_record_ids"],
            "thread source lineage omits its resolved graph",
        )
        if thread["decision"] == "qualified":
            p990.require(len(set(expected_steps)) >= 2, "qualified thread is not cross-step")
            qualified.append(thread_id)
    return sorted(qualified)


def _resolve_graph_sources(
    graphs: Mapping[str, Mapping[str, Any]],
    closures: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Require every qualified graph to descend from a qualified edge gate."""
    qualified: list[str] = []
    for graph_id, graph in graphs.items():
        if graph["decision"] != "qualified":
            continue
        sources = [
            closures[source_id]
            for source_id in graph["source_record_ids"]
            if source_id in closures
            and closures[source_id]["closure_type"] == "edge_qualified"
            and closures[source_id]["decision"] == "qualified"
        ]
        p990.require(sources, "qualified graph lacks a qualified edge certificate")
        for source in sources:
            for field in (
                "protocol_sha256", "gate_sha256", "model_scope",
                "split_scope", "budget_id",
            ):
                p990.require(
                    source[field] == graph[field],
                    f"qualified graph edge source {field} mismatch",
                )
        qualified.append(graph_id)
    return sorted(qualified)


def _resolve_closures(
    closures: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    qualified: list[str] = []
    for closure_id, closure in closures.items():
        contract = p990.definitions_payload()["closure_contracts"][
            closure["closure_type"]
        ]
        prerequisite_ids = closure["prerequisite_closure_ids"]
        expected_types = contract["prerequisite_closure_types"]
        for prerequisite_id, expected_type in zip(
            prerequisite_ids, expected_types, strict=True
        ):
            p990.require(
                prerequisite_id in closures,
                f"closure prerequisite does not resolve: {prerequisite_id}",
            )
            prerequisite = closures[prerequisite_id]
            p990.require(
                prerequisite["decision"] == "qualified",
                "closure prerequisite is not qualified",
            )
            p990.require(
                prerequisite["closure_type"] == expected_type,
                "closure prerequisite type mismatch",
            )
            for field in (
                "protocol_sha256", "gate_sha256", "model_scope",
                "split_scope", "budget_id",
            ):
                p990.require(
                    closure[field] == prerequisite[field],
                    f"closure prerequisite {field} mismatch",
                )

        for evidence_type, evidence_id in closure["evidence_refs"].items():
            p990.require(
                isinstance(evidence_id, str) and evidence_id in evidence,
                f"closure evidence does not resolve: {evidence_id}",
            )
            resolved = evidence[evidence_id]
            p990.require(resolved["decision"] == "qualified", "closure evidence rejected")
            p990.require(
                resolved["evidence_type"] == evidence_type,
                "closure evidence type mismatch",
            )
            for field in (
                "protocol_sha256", "gate_sha256", "model_scope",
                "split_scope", "budget_id",
            ):
                p990.require(
                    resolved[field] == closure[field],
                    f"closure evidence {field} mismatch",
                )
            p990.require(
                resolved["subject_id"] == closure["subject_id"],
                "closure evidence subject mismatch",
            )

        if closure["closure_type"] == "cross_model_replicated":
            target_type = closure["target_closure_type"]
            refs = closure["per_model_certificate_refs"]
            denominators = closure["per_model_denominators"]
            for model in p990.MODEL_ORDER:
                reference = refs[model]
                p990.require(reference in closures, "cross-model certificate missing")
                resolved = closures[reference]
                p990.require(resolved["decision"] == "qualified", "cross-model source rejected")
                p990.require(
                    resolved["closure_type"] == target_type,
                    "cross-model target closure type mismatch",
                )
                p990.require(
                    resolved["model_scope"] == [model],
                    "cross-model certificate has the wrong model scope",
                )
                for field in (
                    "protocol_sha256", "gate_sha256", "split_scope", "budget_id"
                ):
                    p990.require(
                        resolved[field] == closure[field],
                        f"cross-model certificate {field} mismatch",
                    )
                p990.require(
                    denominators[model] == _registered_denominator(resolved),
                    "cross-model denominator differs from resolved certificate",
                )
            p990.require(
                closure["model_scope"] == list(p990.MODEL_ORDER),
                "cross-model closure scope is incomplete",
            )

        # The closure gate has already been content-hash validated.  This read
        # makes the dependency explicit in the resolution graph.
        p990.require(closure["gate_sha256"] in gates, "closure gate disappeared")
        if closure["decision"] == "qualified":
            qualified.append(closure_id)
    return sorted(qualified)


def _assert_dependency_acyclic(
    evidence: Mapping[str, Mapping[str, Any]],
    graphs: Mapping[str, Mapping[str, Any]],
    threads: Mapping[str, Mapping[str, Any]],
    closures: Mapping[str, Mapping[str, Any]],
) -> None:
    records: dict[str, Mapping[str, Any]] = {
        **evidence, **graphs, **threads, **closures,
    }
    dependencies: dict[str, set[str]] = {}
    for record_id, record in records.items():
        refs = set(record.get("source_record_ids", []))
        if record_id in closures:
            refs.update(record["prerequisite_closure_ids"])
            refs.update(record["evidence_refs"].values())
            if record["closure_type"] == "cross_model_replicated":
                refs.update(record["per_model_certificate_refs"].values())
        if record_id in threads:
            refs.add(record["referenced_graph_id"])
        dependencies[record_id] = {ref for ref in refs if ref in records}

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(record_id: str) -> None:
        p990.require(record_id not in visiting, "reference dependency cycle detected")
        if record_id in visited:
            return
        visiting.add(record_id)
        for dependency in sorted(dependencies[record_id]):
            visit(dependency)
        visiting.remove(record_id)
        visited.add(record_id)

    for identifier in sorted(records):
        visit(identifier)


def resolve_registry(
    registry: Mapping[str, Any],
    *,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    """Validate and resolve a complete registry or raise ``RuntimeError``."""
    _require_exact_fields(registry, REGISTRY_FIELDS, "reference registry")
    p990.verify_self_hash(registry, REGISTRY_HASH_FIELD, "reference registry")
    p990.require(registry["phase"] == PHASE, "registry phase changed")
    p990.require(registry["schema_version"] == SCHEMA_VERSION, "registry schema changed")
    p990.require(registry["experiment"] == EXPERIMENT, "registry experiment changed")
    p990.require(registry["role"] == ROLE, "registry role changed")
    protocol_sha256 = _require_sha(
        registry["phase990_protocol_sha256"], "registry phase990 protocol"
    )
    _require_sha(expected_protocol_sha256, "externally pinned phase990 protocol")
    p990.require(
        protocol_sha256 == expected_protocol_sha256,
        "registry protocol differs from the externally pinned Phase990 protocol",
    )

    gates = _index_unique(registry["gates"], "gate_sha256", "gate")
    evidence = _index_unique(registry["evidence"], "evidence_id", "evidence")
    graphs = _index_unique(
        registry["effective_graphs"], "record_id", "effective graph"
    )
    threads = _index_unique(
        registry["instance_threads"], "record_id", "instance thread"
    )
    closures = _index_unique(registry["closures"], "record_id", "closure")
    p990.require(gates, "at least one gate is required")

    global_ids: set[str] = set()
    for label, index in (
        ("evidence", evidence), ("effective graph", graphs),
        ("instance thread", threads), ("closure", closures),
    ):
        overlap = global_ids & set(index)
        p990.require(not overlap, f"global record id collision in {label}: {sorted(overlap)}")
        global_ids.update(index)

    for gate in gates.values():
        _validate_gate(gate, protocol_sha256)
    for row in evidence.values():
        _validate_evidence_shape(row, protocol_sha256, gates)
    for row in graphs.values():
        _validate_graph_shape(row, protocol_sha256, gates)
    for row in threads.values():
        _validate_thread_shape(row, protocol_sha256, gates)
    for row in closures.values():
        _validate_closure_shape(row, protocol_sha256, gates)

    all_records = {**evidence, **graphs, **threads, **closures}
    for row in [*graphs.values(), *threads.values(), *closures.values()]:
        _assert_sources_resolve(row, all_records)
    _assert_dependency_acyclic(evidence, graphs, threads, closures)

    qualified_graphs = _resolve_graph_sources(graphs, closures)
    qualified_threads = _resolve_threads(threads, graphs)
    qualified_closures = _resolve_closures(closures, evidence, gates)
    return {
        "passed": True,
        "registry_sha256": registry[REGISTRY_HASH_FIELD],
        "counts": {
            "gates": len(gates),
            "evidence": len(evidence),
            "effective_graphs": len(graphs),
            "instance_threads": len(threads),
            "closures": len(closures),
        },
        "qualified_graph_ids": qualified_graphs,
        "qualified_thread_ids": qualified_threads,
        "qualified_closure_ids": qualified_closures,
        "cuda_used": False,
        "model_weights_loaded": False,
    }


def _common(
    record_type: str,
    record_id: str,
    protocol_sha256: str,
    sources: list[str],
    decision: str = "qualified",
) -> dict[str, Any]:
    return {
        "phase": p990.PHASE,
        "schema_version": p990.SCHEMA_VERSION,
        "experiment": p990.EXPERIMENT,
        "record_type": record_type,
        "record_id": record_id,
        "protocol_sha256": protocol_sha256,
        "source_record_ids": sources,
        "decision": decision,
        "reason_codes": ["synthetic_resolver_fixture"],
        "created_at_utc": "2000-01-01T00:00:00+00:00",
    }


def _make_evidence(
    evidence_id: str,
    evidence_type: str,
    protocol_sha256: str,
    gate_sha256: str,
    model_scope: list[str],
    split_scope: list[str],
    budget_id: str,
    subject_id: str,
    instance_id: str,
) -> dict[str, Any]:
    payload = {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "synthetic_fixture": True,
    }
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "protocol_sha256": protocol_sha256,
        "gate_sha256": gate_sha256,
        "model_scope": model_scope,
        "split_scope": split_scope,
        "budget_id": budget_id,
        "subject_id": subject_id,
        "instance_id": instance_id,
        "decision": "qualified",
        "payload": payload,
        "payload_sha256": p990.sha256_json(payload),
    }


def _make_closure(
    closure_id: str,
    closure_type: str,
    protocol_sha256: str,
    gate_sha256: str,
    thresholds: dict[str, Any],
    model_scope: list[str],
    split_scope: list[str],
    budget_id: str,
    subject_type: str,
    subject_id: str,
    prerequisite_ids: list[str],
    evidence_refs: dict[str, str],
    denominator: int = 32,
) -> dict[str, Any]:
    record = {
        **_common(
            "closure_certificate",
            closure_id,
            protocol_sha256,
            [*prerequisite_ids, *evidence_refs.values()],
        ),
        "closure_id": closure_id,
        "closure_type": closure_type,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "gate_sha256": gate_sha256,
        "prerequisite_closure_ids": prerequisite_ids,
        "prerequisite_closure_types": list(
            p990.CLOSURE_PREREQUISITE_TYPES[closure_type]
        ),
        "model_scope": model_scope,
        "split_scope": split_scope,
        "thresholds": thresholds,
        "budget_id": budget_id,
        "evidence_refs": evidence_refs,
        "counts_by_semantic_world_and_stratum": {
            "denominator": denominator,
            "qualified": denominator,
        },
        "pooled_denominator": False,
    }
    return record


def build_positive_fixture() -> dict[str, Any]:
    """Construct a deterministic positive registry used only by self-tests."""
    protocol_sha256 = "a" * 64
    thresholds = {"minimum_passes": 26, "denominator": 32}
    budget_id = "fixture-budget-v1"
    split_scope = ["confirmation"]
    gate_body = {
        "protocol_sha256": protocol_sha256,
        "model_scope": list(p990.MODEL_ORDER),
        "split_scope": split_scope,
        "budget_id": budget_id,
        "thresholds_sha256": p990.sha256_json(thresholds),
    }
    gate = {**gate_body, "gate_sha256": p990.sha256_json(gate_body)}
    gate_sha256 = gate["gate_sha256"]

    evidence_rows: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    replay_ids: dict[str, str] = {}
    for model in p990.MODEL_ORDER:
        closure_id = f"replay-{model}"
        subject_id = f"run-{model}"
        refs: dict[str, str] = {}
        for evidence_type in p990.CLOSURE_EVIDENCE_FIELDS["replay_integrity"]:
            evidence_id = f"ev-{model}-replay-{evidence_type}"
            refs[evidence_type] = evidence_id
            evidence_rows.append(_make_evidence(
                evidence_id, evidence_type, protocol_sha256, gate_sha256,
                [model], split_scope, budget_id, subject_id, f"instance-{model}",
            ))
        closures.append(_make_closure(
            closure_id, "replay_integrity", protocol_sha256, gate_sha256,
            thresholds, [model], split_scope, budget_id, "run", subject_id,
            [], refs,
        ))
        replay_ids[model] = closure_id

    edge_refs: dict[str, str] = {}
    for evidence_type in p990.CLOSURE_EVIDENCE_FIELDS["edge_qualified"]:
        evidence_id = f"ev-qwen-edge-{evidence_type}"
        edge_refs[evidence_type] = evidence_id
        evidence_rows.append(_make_evidence(
            evidence_id, evidence_type, protocol_sha256, gate_sha256,
            ["qwen3"], split_scope, budget_id, "edge-qwen", "instance-qwen3",
        ))
    edge_closure = _make_closure(
        "edge-qwen", "edge_qualified", protocol_sha256, gate_sha256,
        thresholds, ["qwen3"], split_scope, budget_id, "causal_edge",
        "edge-qwen", [replay_ids["qwen3"]], edge_refs,
    )
    closures.append(edge_closure)

    cross_refs: dict[str, str] = {}
    for evidence_type in p990.CLOSURE_EVIDENCE_FIELDS["cross_model_replicated"]:
        evidence_id = f"ev-cross-{evidence_type}"
        cross_refs[evidence_type] = evidence_id
        evidence_rows.append(_make_evidence(
            evidence_id, evidence_type, protocol_sha256, gate_sha256,
            list(p990.MODEL_ORDER), split_scope, budget_id,
            "cross-replay", "cross-model-instance",
        ))
    cross = _make_closure(
        "cross-replay", "cross_model_replicated", protocol_sha256,
        gate_sha256, thresholds, list(p990.MODEL_ORDER), split_scope,
        budget_id, "abstract_cross_model_claim", "cross-replay", [],
        cross_refs,
    )
    cross.update({
        "target_closure_type": "replay_integrity",
        "per_model_certificate_refs": dict(replay_ids),
        "per_model_denominators": {model: 32 for model in p990.MODEL_ORDER},
    })
    cross["source_record_ids"] = [*cross["source_record_ids"], *replay_ids.values()]
    closures.append(cross)

    graph = {
        **_common(
            "effective_graph", "graph-qwen", protocol_sha256, ["edge-qwen"]
        ),
        "gate_sha256": gate_sha256,
        "scope_sha256": "0" * 64,
        "instance_id": "instance-qwen3",
        "model_scope": ["qwen3"],
        "split_scope": split_scope,
        "budget_id": budget_id,
        "anchor_generation_step": 1,
        "generation_window": {"start_step": 0, "end_step": 2},
        "event_generation_steps": [0, 1, 2],
        "qualified_events": {"event-0": 0, "event-1": 1, "event-2": 2},
        "input_sha256": "b" * 64,
        "target_id": "answer-value",
        "intervention_family": "matched-residual-state",
        "granularity": "relative-layer-block",
        "operational_thresholds": thresholds,
        "search_budget": {"candidate_blocks": 4},
    }
    graph["scope_sha256"] = _scope_sha(graph)
    thread = {
        **_common(
            "instance_thread", "thread-qwen", protocol_sha256,
            ["graph-qwen"],
        ),
        "gate_sha256": gate_sha256,
        "scope_sha256": graph["scope_sha256"],
        "instance_id": graph["instance_id"],
        "model_scope": graph["model_scope"],
        "split_scope": graph["split_scope"],
        "budget_id": graph["budget_id"],
        "referenced_graph_id": graph["record_id"],
        "referenced_generation_window": deepcopy(graph["generation_window"]),
        "event_generation_steps": [0, 2],
        "referenced_event_ids": ["event-0", "event-2"],
        "external_anchors": {"trigger": 0, "read": 1, "complete": 2, "exit": 2},
        "link_kind": "generated_token_feedback",
    }
    payload = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": ROLE,
        "phase990_protocol_sha256": protocol_sha256,
        "gates": [gate],
        "evidence": evidence_rows,
        "effective_graphs": [graph],
        "instance_threads": [thread],
        "closures": closures,
    }
    return _seal_registry_payload(payload)


def _find(rows: list[dict[str, Any]], field: str, value: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get(field) == value]
    p990.require(len(matches) == 1, f"fixture lookup failed: {field}={value}")
    return matches[0]


def self_test() -> dict[str, Any]:
    positive = build_positive_fixture()
    expected_protocol = str(positive["phase990_protocol_sha256"])
    first = resolve_registry(
        positive, expected_protocol_sha256=expected_protocol
    )
    second = resolve_registry(
        build_positive_fixture(), expected_protocol_sha256=expected_protocol
    )
    p990.require(first == second, "resolver positive fixture is nondeterministic")

    mutations: dict[str, Any] = {}

    def register(name: str, mutate: Any, reseal: bool = True) -> None:
        candidate = deepcopy(positive)
        mutate(candidate)
        if reseal:
            candidate = _reseal_registry(candidate)
        rejected = False
        try:
            resolve_registry(
                candidate, expected_protocol_sha256=expected_protocol
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            rejected = True
        p990.require(rejected, f"mutation was accepted: {name}")
        mutations[name] = True

    register(
        "registry_self_hash_tamper",
        lambda value: value.__setitem__("role", "tampered"),
        reseal=False,
    )
    register(
        "unknown_registry_field",
        lambda value: value.__setitem__("unexpected", True),
    )
    register(
        "externally_pinned_protocol_mismatch",
        lambda value: value.__setitem__("phase990_protocol_sha256", "c" * 64),
    )
    register(
        "duplicate_global_record_id",
        lambda value: value["evidence"].append(deepcopy(value["evidence"][0])),
    )
    register(
        "gate_content_rehash_missing",
        lambda value: value["gates"][0].__setitem__("budget_id", "other-budget"),
    )
    register(
        "missing_closure_prerequisite",
        lambda value: _find(value["closures"], "record_id", "edge-qwen")[
            "prerequisite_closure_ids"
        ].__setitem__(0, "missing-replay"),
    )
    register(
        "unqualified_closure_prerequisite",
        lambda value: _find(value["closures"], "record_id", "replay-qwen3").__setitem__(
            "decision", "rejected"
        ),
    )
    register(
        "wrong_closure_prerequisite_type",
        lambda value: _find(value["closures"], "record_id", "edge-qwen")[
            "prerequisite_closure_ids"
        ].__setitem__(0, "edge-qwen"),
    )
    register(
        "closure_protocol_mismatch",
        lambda value: _find(value["closures"], "record_id", "edge-qwen").__setitem__(
            "protocol_sha256", "c" * 64
        ),
    )
    register(
        "closure_gate_missing",
        lambda value: _find(value["closures"], "record_id", "edge-qwen").__setitem__(
            "gate_sha256", "d" * 64
        ),
    )
    register(
        "closure_evidence_missing",
        lambda value: _find(value["closures"], "record_id", "edge-qwen")[
            "evidence_refs"
        ].__setitem__("target_damage", "missing-evidence"),
    )
    register(
        "closure_evidence_wrong_type",
        lambda value: _find(
            value["evidence"], "evidence_id",
            _find(value["closures"], "record_id", "edge-qwen")["evidence_refs"][
                "target_damage"
            ],
        ).__setitem__("evidence_type", "wrong-type"),
    )
    register(
        "closure_evidence_payload_hash_mismatch",
        lambda value: _find(
            value["evidence"], "evidence_id",
            _find(value["closures"], "record_id", "edge-qwen")["evidence_refs"][
                "target_damage"
            ],
        )["payload"].__setitem__("tampered", True),
    )
    register(
        "closure_evidence_scope_mismatch",
        lambda value: _find(
            value["evidence"], "evidence_id",
            _find(value["closures"], "record_id", "edge-qwen")["evidence_refs"][
                "target_damage"
            ],
        ).__setitem__("model_scope", ["glm4"]),
    )
    register(
        "cross_model_reference_missing",
        lambda value: _find(value["closures"], "record_id", "cross-replay")[
            "per_model_certificate_refs"
        ].__setitem__("glm4", "missing-glm-certificate"),
    )
    register(
        "cross_model_target_type_mismatch",
        lambda value: _find(value["closures"], "record_id", "cross-replay").__setitem__(
            "target_closure_type", "edge_qualified"
        ),
    )
    register(
        "cross_model_denominator_mismatch",
        lambda value: _find(value["closures"], "record_id", "cross-replay")[
            "per_model_denominators"
        ].__setitem__("deepseek7b", 31),
    )
    register(
        "thread_graph_missing",
        lambda value: value["instance_threads"][0].__setitem__(
            "referenced_graph_id", "missing-graph"
        ),
    )
    register(
        "thread_graph_not_qualified",
        lambda value: value["effective_graphs"][0].__setitem__(
            "decision", "inconclusive"
        ),
    )
    register(
        "qualified_graph_without_edge_certificate",
        lambda value: value["effective_graphs"][0]["source_record_ids"].__setitem__(
            0, "replay-qwen3"
        ),
    )
    register(
        "thread_window_mismatch",
        lambda value: value["instance_threads"][0][
            "referenced_generation_window"
        ].__setitem__("end_step", 1),
    )
    register(
        "thread_foreign_event",
        lambda value: value["instance_threads"][0]["referenced_event_ids"].__setitem__(
            1, "foreign-event"
        ),
    )
    register(
        "thread_event_step_mismatch",
        lambda value: value["instance_threads"][0]["event_generation_steps"].__setitem__(
            1, 1
        ),
    )
    register(
        "thread_instance_mismatch",
        lambda value: value["instance_threads"][0].__setitem__(
            "instance_id", "other-instance"
        ),
    )
    register(
        "thread_source_cycle",
        lambda value: value["effective_graphs"][0]["source_record_ids"].__setitem__(
            0, "thread-qwen"
        ),
    )

    return {
        "passed": True,
        "positive_registry_sha256": positive[REGISTRY_HASH_FIELD],
        "positive_counts": first["counts"],
        "positive_qualified_graphs": first["qualified_graph_ids"],
        "positive_qualified_threads": first["qualified_thread_ids"],
        "positive_qualified_closures": first["qualified_closure_ids"],
        "mutation_rejection_count": len(mutations),
        "mutation_rejections": dict(sorted(mutations.items())),
        "cuda_used": False,
        "model_weights_loaded": False,
    }


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments != ["--self-test"]:
        raise SystemExit("usage: phase991_reference_resolver.py --self-test")
    print(p990.canonical_json(self_test()))


if __name__ == "__main__":
    main()
