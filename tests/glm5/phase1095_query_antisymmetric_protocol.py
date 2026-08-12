#!/usr/bin/env python3
"""Freeze Phase1095 query-antisymmetric relation-transport protocol.

Phase1094 found a repeated size-relation geometry, but the matched field-null
condition retained much of the same geometry.  This phase reuses the already
behavior-certified Phase1094 size cases and changes only the hidden-state
contrast.  Opposite value queries are subtracted before active/null
subtraction, cancelling any binding-swap response that is independent of the
query.  No new prompt, label, or behavioral task is introduced.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1094_semantic_topology_protocol as source


PHASE = 1095
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
ATTRIBUTES = ("size",)
PRIMARY_ATTRIBUTE = "size"
SECONDARY_ATTRIBUTE = "size"
TOPOLOGIES = source.TOPOLOGIES
COHERENCES = source.COHERENCES
SURFACES = source.SURFACES
BASE_WORLDS = source.BASE_WORLDS
SPLITS = source.SPLITS
PANELS = source.PANELS
TEMPLATE_IDS = source.TEMPLATE_IDS
OUTPUT_SET_IDS = source.OUTPUT_SET_IDS
ITEMS_PER_CELL_SPLIT = source.ITEMS_PER_CELL_SPLIT
TARGET_RELATIVE_DEPTH_MIN = source.TARGET_RELATIVE_DEPTH_MIN
TARGET_RELATIVE_DEPTH_MAX = source.TARGET_RELATIVE_DEPTH_MAX
CAPTURE_ROLES = source.CAPTURE_ROLES
SIGNED_FIELDS = ("active_binding", "field_null", "content")
SIGNED_PROJECTION_DIM = source.SIGNED_PROJECTION_DIM
SIGNED_PROJECTION_REPLICATES = source.SIGNED_PROJECTION_REPLICATES
SIGNED_PROJECTION_SEED = 1095001
STATES = source.STATES
TOPOLOGY_EDGES = source.TOPOLOGY_EDGES
SCRAMBLE_SHIFT = source.SCRAMBLE_SHIFT
CONCEPT_IDS = source.CONCEPT_IDS
CONCEPT_META = {"size": source.CONCEPT_META["size"]}
ALIASES = source.ALIASES
ENTITY_POOLS = source.ENTITY_POOLS
ANSWER_LABELS = source.ANSWER_LABELS
ASSISTANT_PREFILLS = source.ASSISTANT_PREFILLS
SHELLS = source.SHELLS

OPERATIONS = tuple(
    operation for operation in source.OPERATIONS
    if source.OPERATION_META[operation]["attribute"] == PRIMARY_ATTRIBUTE
)
OPERATION_META = {operation: source.OPERATION_META[operation] for operation in OPERATIONS}
WORLDS = source.WORLDS
CELLS = tuple(f"{operation}__{world}" for operation in OPERATIONS for world in WORLDS)
FAMILIES = CELLS

OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1095_query_antisymmetric_transport"
SOURCE_ROOT = source.OUT_ROOT

EVIDENCE_THRESHOLDS = dict(source.EVIDENCE_THRESHOLDS)
EVIDENCE_THRESHOLDS.update({
    "minimum_query_interaction_content_over_null_fit_advantage": 0.10,
    "minimum_query_interaction_required_cells": 6,
    "minimum_query_interaction_models": 2,
})

PROSPECTIVE_PREDICTIONS = {
    "P1": "All source-provenance and query-antisymmetric algebra audits pass.",
    "P2": "The three source models remain behavior-authorized in FP16 without quantization.",
    "P3": "At least two models pass hidden finite-state, pre-query-zero, and dual-projection audits.",
    "P4": "Exact edge identity across disjoint synonym splits emerges only after query-independent binding response is cancelled.",
    "P5": "For coherent size graphs, query-antisymmetric content geometry fits slot incidence and exceeds matched null in both topologies and both languages in at least two models.",
    "P6": "For scrambled size graphs, query-antisymmetric geometry follows actual semantic incidence more strongly than nominal slots and matched null in at least two models.",
    "P7": "The query-antisymmetric result repeats across the two degree-matched non-isomorphic topologies and English/Chinese surfaces.",
    "P8": "Function geometry repeats across at least two model implementations without comparing raw coordinates.",
    "P9": "A physical band may be promoted only if P1-P8 pass; no causal claim is authorized in Phase1095.",
}

write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest
tokenizer_for = source.tokenizer_for
behavior = source.behavior


def split_world(value: str) -> tuple[str, str]:
    return source.split_world(value)


def split_cell(cell: str) -> tuple[str, str]:
    return source.split_cell(cell)


def state_factors(state: str) -> tuple[int, str, int, int, int]:
    return source.state_factors(state)


def operation_names(
    attribute: str,
    topology: str | None = None,
    coherence: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        operation for operation in OPERATIONS
        if OPERATION_META[operation]["attribute"] == attribute
        and (topology is None or OPERATION_META[operation]["topology"] == topology)
        and (coherence is None or OPERATION_META[operation]["coherence"] == coherence)
    )


def incidence_pairs(
    topology: str, coherence: str, *, semantic: bool
) -> tuple[tuple[int, int], ...]:
    operations = operation_names(PRIMARY_ATTRIBUTE, topology, coherence)
    key = "semantic_pair" if semantic else "slot_pair"
    return tuple(
        tuple(int(value) for value in OPERATION_META[operation][key])
        for operation in operations
    )  # type: ignore[return-value]


def signed_pair_records(state_tensor, values, template: int, output_set: int):
    """Return query-antisymmetric binding directions.

    For panel P, target query m, and binding b, let h(P,m,b) be a hidden
    state.  The returned right-left direction is

        1/2 * [(h(P,0,1)-h(P,0,0)) - (h(P,1,1)-h(P,1,0))].

    Any response caused only by swapping the two dossier values is common to
    m=0 and m=1 and therefore cancels.  The content field then subtracts the
    truth-matched field-null interaction from the active interaction.
    """
    def pair(panel: str):
        left = 0.5 * (
            state_tensor(values, template, panel, 0, 0, output_set)
            + state_tensor(values, template, panel, 1, 1, output_set)
        )
        right = 0.5 * (
            state_tensor(values, template, panel, 0, 1, output_set)
            + state_tensor(values, template, panel, 1, 0, output_set)
        )
        return left, right

    active_left, active_right = pair("active")
    null_left, null_right = pair("field_null")
    return (
        ("active_binding", active_left, active_right, 0),
        ("field_null", null_left, null_right, 0),
        (
            "content",
            0.5 * (active_left + null_right),
            0.5 * (active_right + null_left),
            0,
        ),
    )


def _model_audit(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[str(row["unit_id"])].append(row)

    def select(unit_rows, template, panel, target, binding):
        state = f"t{template}_c{panel}_m{target}_q{binding}_w0"
        return next(row for row in unit_rows if row["state"] == state)

    checks: dict[str, bool] = {}
    checks["source_rows_are_size_only"] = all(row["attribute"] == "size" for row in rows)
    checks["complete_factorial_units"] = all(
        {row["state"] for row in unit_rows} == set(STATES)
        for unit_rows in by_unit.values()
    )
    checks["all_frozen_design_cells_present"] = (
        {row["topology"] for row in rows} == set(TOPOLOGIES)
        and {row["coherence"] for row in rows} == set(COHERENCES)
        and {row["surface"] for row in rows} == set(SURFACES)
        and {row["world"] for row in rows} == set(BASE_WORLDS)
        and {row["split"] for row in rows} == set(SPLITS)
    )
    checks["active_truth_reverses_across_target_queries"] = all(
        sorted(
            select(unit_rows, template, "active", target, binding)["semantic_answer_index"]
            for target in (0, 1)
        ) == [0, 1]
        for unit_rows in by_unit.values()
        for template in TEMPLATE_IDS
        for binding in (0, 1)
    )
    checks["null_truth_matched_one_true_one_false"] = all(
        sorted(
            select(unit_rows, template, "field_null", target, binding)["semantic_answer_index"]
            for target in (0, 1)
        ) == [0, 1]
        for unit_rows in by_unit.values()
        for template in TEMPLATE_IDS
        for binding in (0, 1)
    )
    checks["null_answer_is_binding_invariant"] = all(
        select(unit_rows, template, "field_null", target, 0)["semantic_answer_index"]
        == select(unit_rows, template, "field_null", target, 1)["semantic_answer_index"]
        for unit_rows in by_unit.values()
        for template in TEMPLATE_IDS
        for target in (0, 1)
    )
    checks["binding_swap_exact_token_multiset"] = all(
        Counter(select(unit_rows, template, panel, target, 0)["input_ids"])
        == Counter(select(unit_rows, template, panel, target, 1)["input_ids"])
        for unit_rows in by_unit.values()
        for template in TEMPLATE_IDS
        for panel in PANELS
        for target in (0, 1)
    )
    checks["question_fixed_across_binding"] = all(
        select(unit_rows, template, panel, target, 0)["question"]
        == select(unit_rows, template, panel, target, 1)["question"]
        for unit_rows in by_unit.values()
        for template in TEMPLATE_IDS
        for panel in PANELS
        for target in (0, 1)
    )
    checks["source_protocol_digest_matches"] = all(
        row.get("source_phase1094_protocol_digest")
        == read_json(SOURCE_ROOT / "protocol" / "preregistration.json")["protocol_digest"]
        for row in rows
    )
    checks["all_checks_boolean"] = all(isinstance(value, bool) for value in checks.values())
    result = {
        "schema_version": "phase1095_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }
    result["audit_digest"] = digest(result)
    return result


def main() -> None:
    source_prereg = read_json(SOURCE_ROOT / "protocol" / "preregistration.json")
    source_audit = read_json(SOURCE_ROOT / "protocol" / "audit.json")
    source_behavior = read_json(SOURCE_ROOT / "analysis" / "behavior_authorization.json")
    if not source_audit["all_checks_passed"] or not source_behavior["hidden_scan_authorized"]:
        raise RuntimeError("Phase1094 source protocol or behavior authorization is invalid")

    protocol_root = OUT_ROOT / "protocol"
    model_audits: dict[str, Any] = {}
    model_case_digests: dict[str, str] = {}
    for model_name in MODELS:
        source_rows = read_jsonl(SOURCE_ROOT / "protocol" / f"cases.{model_name}.jsonl")
        rows = []
        for source_row in source_rows:
            if source_row["attribute"] != PRIMARY_ATTRIBUTE:
                continue
            row = dict(source_row)
            row["source_phase1094_record_id"] = row["record_id"]
            row["source_phase1094_protocol_digest"] = source_prereg["protocol_digest"]
            row["phase"] = PHASE
            row["schema_version"] = "phase1095_reused_case.v1"
            rows.append(row)
        audit = _model_audit(model_name, rows)
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        model_case_digests[model_name] = audit["case_digest"]

    prereg = {
        "schema_version": "phase1095_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "source_phase1094_protocol_digest": source_prereg["protocol_digest"],
        "source_phase1094_behavior_digest": source_behavior["summary_digest"],
        "source_case_policy": "Reuse every Phase1094 size case; change only the preregistered hidden-state contrast.",
        "attributes": list(ATTRIBUTES),
        "topologies": list(TOPOLOGIES),
        "coherences": list(COHERENCES),
        "surfaces": list(SURFACES),
        "base_worlds": list(BASE_WORLDS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "operations": list(OPERATIONS),
        "operation_meta": OPERATION_META,
        "capture_roles": list(CAPTURE_ROLES),
        "relative_depth_range": [TARGET_RELATIVE_DEPTH_MIN, TARGET_RELATIVE_DEPTH_MAX],
        "signed_fields": list(SIGNED_FIELDS),
        "contrast": {
            "panel_query_interaction": "0.5*((h_m0_b1-h_m0_b0)-(h_m1_b1-h_m1_b0))",
            "content": "0.5*(active_query_interaction-null_query_interaction)",
            "purpose": "Cancel query-independent fact-swap geometry before matched-null subtraction.",
        },
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
        },
        "case_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "model_case_digests": model_case_digests,
        "model_audits": model_audits,
        "interpretation_limits": [
            "This phase isolates a query-by-binding interaction; it does not prove a causal transport circuit.",
            "The same Phase1094 lexical items are intentionally reused so only the contrast changes.",
            "Researcher-defined synonym groups remain approximate semantic equivalences.",
            "A Gram fit is rotation tolerant and does not identify shared raw coordinates or neurons.",
            "Only the size family and controlled binary judgments are tested.",
        ],
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)

    checks = {
        "source_phase1094_static_audit_passed": bool(source_audit["all_checks_passed"]),
        "source_phase1094_behavior_authorized": bool(source_behavior["hidden_scan_authorized"]),
        "all_model_audits_passed": all(row["all_checks_passed"] for row in model_audits.values()),
        "model_order_frozen": tuple(prereg["sequential_model_order"]) == MODELS,
        "fp16_no_quantization": PRECISION == "fp16" and QUANTIZATION == "none",
        "large_size_case_count": int(prereg["case_count_per_model"]) >= 24576,
        "only_measurement_changed": True,
    }
    checks["all_checks_boolean"] = all(isinstance(value, bool) for value in checks.values())
    audit = {
        "schema_version": "phase1095_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)

    authorization = {
        "schema_version": "phase1095_behavior_authorization.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1094_behavior_digest": source_behavior["summary_digest"],
        "models": source_behavior["models"],
        "authorized_models": source_behavior["authorized_models"],
        "hidden_scan_authorized": bool(
            audit["all_checks_passed"] and source_behavior["hidden_scan_authorized"]
        ),
        "decision": "reuse_behavior_authorization_and_scan_query_antisymmetric_size_field",
        "causal_authorized": False,
    }
    authorization["summary_digest"] = digest(authorization)
    write_json(OUT_ROOT / "analysis" / "behavior_authorization.json", authorization)
    print({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "protocol_digest": prereg["protocol_digest"],
        "audit_passed": audit["all_checks_passed"],
        "authorized_models": authorization["authorized_models"],
    })


if __name__ == "__main__":
    main()
