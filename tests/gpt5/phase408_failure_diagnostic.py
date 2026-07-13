#!/usr/bin/env python3
"""Explain Phase408 discovery failures without changing the frozen gates."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase408_partition_interface_analysis import enrich_row, response_map
from phase408_partition_interface_collection import read_jsonl, write_json, write_jsonl
from phase408_partition_interface_protocol import (
    FAMILIES,
    INTERFACES,
    MODELS,
    OUT,
    token_words,
)


SURFACES = ("r000", "r001", "r002", "r003")


def condition_status(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    numeric_invalid = sum(row["runtime_numeric_status"] != "valid" for row in rows)
    semantic_counts = Counter(row["semantic_class"] for row in rows)
    mapping, separates, label_aligned = response_map(rows, family)
    if numeric_invalid:
        status = "numeric_invalid"
    elif mapping is None:
        status = "missing_or_nonexclusive_response"
    elif not separates:
        status = "state_collapse"
    elif label_aligned:
        status = "bijective_label_aligned"
    else:
        status = "bijective_relabelled"
    return {
        "model": rows[0]["model"],
        "family_id": family,
        "anonymous_parallel_group_id": rows[0]["anonymous_parallel_group_id"],
        "interface": rows[0]["interface_private"],
        "surface_id": rows[0]["surface_id_private"],
        "lexical_replica": rows[0]["lexical_replica_private"],
        "case_count": len(rows),
        "semantic_class_counts": dict(semantic_counts),
        "runtime_numeric_invalid_count": numeric_invalid,
        "registered_raw_class_count": sum(
            row["raw_response_class"] is not None for row in rows
        ),
        "mapping_private": mapping,
        "separates_states": separates,
        "label_aligned": label_aligned,
        "condition_status": status,
    }


def main() -> None:
    group_path = OUT / "analysis/discovery/phase408_group_audits.jsonl"
    if not group_path.is_file():
        raise FileNotFoundError("Run Phase408 discovery analysis first")
    group_rows = read_jsonl(group_path)
    raw_rows: list[dict[str, Any]] = []
    for model in MODELS:
        raw_rows.extend(
            read_jsonl(
                OUT / "collection/discovery/private" / model / "rows.jsonl"
            )
        )
    rows = [enrich_row(row) for row in raw_rows]
    grouped: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["model"],
            row["family_id"],
            row["anonymous_parallel_group_id"],
            row["interface_private"],
            row["surface_id_private"],
            row["lexical_replica_private"],
        )
        grouped.setdefault(key, []).append(row)
    condition_rows = [
        condition_status(selected, key[1])
        for key, selected in sorted(grouped.items())
    ]

    registry_sets: dict[tuple[str, str, str, int, str], set[tuple[str, ...]]] = {}
    for row in rows:
        key = (
            row["model"],
            row["family_id"],
            row["anonymous_parallel_group_id"],
            row["lexical_replica_private"],
            row["interface_private"],
        )
        if key not in registry_sets:
            registry_sets[key] = {
                token_words(alias)
                for aliases in row["raw_response_aliases_private"].values()
                for alias in aliases
            }
    overlap_audit: list[dict[str, Any]] = []
    for family in FAMILIES:
        contexts = sorted(
            {
                (model, group_id, lexical)
                for model, row_family, group_id, lexical, _interface in registry_sets
                if row_family == family
            }
        )
        for source, target in combinations(INTERFACES[family], 2):
            overlap_contexts = []
            examples: list[str] = []
            for model, group_id, lexical in contexts:
                source_aliases = registry_sets[
                    (model, family, group_id, lexical, source)
                ]
                target_aliases = registry_sets[
                    (model, family, group_id, lexical, target)
                ]
                overlap = source_aliases & target_aliases
                if overlap:
                    overlap_contexts.append(f"{model}:{group_id}:lex{lexical}")
                    for alias in sorted(overlap):
                        value = " ".join(alias)
                        if value not in examples and len(examples) < 8:
                            examples.append(value)
            overlap_audit.append(
                {
                    "family_id": family,
                    "source_interface": source,
                    "target_interface": target,
                    "context_count": len(contexts),
                    "overlap_context_count": len(overlap_contexts),
                    "all_contexts_overlap": len(overlap_contexts) == len(contexts),
                    "overlap_alias_examples": examples,
                }
            )

    failure_axes: list[dict[str, Any]] = []
    for model in MODELS:
        for family in FAMILIES:
            selected_cases = [
                row
                for row in rows
                if row["model"] == model and row["family_id"] == family
            ]
            selected_conditions = [
                row
                for row in condition_rows
                if row["model"] == model and row["family_id"] == family
            ]
            selected_groups = [
                row
                for row in group_rows
                if row["model"] == model and row["family_id"] == family
            ]
            status_counts = Counter(
                row["condition_status"] for row in selected_conditions
            )
            failure_axes.append(
                {
                    "model": model,
                    "family_id": family,
                    "case_count": len(selected_cases),
                    "condition_cell_count": len(selected_conditions),
                    "group_count": len(selected_groups),
                    "semantic_class_counts": dict(
                        Counter(row["semantic_class"] for row in selected_cases)
                    ),
                    "runtime_numeric_status_counts": dict(
                        Counter(
                            row["runtime_numeric_status"] for row in selected_cases
                        )
                    ),
                    "condition_status_counts": dict(status_counts),
                    "condition_separating_count": sum(
                        row["separates_states"] for row in selected_conditions
                    ),
                    "condition_label_aligned_count": sum(
                        row["label_aligned"] for row in selected_conditions
                    ),
                    "all_condition_separation_group_count": sum(
                        row["condition_separation_count"] == row["condition_count"]
                        for row in selected_groups
                    ),
                    "surface_lexical_stability_group_count": sum(
                        row["stable_interface_response_map_count"] == 3
                        for row in selected_groups
                    ),
                    "task_coordinate_covariance_group_count": sum(
                        row["task_coordinate_covariance_pass"]
                        for row in selected_groups
                    ),
                    "functional_group_count": sum(
                        row["functional_partition_interface_pass"]
                        for row in selected_groups
                    ),
                    "dominant_failure_axis": (
                        status_counts.most_common(1)[0][0]
                        if status_counts
                        else "no_authorized_data"
                    ),
                }
            )

    interface_axes: list[dict[str, Any]] = []
    for model in MODELS:
        for family in FAMILIES:
            for interface in INTERFACES[family]:
                selected = [
                    row
                    for row in condition_rows
                    if row["model"] == model
                    and row["family_id"] == family
                    and row["interface"] == interface
                ]
                status_counts = Counter(row["condition_status"] for row in selected)
                interface_axes.append(
                    {
                        "model": model,
                        "family_id": family,
                        "interface": interface,
                        "condition_cell_count": len(selected),
                        "condition_status_counts": dict(status_counts),
                        "condition_separating_count": sum(
                            row["separates_states"] for row in selected
                        ),
                        "condition_label_aligned_count": sum(
                            row["label_aligned"] for row in selected
                        ),
                        "dominant_condition_status": (
                            status_counts.most_common(1)[0][0]
                            if status_counts
                            else "no_authorized_data"
                        ),
                    }
                )

    payload = {
        "schema_version": "82.5.0",
        "phase_id": "Phase408-FailureAxisDiagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_gate_changed": False,
        "case_count": len(rows),
        "condition_cell_count": len(condition_rows),
        "group_count": len(group_rows),
        "semantic_class_counts": dict(
            Counter(row["semantic_class"] for row in rows)
        ),
        "runtime_numeric_status_counts": dict(
            Counter(row["runtime_numeric_status"] for row in rows)
        ),
        "condition_status_counts": dict(
            Counter(row["condition_status"] for row in condition_rows)
        ),
        "condition_separating_count": sum(
            row["separates_states"] for row in condition_rows
        ),
        "condition_label_aligned_count": sum(
            row["label_aligned"] for row in condition_rows
        ),
        "failure_axis_cell_count": len(failure_axes),
        "interface_failure_axis_cell_count": len(interface_axes),
        "cross_interface_alias_overlap_audit": overlap_audit,
        "claim_boundary": {
            "diagnostic_status_is_new_formal_gate": False,
            "bijective_relabelled_is_internal_state_retention": False,
            "dominant_failure_axis_is_language_encoding_mechanism": False,
        },
    }
    write_jsonl(
        OUT / "analysis/private/phase408_condition_cells.jsonl", condition_rows
    )
    write_jsonl(OUT / "analysis/phase408_failure_axes.jsonl", failure_axes)
    write_jsonl(
        OUT / "analysis/phase408_interface_failure_axes.jsonl", interface_axes
    )
    write_json(OUT / "phase408_failure_diagnostic.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
