#!/usr/bin/env python3
"""Classify Phase407 state-response maps without changing formal success gates."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase407_event_horizon_protocol import (  # noqa: E402
    FAMILIES,
    HISTORY_MODES,
    INTERFACES,
    MODELS,
    OUT,
    STATE_IDS,
    SURFACE_REPLICAS,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def classify_mapping(
    states: tuple[str, ...], mapping: dict[str, str | None]
) -> str:
    if set(mapping) != set(states) or any(mapping[state] is None for state in states):
        return "incomplete_registered_mapping"
    outputs = [mapping[state] for state in states]
    if len(set(outputs)) != len(states):
        return "registered_state_collapse"
    if all(mapping[state] == state for state in states):
        return "registered_identity_mapping"
    return "registered_bijective_nonidentity_mapping"


def signature(states: tuple[str, ...], mapping: dict[str, str | None]) -> str:
    return "|".join(f"{state}->{mapping.get(state) or 'missing'}" for state in states)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        rows.extend(
            read_jsonl(
                OUT
                / "analysis/discovery/private"
                / model
                / "semantic_rows.jsonl"
            )
        )

    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model"],
                row["family_id"],
                row["anonymous_parallel_group_id"],
                row["surface_id_private"],
                row["interface_private"],
                row["history_mode_private"],
            )
        ].append(row)

    cells = []
    for key, selected in sorted(grouped.items()):
        model, family, group_id, surface, interface, history = key
        states = STATE_IDS[family]
        mapping = {
            row["state_id_private"]: row["normalized_semantic_state_private"]
            for row in selected
        }
        if len(selected) != len(states):
            raise RuntimeError(f"Phase407 response cell size mismatch: {key}")
        kind = classify_mapping(states, mapping)
        cells.append(
            {
                "schema_version": "81.5.0",
                "phase_id": "Phase407-ResponsePartitionDiagnostic",
                "model": model,
                "family_id": family,
                "anonymous_parallel_group_id": group_id,
                "surface_id": surface,
                "interface": interface,
                "history_mode": history,
                "state_count": len(states),
                "mapping_class": kind,
                "mapping_signature_private": signature(states, mapping),
                "registered_parse_count": sum(value is not None for value in mapping.values()),
                "registered_correct_count": sum(
                    mapping.get(state) == state for state in states
                ),
            }
        )

    expected_cells = sum(
        len(MODELS)
        * 12
        * len(SURFACE_REPLICAS)
        * len(INTERFACES[family])
        * len(HISTORY_MODES)
        for family in FAMILIES
    )
    if len(cells) != expected_cells:
        raise RuntimeError(f"Phase407 response cell count {len(cells)} != {expected_cells}")

    surface_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        surface_groups[
            (
                cell["model"],
                cell["family_id"],
                cell["anonymous_parallel_group_id"],
                cell["interface"],
                cell["history_mode"],
            )
        ].append(cell)
    surface_stability = []
    for key, selected in sorted(surface_groups.items()):
        signatures = {cell["mapping_signature_private"] for cell in selected}
        classes = {cell["mapping_class"] for cell in selected}
        stable = len(selected) == 4 and len(signatures) == 1
        surface_stability.append(
            {
                "model": key[0],
                "family_id": key[1],
                "anonymous_parallel_group_id": key[2],
                "interface": key[3],
                "history_mode": key[4],
                "surface_count": len(selected),
                "mapping_stable_across_all_surfaces": stable,
                "stable_mapping_class": next(iter(classes)) if stable else None,
                "stable_mapping_signature_private": (
                    next(iter(signatures)) if stable else None
                ),
            }
        )

    axes = []
    axis_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        axis_groups[
            (
                cell["model"],
                cell["family_id"],
                cell["interface"],
                cell["history_mode"],
            )
        ].append(cell)
    for key, selected in sorted(axis_groups.items()):
        counts = Counter(cell["mapping_class"] for cell in selected)
        axes.append(
            {
                "model": key[0],
                "family_id": key[1],
                "interface": key[2],
                "history_mode": key[3],
                "cell_count": len(selected),
                "mapping_class_counts": dict(sorted(counts.items())),
            }
        )

    all_counts = Counter(cell["mapping_class"] for cell in cells)
    stable_counts = Counter(
        row["stable_mapping_class"]
        for row in surface_stability
        if row["mapping_stable_across_all_surfaces"]
    )
    nonidentity_patterns = Counter(
        (
            cell["model"],
            cell["family_id"],
            cell["interface"],
            cell["history_mode"],
            cell["mapping_signature_private"],
        )
        for cell in cells
        if cell["mapping_class"] == "registered_bijective_nonidentity_mapping"
    )
    payload = {
        "schema_version": "81.5.0",
        "phase_id": "Phase407-ResponsePartitionDiagnostic",
        "created_at": now(),
        "case_count": len(rows),
        "condition_cell_count": len(cells),
        "mapping_class_counts": dict(sorted(all_counts.items())),
        "surface_mapping_group_count": len(surface_stability),
        "surface_stable_mapping_count": sum(
            row["mapping_stable_across_all_surfaces"]
            for row in surface_stability
        ),
        "surface_stable_mapping_class_counts": dict(sorted(stable_counts.items())),
        "registered_bijective_nonidentity_pattern_counts": [
            {
                "model": key[0],
                "family_id": key[1],
                "interface": key[2],
                "history_mode": key[3],
                "mapping_signature_private": key[4],
                "cell_count": count,
            }
            for key, count in sorted(
                nonidentity_patterns.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "axis_summary": axes,
        "cell_ledger_path": "analysis/private/phase407_response_mapping_cells.jsonl",
        "surface_stability_path": "analysis/private/phase407_surface_mapping_stability.jsonl",
        "formal_gate_changed": False,
        "authorization": {
            "promote_nonidentity_mapping_to_language_state": False,
            "run_calibration_from_partition_diagnostic": False,
            "run_physical_mapping": False,
        },
        "claim_boundary": {
            "registered_bijection_is_semantic_correctness": False,
            "stable_wrong_mapping_is_internal_operator": False,
            "external_parser_partition_is_internal_state_partition": False,
            "diagnostic_can_override_formal_gate": False,
        },
    }
    write_jsonl(
        OUT / "analysis/private/phase407_response_mapping_cells.jsonl", cells
    )
    write_jsonl(
        OUT / "analysis/private/phase407_surface_mapping_stability.jsonl",
        surface_stability,
    )
    write_json(OUT / "phase407_response_partition_diagnostic.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
