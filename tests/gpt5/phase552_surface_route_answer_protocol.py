#!/usr/bin/env python3
"""Freeze a semantic-route by surface-form by answer-identity Phase552 design."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from phase548_shared_attention_compute_protocol import render_chat, token_edit_distance
from phase551_model_specific_behavior_analysis import QUALIFICATION_PATH
from phase551_model_specific_route_protocol import (
    FROZEN_SCAFFOLDS_PATH,
    MODELS,
    OUT_DIR as PHASE551_OUT,
    ROOT,
    SPLITS,
    case_spec,
    read_jsonl,
    tokenizer_for,
)


PHASE = "Phase552"
SCHEMA_VERSION = "phase552_surface_route_answer_factorial.v1"
OUT_DIR = ROOT / "tests/gpt5/result/phase552_surface_route_answer"
CASES_PATH = OUT_DIR / "phase552_registered_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase552_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase552_static_audit.json"
WORLDS_PER_SPLIT = 73
CELLS = tuple(
    f"route{route}_surface{surface}_answer_{answer}"
    for route in (0, 1) for surface in (0, 1) for answer in ("a", "b")
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def factors(cell: str) -> tuple[int, int, str]:
    route = 0 if cell.startswith("route0") else 1
    surface = 0 if "_surface0_" in cell else 1
    answer = "a" if cell.endswith("answer_a") else "b"
    return route, surface, answer


def surface_contracts() -> list[dict[str, Any]]:
    frozen = read_json(FROZEN_SCAFFOLDS_PATH)
    qualification = {
        (row["model"], row["mechanism_id"]): row for row in read_jsonl(QUALIFICATION_PATH)
    }
    contracts = []
    for selection in frozen["selections"]:
        key = (selection["model"], selection["mechanism_id"])
        if not qualification[key]["observer_collection_authorized"]:
            continue
        primary = selection["selected_scaffold_id"]
        alternatives = [
            row for row in selection["all_candidate_metrics"]
            if row["scaffold_id"] != primary and row["selection_gate_pass"]
        ]
        if not alternatives:
            continue
        alternate = sorted(
            alternatives,
            key=lambda row: (
                row["all_four_correct_anchor_count"],
                row["minimum_cell_correct_count"],
                -row["mean_prompt_token_count"],
            ),
            reverse=True,
        )[0]
        contracts.append({
            "model": selection["model"],
            "family_id": selection["family_id"],
            "mechanism_id": selection["mechanism_id"],
            "surface0_scaffold_id": primary,
            "surface1_scaffold_id": alternate["scaffold_id"],
            "surface_selection_used_phase552_data": False,
            "phase551_validation_behavior_pass": True,
        })
    return contracts


def row_from_cell(
    tokenizer: Any,
    contract: dict[str, Any],
    split: str,
    world_index: int,
    cell: str,
) -> tuple[dict[str, Any], list[int]]:
    route, surface, answer = factors(cell)
    scaffold = contract[f"surface{surface}_scaffold_id"]
    base_cell = f"route{route}_answer_{answer}"
    offset = 8000 if split == "discovery" else 12000
    spec = case_spec(
        contract["mechanism_id"], split, world_index + offset, base_cell, scaffold,
    )
    prompt = render_chat(tokenizer, contract["model"], spec["raw_prompt"])
    ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
    anchor_id = (
        f"phase552_{contract['model']}_{contract['mechanism_id']}_{split}_{world_index:03d}"
    )
    return ({
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": f"{anchor_id}_{cell}",
        "anchor_id": anchor_id,
        "model": contract["model"],
        "family_id": contract["family_id"],
        "mechanism_id": contract["mechanism_id"],
        "split": split,
        "world_index": world_index,
        "factorial_cell": cell,
        "route_factor": route,
        "surface_factor": surface,
        "answer_factor": answer,
        "scaffold_id": scaffold,
        "raw_prompt": spec["raw_prompt"],
        "prompt": prompt,
        "source_fragment": spec["source_fragment"],
        "query_fragment": spec["query_fragment"],
        "target": spec["target"],
        "target_aliases": [spec["target"]],
        "distractors": spec["distractors"],
        "all_candidates": spec["all_candidates"],
        "strict_expected": spec["target"],
        "strict_kind": "plain",
        "entity_key": spec["entity_key"],
        "prompt_token_count": len(ids),
        "semantic_event_is_natural_answer": True,
        "arbitrary_label_output": False,
        "observer_only": True,
        "compute_edge": False,
        "causal": False,
        "single_neuron": False,
        "sealed": False,
    }, ids)


def build_rows(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        for contract in contracts:
            if contract["model"] != model:
                continue
            for split in SPLITS:
                for world_index in range(WORLDS_PER_SPLIT):
                    anchor = [
                        row_from_cell(tokenizer, contract, split, world_index, cell)
                        for cell in CELLS
                    ]
                    reference_ids = anchor[0][1]
                    for row, ids in anchor:
                        row["token_edit_distance_from_reference"] = token_edit_distance(reference_ids, ids)
                        row["token_length_delta_from_reference"] = len(ids) - len(reference_ids)
                        rows.append(row)
    return rows


def validate(rows: list[dict[str, Any]], contracts: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(contracts) * len(SPLITS) * WORLDS_PER_SPLIT * len(CELLS)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["anchor_id"]].append(row)
    relation_errors = 0
    for group in groups.values():
        by_cell = {row["factorial_cell"]: row for row in group}
        if set(by_cell) != set(CELLS):
            relation_errors += 1
            continue
        for answer in ("a", "b"):
            targets = {
                by_cell[f"route{route}_surface{surface}_answer_{answer}"]["target"]
                for route in (0, 1) for surface in (0, 1)
            }
            if len(targets) != 1:
                relation_errors += 1
        if by_cell["route0_surface0_answer_a"]["target"] == by_cell["route0_surface0_answer_b"]["target"]:
            relation_errors += 1
    phase551_entities = {
        row["entity_key"]
        for row in read_jsonl(PHASE551_OUT / "phase551_validation_cases.jsonl")
    }
    entities = {row["entity_key"] for row in rows}
    audit = {
        "schema_version": "phase552_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "expected_case_count": expected,
        "contract_count": len(contracts),
        "model_case_counts": dict(Counter(row["model"] for row in rows)),
        "anchor_count": len(groups),
        "rows_per_anchor": sorted({len(group) for group in groups.values()}),
        "factorial_relation_error_count": relation_errors,
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_prompt_count": len(rows) - len({(row["model"], row["prompt"]) for row in rows}),
        "phase551_entity_overlap_count": len(phase551_entities & entities),
        "prompt_token_count_range_by_model": {
            model: [
                min(row["prompt_token_count"] for row in rows if row["model"] == model),
                max(row["prompt_token_count"] for row in rows if row["model"] == model),
            ] for model in MODELS if any(row["model"] == model for row in rows)
        },
        "sealed_row_count": sum(bool(row["sealed"]) for row in rows),
    }
    audit["valid"] = bool(
        len(rows) == expected
        and audit["rows_per_anchor"] == [len(CELLS)]
        and max(maximum for _minimum, maximum in audit["prompt_token_count_range_by_model"].values()) <= 512
        and all(audit[key] == 0 for key in (
            "factorial_relation_error_count", "duplicate_case_id_count", "duplicate_prompt_count",
            "phase551_entity_overlap_count", "sealed_row_count",
        ))
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    contracts = surface_contracts()
    rows = build_rows(contracts)
    write_jsonl(CASES_PATH, rows)
    audit = validate(rows, contracts)
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Independent semantic route, surface form, and answer identity factorial",
        "models_in_required_execution_order": list(MODELS),
        "factorial_cells": list(CELLS),
        "worlds_per_split": WORLDS_PER_SPLIT,
        "splits": list(SPLITS),
        "surface_contracts": contracts,
        "behavior_gate": {
            "all_eight_cells_correct_lcb95_min": 0.90,
            "unrecoverable_anchor_ucb95_max": 0.05,
            "both_independent_splits_required": True,
        },
        "observer_gate": {
            "route_effect_compared_with_surface_and_answer_effects": True,
            "full_layer_components_and_roles": True,
            "intervention_authorized": False,
        },
        "evidence_boundaries": {
            "surface_scaffolds_selected_only_from_phase551_calibration": True,
            "phase552_data_used_for_surface_selection": False,
            "observer_is_compute_edge": False,
            "new_sealed_split_read": False,
            "head_channel_neuron_search": False,
        },
        "registered_cases_path": str(CASES_PATH.relative_to(ROOT)),
        "registered_cases_sha256": sha256_file(CASES_PATH),
        "static_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
