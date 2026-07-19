#!/usr/bin/env python3
"""Select Phase551 scaffolds on calibration worlds and freeze fresh validation cases."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from phase551_model_specific_route_protocol import (
    CALIBRATION_AUDIT_PATH,
    CALIBRATION_CASES_PATH,
    CALIBRATION_WORLDS,
    CELLS,
    FROZEN_SCAFFOLDS_PATH,
    MECHANISMS,
    MODELS,
    OUT_DIR,
    PHASE,
    PROTOCOL_PATH,
    ROOT,
    SCAFFOLDS,
    SPLITS,
    VALIDATION_AUDIT_PATH,
    VALIDATION_CASES_PATH,
    VALIDATION_PROTOCOL_PATH,
    VALIDATION_WORLDS,
    now,
    read_jsonl,
    row_from_spec,
    sha256_file,
    tokenizer_for,
    token_edit_distance,
    validate_rows,
    write_json,
    write_jsonl,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        anchors[row["anchor_id"]].append(row)
    all_correct = sum(
        len(group) == len(CELLS)
        and {row["factorial_cell"] for row in group} == set(CELLS)
        and all(row["semantic_correct"] for row in group)
        for group in anchors.values()
    )
    unrecoverable = sum(
        any(not row["semantic_event_recoverable"] for row in group)
        for group in anchors.values()
    )
    by_cell = {
        cell: sum(row["semantic_correct"] for row in rows if row["factorial_cell"] == cell)
        for cell in CELLS
    }
    return {
        "anchor_count": len(anchors),
        "all_four_correct_anchor_count": all_correct,
        "unrecoverable_anchor_count": unrecoverable,
        "correct_count_by_cell": by_cell,
        "minimum_cell_correct_count": min(by_cell.values()),
        "mean_prompt_token_count": mean(row["prompt_token_count"] for row in rows),
    }


def select_scaffolds() -> dict[str, Any]:
    static = read_json(CALIBRATION_AUDIT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if not static["valid"] or protocol["calibration_cases_sha256"] != sha256_file(CALIBRATION_CASES_PATH):
        raise RuntimeError("Phase551 calibration registry drift")
    rows = []
    execution = {}
    for model in MODELS:
        execution_path = OUT_DIR / f"phase551_calibration_{model}_behavior_execution.json"
        execution[model] = read_json(execution_path)
        behavior_path = OUT_DIR / f"phase551_calibration_{model}_behavior_rows.jsonl"
        rows.extend(read_jsonl(behavior_path))
    selections = []
    for model in MODELS:
        for mechanism in MECHANISMS:
            candidates = []
            for scaffold in SCAFFOLDS:
                selected = [
                    row for row in rows
                    if row["model"] == model
                    and row["mechanism_id"] == mechanism
                    and row["scaffold_id"] == scaffold
                ]
                metrics = candidate_metrics(selected)
                metrics.update({"scaffold_id": scaffold})
                metrics["selection_gate_pass"] = (
                    metrics["anchor_count"] == CALIBRATION_WORLDS
                    and metrics["all_four_correct_anchor_count"] >= 22
                    and metrics["minimum_cell_correct_count"] >= 23
                    and metrics["unrecoverable_anchor_count"] <= 1
                )
                candidates.append(metrics)
            ranked = sorted(
                candidates,
                key=lambda row: (
                    row["all_four_correct_anchor_count"],
                    row["minimum_cell_correct_count"],
                    -row["mean_prompt_token_count"],
                    -SCAFFOLDS.index(row["scaffold_id"]),
                ),
                reverse=True,
            )
            best = ranked[0]
            selections.append({
                "model": model,
                "mechanism_id": mechanism,
                "family_id": next(
                    row["family_id"] for row in rows
                    if row["model"] == model and row["mechanism_id"] == mechanism
                ),
                "selected_scaffold_id": best["scaffold_id"],
                "calibration_gate_pass": best["selection_gate_pass"],
                "validation_authorized": best["selection_gate_pass"],
                "selected_metrics": best,
                "all_candidate_metrics": candidates,
                "selection_used_validation_data": False,
                "compute_edge": False,
                "causal": False,
                "single_neuron": False,
                "sealed": False,
            })
    frozen = {
        "schema_version": "phase551_frozen_scaffolds.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "status": "model_specific_scaffolds_frozen_before_validation",
        "calibration_cases_sha256": sha256_file(CALIBRATION_CASES_PATH),
        "execution": execution,
        "selection_count": len(selections),
        "validation_authorized_count": sum(row["validation_authorized"] for row in selections),
        "selections": selections,
        "new_sealed_split_read": False,
    }
    write_json(FROZEN_SCAFFOLDS_PATH, frozen)
    return frozen


def register_validation(frozen: dict[str, Any]) -> dict[str, Any]:
    rows = []
    authorized = [row for row in frozen["selections"] if row["validation_authorized"]]
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        for selection in authorized:
            if selection["model"] != model:
                continue
            mechanism = selection["mechanism_id"]
            scaffold = selection["selected_scaffold_id"]
            for split in SPLITS:
                for world_index in range(VALIDATION_WORLDS):
                    anchor = []
                    for cell in CELLS:
                        anchor.append(row_from_spec(
                            tokenizer, model, mechanism, split, world_index,
                            cell, scaffold, "validation",
                        ))
                    reference_ids = anchor[0][1]
                    for row, ids in anchor:
                        row["token_edit_distance_from_route0_answer_a"] = token_edit_distance(reference_ids, ids)
                        row["token_length_delta_from_route0_answer_a"] = len(ids) - len(reference_ids)
                        rows.append(row)
    expected = len(authorized) * len(SPLITS) * VALIDATION_WORLDS * len(CELLS)
    audit = validate_rows(rows, expected, len(CELLS))
    calibration_entities = {row["entity_key"] for row in read_jsonl(CALIBRATION_CASES_PATH)}
    validation_entities = {row["entity_key"] for row in rows}
    previous_entities = set()
    for previous in (
        ROOT / "tests/gpt5/result/phase548_shared_attention_compute/phase548_registered_cases.jsonl",
        ROOT / "tests/gpt5/result/phase549_route_answer_factorial/phase549_registered_cases.jsonl",
    ):
        if previous.exists():
            previous_entities.update(row["entity_key"] for row in read_jsonl(previous))
    audit.update({
        "schema_version": "phase551_validation_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "authorized_contract_count": len(authorized),
        "calibration_entity_overlap_count": len(calibration_entities & validation_entities),
        "phase548_549_entity_overlap_count": len(previous_entities & validation_entities),
        "frozen_scaffolds_sha256": sha256_file(FROZEN_SCAFFOLDS_PATH),
    })
    audit["valid"] = bool(
        audit["valid"]
        and audit["calibration_entity_overlap_count"] == 0
        and audit["phase548_549_entity_overlap_count"] == 0
    )
    audit["status"] = "static_pass_no_validation_run" if audit["valid"] else "static_fail"
    write_jsonl(VALIDATION_CASES_PATH, rows)
    write_json(VALIDATION_AUDIT_PATH, audit)
    validation_protocol = {
        "schema_version": "phase551_validation_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "frozen_scaffolds_path": str(FROZEN_SCAFFOLDS_PATH.relative_to(ROOT)),
        "frozen_scaffolds_sha256": sha256_file(FROZEN_SCAFFOLDS_PATH),
        "validation_cases_path": str(VALIDATION_CASES_PATH.relative_to(ROOT)),
        "validation_cases_sha256": sha256_file(VALIDATION_CASES_PATH),
        "validation_audit_path": str(VALIDATION_AUDIT_PATH.relative_to(ROOT)),
        "validation_audit_sha256": sha256_file(VALIDATION_AUDIT_PATH),
        "models_in_required_execution_order": list(MODELS),
        "authorized_contract_count": len(authorized),
        "validation_worlds_per_split": VALIDATION_WORLDS,
        "splits": list(SPLITS),
        "factorial_cells": list(CELLS),
        "evidence_boundaries": {
            "selection_used_validation_data": False,
            "validation_behavior_is_physical_evidence": False,
            "new_sealed_split_read": False,
            "intervention_authorized": False,
        },
    }
    write_json(VALIDATION_PROTOCOL_PATH, validation_protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> None:
    frozen = select_scaffolds()
    audit = register_validation(frozen)
    print(json.dumps({"frozen": frozen, "validation_audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
