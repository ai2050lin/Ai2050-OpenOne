#!/usr/bin/env python3
"""Aggregate repaired behavior gates and freeze the complete 18-mechanism matrix."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase361_repaired_contract_case_bank import (  # noqa: E402
    MODELS, OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


SPLITS = ("physical_discovery", "physical_calibration", "physical_heldout", "causal_sealed")
CONDITIONS = ("A", "B", "C", "D")
PHASE360 = ROOT / "tests/gpt5/result/phase360_denominator_freeze/phase360_denominator_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    contracts = read_jsonl(root / "phase361_repaired_contract_registry.jsonl")
    phrase = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase361_phrase_rows.jsonl")]
    rollout = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase361_rollout_rows.jsonl")]
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    cells = []
    for model in MODELS:
        for contract in contracts:
            family, mechanism = contract["family_id"], contract["mechanism_id"]
            cell: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "family_id": family, "mechanism_id": mechanism,
                "manipulated_variable": contract["manipulated_variable"],
            }
            split_gates = []
            for split in SPLITS:
                condition_gates = []
                for condition in CONDITIONS:
                    p_rows = [
                        row for row in phrase
                        if row["model"] == model and row["family_id"] == family
                        and row["mechanism_id"] == mechanism and row["split"] == split
                        and row["contrast_condition"].startswith(condition)
                    ]
                    r_rows = [
                        row for row in rollout
                        if row["model"] == model and row["family_id"] == family
                        and row["mechanism_id"] == mechanism and row["split"] == split
                        and row["contrast_condition"].startswith(condition)
                    ]
                    valid = sum(row["score_valid"] for row in p_rows) / len(p_rows)
                    semantic = sum(row["semantic_correct"] for row in r_rows) / len(r_rows)
                    gate = bool(valid == 1 and semantic >= 0.8)
                    condition_gates.append(gate)
                    prefix = f"{split}__{condition}"
                    cell.update({
                        f"{prefix}_case_count": len(r_rows),
                        f"{prefix}_phrase_valid_rate": round(valid, 7),
                        f"{prefix}_semantic_accuracy": round(semantic, 7),
                        f"{prefix}_gate_pass": gate,
                    })
                split_pass = all(condition_gates)
                cell[f"{split}_gate_pass"] = split_pass
                split_gates.append(split_pass)
            cell["full_behavior_contract_gate_pass"] = all(split_gates)
            cell["trace_discovery_calibration_gate_pass"] = (
                cell["physical_discovery_gate_pass"] and cell["physical_calibration_gate_pass"]
            )
            cell["physical_heldout_trace_revealed"] = False
            cell["causal_sealed_trace_revealed"] = False
            cells.append(cell)

    cross = []
    for contract in contracts:
        values = [
            row for row in cells
            if row["family_id"] == contract["family_id"] and row["mechanism_id"] == contract["mechanism_id"]
        ]
        cross.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "family_id": contract["family_id"], "mechanism_id": contract["mechanism_id"],
            "full_behavior_model_count": sum(row["full_behavior_contract_gate_pass"] for row in values),
            "trace_entry_model_count": sum(row["trace_discovery_calibration_gate_pass"] for row in values),
            "cross_model_full_behavior_gate_pass": all(row["full_behavior_contract_gate_pass"] for row in values),
            "cross_model_trace_entry": all(row["trace_discovery_calibration_gate_pass"] for row in values),
        })

    previous = read_json(PHASE360)
    repaired = {(row["family_id"], row["mechanism_id"]): row for row in cross}
    frozen_matrix = []
    for old in previous["mechanisms"]:
        key = (old["family_id"], old["mechanism_id"])
        if key not in repaired:
            frozen_matrix.append({**old, "phase361_repaired": False, "admission_state_frozen": True})
            continue
        result = repaired[key]
        trace = result["cross_model_trace_entry"]
        full = result["cross_model_full_behavior_gate_pass"]
        frozen_matrix.append({
            **old,
            "contract_gate_pass": True,
            "contract_mapping_status": "repaired_contract_qualified",
            "behavior_measured_on_three_models": True,
            "cross_model_trace_entry": trace,
            "cross_model_full_behavior_gate_pass": full,
            "blind_discovery_admission": "blind_discovery_admitted" if trace else "behavior_rejected",
            "phase361_repaired": True,
            "admission_state_frozen": True,
        })
    admitted = [row for row in frozen_matrix if row["blind_discovery_admission"] == "blind_discovery_admitted"]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "repaired_contract_count": 7,
            "registered_case_count": len(phrase),
            "model_count": 3,
            "model_contract_cell_count": len(cells),
            "mechanism_matrix_count": len(frozen_matrix),
        },
        "quality": {
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "invalid_phrase_row_count": sum(not row["score_valid"] for row in phrase),
            "actual_model_batch_size": 1,
        },
        "results": {
            "repaired_cross_model_trace_entry_count": sum(row["cross_model_trace_entry"] for row in cross),
            "repaired_cross_model_full_behavior_count": sum(row["cross_model_full_behavior_gate_pass"] for row in cross),
            "total_blind_discovery_admitted_count": len(admitted),
            "total_full_behavior_pass_count": sum(row["cross_model_full_behavior_gate_pass"] for row in frozen_matrix),
        },
        "admitted_mechanisms": [
            {"family_id": row["family_id"], "mechanism_id": row["mechanism_id"]} for row in admitted
        ],
        "frozen_mechanism_matrix": frozen_matrix,
        "evidence_boundary": {
            "admission_matrix_frozen": True,
            "behavior_qualification_is_internal_trace": False,
            "physical_heldout_trace_revealed": False,
            "causal_sealed_trace_revealed": False,
            "causal_intervention_executed": False,
            "single_global_progress_percentage_valid": False,
            "language_encoding_closed": False,
        },
        "next_decision": "run_r0_r1_on_admitted_mechanisms" if admitted else "stop_no_admitted_mechanisms",
    }
    write_jsonl(root / "phase361_model_behavior_cells.jsonl", cells)
    write_jsonl(root / "phase361_repaired_cross_model_summary.jsonl", cross)
    write_jsonl(root / "phase361_frozen_mechanism_matrix.jsonl", frozen_matrix)
    write_json(root / "phase361_behavior_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))
