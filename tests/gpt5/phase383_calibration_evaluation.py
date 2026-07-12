#!/usr/bin/env python3
"""Evaluate frozen Phase383 signed event cells on independent calibration groups."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
MODELS = ("qwen3", "glm4", "deepseek7b")

from phase383_signed_event_map import (  # noqa: E402
    crossmodel_cells,
    model_cells,
    process_model,
    write_json,
    write_jsonl,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["mechanism_id"],
        row["contrast_axis"],
        row["event_type"],
        row["receiver_role"],
        row["source_role"],
        row["depth_bin"],
    )


def main() -> None:
    freeze = read_json(OUT / "phase383_discovery_map_freeze.json")
    if not freeze["authorization"]["calibration_collection"]:
        raise RuntimeError("Phase383 calibration is not authorized")
    contract = copy.deepcopy(read_json(OUT / "phase383_signed_event_contract.json"))
    contract["candidate_gates"]["minimum_group_count"] = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    model_summaries = []
    private_root = OUT / "calibration/private"
    for model in MODELS:
        rows, summary = process_model(model, "calibration", device)
        path = private_root / f"phase383_{model}_calibration_event_rows.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        summary["parquet_relative_path"] = str(path.relative_to(OUT))
        summary["parquet_byte_count"] = path.stat().st_size
        model_summaries.append(summary)
        all_rows.extend(rows)
    cells = model_cells(all_rows, contract)
    cross_rows = crossmodel_cells(cells)
    write_jsonl(OUT / "phase383_calibration_model_cells.jsonl", cells)
    write_jsonl(OUT / "phase383_calibration_crossmodel_cells.jsonl", cross_rows)
    calibration_by_signature = {signature(row): row for row in cross_rows}
    replication_rows = []
    for frozen in freeze["frozen_candidates"]:
        calibrated = calibration_by_signature.get(signature(frozen))
        replication_rows.append(
            {
                "schema_version": "57.5.0",
                "phase_id": "Phase383-CalibrationEvaluation",
                "mechanism_id": frozen["mechanism_id"],
                "contrast_axis": frozen["contrast_axis"],
                "event_type": frozen["event_type"],
                "receiver_role": frozen["receiver_role"],
                "source_role": frozen["source_role"],
                "depth_bin": frozen["depth_bin"],
                "discovery_level2_pass": frozen["heterogeneous_level2_pass"],
                "discovery_level3_pass": frozen["level3_pass"],
                "calibration_cell_available": calibrated is not None,
                "calibration_passing_models": (
                    calibrated["passing_models"] if calibrated else []
                ),
                "calibration_level2_pass": bool(
                    calibrated and calibrated["heterogeneous_level2_pass"]
                ),
                "calibration_level3_pass": bool(
                    calibrated and calibrated["level3_pass"]
                ),
                "terminal_interface_cell": frozen["terminal_interface_cell"],
                "upstream_cell": frozen["upstream_cell"],
                "causal_path_established": False,
            }
        )
    write_jsonl(OUT / "phase383_calibration_replication_rows.jsonl", replication_rows)
    level2 = [row for row in replication_rows if row["calibration_level2_pass"]]
    level3 = [row for row in replication_rows if row["calibration_level3_pass"]]
    upstream = [row for row in level2 if row["upstream_cell"]]
    terminal = [row for row in level2 if row["terminal_interface_cell"]]
    summary = {
        "schema_version": "57.5.0",
        "phase_id": "Phase383-CalibrationEvaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": len(MODELS),
            "mechanism_count": 4,
            "calibration_parallel_group_count": 8,
            "case_count": sum(row["case_count"] for row in model_summaries),
            "event_row_count": len(all_rows),
            "frozen_candidate_count": len(replication_rows),
        },
        "models": model_summaries,
        "frozen_contract": {
            "descriptor_thresholds_changed": False,
            "calibration_group_count": 2,
            "discovery_candidates_added_after_opening": False,
        },
        "results": {
            "calibration_level2_replication_count": len(level2),
            "calibration_level3_replication_count": len(level3),
            "upstream_level2_replication_count": len(upstream),
            "terminal_interface_level2_replication_count": len(terminal),
            "replication_counts_by_event_type": dict(
                Counter(row["event_type"] for row in level2)
            ),
            "replication_counts_by_depth_bin": dict(
                Counter(str(row["depth_bin"]) for row in level2)
            ),
            "attention_source_write_replication_count": sum(
                row["event_type"] == "attention_source_write" for row in level2
            ),
            "language_path_discovered": False,
            "terminal_prediction_gain_computed": False,
        },
        "claim_boundary": {
            "calibration_replication_is_causal_path": False,
            "late_current_state_is_upstream_formation_rule": False,
            "physical_holdout_is_unopened": True,
            "exact_head_and_channel_coordinates_are_scanned": False,
        },
        "authorization": {
            "physical_holdout_collection": len(upstream) > 0,
            "causal_intervention": False,
            "exact_subunit_coordinate_expansion": len(upstream) == 0,
        },
        "next_decision": (
            "open_upstream_physical_holdout"
            if upstream
            else "keep_physical_holdout_sealed_and_expand_exact_head_channel_coordinates"
        ),
    }
    write_json(OUT / "phase383_calibration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
