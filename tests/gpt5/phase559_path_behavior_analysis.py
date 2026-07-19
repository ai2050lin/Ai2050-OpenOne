#!/usr/bin/env python3
"""Apply the frozen Phase559 path gate and register complete Qwen3 anchors."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase558_fixed_identity_color_behavior_analysis import rate, split_report  # noqa: E402


OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PROTOCOL_PATH = OUT_DIR / "phase559_frozen_protocol.json"
PATH_CONTRACT_PATH = OUT_DIR / "phase559_path_behavior_frozen_contract.json"
ROWS_PATH = OUT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
EXECUTION_PATH = OUT_DIR / "phase559_qwen3_path_behavior_execution_summary.json"
SUMMARY_PATH = OUT_DIR / "phase559_path_behavior_summary.json"
ANCHOR_REGISTRY_PATH = OUT_DIR / "phase559_path_anchor_registry.json"
EXPECTED_ROWS = 7168


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    contract = read_json(PATH_CONTRACT_PATH)
    execution = read_json(EXECUTION_PATH)
    rows = read_jsonl(ROWS_PATH)
    if contract["authorized_models"] != ["qwen3"]:
        raise RuntimeError("Phase559 path authorization drift")
    if execution["status"] != "complete" or len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Phase559 path denominator is incomplete")
    if any(row["sealed"] for row in rows):
        raise RuntimeError("Phase559 path rows contain sealed data")

    path_gate = contract["path_gate"]
    reports: dict[str, dict[str, Any]] = {}
    anchors: list[dict[str, Any]] = []
    for split in contract["selected_splits"]:
        report = split_report(rows, split, protocol["behavior_gate"])
        gate_pass = bool(
            report["all_32_correct_world_rate"] >= path_gate["world_all_32_rate_min_per_split"]
            and report["minimum_cell_wilson_95_lcb"] >= path_gate["minimum_cell_wilson_95_lcb"]
            and report["unrecoverable_wilson_95_ucb"] <= path_gate["unrecoverable_wilson_95_ucb_max"]
        )
        report["path_gate_pass"] = gate_pass
        reports[split] = report
        for anchor_id in report["all_correct_world_ids"]:
            anchors.append({
                "model": "qwen3",
                "split": split,
                "anchor_id": anchor_id,
                "authorized_for_internal_collection": gate_pass and split in (
                    "path_discovery", "path_confirmation"
                ),
                "reserved_for_unseen_validation": split == "unseen_recombination",
            })

    all_splits_pass = all(reports[split]["path_gate_pass"] for split in contract["selected_splits"])
    registry = {
        "schema_version": "phase559_path_anchor_registry.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "authorized_models": ["qwen3"] if all_splits_pass else [],
        "anchors": anchors,
        "internal_collection_anchor_count": sum(
            row["authorized_for_internal_collection"] for row in anchors
        ) if all_splits_pass else 0,
        "unseen_validation_anchor_count": sum(
            row["reserved_for_unseen_validation"] for row in anchors
        ),
        "sealed_split_read": False,
    }
    write_json(ANCHOR_REGISTRY_PATH, registry)
    summary = {
        "schema_version": "phase559_path_behavior_summary.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "model": "qwen3",
        "row_count": len(rows),
        "semantic_accuracy": rate(rows, "semantic_correct"),
        "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
        "failure_count": sum(not row["semantic_correct"] for row in rows),
        "all_selected_splits_pass": all_splits_pass,
        "authorized_for_internal_collection": all_splits_pass,
        "split_reports": reports,
        "path_contract_sha256": sha256_file(PATH_CONTRACT_PATH),
        "rows_sha256": sha256_file(ROWS_PATH),
        "anchor_registry_path": str(ANCHOR_REGISTRY_PATH.relative_to(ROOT)),
        "cuda_used": execution["cuda_used"],
        "torch_dtype": execution["torch_dtype"],
        "sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "semantic_accuracy": summary["semantic_accuracy"],
        "strict_sequence_accuracy": summary["strict_sequence_accuracy"],
        "failure_count": summary["failure_count"],
        "all_selected_splits_pass": all_splits_pass,
        "internal_collection_anchor_count": registry["internal_collection_anchor_count"],
        "unseen_validation_anchor_count": registry["unseen_validation_anchor_count"],
        "splits": {
            split: {
                "semantic_accuracy": report["semantic_accuracy"],
                "all_32_correct_world_rate": report["all_32_correct_world_rate"],
                "minimum_cell_wilson_95_lcb": report["minimum_cell_wilson_95_lcb"],
                "unrecoverable_wilson_95_ucb": report["unrecoverable_wilson_95_ucb"],
                "path_gate_pass": report["path_gate_pass"],
            }
            for split, report in reports.items()
        },
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
