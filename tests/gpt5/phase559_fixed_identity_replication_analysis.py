#!/usr/bin/env python3
"""Analyze Phase559's independent larger-denominator behavior replication."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase558_fixed_identity_color_behavior_analysis import rate, split_report  # noqa: E402


OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PROTOCOL_PATH = OUT_DIR / "phase559_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase559_static_audit.json"
SUMMARY_PATH = OUT_DIR / "phase559_behavior_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("behavior_discovery", "behavior_confirmation")


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
    audit = read_json(AUDIT_PATH)
    if not audit["valid"]:
        raise RuntimeError("Phase559 static protocol failed")
    gate = protocol["behavior_gate"]
    reports = []
    authorized_models = []
    for model in MODELS:
        rows_path = OUT_DIR / f"phase559_{model}_behavior_rows.jsonl"
        execution = read_json(OUT_DIR / f"phase559_{model}_behavior_execution_summary.json")
        rows = read_jsonl(rows_path)
        if execution["status"] != "complete" or len(rows) != 8192:
            raise RuntimeError(f"Phase559 behavior incomplete for {model}")
        split_reports = {split: split_report(rows, split, gate) for split in SPLITS}
        authorized = all(split_reports[split]["behavior_gate_pass"] for split in SPLITS)
        if authorized:
            authorized_models.append(model)
        reports.append({
            "model": model,
            "row_count": len(rows),
            "semantic_accuracy": rate(rows, "semantic_correct"),
            "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
            "failure_count": sum(not row["semantic_correct"] for row in rows),
            "authorized_for_path_behavior": authorized,
            "split_reports": split_reports,
            "rows_sha256": sha256_file(rows_path),
            "cuda_used": execution["cuda_used"],
            "torch_dtype": execution["torch_dtype"],
            "sealed_split_read": execution["sealed_split_read"],
        })
    summary = {
        "schema_version": "phase559_behavior_summary.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "behavior_open_case_count": sum(row["row_count"] for row in reports),
        "authorized_models": authorized_models,
        "model_reports": reports,
        "phase558_thresholds_changed": False,
        "phase558_surfaces_changed": False,
        "sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "authorized_models": authorized_models,
        "model_reports": [{
            "model": row["model"],
            "semantic_accuracy": row["semantic_accuracy"],
            "strict_sequence_accuracy": row["strict_sequence_accuracy"],
            "failure_count": row["failure_count"],
            "authorized_for_path_behavior": row["authorized_for_path_behavior"],
            "discovery_world_rate": row["split_reports"]["behavior_discovery"]["all_32_correct_world_rate"],
            "confirmation_world_rate": row["split_reports"]["behavior_confirmation"]["all_32_correct_world_rate"],
            "discovery_min_cell_lcb": row["split_reports"]["behavior_discovery"]["minimum_cell_wilson_95_lcb"],
            "confirmation_min_cell_lcb": row["split_reports"]["behavior_confirmation"]["minimum_cell_wilson_95_lcb"],
            "discovery_unrecoverable_ucb": row["split_reports"]["behavior_discovery"]["unrecoverable_wilson_95_ucb"],
            "confirmation_unrecoverable_ucb": row["split_reports"]["behavior_confirmation"]["unrecoverable_wilson_95_ucb"],
        } for row in reports],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
