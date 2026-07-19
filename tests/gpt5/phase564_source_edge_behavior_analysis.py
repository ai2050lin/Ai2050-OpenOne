#!/usr/bin/env python3
"""Analyze Phase564 behavior gates and freeze the edge denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase558_fixed_identity_color_behavior_analysis import rate, split_report  # noqa: E402


OUT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
PROTOCOL_PATH = OUT_DIR / "phase564_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase564_static_audit.json"
BEHAVIOR_SUMMARY_PATH = OUT_DIR / "phase564_behavior_summary.json"
EDGE_CONTRACT_PATH = OUT_DIR / "phase564_edge_behavior_frozen_contract.json"
EDGE_SUMMARY_PATH = OUT_DIR / "phase564_edge_behavior_summary.json"
ANCHOR_REGISTRY_PATH = OUT_DIR / "phase564_edge_anchor_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(mode: str, model: str) -> tuple[Path, Path]:
    prefix = f"phase564_{model}_{mode}_behavior"
    return (
        OUT_DIR / f"{prefix}_rows.jsonl",
        OUT_DIR / f"{prefix}_execution_summary.json",
    )


def analyze_behavior() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit["valid"]:
        raise RuntimeError("Phase564 static protocol failed")
    gate = protocol["behavior_gate"]
    splits = tuple(protocol["behavior_splits"])
    expected = int(protocol["behavior_case_count_per_model"])
    reports = []
    authorized_models = []
    for model in MODELS:
        rows_path, execution_path = paths("behavior", model)
        execution = read_json(execution_path)
        rows = read_jsonl(rows_path)
        if execution["status"] != "complete" or len(rows) != expected:
            raise RuntimeError(f"Phase564 behavior incomplete for {model}")
        split_reports = {split: split_report(rows, split, gate) for split in splits}
        authorized = all(split_reports[split]["behavior_gate_pass"] for split in splits)
        if authorized:
            authorized_models.append(model)
        reports.append({
            "model": model,
            "row_count": len(rows),
            "semantic_accuracy": rate(rows, "semantic_correct"),
            "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
            "failure_count": sum(not row["semantic_correct"] for row in rows),
            "authorized_for_edge_behavior": authorized,
            "split_reports": split_reports,
            "rows_sha256": sha256_file(rows_path),
            "cuda_used": execution["cuda_used"],
            "torch_dtype": execution["torch_dtype"],
            "sealed_split_read": execution["sealed_split_read"],
        })
    summary = {
        "schema_version": "phase564_behavior_summary.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "behavior_open_case_count": sum(report["row_count"] for report in reports),
        "authorized_models": authorized_models,
        "model_reports": reports,
        "parent_contract_changed": False,
        "sealed_split_read": False,
    }
    write_json(BEHAVIOR_SUMMARY_PATH, summary)
    selected_splits = tuple(protocol["edge_splits"])
    contract = {
        "schema_version": "phase564_edge_behavior_frozen_contract.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "parent_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_behavior_summary_sha256": sha256_file(BEHAVIOR_SUMMARY_PATH),
        "authorized_models": authorized_models,
        "selected_splits": list(selected_splits),
        "row_counts": {
            split: int(protocol["split_world_counts"][split]) * 32 for split in selected_splits
        },
        "expected_rows_per_model": sum(
            int(protocol["split_world_counts"][split]) * 32 for split in selected_splits
        ),
        "path_gate": {
            "world_all_32_rate_min_per_split": 0.80,
            "minimum_cell_wilson_95_lcb": 0.90,
            "unrecoverable_wilson_95_ucb_max": 0.05,
            "all_splits_required": True,
            "internal_anchor_requires_all_32_correct": True,
        },
        "sealed_split_read": False,
    }
    write_json(EDGE_CONTRACT_PATH, contract)
    print(json.dumps({
        "authorized_models": authorized_models,
        "models": [{
            "model": report["model"],
            "semantic_accuracy": report["semantic_accuracy"],
            "failure_count": report["failure_count"],
            "authorized": report["authorized_for_edge_behavior"],
            "discovery_world_rate": report["split_reports"][splits[0]]["all_32_correct_world_rate"],
            "confirmation_world_rate": report["split_reports"][splits[1]]["all_32_correct_world_rate"],
            "discovery_min_lcb": report["split_reports"][splits[0]]["minimum_cell_wilson_95_lcb"],
            "confirmation_min_lcb": report["split_reports"][splits[1]]["minimum_cell_wilson_95_lcb"],
        } for report in reports],
    }, ensure_ascii=False, indent=2))
    return summary


def analyze_edge_behavior() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    contract = read_json(EDGE_CONTRACT_PATH)
    gate = contract["path_gate"]
    model_reports = []
    anchors = []
    internally_authorized = []
    for model in contract["authorized_models"]:
        rows_path, execution_path = paths("edge", model)
        rows = read_jsonl(rows_path)
        execution = read_json(execution_path)
        if execution["status"] != "complete" or len(rows) != contract["expected_rows_per_model"]:
            raise RuntimeError(f"Phase564 edge behavior incomplete for {model}")
        reports: dict[str, dict[str, Any]] = {}
        for split in contract["selected_splits"]:
            report = split_report(rows, split, protocol["behavior_gate"])
            report["path_gate_pass"] = bool(
                report["all_32_correct_world_rate"] >= gate["world_all_32_rate_min_per_split"]
                and report["minimum_cell_wilson_95_lcb"] >= gate["minimum_cell_wilson_95_lcb"]
                and report["unrecoverable_wilson_95_ucb"] <= gate["unrecoverable_wilson_95_ucb_max"]
            )
            reports[split] = report
            for anchor_id in report["all_correct_world_ids"]:
                anchors.append({
                    "model": model,
                    "split": split,
                    "anchor_id": anchor_id,
                    "authorized_for_internal_collection": bool(report["path_gate_pass"]),
                    "sealed": False,
                })
        all_pass = all(reports[split]["path_gate_pass"] for split in contract["selected_splits"])
        if all_pass:
            internally_authorized.append(model)
        model_reports.append({
            "model": model,
            "row_count": len(rows),
            "semantic_accuracy": rate(rows, "semantic_correct"),
            "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
            "failure_count": sum(not row["semantic_correct"] for row in rows),
            "all_splits_pass": all_pass,
            "authorized_for_internal_collection": all_pass,
            "split_reports": reports,
            "rows_sha256": sha256_file(rows_path),
            "cuda_used": execution["cuda_used"],
            "torch_dtype": execution["torch_dtype"],
        })
    registry = {
        "schema_version": "phase564_edge_anchor_registry.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "authorized_models": internally_authorized,
        "anchors": anchors,
        "anchor_counts": {
            f"{model}:{split}": sum(
                row["model"] == model and row["split"] == split
                and row["authorized_for_internal_collection"]
                for row in anchors
            )
            for model in internally_authorized
            for split in contract["selected_splits"]
        },
        "sealed_split_read": False,
    }
    write_json(ANCHOR_REGISTRY_PATH, registry)
    summary = {
        "schema_version": "phase564_edge_behavior_summary.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "authorized_models": internally_authorized,
        "model_reports": model_reports,
        "edge_contract_sha256": sha256_file(EDGE_CONTRACT_PATH),
        "anchor_registry_sha256": sha256_file(ANCHOR_REGISTRY_PATH),
        "sealed_split_read": False,
    }
    write_json(EDGE_SUMMARY_PATH, summary)
    print(json.dumps({
        "authorized_models": internally_authorized,
        "anchor_counts": registry["anchor_counts"],
        "models": [{
            "model": report["model"],
            "semantic_accuracy": report["semantic_accuracy"],
            "failure_count": report["failure_count"],
            "all_splits_pass": report["all_splits_pass"],
        } for report in model_reports],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("behavior", "edge"))
    args = parser.parse_args()
    if args.mode == "behavior":
        analyze_behavior()
    else:
        analyze_edge_behavior()
