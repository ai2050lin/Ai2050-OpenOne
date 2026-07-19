#!/usr/bin/env python3
"""Analyze the frozen Phase568 explicit-relation behavior denominator."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase568_explicit_relation_binding"
PROTOCOL_PATH = OUT_DIR / "phase568_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase568_static_audit.json"
SUMMARY_PATH = OUT_DIR / "phase568_behavior_summary.json"
TRIPLET_REGISTRY_PATH = OUT_DIR / "phase568_behavior_qualified_triplet_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
GATE_SPLITS = ("gate_discovery", "gate_confirmation")
ROLE_SPLITS = ("role_discovery", "role_confirmation")
EXPECTED_MODEL_ROWS = 22464
EXPECTED_ROWS_PER_WORLD = 108
Z = 1.96


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denominator = 1.0 + Z * Z / n
    center = (p + Z * Z / (2.0 * n)) / denominator
    radius = Z * math.sqrt((p * (1.0 - p) + Z * Z / (4.0 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def grouped_metric(
    rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    report = {}
    for key, group in sorted(groups.items()):
        correct = sum(bool(row["semantic_correct"]) for row in group)
        lcb, ucb = wilson(correct, len(group))
        report[key] = {
            "n": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "wilson_95_lcb": lcb,
            "wilson_95_ucb": ucb,
        }
    return report


def split_report(rows: list[dict[str, Any]], split: str, gate: dict[str, Any]) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    triplets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        worlds[row["anchor_id"]].append(row)
        triplets[row["triplet_id"]].append(row)
    invalid_worlds = sum(len(group) != EXPECTED_ROWS_PER_WORLD for group in worlds.values())
    invalid_triplets = sum(len(group) != 3 for group in triplets.values())
    if invalid_worlds or invalid_triplets:
        raise RuntimeError(
            f"Phase568 denominator drift in {split}: worlds={invalid_worlds}, "
            f"triplets={invalid_triplets}"
        )
    all_correct_worlds = sorted(
        anchor for anchor, group in worlds.items() if all(row["semantic_correct"] for row in group)
    )
    all_correct_triplets = sorted(
        triplet for triplet, group in triplets.items()
        if all(row["semantic_correct"] for row in group)
    )
    cell_metrics = grouped_metric(selected, lambda row: row["factorial_cell"])
    axis_metrics = {
        "binding": grouped_metric(selected, lambda row: str(row["binding"])),
        "query_object": grouped_metric(selected, lambda row: str(row["query_object_index"])),
        "query_relation": grouped_metric(selected, lambda row: row["query_relation"]),
        "surface": grouped_metric(selected, lambda row: str(row["surface_id"])),
        "fact_order": grouped_metric(selected, lambda row: str(row["fact_order"])),
        "value_regime": grouped_metric(selected, lambda row: row["value_regime"]),
    }
    min_cell_accuracy = min(
        (float(metric["accuracy"]) for metric in cell_metrics.values()), default=0.0
    )
    min_cell_lcb = min(
        (float(metric["wilson_95_lcb"]) for metric in cell_metrics.values()), default=0.0
    )
    min_axis_lcb = min(
        (
            float(metric["wilson_95_lcb"])
            for axis in axis_metrics.values()
            for metric in axis.values()
        ),
        default=0.0,
    )
    semantic_accuracy = rate(selected, "semantic_correct")
    triplet_rate = len(all_correct_triplets) / len(triplets) if triplets else 0.0
    unrecoverable = sum(not row["semantic_event_recoverable"] for row in selected)
    unrecoverable_lcb, unrecoverable_ucb = wilson(unrecoverable, len(selected))
    gate_applies = split in GATE_SPLITS
    gate_checks = {
        "semantic_accuracy": semantic_accuracy >= gate["semantic_accuracy_min_per_gate_split"],
        "triplet_all_three_rate": (
            triplet_rate >= gate["all_three_bindings_correct_triplet_rate_min"]
        ),
        "minimum_axis_wilson_95_lcb": (
            min_axis_lcb >= gate["minimum_axis_wilson_95_lcb"]
        ),
        "minimum_factorial_cell_accuracy": (
            min_cell_accuracy >= gate["minimum_factorial_cell_accuracy"]
        ),
        "unrecoverable_wilson_95_ucb": (
            unrecoverable_ucb <= gate["unrecoverable_wilson_95_ucb_max"]
        ),
    }
    return {
        "split": split,
        "row_count": len(selected),
        "semantic_accuracy": semantic_accuracy,
        "strict_sequence_accuracy": rate(selected, "strict_sequence_correct"),
        "unrecoverable_count": unrecoverable,
        "unrecoverable_rate": unrecoverable / len(selected) if selected else 0.0,
        "unrecoverable_wilson_95_lcb": unrecoverable_lcb,
        "unrecoverable_wilson_95_ucb": unrecoverable_ucb,
        "world_count": len(worlds),
        "all_108_correct_world_count": len(all_correct_worlds),
        "all_108_correct_world_rate": len(all_correct_worlds) / len(worlds) if worlds else 0.0,
        "all_correct_world_ids": all_correct_worlds,
        "counterfactual_triplet_count": len(triplets),
        "all_three_bindings_correct_triplet_count": len(all_correct_triplets),
        "all_three_bindings_correct_triplet_rate": triplet_rate,
        "all_correct_triplet_ids": all_correct_triplets,
        "minimum_factorial_cell_accuracy": min_cell_accuracy,
        "minimum_cell_wilson_95_lcb_report_only": min_cell_lcb,
        "minimum_axis_wilson_95_lcb": min_axis_lcb,
        "cell_metrics": cell_metrics,
        "axis_metrics": axis_metrics,
        "behavior_gate_applies": gate_applies,
        "gate_checks": gate_checks,
        "behavior_gate_pass": bool(gate_applies and all(gate_checks.values())),
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit["valid"] or audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase568 static audit failed")
    gate = protocol["behavior_gate"]
    reports = []
    qualified_triplets: list[dict[str, Any]] = []
    authorized_models = []
    for model in MODELS:
        rows_path = OUT_DIR / f"phase568_{model}_behavior_rows.jsonl"
        execution_path = OUT_DIR / f"phase568_{model}_behavior_execution_summary.json"
        if not rows_path.exists() or not execution_path.exists():
            raise RuntimeError(f"Phase568 behavior is incomplete for {model}")
        execution = read_json(execution_path)
        rows = read_jsonl(rows_path)
        if execution["status"] != "complete" or len(rows) != EXPECTED_MODEL_ROWS:
            raise RuntimeError(f"Phase568 behavior denominator mismatch for {model}")
        if len({row["case_id"] for row in rows}) != EXPECTED_MODEL_ROWS:
            raise RuntimeError(f"Phase568 duplicate behavior rows for {model}")
        if any(row["sealed"] for row in rows):
            raise RuntimeError(f"Phase568 sealed row leaked into behavior run for {model}")
        if execution["rows_sha256"] != sha256_file(rows_path):
            raise RuntimeError(f"Phase568 behavior hash mismatch for {model}")
        split_reports = {
            split: split_report(rows, split, gate) for split in protocol["open_splits"]
        }
        behavior_pass = all(
            split_reports[split]["behavior_gate_pass"] for split in GATE_SPLITS
        )
        role_counts_pass = all(
            split_reports[split]["all_three_bindings_correct_triplet_count"]
            >= gate["minimum_all_correct_role_triplets_per_split"]
            for split in ROLE_SPLITS
        )
        authorized = bool(behavior_pass and role_counts_pass)
        if authorized:
            authorized_models.append(model)
        for split in ROLE_SPLITS + ("unseen_recombination",):
            qualified_triplets.extend({
                "model": model,
                "split": split,
                "triplet_id": triplet_id,
                "authorized_for_internal_collection": authorized and split in ROLE_SPLITS,
                "reserved_for_unseen_only": split == "unseen_recombination",
            } for triplet_id in split_reports[split]["all_correct_triplet_ids"])
        reports.append({
            "model": model,
            "row_count": len(rows),
            "semantic_accuracy": rate(rows, "semantic_correct"),
            "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
            "gate_discovery_pass": split_reports["gate_discovery"]["behavior_gate_pass"],
            "gate_confirmation_pass": split_reports["gate_confirmation"]["behavior_gate_pass"],
            "role_all_correct_triplet_count_gate_pass": role_counts_pass,
            "authorized_for_internal_collection": authorized,
            "split_reports": split_reports,
            "rows_sha256": sha256_file(rows_path),
            "cuda_used": execution["cuda_used"],
            "torch_dtype": execution["torch_dtype"],
            "sealed_split_read": execution["sealed_split_read"],
        })
    registry = {
        "schema_version": "phase568_behavior_qualified_triplet_registry.v1",
        "phase_id": "Phase568",
        "created_at": now(),
        "authorized_models": authorized_models,
        "triplets": qualified_triplets,
        "sealed_split_read": False,
    }
    write_json(TRIPLET_REGISTRY_PATH, registry)
    summary = {
        "schema_version": "phase568_behavior_summary.v1",
        "phase_id": "Phase568",
        "created_at": now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "registered_case_count": protocol["registered_case_count"],
        "open_case_count": protocol["open_case_count"],
        "sealed_case_count_unread": protocol["sealed_case_count"],
        "authorized_models": authorized_models,
        "model_reports": reports,
        "triplet_registry_path": str(TRIPLET_REGISTRY_PATH.relative_to(ROOT)),
        "all_108_world_metric_used_for_gate": False,
        "sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "authorized_models": authorized_models,
        "model_reports": [{
            "model": report["model"],
            "semantic_accuracy": report["semantic_accuracy"],
            "strict_sequence_accuracy": report["strict_sequence_accuracy"],
            "gate_discovery_pass": report["gate_discovery_pass"],
            "gate_confirmation_pass": report["gate_confirmation_pass"],
            "role_count_gate_pass": report["role_all_correct_triplet_count_gate_pass"],
            "authorized": report["authorized_for_internal_collection"],
            "discovery_triplet_rate": report["split_reports"]["gate_discovery"][
                "all_three_bindings_correct_triplet_rate"
            ],
            "confirmation_triplet_rate": report["split_reports"]["gate_confirmation"][
                "all_three_bindings_correct_triplet_rate"
            ],
            "discovery_min_cell_accuracy": report["split_reports"]["gate_discovery"][
                "minimum_factorial_cell_accuracy"
            ],
            "confirmation_min_cell_accuracy": report["split_reports"]["gate_confirmation"][
                "minimum_factorial_cell_accuracy"
            ],
        } for report in reports],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
