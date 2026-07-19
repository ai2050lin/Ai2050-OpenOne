#!/usr/bin/env python3
"""Analyze the frozen Phase558 fixed-identity behavior denominator."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase558_fixed_identity_color"
PROTOCOL_PATH = OUT_DIR / "phase558_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase558_static_audit.json"
SUMMARY_PATH = OUT_DIR / "phase558_behavior_summary.json"
ANCHOR_REGISTRY_PATH = OUT_DIR / "phase558_behavior_qualified_anchor_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
BEHAVIOR_SPLITS = ("behavior_discovery", "behavior_confirmation")
PATH_SPLITS = ("path_discovery", "path_confirmation")
Z = 1.96


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def split_report(rows: list[dict[str, Any]], split: str, gate: dict[str, Any]) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    surfaces: dict[int, list[dict[str, Any]]] = defaultdict(list)
    orders: dict[int, list[dict[str, Any]]] = defaultdict(list)
    color_regimes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        worlds[row["anchor_id"]].append(row)
        cells[row["factorial_cell"]].append(row)
        pairs[row["pair_id"]].append(row)
        surfaces[int(row["surface_id"])].append(row)
        orders[int(row["fact_order"])].append(row)
        color_regimes[row["color_regime"]].append(row)
    all_correct_worlds = sorted(
        anchor for anchor, group in worlds.items()
        if len(group) == 32 and all(row["semantic_correct"] for row in group)
    )
    all_correct_pairs = sum(
        len(group) == 2 and all(row["semantic_correct"] for row in group)
        for group in pairs.values()
    )
    cell_metrics = {}
    for cell, group in sorted(cells.items()):
        correct = sum(row["semantic_correct"] for row in group)
        lcb, ucb = wilson(correct, len(group))
        cell_metrics[cell] = {
            "n": len(group), "correct": correct, "accuracy": correct / len(group),
            "wilson_95_lcb": lcb, "wilson_95_ucb": ucb,
        }
    unrecoverable = sum(not row["semantic_event_recoverable"] for row in selected)
    unrecoverable_lcb, unrecoverable_ucb = wilson(unrecoverable, len(selected))
    world_rate = len(all_correct_worlds) / len(worlds) if worlds else 0.0
    min_cell_lcb = min((metric["wilson_95_lcb"] for metric in cell_metrics.values()), default=0.0)
    behavior_gate_applies = split in BEHAVIOR_SPLITS
    gate_pass = bool(
        behavior_gate_applies
        and world_rate >= gate["world_all_32_rate_min_per_behavior_split"]
        and min_cell_lcb >= gate["minimum_cell_wilson_95_lcb"]
        and unrecoverable_ucb <= gate["unrecoverable_wilson_95_ucb_max"]
    )
    return {
        "split": split,
        "row_count": len(selected),
        "semantic_accuracy": rate(selected, "semantic_correct"),
        "strict_sequence_accuracy": rate(selected, "strict_sequence_correct"),
        "unrecoverable_count": unrecoverable,
        "unrecoverable_rate": unrecoverable / len(selected) if selected else 0.0,
        "unrecoverable_wilson_95_lcb": unrecoverable_lcb,
        "unrecoverable_wilson_95_ucb": unrecoverable_ucb,
        "world_count": len(worlds),
        "all_32_correct_world_count": len(all_correct_worlds),
        "all_32_correct_world_rate": world_rate,
        "all_correct_world_ids": all_correct_worlds,
        "counterfactual_pair_count": len(pairs),
        "both_bindings_correct_pair_count": all_correct_pairs,
        "both_bindings_correct_pair_rate": all_correct_pairs / len(pairs) if pairs else 0.0,
        "minimum_cell_wilson_95_lcb": min_cell_lcb,
        "cell_metrics": cell_metrics,
        "surface_accuracy": {
            str(key): rate(group, "semantic_correct") for key, group in sorted(surfaces.items())
        },
        "fact_order_accuracy": {
            str(key): rate(group, "semantic_correct") for key, group in sorted(orders.items())
        },
        "color_regime_accuracy": {
            key: rate(group, "semantic_correct") for key, group in sorted(color_regimes.items())
        },
        "behavior_gate_applies": behavior_gate_applies,
        "behavior_gate_pass": gate_pass,
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit["valid"]:
        raise RuntimeError("Phase558 static audit failed")
    gate = protocol["behavior_gate"]
    reports = []
    qualified_anchors: list[dict[str, Any]] = []
    authorized_models = []
    for model in MODELS:
        rows_path = OUT_DIR / f"phase558_{model}_behavior_rows.jsonl"
        execution_path = OUT_DIR / f"phase558_{model}_behavior_execution_summary.json"
        if not rows_path.exists() or not execution_path.exists():
            raise RuntimeError(f"Phase558 behavior is incomplete for {model}")
        execution = read_json(execution_path)
        rows = read_jsonl(rows_path)
        if execution["status"] != "complete" or len(rows) != 9216:
            raise RuntimeError(f"Phase558 behavior denominator mismatch for {model}")
        split_reports = {
            split: split_report(rows, split, gate)
            for split in protocol["open_splits"]
        }
        behavior_pass = all(split_reports[split]["behavior_gate_pass"] for split in BEHAVIOR_SPLITS)
        path_counts_pass = all(
            split_reports[split]["all_32_correct_world_count"]
            >= gate["minimum_all_correct_path_worlds_per_split"]
            for split in PATH_SPLITS
        )
        authorized = bool(behavior_pass and path_counts_pass)
        if authorized:
            authorized_models.append(model)
        for split in PATH_SPLITS + ("unseen_recombination",):
            qualified_anchors.extend({
                "model": model,
                "split": split,
                "anchor_id": anchor_id,
                "authorized_for_internal_collection": authorized and split in PATH_SPLITS,
                "reserved_for_unseen_only": split == "unseen_recombination",
            } for anchor_id in split_reports[split]["all_correct_world_ids"])
        reports.append({
            "model": model,
            "row_count": len(rows),
            "semantic_accuracy": rate(rows, "semantic_correct"),
            "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
            "behavior_discovery_pass": split_reports["behavior_discovery"]["behavior_gate_pass"],
            "behavior_confirmation_pass": split_reports["behavior_confirmation"]["behavior_gate_pass"],
            "path_all_correct_count_gate_pass": path_counts_pass,
            "authorized_for_internal_collection": authorized,
            "split_reports": split_reports,
            "rows_sha256": sha256_file(rows_path),
            "cuda_used": execution["cuda_used"],
            "torch_dtype": execution["torch_dtype"],
            "sealed_split_read": execution["sealed_split_read"],
        })
    registry = {
        "schema_version": "phase558_behavior_qualified_anchor_registry.v1",
        "phase_id": "Phase558",
        "created_at": now(),
        "authorized_models": authorized_models,
        "anchors": qualified_anchors,
        "sealed_split_read": False,
    }
    write_json(ANCHOR_REGISTRY_PATH, registry)
    summary = {
        "schema_version": "phase558_behavior_summary.v1",
        "phase_id": "Phase558",
        "created_at": now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "registered_case_count": protocol["registered_case_count"],
        "open_case_count": protocol["open_case_count"],
        "sealed_case_count_unread": protocol["sealed_case_count"],
        "authorized_models": authorized_models,
        "model_reports": reports,
        "anchor_registry_path": str(ANCHOR_REGISTRY_PATH.relative_to(ROOT)),
        "sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "authorized_models": authorized_models,
        "model_reports": [{
            "model": report["model"],
            "semantic_accuracy": report["semantic_accuracy"],
            "strict_sequence_accuracy": report["strict_sequence_accuracy"],
            "behavior_discovery_pass": report["behavior_discovery_pass"],
            "behavior_confirmation_pass": report["behavior_confirmation_pass"],
            "path_all_correct_count_gate_pass": report["path_all_correct_count_gate_pass"],
            "authorized_for_internal_collection": report["authorized_for_internal_collection"],
            "discovery_world_rate": report["split_reports"]["behavior_discovery"]["all_32_correct_world_rate"],
            "confirmation_world_rate": report["split_reports"]["behavior_confirmation"]["all_32_correct_world_rate"],
            "discovery_min_cell_lcb": report["split_reports"]["behavior_discovery"]["minimum_cell_wilson_95_lcb"],
            "confirmation_min_cell_lcb": report["split_reports"]["behavior_confirmation"]["minimum_cell_wilson_95_lcb"],
        } for report in reports],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
