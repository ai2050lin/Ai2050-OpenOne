#!/usr/bin/env python3
"""Analyze Phase569 stable-correct and stable-relation-confusion phenotypes."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase569_relation_competition"
PROTOCOL_PATH = OUT_DIR / "phase569_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase569_static_audit.json"
SUMMARY_PATH = OUT_DIR / "phase569_behavior_summary.json"
REGISTRY_PATH = OUT_DIR / "phase569_path_phenotype_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
PHENOTYPE_SPLITS = ("phenotype_discovery", "phenotype_confirmation")
PATH_SPLITS = ("path_discovery", "path_confirmation")
EXPECTED_MODEL_ROWS = 48384
TRACE_CASE_CAP = 96


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def split_summary(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    correct = sum(bool(row["semantic_correct"]) for row in selected)
    confusion = sum(bool(row["relation_confusion"]) for row in selected)
    recoverable = sum(bool(row["semantic_event_recoverable"]) for row in selected)
    strict = sum(bool(row["strict_sequence_correct"]) for row in selected)
    events: dict[str, int] = defaultdict(int)
    for row in selected:
        events[row["semantic_event"]] += 1
    triplets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        triplets[row["triplet_id"]].append(row)
        worlds[row["anchor_id"]].append(row)
    if any(len(group) != 3 for group in triplets.values()):
        raise RuntimeError(f"Phase569 triplet denominator drift in {split}")
    if any(len(group) != 108 for group in worlds.values()):
        raise RuntimeError(f"Phase569 world denominator drift in {split}")
    return {
        "split": split,
        "row_count": len(selected),
        "semantic_correct_count": correct,
        "semantic_accuracy": rate(correct, len(selected)),
        "strict_sequence_correct_count": strict,
        "strict_sequence_accuracy": rate(strict, len(selected)),
        "relation_confusion_count": confusion,
        "relation_confusion_rate_all_rows": rate(confusion, len(selected)),
        "relation_confusion_share_of_registered_errors": rate(
            confusion, len(selected) - correct
        ),
        "recoverable_count": recoverable,
        "recoverable_rate": rate(recoverable, len(selected)),
        "event_counts": dict(sorted(events.items())),
        "world_count": len(worlds),
        "all_108_correct_world_count": sum(
            all(row["semantic_correct"] for row in group) for group in worlds.values()
        ),
        "triplet_count": len(triplets),
        "all_three_correct_triplet_count": sum(
            all(row["semantic_correct"] for row in group) for group in triplets.values()
        ),
    }


def cell_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["factorial_cell"]].append(row)
    metrics = {}
    for cell, group in sorted(groups.items()):
        correct = sum(bool(row["semantic_correct"]) for row in group)
        confusion = sum(bool(row["relation_confusion"]) for row in group)
        errors = len(group) - correct
        metrics[cell] = {
            "n": len(group),
            "correct_count": correct,
            "accuracy": rate(correct, len(group)),
            "registered_error_count": errors,
            "relation_confusion_count": confusion,
            "relation_confusion_rate_all_rows": rate(confusion, len(group)),
            "relation_confusion_share_of_registered_errors": rate(confusion, errors),
        }
    return metrics


def stable_cells(
    by_split: dict[str, dict[str, dict[str, Any]]], gate: dict[str, Any]
) -> tuple[list[str], list[str]]:
    shared = set(by_split[PHENOTYPE_SPLITS[0]]) & set(by_split[PHENOTYPE_SPLITS[1]])
    correct_cells = []
    confusion_cells = []
    for cell in sorted(shared):
        metrics = [by_split[split][cell] for split in PHENOTYPE_SPLITS]
        if all(
            metric["accuracy"] >= gate["stable_correct_cell_accuracy_min"]
            for metric in metrics
        ):
            correct_cells.append(cell)
        if all(
            metric["relation_confusion_rate_all_rows"]
            >= gate["stable_confusion_cell_rate_min"]
            and metric["relation_confusion_share_of_registered_errors"]
            >= gate["stable_confusion_share_of_registered_errors_min"]
            and metric["accuracy"] <= gate["stable_confusion_cell_accuracy_max"]
            for metric in metrics
        ):
            confusion_cells.append(cell)
    return correct_cells, confusion_cells


def phenotype_rows(
    rows: list[dict[str, Any]], split: str, phenotype: str, cells: set[str]
) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["split"] == split and row["factorial_cell"] in cells
    ]
    if phenotype == "stable_correct":
        return [row for row in selected if row["semantic_correct"]]
    if phenotype == "stable_relation_confusion":
        return [row for row in selected if row["relation_confusion"]]
    raise KeyError(phenotype)


def balanced_trace_selection(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: item["semantic_case_id"]):
        strata[(
            row["factorial_cell"], row["target"], row["other_relation_target"]
        )].append(row)
    selected = []
    keys = sorted(strata)
    while len(selected) < cap and keys:
        next_keys = []
        for key in keys:
            if len(selected) >= cap:
                break
            if strata[key]:
                selected.append(strata[key].popleft())
            if strata[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def path_report(
    rows: list[dict[str, Any]], split: str, phenotype: str, cells: set[str], gate: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = phenotype_rows(rows, split, phenotype, cells)
    distinct_cells = {row["factorial_cell"] for row in eligible}
    target_other_pairs = {
        (row["target"], row["other_relation_target"]) for row in eligible
    }
    checks = {
        "minimum_cases": (
            len(eligible) >= gate["minimum_path_cases_per_phenotype_per_split"]
        ),
        "minimum_distinct_cells": (
            len(distinct_cells) >= gate["minimum_path_distinct_cells_per_phenotype"]
        ),
        "minimum_target_other_pairs": (
            len(target_other_pairs) >= gate["minimum_path_target_other_pairs_per_phenotype"]
        ),
    }
    trace_rows = balanced_trace_selection(eligible, TRACE_CASE_CAP)
    report = {
        "split": split,
        "phenotype": phenotype,
        "eligible_case_count": len(eligible),
        "distinct_cell_count": len(distinct_cells),
        "distinct_target_other_pair_count": len(target_other_pairs),
        "trace_case_count": len(trace_rows),
        "checks": checks,
        "qualified": all(checks.values()),
    }
    return report, trace_rows


def analyze_model(model: str, protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows_path = OUT_DIR / f"phase569_{model}_behavior_rows.jsonl"
    execution_path = OUT_DIR / f"phase569_{model}_behavior_execution_summary.json"
    contract_path = OUT_DIR / f"phase569_{model}_behavior_run_contract.json"
    for path in (rows_path, execution_path, contract_path):
        if not path.exists():
            raise RuntimeError(f"Missing Phase569 artifact: {path}")
    execution = read_json(execution_path)
    contract = read_json(contract_path)
    rows = list(iter_jsonl(rows_path))
    if len(rows) != EXPECTED_MODEL_ROWS or len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"Phase569 {model} behavior denominator is incomplete")
    if any(row["model"] != model or row["sealed"] for row in rows):
        raise RuntimeError(f"Phase569 {model} output identity/seal drift")
    if any(row["torch_dtype"] != "torch.bfloat16" or row["quantized_8bit"] for row in rows):
        raise RuntimeError(f"Phase569 {model} precision drift")
    if execution["rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError(f"Phase569 {model} output hash drift")
    if contract["open_cases_sha256"] != protocol["open_cases_sha256"]:
        raise RuntimeError(f"Phase569 {model} source bank drift")
    split_reports = [split_summary(rows, split) for split in (*PHENOTYPE_SPLITS, *PATH_SPLITS)]
    metrics_by_split = {
        split: cell_metrics(row for row in rows if row["split"] == split)
        for split in PHENOTYPE_SPLITS
    }
    gate = protocol["phenotype_gate"]
    correct_cells, confusion_cells = stable_cells(metrics_by_split, gate)
    phenotype_cells = {
        "stable_correct": set(correct_cells),
        "stable_relation_confusion": set(confusion_cells),
    }
    path_reports = []
    registry_entries = []
    for phenotype, cells in phenotype_cells.items():
        for split in PATH_SPLITS:
            report, trace_rows = path_report(rows, split, phenotype, cells, gate)
            path_reports.append(report)
            registry_entries.append({
                "model": model,
                "phenotype": phenotype,
                "split": split,
                "qualified": report["qualified"],
                "case_count": len(trace_rows),
                "semantic_case_ids": [row["semantic_case_id"] for row in trace_rows],
                "case_ids": [row["case_id"] for row in trace_rows],
            })
    authorized = bool(
        correct_cells
        and confusion_cells
        and len(path_reports) == 4
        and all(report["qualified"] for report in path_reports)
    )
    report = {
        "model": model,
        "registered_case_count": len(rows),
        "overall_semantic_accuracy": rate(
            sum(bool(row["semantic_correct"]) for row in rows), len(rows)
        ),
        "overall_strict_sequence_accuracy": rate(
            sum(bool(row["strict_sequence_correct"]) for row in rows), len(rows)
        ),
        "overall_relation_confusion_count": sum(
            bool(row["relation_confusion"]) for row in rows
        ),
        "split_reports": split_reports,
        "phenotype_cell_metrics": metrics_by_split,
        "stable_correct_cell_count": len(correct_cells),
        "stable_correct_cells": correct_cells,
        "stable_relation_confusion_cell_count": len(confusion_cells),
        "stable_relation_confusion_cells": confusion_cells,
        "path_reports": path_reports,
        "authorized_for_coarse_internal_trace": authorized,
        "cuda_used": execution["cuda_used"],
        "torch_dtype": execution["torch_dtype"],
        "quantized_8bit": execution["quantized_8bit"],
        "sealed_split_read": False,
    }
    registry = {
        "model": model,
        "authorized_for_coarse_internal_trace": authorized,
        "entries": registry_entries,
    }
    return report, registry


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit["valid"] or audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase569 static audit failed")
    reports = []
    registries = []
    for model in MODELS:
        report, registry = analyze_model(model, protocol)
        reports.append(report)
        registries.append(registry)
    authorized = [
        report["model"] for report in reports
        if report["authorized_for_coarse_internal_trace"]
    ]
    summary = {
        "schema_version": "phase569_behavior_summary.v1",
        "phase_id": "Phase569",
        "created_at": now(),
        "status": "complete",
        "registered_semantic_case_count": protocol["registered_semantic_case_count"],
        "open_semantic_case_count": protocol["open_semantic_case_count"],
        "open_model_evaluation_count": protocol["open_model_evaluation_count"],
        "phenotype_error_denominator_definition": (
            "all semantic-incorrect rows in the registered factorial cell"
        ),
        "model_reports": reports,
        "authorized_models_for_coarse_internal_trace": authorized,
        "sealed_split_read": False,
        "closure_claimed": False,
    }
    registry = {
        "schema_version": "phase569_path_phenotype_registry.v1",
        "phase_id": "Phase569",
        "created_at": now(),
        "trace_case_cap_per_model_phenotype_split": TRACE_CASE_CAP,
        "authorized_models": authorized,
        "models": registries,
        "sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    write_json(REGISTRY_PATH, registry)
    print(json.dumps({
        "authorized_models": authorized,
        "models": [
            {
                "model": report["model"],
                "accuracy": report["overall_semantic_accuracy"],
                "stable_correct_cells": report["stable_correct_cell_count"],
                "stable_confusion_cells": report["stable_relation_confusion_cell_count"],
                "authorized": report["authorized_for_coarse_internal_trace"],
            }
            for report in reports
        ],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
