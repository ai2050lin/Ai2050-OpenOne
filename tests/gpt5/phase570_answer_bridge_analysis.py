#!/usr/bin/env python3
"""Analyze Phase570 late answer-competition bridge causal screens."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase570_answer_bridge_causal"
PROTOCOL_PATH = OUT_DIR / "phase570_frozen_protocol.json"
SUMMARY_PATH = OUT_DIR / "phase570_causal_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = (
    "baseline",
    "target_projection_remove",
    "random_matched_remove",
    "wrong_layer_projection_remove",
)
PHENOTYPES = ("stable_correct", "stable_relation_confusion")
MINIMUM_CASES = 48
SPECIFICITY_RATIO = 1.25
BEHAVIOR_CHANGE_MIN = 0.10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
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


def condition_report(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    events: dict[str, int] = defaultdict(int)
    for row in selected:
        events[row["semantic_event"]] += 1
    return {
        "condition": condition,
        "n": len(selected),
        "semantic_accuracy": rate(
            sum(bool(row["semantic_correct"]) for row in selected), len(selected)
        ),
        "relation_confusion_rate": rate(
            sum(bool(row["relation_confusion"]) for row in selected), len(selected)
        ),
        "recoverable_rate": rate(
            sum(bool(row["semantic_event_recoverable"]) for row in selected), len(selected)
        ),
        "mean_first_step_target_minus_other_margin": (
            sum(row["first_step_target_minus_other_margin"] for row in selected)
            / len(selected) if selected else 0.0
        ),
        "event_counts": dict(sorted(events.items())),
    }


def paired_shift(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    by_condition = {
        item: {row["case_id"]: row for row in rows if row["condition"] == item}
        for item in ("baseline", condition)
    }
    if set(by_condition["baseline"]) != set(by_condition[condition]):
        raise RuntimeError(f"Phase570 paired denominator drift for {condition}")
    shifts = [
        by_condition[condition][case_id]["first_step_target_minus_other_margin"]
        - baseline["first_step_target_minus_other_margin"]
        for case_id, baseline in by_condition["baseline"].items()
    ]
    return {
        "condition": condition,
        "n": len(shifts),
        "mean_margin_shift_from_baseline": sum(shifts) / len(shifts),
        "negative_shift_rate": rate(sum(value < 0.0 for value in shifts), len(shifts)),
        "positive_shift_rate": rate(sum(value > 0.0 for value in shifts), len(shifts)),
    }


def analyze_model(model: str) -> dict[str, Any]:
    rows_path = OUT_DIR / f"phase570_{model}_causal_rows.jsonl"
    execution_path = OUT_DIR / f"phase570_{model}_execution_summary.json"
    if not rows_path.exists() or not execution_path.exists():
        raise RuntimeError(f"Missing Phase570 model artifacts for {model}")
    execution = read_json(execution_path)
    rows = list(iter_jsonl(rows_path))
    if execution["causal_rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError(f"Phase570 {model} causal hash drift")
    if any(row["model"] != model or row["sealed"] for row in rows):
        raise RuntimeError(f"Phase570 {model} identity/seal drift")
    reports = {}
    shifts = {}
    checks = {}
    for phenotype in PHENOTYPES:
        phenotype_rows = [row for row in rows if row["intended_phenotype"] == phenotype]
        condition_reports = {
            condition: condition_report(phenotype_rows, condition)
            for condition in CONDITIONS
        }
        paired = {
            condition: paired_shift(phenotype_rows, condition)
            for condition in CONDITIONS if condition != "baseline"
        }
        reports[phenotype] = condition_reports
        shifts[phenotype] = paired
    correct_target = shifts["stable_correct"]["target_projection_remove"][
        "mean_margin_shift_from_baseline"
    ]
    confusion_target = shifts["stable_relation_confusion"]["target_projection_remove"][
        "mean_margin_shift_from_baseline"
    ]
    specificity = {}
    for phenotype in PHENOTYPES:
        target = abs(shifts[phenotype]["target_projection_remove"]["mean_margin_shift_from_baseline"])
        random = abs(shifts[phenotype]["random_matched_remove"]["mean_margin_shift_from_baseline"])
        wrong = abs(shifts[phenotype]["wrong_layer_projection_remove"]["mean_margin_shift_from_baseline"])
        specificity[phenotype] = {
            "target_absolute_mean_shift": target,
            "random_absolute_mean_shift": random,
            "wrong_layer_absolute_mean_shift": wrong,
            "target_vs_largest_control_ratio": target / max(random, wrong, 1e-12),
            "passes_ratio": target >= SPECIFICITY_RATIO * max(random, wrong, 1e-12),
        }
    correct_baseline = reports["stable_correct"]["baseline"]["semantic_accuracy"]
    correct_remove = reports["stable_correct"]["target_projection_remove"]["semantic_accuracy"]
    confusion_baseline = reports["stable_relation_confusion"]["baseline"][
        "relation_confusion_rate"
    ]
    confusion_remove = reports["stable_relation_confusion"]["target_projection_remove"][
        "relation_confusion_rate"
    ]
    confusion_accuracy_baseline = reports["stable_relation_confusion"]["baseline"][
        "semantic_accuracy"
    ]
    confusion_accuracy_remove = reports["stable_relation_confusion"][
        "target_projection_remove"
    ]["semantic_accuracy"]
    checks = {
        "minimum_cases": all(
            reports[phenotype]["baseline"]["n"] >= MINIMUM_CASES
            for phenotype in PHENOTYPES
        ),
        "bidirectional_margin": correct_target < 0.0 and confusion_target > 0.0,
        "specific_over_random_and_wrong_layer": all(
            specificity[phenotype]["passes_ratio"] for phenotype in PHENOTYPES
        ),
        "correct_behavior_degrades": (
            correct_baseline - correct_remove >= BEHAVIOR_CHANGE_MIN
        ),
        "confusion_behavior_improves": (
            confusion_baseline - confusion_remove >= BEHAVIOR_CHANGE_MIN
            or confusion_accuracy_remove - confusion_accuracy_baseline >= BEHAVIOR_CHANGE_MIN
        ),
    }
    screen_pass = all(checks.values())
    return {
        "model": model,
        "target_layer": execution["target_layer"],
        "wrong_layer_control": execution["wrong_layer_control"],
        "selected_causal_case_counts": execution["selected_causal_case_counts"],
        "condition_reports_by_phenotype": reports,
        "paired_margin_shifts_by_phenotype": shifts,
        "specificity_by_phenotype": specificity,
        "screen_checks": checks,
        "late_answer_bridge_causal_screen_pass": screen_pass,
        "causal_scope": "late answer competition bridge only",
        "upstream_relation_encoding_claimed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    reports = [analyze_model(model) for model in MODELS]
    passed = [
        report["model"] for report in reports
        if report["late_answer_bridge_causal_screen_pass"]
    ]
    summary = {
        "schema_version": "phase570_causal_summary.v1",
        "phase_id": "Phase570",
        "created_at": now(),
        "status": "complete",
        "screening_thresholds": {
            "minimum_cases_per_phenotype": MINIMUM_CASES,
            "target_vs_largest_control_absolute_shift_ratio": SPECIFICITY_RATIO,
            "minimum_behavior_rate_change": BEHAVIOR_CHANGE_MIN,
            "screen_is_not_mechanism_closure": True,
        },
        "model_reports": reports,
        "passed_models": passed,
        "passed_model_count": len(passed),
        "cross_model_late_answer_bridge_screen_pass": len(passed) >= 2,
        "causal_scope": protocol["causal_screen_scope"],
        "upstream_relation_encoding_claimed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "passed_models": passed,
        "cross_model_pass": summary["cross_model_late_answer_bridge_screen_pass"],
        "models": [
            {
                "model": report["model"],
                "checks": report["screen_checks"],
                "correct_target_shift": report["paired_margin_shifts_by_phenotype"]
                ["stable_correct"]["target_projection_remove"]["mean_margin_shift_from_baseline"],
                "confusion_target_shift": report["paired_margin_shifts_by_phenotype"]
                ["stable_relation_confusion"]["target_projection_remove"]
                ["mean_margin_shift_from_baseline"],
            }
            for report in reports
        ],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
