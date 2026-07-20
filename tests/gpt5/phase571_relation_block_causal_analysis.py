#!/usr/bin/env python3
"""Analyze Phase571 coarse block causal gates and freeze the stage decision."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase571_relation_block"
MODELS = ("qwen3", "glm4", "deepseek7b")
EXECUTED_MODELS = ("qwen3", "deepseek7b")
PHENOTYPES = ("stable_correct", "stable_relation_confusion")
CONDITIONS = (
    "baseline",
    "signed_block_remove",
    "full_block_remove",
    "full_block_remove_restore",
    "wrong_depth_full_remove",
    "wrong_role_full_remove",
    "random_matched_replace",
)
PROTOCOL_PATH = OUT_DIR / "phase571_frozen_protocol.json"
REGISTRY_PATH = OUT_DIR / "phase571_continuous_block_registry.json"
ANALYSIS_PATH = OUT_DIR / "phase571_coarse_block_causal_analysis.json"
DECISION_PATH = OUT_DIR / "phase571_stage_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
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
    baseline = {row["case_id"]: row for row in rows if row["condition"] == "baseline"}
    changed = {row["case_id"]: row for row in rows if row["condition"] == condition}
    if set(baseline) != set(changed):
        raise RuntimeError(f"Phase571 paired denominator drift for {condition}")
    shifts = [
        changed[case_id]["first_step_target_minus_other_margin"]
        - baseline_row["first_step_target_minus_other_margin"]
        for case_id, baseline_row in baseline.items()
    ]
    exact_matches = sum(
        changed[case_id]["normalized_generated"] == baseline_row["normalized_generated"]
        for case_id, baseline_row in baseline.items()
    )
    semantic_matches = sum(
        changed[case_id]["semantic_event"] == baseline_row["semantic_event"]
        for case_id, baseline_row in baseline.items()
    )
    return {
        "condition": condition,
        "n": len(shifts),
        "mean_margin_shift_from_baseline": sum(shifts) / len(shifts),
        "negative_shift_rate": rate(sum(value < 0.0 for value in shifts), len(shifts)),
        "positive_shift_rate": rate(sum(value > 0.0 for value in shifts), len(shifts)),
        "exact_output_match_baseline_rate": rate(exact_matches, len(shifts)),
        "semantic_event_match_baseline_rate": rate(semantic_matches, len(shifts)),
    }


def analyze_model(model: str, frozen: dict[str, Any]) -> dict[str, Any]:
    rows_path = OUT_DIR / f"phase571_{model}_coarse_block_causal_rows.jsonl.gz"
    execution_path = OUT_DIR / f"phase571_{model}_coarse_block_execution_summary.json"
    execution = read_json(execution_path)
    rows = list(iter_jsonl(rows_path))
    if execution["rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError(f"Phase571 {model} causal hash drift")
    expected = 128 * len(PHENOTYPES) * len(CONDITIONS)
    if len(rows) != expected or any(row["sealed"] for row in rows):
        raise RuntimeError(f"Phase571 {model} causal denominator/seal drift")
    reports = {}
    shifts = {}
    for phenotype in PHENOTYPES:
        phenotype_rows = [row for row in rows if row["causal_phenotype"] == phenotype]
        reports[phenotype] = {
            condition: condition_report(phenotype_rows, condition)
            for condition in CONDITIONS
        }
        shifts[phenotype] = {
            condition: paired_shift(phenotype_rows, condition)
            for condition in CONDITIONS if condition != "baseline"
        }
    gate = frozen["coarse_block_gate"]
    correct = reports["stable_correct"]
    confusion = reports["stable_relation_confusion"]
    correct_damage = (
        correct["baseline"]["semantic_accuracy"]
        - correct["full_block_remove"]["semantic_accuracy"]
    )
    confusion_repair = (
        confusion["full_block_remove"]["semantic_accuracy"]
        - confusion["baseline"]["semantic_accuracy"]
    )
    correct_control_damage = max(
        correct["baseline"]["semantic_accuracy"]
        - correct[condition]["semantic_accuracy"]
        for condition in ("wrong_depth_full_remove", "wrong_role_full_remove")
    )
    confusion_control_repair = max(
        confusion[condition]["semantic_accuracy"]
        - confusion["baseline"]["semantic_accuracy"]
        for condition in ("wrong_depth_full_remove", "wrong_role_full_remove")
    )
    correct_signed_shift = shifts["stable_correct"]["signed_block_remove"][
        "mean_margin_shift_from_baseline"
    ]
    confusion_signed_shift = shifts["stable_relation_confusion"]["signed_block_remove"][
        "mean_margin_shift_from_baseline"
    ]
    checks = {
        "minimum_paired_cases": all(
            reports[phenotype]["baseline"]["n"]
            >= gate["minimum_paired_cases_per_phenotype"]
            for phenotype in PHENOTYPES
        ),
        "signed_projection_bidirectional_diagnostic": (
            correct_signed_shift < 0.0 and confusion_signed_shift > 0.0
        ),
        "correct_natural_behavior_necessity": (
            correct_damage >= gate["minimum_correct_accuracy_damage"]
        ),
        "confusion_natural_behavior_repair": (
            confusion_repair >= gate["minimum_confusion_accuracy_repair"]
        ),
        "same_case_restore_correct": (
            correct["baseline"]["semantic_accuracy"]
            - correct["full_block_remove_restore"]["semantic_accuracy"]
            <= gate["maximum_restore_rate_loss"]
        ),
        "same_case_restore_confusion": (
            confusion["baseline"]["relation_confusion_rate"]
            - confusion["full_block_remove_restore"]["relation_confusion_rate"]
            <= gate["maximum_restore_rate_loss"]
        ),
        "correct_effect_specific_over_wrong_depth_and_role": (
            correct_damage - correct_control_damage
            >= gate["minimum_behavior_specificity_advantage"]
        ),
        "confusion_repair_specific_over_wrong_depth_and_role": (
            confusion_repair - confusion_control_repair
            >= gate["minimum_behavior_specificity_advantage"]
        ),
    }
    required = [
        key for key in checks if key != "signed_projection_bidirectional_diagnostic"
    ]
    passed = all(checks[key] for key in required)
    return {
        "model": model,
        "selected_block": execution["selected_block"],
        "candidate_paired_cases_per_phenotype": execution[
            "candidate_paired_cases_per_phenotype"
        ],
        "baseline_valid_pair_count": execution["baseline_valid_pair_count"],
        "baseline_drift_pair_count": execution["baseline_drift_pair_count"],
        "condition_reports_by_phenotype": reports,
        "paired_effects_by_phenotype": shifts,
        "derived_behavior_effects": {
            "correct_damage": correct_damage,
            "confusion_repair": confusion_repair,
            "largest_wrong_depth_or_role_correct_damage": correct_control_damage,
            "largest_wrong_depth_or_role_confusion_repair": confusion_control_repair,
            "correct_specificity_advantage": correct_damage - correct_control_damage,
            "confusion_specificity_advantage": confusion_repair - confusion_control_repair,
        },
        "coarse_block_checks": checks,
        "coarse_block_gate_pass": passed,
        "signed_projection_diagnostic_is_not_a_gate": True,
        "relation_selection_mechanism_claimed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }


def analyze() -> dict[str, Any]:
    frozen = read_json(PROTOCOL_PATH)
    registry = read_json(REGISTRY_PATH)
    reports = [analyze_model(model, frozen) for model in EXECUTED_MODELS]
    passed = [report["model"] for report in reports if report["coarse_block_gate_pass"]]
    skipped = [model for model in MODELS if model not in EXECUTED_MODELS]
    analysis = {
        "schema_version": "phase571_coarse_block_causal_analysis.v1",
        "phase_id": "Phase571",
        "created_at": now(),
        "status": "complete",
        "coarse_block_gate": frozen["coarse_block_gate"],
        "model_reports": reports,
        "skipped_models_without_confirmed_block": skipped,
        "passed_models": passed,
        "passed_model_count": len(passed),
        "donor_stage_authorized": bool(passed),
        "relation_selection_mechanism_claimed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, analysis)
    decision = {
        "schema_version": "phase571_stage_decision.v1",
        "phase_id": "Phase571",
        "created_at": now(),
        "continuous_block_registry_sha256": sha256_file(REGISTRY_PATH),
        "coarse_block_analysis_sha256": sha256_file(ANALYSIS_PATH),
        "confirmed_observer_blocks": registry["authorized_models"],
        "coarse_causal_gate_passed_models": passed,
        "advance_to_relation_donor_stage": bool(passed),
        "advance_to_head_channel_parameter_neuron_scan": False,
        "stop_reason_if_no_pass": (
            "No confirmed coarse block jointly passed natural necessity, confusion repair, "
            "same-case restoration, and wrong-depth/wrong-role specificity."
            if not passed else None
        ),
        "single_layer_patch_route_reopened": False,
        "sealed_split_read": False,
    }
    write_json(DECISION_PATH, decision)
    print(json.dumps({
        "passed_models": passed,
        "skipped_models": skipped,
        "advance_to_donor": decision["advance_to_relation_donor_stage"],
        "models": [
            {
                "model": report["model"],
                "effects": report["derived_behavior_effects"],
                "checks": report["coarse_block_checks"],
            }
            for report in reports
        ],
    }, ensure_ascii=False, indent=2))
    return analysis


if __name__ == "__main__":
    analyze()
