#!/usr/bin/env python3
"""Analyze Phase571 Qwen3 donor interventions and freeze the final decision."""

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
MODEL = "qwen3"
PHENOTYPES = ("stable_correct", "stable_relation_confusion")
CONDITIONS = (
    "baseline",
    "self_entry_restore",
    "matched_correct_answer_entry",
    "matched_correct_answer_exit",
    "matched_correct_query_entry",
    "matched_correct_target_fact_entry",
    "wrong_target_answer_entry",
    "random_matched_answer_entry",
)
ROWS_PATH = OUT_DIR / "phase571_qwen3_relation_donor_rows.jsonl.gz"
EXECUTION_PATH = OUT_DIR / "phase571_qwen3_relation_donor_execution_summary.json"
PROTOCOL_PATH = OUT_DIR / "phase571_relation_donor_frozen_protocol.json"
REGISTRY_PATH = OUT_DIR / "phase571_relation_donor_registry.json"
ANALYSIS_PATH = OUT_DIR / "phase571_relation_donor_analysis.json"
DECISION_PATH = OUT_DIR / "phase571_final_stage_decision.json"


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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
        "strict_sequence_accuracy": rate(
            sum(bool(row["strict_sequence_correct"]) for row in selected), len(selected)
        ),
        "relation_confusion_rate": rate(
            sum(bool(row["relation_confusion"]) for row in selected), len(selected)
        ),
        "recoverable_rate": rate(
            sum(bool(row["semantic_event_recoverable"]) for row in selected), len(selected)
        ),
        "wrong_donor_target_output_rate": rate(
            sum(
                row.get("selected_candidate") == row.get("wrong_donor_target")
                for row in selected
            ),
            len(selected),
        ),
        "mean_first_step_target_minus_other_margin": (
            sum(row["first_step_target_minus_other_margin"] for row in selected)
            / len(selected)
            if selected
            else 0.0
        ),
        "event_counts": dict(sorted(events.items())),
    }


def paired_shift(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in rows if row["condition"] == "baseline"}
    changed = {row["case_id"]: row for row in rows if row["condition"] == condition}
    if set(baseline) != set(changed):
        raise RuntimeError(f"Phase571 donor paired denominator drift for {condition}")
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


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    registry = read_json(REGISTRY_PATH)
    execution = read_json(EXECUTION_PATH)
    rows = list(iter_jsonl(ROWS_PATH))
    if protocol["registry_sha256"] != sha256_file(REGISTRY_PATH):
        raise RuntimeError("Phase571 donor registry hash drift")
    if execution["rows_sha256"] != sha256_file(ROWS_PATH):
        raise RuntimeError("Phase571 donor row hash drift")
    expected = 128 * len(PHENOTYPES) * len(CONDITIONS)
    if len(rows) != expected or any(row["sealed"] for row in rows):
        raise RuntimeError("Phase571 donor denominator/seal drift")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise RuntimeError("Phase571 donor condition drift")

    reports: dict[str, Any] = {}
    shifts: dict[str, Any] = {}
    for phenotype in PHENOTYPES:
        phenotype_rows = [
            row for row in rows if row["receiver_phenotype"] == phenotype
        ]
        reports[phenotype] = {
            condition: condition_report(phenotype_rows, condition)
            for condition in CONDITIONS
        }
        shifts[phenotype] = {
            condition: paired_shift(phenotype_rows, condition)
            for condition in CONDITIONS
            if condition != "baseline"
        }

    correct = reports["stable_correct"]
    confusion = reports["stable_relation_confusion"]
    gate = protocol["donor_gate"]

    def repair(condition: str) -> float:
        return (
            confusion[condition]["semantic_accuracy"]
            - confusion["baseline"]["semantic_accuracy"]
        )

    def preservation(condition: str) -> float:
        return correct[condition]["semantic_accuracy"]

    entry_repair = repair("matched_correct_answer_entry")
    exit_repair = repair("matched_correct_answer_exit")
    query_repair = repair("matched_correct_query_entry")
    fact_repair = repair("matched_correct_target_fact_entry")
    wrong_repair = repair("wrong_target_answer_entry")
    random_repair = repair("random_matched_answer_entry")
    upstream_repair = max(query_repair, fact_repair)
    self_restore_matches = [
        shifts[phenotype]["self_entry_restore"]["semantic_event_match_baseline_rate"]
        for phenotype in PHENOTYPES
    ]
    self_restore_semantic_match = sum(self_restore_matches) / len(self_restore_matches)

    frozen_checks = {
        "minimum_paired_cases": all(
            reports[phenotype]["baseline"]["n"] >= 128 for phenotype in PHENOTYPES
        ),
        "self_restore_semantic_match": (
            self_restore_semantic_match
            >= gate["minimum_self_restore_semantic_match"]
        ),
        "answer_entry_confusion_repair": (
            entry_repair >= gate["minimum_confusion_repair"]
        ),
        "answer_entry_specific_over_wrong_target": (
            entry_repair - wrong_repair
            >= gate["minimum_specificity_over_wrong_target"]
        ),
        "answer_entry_correct_preservation": (
            preservation("matched_correct_answer_entry")
            >= gate["minimum_correct_preservation"]
        ),
        "query_or_fact_upstream_repair": (
            upstream_repair
            >= gate["minimum_query_or_fact_repair_for_upstream_claim"]
        ),
    }
    relation_gate_pass = all(frozen_checks.values())
    random_control_specificity = entry_repair - random_repair
    robust_entry_specificity = entry_repair - max(wrong_repair, random_repair)

    analysis = {
        "schema_version": "phase571_relation_donor_analysis.v1",
        "phase_id": "Phase571",
        "created_at": now(),
        "status": "complete",
        "model": MODEL,
        "candidate_pair_count": execution["candidate_pair_count"],
        "baseline_valid_pair_count": execution["baseline_valid_pair_count"],
        "baseline_drift_pair_count": execution["baseline_drift_pair_count"],
        "final_pair_count": execution["final_pair_count"],
        "condition_reports_by_phenotype": reports,
        "paired_effects_by_phenotype": shifts,
        "derived_effects": {
            "matched_answer_entry_confusion_repair": entry_repair,
            "matched_answer_exit_confusion_repair": exit_repair,
            "matched_query_entry_confusion_repair": query_repair,
            "matched_target_fact_entry_confusion_repair": fact_repair,
            "wrong_target_answer_entry_confusion_repair": wrong_repair,
            "random_answer_entry_confusion_repair": random_repair,
            "largest_upstream_role_repair": upstream_repair,
            "matched_answer_entry_correct_preservation": preservation(
                "matched_correct_answer_entry"
            ),
            "matched_answer_exit_correct_preservation": preservation(
                "matched_correct_answer_exit"
            ),
            "matched_query_entry_correct_preservation": preservation(
                "matched_correct_query_entry"
            ),
            "matched_target_fact_entry_correct_preservation": preservation(
                "matched_correct_target_fact_entry"
            ),
            "self_restore_semantic_match": self_restore_semantic_match,
            "answer_entry_specificity_over_wrong_target": entry_repair - wrong_repair,
            "answer_entry_specificity_over_random": random_control_specificity,
            "answer_entry_specificity_over_strongest_control": robust_entry_specificity,
        },
        "frozen_donor_gate": gate,
        "frozen_donor_checks": frozen_checks,
        "relation_selection_donor_gate_pass": relation_gate_pass,
        "terminal_answer_content_transport_observed": exit_repair >= 0.10,
        "terminal_answer_content_transport_is_not_relation_selection": True,
        "random_control_prevents_specific_entry_mechanism_claim": (
            robust_entry_specificity < gate["minimum_specificity_over_wrong_target"]
        ),
        "relation_selection_mechanism_claimed": False,
        "closure_claimed": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, analysis)

    decision = {
        "schema_version": "phase571_final_stage_decision.v1",
        "phase_id": "Phase571",
        "created_at": now(),
        "relation_donor_analysis_sha256": sha256_file(ANALYSIS_PATH),
        "coarse_block_causal_gate_passed_models": ["qwen3"],
        "relation_selection_donor_gate_passed_models": (
            ["qwen3"] if relation_gate_pass else []
        ),
        "observed_scope": (
            "late_answer_content_transport_only"
            if exit_repair >= 0.10 and not relation_gate_pass
            else "no_donor_mechanism"
        ),
        "advance_to_head_channel_parameter_neuron_scan": False,
        "advance_to_sealed_split": False,
        "phase571_stage_complete": True,
        "stop_reason": (
            "Matched answer-exit state transported target content, but the frozen relation "
            "donor gate failed correct preservation and upstream query/fact repair; random "
            "answer-entry replacement also exceeded matched entry repair."
            if not relation_gate_pass
            else None
        ),
        "next_phase_required": True,
        "next_phase_scope": (
            "Redesign discovery around natural, role-conditioned multi-position trajectories; "
            "do not descend this terminal answer block to heads or neurons."
        ),
        "relation_selection_mechanism_claimed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }
    write_json(DECISION_PATH, decision)
    print(
        json.dumps(
            {
                "derived_effects": analysis["derived_effects"],
                "checks": frozen_checks,
                "relation_gate_pass": relation_gate_pass,
                "decision": decision["observed_scope"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return analysis


if __name__ == "__main__":
    analyze()
