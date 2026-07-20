#!/usr/bin/env python3
"""Analyze Phase572 joint-role interventions and freeze the route decision."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase572_relation_joint"
ROWS_PATH = OUT_DIR / "phase572_qwen3_joint_causal_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase572_qwen3_joint_causal_summary.json"
PROTOCOL_PATH = OUT_DIR / "phase572_frozen_protocol.json"
ANALYSIS_PATH = OUT_DIR / "phase572_joint_causal_analysis.json"
DECISION_PATH = OUT_DIR / "phase572_stage_decision.json"
PHENOTYPES = ("stable_correct", "stable_relation_confusion")


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


def paired_report(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in rows if row["condition"] == "baseline"}
    changed = {row["case_id"]: row for row in rows if row["condition"] == condition}
    if set(baseline) != set(changed):
        raise RuntimeError(f"Phase572 paired denominator drift: {condition}")
    shifts = [
        changed[case_id]["first_step_target_minus_other_margin"]
        - baseline_row["first_step_target_minus_other_margin"]
        for case_id, baseline_row in baseline.items()
    ]
    semantic_matches = sum(
        changed[case_id]["semantic_event"] == baseline_row["semantic_event"]
        for case_id, baseline_row in baseline.items()
    )
    exact_matches = sum(
        changed[case_id]["normalized_generated"] == baseline_row["normalized_generated"]
        for case_id, baseline_row in baseline.items()
    )
    return {
        "condition": condition,
        "n": len(shifts),
        "mean_margin_shift": sum(shifts) / len(shifts),
        "positive_shift_rate": rate(sum(value > 0.0 for value in shifts), len(shifts)),
        "negative_shift_rate": rate(sum(value < 0.0 for value in shifts), len(shifts)),
        "semantic_event_match_baseline_rate": rate(semantic_matches, len(shifts)),
        "exact_output_match_baseline_rate": rate(exact_matches, len(shifts)),
    }


def analyze() -> dict[str, Any]:
    frozen = read_json(PROTOCOL_PATH)
    summary = read_json(SUMMARY_PATH)
    rows = list(iter_jsonl(ROWS_PATH))
    if summary["rows_sha256"] != sha256_file(ROWS_PATH):
        raise RuntimeError("Phase572 causal row hash drift")
    expected = frozen["final_pair_count"] * 2 * len(frozen["conditions"])
    if len(rows) != expected or any(row["sealed"] for row in rows):
        raise RuntimeError("Phase572 causal denominator/seal drift")
    reports = {}
    paired = {}
    for phenotype in PHENOTYPES:
        phenotype_rows = [row for row in rows if row["receiver_phenotype"] == phenotype]
        reports[phenotype] = {
            condition: condition_report(phenotype_rows, condition)
            for condition in frozen["conditions"]
        }
        paired[phenotype] = {
            condition: paired_report(phenotype_rows, condition)
            for condition in frozen["conditions"]
            if condition != "baseline"
        }
    correct = reports["stable_correct"]
    confusion = reports["stable_relation_confusion"]

    def repair(condition: str) -> float:
        return confusion[condition]["semantic_accuracy"] - confusion["baseline"]["semantic_accuracy"]

    repairs = {
        "answer": repair("matched_answer_entry"),
        "query": repair("matched_query_entry"),
        "fact": repair("matched_fact_entry"),
        "query_answer": repair("matched_query_answer_entry"),
        "fact_answer": repair("matched_fact_answer_entry"),
        "query_fact": repair("matched_query_fact_entry"),
        "query_fact_answer": repair("matched_query_fact_answer_entry"),
        "wrong_target_query_fact_answer": repair("wrong_target_query_fact_answer_entry"),
        "random_query_fact_answer": repair("random_query_fact_answer_entry"),
    }
    preservations = {
        key: correct[condition]["semantic_accuracy"]
        for key, condition in {
            "answer": "matched_answer_entry",
            "query": "matched_query_entry",
            "fact": "matched_fact_entry",
            "query_answer": "matched_query_answer_entry",
            "fact_answer": "matched_fact_answer_entry",
            "query_fact": "matched_query_fact_entry",
            "query_fact_answer": "matched_query_fact_answer_entry",
            "wrong_target_query_fact_answer": "wrong_target_query_fact_answer_entry",
            "random_query_fact_answer": "random_query_fact_answer_entry",
        }.items()
    }
    joint = repairs["query_fact_answer"]
    best_single = max(repairs["answer"], repairs["query"], repairs["fact"])
    best_pair = max(
        repairs["query_answer"], repairs["fact_answer"], repairs["query_fact"]
    )
    leave_one_out = {
        "query_contribution": joint - repairs["fact_answer"],
        "fact_contribution": joint - repairs["query_answer"],
        "answer_contribution": joint - repairs["query_fact"],
    }
    gate = frozen["joint_gate"]
    positive_role_count = sum(
        value >= gate["minimum_leave_one_out_contribution"]
        for value in leave_one_out.values()
    )
    self_matches = [
        paired[phenotype]["self_qfa_entry_restore"]["semantic_event_match_baseline_rate"]
        for phenotype in PHENOTYPES
    ]
    self_restore_match = sum(self_matches) / len(self_matches)
    checks = {
        "minimum_paired_cases": all(reports[p]["baseline"]["n"] >= 128 for p in PHENOTYPES),
        "self_restore_semantic_match": self_restore_match >= gate["minimum_self_restore_semantic_match"],
        "joint_confusion_repair": joint >= gate["minimum_joint_confusion_repair"],
        "joint_correct_preservation": preservations["query_fact_answer"] >= gate["minimum_joint_correct_preservation"],
        "joint_specific_over_wrong_target": (
            joint - repairs["wrong_target_query_fact_answer"]
            >= gate["minimum_joint_specificity_over_each_control"]
        ),
        "joint_specific_over_random": (
            joint - repairs["random_query_fact_answer"]
            >= gate["minimum_joint_specificity_over_each_control"]
        ),
        "joint_gain_over_best_single": joint - best_single >= gate["minimum_joint_gain_over_best_single_role"],
        "joint_gain_over_best_pair": joint - best_pair >= gate["minimum_joint_gain_over_best_two_role_subset"],
        "at_least_two_roles_have_leave_one_out_contribution": (
            positive_role_count
            >= gate["minimum_two_roles_with_positive_leave_one_out_contribution"]
        ),
    }
    joint_pass = all(checks.values())
    only_answer_containing = (
        repairs["answer"] >= gate["minimum_joint_confusion_repair"]
        and repairs["query"] < gate["minimum_joint_confusion_repair"]
        and repairs["fact"] < gate["minimum_joint_confusion_repair"]
        and repairs["query_fact"] < gate["minimum_joint_confusion_repair"]
        and leave_one_out["answer_contribution"]
        >= gate["minimum_leave_one_out_contribution"]
        and leave_one_out["query_contribution"]
        < gate["minimum_leave_one_out_contribution"]
        and leave_one_out["fact_contribution"]
        < gate["minimum_leave_one_out_contribution"]
    )
    protocol_phase = frozen["phase_id"]
    analysis = {
        "schema_version": "phase572_joint_causal_analysis.v1",
        "phase_id": protocol_phase,
        "created_at": now(),
        "status": "complete",
        "model": frozen["model"],
        "candidate_pair_count": summary["candidate_pair_count"],
        "baseline_valid_pair_count": summary["baseline_valid_pair_count"],
        "baseline_drift_pair_count": summary["baseline_drift_pair_count"],
        "final_pair_count": summary["final_pair_count"],
        "condition_reports_by_phenotype": reports,
        "paired_reports_by_phenotype": paired,
        "confusion_repair_by_role_set": repairs,
        "correct_preservation_by_role_set": preservations,
        "leave_one_out_contributions": leave_one_out,
        "best_single_repair": best_single,
        "best_two_role_repair": best_pair,
        "joint_gain_over_best_single": joint - best_single,
        "joint_gain_over_best_two_role_subset": joint - best_pair,
        "self_restore_semantic_match": self_restore_match,
        "joint_gate_checks": checks,
        "joint_relation_state_gate_pass": joint_pass,
        "only_answer_containing_sets_show_material_repair": only_answer_containing,
        "observed_scope": (
            "candidate_distributed_relation_state"
            if joint_pass
            else "late_answer_content_transport_not_joint_relation_state"
            if only_answer_containing
            else "late_joint_role_route_closed"
        ),
        "relation_selection_mechanism_claimed": False,
        "closure_claimed": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, analysis)
    decision = {
        "schema_version": "phase572_stage_decision.v1",
        "phase_id": protocol_phase,
        "created_at": now(),
        "analysis_sha256": sha256_file(ANALYSIS_PATH),
        "joint_relation_state_gate_passed": joint_pass,
        "observed_scope": analysis["observed_scope"],
        "late_block_head_channel_parameter_neuron_scan_allowed": False,
        "late_static_joint_role_route_closed": not joint_pass,
        "next_phase_required": True,
        "next_phase_scope": (
            "Move upstream and discover natural relation-conditioned transition events "
            "before answer content consolidation; use fresh worlds and coordinate-free "
            "role-to-role effects rather than additional late-state patch combinations."
        ),
        "relation_selection_mechanism_claimed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }
    write_json(DECISION_PATH, decision)
    print(
        json.dumps(
            {
                "repairs": repairs,
                "preservations": preservations,
                "leave_one_out": leave_one_out,
                "checks": checks,
                "joint_gate_pass": joint_pass,
                "observed_scope": analysis["observed_scope"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return analysis


if __name__ == "__main__":
    analyze()
