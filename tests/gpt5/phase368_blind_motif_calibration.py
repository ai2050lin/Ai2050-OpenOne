#!/usr/bin/env python3
"""Run frozen Phase367 motifs once on untouched blind calibration groups."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase367_blind_motif_discovery import (
    MIN_BASELINE_IMPROVEMENT,
    MIN_GROUP_WEIGHTED_ACCURACY,
    CandidateStats,
    group_weighted_accuracy,
    iter_occurrences,
    load_cases,
    load_thresholds,
    matched_random_map,
    persistence_accuracy,
    transition_states,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation"
DISCOVERY = PHASE_ROOT / "blind_motif_discovery"
OUT = PHASE_ROOT / "blind_motif_calibration"
CANDIDATES = DISCOVERY / "phase367_frozen_blind_motif_candidates.jsonl"
DISCOVERY_SUMMARY = DISCOVERY / "phase367_blind_motif_discovery_summary.json"
MIN_CALIBRATION_GROUP_SUPPORT = 4


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["anonymous_model_id"], row["generation_time"], row["start_anchor_index"],
        row["path_length"], row["source_role_alias"], row["receiver_role_alias"],
        row["feature"], tuple(row["transition_sequence"]),
    )


def main() -> None:
    discovery_summary = read_json(DISCOVERY_SUMMARY)
    candidate_digest = hashlib.sha256(CANDIDATES.read_bytes()).hexdigest()
    if candidate_digest != discovery_summary["candidate_file"]["sha256"]:
        raise RuntimeError("Frozen discovery candidate digest mismatch")
    candidate_rows = read_jsonl(CANDIDATES)
    candidates_by_key = {candidate_key(row): row for row in candidate_rows}

    thresholds = load_thresholds()
    cases, model_by_anonymous = load_cases("blind_calibration")
    states_by_case = {case.case_id: transition_states(case, thresholds) for case in cases}
    random_map = matched_random_map(cases)
    stats = {key: CandidateStats() for key in candidates_by_key}
    for case in cases:
        for occurrence in iter_occurrences(case, states_by_case, random_map):
            candidate = stats.get(occurrence["key"])
            if candidate is None:
                continue
            group = case.group_id
            candidate.actual[group][occurrence["target"]] += 1
            candidate.shuffled[group][occurrence["shuffled_target"]] += 1
            candidate.random[group][occurrence["random_target"]] += 1
            candidate.persistence_correct[group] += occurrence["target"] == occurrence["last_state"]
            candidate.persistence_total[group] += 1
            candidate.occurrence_count += 1

    result_rows = []
    survivor_rows = []
    for key, frozen in candidates_by_key.items():
        candidate = stats[key]
        support = len(candidate.actual)
        actual_accuracy = group_weighted_accuracy(
            candidate.actual, frozen["frozen_next_state_prediction"]
        )
        common_accuracy = group_weighted_accuracy(
            candidate.actual, frozen["frozen_common_transition_prediction"]
        )
        persistence = persistence_accuracy(candidate.persistence_correct, candidate.persistence_total)
        shuffled_accuracy = group_weighted_accuracy(
            candidate.shuffled, frozen["frozen_shuffled_control_prediction"]
        )
        random_accuracy = group_weighted_accuracy(
            candidate.random, frozen["frozen_random_control_prediction"]
        )
        strongest = max(common_accuracy, persistence, shuffled_accuracy, random_accuracy)
        improvement = actual_accuracy - strongest
        passed = (
            support >= MIN_CALIBRATION_GROUP_SUPPORT
            and actual_accuracy >= MIN_GROUP_WEIGHTED_ACCURACY
            and improvement >= MIN_BASELINE_IMPROVEMENT
        )
        row = {
            "schema_version": "45.0.0",
            "candidate_id": frozen["candidate_id"],
            "equivalence_signature": frozen["equivalence_signature"],
            "anonymous_model_id": frozen["anonymous_model_id"],
            "generation_time": frozen["generation_time"],
            "start_anchor_index": frozen["start_anchor_index"],
            "path_length": frozen["path_length"],
            "source_role_alias": frozen["source_role_alias"],
            "receiver_role_alias": frozen["receiver_role_alias"],
            "feature": frozen["feature"],
            "transition_sequence": frozen["transition_sequence"],
            "frozen_next_state_prediction": frozen["frozen_next_state_prediction"],
            "calibration_independent_group_support": support,
            "calibration_occurrence_count": candidate.occurrence_count,
            "calibration_group_weighted_next_state_accuracy": actual_accuracy,
            "calibration_common_transition_accuracy": common_accuracy,
            "calibration_persistence_accuracy": persistence,
            "calibration_order_shuffled_control_accuracy": shuffled_accuracy,
            "calibration_matched_random_bundle_control_accuracy": random_accuracy,
            "calibration_strongest_baseline_accuracy": strongest,
            "calibration_improvement_over_strongest_baseline": improvement,
            "calibration_gate_passed": passed,
            "candidate_or_prediction_refit_on_calibration": False,
        }
        result_rows.append(row)
        if passed:
            survivor_rows.append(row)

    signature_models: dict[str, set[str]] = defaultdict(set)
    for row in survivor_rows:
        signature_models[row["equivalence_signature"]].add(row["anonymous_model_id"])
    for row in survivor_rows:
        row["cross_model_calibration_model_count"] = len(signature_models[row["equivalence_signature"]])

    OUT.mkdir(parents=True, exist_ok=True)
    all_path = OUT / "phase368_all_frozen_candidate_results.jsonl"
    survivor_path = OUT / "phase368_blind_calibration_survivors.jsonl"
    with all_path.open("w", encoding="utf-8") as handle:
        for row in sorted(result_rows, key=lambda value: value["candidate_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    with survivor_path.open("w", encoding="utf-8") as handle:
        for row in sorted(survivor_rows, key=lambda value: value["candidate_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    model_counts = Counter(model_by_anonymous[row["anonymous_model_id"]] for row in survivor_rows)
    three_model_signatures = sorted(
        signature for signature, models in signature_models.items() if len(models) == 3
    )
    summary = {
        "schema_version": "45.0.0",
        "phase_id": "Phase368",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "evaluate_frozen_blind_motifs_once_on_untouched_independent_calibration_groups",
        "denominator": {
            "calibration_case_count": len(cases),
            "calibration_independent_group_count": len({case.group_id for case in cases}),
            "frozen_discovery_candidate_count": len(candidate_rows),
            "candidate_digest": candidate_digest,
        },
        "frozen_gates": {
            "minimum_independent_group_support_per_model": MIN_CALIBRATION_GROUP_SUPPORT,
            "minimum_group_weighted_next_state_accuracy": MIN_GROUP_WEIGHTED_ACCURACY,
            "minimum_improvement_over_strongest_baseline": MIN_BASELINE_IMPROVEMENT,
            "candidate_shape_refit": False,
            "threshold_refit": False,
            "prediction_refit": False,
        },
        "results": {
            "calibrated_candidate_count": len(survivor_rows),
            "calibrated_candidate_count_by_model": dict(sorted(model_counts.items())),
            "cross_model_three_of_three_signature_count": len(three_model_signatures),
            "cross_model_three_of_three_signatures": three_model_signatures,
            "semantic_or_target_labels_used": False,
            "condition_average_used": False,
            "physical_confirmation_opened": False,
        },
        "files": {
            "all_candidate_results": str(all_path.relative_to(OUT)),
            "survivors": str(survivor_path.relative_to(OUT)),
            "all_candidate_results_sha256": hashlib.sha256(all_path.read_bytes()).hexdigest(),
            "survivors_sha256": hashlib.sha256(survivor_path.read_bytes()).hexdigest(),
        },
        "authorization": {
            "posthoc_semantic_audit_authorized": len(three_model_signatures) > 0,
            "physical_confirmation_authorized": False,
            "causal_intervention_authorized": False,
        },
        "claim_boundary": {
            "model_specific_calibrated_motifs_may_exist": len(survivor_rows) > 0,
            "cross_model_language_path_candidate_exists": len(three_model_signatures) > 0,
            "language_path_discovered": False,
            "language_mechanism_closed": False,
        },
        "next_decision": (
            "posthoc_label_audit_three_model_signatures_without_opening_physical_confirmation"
            if three_model_signatures else
            "stop_without_label_reveal_and_revise_cross_model_dynamic_equivalence"
        ),
    }
    write_json(OUT / "phase368_blind_motif_calibration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
