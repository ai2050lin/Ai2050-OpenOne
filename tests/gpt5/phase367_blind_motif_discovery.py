#!/usr/bin/env python3
"""Discover future-predictive directed-flow motifs on blind discovery groups only."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation"
DESCRIPTORS = PHASE_ROOT / "label_blind_flow_descriptors"
THRESHOLDS = PHASE_ROOT / "blind_threshold_custodian" / "phase366_frozen_descriptor_floors.jsonl"
OUT = PHASE_ROOT / "blind_motif_discovery"
MODELS = ("qwen3", "glm4", "deepseek7b")
PATH_LENGTHS = (2, 3, 4, 6, 8)
ANCHOR_COUNT = 10
MIN_DISCOVERY_GROUP_SUPPORT = 8
MIN_GROUP_WEIGHTED_ACCURACY = 0.75
MIN_BASELINE_IMPROVEMENT = 0.10


@dataclass
class CaseSeries:
    case_id: str
    anonymous_model_id: str
    group_id: str
    condition_slot: str
    split: str
    values: dict[tuple[Any, ...], dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
    layer_count: int = 0


@dataclass
class CandidateStats:
    actual: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    shuffled: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    random: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    persistence_correct: Counter[str] = field(default_factory=Counter)
    persistence_total: Counter[str] = field(default_factory=Counter)
    occurrence_count: int = 0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mode(counter: Counter[int]) -> int:
    if not counter:
        raise ValueError("Cannot select mode from empty counter")
    return min(value for value, count in counter.items() if count == max(counter.values()))


def group_mode(counter_by_group: dict[str, Counter[int]]) -> int:
    votes = Counter(mode(counter) for counter in counter_by_group.values() if counter)
    return mode(votes)


def group_weighted_accuracy(counter_by_group: dict[str, Counter[int]], prediction: int) -> float:
    if not counter_by_group:
        return 0.0
    accuracies = [counter[prediction] / sum(counter.values()) for counter in counter_by_group.values()]
    return sum(accuracies) / len(accuracies)


def persistence_accuracy(correct: Counter[str], total: Counter[str]) -> float:
    groups = [group for group, count in total.items() if count]
    if not groups:
        return 0.0
    return sum(correct[group] / total[group] for group in groups) / len(groups)


def depth_bin(layer_index: int, layer_count: int) -> str:
    fraction = (layer_index + 0.5) / layer_count
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def load_thresholds() -> dict[tuple[Any, ...], float]:
    return {
        (
            row["anonymous_model_id"], row["generation_time"], row["relative_depth_bin"],
            row["source_role_alias"], row["receiver_role_alias"], row["feature"],
        ): float(row["frozen_effect_floor"])
        for row in read_jsonl(THRESHOLDS)
    }


def load_cases(split: str) -> tuple[list[CaseSeries], dict[str, str]]:
    cases: list[CaseSeries] = []
    model_by_anonymous: dict[str, str] = {}
    for model in MODELS:
        by_case: dict[str, CaseSeries] = {}
        path = DESCRIPTORS / "private" / f"{model}_directed_flow_descriptors.jsonl"
        for row in read_jsonl(path):
            if row["split"] != split:
                continue
            case = by_case.get(row["anonymous_case_id"])
            if case is None:
                case = CaseSeries(
                    case_id=row["anonymous_case_id"],
                    anonymous_model_id=row["anonymous_model_id"],
                    group_id=row["anonymous_group_id"],
                    condition_slot=row["anonymous_condition_slot"],
                    split=row["split"],
                )
                by_case[case.case_id] = case
                model_by_anonymous[case.anonymous_model_id] = model
            series_prefix = (
                row["generation_time"], row["source_role_alias"], row["receiver_role_alias"],
            )
            for feature, value in row["features"].items():
                case.values[series_prefix + (feature,)][int(row["layer_index"])] = float(value)
            case.layer_count = max(case.layer_count, int(row["layer_index"]) + 1)
        cases.extend(by_case.values())
    return sorted(cases, key=lambda case: (case.anonymous_model_id, case.case_id)), model_by_anonymous


def anchor_layers(layer_count: int) -> tuple[int, ...]:
    anchors = tuple(round(index * (layer_count - 1) / (ANCHOR_COUNT - 1)) for index in range(ANCHOR_COUNT))
    if len(set(anchors)) != ANCHOR_COUNT:
        raise RuntimeError(f"Layer count {layer_count} cannot support {ANCHOR_COUNT} unique anchors")
    return anchors


def transition_states(
    case: CaseSeries,
    thresholds: dict[tuple[Any, ...], float],
) -> dict[tuple[Any, ...], tuple[int, ...]]:
    anchors = anchor_layers(case.layer_count)
    result = {}
    for series_key, layer_values in case.values.items():
        generation_time, source, receiver, feature = series_key
        states = []
        for start, end in zip(anchors[:-1], anchors[1:], strict=True):
            threshold_key = (
                case.anonymous_model_id, generation_time, depth_bin(start, case.layer_count),
                source, receiver, feature,
            )
            floor = thresholds[threshold_key]
            delta = layer_values[end] - layer_values[start]
            states.append(1 if delta > floor else -1 if delta < -floor else 0)
        result[series_key] = tuple(states)
    return result


def matched_random_map(cases: list[CaseSeries]) -> dict[str, str]:
    result = {}
    by_model: dict[str, list[CaseSeries]] = defaultdict(list)
    for case in cases:
        by_model[case.anonymous_model_id].append(case)
    for model_cases in by_model.values():
        ordered = sorted(model_cases, key=lambda case: case.case_id)
        for index, case in enumerate(ordered):
            offset = 1
            while ordered[(index + offset) % len(ordered)].group_id == case.group_id:
                offset += 1
            result[case.case_id] = ordered[(index + offset) % len(ordered)].case_id
    return result


def iter_occurrences(
    case: CaseSeries,
    states_by_case: dict[str, dict[tuple[Any, ...], tuple[int, ...]]],
    random_map: dict[str, str],
) -> Iterator[dict[str, Any]]:
    case_states = states_by_case[case.case_id]
    random_states = states_by_case[random_map[case.case_id]]
    for series_key, states in case_states.items():
        generation_time, source, receiver, feature = series_key
        shuffled_key = ((generation_time + 1) % 3, source, receiver, feature)
        shuffled_states = case_states.get(shuffled_key)
        for length in PATH_LENGTHS:
            for start in range(len(states) - length):
                sequence = states[start:start + length]
                target_index = start + length
                target = states[target_index]
                context = (
                    case.anonymous_model_id, generation_time, target_index,
                    source, receiver, feature,
                )
                key = context[:-5] + (
                    generation_time, start, length, source, receiver, feature, sequence,
                )
                shuffled_target = (
                    shuffled_states[target_index]
                    if shuffled_states is not None
                    else states[len(states) - 1 - target_index]
                )
                random_series = random_states.get(series_key)
                if random_series is None:
                    raise RuntimeError(f"Matched random bundle lacks series {series_key}")
                yield {
                    "key": key,
                    "context": context,
                    "target": target,
                    "last_state": sequence[-1],
                    "shuffled_target": shuffled_target,
                    "random_target": random_series[target_index],
                    "all_zero": not any(sequence),
                }


def motif_fields(key: tuple[Any, ...]) -> dict[str, Any]:
    anonymous_model_id, generation_time, start, length, source, receiver, feature, sequence = key
    return {
        "anonymous_model_id": anonymous_model_id,
        "generation_time": generation_time,
        "start_anchor_index": start,
        "path_length": length,
        "source_role_alias": source,
        "receiver_role_alias": receiver,
        "feature": feature,
        "transition_sequence": list(sequence),
    }


def equivalence_signature(fields: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in fields.items() if key != "anonymous_model_id"
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def main() -> None:
    thresholds = load_thresholds()
    cases, model_by_anonymous = load_cases("blind_discovery")
    states_by_case = {case.case_id: transition_states(case, thresholds) for case in cases}
    random_map = matched_random_map(cases)
    group_indices = {group: index for index, group in enumerate(sorted({case.group_id for case in cases}))}

    candidate_group_masks: dict[tuple[Any, ...], int] = {}
    context_targets: dict[tuple[Any, ...], dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    occurrence_count = zero_window_count = 0
    for case in cases:
        group_bit = 1 << group_indices[case.group_id]
        for occurrence in iter_occurrences(case, states_by_case, random_map):
            occurrence_count += 1
            context_targets[occurrence["context"]][case.group_id][occurrence["target"]] += 1
            if occurrence["all_zero"]:
                zero_window_count += 1
                continue
            key = occurrence["key"]
            candidate_group_masks[key] = candidate_group_masks.get(key, 0) | group_bit

    recurrent_keys = {
        key for key, mask in candidate_group_masks.items()
        if mask.bit_count() >= MIN_DISCOVERY_GROUP_SUPPORT
    }
    context_predictions = {
        key: group_mode(value) for key, value in context_targets.items()
    }
    stats: dict[tuple[Any, ...], CandidateStats] = {
        key: CandidateStats() for key in recurrent_keys
    }
    for case in cases:
        for occurrence in iter_occurrences(case, states_by_case, random_map):
            key = occurrence["key"]
            candidate = stats.get(key)
            if candidate is None:
                continue
            group = case.group_id
            candidate.actual[group][occurrence["target"]] += 1
            candidate.shuffled[group][occurrence["shuffled_target"]] += 1
            candidate.random[group][occurrence["random_target"]] += 1
            candidate.persistence_correct[group] += occurrence["target"] == occurrence["last_state"]
            candidate.persistence_total[group] += 1
            candidate.occurrence_count += 1

    evaluated_rows = []
    frozen_rows = []
    for key, candidate in stats.items():
        fields = motif_fields(key)
        actual_prediction = group_mode(candidate.actual)
        shuffled_prediction = group_mode(candidate.shuffled)
        random_prediction = group_mode(candidate.random)
        actual_accuracy = group_weighted_accuracy(candidate.actual, actual_prediction)
        context_prediction = context_predictions[(
            fields["anonymous_model_id"], fields["generation_time"],
            fields["start_anchor_index"] + fields["path_length"],
            fields["source_role_alias"], fields["receiver_role_alias"], fields["feature"],
        )]
        common_accuracy = group_weighted_accuracy(candidate.actual, context_prediction)
        persistent_accuracy = persistence_accuracy(candidate.persistence_correct, candidate.persistence_total)
        shuffled_accuracy = group_weighted_accuracy(candidate.shuffled, shuffled_prediction)
        random_accuracy = group_weighted_accuracy(candidate.random, random_prediction)
        strongest_baseline = max(common_accuracy, persistent_accuracy, shuffled_accuracy, random_accuracy)
        improvement = actual_accuracy - strongest_baseline
        support = len(candidate.actual)
        passed = (
            support >= MIN_DISCOVERY_GROUP_SUPPORT
            and actual_accuracy >= MIN_GROUP_WEIGHTED_ACCURACY
            and improvement >= MIN_BASELINE_IMPROVEMENT
        )
        row = {
            "schema_version": "44.0.0",
            **fields,
            "candidate_id": "motif_" + hashlib.sha256(repr(key).encode()).hexdigest()[:24],
            "equivalence_signature": equivalence_signature(fields),
            "independent_group_support": support,
            "occurrence_count": candidate.occurrence_count,
            "active_transition_count": sum(value != 0 for value in fields["transition_sequence"]),
            "frozen_next_state_prediction": actual_prediction,
            "frozen_common_transition_prediction": context_prediction,
            "frozen_shuffled_control_prediction": shuffled_prediction,
            "frozen_random_control_prediction": random_prediction,
            "group_weighted_next_state_accuracy": actual_accuracy,
            "common_transition_accuracy": common_accuracy,
            "persistence_accuracy": persistent_accuracy,
            "order_shuffled_control_accuracy": shuffled_accuracy,
            "matched_random_bundle_control_accuracy": random_accuracy,
            "strongest_baseline_accuracy": strongest_baseline,
            "improvement_over_strongest_baseline": improvement,
            "discovery_gate_passed": passed,
        }
        evaluated_rows.append(row)
        if passed:
            frozen_rows.append(row)

    signature_models: dict[str, set[str]] = defaultdict(set)
    for row in frozen_rows:
        signature_models[row["equivalence_signature"]].add(row["anonymous_model_id"])
    for row in frozen_rows:
        row["cross_model_discovery_model_count"] = len(signature_models[row["equivalence_signature"]])

    frozen_rows.sort(key=lambda row: (row["candidate_id"], row["anonymous_model_id"]))
    candidate_path = OUT / "phase367_frozen_blind_motif_candidates.jsonl"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("w", encoding="utf-8") as handle:
        for row in frozen_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    candidate_digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    model_counts = Counter(model_by_anonymous[row["anonymous_model_id"]] for row in frozen_rows)
    length_counts = Counter(row["path_length"] for row in frozen_rows)
    summary = {
        "schema_version": "44.0.0",
        "phase_id": "Phase367",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_label_blind_future_predictive_directed_flow_motifs_on_discovery_groups_only",
        "denominator": {
            "model_count": len(MODELS),
            "discovery_case_count": len(cases),
            "discovery_independent_group_count": len(group_indices),
            "normalized_depth_anchor_count": ANCHOR_COUNT,
            "path_lengths": list(PATH_LENGTHS),
            "enumerated_window_occurrence_count": occurrence_count,
            "all_zero_window_count": zero_window_count,
            "unique_nonzero_motif_count": len(candidate_group_masks),
            "minimum_group_support_recurrent_motif_count": len(recurrent_keys),
        },
        "frozen_gates": {
            "minimum_independent_group_support_per_model": MIN_DISCOVERY_GROUP_SUPPORT,
            "minimum_group_weighted_next_state_accuracy": MIN_GROUP_WEIGHTED_ACCURACY,
            "minimum_improvement_over_strongest_baseline": MIN_BASELINE_IMPROVEMENT,
            "independent_group_is_analysis_unit": True,
            "all_zero_paths_are_candidates": False,
        },
        "controls": {
            "common_transition": True,
            "current_state_persistence": True,
            "generation_time_cyclic_shuffle_with_depth_reversal_fallback": True,
            "matched_size_different_group_random_bundle": True,
        },
        "results": {
            "evaluated_recurrent_motif_count": len(evaluated_rows),
            "frozen_discovery_candidate_count": len(frozen_rows),
            "candidate_count_by_model": dict(sorted(model_counts.items())),
            "candidate_count_by_path_length": {str(key): value for key, value in sorted(length_counts.items())},
            "cross_model_three_of_three_signature_count": sum(len(models) == 3 for models in signature_models.values()),
            "calibration_rows_used": False,
            "semantic_or_target_labels_used": False,
            "condition_average_used": False,
            "top_k_selection_used": False,
        },
        "candidate_file": {
            "relative_path": str(candidate_path.relative_to(OUT)),
            "sha256": candidate_digest,
        },
        "authorization": {
            "blind_calibration_authorized": len(frozen_rows) > 0,
            "semantic_label_reveal_authorized": False,
            "physical_confirmation_authorized": False,
            "causal_intervention_authorized": False,
        },
        "claim_boundary": {
            "candidates_are_frozen_discovery_motifs": len(frozen_rows) > 0,
            "candidates_are_language_paths": False,
            "independent_calibration_executed": False,
            "language_mechanism_closed": False,
        },
        "next_decision": (
            "run_frozen_candidates_once_on_blind_calibration_groups"
            if frozen_rows else "stop_without_label_reveal_and_revise_dynamic_object"
        ),
    }
    write_json(OUT / "phase367_blind_motif_discovery_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
