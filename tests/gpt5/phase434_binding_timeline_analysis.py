#!/usr/bin/env python3
"""Analyze Phase434 behavior and label-blind binding geometry."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase434_binding_timeline_collect import POSITION_ROLES  # noqa: E402
from phase434_binding_timeline_protocol import (  # noqa: E402
    BEHAVIOR_SPLITS,
    LANGUAGE_MODEL,
    MODELS,
    OUT,
    PHYSICAL_SPLIT,
    PHASE_ID as PROTOCOL_PHASE_ID,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    STRESS_SPLIT,
    TIMINGS,
    freeze,
    read_json,
    read_jsonl,
    write_json,
)


PHASE_ID = "Phase434-BindingTimelineAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase434_binding_timeline"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
LATE_TIMINGS = ("after_records", "near_query")
PREGEN_POSITIONS = tuple(
    role for role in POSITION_ROLES if role != "teacher_branch_boundary"
)
POSITION_COLORS = {
    "selector_slot_end": "#ef4444",
    "role_a_result_end": "#f59e0b",
    "role_b_result_end": "#eab308",
    "after_records_end": "#22c55e",
    "question_end": "#06b6d4",
    "instruction_end": "#3b82f6",
    "assistant_boundary": "#8b5cf6",
    "prompt_terminal": "#ec4899",
    "teacher_branch_boundary": "#64748b",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase434 non-finite scalar: {value}")
    return round(float(value), 9)


def wilson(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "estimate": 0.0, "lcb": 0.0, "ucb": 1.0}
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        estimate * (1 - estimate) / total + z * z / (4 * total * total)
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": clean(estimate),
        "lcb": clean(max(0.0, center - radius)),
        "ucb": clean(min(1.0, center + radius)),
    }


def condition_good(row: dict[str, Any]) -> bool:
    return bool(
        row["teacher_sequence_correct"]
        and row["natural_target_first"]
        and not row["natural_opposite_first"]
        and row["natural_interface_valid"]
        and row["natural_exact_target_contract"]
        and not row["natural_revision"]
        and row["natural_boundary"]
        and row["natural_stop"]
        and not row["natural_censoring"]
        and row["natural_common_prefix_exact"]
        and row["natural_reaches_branch_boundary"]
        and row["natural_branch_correct"]
        and row["natural_complete_event_correct"]
    )


def cell_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return row["timing"], row["record_order"], row["mapping"], row["role"]


def behavior_cells(rows: list[dict[str, Any]], split: str, candidate: bool) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["split"] == split and bool(row["candidate"]) == candidate
    ]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[cell_key(row)].append(row)
    return [
        {
            "timing": key[0],
            "record_order": key[1],
            "mapping": key[2],
            "role": key[3],
            "condition_count": len(values),
            "complete_event": wilson(sum(condition_good(row) for row in values), len(values)),
            "branch_correct": wilson(sum(row["natural_branch_correct"] for row in values), len(values)),
            "teacher_correct": wilson(sum(row["teacher_sequence_correct"] for row in values), len(values)),
            "choice_counts": dict(Counter(row["actual_choice"] for row in values)),
        }
        for key, values in sorted(groups.items())
    ]


def grouped_contract(
    rows: list[dict[str, Any]], split: str, candidate: bool
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == split and bool(row["candidate"]) == candidate
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[row["semantic_group_id"]].append(row)
    outcomes = []
    expected_total = 40 if candidate else 10
    for values in grouped.values():
        if candidate:
            required = [row for row in values if row["timing"] in LATE_TIMINGS]
            outcomes.append(
                len(values) == expected_total
                and len(required) == 16
                and all(condition_good(row) for row in required)
            )
        else:
            outcomes.append(
                len(values) == expected_total and all(condition_good(row) for row in values)
            )
    return {
        "group_count": len(grouped),
        "required_condition_scope": "late_timings" if candidate else "all_timings",
        "group_contract": wilson(sum(outcomes), len(outcomes)),
        "condition_complete_event": wilson(
            sum(condition_good(row) for row in selected), len(selected)
        ),
        "natural_common_prefix": wilson(
            sum(row["natural_common_prefix_exact"] for row in selected), len(selected)
        ),
        "actual_choice_counts": dict(Counter(row["actual_choice"] for row in selected)),
    }


def timing_reproducibility(
    discovery_cells: list[dict[str, Any]], holdout_cells: list[dict[str, Any]]
) -> dict[str, Any]:
    def indexed(values: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], float]:
        return {
            (row["timing"], row["record_order"], row["mapping"], row["role"]): float(row["complete_event"]["estimate"])
            for row in values
        }

    discovery = indexed(discovery_cells)
    holdout = indexed(holdout_cells)
    details = []
    agreements = []
    for order in ("ab", "ba"):
        for mapping in ("direct", "swapped"):
            for role in ("a", "b"):
                late_d = statistics.mean(discovery[(timing, order, mapping, role)] for timing in LATE_TIMINGS)
                late_h = statistics.mean(holdout[(timing, order, mapping, role)] for timing in LATE_TIMINGS)
                for timing in TIMINGS:
                    gap_d = late_d - discovery[(timing, order, mapping, role)]
                    gap_h = late_h - holdout[(timing, order, mapping, role)]
                    if abs(gap_d) >= 0.05:
                        agreements.append((gap_d > 0) == (gap_h > 0))
                    details.append(
                        {
                            "timing": timing,
                            "record_order": order,
                            "mapping": mapping,
                            "role": role,
                            "discovery_late_minus_timing": clean(gap_d),
                            "holdout_late_minus_timing": clean(gap_h),
                        }
                    )
    return {
        "direction_tested_count": len(agreements),
        "direction_agreement": wilson(sum(agreements), len(agreements)),
        "details": details,
    }


def alias_failure_audit(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == split and row["candidate"] and not condition_good(row)
    ]
    return {
        "failure_count": len(selected),
        "by_timing": dict(Counter(row["timing"] for row in selected)),
        "by_record_order": dict(Counter(row["record_order"] for row in selected)),
        "by_mapping": dict(Counter(row["mapping"] for row in selected)),
        "by_role": dict(Counter(row["role"] for row in selected)),
        "by_role_alias": dict(Counter(str(row["role_alias_index"]) for row in selected)),
        "by_cue_alias": dict(Counter(str(row["cue_alias_index"]) for row in selected)),
        "three_factor": {
            "::".join(key): value
            for key, value in sorted(
                Counter((row["timing"], row["mapping"], row["role"]) for row in selected).items()
            )
        },
    }


def first_record_source(row: dict[str, Any]) -> str:
    first_role = "a" if row["record_order"] == "ab" else "b"
    if row["mapping"] == "direct":
        return "source_1" if first_role == "a" else "source_2"
    return "source_2" if first_role == "a" else "source_1"


def record_position_audit(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["split"] == split and row["candidate"]
    ]
    by_timing = {}
    for timing in TIMINGS:
        timing_rows = [row for row in selected if row["timing"] == timing]
        position_payload = {}
        for position in ("first", "second"):
            subset = [
                row
                for row in timing_rows
                if (("a" if row["record_order"] == "ab" else "b") == row["role"])
                == (position == "first")
            ]
            position_payload[position] = {
                "complete_event": wilson(
                    sum(row["natural_complete_event_correct"] for row in subset),
                    len(subset),
                ),
                "registered_source_choice_correct": wilson(
                    sum(row["actual_choice"] == row["semantic_target_source"] for row in subset),
                    len(subset),
                ),
            }
        by_timing[timing] = position_payload
    first_target = [
        row
        for row in selected
        if ("a" if row["record_order"] == "ab" else "b") == row["role"]
    ]
    second_target = [
        row
        for row in selected
        if ("a" if row["record_order"] == "ab" else "b") != row["role"]
    ]
    failures = [row for row in selected if not condition_good(row)]
    return {
        "posthoc_descriptive_only": True,
        "gate_thresholds_unchanged": True,
        "first_record_selected": wilson(
            sum(row["actual_choice"] == first_record_source(row) for row in selected),
            len(selected),
        ),
        "target_is_first_record": {
            "complete_event": wilson(
                sum(row["natural_complete_event_correct"] for row in first_target),
                len(first_target),
            ),
            "registered_source_choice_correct": wilson(
                sum(row["actual_choice"] == row["semantic_target_source"] for row in first_target),
                len(first_target),
            ),
        },
        "target_is_second_record": {
            "complete_event": wilson(
                sum(row["natural_complete_event_correct"] for row in second_target),
                len(second_target),
            ),
            "registered_source_choice_correct": wilson(
                sum(row["actual_choice"] == row["semantic_target_source"] for row in second_target),
                len(second_target),
            ),
        },
        "failed_conditions_select_first_record": wilson(
            sum(row["actual_choice"] == first_record_source(row) for row in failures),
            len(failures),
        ),
        "by_timing": by_timing,
    }


def analyze_behavior_model(model: str) -> dict[str, Any]:
    root = OUT / "behavior" / model / "behavior"
    rows = read_jsonl(root / "phase434_behavior_rows.jsonl")
    complete = read_json(root / "phase434_behavior_complete.json")
    protocol = read_json(OUT / "phase434_protocol.json")
    token_fraction = complete["token_contract_valid_count"] / max(1, complete["condition_count"])
    splits = {}
    cells = {}
    for split in BEHAVIOR_SPLITS:
        candidate = grouped_contract(rows, split, True)
        control = grouped_contract(rows, split, False)
        split_pass = bool(
            candidate["group_contract"]["lcb"]
            >= protocol["numeric_gates"]["late_behavior_group_lcb_min"]
            and control["group_contract"]["lcb"]
            >= protocol["numeric_gates"]["control_all_timing_group_lcb_min"]
        )
        splits[split] = {
            "candidate": candidate,
            "control": control,
            "behavior_qualification_pass": split_pass,
        }
        cells[split] = {
            "candidate": behavior_cells(rows, split, True),
            "control": behavior_cells(rows, split, False),
        }
    stress = [row for row in rows if row["split"] == STRESS_SPLIT]
    token_pass = token_fraction >= protocol["numeric_gates"]["token_contract_valid_fraction_min"]
    eligible = token_pass and all(
        splits[split]["behavior_qualification_pass"] for split in BEHAVIOR_SPLITS
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "condition_count": len(rows),
        "token_contract": {
            "valid": complete["token_contract_valid_count"],
            "total": complete["condition_count"],
            "fraction": clean(token_fraction),
            "common_prefix_length_range": complete["common_prefix_length_range"],
            "pass": token_pass,
        },
        "splits": splits,
        "cells": cells,
        "timing_reproducibility": timing_reproducibility(
            cells["behavior_discovery"]["candidate"],
            cells["behavior_holdout"]["candidate"],
        ),
        "holdout_failure_audit": alias_failure_audit(rows, "behavior_holdout"),
        "record_position_audit": {
            split: record_position_audit(rows, split) for split in BEHAVIOR_SPLITS
        },
        "stress": {
            "condition_count": len(stress),
            "nonblocking": True,
            "choice_counts": dict(Counter(row["actual_choice"] for row in stress)),
            "by_mode": {
                mode: dict(Counter(row["actual_choice"] for row in stress if row["route_mode"] == mode))
                for mode in ("conflict_slots", "neutral_only")
            },
        },
        "behavior_eligible": eligible,
        "physical": False,
        "predictive": False,
        "causal": False,
    }
    write_json(OUT / f"phase434_{model}_behavior_audit.json", output)
    return output


def analyze_behavior_gate() -> dict[str, Any]:
    if not (OUT / "phase434_protocol.json").exists():
        freeze()
    behavior = {model: analyze_behavior_model(model) for model in MODELS}
    eligible = [model for model in MODELS if behavior[model]["behavior_eligible"]]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "behavior_gate",
        "behavior": behavior,
        "eligible_models": eligible,
        "model_specific_physical_allowed": bool(eligible),
        "cross_model_behavior_qualified": len(eligible) >= 2,
        "sealed_rows_read": False,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase434_behavior_gate.json", output)
    return output


def iter_gzip_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def median(values: list[float]) -> float:
    return clean(statistics.median(values)) if values else 0.0


def group_bundles(path: Path) -> Iterator[list[dict[str, Any]]]:
    current_id: str | None = None
    bundle: list[dict[str, Any]] = []
    for row in iter_gzip_rows(path):
        group_id = row["semantic_group_id"]
        if current_id is not None and group_id != current_id:
            yield bundle
            bundle = []
        current_id = group_id
        bundle.append(row)
    if bundle:
        yield bundle


def geometry_ledger(path: Path) -> dict[str, Any]:
    effects: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    control_distances: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    candidate_groups = 0
    control_groups = 0
    layer_count = 0
    for bundle in group_bundles(path):
        first = bundle[0]
        fold = "discovery" if int(first["group_index"]) < 96 else "holdout"
        layer_count = max(layer_count, max(int(row["layer"]) for row in bundle) + 1)
        indexed = {
            (
                row["timing"],
                row["record_order"],
                row["mapping"],
                row["role"],
                int(row["layer"]),
            ): row
            for row in bundle
        }
        if first["candidate"]:
            candidate_groups += 1
            for timing in TIMINGS:
                for order in ("ab", "ba"):
                    for layer in range(layer_count):
                        rows = {
                            (mapping, role): indexed[(timing, order, mapping, role, layer)]
                            for mapping in ("direct", "swapped")
                            for role in ("a", "b")
                        }
                        for position in POSITION_ROLES:
                            sketches = {
                                key: value["position_metrics"][position]["state_sketch"]
                                for key, value in rows.items()
                            }
                            within = statistics.mean(
                                (
                                    cosine_distance(sketches[("direct", "a")], sketches[("swapped", "b")]),
                                    cosine_distance(sketches[("direct", "b")], sketches[("swapped", "a")]),
                                )
                            )
                            between = statistics.mean(
                                (
                                    cosine_distance(sketches[("direct", "a")], sketches[("swapped", "a")]),
                                    cosine_distance(sketches[("direct", "b")], sketches[("swapped", "b")]),
                                )
                            )
                            effect = (between - within) / max(1e-8, abs(between) + abs(within))
                            effects[(fold, layer, position, timing)].append(effect)
        else:
            control_groups += 1
            for timing in TIMINGS:
                for layer in range(layer_count):
                    row_a = next(
                        row for row in bundle if row["timing"] == timing and row["role"] == "a" and int(row["layer"]) == layer
                    )
                    row_b = next(
                        row for row in bundle if row["timing"] == timing and row["role"] == "b" and int(row["layer"]) == layer
                    )
                    for position in POSITION_ROLES:
                        control_distances[(fold, layer, position, timing)].append(
                            cosine_distance(
                                row_a["position_metrics"][position]["state_sketch"],
                                row_b["position_metrics"][position]["state_sketch"],
                            )
                        )
    cells = []
    for key, values in sorted(effects.items()):
        fold, layer, position, timing = key
        cells.append(
            {
                "fold": fold,
                "layer": layer,
                "relative_depth": clean(layer / max(1, layer_count - 1)),
                "position_role": position,
                "timing": timing,
                "independent_group_count": len(values) // 2,
                "paired_order_effect_count": len(values),
                "binding_geometry_effect_median": median(values),
                "binding_geometry_effect_positive": wilson(sum(value > 0 for value in values), len(values)),
                "control_role_distance_median": median(control_distances.get(key, [])),
                "output_label_blind": True,
            }
        )
    return {
        "candidate_group_count": candidate_groups,
        "control_group_count": control_groups,
        "layer_count": layer_count,
        "cells": cells,
    }


def freeze_geometry_window(ledger: dict[str, Any], model: str) -> dict[str, Any]:
    path = OUT / f"phase434_{model}_geometry_window_freeze.json"
    if path.exists():
        return read_json(path)
    discovery = [
        row
        for row in ledger["cells"]
        if row["fold"] == "discovery" and row["position_role"] in PREGEN_POSITIONS
    ]
    ranked = sorted(
        discovery,
        key=lambda row: (
            -float(row["binding_geometry_effect_median"]),
            int(row["layer"]),
            POSITION_ROLES.index(row["position_role"]),
            TIMINGS.index(row["timing"]),
        ),
    )
    selected = ranked[0]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "discovery_groups": 96,
        "selected_layer": selected["layer"],
        "selected_position_role": selected["position_role"],
        "selected_timing": selected["timing"],
        "discovery_binding_geometry_effect": selected["binding_geometry_effect_median"],
        "selection_uses_output_labels": False,
        "unknown_hidden_permutation_assumed": False,
        "holdout_reselection_forbidden": True,
        "sealed_reselection_forbidden": True,
        "top_discovery_windows": ranked[:20],
    }
    write_json(path, output)
    return output


def temporal_replication(
    ledger: dict[str, Any], window: dict[str, Any]
) -> dict[str, Any]:
    selected = [
        row
        for row in ledger["cells"]
        if row["layer"] == window["selected_layer"]
        and row["position_role"] == window["selected_position_role"]
    ]
    by_fold = {
        fold: {
            row["timing"]: float(row["binding_geometry_effect_median"])
            for row in selected
            if row["fold"] == fold
        }
        for fold in ("discovery", "holdout")
    }
    agreements = []
    comparisons = []
    for left_index, left in enumerate(TIMINGS):
        for right in TIMINGS[left_index + 1 :]:
            discovery_delta = by_fold["discovery"][left] - by_fold["discovery"][right]
            holdout_delta = by_fold["holdout"][left] - by_fold["holdout"][right]
            if abs(discovery_delta) >= 0.02:
                agreements.append((discovery_delta > 0) == (holdout_delta > 0))
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "discovery_delta": clean(discovery_delta),
                    "holdout_delta": clean(holdout_delta),
                }
            )
    spread = {
        fold: clean(max(values.values()) - min(values.values()))
        for fold, values in by_fold.items()
    }
    return {
        "effects_by_fold_timing": by_fold,
        "direction_agreement": wilson(sum(agreements), len(agreements)),
        "effect_spread": spread,
        "comparisons": comparisons,
    }


def add_vectors(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def divide_vector(value: list[float], count: int) -> list[float]:
    return [item / max(1, count) for item in value]


def choice_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    per_class = {}
    recalls = []
    for label in ("source_1", "source_2"):
        indices = [index for index, value in enumerate(actual) if value == label]
        successes = sum(predicted[index] == label for index in indices)
        metric = wilson(successes, len(indices))
        per_class[label] = metric
        recalls.append(float(metric["estimate"]))
    return {
        "condition_count": len(actual),
        "accuracy": clean(sum(a == p for a, p in zip(actual, predicted)) / max(1, len(actual))),
        "balanced_accuracy": clean(statistics.mean(recalls)),
        "per_class": per_class,
    }


def fixed_window_prediction(path: Path, window: dict[str, Any]) -> dict[str, Any]:
    layer = int(window["selected_layer"])
    position = window["selected_position_role"]
    timing = window["selected_timing"]
    sums = {"source_1": [0.0] * 16, "source_2": [0.0] * 16}
    counts = Counter()
    holdout_rows: list[tuple[dict[str, Any], list[float]]] = []
    control_rows: list[tuple[dict[str, Any], list[float]]] = []
    for row in iter_gzip_rows(path):
        if int(row["layer"]) != layer or row["timing"] != timing:
            continue
        sketch = row["position_metrics"][position]["state_sketch"]
        if row["candidate"] and int(row["group_index"]) < 96:
            label = row["semantic_target_source"]
            sums[label] = add_vectors(sums[label], sketch)
            counts[label] += 1
        elif row["candidate"]:
            holdout_rows.append((row, sketch))
        else:
            control_rows.append((row, sketch))
    centroids = {label: divide_vector(sums[label], counts[label]) for label in sums}

    def predict(sketch: list[float]) -> str:
        distances = {
            label: cosine_distance(sketch, centroid) for label, centroid in centroids.items()
        }
        return min(distances, key=lambda label: (distances[label], label))

    actual = [row["semantic_target_source"] for row, _ in holdout_rows]
    predicted = [predict(sketch) for _, sketch in holdout_rows]
    candidate_metrics = choice_metrics(actual, predicted)
    candidate_pairs: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for (row, _), prediction in zip(holdout_rows, predicted):
        candidate_pairs[(row["semantic_group_id"], row["record_order"], row["mapping"])][row["role"]] = prediction
    candidate_flips = [
        values["a"] != values["b"] for values in candidate_pairs.values() if set(values) == {"a", "b"}
    ]
    control_actual = [row["semantic_target_source"] for row, _ in control_rows]
    control_predicted = [predict(sketch) for _, sketch in control_rows]
    control_metrics = choice_metrics(control_actual, control_predicted)
    control_pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for (row, _), prediction in zip(control_rows, control_predicted):
        control_pairs[row["semantic_group_id"]][row["role"]] = prediction
    control_invariance = [
        values["a"] == values["b"] for values in control_pairs.values() if set(values) == {"a", "b"}
    ]
    return {
        "layer": layer,
        "position_role": position,
        "timing": timing,
        "discovery_centroid_counts": dict(counts),
        "candidate_holdout": candidate_metrics,
        "candidate_predicted_role_flip": wilson(sum(candidate_flips), len(candidate_flips)),
        "control_holdout": control_metrics,
        "control_predicted_role_invariance": wilson(sum(control_invariance), len(control_invariance)),
        "feature_discovery_output_label_blind": True,
        "prediction_uses_frozen_experimental_source_labels": True,
    }


def analyze_physical_model(model: str, stage: str = "physical") -> dict[str, Any]:
    root = OUT / stage / model / "physical"
    path = root / "phase434_physical_rows.jsonl.gz"
    complete = read_json(root / "phase434_physical_complete.json")
    ledger = geometry_ledger(path)
    window = freeze_geometry_window(ledger, model) if stage == "physical" else read_json(OUT / f"phase434_{model}_geometry_window_freeze.json")
    holdout = next(
        row
        for row in ledger["cells"]
        if row["fold"] == "holdout"
        and row["layer"] == window["selected_layer"]
        and row["position_role"] == window["selected_position_role"]
        and row["timing"] == window["selected_timing"]
    )
    temporal = temporal_replication(ledger, window)
    prediction = fixed_window_prediction(path, window)
    protocol = read_json(OUT / "phase434_protocol.json")
    trace_total = int(complete["trace_row_count"])
    finite_fraction = int(complete["finite_sketch_count"]) / max(1, trace_total)
    g1 = bool(
        complete["hook_hidden_state_max_abs_error"] <= protocol["numeric_gates"]["hook_hidden_state_max_abs_error"]
        and finite_fraction >= protocol["numeric_gates"]["physical_finite_fraction_min"]
    )
    g2 = float(holdout["binding_geometry_effect_median"]) >= protocol["numeric_gates"]["binding_geometry_effect_min"]
    g3 = bool(
        temporal["direction_agreement"]["estimate"]
        >= protocol["numeric_gates"]["binding_geometry_holdout_direction_agreement_min"]
        and temporal["effect_spread"]["holdout"] >= protocol["numeric_gates"]["binding_geometry_effect_min"]
    )
    g5 = all(
        prediction["candidate_holdout"]["per_class"][label]["lcb"] >= 0.90
        for label in ("source_1", "source_2")
    )
    g6 = bool(
        prediction["candidate_predicted_role_flip"]["lcb"] >= 0.95
        and prediction["control_predicted_role_invariance"]["lcb"] >= 0.95
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "coordinate_identity": {
            "hook_hidden_state_max_abs_error": complete["hook_hidden_state_max_abs_error"],
            "finite_fraction": clean(finite_fraction),
            "pass": g1,
        },
        "geometry_window_freeze": window,
        "geometry_holdout": holdout,
        "temporal_replication": temporal,
        "fixed_window_prediction": prediction,
        "gates": {
            "G1_token_position_cache_component_identity": g1,
            "G2_label_blind_binding_geometry": g2,
            "G3_binding_time_replication": g3,
            "G4_source_specific_transport": False,
            "G5_complete_event_holdout_prediction": g5,
            "G6_matched_control_specificity": g6,
        },
        "source_transport_status": "not_inferred_from_state_geometry; requires legal component-write follow-up",
        "geometry_ledger": ledger,
        "physical": True,
        "observer": True,
        "predictive": g5,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / f"phase434_{model}_{stage}_physical_audit.json", output)
    return output


def analyze_open() -> dict[str, Any]:
    behavior_gate = read_json(OUT / "phase434_behavior_gate.json")
    eligible = behavior_gate["eligible_models"]
    physical = {
        model: analyze_physical_model(model)
        for model in eligible
        if (OUT / "physical" / model / "physical/phase434_physical_complete.json").exists()
    }
    qwen_behavior = behavior_gate["behavior"][LANGUAGE_MODEL]
    qwen_physical = physical.get(LANGUAGE_MODEL)
    gates = {
        "G0_natural_behavior_qualification": qwen_behavior["behavior_eligible"],
        "G1_token_position_cache_component_identity": bool(qwen_physical and qwen_physical["gates"]["G1_token_position_cache_component_identity"]),
        "G2_label_blind_binding_geometry": bool(qwen_physical and qwen_physical["gates"]["G2_label_blind_binding_geometry"]),
        "G3_binding_time_replication": bool(qwen_physical and qwen_physical["gates"]["G3_binding_time_replication"]),
        "G4_source_specific_transport": False,
        "G5_complete_event_holdout_prediction": bool(qwen_physical and qwen_physical["gates"]["G5_complete_event_holdout_prediction"]),
        "G6_matched_control_specificity": bool(qwen_physical and qwen_physical["gates"]["G6_matched_control_specificity"]),
    }
    unlock = all(gates.values())
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "open",
        "behavior_gate": behavior_gate,
        "physical": physical,
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
        "sealed_unlock": unlock,
        "sealed_rows_read": False,
        "source_transport_not_inferred_from_geometry": True,
        "cross_model_binding_geometry": sum(
            all(value["gates"].values()) for value in physical.values()
        ) >= 2,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase434_open_gate.json", output)
    return output


def build_summary() -> dict[str, Any]:
    protocol = read_json(OUT / "phase434_protocol.json")
    behavior_gate = read_json(OUT / "phase434_behavior_gate.json")
    open_path = OUT / "phase434_open_gate.json"
    open_gate = read_json(open_path) if open_path.exists() else None
    sealed_path = OUT / "sealed/phase434_sealed_result.json"
    sealed = read_json(sealed_path) if sealed_path.exists() else None
    status = (
        "behavior_failed_physical_not_run"
        if not behavior_gate["eligible_models"]
        else "open_geometry_complete_source_transport_pending"
        if open_gate and not open_gate["sealed_unlock"]
        else "open_gates_passed_sealed_pending"
        if open_gate
        else "behavior_qualified_physical_pending"
    )
    summary = {
        "schema_version": "phase434_binding_timeline_summary.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "status": status,
        "denominator": protocol["denominator_audit"],
        "behavior": behavior_gate,
        "open": open_gate,
        "sealed": sealed,
        "evidence": {
            "physical": bool(open_gate and open_gate["physical"]),
            "observer": True,
            "predictive": bool(
                open_gate
                and open_gate["gates"]["G5_complete_event_holdout_prediction"]
            ),
            "source_transport": False,
            "causal": False,
            "single_neuron": False,
            "mechanism_closure": False,
        },
        "closure": {
            "strict_mechanisms": "0/72",
            "overall_scientific_progress_percent": 21,
            "cautious_interval_percent": [18, 24],
        },
    }
    write_json(OUT / "phase434_final_summary.json", summary)
    return summary


def behavior_visual_nodes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for model_index, model in enumerate(MODELS):
        cells = summary["behavior"]["behavior"][model]["cells"]["behavior_holdout"]["candidate"]
        for timing_index, timing in enumerate(TIMINGS):
            selected = [row for row in cells if row["timing"] == timing]
            score = statistics.mean(float(row["complete_event"]["estimate"]) for row in selected)
            nodes.append(
                {
                    "id": f"phase434:{model}:{timing}",
                    "label": f"{model} / {timing}",
                    "type": "binding_timing_behavior",
                    "model": model,
                    "layer": -1,
                    "relative_depth": 0.0,
                    "position_role": timing,
                    "position": [float(timing_index * 5), float(model_index * 5), -5.0],
                    "score": clean(score),
                    "color": ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#8b5cf6"][timing_index],
                    "size": 0.7,
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "evidence_level": "open_natural_behavior",
                    "show_label": timing in {"before_records", "near_query"},
                }
            )
    return nodes


def publish_visual() -> dict[str, Any]:
    summary = build_summary()
    nodes = behavior_visual_nodes(summary)
    physical_stage_run = bool(summary.get("open", {}).get("physical"))
    qwen_physical = (
        summary["open"]["physical"].get(LANGUAGE_MODEL)
        if summary.get("open")
        else None
    )
    if qwen_physical:
        window = qwen_physical["geometry_window_freeze"]
        sample_layers = {
            0,
            qwen_physical["geometry_ledger"]["layer_count"] // 4,
            qwen_physical["geometry_ledger"]["layer_count"] // 2,
            (3 * qwen_physical["geometry_ledger"]["layer_count"]) // 4,
            qwen_physical["geometry_ledger"]["layer_count"] - 1,
            int(window["selected_layer"]),
        }
        for cell in qwen_physical["geometry_ledger"]["cells"]:
            if (
                cell["fold"] != "holdout"
                or cell["layer"] not in sample_layers
                or cell["timing"] != window["selected_timing"]
            ):
                continue
            primary = (
                cell["layer"] == window["selected_layer"]
                and cell["position_role"] == window["selected_position_role"]
            )
            nodes.append(
                {
                    "id": f"phase434:qwen3:L{cell['layer']}:{cell['position_role']}",
                    "label": f"L{cell['layer']} / {cell['position_role']}",
                    "type": "label_blind_binding_geometry" if primary else "binding_geometry_sample",
                    "model": "qwen3",
                    "layer": cell["layer"],
                    "relative_depth": cell["relative_depth"],
                    "position_role": cell["position_role"],
                    "timing": cell["timing"],
                    "position": [float(cell["layer"]), float(POSITION_ROLES.index(cell["position_role"]) * 2.5), 0.0],
                    "score": cell["binding_geometry_effect_median"],
                    "color": POSITION_COLORS[cell["position_role"]],
                    "size": 1.1 if primary else 0.5,
                    "physical": True,
                    "observer": True,
                    "predictive": bool(primary and qwen_physical["predictive"]),
                    "causal": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "output_label_blind": True,
                    "evidence_level": "open_physical_holdout",
                    "show_label": primary,
                }
            )
    edges = []
    for model in MODELS:
        model_nodes = [node for node in nodes if node["model"] == model and node["type"] == "binding_timing_behavior"]
        model_nodes.sort(key=lambda node: TIMINGS.index(node["position_role"]))
        for left, right in zip(model_nodes, model_nodes[1:]):
            edges.append(
                {
                    "id": f"{left['id']}->{right['id']}",
                    "source": left["id"],
                    "target": right["id"],
                    "type": "registered_timing_order",
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "evidence_level": "design_order_not_causal",
                    "color": "#64748b",
                    "weight": 0.4,
                }
            )
    if physical_stage_run:
        evidence_scope = (
            "open natural behavior and label-blind paired state geometry; "
            "source transport unconfirmed; non-causal"
        )
        description = (
            "五个选择器时序、记录顺序与映射置换；自然行为和"
            "无输出标签状态几何分账，非因果。"
        )
    else:
        evidence_scope = (
            "open behavior only; no behavior-qualified model; physical, sealed, "
            "causal, and neuronal stages not run"
        )
        description = (
            "五个选择器时序、记录顺序与映射置换；仅开放行为证据，"
            "物理、密封、因果与神经元阶段未运行。"
        )
    payload = {
        "schema_version": "phase434_binding_timeline_graph.v1",
        "phase_id": PHASE_ID,
        "title": "Phase434 关系绑定形成时序图谱",
        "model": "multi_model",
        "evidence_scope": evidence_scope,
        "graph": {
            "meta": {
                "gates": summary["open"]["gates"] if summary.get("open") else {},
                "eligible_models": summary["behavior"]["eligible_models"],
                "sealed_pass": False,
                "source_transport_unconfirmed": True,
                "physical_stage_run": physical_stage_run,
                "causal": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    VIS.mkdir(parents=True, exist_ok=True)
    filename = "phase434_binding_timeline.json"
    write_json(VIS / filename, payload)
    manifest = {
        "schema_version": "phase434_binding_timeline_manifest.v1",
        "generated_at": now(),
        "default_item_id": "phase434_binding_timeline",
        "items": [
            {
                "id": "phase434_binding_timeline",
                "label": "Phase434 关系绑定形成时序",
                "filename": filename,
                "model": "multi_model",
                "phase": 434,
                "evidence_scope": payload["evidence_scope"],
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase434_binding_timeline",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase434 关系绑定形成时序",
        "description": description,
        "manifest_path": "/vis_data/phase434_binding_timeline/manifest.json",
        "manifest_schema": "phase434_binding_timeline_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase434_binding_timeline",
        "models": list(MODELS),
        "evidence_scope": payload["evidence_scope"],
        "color": "#f97316",
    }
    registry["sources"] = [
        item for item in registry["sources"] if item["id"] != source["id"]
    ] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    return {"manifest": manifest, "node_count": len(nodes), "edge_count": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("behavior", "open", "summary"), required=True)
    parser.add_argument("--publish-visual", action="store_true")
    args = parser.parse_args()
    if args.stage == "behavior":
        output = analyze_behavior_gate()
    elif args.stage == "open":
        output = analyze_open()
    else:
        output = build_summary()
    if args.publish_visual:
        output = {"analysis": output, "visual": publish_visual()}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
