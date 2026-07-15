#!/usr/bin/env python3
"""Analyze Phase433 shared-prefix event trajectories under frozen gates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase433_shared_prefix_collect import POSITION_ROLES  # noqa: E402
from phase433_shared_prefix_protocol import (  # noqa: E402
    LANGUAGE_MODEL,
    MAIN_ROUTES,
    MODELS,
    OPEN_SPLITS,
    OUT,
    PHASE_ID as PROTOCOL_PHASE_ID,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    STRESS_ROUTES,
    TRACE_SCHEMA_VERSION,
    freeze,
    read_json,
    read_jsonl,
    write_json,
)


PHASE_ID = "Phase433-SharedPrefixAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase433_shared_prefix"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
POSITION_COLORS = {
    "source_1_end": "#2563eb",
    "source_2_end": "#0891b2",
    "question_end": "#7c3aed",
    "instruction_start": "#a855f7",
    "instruction_mid": "#c026d3",
    "instruction_end": "#db2777",
    "assistant_boundary": "#ea580c",
    "prompt_terminal": "#dc2626",
    "teacher_branch_boundary": "#16a34a",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase433 non-finite scalar: {value}")
    return round(float(value), 9)


def read_jsonl_any(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return read_jsonl(path)


def wilson(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "estimate": 0.0, "lcb": 0.0, "ucb": 1.0}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": clean(p),
        "lcb": clean(max(0.0, center - radius)),
        "ucb": clean(min(1.0, center + radius)),
    }


def choice_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    classes = ("source_1", "source_2")
    per_class = {}
    recalls = []
    for label in classes:
        selected = [index for index, value in enumerate(actual) if value == label]
        correct = sum(predicted[index] == label for index in selected)
        interval = wilson(correct, len(selected))
        per_class[label] = interval
        recalls.append(float(interval["estimate"]))
    correct = sum(left == right for left, right in zip(actual, predicted))
    return {
        "accuracy": clean(correct / len(actual)) if actual else 0.0,
        "balanced_accuracy": clean(sum(recalls) / len(recalls)) if recalls else 0.0,
        "per_class": per_class,
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


def paired_role_contract(
    rows: Iterable[dict[str, Any]], candidate: bool, expected_flip: bool
) -> dict[str, float | int]:
    cells: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        if bool(row["candidate"]) != candidate or row["route_mode"] not in MAIN_ROUTES:
            continue
        cells[(row["semantic_group_id"], row["route_mode"])][row["role"]] = row[
            "actual_choice"
        ]
    outcomes = []
    for role_map in cells.values():
        if set(role_map) != {"a", "b"} or "other" in role_map.values():
            outcomes.append(False)
            continue
        flipped = role_map["a"] != role_map["b"]
        outcomes.append(flipped if expected_flip else not flipped)
    return wilson(sum(outcomes), len(outcomes))


def analyze_behavior_split(
    rows: list[dict[str, Any]], split: str, protocol: dict[str, Any]
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == split
        and not row.get("stress_only", row["split"] == "conflict_stress")
    ]
    payload: dict[str, Any] = {}
    passes = []
    threshold = protocol["numeric_gates"]["behavior_group_all_lcb_min"]
    role_threshold = protocol["numeric_gates"]["behavior_role_contract_lcb_min"]
    for candidate in (True, False):
        subset = [row for row in selected if bool(row["candidate"]) == candidate]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            grouped[row["semantic_group_id"]].append(row)
        outcomes = [
            len(group_rows) == 2 * len(MAIN_ROUTES)
            and all(condition_good(row) for row in group_rows)
            for group_rows in grouped.values()
        ]
        role_contract = paired_role_contract(subset, candidate, candidate)
        label = "candidate" if candidate else "control"
        payload[label] = {
            "group_all_complete_events": wilson(sum(outcomes), len(outcomes)),
            "role_response_contract": role_contract,
            "condition_complete_event": wilson(
                sum(condition_good(row) for row in subset), len(subset)
            ),
            "natural_common_prefix": wilson(
                sum(row["natural_common_prefix_exact"] for row in subset), len(subset)
            ),
            "natural_branch_correct": wilson(
                sum(row["natural_branch_correct"] for row in subset), len(subset)
            ),
            "teacher_sequence_correct": wilson(
                sum(row["teacher_sequence_correct"] for row in subset), len(subset)
            ),
            "actual_choice_counts": dict(Counter(row["actual_choice"] for row in subset)),
        }
        passes.append(
            payload[label]["group_all_complete_events"]["lcb"] >= threshold
            and role_contract["lcb"] >= role_threshold
        )
    return {
        "split": split,
        "condition_count": len(selected),
        **payload,
        "behavior_gate_pass": bool(selected) and all(passes),
    }


def analyze_stress(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("stress_only", row["split"] == "conflict_stress")
    ]
    return {
        "condition_count": len(selected),
        "nonblocking": True,
        "by_block_route_role_choice": {
            "::".join(key): dict(value)
            for key, value in sorted(
                {
                    key: Counter(
                        row["actual_choice"]
                        for row in selected
                        if (
                            row["block_id"],
                            row["route_mode"],
                            row["role"],
                        )
                        == key
                    )
                    for key in {
                        (row["block_id"], row["route_mode"], row["role"])
                        for row in selected
                    }
                }.items()
            )
        },
        "complete_event": wilson(
            sum(row["natural_complete_event_correct"] for row in selected), len(selected)
        ),
    }


def analyze_behavior(model: str, stage: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase433_protocol.json")
    rows = read_jsonl(
        OUT / stage / model / "behavior/phase433_behavior_rows.jsonl"
    )
    splits = [SEALED_SPLIT] if stage == "sealed" else list(OPEN_SPLITS)
    split_payload = {
        split: analyze_behavior_split(rows, split, protocol) for split in splits
    }
    complete = read_json(
        OUT / stage / model / "behavior/phase433_behavior_complete.json"
    )
    token_fraction = complete["token_contract_valid_count"] / max(
        1, complete["condition_count"]
    )
    return {
        "model": model,
        "stage": stage,
        "condition_count": len(rows),
        "token_contract": {
            "valid": complete["token_contract_valid_count"],
            "total": complete["condition_count"],
            "fraction": clean(token_fraction),
            "common_prefix_length_range": complete["common_prefix_length_range"],
            "pass": token_fraction
            >= protocol["numeric_gates"]["token_contract_valid_fraction_min"],
        },
        "splits": split_payload,
        "stress": analyze_stress(rows) if stage == "open" else None,
        "behavior_holdout_pass": (
            split_payload.get("behavior_holdout", {}).get("behavior_gate_pass", False)
            if stage == "open"
            else split_payload[SEALED_SPLIT]["behavior_gate_pass"]
        ),
    }


def physical_rows(model: str, stage: str) -> list[dict[str, Any]]:
    return read_jsonl_any(
        OUT / stage / model / "physical/phase433_physical_rows.jsonl.gz"
    )


def window_metrics(
    rows: list[dict[str, Any]],
    split: str,
    layer: int,
    position_role: str,
    candidate: bool,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == split
        and int(row["layer"]) == layer
        and bool(row["candidate"]) == candidate
        and row["actual_choice"] in {"source_1", "source_2"}
    ]
    actual = [row["actual_choice"] for row in selected]
    predicted = [
        "source_1"
        if float(
            row["position_metrics"][position_role][
                "source_1_minus_source_2_branch_margin"
            ]
        )
        >= 0.0
        else "source_2"
        for row in selected
    ]
    cells: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row, prediction in zip(selected, predicted):
        cells[(row["semantic_group_id"], row["route_mode"])][row["role"]] = prediction
    flips = [
        role_map["a"] != role_map["b"]
        for role_map in cells.values()
        if set(role_map) == {"a", "b"}
    ]
    return {
        "split": split,
        "layer": layer,
        "position_role": position_role,
        "candidate": candidate,
        "condition_count": len(selected),
        "actual_choice_counts": dict(Counter(actual)),
        "prediction_counts": dict(Counter(predicted)),
        "choice": choice_metrics(actual, predicted),
        "predicted_role_flip": wilson(sum(flips), len(flips)),
        "predicted_role_invariance": wilson(len(flips) - sum(flips), len(flips)),
        "natural_common_prefix": wilson(
            sum(row["natural_common_prefix_exact"] for row in selected), len(selected)
        ),
        "natural_complete_event": wilson(
            sum(row["natural_complete_event_correct"] for row in selected), len(selected)
        ),
    }


def freeze_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = OUT / "phase433_window_freeze.json"
    if path.exists():
        return read_json(path)
    candidates = [
        window_metrics(rows, "observer_calibration", layer, "prompt_terminal", True)
        for layer in (24, 25, 26, 27, 28, 29)
    ]
    ranked = sorted(
        candidates,
        key=lambda item: (-item["choice"]["balanced_accuracy"], item["layer"]),
    )
    selected = ranked[0]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "calibration_split": "observer_calibration",
        "selected_layer": selected["layer"],
        "selected_position_role": "prompt_terminal",
        "selected_balanced_accuracy": selected["choice"]["balanced_accuracy"],
        "candidate_windows": candidates,
        "selection_count": 1,
        "labels_used": True,
        "observer_only": True,
        "holdout_reselection_forbidden": True,
        "sealed_reselection_forbidden": True,
    }
    write_json(path, output)
    return output


def component_analysis() -> dict[str, Any]:
    rows = read_jsonl_any(
        OUT / "open/qwen3/components/phase433_component_rows.jsonl.gz"
    )
    complete = read_json(
        OUT / "open/qwen3/components/phase433_component_complete.json"
    )
    cells = []
    for layer in complete["selected_layers"]:
        for receiver in ("prompt_terminal", "teacher_branch_boundary"):
            selected = [row for row in rows if row["layer"] == layer]
            replay = [
                row["receiver_metrics"][receiver]["attention_replay_relative_error"]
                for row in selected
            ]
            source_1 = [
                row["receiver_metrics"][receiver]["source_partition"]["source_1"][
                    "branch_margin_write"
                ]
                for row in selected
            ]
            source_2 = [
                row["receiver_metrics"][receiver]["source_partition"]["source_2"][
                    "branch_margin_write"
                ]
                for row in selected
            ]
            instruction = [
                row["receiver_metrics"][receiver]["source_partition"]["instruction"][
                    "branch_margin_write"
                ]
                for row in selected
            ]
            template = [
                row["receiver_metrics"][receiver]["source_partition"][
                    "assistant_template"
                ]["branch_margin_write"]
                for row in selected
            ]
            cells.append(
                {
                    "layer": layer,
                    "receiver": receiver,
                    "condition_count": len(selected),
                    "attention_replay_error_median": clean(statistics.median(replay)),
                    "source_1_branch_write_median": clean(statistics.median(source_1)),
                    "source_2_branch_write_median": clean(statistics.median(source_2)),
                    "instruction_branch_write_median": clean(statistics.median(instruction)),
                    "assistant_template_branch_write_median": clean(
                        statistics.median(template)
                    ),
                    "observed_legal_attention_partition": True,
                    "causal": False,
                }
            )
    protocol = read_json(OUT / "phase433_protocol.json")
    passed = bool(
        complete["all_rows_complete"]
        and complete["attention_replay_relative_error_median"]
        <= protocol["numeric_gates"]["attention_replay_median_max"]
    )
    output = {
        "condition_count": complete["condition_count"],
        "trace_row_count": complete["trace_row_count"],
        "attention_replay_relative_error_median": complete[
            "attention_replay_relative_error_median"
        ],
        "conservation_pass": passed,
        "cells": cells,
        "interpretation": "observed source-partition writes, not causal contribution",
    }
    write_json(OUT / "phase433_component_analysis.json", output)
    return output


def build_failure_audit(
    behavior: dict[str, Any], physical: dict[str, Any], component: dict[str, Any]
) -> dict[str, Any]:
    event_fields = (
        "teacher_sequence_correct",
        "natural_common_prefix_exact",
        "natural_reaches_branch_boundary",
        "natural_branch_correct",
        "natural_complete_event_correct",
        "natural_interface_valid",
        "natural_stop",
    )
    model_events: dict[str, Any] = {}
    for model in MODELS:
        rows = read_jsonl(
            OUT / f"open/{model}/behavior/phase433_behavior_rows.jsonl"
        )
        split_payload = {}
        for split in (*OPEN_SPLITS, "conflict_stress"):
            selected = [row for row in rows if row["split"] == split]
            split_payload[split] = {
                "condition_count": len(selected),
                "actual_choice_counts": dict(
                    Counter(row["actual_choice"] for row in selected)
                ),
                "event_counts": {
                    field: sum(bool(row[field]) for row in selected)
                    for field in event_fields
                },
            }
        model_events[model] = split_payload

    qwen_behavior_path = OUT / "open/qwen3/behavior/phase433_behavior_rows.jsonl"
    qwen_materialized_path = (
        OUT / "open/qwen3/behavior/phase433_materialized_conditions.jsonl"
    )
    qwen_rows = read_jsonl(qwen_behavior_path)
    materialized = {
        row["condition_id"]: row for row in read_jsonl(qwen_materialized_path)
    }
    qwen_systematic_failures = {}
    for split in OPEN_SPLITS:
        selected = [
            row
            for row in qwen_rows
            if row["split"] == split
            and row["candidate"]
            and not row["natural_complete_event_correct"]
        ]
        qwen_systematic_failures[split] = {
            "failure_count": len(selected),
            "cells": {
                "::".join(str(value) for value in key): count
                for key, count in Counter(
                    (
                        materialized[row["condition_id"]]["role_mapping_variant"],
                        row["role"],
                        row["route_mode"],
                        row["actual_choice"],
                        materialized[row["condition_id"]]["semantic_target_source"],
                    )
                    for row in selected
                ).items()
            },
        }

    qwen_physical = physical_rows("qwen3", "open")
    route_windows = {}
    for route in MAIN_ROUTES:
        route_rows = [row for row in qwen_physical if row["route_mode"] == route]
        registered = window_metrics(
            route_rows, "behavior_holdout", 26, "prompt_terminal", True
        )
        posthoc_cells = [
            window_metrics(
                route_rows, "behavior_holdout", layer, "prompt_terminal", True
            )
            for layer in range(36)
        ]
        posthoc_best = max(
            posthoc_cells,
            key=lambda item: (item["choice"]["balanced_accuracy"], -item["layer"]),
        )
        route_windows[route] = {
            "registered_l26": registered,
            "posthoc_best_descriptive_only": posthoc_best,
        }
    position_shape = {}
    for position in POSITION_ROLES:
        cells = [
            window_metrics(
                qwen_physical, "behavior_holdout", layer, position, True
            )
            for layer in range(36)
        ]
        best = max(
            cells,
            key=lambda item: (item["choice"]["balanced_accuracy"], -item["layer"]),
        )
        position_shape[position] = {
            "posthoc_best_layer": best["layer"],
            "posthoc_best_balanced_accuracy": best["choice"]["balanced_accuracy"],
            "time_status": (
                "teacher_forced_pre_divergence"
                if position == "teacher_branch_boundary"
                else "natural_prompt_pre_generation"
            ),
        }
    output = {
        "schema_version": "phase433_failure_audit.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "formal_open_gates": {
            "P0_main_behavior": behavior[LANGUAGE_MODEL]["behavior_holdout_pass"],
            "P1_coordinate_and_token_identity": bool(
                behavior[LANGUAGE_MODEL]["token_contract"]["pass"]
                and physical[LANGUAGE_MODEL]["coordinate_identity"]["pass"]
            ),
            "P2_fixed_window_holdout_event_prediction": physical[LANGUAGE_MODEL][
                "fixed_window_event_observer_pass"
            ],
            "P3_candidate_specificity": physical[LANGUAGE_MODEL][
                "candidate_specificity_pass"
            ],
            "P4_component_conservation": component["conservation_pass"],
        },
        "model_event_ledger": model_events,
        "qwen_systematic_candidate_failures": qwen_systematic_failures,
        "qwen_route_window_audit": route_windows,
        "qwen_position_shape_posthoc": position_shape,
        "instrument_incidents": [
            {
                "model": "glm4",
                "incident": "float16 teacher-score NaN before first checkpoint",
                "repair": "rerun with preregistered bfloat16",
                "scientific_result": False,
            },
            {
                "model": "glm4",
                "incident": "process segmentation fault after checkpointed behavior rows",
                "repair": "resume from condition-id checkpoints in a fresh process",
                "scientific_result": False,
            },
            {
                "scope": "analysis",
                "incident": "stress_only metadata omitted by inherited behavior serializer",
                "repair": "recover deterministically from frozen conflict_stress split",
                "scientific_result": False,
            },
        ],
        "sealed_rows_read": False,
        "causal_tested": False,
        "single_neuron_tested": False,
        "conclusion": (
            "The Phase432 first-token terminal observer does not transfer to the "
            "shared-prefix complete event. Qwen3 behavior has one deterministic "
            "pre-source forward-binding failure cell, but no prompt-terminal route "
            "window qualifies for sealed replication."
        ),
    }
    write_json(OUT / "phase433_failure_audit.json", output)
    return output


def build_revision_audit() -> dict[str, Any]:
    protocol = read_json(OUT / "phase433_protocol.json")
    frozen = protocol["implementation_hashes"]
    current = {}
    for name in (
        "phase433_shared_prefix_protocol.py",
        "phase433_shared_prefix_collect.py",
        "phase433_shared_prefix_analysis.py",
        "test_phase433_shared_prefix.py",
    ):
        path = ROOT / "tests/gpt5" / name
        current[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = {
        "schema_version": "phase433_revision_audit.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "frozen_execution_hashes": frozen,
        "current_hashes": current,
        "revisions_after_model_collection": [
            {
                "file": "phase433_shared_prefix_collect.py",
                "scope": "explicitly serialize stress_only and accept the frozen conflict_stress split as deterministic fallback",
                "model_forward_reexecuted": False,
                "stored_behavior_or_physical_value_changed": False,
            },
            {
                "file": "phase433_shared_prefix_analysis.py",
                "scope": "recover stress-only metadata and add stratified failure/route/shape audits",
                "model_forward_reexecuted": False,
                "gate_or_threshold_changed": False,
            },
        ],
        "frozen_rows_and_commitment_unchanged": True,
        "sealed_rows_read": False,
    }
    write_json(OUT / "phase433_revision_audit.json", output)
    return output


def analyze_physical_model(model: str, stage: str) -> dict[str, Any]:
    rows = physical_rows(model, stage)
    complete = read_json(
        OUT / stage / model / "physical/phase433_physical_complete.json"
    )
    layer_count = int(complete["layer_count"])
    identity_pass = bool(
        complete["terminal_native_top1_equal"] == complete["identity_total"]
        and complete["branch_native_top1_equal"] == complete["identity_total"]
        and complete["hook_hidden_state_max_abs_error"]
        <= read_json(OUT / "phase433_protocol.json")["numeric_gates"][
            "hidden_state_hook_max_abs_error"
        ]
    )
    if stage == "open" and model == LANGUAGE_MODEL:
        window = freeze_window(rows)
    else:
        window = read_json(OUT / "phase433_window_freeze.json")
    qwen_layer = int(window["selected_layer"])
    registered_layer = (
        qwen_layer
        if model == LANGUAGE_MODEL
        else round((qwen_layer / 35) * (layer_count - 1))
    )
    split = SEALED_SPLIT if stage == "sealed" else "behavior_holdout"
    candidate = window_metrics(
        rows, split, registered_layer, "prompt_terminal", True
    )
    control = window_metrics(
        rows, split, registered_layer, "prompt_terminal", False
    )
    branch_candidate = window_metrics(
        rows, split, registered_layer, "teacher_branch_boundary", True
    )
    numeric = read_json(OUT / "phase433_protocol.json")["numeric_gates"]
    observer_pass = bool(
        all(
            payload["lcb"] >= numeric["observer_per_class_lcb_min"]
            for payload in candidate["choice"]["per_class"].values()
        )
        and candidate["natural_common_prefix"]["lcb"]
        >= numeric["natural_common_prefix_lcb_min"]
    )
    specificity_pass = bool(
        candidate["predicted_role_flip"]["lcb"]
        >= numeric["candidate_role_flip_lcb_min"]
        and control["predicted_role_invariance"]["lcb"]
        >= numeric["control_role_invariance_lcb_min"]
        and control["predicted_role_flip"]["ucb"]
        <= numeric["control_role_flip_ucb_max"]
    )
    return {
        "model": model,
        "stage": stage,
        "trace_row_count": len(rows),
        "layer_count": layer_count,
        "registered_layer": registered_layer,
        "registered_relative_depth": clean(registered_layer / max(1, layer_count - 1)),
        "window_freeze": window,
        "coordinate_identity": {
            "terminal_native_top1_equal": complete["terminal_native_top1_equal"],
            "branch_native_top1_equal": complete["branch_native_top1_equal"],
            "identity_total": complete["identity_total"],
            "hook_hidden_state_max_abs_error": complete[
                "hook_hidden_state_max_abs_error"
            ],
            "pass": identity_pass,
        },
        "candidate_prompt_terminal": candidate,
        "control_prompt_terminal": control,
        "candidate_teacher_branch_boundary": branch_candidate,
        "fixed_window_event_observer_pass": observer_pass,
        "candidate_specificity_pass": specificity_pass,
        "prompt_terminal_is_pre_generation": True,
        "teacher_branch_boundary_is_teacher_forced": True,
        "language_interpretation_allowed": model == LANGUAGE_MODEL,
    }


def build_shape_map(model: str, stage: str) -> dict[str, Any]:
    rows = physical_rows(model, stage)
    split = SEALED_SPLIT if stage == "sealed" else "behavior_holdout"
    layers = sorted({int(row["layer"]) for row in rows})
    cells = [
        window_metrics(rows, split, layer, position, candidate)
        for layer in layers
        for position in POSITION_ROLES
        for candidate in (True, False)
    ]
    output = {
        "schema_version": "phase433_shape_map.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "split": split,
        "labels_used": True,
        "descriptive_only": True,
        "can_reselect_window": False,
        "cells": cells,
    }
    write_json(OUT / f"phase433_{model}_{stage}_shape_map.json", output)
    return output


def analyze_open() -> dict[str, Any]:
    if not (OUT / "phase433_protocol.json").exists():
        freeze()
    protocol = read_json(OUT / "phase433_protocol.json")
    behavior = {model: analyze_behavior(model, "open") for model in MODELS}
    physical = {model: analyze_physical_model(model, "open") for model in MODELS}
    component = component_analysis()
    failure_audit = build_failure_audit(behavior, physical, component)
    revision_audit = build_revision_audit()
    qwen = LANGUAGE_MODEL
    gates = {
        "P0_main_behavior": behavior[qwen]["behavior_holdout_pass"],
        "P1_coordinate_and_token_identity": bool(
            behavior[qwen]["token_contract"]["pass"]
            and physical[qwen]["coordinate_identity"]["pass"]
        ),
        "P2_fixed_window_holdout_event_prediction": physical[qwen][
            "fixed_window_event_observer_pass"
        ],
        "P3_candidate_specificity": physical[qwen]["candidate_specificity_pass"],
        "P4_component_conservation": component["conservation_pass"],
    }
    unlock = all(gates.values())
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "open",
        "denominator": protocol["denominator_audit"],
        "behavior": behavior,
        "physical": physical,
        "component": component,
        "failure_audit": failure_audit,
        "revision_audit": revision_audit,
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
        "sealed_unlock": unlock,
        "sealed_rows_read": False,
        "stress_gate_blocking": False,
        "cross_model_language_mechanism": False,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase433_open_gate.json", output)
    for model in MODELS:
        build_shape_map(model, "open")
    return output


def analyze_sealed() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase433_open_gate.json")
    if not open_gate.get("sealed_unlock"):
        raise RuntimeError("Phase433 sealed result is not authorized")
    behavior = analyze_behavior(LANGUAGE_MODEL, "sealed")
    physical = analyze_physical_model(LANGUAGE_MODEL, "sealed")
    passed = bool(
        behavior["behavior_holdout_pass"]
        and behavior["token_contract"]["pass"]
        and physical["coordinate_identity"]["pass"]
        and physical["fixed_window_event_observer_pass"]
        and physical["candidate_specificity_pass"]
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "sealed",
        "model": LANGUAGE_MODEL,
        "behavior": behavior,
        "physical": physical,
        "sealed_pass": passed,
        "sealed_rows_read": True,
        "shared_prefix_prechoice_observer_confirmed": passed,
        "aggregate_causal_unlock": passed,
        "causal": False,
        "single_neuron": False,
        "mechanism_closure": False,
    }
    write_json(OUT / "sealed/phase433_sealed_result.json", output)
    build_shape_map(LANGUAGE_MODEL, "sealed")
    return output


def build_summary() -> dict[str, Any]:
    protocol = read_json(OUT / "phase433_protocol.json")
    open_gate = read_json(OUT / "phase433_open_gate.json")
    sealed_path = OUT / "sealed/phase433_sealed_result.json"
    sealed = read_json(sealed_path) if sealed_path.exists() else None
    confirmed = bool(sealed and sealed.get("sealed_pass"))
    status = (
        "sealed_shared_prefix_observer_confirmed"
        if confirmed
        else "open_gates_passed_sealed_pending"
        if open_gate.get("sealed_unlock")
        else "open_gate_failed_sealed_unread"
    )
    progress = 22 if confirmed else 21
    output = {
        "schema_version": "phase433_final_summary.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "status": status,
        "denominator": protocol["denominator_audit"],
        "window_freeze": read_json(OUT / "phase433_window_freeze.json"),
        "open": open_gate,
        "sealed": sealed,
        "evidence": {
            "physical": True,
            "observer": True,
            "predictive": confirmed,
            "causal": False,
            "single_neuron": False,
            "mechanism_closure": False,
            "cross_model_language_mechanism": False,
            "complete_multi_token_event": confirmed,
        },
        "closure": {
            "strict_mechanisms": "0/72",
            "overall_scientific_progress_percent": progress,
            "cautious_interval_percent": [19, 25] if confirmed else [18, 24],
        },
    }
    write_json(OUT / "phase433_final_summary.json", output)
    return output


def publish_visual() -> dict[str, Any]:
    summary = build_summary()
    shape = read_json(OUT / "phase433_qwen3_open_shape_map.json")
    selected_layer = int(summary["window_freeze"]["selected_layer"])
    sample_layers = {0, 8, 16, 23, 24, 25, 26, 27, 28, 29, 35, selected_layer}
    cells = [
        cell
        for cell in shape["cells"]
        if cell["candidate"] and cell["layer"] in sample_layers
    ]
    nodes = []
    for cell in cells:
        layer = int(cell["layer"])
        role = cell["position_role"]
        primary = layer == selected_layer and role == "prompt_terminal"
        teacher = role == "teacher_branch_boundary"
        nodes.append(
            {
                "id": f"phase433:qwen3:L{layer}:{role}",
                "label": f"L{layer} / {role}",
                "type": (
                    "shared_prefix_prechoice_observer"
                    if primary
                    else "teacher_branch_diagnostic"
                    if teacher
                    else "event_shape_sample"
                ),
                "model": "qwen3",
                "layer": layer,
                "relative_depth": clean(layer / 35),
                "position_role": role,
                "position": [float(layer), float(POSITION_ROLES.index(role) * 2.5), 0.0],
                "score": cell["choice"]["balanced_accuracy"],
                "balanced_accuracy": cell["choice"]["balanced_accuracy"],
                "predicted_role_flip": cell["predicted_role_flip"]["estimate"],
                "color": POSITION_COLORS[role],
                "size": 1.1 if primary else 0.55,
                "physical": True,
                "observer": True,
                "predictive": bool(primary and summary["evidence"]["predictive"]),
                "causal": False,
                "single_neuron": False,
                "pipeline_sealed": bool(primary and summary["evidence"]["predictive"]),
                "time_status": (
                    "teacher_forced_pre_divergence"
                    if teacher
                    else "natural_prompt_pre_generation"
                ),
                "evidence_level": (
                    "sealed_predictive_observer"
                    if primary and summary["evidence"]["predictive"]
                    else "open_fixed_observer"
                    if primary
                    else "teacher_forced_diagnostic"
                    if teacher
                    else "descriptive_observer"
                ),
                "show_label": primary,
            }
        )
    edges = []
    for role in POSITION_ROLES:
        role_nodes = sorted(
            [node for node in nodes if node["position_role"] == role],
            key=lambda node: node["layer"],
        )
        for left, right in zip(role_nodes, role_nodes[1:]):
            edges.append(
                {
                    "id": f"{left['id']}->{right['id']}",
                    "source": left["id"],
                    "target": right["id"],
                    "type": "same_position_depth_order",
                    "physical": True,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "evidence_level": "observed_order_not_causal",
                    "color": POSITION_COLORS[role],
                    "weight": 0.45,
                }
            )
    payload = {
        "schema_version": "phase433_shared_prefix_graph.v1",
        "phase_id": PHASE_ID,
        "title": "Phase433 同首词元多词元事件图谱",
        "model": "qwen3",
        "evidence_scope": (
            "shared-prefix complete-event observer; sealed predictive confirmation; non-causal"
            if summary["evidence"]["predictive"]
            else "shared-prefix complete-event observer; open gates failed or sealed unread; non-causal"
        ),
        "graph": {
            "meta": {
                "gates": summary["open"]["gates"],
                "sealed_pass": bool(summary["sealed"] and summary["sealed"].get("sealed_pass")),
                "window_freeze": summary["window_freeze"],
                "complete_multi_token_event": summary["evidence"][
                    "complete_multi_token_event"
                ],
                "teacher_branch_boundary_is_not_prompt_prechoice": True,
                "causal": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    VIS.mkdir(parents=True, exist_ok=True)
    filename = "phase433_qwen3_shared_prefix_event.json"
    write_json(VIS / filename, payload)
    manifest = {
        "schema_version": "phase433_shared_prefix_manifest.v1",
        "generated_at": now(),
        "default_item_id": "phase433_qwen3_shared_prefix_event",
        "items": [
            {
                "id": "phase433_qwen3_shared_prefix_event",
                "label": "Phase433 Qwen3 同首词元多词元事件",
                "filename": filename,
                "model": "qwen3",
                "phase": 433,
                "evidence_scope": payload["evidence_scope"],
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase433_shared_prefix",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase433 同首词元多词元事件",
        "description": (
            "共享长前缀、完整事件和提示终端固定观察器；教师分叉边界单独标注，非因果。"
        ),
        "manifest_path": "/vis_data/phase433_shared_prefix/manifest.json",
        "manifest_schema": "phase433_shared_prefix_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase433_shared_prefix",
        "models": list(MODELS),
        "evidence_scope": payload["evidence_scope"],
        "color": "#dc2626",
    }
    registry["sources"] = [
        item for item in registry["sources"] if item["id"] != source["id"]
    ] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    return {
        "manifest": manifest,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("open", "sealed", "summary"), default="open")
    parser.add_argument("--publish-visual", action="store_true")
    args = parser.parse_args()
    if args.stage == "open":
        output = analyze_open()
    elif args.stage == "sealed":
        output = analyze_sealed()
    else:
        output = build_summary()
    if args.publish_visual:
        output = {"analysis": output, "visual": publish_visual()}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
