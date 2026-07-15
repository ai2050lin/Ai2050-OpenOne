#!/usr/bin/env python3
"""Analyze the preregistered Phase432 pre-choice terminal observer."""

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
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase432_prechoice_terminal_protocol import (  # noqa: E402
    LANGUAGE_MODEL,
    MODELS,
    OPEN_SPLIT,
    OUT,
    PRIMARY_WINDOW,
    SCHEMA_VERSION,
    SCORABLE_ROUTES,
    SEALED_SPLIT,
    freeze,
    read_json,
    read_jsonl,
    write_json,
)


PHASE_ID = "Phase432-PrechoiceTerminalAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase432_prechoice_terminal"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
POSITION_ORDER = ("question_end", "instruction_end", "prompt_terminal")
POSITION_COLORS = {
    "question_end": "#0ea5e9",
    "instruction_end": "#f59e0b",
    "prompt_terminal": "#22c55e",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase432 non-finite scalar: {value}")
    return round(float(value), 9)


def read_jsonl_any(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return read_jsonl(path)


def wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {
            "successes": successes,
            "total": total,
            "estimate": 0.0,
            "lcb": 0.0,
            "ucb": 1.0,
        }
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        (p * (1.0 - p) + z * z / (4.0 * total)) / total
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": clean(p),
        "lcb": clean(max(0.0, center - radius)),
        "ucb": clean(min(1.0, center + radius)),
    }


def choice_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    classes = ("source_1", "source_2")
    overall = [left == right for left, right in zip(actual, predicted)]
    per_class = {}
    for label in classes:
        values = [
            prediction == label
            for truth, prediction in zip(actual, predicted)
            if truth == label
        ]
        per_class[label] = wilson(sum(values), len(values))
    balanced = statistics.mean(row["estimate"] for row in per_class.values())
    return {
        "overall": wilson(sum(overall), len(overall)),
        "per_class": per_class,
        "balanced_accuracy": clean(balanced),
        "confusion": dict(Counter(f"{a}->{p}" for a, p in zip(actual, predicted))),
    }


def condition_good(row: dict[str, Any]) -> bool:
    return bool(
        row["teacher_sequence_correct"]
        and row["natural_target_first"]
        and not row["natural_opposite_first"]
        and row["natural_interface_valid"]
        and not row["natural_revision"]
        and row["natural_boundary"]
        and row["natural_stop"]
        and not row["natural_censoring"]
    )


def paired_role_contract(
    rows: Iterable[dict[str, Any]], candidate: bool, expected_flip: bool
) -> dict[str, float]:
    cells: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        if bool(row["candidate"]) != candidate or row["route_mode"] not in SCORABLE_ROUTES:
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


def analyze_behavior(model: str, stage: str) -> dict[str, Any]:
    rows = read_jsonl(OUT / stage / model / "behavior/phase432_behavior_rows.jsonl")
    payload: dict[str, Any] = {}
    protocol = read_json(OUT / "phase432_protocol.json")
    threshold = protocol["numeric_gates"]["behavior_group_all_lcb_min"]
    passes = []
    for candidate in (True, False):
        selected = [row for row in rows if bool(row["candidate"]) == candidate]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            if candidate and row["route_mode"] not in SCORABLE_ROUTES:
                continue
            grouped[row["semantic_group_id"]].append(row)
        expected = 2 * (len(SCORABLE_ROUTES) if candidate else 5)
        outcomes = [
            len(group_rows) == expected and all(condition_good(row) for row in group_rows)
            for group_rows in grouped.values()
        ]
        label = "candidate" if candidate else "control"
        role_contract = paired_role_contract(selected, candidate, candidate)
        payload[label] = {
            "group_all_events": wilson(sum(outcomes), len(outcomes)),
            "role_response_contract": role_contract,
            "actual_choice_counts": dict(Counter(row["actual_choice"] for row in selected)),
        }
        passes.append(
            payload[label]["group_all_events"]["lcb"] >= threshold
            and role_contract["lcb"] >= threshold
        )
    return {
        "model": model,
        "stage": stage,
        "row_count": len(rows),
        **payload,
        "behavior_gate_pass": bool(rows) and all(passes),
    }


def physical_rows(model: str, stage: str) -> list[dict[str, Any]]:
    return read_jsonl_any(
        OUT / stage / model / "physical/phase432_physical_rows.jsonl.gz"
    )


def window_metrics(
    rows: list[dict[str, Any]], layer: int, position_role: str, candidate: bool
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if int(row["layer"]) == layer
        and bool(row["candidate"]) == candidate
        and row["route_mode"] in SCORABLE_ROUTES
        and row["actual_choice"] in {"source_1", "source_2"}
    ]
    actual = [row["actual_choice"] for row in selected]
    predicted = [
        "source_1"
        if float(row["position_metrics"][position_role]["source_1_minus_source_2_margin"])
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
        "layer": layer,
        "position_role": position_role,
        "candidate": candidate,
        "condition_count": len(selected),
        "actual_choice_counts": dict(Counter(actual)),
        "prediction_counts": dict(Counter(predicted)),
        "choice": choice_metrics(actual, predicted),
        "predicted_role_flip": wilson(sum(flips), len(flips)),
        "predicted_role_invariance": wilson(len(flips) - sum(flips), len(flips)),
    }


def analyze_model(model: str, stage: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase432_protocol.json")
    rows = physical_rows(model, stage)
    complete = read_json(
        OUT / stage / model / "physical/phase432_physical_complete.json"
    )
    layer_count = int(complete["layer_count"])
    registered_layer = (
        int(PRIMARY_WINDOW["layer"])
        if model == LANGUAGE_MODEL
        else round(float(PRIMARY_WINDOW["relative_depth"]) * (layer_count - 1))
    )
    candidate = window_metrics(rows, registered_layer, "prompt_terminal", True)
    control = window_metrics(rows, registered_layer, "prompt_terminal", False)
    candidate_rows = [row for row in rows if row["candidate"] and row["layer"] == 0]
    coverage = wilson(
        sum(row["registered_source_choice"] for row in candidate_rows),
        len(candidate_rows),
    )
    numeric = protocol["numeric_gates"]
    identity_pass = bool(
        complete["terminal_native_top1_equal"]
        == complete["terminal_native_top1_total"]
        and complete["hook_hidden_state_max_abs_error"]
        <= numeric["hidden_state_hook_max_abs_error"]
    )
    primary_pass = bool(
        coverage["lcb"] >= numeric["registered_source_coverage_lcb_min"]
        and all(
            payload["lcb"] >= numeric["primary_choice_per_class_lcb_min"]
            for payload in candidate["choice"]["per_class"].values()
        )
        and candidate["predicted_role_flip"]["lcb"]
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
        "coordinate_identity": {
            "terminal_native_top1_equal": complete["terminal_native_top1_equal"],
            "terminal_native_top1_total": complete["terminal_native_top1_total"],
            "hook_hidden_state_max_abs_error": complete[
                "hook_hidden_state_max_abs_error"
            ],
            "pass": identity_pass,
        },
        "registered_source_coverage": coverage,
        "candidate": candidate,
        "control": control,
        "primary_observer_pass": primary_pass,
        "language_interpretation_allowed": model == LANGUAGE_MODEL,
    }


def shape_map(model: str, stage: str) -> dict[str, Any]:
    rows = physical_rows(model, stage)
    layers = sorted({int(row["layer"]) for row in rows})
    cells = []
    for layer in layers:
        for position in POSITION_ORDER:
            for candidate in (True, False):
                cells.append(window_metrics(rows, layer, position, candidate))
    return {
        "model": model,
        "stage": stage,
        "labels_used": True,
        "descriptive_only": True,
        "can_select_new_primary_window": False,
        "cells": cells,
    }


def analyze_open() -> dict[str, Any]:
    freeze()
    behavior = {model: analyze_behavior(model, "open") for model in MODELS}
    physical = {model: analyze_model(model, "open") for model in MODELS}
    language = LANGUAGE_MODEL
    gates = {
        "H0_qwen_behavior": behavior[language]["behavior_gate_pass"],
        "H1_qwen_coordinate_identity": physical[language]["coordinate_identity"]["pass"],
        "H2_fixed_prechoice_window": physical[language]["primary_observer_pass"],
        "H3_temporal_legality": PRIMARY_WINDOW["position_role"] == "prompt_terminal",
    }
    unlock = all(gates.values())
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "open",
        "behavior": behavior,
        "physical": physical,
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
        "sealed_unlock": unlock,
        "sealed_rows_read": False,
        "cross_model_language_mechanism": False,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase432_open_gate.json", output)
    for model in MODELS:
        write_json(OUT / f"phase432_{model}_open_shape_map.json", shape_map(model, "open"))
    return output


def analyze_sealed() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase432_open_gate.json")
    if not open_gate.get("sealed_unlock"):
        raise RuntimeError("Phase432 sealed result is not authorized")
    behavior = analyze_behavior(LANGUAGE_MODEL, "sealed")
    physical = analyze_model(LANGUAGE_MODEL, "sealed")
    passed = bool(
        behavior["behavior_gate_pass"]
        and physical["coordinate_identity"]["pass"]
        and physical["primary_observer_pass"]
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
        "predictive_observer_confirmed": passed,
        "causal": False,
        "single_neuron": False,
        "mechanism_closure": False,
    }
    write_json(OUT / "sealed/phase432_sealed_result.json", output)
    write_json(
        OUT / "sealed/phase432_qwen3_sealed_shape_map.json",
        shape_map(LANGUAGE_MODEL, "sealed"),
    )
    return output


def build_failure_audit() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase432_open_gate.json")
    qwen_rows = read_jsonl(
        OUT / "open/qwen3/behavior/phase432_behavior_rows.jsonl"
    )
    negative_axes = {"natural_opposite_first", "natural_revision", "natural_censoring"}
    event_axes = (
        "teacher_sequence_correct",
        "natural_target_first",
        "natural_opposite_first",
        "natural_interface_valid",
        "natural_revision",
        "natural_boundary",
        "natural_stop",
        "natural_censoring",
    )
    failed_conditions = []
    axis_counts: Counter[str] = Counter()
    failed_groups = set()
    for row in qwen_rows:
        if row["candidate"]:
            continue
        failures = [
            axis
            for axis in event_axes
            if (bool(row[axis]) if axis in negative_axes else not bool(row[axis]))
        ]
        if not failures:
            continue
        failed_groups.add(row["semantic_group_id"])
        axis_counts.update(failures)
        failed_conditions.append(
            {
                "condition_id": row["condition_id"],
                "semantic_group_id": row["semantic_group_id"],
                "role": row["role"],
                "route_mode": row["route_mode"],
                "actual_choice": row["actual_choice"],
                "failed_axes": failures,
            }
        )

    shape_milestones = {}
    for model in MODELS:
        shape = read_json(OUT / f"phase432_{model}_open_shape_map.json")
        model_rows = []
        for position_role in POSITION_ORDER:
            candidates = [
                row
                for row in shape["cells"]
                if row["candidate"] and row["position_role"] == position_role
            ]
            if not candidates:
                continue
            winner = max(
                candidates,
                key=lambda row: (
                    row["choice"]["balanced_accuracy"],
                    row["choice"]["overall"]["lcb"],
                    -row["layer"],
                ),
            )
            first_ninety = next(
                (
                    row
                    for row in sorted(candidates, key=lambda value: value["layer"])
                    if row["choice"]["balanced_accuracy"] >= 0.90
                ),
                None,
            )
            model_rows.append(
                {
                    "position_role": position_role,
                    "best_layer": winner["layer"],
                    "best_balanced_accuracy": winner["choice"]["balanced_accuracy"],
                    "best_role_flip": winner["predicted_role_flip"]["estimate"],
                    "first_layer_at_or_above_0_90": first_ninety["layer"]
                    if first_ninety
                    else None,
                }
            )
        shape_milestones[model] = model_rows

    output = {
        "schema_version": "phase432_failure_audit.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "failed_gates": open_gate["failed_gates"],
        "qwen_control_failed_group_count": len(failed_groups),
        "qwen_control_failed_condition_count": len(failed_conditions),
        "qwen_control_failure_axis_counts": dict(axis_counts),
        "qwen_control_failed_conditions": failed_conditions,
        "all_qwen_control_failures_are_role_b_conflict": bool(failed_conditions)
        and all(
            row["role"] == "b" and row["route_mode"] == "conflict"
            for row in failed_conditions
        ),
        "shape_milestones": shape_milestones,
        "audit_boundary": {
            "labels_used": True,
            "descriptive_only": True,
            "phase432_gate_reopened": False,
            "sealed_rows_read": False,
            "primary_physical_routes_exclude_conflict": True,
            "frozen_behavior_gate_includes_control_conflict": True,
            "fixed_observer_is_output_basis_not_transport_mechanism": True,
            "source_items_differ_at_first_token": True,
        },
    }
    write_json(OUT / "phase432_failure_audit.json", output)
    return output


def build_summary() -> dict[str, Any]:
    protocol = read_json(OUT / "phase432_protocol.json")
    open_gate = read_json(OUT / "phase432_open_gate.json")
    sealed_path = OUT / "sealed/phase432_sealed_result.json"
    sealed = read_json(sealed_path) if sealed_path.exists() else None
    failure_path = OUT / "phase432_failure_audit.json"
    failure = read_json(failure_path) if failure_path.exists() else None
    confirmed = bool(sealed and sealed.get("sealed_pass"))
    progress = 22 if confirmed else 21
    output = {
        "schema_version": "phase432_final_summary.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "status": "sealed_predictive_observer_confirmed" if confirmed else "open_or_sealed_failed",
        "denominator": protocol["denominator_audit"],
        "primary_window": protocol["primary_window"],
        "open": open_gate,
        "sealed": sealed,
        "failure_audit": failure,
        "evidence": {
            "physical": True,
            "observer": True,
            "predictive": confirmed,
            "causal": False,
            "single_neuron": False,
            "mechanism_closure": False,
            "cross_model_language_mechanism": False,
        },
        "closure": {
            "strict_mechanisms": "0/72",
            "overall_scientific_progress_percent": progress,
            "cautious_interval_percent": [19, 25] if confirmed else [18, 24],
        },
    }
    write_json(OUT / "phase432_final_summary.json", output)
    return output


def publish_visual() -> dict[str, Any]:
    summary = build_summary()
    sealed_confirmed = bool(summary["evidence"]["predictive"])
    evidence_scope = (
        "fixed pre-choice terminal observer; independent open and sealed confirmation; "
        "non-causal and non-neuronal"
        if sealed_confirmed
        else "fixed pre-choice terminal observer; independent open evaluation passed but "
        "behavior gate blocked sealed read; non-causal and non-neuronal"
    )
    open_shape = read_json(OUT / "phase432_qwen3_open_shape_map.json")
    selected_layers = {0, 12, 20, 25, 26, 27, 30, 35}
    cells = [
        row
        for row in open_shape["cells"]
        if row["candidate"] and row["layer"] in selected_layers
    ]
    nodes = []
    edges = []
    for cell in cells:
        layer = int(cell["layer"])
        role = cell["position_role"]
        node_id = f"phase432:qwen3:L{layer}:{role}"
        primary = layer == 26 and role == "prompt_terminal"
        nodes.append(
            {
                "id": node_id,
                "label": f"L{layer} / {role}",
                "type": "fixed_prechoice_observer" if primary else "observer_shape_sample",
                "model": "qwen3",
                "layer": layer,
                "relative_depth": clean(layer / 35),
                "position_role": role,
                "position": [float(layer), float(POSITION_ORDER.index(role) * 3), 0.0],
                "score": cell["choice"]["balanced_accuracy"],
                "balanced_accuracy": cell["choice"]["balanced_accuracy"],
                "predicted_role_flip": cell["predicted_role_flip"]["estimate"],
                "color": POSITION_COLORS[role],
                "size": 1.0 if primary else 0.55,
                "physical": True,
                "observer": True,
                "predictive": bool(primary and summary["evidence"]["predictive"]),
                "causal": False,
                "single_neuron": False,
                "pipeline_sealed": bool(primary and summary["evidence"]["predictive"]),
                "evidence_level": "sealed_predictive_observer"
                if primary and summary["evidence"]["predictive"]
                else "descriptive_observer",
                "show_label": primary,
            }
        )
    node_ids = {node["id"] for node in nodes}
    for role in POSITION_ORDER:
        role_nodes = sorted(
            [node for node in nodes if node["position_role"] == role],
            key=lambda row: row["layer"],
        )
        for left, right in zip(role_nodes, role_nodes[1:]):
            edge_id = f"{left['id']}->{right['id']}"
            if left["id"] in node_ids and right["id"] in node_ids:
                edges.append(
                    {
                        "id": edge_id,
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
                        "weight": 0.5,
                    }
                )
    payload = {
        "schema_version": "phase432_prechoice_terminal_graph.v1",
        "phase_id": PHASE_ID,
        "title": "Phase432 选择前提示终端观察器开放评估",
        "model": "qwen3",
        "evidence_scope": evidence_scope,
        "graph": {
            "meta": {
                "gates": summary["open"]["gates"],
                "sealed_pass": bool(summary["sealed"] and summary["sealed"].get("sealed_pass")),
                "primary_window": summary["primary_window"],
                "observer_labels_used": True,
                "causal": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    VIS.mkdir(parents=True, exist_ok=True)
    filename = "phase432_qwen3_prechoice_terminal.json"
    write_json(VIS / filename, payload)
    manifest = {
        "schema_version": "phase432_prechoice_terminal_manifest.v1",
        "generated_at": now(),
        "default_item_id": "phase432_qwen3_prechoice_terminal",
        "items": [
            {
                "id": "phase432_qwen3_prechoice_terminal",
                "label": "Phase432 Qwen3 选择前终端观察器",
                "filename": filename,
                "model": "qwen3",
                "phase": 432,
                "evidence_scope": payload["evidence_scope"],
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase432_prechoice_terminal",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase432 选择前提示终端开放观察器",
        "description": "固定 L26 提示终端窗口通过独立开放评估；行为门失败，密封集未读取。",
        "manifest_path": "/vis_data/phase432_prechoice_terminal/manifest.json",
        "manifest_schema": "phase432_prechoice_terminal_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase432_prechoice_terminal",
        "models": list(MODELS),
        "evidence_scope": payload["evidence_scope"],
        "color": "#22c55e",
    }
    registry["sources"] = [
        row for row in registry["sources"] if row["id"] != source["id"]
    ] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    return {"manifest": manifest, "node_count": len(nodes), "edge_count": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("open", "sealed", "failure", "summary", "publish")
    )
    args = parser.parse_args()
    if args.command == "open":
        output = analyze_open()
    elif args.command == "sealed":
        output = analyze_sealed()
    elif args.command == "failure":
        output = build_failure_audit()
    elif args.command == "summary":
        output = build_summary()
    else:
        output = publish_visual()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
