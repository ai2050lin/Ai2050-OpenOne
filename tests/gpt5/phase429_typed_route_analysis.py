#!/usr/bin/env python3
"""Analyze Phase429 interfaces and typed behavior gates at independent-group level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase429_typed_route_protocol import (  # noqa: E402
    BEHAVIOR_BLOCKS,
    CONTRACT_VARIANTS,
    INTERFACES,
    MODELS,
    OBSERVER_BLOCKS,
    OUT,
    SCHEMA_VERSION,
    SCORABLE_ROUTES,
)


PHASE_ID = "Phase429-TypedRouteAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase429_typed_route"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
Z_95 = 1.959963984540054
ROUTE_COLORS = {
    "none": "#64748b",
    "source_only": "#22c55e",
    "query_only": "#06b6d4",
    "consistent": "#f59e0b",
    "conflict": "#ef4444",
}
INTERFACE_COLORS = {
    "direct_item": "#14b8a6",
    "short_code": "#3b82f6",
    "result_field": "#8b5cf6",
    "forced_choice": "#ec4899",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase429 non-finite scalar: {value}")
    return round(float(value), 10)


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"estimate": 0.0, "lcb": 0.0, "ucb": 1.0}
    p = successes / total
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = (
        Z_95
        * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
        / denominator
    )
    return {
        "estimate": clean(p),
        "lcb": clean(max(0.0, center - radius)),
        "ucb": clean(min(1.0, center + radius)),
    }


def verify_frozen_implementation() -> dict[str, str]:
    protocol = read_json(OUT / "phase429_protocol.json")
    actual = {}
    for filename, expected in protocol["implementation_commitments"].items():
        digest = hashlib.sha256((ROOT / "tests/gpt5" / filename).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Phase429 implementation changed after freeze: {filename}")
        actual[filename] = digest
    return actual


def completed_rows(model: str, stage: str) -> list[dict[str, Any]]:
    root = OUT / "models" / model / stage
    complete = read_json(root / "phase429_collection_complete.json")
    if not complete.get("all_rows_complete"):
        raise RuntimeError(f"Phase429 incomplete {model} {stage}")
    rows = read_jsonl(root / "phase429_rows.jsonl")
    if len(rows) != int(complete["condition_count"]):
        raise RuntimeError(f"Phase429 row count mismatch {model} {stage}")
    if any(row["physical"] or row["causal"] for row in rows):
        raise RuntimeError(f"Phase429 behavior stage contains physical claims: {model}")
    return rows


def instrument_audit(stage: str, models: Iterable[str] = MODELS) -> dict[str, Any]:
    instrument_stage = f"{stage}_instrument"
    model_rows = {}
    for model in models:
        rows = completed_rows(model, instrument_stage)
        model_rows[model] = {
            "condition_count": len(rows),
            "finite": all(math.isfinite(float(row["teacher_sequence_logprob_margin"])) for row in rows),
            "parser_complete": all("natural_interface_valid" in row for row in rows),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": instrument_stage,
        "models": model_rows,
        "valid": all(row["finite"] and row["parser_complete"] for row in model_rows.values()),
        "updates_thresholds": False,
        "physical_hooks_installed": False,
        "sealed_rows_read": False,
    }


def observer_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        raise RuntimeError("Empty Phase429 observer summary")
    metrics = {
        "teacher_correct": wilson(sum(bool(row["teacher_sequence_correct"]) for row in rows), total),
        "target_first": wilson(sum(bool(row["natural_target_first"]) for row in rows), total),
        "opposite_first": wilson(sum(bool(row["natural_opposite_first"]) for row in rows), total),
        "event_coverage": wilson(sum(bool(row["natural_event_coverage"]) for row in rows), total),
        "interface_valid": wilson(sum(bool(row["natural_interface_valid"]) for row in rows), total),
        "revision": wilson(sum(bool(row["natural_revision"]) for row in rows), total),
        "boundary": wilson(sum(bool(row["natural_boundary"]) for row in rows), total),
        "stop": wilson(sum(bool(row["natural_stop"]) for row in rows), total),
        "censoring": wilson(sum(bool(row["natural_censoring"]) for row in rows), total),
    }
    return {
        "independent_group_count": total,
        "condition_count": total,
        "teacher_margin_median": median(row["teacher_sequence_logprob_margin"] for row in rows),
        "metrics": metrics,
    }


def observer_gate(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    checks = {
        "group_count": summary["independent_group_count"] >= thresholds["groups_per_block_split"],
        "teacher_all_lcb": metrics["teacher_correct"]["lcb"] >= thresholds["teacher_all_lcb_min"],
        "target_first_lcb": metrics["target_first"]["lcb"] >= thresholds["target_first_lcb_min"],
        "opposite_first_ucb": metrics["opposite_first"]["ucb"] <= thresholds["opposite_first_ucb_max"],
        "event_coverage_lcb": metrics["event_coverage"]["lcb"] >= thresholds["event_coverage_lcb_min"],
        "teacher_margin": summary["teacher_margin_median"] > thresholds["teacher_margin_median_min"],
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def analyze_observer() -> dict[str, Any]:
    protocol = read_json(OUT / "phase429_protocol.json")
    implementation = verify_frozen_implementation()
    instrument = instrument_audit("observer")
    if not instrument["valid"]:
        raise RuntimeError("Phase429 observer instrument failed")
    all_rows = [row for model in MODELS for row in completed_rows(model, "observer")]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["model"], row["interface"], row["block_id"], row["split"])].append(row)
    summaries = []
    lookup = {}
    for key, values in sorted(grouped.items()):
        model, interface, block_id, split = key
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "model": model,
            "interface": interface,
            "block_id": block_id,
            "split": split,
            **observer_summary(values),
        }
        summaries.append(summary)
        lookup[key] = summary
    thresholds = protocol["typed_thresholds"]["observer"]
    model_freeze = {}
    for model in MODELS:
        scores = []
        for interface in INTERFACES:
            calibration = [
                lookup[(model, interface, block["block_id"], "interface_calibration")]
                for block in OBSERVER_BLOCKS
            ]
            score = (
                min(row["metrics"]["target_first"]["lcb"] for row in calibration),
                min(row["metrics"]["teacher_correct"]["lcb"] for row in calibration),
                -max(row["metrics"]["opposite_first"]["ucb"] for row in calibration),
                -INTERFACES.index(interface),
            )
            scores.append((score, interface))
        selected = max(scores)[1]
        calibration_gates = {}
        holdout_gates = {}
        for block in OBSERVER_BLOCKS:
            block_id = block["block_id"]
            calibration_gates[block_id] = observer_gate(
                lookup[(model, selected, block_id, "interface_calibration")], thresholds
            )
            holdout_gates[block_id] = observer_gate(
                lookup[(model, selected, block_id, "interface_holdout")], thresholds
            )
        authorized = all(row["gate_pass"] for row in holdout_gates.values())
        model_freeze[model] = {
            "selected_interface": selected,
            "selection_score": list(next(score for score, interface in scores if interface == selected)),
            "selection_split": "interface_calibration",
            "qualification_split": "interface_holdout",
            "calibration_gates": calibration_gates,
            "holdout_gates": holdout_gates,
            "behavior_authorized": authorized,
            "selection_reused_behavior_data": False,
        }
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models": model_freeze,
        "authorized_models": [model for model, row in model_freeze.items() if row["behavior_authorized"]],
        "thresholds_updated": False,
        "behavior_data_read": False,
        "sealed_rows_read": False,
        "physical_hooks_installed": False,
        "implementation_sha256": implementation,
    }
    write_jsonl(OUT / "phase429_observer_summaries.jsonl", summaries)
    write_json(OUT / "phase429_observer_instrument_audit.json", instrument)
    write_json(OUT / "phase429_interface_freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    return freeze


def behavior_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["semantic_group_id"]].append(row)
    group_events = []
    for group_id, values in sorted(by_group.items()):
        if len(values) != 2 or {row["role"] for row in values} != {"a", "b"}:
            raise RuntimeError(f"Phase429 unpaired role group: {group_id}")
        group_events.append(
            {
                "teacher_all": all(row["teacher_sequence_correct"] for row in values),
                "target_first_all": all(row["natural_target_first"] for row in values),
                "opposite_first_any": any(row["natural_opposite_first"] for row in values),
                "event_coverage_all": all(row["natural_event_coverage"] for row in values),
                "interface_valid_all": all(row["natural_interface_valid"] for row in values),
                "revision_any": any(row["natural_revision"] for row in values),
                "boundary_all": all(row["natural_boundary"] for row in values),
                "stop_all": all(row["natural_stop"] for row in values),
                "censor_any": any(row["natural_censoring"] for row in values),
                "teacher_margin": median(row["teacher_sequence_logprob_margin"] for row in values),
                "target_condition_fraction": mean(float(row["natural_target_first"]) for row in values),
                "opposite_condition_fraction": mean(float(row["natural_opposite_first"]) for row in values),
            }
        )
    n = len(group_events)
    binary_keys = (
        "teacher_all",
        "target_first_all",
        "opposite_first_any",
        "event_coverage_all",
        "interface_valid_all",
        "revision_any",
        "boundary_all",
        "stop_all",
        "censor_any",
    )
    return {
        "independent_group_count": n,
        "condition_count": len(rows),
        "teacher_margin_median": median(row["teacher_margin"] for row in group_events),
        "metrics": {key: wilson(sum(bool(row[key]) for row in group_events), n) for key in binary_keys},
        "condition_target_first_fraction": mean(row["target_condition_fraction"] for row in group_events),
        "condition_opposite_first_fraction": mean(row["opposite_condition_fraction"] for row in group_events),
    }


def typed_gates(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    content_t = thresholds["content"]
    content_checks = {
        "group_count": summary["independent_group_count"] >= content_t["groups_per_block_split"],
        "teacher_all_lcb": metrics["teacher_all"]["lcb"] >= content_t["teacher_all_lcb_min"],
        "target_first_all_lcb": metrics["target_first_all"]["lcb"] >= content_t["target_first_all_lcb_min"],
        "opposite_first_any_ucb": metrics["opposite_first_any"]["ucb"] <= content_t["opposite_first_any_ucb_max"],
        "event_coverage_all_lcb": metrics["event_coverage_all"]["lcb"] >= content_t["event_coverage_all_lcb_min"],
        "teacher_margin": summary["teacher_margin_median"] > content_t["teacher_margin_median_min"],
    }
    interface_checks = {
        "valid_all_lcb": metrics["interface_valid_all"]["lcb"] >= thresholds["interface"]["valid_all_lcb_min"]
    }
    revision_checks = {
        "revision_any_ucb": metrics["revision_any"]["ucb"] <= thresholds["revision"]["revision_any_ucb_max"]
    }
    boundary_checks = {
        "boundary_all_lcb": metrics["boundary_all"]["lcb"] >= thresholds["boundary"]["boundary_all_lcb_min"]
    }
    termination_checks = {
        "stop_all_lcb": metrics["stop_all"]["lcb"] >= thresholds["termination"]["stop_all_lcb_min"],
        "censor_any_ucb": metrics["censor_any"]["ucb"] <= thresholds["termination"]["censor_any_ucb_max"],
    }
    output = {}
    for name, checks in (
        ("content", content_checks),
        ("interface", interface_checks),
        ("revision", revision_checks),
        ("boundary", boundary_checks),
        ("termination", termination_checks),
    ):
        output[name] = {"checks": checks, "gate_pass": all(checks.values())}
    output["complete_generation"] = {
        "gate_pass": all(output[name]["gate_pass"] for name in ("content", "interface", "revision", "boundary", "termination"))
    }
    return output


def build_candidate_audits(
    summaries: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = {
        (row["model"], row["block_id"], row["contract_variant"], row["split"], row["route_mode"]): row
        for row in summaries
    }
    candidates = [block for block in BEHAVIOR_BLOCKS if block["candidate"]]
    output = []
    for model in read_json(OUT / "phase429_interface_freeze.json")["authorized_models"]:
        for block in candidates:
            for contract in CONTRACT_VARIANTS:
                route_gates = {}
                for route in SCORABLE_ROUTES:
                    split_rows = {}
                    for split in ("behavior_calibration", "behavior_holdout"):
                        candidate = lookup[(model, block["block_id"], contract, split, route)]
                        control = lookup[(model, block["matched_control_block_id"], contract, split, route)]
                        split_rows[split] = {
                            "candidate": candidate,
                            "control": control,
                            "paired_content_pass": bool(
                                candidate["typed_gates"]["content"]["gate_pass"]
                                and control["typed_gates"]["content"]["gate_pass"]
                            ),
                            "candidate_complete_generation_pass": candidate["typed_gates"]["complete_generation"]["gate_pass"],
                        }
                    route_gates[route] = {
                        "splits": split_rows,
                        "paired_content_qualified": all(row["paired_content_pass"] for row in split_rows.values()),
                        "complete_generation_qualified": all(row["candidate_complete_generation_pass"] for row in split_rows.values()),
                    }
                specificity = {}
                for split in ("behavior_calibration", "behavior_holdout"):
                    candidate_consistent = lookup[(model, block["block_id"], contract, split, "consistent")]
                    candidate_none = lookup[(model, block["block_id"], contract, split, "none")]
                    control_consistent = lookup[(model, block["matched_control_block_id"], contract, split, "consistent")]
                    control_none = lookup[(model, block["matched_control_block_id"], contract, split, "none")]
                    candidate_effect = clean(
                        candidate_consistent["metrics"]["target_first_all"]["estimate"]
                        - candidate_none["metrics"]["target_first_all"]["estimate"]
                    )
                    control_effect = clean(
                        control_consistent["metrics"]["target_first_all"]["estimate"]
                        - control_none["metrics"]["target_first_all"]["estimate"]
                    )
                    delta = clean(candidate_effect - control_effect)
                    specificity[split] = {
                        "candidate_effect": candidate_effect,
                        "control_effect": control_effect,
                        "specificity_delta": delta,
                        "gate_pass": delta >= protocol["typed_thresholds"]["specificity_effect_min"],
                    }
                dual_route = bool(
                    route_gates["consistent"]["paired_content_qualified"]
                    and (
                        route_gates["source_only"]["paired_content_qualified"]
                        or route_gates["query_only"]["paired_content_qualified"]
                    )
                )
                specificity_pass = all(row["gate_pass"] for row in specificity.values())
                output.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "model": model,
                        "block_id": block["block_id"],
                        "family_id": block["family_id"],
                        "mechanism_id": block["mechanism_id"],
                        "matched_control_block_id": block["matched_control_block_id"],
                        "contract_variant": contract,
                        "route_gates": route_gates,
                        "specificity": specificity,
                        "dual_route_content_qualified": dual_route,
                        "specificity_qualified": specificity_pass,
                        "physical_content_authorized": bool(dual_route and specificity_pass),
                        "complete_generation_qualified": bool(
                            dual_route
                            and route_gates["consistent"]["complete_generation_qualified"]
                            and any(route_gates[route]["complete_generation_qualified"] for route in ("source_only", "query_only"))
                        ),
                        "physical_tested": False,
                        "causal_tested": False,
                    }
                )
    return output


def graph_for_model(
    model: str,
    observer_summaries: list[dict[str, Any]],
    behavior_summaries: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    nodes = []
    edges = []
    selected = selection["selected_interface"]
    interface_rows = [
        row
        for row in observer_summaries
        if row["model"] == model and row["split"] == "interface_holdout"
    ]
    by_interface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interface_rows:
        by_interface[row["interface"]].append(row)
    for index, interface in enumerate(INTERFACES):
        values = by_interface.get(interface, [])
        score = min((row["metrics"]["target_first"]["estimate"] for row in values), default=0.0)
        node_id = f"phase429:{model}:interface:{interface}"
        nodes.append(
            {
                "id": node_id,
                "label": f"观察接口 / {interface}",
                "type": "observer_interface_qualification",
                "model": model,
                "interface": interface,
                "selected": interface == selected,
                "qualified": bool(selection["behavior_authorized"] and interface == selected),
                "score": score,
                "size": 0.85 if interface == selected else 0.6,
                "color": INTERFACE_COLORS[interface],
                "position": [float((index - 1.5) * 8), 54.0, 0.0],
                "physical": False,
                "observer": True,
                "native_compute_event": False,
                "compute_edge": False,
                "predictive": False,
                "causal": False,
                "pipeline_sealed": False,
                "strict_double_blind": False,
                "evidence_level": "independent_observer_qualification",
                "show_label": True,
            }
        )
    model_rows = [
        row
        for row in behavior_summaries
        if row["model"] == model and row["split"] == "behavior_holdout"
    ]
    block_order = [block["block_id"] for block in BEHAVIOR_BLOCKS]
    contract_order = list(CONTRACT_VARIANTS)
    for row in model_rows:
        block_index = block_order.index(row["block_id"])
        contract_index = contract_order.index(row["contract_variant"])
        route_index = ("none", "source_only", "query_only", "consistent", "conflict").index(row["route_mode"])
        node_id = f"phase429:{model}:{row['block_id']}:{row['contract_variant']}:{row['route_mode']}"
        nodes.append(
            {
                "id": node_id,
                "label": f"{row['mechanism_id']} / {row['contract_variant']} / {row['route_mode']}",
                "type": "typed_behavior_route_observation",
                "model": model,
                "block_id": row["block_id"],
                "family_id": row["family_id"],
                "mechanism_id": row["mechanism_id"],
                "candidate": row["candidate"],
                "contract_variant": row["contract_variant"],
                "route_mode": row["route_mode"],
                "independent_group_count": row["independent_group_count"],
                "content_gate_pass": row["typed_gates"]["content"]["gate_pass"],
                "interface_gate_pass": row["typed_gates"]["interface"]["gate_pass"],
                "revision_gate_pass": row["typed_gates"]["revision"]["gate_pass"],
                "boundary_gate_pass": row["typed_gates"]["boundary"]["gate_pass"],
                "termination_gate_pass": row["typed_gates"]["termination"]["gate_pass"],
                "score": row["metrics"]["target_first_all"]["estimate"],
                "size": 0.65,
                "color": ROUTE_COLORS[row["route_mode"]],
                "position": [float((route_index - 2) * 8), float(38 - (block_index * 2 + contract_index) * 8), 0.0],
                "physical": False,
                "observer": True,
                "native_compute_event": False,
                "compute_edge": False,
                "predictive": False,
                "causal": False,
                "pipeline_sealed": False,
                "strict_double_blind": False,
                "evidence_level": "independent_typed_behavior_observation",
                "show_label": route_index in {1, 2, 3},
            }
        )
    selected_id = f"phase429:{model}:interface:{selected}"
    for node in [row for row in nodes if row["type"] == "typed_behavior_route_observation"]:
        if node["route_mode"] == "consistent":
            edges.append(
                {
                    "id": f"{selected_id}->{node['id']}",
                    "source": selected_id,
                    "target": node["id"],
                    "type": "observer_coordinate_relation",
                    "physical": False,
                    "observer": True,
                    "compute_edge": False,
                    "predictive": False,
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "evidence_level": "qualified_measurement_coordinate",
                    "color": "#94a3b8",
                    "weight": 1.0,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": model,
        "title": f"Phase429 {model} 观察接口与类型化双路线图谱",
        "evidence_scope": "observer and typed behavior only; non-physical, non-predictive, non-causal",
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "phase": 429,
                "selected_interface": selected,
                "behavior_authorized": selection["behavior_authorized"],
                "physical_node_count": 0,
                "compute_edge_count": 0,
                "pipeline_sealed": False,
                "strict_double_blind": False,
                "causal": False,
            },
        },
    }


def publish_visual(
    observer_summaries: list[dict[str, Any]], behavior_summaries: list[dict[str, Any]]
) -> None:
    freeze = read_json(OUT / "phase429_interface_freeze.json")
    VIS.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase429_{model}_typed_route.json"
        write_json(
            VIS / filename,
            graph_for_model(model, observer_summaries, behavior_summaries, freeze["models"][model]),
        )
        items.append(
            {
                "id": f"phase429_{model}_typed_route",
                "label": f"Phase429 {model} 观察接口与类型化双路线图谱",
                "filename": filename,
                "model": model,
                "phase": 429,
                "evidence_scope": "observer and typed behavior only; non-physical and non-causal",
            }
        )
    manifest = {
        "schema_version": "phase429_typed_route_manifest.v1",
        "generated_at": now(),
        "default_item_id": items[0]["id"],
        "items": items,
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase429_typed_route",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase429 观察接口与类型化双路线",
        "description": "独立接口资格、完全交叉和无示例双分母，以及内容、接口、改口、边界、停止分门。",
        "manifest_path": "/vis_data/phase429_typed_route/manifest.json",
        "manifest_schema": manifest["schema_version"],
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase429_typed_route",
        "models": list(MODELS),
        "evidence_scope": "独立观察接口与类型化行为；非物理、非预测、非因果、非神经元机制",
        "color": "#14b8a6",
    }
    registry["sources"] = [row for row in registry["sources"] if row["id"] != source["id"]] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze_behavior() -> dict[str, Any]:
    protocol = read_json(OUT / "phase429_protocol.json")
    verify_frozen_implementation()
    interface_freeze = read_json(OUT / "phase429_interface_freeze.json")
    authorized_models = interface_freeze["authorized_models"]
    instrument = instrument_audit("behavior", authorized_models)
    if not instrument["valid"]:
        raise RuntimeError("Phase429 behavior instrument failed")
    all_rows = [row for model in authorized_models for row in completed_rows(model, "behavior")]
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["model"], row["block_id"], row["contract_variant"], row["split"], row["route_mode"])].append(row)
    summaries = []
    for key, values in sorted(grouped.items()):
        model, block_id, contract, split, route = key
        first = values[0]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "model": model,
            "block_id": block_id,
            "family_id": first["family_id"],
            "mechanism_id": first["mechanism_id"],
            "candidate": first["candidate"],
            "matched_control_block_id": first["matched_control_block_id"],
            "interface": first["interface"],
            "contract_variant": contract,
            "split": split,
            "route_mode": route,
            **behavior_summary(values),
        }
        summary["typed_gates"] = typed_gates(summary, protocol["typed_thresholds"])
        summaries.append(summary)
    audits = build_candidate_audits(summaries, protocol)
    authorized = [row for row in audits if row["physical_content_authorized"]]
    cross_model = []
    grouped_auth: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in authorized:
        grouped_auth[(row["block_id"], row["contract_variant"])].append(row)
    for (block_id, contract), values in sorted(grouped_auth.items()):
        if len(values) >= protocol["typed_thresholds"]["cross_model_replication_min"]:
            cross_model.append(
                {
                    "block_id": block_id,
                    "contract_variant": contract,
                    "models": sorted(row["model"] for row in values),
                    "replication_count": len(values),
                }
            )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "physical_content_unlock": bool(authorized),
        "authorized_candidates": [
            {
                "model": row["model"],
                "block_id": row["block_id"],
                "contract_variant": row["contract_variant"],
                "matched_control_block_id": row["matched_control_block_id"],
            }
            for row in authorized
        ],
        "cross_model_content_candidates": cross_model,
        "cross_model_content_candidate_count": len(cross_model),
        "sealed_unlock": False,
        "sealed_rows_read": False,
        "physical_hooks_run": False,
        "thresholds_updated": False,
    }
    observer_summaries = read_jsonl(OUT / "phase429_observer_summaries.jsonl")
    write_jsonl(OUT / "phase429_behavior_route_summaries.jsonl", summaries)
    write_jsonl(OUT / "phase429_candidate_audits.jsonl", audits)
    write_json(OUT / "phase429_behavior_instrument_audit.json", instrument)
    write_json(OUT / "phase429_open_behavior_gate.json", gate)
    publish_visual(observer_summaries, summaries)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "registered_observer_condition_count": protocol["validation"]["observer_formal_condition_count"],
        "registered_behavior_group_count": protocol["validation"]["behavior_formal_group_count"],
        "executed_behavior_condition_count": len(all_rows),
        "authorized_observer_models": authorized_models,
        "selected_interfaces": {
            model: interface_freeze["models"][model]["selected_interface"] for model in MODELS
        },
        "physical_content_unlock": gate["physical_content_unlock"],
        "authorized_physical_candidate_count": len(authorized),
        "cross_model_content_candidate_count": len(cross_model),
        "sealed_tested": False,
        "physical_tested": False,
        "predictive_tested": False,
        "causal_tested": False,
        "strict_mechanism_closure": "0/72",
        "overall_scientific_progress_percent": 21,
        "progress_interval_percent": [18, 24],
        "conclusion": (
            "At least one independently qualified model-task-contract passed typed open content and specificity gates; finite architecture-level physical recording is authorized, but no physical mechanism is yet established."
            if authorized
            else "No model-task-contract passed independent observer, candidate, matched-control, calibration, holdout and specificity gates; physical and sealed stages remain closed."
        ),
    }
    write_json(OUT / "phase429_global_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("observer", "behavior"), required=True)
    args = parser.parse_args()
    if args.stage == "observer":
        analyze_observer()
    else:
        analyze_behavior()


if __name__ == "__main__":
    main()
