#!/usr/bin/env python3
"""Aggregate Phase420 source paths, rewrite geometry and prediction gates."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase420_typed_path_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase420_typed_path_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_network", "reasoning", "grammar", "protocol_control")
INTERFACES = ("chat", "completion")
DEPTHS = ("early", "middle", "late")
SOURCE_ROLES = ("history_answer", "current_evidence")
ROLE_CONTROLS = {
    "history_answer": "history_length_control",
    "current_evidence": "current_length_control",
}
FAMILY_COLORS = {
    "knowledge_network": "#22c55e",
    "reasoning": "#f59e0b",
    "grammar": "#38bdf8",
    "protocol_control": "#ef4444",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def effect(cells: dict[tuple[str, str], dict[str, Any]], field: str, transform: Any = float) -> float:
    aa = transform(cells[("a", "a")][field])
    ab = transform(cells[("a", "b")][field])
    ba = transform(cells[("b", "a")][field])
    bb = transform(cells[("b", "b")][field])
    return 0.5 * ((ab - aa) + (ba - bb))


def behavior_effect_rows(all_behavior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in all_behavior:
        groups[(row["model"], row["group_id"], row["interface"])][
            (row["current_identity"], row["history_identity"])
        ] = row
    output = []
    for (model, group_id, interface), cells in sorted(groups.items()):
        if len(cells) != 4:
            raise RuntimeError(f"Incomplete Phase420 behavior cells: {model}/{group_id}/{interface}")
        first = cells[("a", "a")]
        margin_effect = effect(cells, "target_first_step_margin")
        target_effect = effect(cells, "target_event_match", int)
        opposite_effect = effect(cells, "opposite_identity_event_match", int)
        censor_effect = effect(cells, "right_censored", int)
        output.append(
            {
                "schema_version": "93.0.0",
                "phase_id": "Phase420-CrossedBehaviorEffect",
                "created_at": now(),
                "model": model,
                "group_id": group_id,
                "family_id": first["family_id"],
                "mechanism_id": first["mechanism_id"],
                "split": first["split"],
                "interface": interface,
                "conflict_effect_on_target_first_step_margin": margin_effect,
                "margin_effect_direction": "negative" if margin_effect < 0 else "positive" if margin_effect > 0 else "zero",
                "conflict_effect_on_target_event": target_effect,
                "conflict_effect_on_opposite_event": opposite_effect,
                "conflict_effect_on_right_censor": censor_effect,
                "target_event_changed": abs(target_effect) > 1e-12,
                "opposite_event_changed": abs(opposite_effect) > 1e-12,
                "right_censor_changed": abs(censor_effect) > 1e-12,
                "fully_crossed_current_by_history_identity": True,
                "physical": False,
                "predictive": False,
                "causal": False,
            }
        )
    return output


def source_specificity_value(row: dict[str, Any], role: str) -> float:
    control = ROLE_CONTROLS[role]
    role_mean = 0.5 * (
        float(row[f"{role}_compatible_write_norm_per_token_mean"])
        + float(row[f"{role}_conflict_write_norm_per_token_mean"])
    )
    control_mean = 0.5 * (
        float(row[f"{control}_compatible_write_norm_per_token_mean"])
        + float(row[f"{control}_conflict_write_norm_per_token_mean"])
    )
    return role_mean - control_mean


def source_coordinate_replication(head_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    coordinate_index = {}
    for model in MODELS:
        for role in SOURCE_ROLES:
            for interface in INTERFACES:
                for family in FAMILIES:
                    discovery = [
                        row
                        for row in head_rows
                        if row["model"] == model
                        and row["split"] == "discovery"
                        and row["interface"] == interface
                        and row["family_id"] == family
                    ]
                    by_coordinate: dict[tuple[int, int, str], list[float]] = defaultdict(list)
                    for row in discovery:
                        by_coordinate[(int(row["layer"]), int(row["head_index"]), row["depth_bin"])].append(
                            source_specificity_value(row, role)
                        )
                    coordinate = max(
                        by_coordinate,
                        key=lambda key: (median(by_coordinate[key]), -key[0], -key[1]),
                    )
                    coordinate_index[(model, role, interface, family)] = coordinate
                    split_payload = {}
                    for split in ("discovery", "calibration", "behavior_holdout"):
                        values = [
                            source_specificity_value(row, role)
                            for row in head_rows
                            if row["model"] == model
                            and row["split"] == split
                            and row["interface"] == interface
                            and row["family_id"] == family
                            and (int(row["layer"]), int(row["head_index"]), row["depth_bin"]) == coordinate
                        ]
                        split_payload[f"{split}_case_count"] = len(values)
                        split_payload[f"{split}_positive_count"] = sum(value > 0 for value in values)
                        split_payload[f"{split}_median_specificity"] = median(values)
                    output.append(
                        {
                            "schema_version": "93.0.0",
                            "phase_id": "Phase420-FrozenSourceCoordinateReplication",
                            "created_at": now(),
                            "model": model,
                            "source_role": role,
                            "control_role": ROLE_CONTROLS[role],
                            "interface": interface,
                            "family_id": family,
                            "selected_layer": coordinate[0],
                            "selected_head": coordinate[1],
                            "selected_depth_bin": coordinate[2],
                            **split_payload,
                            "selection_uses_discovery_only": True,
                            "physical": True,
                            "predictive": False,
                            "causal": False,
                        }
                    )
    validation = {}
    for model in MODELS:
        validation[model] = {}
        for role in SOURCE_ROLES:
            rows = [row for row in output if row["model"] == model and row["source_role"] == role]
            positive = sum(
                row["calibration_positive_count"] + row["behavior_holdout_positive_count"]
                for row in rows
            )
            total = sum(
                row["calibration_case_count"] + row["behavior_holdout_case_count"]
                for row in rows
            )
            validation[model][role] = {
                "positive_count": positive,
                "case_count": total,
                "positive_rate": positive / total,
                "replication_gate_pass": positive / total >= 0.80,
            }
        distinct = 0
        for interface in INTERFACES:
            for family in FAMILIES:
                distinct += int(
                    coordinate_index[(model, "history_answer", interface, family)]
                    != coordinate_index[(model, "current_evidence", interface, family)]
                )
        validation[model]["role_coordinate_separation"] = {
            "distinct_coordinate_count": distinct,
            "cell_count": len(INTERFACES) * len(FAMILIES),
            "distinct_rate": distinct / (len(INTERFACES) * len(FAMILIES)),
            "separation_gate_pass": distinct / (len(INTERFACES) * len(FAMILIES)) >= 0.75,
        }
    return output, validation


def rewrite_summary(rewrite_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rewrite_rows:
        groups[(row["model"], row["split"], row["depth_bin"])].append(row)
    output = []
    for (model, split, depth), rows in sorted(groups.items()):
        classes = Counter(row["stable_rewrite_class"] for row in rows)
        dominant, dominant_count = classes.most_common(1)[0]
        output.append(
            {
                "schema_version": "93.0.0",
                "phase_id": "Phase420-RewriteGeometrySummary",
                "created_at": now(),
                "model": model,
                "split": split,
                "depth_bin": depth,
                "cell_count": len(rows),
                "stable_four_cell_count": sum(row["rewrite_class_stable_across_four_cells"] for row in rows),
                "stable_four_cell_rate": sum(row["rewrite_class_stable_across_four_cells"] for row in rows) / len(rows),
                "dominant_rewrite_class": dominant,
                "dominant_rewrite_class_count": dominant_count,
                "rewrite_class_counts": dict(classes),
                "median_cancellation_compatibility_effect": median(
                    [float(row["cancellation_index_compatibility_effect"]) for row in rows]
                ),
                "median_novelty_compatibility_effect": median(
                    [float(row["rewrite_novelty_compatibility_effect"]) for row in rows]
                ),
                "descriptive_geometry_only": True,
                "causal": False,
            }
        )
    return output


def feature_rows(
    behavior_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    rewrite_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    behavior = {
        (row["model"], row["group_id"], row["interface"]): row for row in behavior_rows
        if row["split"] != "physical_holdout"
    }
    head_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        head_groups[(row["model"], row["group_id"], row["interface"], row["depth_bin"])].append(row)
    source_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_groups[(row["model"], row["group_id"], row["interface"], row["depth_bin"], row["source_role"])].append(row)
    rewrite_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rewrite_rows:
        rewrite_groups[(row["model"], row["group_id"], row["interface"], row["depth_bin"])].append(row)
    output = []
    for key, target in sorted(behavior.items()):
        model, group_id, interface = key
        features = {}
        for depth in DEPTHS:
            heads = head_groups[(model, group_id, interface, depth)]
            for role in SOURCE_ROLES:
                control = ROLE_CONTROLS[role]
                specificity = sorted(
                    [source_specificity_value(row, role) for row in heads], reverse=True
                )
                features[f"path_{role}_source_specificity_top5_{depth}"] = mean(specificity[:5])
                effects = [
                    float(row[f"{role}_compatibility_effect"])
                    - float(row[f"{control}_compatibility_effect"])
                    for row in heads
                ]
                strongest = max(effects, key=abs)
                features[f"path_{role}_compatibility_effect_{depth}"] = strongest
            competition = [float(row["history_current_competition_effect"]) for row in heads]
            features[f"path_history_current_competition_effect_{depth}"] = max(
                competition, key=abs
            )
            for role in SOURCE_ROLES:
                paths = source_groups[(model, group_id, interface, depth, role)]
                features[f"path_{role}_vector_effect_norm_{depth}"] = max(
                    float(row["compatibility_effect_vector_norm"]) for row in paths
                )
            rewrites = rewrite_groups[(model, group_id, interface, depth)]
            for field in (
                "attention_output_norm_compatibility_effect",
                "mlp_output_norm_compatibility_effect",
                "cancellation_index_compatibility_effect",
                "rewrite_novelty_compatibility_effect",
            ):
                prefix = "baseline_absolute" if field.startswith(("attention", "mlp")) else "path_rewrite"
                features[f"{prefix}_{field}_{depth}"] = median(
                    [float(row[field]) for row in rewrites]
                )
        output.append(
            {
                "schema_version": "93.0.0",
                "phase_id": "Phase420-PredictionFeatureRow",
                "created_at": now(),
                "model": model,
                "group_id": group_id,
                "family_id": target["family_id"],
                "split": target["split"],
                "interface": interface,
                "target_margin_negative": target[
                    "conflict_effect_on_target_first_step_margin"
                ] < 0,
                "target_event_changed": target["target_event_changed"],
                "target_right_censor_changed": target["right_censor_changed"],
                "features": features,
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
    return output


def accuracy(predictions: list[bool], labels: list[bool]) -> float:
    return sum(left == right for left, right in zip(predictions, labels)) / len(labels)


def fit_stump(rows: list[dict[str, Any]], feature_names: list[str], target: str) -> dict[str, Any]:
    best = None
    labels = [bool(row[target]) for row in rows]
    for feature in feature_names:
        values = sorted({float(row["features"][feature]) for row in rows})
        thresholds = [values[0] - 1e-9, values[-1] + 1e-9]
        thresholds.extend((left + right) / 2 for left, right in zip(values, values[1:]))
        for threshold in thresholds:
            for orientation in (1, -1):
                predictions = [
                    (float(row["features"][feature]) > threshold)
                    if orientation == 1
                    else (float(row["features"][feature]) <= threshold)
                    for row in rows
                ]
                score = accuracy(predictions, labels)
                candidate = (score, feature, threshold, orientation)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        raise RuntimeError("Cannot fit Phase420 decision stump")
    return {
        "feature": best[1],
        "threshold": best[2],
        "orientation": best[3],
        "discovery_accuracy": best[0],
    }


def stump_predict(rule: dict[str, Any], rows: list[dict[str, Any]]) -> list[bool]:
    return [
        (float(row["features"][rule["feature"]]) > rule["threshold"])
        if rule["orientation"] == 1
        else (float(row["features"][rule["feature"]]) <= rule["threshold"])
        for row in rows
    ]


def prediction_audit(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        model_rows = [row for row in features if row["model"] == model]
        discovery = [row for row in model_rows if row["split"] == "discovery"]
        calibration = [row for row in model_rows if row["split"] == "calibration"]
        holdout = [row for row in model_rows if row["split"] == "behavior_holdout"]
        names = sorted(discovery[0]["features"])
        path_names = [name for name in names if name.startswith("path_")]
        absolute_names = [name for name in names if name.startswith("baseline_absolute_")]
        for target in (
            "target_margin_negative",
            "target_event_changed",
            "target_right_censor_changed",
        ):
            path_rule = fit_stump(discovery, path_names, target)
            absolute_rule = fit_stump(discovery, absolute_names, target)
            majority = sum(bool(row[target]) for row in discovery) * 2 >= len(discovery)
            payload = {}
            for split, rows in (
                ("discovery", discovery),
                ("calibration", calibration),
                ("behavior_holdout", holdout),
            ):
                labels = [bool(row[target]) for row in rows]
                payload[f"path_{split}_accuracy"] = accuracy(stump_predict(path_rule, rows), labels)
                payload[f"absolute_baseline_{split}_accuracy"] = accuracy(
                    stump_predict(absolute_rule, rows), labels
                )
                payload[f"majority_baseline_{split}_accuracy"] = accuracy(
                    [majority] * len(rows), labels
                )
            best_calibration_baseline = max(
                payload["absolute_baseline_calibration_accuracy"],
                payload["majority_baseline_calibration_accuracy"],
            )
            best_holdout_baseline = max(
                payload["absolute_baseline_behavior_holdout_accuracy"],
                payload["majority_baseline_behavior_holdout_accuracy"],
            )
            gate = bool(
                payload["path_calibration_accuracy"] > best_calibration_baseline
                and payload["path_behavior_holdout_accuracy"] > best_holdout_baseline
            )
            output.append(
                {
                    "schema_version": "93.0.0",
                    "phase_id": "Phase420-FrozenPredictionAudit",
                    "created_at": now(),
                    "model": model,
                    "prediction_target": target,
                    "path_rule": path_rule,
                    "absolute_norm_baseline_rule": absolute_rule,
                    "majority_baseline_value": majority,
                    **payload,
                    "strict_prediction_gate_pass": gate,
                    "physical_holdout_used": False,
                    "causal": False,
                }
            )
    return output


def aggregate_graph_head_nodes(
    model: str,
    role: str,
    coordinate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    edges = []
    for row in coordinate_rows:
        if row["model"] != model or row["source_role"] != role:
            continue
        interface_offset = -5.0 if row["interface"] == "chat" else 5.0
        node_id = (
            f"phase420:{model}:{role}:{row['interface']}:{row['family_id']}:"
            f"L{row['selected_layer']}:H{row['selected_head']}"
        )
        validation_rate = (
            row["calibration_positive_count"] + row["behavior_holdout_positive_count"]
        ) / (row["calibration_case_count"] + row["behavior_holdout_case_count"])
        nodes.append(
            {
                "id": node_id,
                "label": f"L{row['selected_layer']} H{row['selected_head']} · {row['family_id']} · {row['interface']}",
                "type": "attention_head_event",
                "model": model,
                **row,
                "score": min(1.0, validation_rate),
                "size": 0.45 + 0.55 * validation_rate,
                "color": FAMILY_COLORS[row["family_id"]],
                "position": [
                    interface_offset,
                    4.0 + 20.0 * row["selected_layer"] / max(1, 39),
                    FAMILIES.index(row["family_id"]) * 8.0,
                ],
                "show_label": True,
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
    ordered = sorted(nodes, key=lambda row: (row["interface"], row["family_id"]))
    for left, right in zip(ordered, ordered[1:]):
        if left["interface"] == right["interface"]:
            edges.append(
                {
                    "id": f"{left['id']}->{right['id']}",
                    "source": left["id"],
                    "target": right["id"],
                    "relation": "display_order_not_causal",
                    "causal": False,
                }
            )
    return nodes, edges


def build_source_graph(
    model: str,
    role: str,
    coordinates: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    nodes, edges = aggregate_graph_head_nodes(model, role, coordinates)
    role_label = "历史来源到查询" if role == "history_answer" else "当前事实到查询"
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase420-TypedSourcePathAtlas",
        "title": f"Phase420 {model} {role_label}图",
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_replication_gate": summary["source_replication_gates"][model][role],
        },
        "evidence_boundary": [
            "Head coordinates were selected on discovery groups only and evaluated unchanged on calibration and behavior holdout groups.",
            "Source writes use actual attention probabilities, value states and output-projection head blocks.",
            "Nodes are natural observational paths; they do not establish necessity, sufficiency or mediation.",
        ],
    }


def build_rewrite_graph(
    model: str,
    rewrites: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in rewrites if row["model"] == model]
    nodes = []
    edges = []
    for depth_index, depth in enumerate(DEPTHS):
        split_rows = [row for row in rows if row["split"] == "behavior_holdout" and row["depth_bin"] == depth]
        row = split_rows[0]
        node_id = f"phase420:{model}:rewrite:{depth}"
        nodes.append(
            {
                "id": node_id,
                "label": f"{depth} · {row['dominant_rewrite_class']}",
                "type": "mlp_rewrite_event",
                "model": model,
                **row,
                "score": row["stable_four_cell_rate"],
                "size": 0.4 + 0.6 * row["stable_four_cell_rate"],
                "color": ("#14b8a6", "#f59e0b", "#ef4444")[depth_index],
                "position": [0.0, depth_index * 9.0, 0.0],
                "show_label": True,
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
    for left, right in zip(nodes, nodes[1:]):
        edges.append(
            {
                "id": f"{left['id']}->{right['id']}",
                "source": left["id"],
                "target": right["id"],
                "relation": "same_block_depth_order_not_causal",
                "causal": False,
            }
        )
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase420-AttentionMLPRewriteAtlas",
        "title": f"Phase420 {model} 注意力—MLP 重写图",
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges)},
        "evidence_boundary": [
            "Rewrite classes summarize the angle and cancellation geometry of same-layer residual writes.",
            "Stable block geometry is not proof that the MLP causally rewrites history information.",
        ],
    }


def build_prediction_graph(model: str, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in predictions if row["model"] == model]
    nodes = []
    for index, row in enumerate(rows):
        node_id = f"phase420:{model}:prediction:{row['prediction_target']}"
        score = row["path_behavior_holdout_accuracy"]
        nodes.append(
            {
                "id": node_id,
                "label": f"{row['prediction_target']} · {'通过' if row['strict_prediction_gate_pass'] else '未通过'}",
                "type": "prediction_gate",
                "model": model,
                **row,
                "score": score,
                "size": 0.45 + 0.55 * score,
                "color": "#22c55e" if row["strict_prediction_gate_pass"] else "#ef4444",
                "position": [(index - 1) * 7.0, 0.0, 0.0],
                "show_label": True,
                "physical": True,
                "predictive": row["strict_prediction_gate_pass"],
                "causal": False,
            }
        )
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase420-PhysicalBehaviorPredictionAtlas",
        "title": f"Phase420 {model} 物理—行为预测图",
        "model": model,
        "graph": {"nodes": nodes, "edges": []},
        "metrics": {
            "node_count": len(nodes),
            "strict_prediction_pass_count": sum(row["strict_prediction_gate_pass"] for row in rows),
        },
        "evidence_boundary": [
            "Rules were selected on discovery groups and evaluated without refitting on calibration and behavior holdout groups.",
            "No physical holdout or causal intervention was consumed because the strict prediction gate failed.",
        ],
    }


def register_source() -> None:
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase420_typed_path_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase420 有类型来源写入与重写图谱",
        "description": "三模型完全交叉当前/历史答案身份下的历史来源、当前来源、注意力头、同层 MLP 重写与留出预测图。",
        "manifest_path": "/vis_data/phase420_typed_path_atlas/manifest.json",
        "manifest_schema": "phase420_typed_path_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase420_typed_path_atlas",
        "models": list(MODELS),
        "evidence_scope": "自然来源写入候选可复现；功能预测门未通过；非因果",
        "color": "#06b6d4",
    }
    registry["sources"] = [item for item in registry["sources"] if item["id"] != source["id"]] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze() -> dict[str, Any]:
    all_behavior = []
    all_heads = []
    all_sources = []
    all_rewrites = []
    trace_summaries = {}
    for model in MODELS:
        model_root = OUT / "models" / model
        complete = read_json(model_root / "phase420_trace_complete.json")
        if not complete["all_development_rows_pass"] or not complete["physical_holdout_remains_sealed"]:
            raise RuntimeError(f"Phase420 model trace is not qualified: {model}")
        trace_summaries[model] = complete
        all_behavior.extend(read_jsonl(model_root / "phase420_behavior_rows.jsonl"))
        all_heads.extend(read_jsonl(model_root / "phase420_head_path_rows.jsonl"))
        all_sources.extend(read_jsonl(model_root / "phase420_source_path_rows.jsonl"))
        all_rewrites.extend(read_jsonl(model_root / "phase420_mlp_rewrite_rows.jsonl"))

    behavior = behavior_effect_rows(all_behavior)
    coordinates, source_gates = source_coordinate_replication(all_heads)
    rewrites = rewrite_summary(all_rewrites)
    features = feature_rows(behavior, all_heads, all_sources, all_rewrites)
    predictions = prediction_audit(features)
    source_gate_pass = all(
        source_gates[model][role]["replication_gate_pass"]
        for model in MODELS
        for role in SOURCE_ROLES
    )
    specificity_gate_pass = all(
        source_gates[model]["role_coordinate_separation"]["separation_gate_pass"]
        for model in MODELS
    )
    prediction_gate_pass = all(row["strict_prediction_gate_pass"] for row in predictions)
    calibration_gate_pass = prediction_gate_pass
    holdout_gate_pass = prediction_gate_pass
    selected_depths = Counter(row["selected_depth_bin"] for row in coordinates)
    cross_model_role_structure_pass = selected_depths["late"] / len(coordinates) >= 0.80
    physical_holdout_authorized = bool(
        source_gate_pass
        and specificity_gate_pass
        and prediction_gate_pass
        and calibration_gate_pass
        and holdout_gate_pass
        and cross_model_role_structure_pass
    )
    behavior_counts = {}
    for model in MODELS:
        model_rows = [row for row in behavior if row["model"] == model]
        behavior_counts[model] = {
            "cell_count": len(model_rows),
            "negative_margin_effect_count": sum(
                row["conflict_effect_on_target_first_step_margin"] < 0 for row in model_rows
            ),
            "target_event_changed_count": sum(row["target_event_changed"] for row in model_rows),
            "right_censor_changed_count": sum(row["right_censor_changed"] for row in model_rows),
        }
    summary = {
        "schema_version": "93.0.0",
        "phase_id": "Phase420-TypedNaturalPathAtlasSummary",
        "created_at": now(),
        "valid": True,
        "frozen_group_count": 33,
        "registered_condition_count": 792,
        "behavior_condition_count": len(all_behavior),
        "development_physical_condition_count": sum(
            trace_summaries[model]["development_physical_condition_count"] for model in MODELS
        ),
        "physical_holdout_condition_count": 0,
        "head_path_row_count": len(all_heads),
        "source_path_row_count": len(all_sources),
        "mlp_rewrite_row_count": len(all_rewrites),
        "behavior_effect_cell_count": len(behavior),
        "source_coordinate_cell_count": len(coordinates),
        "prediction_gate_cell_count": len(predictions),
        "behavior_counts": behavior_counts,
        "source_replication_gates": source_gates,
        "gates": {
            "source_write_replication": source_gate_pass,
            "history_current_source_separation": specificity_gate_pass,
            "unseen_behavior_prediction": prediction_gate_pass,
            "calibration_prediction": calibration_gate_pass,
            "behavior_holdout_prediction": holdout_gate_pass,
            "cross_model_role_structure": cross_model_role_structure_pass,
            "physical_holdout_authorized": physical_holdout_authorized,
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
        },
        "selected_source_depth_counts": dict(selected_depths),
        "strict_mechanism_closure_count": 0,
        "strict_mechanism_denominator": 72,
        "evidence_boundary": [
            "Phase419's 33 fixed-current cases were not intrinsically valid two-current-state groups; Phase420 rebuilt 33 paired groups from the qualified source banks.",
            "History-answer and current-evidence source-write coordinates replicate against length-matched wrong-position controls.",
            "The behavior direction is highly imbalanced toward conflict suppression, so a constant majority baseline is already near-perfect.",
            "Typed path features did not strictly beat frozen absolute-norm and majority baselines on calibration and behavior holdout targets.",
            "MLP rewrite classes describe residual geometry and are not a mediated history-information mechanism.",
            "The physical holdout remains sealed and no causal, channel or neuron intervention is authorized.",
        ],
    }
    write_jsonl(OUT / "phase420_behavior_effect_rows.jsonl", behavior)
    write_jsonl(OUT / "phase420_source_coordinate_replication.jsonl", coordinates)
    write_jsonl(OUT / "phase420_rewrite_geometry_summary.jsonl", rewrites)
    write_jsonl(OUT / "phase420_prediction_feature_rows.jsonl", features)
    write_jsonl(OUT / "phase420_prediction_audit.jsonl", predictions)
    write_json(OUT / "phase420_global_summary.json", summary)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        graph_specs = [
            (
                "history_to_query",
                "历史来源到查询图",
                build_source_graph(model, "history_answer", coordinates, summary),
            ),
            (
                "current_to_query",
                "当前事实到查询图",
                build_source_graph(model, "current_evidence", coordinates, summary),
            ),
            (
                "attention_mlp_rewrite",
                "注意力—MLP 重写图",
                build_rewrite_graph(model, rewrites),
            ),
            (
                "physical_behavior_prediction",
                "物理—行为预测图",
                build_prediction_graph(model, predictions),
            ),
        ]
        for suffix, label, graph in graph_specs:
            filename = f"phase420_{model}_{suffix}.json"
            write_json(PUBLIC / filename, graph)
            items.append(
                {
                    "id": f"phase420_{model}_{suffix}",
                    "label": f"Phase420 {model} {label}",
                    "filename": filename,
                    "model": model,
                    "phase": 420,
                    "evidence_scope": "typed natural source path; prediction not closed; non-causal",
                }
            )
    write_json(
        PUBLIC / "manifest.json",
        {
            "schema_version": "phase420_typed_path_atlas_manifest.v1",
            "generated_at": now(),
            "default_item_id": items[0]["id"],
            "items": items,
        },
    )
    write_json(PUBLIC / "phase420_global_summary.json", summary)
    register_source()
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, allow_nan=False))
