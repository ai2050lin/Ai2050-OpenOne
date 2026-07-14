#!/usr/bin/env python3
"""Audit Phase421 physical paths without consuming the physical holdout.

The analysis keeps four questions separate:
1. Did the balanced behavior denominator remain non-degenerate?
2. Do discovery-selected source coordinates replicate at wrong-position controls?
3. Are the corrected MLP geometry measurements above repeat noise?
4. Do frozen physical features reduce continuous-margin error beyond prompt factors?

Only question four can authorize physical-holdout collection.  Nothing in this
file authorizes a causal intervention or a neuron-level claim.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase421_balanced_boundary_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase421_balanced_boundary_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_network", "reasoning", "grammar", "protocol_control")
INTERFACES = ("chat", "completion")
RELATIONS = ("compatible", "conflict")
SPLITS = ("discovery", "calibration", "behavior_holdout")
VALIDATION_SPLITS = ("calibration", "behavior_holdout")
ROLES = ("history_answer", "current_evidence")
DEPTHS = ("early", "middle", "late")
SCHEMA_VERSION = "94.0.0"
SOURCE_REPLICATION_MIN_RATE = 0.80
ROLE_COORDINATE_DISTINCT_MIN_RATE = 0.75
PREDICTION_MIN_SSE_REDUCTION = 0.05
PREDICTION_RIDGE_LAMBDA = 1.0
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite Phase421 analysis scalar: {value}")
    return round(float(value), 10)


def selected_coordinate_index(
    search_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, str], tuple[int, int, str]]:
    output = {}
    for row in search_rows:
        if not row["selected_coordinate"]:
            continue
        key = (
            row["model"],
            row["family_id"],
            row["interface"],
            row["history_relation"],
            row["source_role"],
        )
        output[key] = (int(row["layer"]), int(row["head_index"]), row["depth_bin"])
    expected = len(MODELS) * len(FAMILIES) * len(INTERFACES) * len(RELATIONS) * len(ROLES)
    if len(output) != expected:
        raise RuntimeError(f"Expected {expected} selected coordinates, found {len(output)}")
    return output


def source_replication_audit(
    features: list[dict[str, Any]],
    coordinates: dict[tuple[str, str, str, str, str], tuple[int, int, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        for role in ROLES:
            grouped[(
                row["model"], role, row["family_id"], row["interface"],
                row["history_relation"], row["split"],
            )].append(row)
    cells = []
    for key, rows in sorted(grouped.items()):
        model, role, family, interface, relation, split = key
        values = [float(row[f"{role}_relation_specificity"]) for row in rows]
        random_controls = [float(row[f"{role}_random_head_control"]) for row in rows]
        median_controls = [float(row[f"{role}_same_layer_head_median_control"]) for row in rows]
        layer, head, depth = coordinates[(model, family, interface, relation, role)]
        cells.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase421-FrozenSourceReplication",
            "created_at": now(),
            "model": model,
            "source_role": role,
            "family_id": family,
            "interface": interface,
            "history_relation": relation,
            "split": split,
            "selected_layer": layer,
            "selected_head": head,
            "selected_depth_bin": depth,
            "case_count": len(rows),
            "wrong_position_positive_count": sum(value > 0 for value in values),
            "wrong_position_positive_rate": clean(sum(value > 0 for value in values) / len(values)),
            "median_source_specificity": clean(median(values)),
            "selected_beats_random_head_count": sum(
                value > control for value, control in zip(values, random_controls)
            ),
            "selected_beats_random_head_rate": clean(sum(
                value > control for value, control in zip(values, random_controls)
            ) / len(values)),
            "selected_beats_same_layer_median_count": sum(
                value > control for value, control in zip(values, median_controls)
            ),
            "selected_beats_same_layer_median_rate": clean(sum(
                value > control for value, control in zip(values, median_controls)
            ) / len(values)),
            "selection_uses_discovery_only": True,
            "physical": True,
            "predictive": False,
            "causal": False,
        })

    gates: dict[str, Any] = {}
    for model in MODELS:
        gates[model] = {}
        for role in ROLES:
            validation = [
                row for row in cells
                if row["model"] == model and row["source_role"] == role
                and row["split"] in VALIDATION_SPLITS
            ]
            positive = sum(row["wrong_position_positive_count"] for row in validation)
            total = sum(row["case_count"] for row in validation)
            random_wins = sum(row["selected_beats_random_head_count"] for row in validation)
            median_wins = sum(row["selected_beats_same_layer_median_count"] for row in validation)
            rate = positive / total
            gates[model][role] = {
                "validation_case_count": total,
                "wrong_position_positive_count": positive,
                "wrong_position_positive_rate": clean(rate),
                "selected_beats_random_head_rate": clean(random_wins / total),
                "selected_beats_same_layer_median_rate": clean(median_wins / total),
                "replication_gate_threshold": SOURCE_REPLICATION_MIN_RATE,
                "replication_gate_pass": rate >= SOURCE_REPLICATION_MIN_RATE,
                "random_and_same_layer_controls_are_diagnostics_not_gate": True,
            }
        distinct = 0
        total = 0
        for family in FAMILIES:
            for interface in INTERFACES:
                for relation in RELATIONS:
                    history = coordinates[(model, family, interface, relation, "history_answer")][:2]
                    current = coordinates[(model, family, interface, relation, "current_evidence")][:2]
                    distinct += int(history != current)
                    total += 1
        gates[model]["role_coordinate_separation"] = {
            "distinct_coordinate_count": distinct,
            "cell_count": total,
            "distinct_rate": clean(distinct / total),
            "gate_threshold": ROLE_COORDINATE_DISTINCT_MIN_RATE,
            "separation_gate_pass": distinct / total >= ROLE_COORDINATE_DISTINCT_MIN_RATE,
            "boundary": "structural coordinate separation; not functional double dissociation",
        }
    return cells, gates


def geometry_audit(
    geometry_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in geometry_rows:
        grouped[(row["model"], row["split"], row["depth_bin"], row["history_relation"])].append(row)
    summaries = []
    metrics = (
        "parallel_gain", "orthogonal_rewrite_ratio", "total_mlp_attention_ratio",
        "delta_attention_norm", "delta_mlp_norm", "delta_combined_norm",
    )
    for key, rows in sorted(grouped.items()):
        payload = {
            metric: clean(median([float(row[metric]) for row in rows])) for metric in metrics
        }
        summaries.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase421-IndependentGeometrySummary",
            "created_at": now(),
            "model": key[0],
            "split": key[1],
            "depth_bin": key[2],
            "history_relation": key[3],
            "row_count": len(rows),
            **payload,
            "old_novelty_and_cancellation_metrics_removed_as_algebraically_redundant": True,
            "descriptive_only": True,
            "causal": False,
        })
    noise_summary = {}
    noise_fields = {
        "parallel_gain": "parallel_gain_absolute_repeat_difference",
        "orthogonal_rewrite_ratio": "orthogonal_rewrite_absolute_repeat_difference",
        "total_mlp_attention_ratio": "total_ratio_absolute_repeat_difference",
    }
    for model in MODELS:
        rows = [row for row in noise_rows if row["model"] == model]
        noise_summary[model] = {}
        for metric, field in noise_fields.items():
            values = [float(row[field]) for row in rows]
            maximum = max(values)
            noise_summary[model][metric] = {
                "repeat_row_count": len(values),
                "median_absolute_repeat_difference": clean(median(values)),
                "max_absolute_repeat_difference": clean(maximum),
                "effective_floor": clean(max(maximum, 1e-6)),
            }
    return summaries, noise_summary


BASE_CATEGORICAL = (
    "family_id", "mechanism_id", "interface", "current_identity",
    "current_support_count", "history_reliability_score", "history_relation",
)
BASE_NUMERIC = ("prompt_token_count", "target_token_count")
PATH_FIELDS = tuple(
    [
        f"{role}_{suffix}"
        for role in ROLES
        for suffix in (
            "specificity_change", "selected_relative_depth",
        )
    ]
    + [
        f"{metric}_{depth}_median"
        for metric in (
            "parallel_gain", "orthogonal_rewrite_ratio", "total_mlp_attention_ratio",
            "delta_attention_norm", "delta_mlp_norm",
        )
        for depth in DEPTHS
    ]
)


def categorical_vocabulary(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        field: sorted({str(row[field]) for row in rows}) for field in BASE_CATEGORICAL
    }


def base_feature_dict(row: dict[str, Any], vocab: dict[str, list[str]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for field in BASE_CATEGORICAL:
        value = str(row[field])
        for category in vocab[field][1:]:
            output[f"{field}={category}"] = float(value == category)
    for field in BASE_NUMERIC:
        output[field] = float(row[field])
    binary = {
        "completion": float(row["interface"] == "completion"),
        "current_b": float(row["current_identity"] == "b"),
        "support_3": float(int(row["current_support_count"]) == 3),
        "reliability_3": float(int(row["history_reliability_score"]) == 3),
        "conflict": float(row["history_relation"] == "conflict"),
    }
    names = sorted(binary)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            output[f"interaction:{left}*{right}"] = binary[left] * binary[right]
    for mechanism in vocab["mechanism_id"]:
        output[f"interaction:mechanism={mechanism}*conflict"] = (
            float(str(row["mechanism_id"]) == mechanism) * binary["conflict"]
        )
    return output


def path_feature_dict(row: dict[str, Any]) -> dict[str, float]:
    output = {f"path:{field}": float(row[field]) for field in PATH_FIELDS}
    for role in ROLES:
        relation = float(row[f"{role}_relation_specificity"])
        output[f"path:{role}_selected_minus_random"] = (
            relation - float(row[f"{role}_random_head_control"])
        )
        output[f"path:{role}_selected_minus_same_layer_median"] = (
            relation - float(row[f"{role}_same_layer_head_median_control"])
        )
    return output


def design_rows(
    rows: list[dict[str, Any]],
    vocab: dict[str, list[str]],
    include_path: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    dictionaries = []
    for row in rows:
        values = base_feature_dict(row, vocab)
        if include_path:
            values.update(path_feature_dict(row))
        dictionaries.append(values)
    names = sorted(dictionaries[0])
    matrix = np.asarray([[values[name] for name in names] for values in dictionaries], dtype=np.float64)
    target = np.asarray(
        [float(row["relation_margin_effect_vs_irrelevant"]) for row in rows], dtype=np.float64
    )
    return names, matrix, target


def fit_frozen_linear(
    discovery: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    vocab: dict[str, list[str]],
    include_path: bool,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    names, train_x, train_y = design_rows(discovery, vocab, include_path)
    lower = train_x.min(axis=0)
    upper = train_x.max(axis=0)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_scaled = (train_x - mean) / scale
    target_mean = float(train_y.mean())
    gram = train_scaled.T @ train_scaled
    coefficients = np.linalg.solve(
        gram + PREDICTION_RIDGE_LAMBDA * np.eye(gram.shape[0]),
        train_scaled.T @ (train_y - target_mean),
    )
    rank = int(np.linalg.matrix_rank(train_scaled))
    singular = np.linalg.svd(train_scaled, compute_uv=False)
    outputs = {}
    for split in SPLITS:
        split_rows = [row for row in all_rows if row["split"] == split]
        split_names, split_x, split_y = design_rows(split_rows, vocab, include_path)
        if split_names != names:
            raise RuntimeError("Phase421 design columns changed across splits")
        # Out-of-discovery values cannot receive an unconstrained linear
        # extrapolation.  The clipping boundary is frozen from discovery data.
        split_clipped = np.clip(split_x, lower, upper)
        predictions = target_mean + ((split_clipped - mean) / scale) @ coefficients
        outputs[split] = (split_y, predictions)
    contract = {
        "feature_count_excluding_intercept": len(names),
        "matrix_rank": rank,
        "minimum_singular_value": clean(float(singular[-1])) if len(singular) else 0.0,
        "fit_split": "discovery",
        "refit_on_validation": False,
        "estimator": "ridge_with_discovery_domain_clipping",
        "ridge_lambda": PREDICTION_RIDGE_LAMBDA,
        "algebraically_redundant_source_features_removed": True,
        "feature_names": names,
    }
    return contract, outputs


def sse_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    sse = float(np.dot(error, error))
    return {
        "sse": clean(sse),
        "rmse": clean(math.sqrt(sse / len(target))),
        "target_variance_sse": clean(float(np.dot(target - target.mean(), target - target.mean()))),
    }


def prediction_audit(features: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    gates = {}
    vocab = categorical_vocabulary(features)
    for model in MODELS:
        model_rows = [row for row in features if row["model"] == model]
        discovery = [row for row in model_rows if row["split"] == "discovery"]
        base_contract, base_predictions = fit_frozen_linear(
            discovery, model_rows, vocab, include_path=False
        )
        path_contract, path_predictions = fit_frozen_linear(
            discovery, model_rows, vocab, include_path=True
        )
        split_pass = {}
        for split in SPLITS:
            base_target, base_prediction = base_predictions[split]
            path_target, path_prediction = path_predictions[split]
            if not np.array_equal(base_target, path_target):
                raise RuntimeError("Phase421 baseline/path target order changed")
            base = sse_metrics(base_target, base_prediction)
            path = sse_metrics(path_target, path_prediction)
            reduction = (base["sse"] - path["sse"]) / max(base["sse"], 1e-12)
            gate = split not in VALIDATION_SPLITS or reduction >= PREDICTION_MIN_SSE_REDUCTION
            split_pass[split] = gate
            output.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase421-IncrementalContinuousPrediction",
                "created_at": now(),
                "model": model,
                "split": split,
                "row_count": len(base_target),
                "baseline_sse": base["sse"],
                "baseline_rmse": base["rmse"],
                "path_sse": path["sse"],
                "path_rmse": path["rmse"],
                "relative_sse_reduction": clean(reduction),
                "minimum_required_relative_sse_reduction": PREDICTION_MIN_SSE_REDUCTION,
                "split_gate_pass": gate,
                "baseline_contract": base_contract,
                "path_contract": path_contract,
                "fit_on_discovery_only": True,
                "physical_holdout_used": False,
                "predictive": split in VALIDATION_SPLITS and gate,
                "causal": False,
            })
        model_gate = all(split_pass[split] for split in VALIDATION_SPLITS)
        gates[model] = {
            "calibration_gate_pass": split_pass["calibration"],
            "behavior_holdout_gate_pass": split_pass["behavior_holdout"],
            "both_validation_splits_pass": model_gate,
        }
    return output, gates


def behavior_graph(model: str, audits: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in audits if row["model"] == model]
    nodes = []
    for index, row in enumerate(rows):
        nodes.append({
            "id": f"phase421:{model}:behavior:{row['split']}",
            "label": f"{row['split']} · 正{row['effect_class_rate']['positive']:.2f} / 负{row['effect_class_rate']['negative']:.2f}",
            "type": "balanced_behavior_boundary",
            "model": model,
            **row,
            "score": 1.0 - max(row["effect_class_rate"]["positive"], row["effect_class_rate"]["negative"]),
            "size": 0.7,
            "color": "#14b8a6" if row["behavior_boundary_gate_pass"] else "#ef4444",
            "position": [index * 8.0, 0.0, 0.0],
            "show_label": True,
            "physical": False,
            "predictive": False,
            "causal": False,
        })
    return atlas_payload(
        model, f"Phase421 {model} 平衡行为边界", nodes, [],
        ["The boundary contains positive, negative and near-zero effects.",
         "Behavior balance authorizes development measurement, not a mechanism claim."],
    )


def source_graph(model: str, cells: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row for row in cells
        if row["model"] == model and row["split"] == "behavior_holdout"
    ]
    nodes = []
    for row in rows:
        role_offset = -2.0 if row["source_role"] == "history_answer" else 2.0
        relation_offset = -2.0 if row["history_relation"] == "compatible" else 2.0
        nodes.append({
            "id": f"phase421:{model}:source:{row['source_role']}:{row['family_id']}:{row['interface']}:{row['history_relation']}",
            "label": f"{row['source_role']} · L{row['selected_layer']} H{row['selected_head']}",
            "type": "typed_source_coordinate",
            **row,
            "score": row["wrong_position_positive_rate"],
            "size": 0.4 + 0.5 * row["wrong_position_positive_rate"],
            "color": FAMILY_COLORS[row["family_id"]],
            "position": [(-6.0 if row["interface"] == "chat" else 6.0) + role_offset,
                         row["selected_layer"] * 0.65,
                         FAMILIES.index(row["family_id"]) * 8.0 + relation_offset],
            "show_label": row["wrong_position_positive_rate"] < 0.8,
            "physical": True,
            "predictive": False,
            "causal": False,
        })
    return atlas_payload(
        model, f"Phase421 {model} 冻结来源坐标复现", nodes, [],
        ["Coordinates were selected on discovery groups and frozen before validation.",
         "Wrong-position replication is separate from structural role-coordinate separation.",
         "No node establishes necessity, sufficiency or mediation."],
        {"source_replication_gates": gates[model]},
    )


def geometry_graph(model: str, rows: list[dict[str, Any]], noise: dict[str, Any]) -> dict[str, Any]:
    selected = sorted([
        row for row in rows
        if row["model"] == model and row["split"] == "behavior_holdout"
        and row["history_relation"] == "conflict"
    ], key=lambda row: DEPTHS.index(row["depth_bin"]))
    nodes = []
    for index, row in enumerate(selected):
        nodes.append({
            "id": f"phase421:{model}:geometry:{row['depth_bin']}",
            "label": f"{row['depth_bin']} · g∥ {row['parallel_gain']:.2f} · g⊥ {row['orthogonal_rewrite_ratio']:.2f}",
            "type": "independent_mlp_geometry",
            **row,
            "score": min(1.0, abs(row["parallel_gain"])),
            "size": 0.65,
            "color": ("#38bdf8", "#f59e0b", "#ef4444")[index],
            "position": [0.0, index * 9.0, 0.0],
            "show_label": True,
            "physical": True,
            "predictive": False,
            "causal": False,
        })
    edges = [{
        "id": f"{left['id']}->{right['id']}", "source": left["id"], "target": right["id"],
        "relation": "depth_order_not_causal", "causal": False,
    } for left, right in zip(nodes, nodes[1:])]
    return atlas_payload(
        model, f"Phase421 {model} 独立 MLP 几何", nodes, edges,
        ["Parallel gain, orthogonal rewrite and total ratio are algebraically independent summaries.",
         "Repeat noise is reported separately; reconstruction error is not treated as the metric noise floor.",
         "Geometry remains descriptive unless it passes frozen incremental prediction."],
        {"repeat_noise": noise[model]},
    )


def prediction_graph(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    nodes = []
    for index, row in enumerate(selected):
        nodes.append({
            "id": f"phase421:{model}:prediction:{row['split']}",
            "label": f"{row['split']} · ΔSSE {row['relative_sse_reduction']:.3f}",
            "type": "incremental_prediction_gate",
            **row,
            "score": max(0.0, min(1.0, row["relative_sse_reduction"])),
            "size": 0.7,
            "color": "#22c55e" if row["split_gate_pass"] else "#ef4444",
            "position": [index * 8.0, 0.0, 0.0],
            "show_label": True,
            "physical": True,
            "predictive": row["predictive"],
            "causal": False,
        })
    return atlas_payload(
        model, f"Phase421 {model} 物理特征增量预测", nodes, [],
        ["Both models were fit on discovery rows only.",
         "Calibration and behavior-holdout SSE reductions are reported separately.",
         "Physical holdout remains sealed unless both validation reductions reach 5%."],
    )


def atlas_payload(
    model: str,
    title: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    boundary: list[str],
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase421-BalancedBoundaryAtlas",
        "title": title,
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), **(extra_metrics or {})},
        "evidence_boundary": boundary,
    }


def register_source() -> None:
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase421_balanced_boundary_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase421 平衡竞争边界与增量路径图谱",
        "description": "三模型平衡历史竞争边界、冻结来源坐标、独立 MLP 几何和发现集冻结增量预测。",
        "manifest_path": "/vis_data/phase421_balanced_boundary_atlas/manifest.json",
        "manifest_schema": "phase421_balanced_boundary_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase421_balanced_boundary_atlas",
        "models": list(MODELS),
        "evidence_scope": "balanced natural behavior and qualified development physical paths; non-causal",
        "color": "#14b8a6",
    }
    registry["sources"] = [item for item in registry["sources"] if item["id"] != source["id"]]
    registry["sources"].append(source)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze() -> dict[str, Any]:
    protocol = read_json(OUT / "phase421_protocol.json")
    behavior_audits = read_jsonl(OUT / "phase421_behavior_boundary_audit.jsonl")
    generation_audit = read_json(OUT / "phase421_generation_panel_audit.json")
    search_rows: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    physical_summaries = {}
    behavior_summaries = {}
    for model in MODELS:
        model_root = OUT / "models" / model
        behavior_summaries[model] = read_json(model_root / "phase421_behavior_complete.json")
        physical = read_json(model_root / "phase421_physical_complete.json")
        if not physical["all_development_rows_pass"] or not physical["physical_holdout_remains_sealed"]:
            raise RuntimeError(f"Unqualified Phase421 physical input: {model}")
        physical_summaries[model] = physical
        search_rows.extend(read_jsonl(model_root / "phase421_discovery_coordinate_search.jsonl"))
        features.extend(read_jsonl(model_root / "phase421_fixed_path_feature_rows.jsonl"))
        geometry_rows.extend(read_jsonl(model_root / "phase421_independent_mlp_geometry.jsonl"))
        noise_rows.extend(read_jsonl(model_root / "phase421_geometry_repeat_noise.jsonl"))

    coordinates = selected_coordinate_index(search_rows)
    source_cells, source_gates = source_replication_audit(features, coordinates)
    geometry_summary, noise_summary = geometry_audit(geometry_rows, noise_rows)
    prediction_rows, prediction_gates = prediction_audit(features)
    behavior_gate = all(row["behavior_boundary_gate_pass"] for row in behavior_audits)
    source_gate = all(
        source_gates[model][role]["replication_gate_pass"]
        for model in MODELS for role in ROLES
    )
    role_gate = all(
        source_gates[model]["role_coordinate_separation"]["separation_gate_pass"]
        for model in MODELS
    )
    prediction_gate = all(
        prediction_gates[model]["both_validation_splits_pass"] for model in MODELS
    )
    physical_holdout_authorized = behavior_gate and source_gate and role_gate and prediction_gate
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase421-BalancedBoundaryAtlasSummary",
        "created_at": now(),
        "valid": True,
        "registered_group_count": protocol["group_count"],
        "registered_behavior_condition_count": protocol["condition_count"],
        "measured_behavior_condition_count": sum(
            behavior_summaries[model]["margin_condition_count"] for model in MODELS
        ),
        "measured_generation_condition_count": sum(
            behavior_summaries[model]["generation_panel_count"] for model in MODELS
        ),
        "development_physical_condition_count": sum(
            physical_summaries[model]["physical_development_condition_count"] for model in MODELS
        ),
        "physical_holdout_condition_count": 0,
        "selected_source_coordinate_count": len(coordinates),
        "fixed_path_feature_row_count": len(features),
        "independent_geometry_row_count": len(geometry_rows),
        "repeat_noise_row_count": len(noise_rows),
        "behavior_boundary_audit": behavior_audits,
        "generation_panel_audit": generation_audit,
        "source_replication_gates": source_gates,
        "geometry_repeat_noise": noise_summary,
        "incremental_prediction_gates": prediction_gates,
        "gates": {
            "balanced_behavior_boundary": behavior_gate,
            "source_write_replication": source_gate,
            "structural_history_current_coordinate_separation": role_gate,
            "incremental_continuous_prediction": prediction_gate,
            "physical_holdout_authorized": physical_holdout_authorized,
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
        },
        "strict_mechanism_closure_count": 0,
        "strict_mechanism_denominator": 72,
        "evidence_boundary": [
            "Phase420's local source-path result survives only if frozen coordinates replicate on new balanced groups.",
            "Current-support count, numeric reliability cue and relation type are registered prompt operations, not assumed latent scalar truth.",
            "The former novelty/cancellation pair was removed because it was algebraically redundant; three independent geometry quantities replace it.",
            "Repeat inference, not component-ledger reconstruction error, defines the geometry measurement floor.",
            "The physical prediction model is fit once on discovery groups and evaluated without refitting on calibration and behavior holdout groups.",
            "Physical holdout, causal interventions and neuron scans remain sealed unless every pre-registered gate passes.",
        ],
    }
    write_jsonl(OUT / "phase421_source_replication_audit.jsonl", source_cells)
    write_jsonl(OUT / "phase421_independent_geometry_summary.jsonl", geometry_summary)
    write_jsonl(OUT / "phase421_incremental_prediction_audit.jsonl", prediction_rows)
    write_json(OUT / "phase421_global_summary.json", summary)
    write_json(
        OUT / "phase421_physical_holdout_authorization.json",
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase421-PhysicalHoldoutAuthorization",
            "created_at": now(),
            "physical_holdout_collection_authorized": physical_holdout_authorized,
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
            "required_gates": summary["gates"],
        },
    )

    PUBLIC.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        specs = (
            ("behavior_boundary", "平衡行为边界", behavior_graph(model, behavior_audits)),
            ("typed_source_paths", "冻结来源坐标", source_graph(model, source_cells, source_gates)),
            ("independent_mlp_geometry", "独立 MLP 几何", geometry_graph(model, geometry_summary, noise_summary)),
            ("incremental_prediction", "物理特征增量预测", prediction_graph(model, prediction_rows)),
        )
        for suffix, label, graph in specs:
            filename = f"phase421_{model}_{suffix}.json"
            write_json(PUBLIC / filename, graph)
            items.append({
                "id": f"phase421_{model}_{suffix}",
                "label": f"Phase421 {model} {label}",
                "filename": filename,
                "model": model,
                "phase": 421,
                "evidence_scope": "balanced development atlas; physical holdout status in summary; non-causal",
            })
    write_json(PUBLIC / "manifest.json", {
        "schema_version": "phase421_balanced_boundary_atlas_manifest.v1",
        "generated_at": now(),
        "default_item_id": items[0]["id"],
        "items": items,
    })
    write_json(PUBLIC / "phase421_global_summary.json", summary)
    register_source()
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, allow_nan=False))
