#!/usr/bin/env python3
"""Shared, frozen Phase400 partial-order graph analysis helpers."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase400_partial_order"
MODELS = ("qwen3", "glm4", "deepseek7b")
QUERY_NODES = (
    "source_to_query_route",
    "query_attention",
    "query_mlp",
    "query_residual",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def clean(value: float) -> float:
    return round(float(value), 9)


def trace_root(stage: str, model: str) -> Path:
    return OUT / "dynamic_trace" / stage / "private/models" / model


def load_stage(stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    model_completes: dict[str, Any] = {}
    for model in MODELS:
        root = trace_root(stage, model)
        complete = read_json(root / "complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase400 {stage} collection for {model}")
        events.extend(read_jsonl(root / "event_trajectory_rows.jsonl"))
        predictions.extend(read_jsonl(root / "case_prediction_rows.jsonl"))
        model_completes[model] = complete
    denominator = {
        "case_count": sum(item["case_count"] for item in model_completes.values()),
        "group_model_cell_count": sum(
            item["group_count"] for item in model_completes.values()
        ),
        "quality_group_model_cell_count": sum(
            item["quality_group_count"] for item in model_completes.values()
        ),
        "event_trajectory_row_count": len(events),
        "case_prediction_row_count": len(predictions),
        "all_collection_quality_gates_pass": all(
            item["quality_group_count"] == item["group_count"]
            for item in model_completes.values()
        ),
        "models": model_completes,
    }
    return events, predictions, denominator


def event_matches(event_id: str, config: dict[str, Any]) -> bool:
    # The protocol calls these prefixes, but each entry is a complete role-level ID.
    return event_id in config["prefixes"]


def node_metrics(rows: list[dict[str, Any]], gate: dict[str, float]) -> dict[str, Any]:
    descriptors = [row["partial_order_descriptor"] for row in rows]
    present = [bool(item["interval_present"]) for item in descriptors]
    metrics = {
        "group_count": len(rows),
        "group_interval_count": sum(present),
        "group_interval_pass_rate": sum(present) / max(len(rows), 1),
        # Missing intervals remain zero by construction, so medians use the full denominator.
        "median_interval_duration_layers": median(
            item["duration_layers"] for item in descriptors
        ),
        "median_interval_roq_norm": median(
            item["active_median_roq_norm"] for item in descriptors
        ),
        "median_interval_cross_axis_cosine": median(
            item["active_median_cross_axis_cosine"] for item in descriptors
        ),
        "median_interval_specificity_ratio": median(
            item["active_median_specificity_ratio"] for item in descriptors
        ),
        "group_with_amplification_count": sum(
            bool(item["amplification_layers"]) for item in descriptors
        ),
        "group_with_flip_count": sum(bool(item["flip_layers"]) for item in descriptors),
    }
    metrics["gate_pass"] = bool(
        metrics["group_interval_pass_rate"] >= gate["group_interval_pass_rate_min"]
        and metrics["median_interval_duration_layers"]
        >= gate["median_interval_duration_layers_min"]
        and metrics["median_interval_roq_norm"]
        >= gate["median_interval_roq_norm_min"]
        and metrics["median_interval_cross_axis_cosine"]
        >= gate["median_interval_cross_axis_cosine_min"]
        and metrics["median_interval_specificity_ratio"]
        >= gate["median_interval_specificity_ratio_min"]
    )
    return metrics


def node_score(metrics: dict[str, Any]) -> float:
    return clean(
        metrics["group_interval_pass_rate"]
        * max(metrics["median_interval_duration_layers"], 1.0)
        * metrics["median_interval_roq_norm"]
        * max(metrics["median_interval_cross_axis_cosine"], 0.0)
        * min(metrics["median_interval_specificity_ratio"], 8.0)
    )


def interval_layers(row: dict[str, Any]) -> set[int]:
    return {
        layer
        for start, end in row["partial_order_descriptor"]["qualified_intervals"]
        for layer in range(start, end + 1)
    }


def interval_distance(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    left_intervals = left["partial_order_descriptor"]["qualified_intervals"]
    right_intervals = right["partial_order_descriptor"]["qualified_intervals"]
    if not left_intervals or not right_intervals:
        return None
    distances = []
    for left_start, left_end in left_intervals:
        for right_start, right_end in right_intervals:
            if left_end < right_start:
                distances.append(right_start - left_end)
            elif right_end < left_start:
                distances.append(left_start - right_end)
            else:
                distances.append(0)
    return min(distances)


def index_selected_rows(
    event_rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    selected_ids = {name: item["event_id"] for name, item in selected.items()}
    reverse = {event_id: name for name, event_id in selected_ids.items()}
    for row in event_rows:
        class_name = reverse.get(row["event_id"])
        if class_name is not None:
            result[class_name][row["phase400_public_parallel_group_id"]] = row
    return dict(result)


def edge_metrics(
    selected_rows: dict[str, dict[str, dict[str, Any]]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for source, target, edge_type in protocol["required_edges"]:
        source_rows = selected_rows[source]
        target_rows = selected_rows[target]
        groups = sorted(set(source_rows) | set(target_rows))
        group_results = []
        for group_id in groups:
            left = source_rows.get(group_id)
            right = target_rows.get(group_id)
            distance = interval_distance(left, right) if left and right else None
            if edge_type == "next_semantic_time":
                passed = bool(
                    left
                    and right
                    and left["partial_order_descriptor"]["interval_present"]
                    and right["partial_order_descriptor"]["interval_present"]
                )
            else:
                passed = distance is not None and distance <= protocol["edge_gate"][
                    "same_time_interval_distance_layers_max"
                ]
            group_results.append(
                {
                    "public_parallel_group_id": group_id,
                    "interval_distance_layers": distance,
                    "pass": passed,
                }
            )
        pass_count = sum(item["pass"] for item in group_results)
        pass_rate = pass_count / max(len(group_results), 1)
        output.append(
            {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "group_count": len(group_results),
                "group_pass_count": pass_count,
                "group_pass_rate": clean(pass_rate),
                "gate_pass": pass_rate
                >= protocol["edge_gate"]["group_edge_pass_rate_min"],
                "group_results": group_results,
            }
        )
    return output


def onset_signature(
    selected_rows: dict[str, dict[str, dict[str, Any]]],
    required_nodes: list[str],
    tie_tolerance: float,
) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for name in required_nodes:
        rows = list(selected_rows[name].values())
        onsets = [
            row["partial_order_descriptor"]["onset_layer"] / max(row["layer_count"] - 1, 1)
            for row in rows
            if row["partial_order_descriptor"]["interval_present"]
        ]
        durations = [
            row["partial_order_descriptor"]["duration_layers"] / row["layer_count"]
            for row in rows
        ]
        nodes[name] = {
            "median_relative_onset": clean(median(onsets)) if onsets else None,
            "median_normalized_duration": clean(median(durations)) if durations else 0.0,
        }
    relations = {}
    for left, right in combinations(required_nodes, 2):
        left_onset = nodes[left]["median_relative_onset"]
        right_onset = nodes[right]["median_relative_onset"]
        if left_onset is None or right_onset is None:
            relation = "missing"
        elif abs(left_onset - right_onset) <= tie_tolerance:
            relation = "tie"
        elif left_onset < right_onset:
            relation = "before"
        else:
            relation = "after"
        relations[f"{left}|{right}"] = relation
    return {"nodes": nodes, "pair_relations": relations}


def select_discovery_cell(
    model: str,
    surface: str,
    cell_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[row["event_id"]].append(row)
    selected: dict[str, dict[str, Any]] = {}
    searched = 0
    for class_name, config in protocol["event_classes"].items():
        options = []
        for event_id, rows in grouped.items():
            if not event_matches(event_id, config):
                continue
            searched += 1
            metrics = node_metrics(rows, protocol["discovery_node_gate"])
            options.append(
                {
                    "event_id": event_id,
                    "metrics": metrics,
                    "gate_pass": metrics["gate_pass"],
                    "score": node_score(metrics),
                }
            )
        if not options:
            raise RuntimeError(f"Missing Phase400 event class {model}/{surface}/{class_name}")
        passing = [item for item in options if item["gate_pass"]]
        chosen = max(
            passing or options,
            key=lambda item: (item["score"], item["event_id"]),
        )
        selected[class_name] = {
            "class_name": class_name,
            "required": config["required"],
            "semantic_time": config["semantic_time"],
            "selected_from_passing_set": bool(passing),
            "searched_event_count": len(options),
            **chosen,
        }
    selected_rows = index_selected_rows(cell_rows, selected)
    edges = edge_metrics(selected_rows, protocol)
    required = protocol["required_nodes"]
    required_nodes_pass = all(selected[name]["gate_pass"] for name in required)
    required_edges_pass = all(item["gate_pass"] for item in edges)
    signature = onset_signature(
        selected_rows,
        required,
        protocol["crossmodel_isomorphism_gate"]["relative_onset_tie_tolerance"],
    )
    return {
        "model": model,
        "surface": surface,
        "event_classes": selected,
        "required_edges": edges,
        "required_node_gate_pass": required_nodes_pass,
        "required_edge_gate_pass": required_edges_pass,
        "partial_order_graph_pass": required_nodes_pass and required_edges_pass,
        "onset_signature": signature,
        "searched_event_count": searched,
    }


def assess_frozen_cell(
    frozen: dict[str, Any],
    cell_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        by_event[row["event_id"]].append(row)
    selected = {}
    for class_name, frozen_event in frozen["event_classes"].items():
        rows = by_event[frozen_event["event_id"]]
        if not rows:
            raise RuntimeError(
                f"Missing frozen Phase400 event {frozen['model']}/{frozen['surface']}/"
                f"{frozen_event['event_id']}"
            )
        metrics = node_metrics(rows, protocol["validation_node_gate"])
        selected[class_name] = {
            "class_name": class_name,
            "event_id": frozen_event["event_id"],
            "required": frozen_event["required"],
            "semantic_time": frozen_event["semantic_time"],
            "metrics": metrics,
            "gate_pass": metrics["gate_pass"],
            "frozen_on_discovery": True,
        }
    selected_rows = index_selected_rows(cell_rows, selected)
    edges = edge_metrics(selected_rows, protocol)
    required = protocol["required_nodes"]
    required_nodes_pass = all(selected[name]["gate_pass"] for name in required)
    required_edges_pass = all(item["gate_pass"] for item in edges)
    return {
        "model": frozen["model"],
        "surface": frozen["surface"],
        "event_classes": selected,
        "required_edges": edges,
        "required_node_gate_pass": required_nodes_pass,
        "required_edge_gate_pass": required_edges_pass,
        "partial_order_graph_pass": required_nodes_pass and required_edges_pass,
        "onset_signature": onset_signature(
            selected_rows,
            required,
            protocol["crossmodel_isomorphism_gate"]["relative_onset_tie_tolerance"],
        ),
        "discovery_graph_pass": frozen["partial_order_graph_pass"],
    }


def graph_layers_by_group(
    selected_rows: dict[str, dict[str, dict[str, Any]]], layer_count: int
) -> dict[str, list[int]]:
    groups = sorted(
        set.intersection(*(set(selected_rows[name]) for name in QUERY_NODES))
    )
    output = {}
    for group_id in groups:
        dilated = []
        for name in QUERY_NODES:
            layers = interval_layers(selected_rows[name][group_id])
            expanded = {
                candidate
                for layer in layers
                for candidate in (layer - 1, layer, layer + 1)
                if 0 <= candidate < layer_count
            }
            dilated.append(expanded)
        output[group_id] = sorted(set.intersection(*dilated))
    return output


def vote(margins: list[float], layers: Iterable[int]) -> float | None:
    values = [margins[layer] for layer in layers]
    return clean(median(values)) if values else None


def accuracy(votes: list[float | None]) -> float:
    return sum(value is not None and value > 0.0 for value in votes) / max(len(votes), 1)


def control_layers(
    kind: str,
    true_layers: list[int],
    layer_count: int,
    key: str,
) -> list[int]:
    if not true_layers:
        return []
    if kind == "wrong_depth":
        half = max(layer_count // 2, 1)
        return sorted(
            {
                min(layer + half, layer_count - 1)
                if layer < half
                else max(layer - half, 0)
                for layer in true_layers
            }
        )
    if kind == "depth_reversal":
        return sorted({layer_count - 1 - layer for layer in true_layers})
    if kind == "deterministic_random":
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
        generator = random.Random(seed)
        return sorted(generator.sample(range(layer_count), min(len(true_layers), layer_count)))
    raise ValueError(kind)


def summarize_votes(
    rows: list[dict[str, Any]], votes: dict[str, float | None]
) -> dict[str, Any]:
    values = [votes[row["blind_case_id_private"]] for row in rows]
    by_group: dict[str, list[float | None]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        by_group[row["public_parallel_group_id"]].append(value)
    group_accuracies = {key: accuracy(group) for key, group in by_group.items()}
    finite = [value for value in values if value is not None]
    return {
        "case_count": len(values),
        "available_vote_count": len(finite),
        "case_accuracy": clean(accuracy(values)),
        "minimum_group_accuracy": clean(min(group_accuracies.values(), default=0.0)),
        "median_group_accuracy": clean(median(group_accuracies.values()))
        if group_accuracies
        else 0.0,
        "median_vote_margin": clean(median(finite)) if finite else None,
        "group_accuracies": {key: clean(value) for key, value in group_accuracies.items()},
    }


def prediction_assessment(
    cell: dict[str, Any],
    cell_events: list[dict[str, Any]],
    cell_predictions: list[dict[str, Any]],
    protocol: dict[str, Any],
    frozen_best_single_layer: int | None = None,
) -> dict[str, Any]:
    selected_rows = index_selected_rows(cell_events, cell["event_classes"])
    layer_count = cell_events[0]["layer_count"]
    graph_layers = graph_layers_by_group(selected_rows, layer_count)
    rows = sorted(cell_predictions, key=lambda row: row["blind_case_id_private"])

    single_layer_scores = []
    for layer in range(layer_count):
        values = [
            row["target_minus_distractor_margin_by_coordinate"]["query_end"][layer]
            for row in rows
        ]
        single_layer_scores.append(
            {
                "layer_index": layer,
                "case_accuracy": clean(accuracy(values)),
                "median_margin": clean(median(values)),
            }
        )
    if frozen_best_single_layer is None:
        best = max(
            single_layer_scores,
            key=lambda item: (
                item["case_accuracy"],
                item["median_margin"],
                -item["layer_index"],
            ),
        )
        frozen_best_single_layer = best["layer_index"]
        selected_on_discovery = True
    else:
        best = single_layer_scores[frozen_best_single_layer]
        selected_on_discovery = False

    vote_sets: dict[str, dict[str, float | None]] = {
        name: {}
        for name in (
            "graph",
            "best_single_layer",
            "single_peak_layer",
            "wrong_depth",
            "depth_reversal",
            "deterministic_random",
            "wrong_query_label",
        )
    }
    residual_rows = selected_rows["query_residual"]
    for row in rows:
        group_id = row["public_parallel_group_id"]
        case_id = row["blind_case_id_private"]
        margins = row["target_minus_distractor_margin_by_coordinate"]["query_end"]
        layers = graph_layers.get(group_id, [])
        graph_vote = vote(margins, layers)
        vote_sets["graph"][case_id] = graph_vote
        vote_sets["best_single_layer"][case_id] = margins[frozen_best_single_layer]
        peak_layer = residual_rows[group_id]["partial_order_descriptor"]["peak_layer"]
        vote_sets["single_peak_layer"][case_id] = margins[peak_layer]
        for kind in ("wrong_depth", "depth_reversal", "deterministic_random"):
            layers_for_control = control_layers(
                kind,
                layers,
                layer_count,
                f"phase400|{cell['model']}|{cell['surface']}|{group_id}|{kind}",
            )
            vote_sets[kind][case_id] = vote(margins, layers_for_control)
        vote_sets["wrong_query_label"][case_id] = (
            -graph_vote if graph_vote is not None else None
        )
    summaries = {
        name: summarize_votes(rows, values) for name, values in vote_sets.items()
    }
    graph_accuracy = summaries["graph"]["case_accuracy"]
    improvements = {
        "over_best_single_layer": clean(
            graph_accuracy - summaries["best_single_layer"]["case_accuracy"]
        ),
        "over_wrong_depth": clean(
            graph_accuracy - summaries["wrong_depth"]["case_accuracy"]
        ),
        "over_depth_reversal": clean(
            graph_accuracy - summaries["depth_reversal"]["case_accuracy"]
        ),
        "over_deterministic_random": clean(
            graph_accuracy - summaries["deterministic_random"]["case_accuracy"]
        ),
    }
    terminal_node_recovery = min(
        cell["event_classes"][name]["metrics"]["group_interval_pass_rate"]
        for name in ("terminal_route", "terminal_content")
    )
    edge_lookup = {
        (item["source"], item["target"]): item for item in cell["required_edges"]
    }
    terminal_edge_recovery = min(
        edge_lookup[("query_residual", "terminal_route")]["group_pass_rate"],
        edge_lookup[("terminal_route", "terminal_content")]["group_pass_rate"],
    )
    gate = protocol["prediction_contract"]
    checks = {
        "correct_answer_case_accuracy": graph_accuracy
        >= gate["correct_answer_case_accuracy_min"],
        "minimum_group_accuracy": summaries["graph"]["minimum_group_accuracy"]
        >= gate["group_accuracy_min"],
        "next_time_node_recovery": terminal_node_recovery
        >= gate["next_time_node_recovery_min"],
        "next_time_edge_recovery": terminal_edge_recovery
        >= gate["next_time_edge_recovery_min"],
        "improvement_over_best_single_layer": improvements["over_best_single_layer"]
        >= gate["improvement_over_discovery_frozen_best_single_layer_min"],
        "improvement_over_wrong_depth": improvements["over_wrong_depth"]
        >= gate["improvement_over_wrong_depth_min"],
        "improvement_over_depth_reversal": improvements["over_depth_reversal"]
        >= gate["improvement_over_depth_reversal_min"],
        "improvement_over_deterministic_random": improvements[
            "over_deterministic_random"
        ]
        >= gate["improvement_over_deterministic_random_graph_min"],
        "wrong_query_accuracy": summaries["wrong_query_label"]["case_accuracy"]
        <= gate["wrong_query_accuracy_max"],
    }
    return {
        "graph_layer_contract": protocol["prediction_contract"]["graph_query_layers"],
        "group_graph_layers": graph_layers,
        "empty_graph_group_count": sum(not layers for layers in graph_layers.values()),
        "best_single_layer": {
            **best,
            "selected_on_discovery_in_this_run": selected_on_discovery,
            "frozen_layer_index": frozen_best_single_layer,
        },
        "controls": summaries,
        "improvements": improvements,
        "next_time_node_recovery": clean(terminal_node_recovery),
        "next_time_edge_recovery": clean(terminal_edge_recovery),
        "gate_checks": checks,
        "prediction_pass": all(checks.values()),
    }


def pairwise_isomorphism(
    cells: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    gate = protocol["crossmodel_isomorphism_gate"]
    pairs = []
    for left, right in combinations(cells, 2):
        left_relations = left["onset_signature"]["pair_relations"]
        right_relations = right["onset_signature"]["pair_relations"]
        keys = sorted(set(left_relations) & set(right_relations))
        agreement = sum(left_relations[key] == right_relations[key] for key in keys) / max(
            len(keys), 1
        )
        duration_differences = {
            name: abs(
                left["onset_signature"]["nodes"][name]["median_normalized_duration"]
                - right["onset_signature"]["nodes"][name][
                    "median_normalized_duration"
                ]
            )
            for name in protocol["required_nodes"]
        }
        max_duration_difference = max(duration_differences.values(), default=1.0)
        pairs.append(
            {
                "models": [left["model"], right["model"]],
                "onset_order_agreement": clean(agreement),
                "normalized_duration_differences": {
                    key: clean(value) for key, value in duration_differences.items()
                },
                "max_normalized_duration_difference": clean(max_duration_difference),
                "onset_gate_pass": agreement
                >= gate["pairwise_onset_order_agreement_min"],
                "duration_gate_pass": max_duration_difference
                <= gate["pairwise_normalized_duration_difference_max"],
            }
        )
    node_coverage = sum(
        all(name in cell["event_classes"] for name in protocol["required_nodes"])
        for cell in cells
    ) / max(len(cells), 1)
    required_edge_keys = {
        tuple(edge) for edge in protocol["required_edges"]
    }
    edge_coverage = sum(
        {
            (edge["source"], edge["target"], edge["edge_type"])
            for edge in cell["required_edges"]
        }
        >= required_edge_keys
        for cell in cells
    ) / max(len(cells), 1)
    passed = bool(
        len(cells) == len(MODELS)
        and all(cell["partial_order_graph_pass"] for cell in cells)
        and node_coverage >= gate["required_node_type_coverage_min"]
        and edge_coverage >= gate["required_edge_type_coverage_min"]
        and all(pair["onset_gate_pass"] and pair["duration_gate_pass"] for pair in pairs)
    )
    return {
        "model_count": len(cells),
        "passing_graph_model_count": sum(
            cell["partial_order_graph_pass"] for cell in cells
        ),
        "required_node_type_coverage": clean(node_coverage),
        "required_edge_type_coverage": clean(edge_coverage),
        "model_pairs": pairs,
        "crossmodel_functional_isomorphism_pass": passed,
        "identical_physical_coordinates_claimed": False,
    }


def crossmodel_surfaces(
    cells: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for surface in sorted({cell["surface"] for cell in cells}):
        surface_cells = [cell for cell in cells if cell["surface"] == surface]
        result = pairwise_isomorphism(surface_cells, protocol)
        result.update(
            {
                "surface": surface,
                "all_three_prediction_pass": len(surface_cells) == len(MODELS)
                and all(cell["prediction"]["prediction_pass"] for cell in surface_cells),
            }
        )
        output.append(result)
    return output

