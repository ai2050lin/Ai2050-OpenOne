#!/usr/bin/env python3
"""Aggregate Phase419 exact-token matched history-identity differences."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase419_token_matched_history_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase419_token_matched_history_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_network", "reasoning", "grammar", "protocol_control")
INTERFACES = ("chat", "completion")
HISTORIES = ("compatible", "conflict")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "residual_increment", "layer_output")
DEPTHS = ("early", "middle", "late")
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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def behavior_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["model"], row["family_id"], row["interface"], row["history_condition"])].append(row)
    cell_rows = []
    for key, values in sorted(cells.items()):
        target = sum(row["target_event_match"] for row in values)
        cell_rows.append(
            {
                "schema_version": "92.0.0",
                "phase_id": "Phase419-BehaviorCell",
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "interface": key[2],
                "history_condition": key[3],
                "case_count": len(values),
                "target_event_match_count": target,
                "target_event_match_rate": target / len(values),
                "exact_answer_match_count": sum(row["exact_answer_match"] for row in values),
                "right_censored_count": sum(row["right_censored"] for row in values),
                "physical_instrument_qualified": all(row["condition_pass"] for row in values),
                "causal": False,
            }
        )
    lookup = {(row["model"], row["family_id"], row["interface"], row["history_condition"]): row for row in cell_rows}
    effects = []
    for model in MODELS:
        for family in FAMILIES:
            for interface in INTERFACES:
                compatible = lookup[(model, family, interface, "compatible")]
                conflict = lookup[(model, family, interface, "conflict")]
                effects.append(
                    {
                        "model": model,
                        "family_id": family,
                        "interface": interface,
                        "case_count": compatible["case_count"],
                        "compatible_target_event_rate": compatible["target_event_match_rate"],
                        "conflict_target_event_rate": conflict["target_event_match_rate"],
                        "conflict_minus_compatible_target_event_rate": conflict["target_event_match_rate"] - compatible["target_event_match_rate"],
                        "exact_prompt_token_count_matched": True,
                        "descriptive_not_causal": True,
                    }
                )
    return cell_rows, effects


def scalar_regions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["model"], row["family_id"], row["interface"], row["history_condition"], row["component"], row["depth_bin"])
        groups[key]["l2"].append(float(row["l2_norm"]))
        groups[key]["relative"].append(float(row["relative_write_rate"]))
    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                "schema_version": "92.0.0",
                "phase_id": "Phase419-ScalarRegion",
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "interface": key[2],
                "history_condition": key[3],
                "component": key[4],
                "depth_bin": key[5],
                "measurement_count": len(values["l2"]),
                "median_l2_norm": median(values["l2"]),
                "median_relative_write_rate": median(values["relative"]),
                "physical": True,
                "causal": False,
            }
        )
    return output


def contrast_regions(
    contrasts: list[dict[str, Any]], directions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    direction_lookup = {
        (
            row["model"], row["family_id"], row["validation_partition"], row["contrast_type"],
            row["contrast_name"], row["history_interface"], row["component"], row["depth_bin"],
        ): row
        for row in directions
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        partition = "discovery" if row["split"] == "discovery" else "non_discovery"
        key = (
            row["model"], row["family_id"], partition, row["contrast_type"],
            row["contrast_name"], row["history_interface"], row["component"], row["depth_bin"],
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        direction = direction_lookup[key]
        output.append(
            {
                "schema_version": "92.0.0",
                "phase_id": "Phase419-ContrastRegion",
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "validation_partition": key[2],
                "contrast_type": key[3],
                "contrast_name": key[4],
                "history_interface": key[5],
                "component": key[6],
                "depth_bin": key[7],
                "case_count": len(values),
                "median_mean_delta_l2": median([float(row["mean_delta_l2"]) for row in values]),
                "median_mean_relative_delta": median([float(row["mean_relative_delta"]) for row in values]),
                "median_endpoint_cosine": median([
                    float(row["mean_endpoint_cosine"]) for row in values if row["mean_endpoint_cosine"] is not None
                ]),
                "resultant_direction_concentration": float(direction["resultant_direction_concentration"]),
                "mean_pairwise_direction_cosine": direction["mean_pairwise_direction_cosine"],
                "exact_prompt_token_count_matched": key[3] == "history_identity",
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
    return output


def replication(regions: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in regions:
        key = (row["model"], row["family_id"], row["contrast_type"], row["contrast_name"], row["history_interface"])
        groups[key][row["validation_partition"]].append(row)
    rows = []
    for key, partitions in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        left = max(partitions["discovery"], key=lambda row: row["median_mean_relative_delta"])
        right = max(partitions["non_discovery"], key=lambda row: row["median_mean_relative_delta"])
        rows.append(
            {
                "model": key[0],
                "family_id": key[1],
                "contrast_type": key[2],
                "contrast_name": key[3],
                "history_interface": key[4],
                "discovery_region": [left["component"], left["depth_bin"]],
                "non_discovery_region": [right["component"], right["depth_bin"]],
                "exact_region_match": (left["component"], left["depth_bin"]) == (right["component"], right["depth_bin"]),
                "discovery_score": left["median_mean_relative_delta"],
                "non_discovery_score": right["median_mean_relative_delta"],
                "causal": False,
            }
        )
    return {
        "rows": rows,
        "eligible_cell_count": len(rows),
        "exact_region_match_count": sum(row["exact_region_match"] for row in rows),
        "predictive_qualification_authorized": False,
    }


def cross_model(replication_result: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in replication_result["rows"]:
        groups[(row["family_id"], row["contrast_type"], row["contrast_name"], row["history_interface"])].append(row)
    rows = []
    for key, values in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        regions = [tuple(row["non_discovery_region"]) for row in values]
        rows.append(
            {
                "family_id": key[0],
                "contrast_type": key[1],
                "contrast_name": key[2],
                "history_interface": key[3],
                "model_regions": {row["model"]: row["non_discovery_region"] for row in values},
                "all_three_exact_region_match": len(values) == 3 and len(set(regions)) == 1,
                "cross_model_hidden_space_aligned": False,
                "causal": False,
            }
        )
    return {
        "rows": rows,
        "cell_count": len(rows),
        "all_three_exact_region_match_count": sum(row["all_three_exact_region_match"] for row in rows),
    }


def combined_regions(regions: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in regions:
        if row["model"] == model:
            groups[(row["family_id"], row["contrast_type"], row["contrast_name"], row["history_interface"], row["component"], row["depth_bin"])].append(row)
    return [
        {
            "family_id": key[0],
            "contrast_type": key[1],
            "contrast_name": key[2],
            "history_interface": key[3],
            "component": key[4],
            "depth_bin": key[5],
            "median_mean_relative_delta": median([row["median_mean_relative_delta"] for row in values]),
            "resultant_direction_concentration": median([row["resultant_direction_concentration"] for row in values]),
            "case_count": sum(row["case_count"] for row in values),
        }
        for key, values in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0]))
    ]


def build_graph(model: str, regions: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    rows = combined_regions(regions, model)
    max_score = max(row["median_mean_relative_delta"] for row in rows)
    nodes = []
    node_ids = set()
    contrast_slots = {("history_identity", "chat"): -1.4, ("history_identity", "completion"): 0.0, ("interface_interaction", None): 1.4}
    for row in rows:
        score = min(1.0, row["median_mean_relative_delta"] / max(max_score, 1e-8))
        interface_key = row["history_interface"] or "interaction"
        node_id = f"phase419:{model}:{row['family_id']}:{row['contrast_type']}:{interface_key}:{row['component']}:{row['depth_bin']}"
        node_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": f"{row['family_id']} · {interface_key} · {row['component']} · {row['depth_bin']}",
                "type": "event",
                "model": model,
                **row,
                "score": score,
                "size": 0.18 + 0.82 * score,
                "color": FAMILY_COLORS[row["family_id"]],
                "position": [
                    (COMPONENTS.index(row["component"]) - 2) * 4.4 + contrast_slots[(row["contrast_type"], row["history_interface"])],
                    DEPTHS.index(row["depth_bin"]) * 8.0 + score * 2.5,
                    FAMILIES.index(row["family_id"]) * 8.0,
                ],
                "exact_prompt_token_count_matched": row["contrast_type"] == "history_identity",
                "physical": True,
                "predictive": False,
                "causal": False,
                "show_label": score >= 0.72,
            }
        )
    edges = []
    for family in FAMILIES:
        for contrast_type, interface in contrast_slots:
            interface_key = interface or "interaction"
            for component in COMPONENTS:
                for left, right in zip(DEPTHS, DEPTHS[1:]):
                    source = f"phase419:{model}:{family}:{contrast_type}:{interface_key}:{component}:{left}"
                    target = f"phase419:{model}:{family}:{contrast_type}:{interface_key}:{component}:{right}"
                    if source in node_ids and target in node_ids:
                        edges.append({"id": f"{source}->{target}", "source": source, "target": target, "relation": "observed_depth_order", "causal": False})
            if contrast_type == "history_identity" and interface == "chat":
                for component in COMPONENTS:
                    for depth in DEPTHS:
                        source = f"phase419:{model}:{family}:history_identity:chat:{component}:{depth}"
                        target = f"phase419:{model}:{family}:history_identity:completion:{component}:{depth}"
                        if source in node_ids and target in node_ids:
                            edges.append({"id": f"{source}->{target}", "source": source, "target": target, "relation": "same_history_identity_contrast_across_interfaces", "causal": False})
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase419-TokenMatchedHistoryAtlas",
        "title": f"Phase419 {model} 等词元历史身份差分图谱",
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {
            "condition_count": summary["model_condition_count"][model],
            "exact_token_pair_count": summary["model_exact_token_pair_count"][model],
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "visual_encoding": {
            "fixed_axes": ["component", "depth_bin", "family_id", "history_identity_contrast_interface"],
            "node_size": "model_local_relative_vector_delta",
            "node_color": "formal_language_family",
        },
        "evidence_boundary": [
            "Compatible and conflicting histories have identical full prompt token counts within each interface.",
            "The intended manipulation is prior answer identity; token identity necessarily differs.",
            "Nodes are reduced region summaries and edges are observational, not causal.",
        ],
    }


def register_source() -> None:
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase419_token_matched_history_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase419 等词元历史身份差分图谱",
        "description": "相容历史与冲突历史在全提示词元数完全相等时的三模型物理向量差分。",
        "manifest_path": "/vis_data/phase419_token_matched_history_atlas/manifest.json",
        "manifest_schema": "phase419_token_matched_history_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase419_token_matched_history_atlas",
        "models": list(MODELS),
        "evidence_scope": "等词元历史身份观测差分；方向与区域复现不等同于因果机制",
        "color": "#eab308",
    }
    registry["sources"] = [item for item in registry["sources"] if item["id"] != source["id"]] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze() -> dict[str, Any]:
    all_conditions = []
    all_physical = []
    all_contrasts = []
    all_directions = []
    model_conditions = {}
    model_pairs = {}
    model_physical = {}
    for model in MODELS:
        root = OUT / "models" / model
        complete = read_json(root / "phase419_trace_complete.json")
        if not complete["all_conditions_pass"] or complete["exact_prompt_token_count_pair_count"] != 66:
            raise RuntimeError(f"Phase419 model incomplete: {model}")
        conditions = list(read_jsonl(root / "phase419_condition_rows.jsonl"))
        physical = list(read_jsonl(root / "phase419_prefill_physical_rows.jsonl"))
        contrasts = list(read_jsonl(root / "phase419_vector_contrast_rows.jsonl"))
        directions = list(read_jsonl(root / "phase419_direction_consistency_rows.jsonl"))
        all_conditions.extend(conditions)
        all_physical.extend(physical)
        all_contrasts.extend(contrasts)
        all_directions.extend(directions)
        model_conditions[model] = len(conditions)
        model_pairs[model] = complete["exact_prompt_token_count_pair_count"]
        model_physical[model] = len(physical)
    behavior_cells, behavior_effects = behavior_analysis(all_conditions)
    scalar = scalar_regions(all_physical)
    contrasts = contrast_regions(all_contrasts, all_directions)
    replication_result = replication(contrasts)
    cross_model_result = cross_model(replication_result)
    summary = {
        "schema_version": "92.0.0",
        "phase_id": "Phase419-TokenMatchedHistoryAtlasSummary",
        "created_at": now(),
        "valid": True,
        "model_count": 3,
        "cross_model_aligned_semantic_case_count": 33,
        "condition_count": len(all_conditions),
        "condition_pass_count": sum(row["condition_pass"] for row in all_conditions),
        "exact_prompt_token_count_pair_count": sum(model_pairs.values()),
        "target_event_match_count": sum(row["target_event_match"] for row in all_conditions),
        "exact_answer_match_count": sum(row["exact_answer_match"] for row in all_conditions),
        "right_censored_count": sum(row["right_censored"] for row in all_conditions),
        "physical_row_count": len(all_physical),
        "vector_contrast_row_count": len(all_contrasts),
        "direction_consistency_row_count": len(all_directions),
        "scalar_region_cell_count": len(scalar),
        "contrast_region_cell_count": len(contrasts),
        "model_condition_count": model_conditions,
        "model_exact_token_pair_count": model_pairs,
        "model_physical_row_count": model_physical,
        "replication_audit": {key: value for key, value in replication_result.items() if key != "rows"},
        "cross_model_audit": {key: value for key, value in cross_model_result.items() if key != "rows"},
        "strict_mechanism_closure_count": 0,
        "strict_mechanism_denominator": 72,
        "authorization": {
            "publish_token_matched_history_atlas": True,
            "claim_history_identity_mechanism": False,
            "claim_predictive_state": False,
            "run_causal_intervention_from_phase419_alone": False,
            "run_single_neuron_scan_from_phase419_alone": False,
        },
        "evidence_boundary": [
            "Full prompt token count, history structure, current task and interface serialization are matched within each direct pair.",
            "Compatible and conflicting prior answer token identities differ by design.",
            "Task-family labels remain external labels rather than learned internal functional names.",
            "Region and direction replication is descriptive and does not establish necessity, sufficiency or mediation.",
        ],
    }
    write_jsonl(OUT / "phase419_behavior_cells.jsonl", behavior_cells)
    write_json(OUT / "phase419_behavior_identity_effects.json", {"rows": behavior_effects})
    write_jsonl(OUT / "phase419_scalar_region_rows.jsonl", scalar)
    write_jsonl(OUT / "phase419_contrast_region_rows.jsonl", contrasts)
    write_json(OUT / "phase419_holdout_replication.json", replication_result)
    write_json(OUT / "phase419_cross_model_layout.json", cross_model_result)
    write_json(OUT / "phase419_global_summary.json", summary)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        graph = build_graph(model, contrasts, summary)
        filename = f"phase419_{model}_token_matched_history_atlas.json"
        write_json(PUBLIC / filename, graph)
        items.append({
            "id": f"phase419_{model}_token_matched_history_atlas",
            "label": f"Phase419 {model} 等词元历史身份差分图谱",
            "filename": filename,
            "model": model,
            "phase": 419,
            "evidence_scope": "exact-token matched prior-answer identity physical difference; non-causal",
        })
    write_json(PUBLIC / "manifest.json", {
        "schema_version": "phase419_token_matched_history_atlas_manifest.v1",
        "generated_at": now(),
        "default_item_id": items[0]["id"],
        "items": items,
    })
    write_json(PUBLIC / "phase419_global_summary.json", summary)
    register_source()
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, allow_nan=False))
