#!/usr/bin/env python3
"""Aggregate Phase418 paired differences into a cautious physical atlas."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase418_interface_history_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase418_interface_history_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_network", "reasoning", "grammar", "protocol_control")
INTERFACES = ("chat", "completion")
HISTORIES = ("none", "compatible", "irrelevant", "conflict", "override")
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


def median_mad(values: list[float]) -> tuple[float, float]:
    center = median(values)
    mad = median([abs(value - center) for value in values])
    return center, mad


def validation_split(split: str) -> str:
    return "discovery" if split == "discovery" else "non_discovery"


def behavior_cells(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        groups[(row["model"], row["family_id"], row["interface"], row["history_condition"])].append(row)
    output = []
    for key, values in sorted(groups.items()):
        target_count = sum(row["target_event_match"] for row in values)
        exact_count = sum(row["exact_answer_match"] for row in values)
        rate = target_count / len(values)
        output.append(
            {
                "schema_version": "92.0.0",
                "phase_id": "Phase418-BehaviorQualificationCell",
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "interface": key[2],
                "history_condition": key[3],
                "case_count": len(values),
                "target_event_match_count": target_count,
                "target_event_match_rate": rate,
                "exact_answer_match_count": exact_count,
                "exact_answer_match_rate": exact_count / len(values),
                "right_censored_count": sum(row["right_censored"] for row in values),
                "functional_cell_qualified": len(values) >= 10 and rate >= 0.70,
                "override_is_composite": key[3] == "override",
                "causal": False,
            }
        )
    return output


def standardized_regions(physical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in physical_rows:
        if row["history_condition"] == "none":
            baselines[(row["model"], row["family_id"], int(row["layer"]), row["component"])].append(float(row["l2_norm"]))
    baseline_stats = {key: median_mad(values) for key, values in baselines.items()}
    groups: dict[tuple[str, str, str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in physical_rows:
        key = (row["model"], row["family_id"], row["interface"], row["history_condition"], row["component"], row["depth_bin"])
        center, mad = baseline_stats[(row["model"], row["family_id"], int(row["layer"]), row["component"])]
        scale = max(1.4826 * mad, abs(center) * 1e-6, 1e-8)
        z = (float(row["l2_norm"]) - center) / scale
        groups[key]["l2"].append(float(row["l2_norm"]))
        groups[key]["z"].append(z)
        groups[key]["abs_z"].append(abs(z))
        groups[key]["relative_write"].append(float(row["relative_write_rate"]))

    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                "schema_version": "92.0.0",
                "phase_id": "Phase418-RobustStandardizedRegion",
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "interface": key[2],
                "history_condition": key[3],
                "component": key[4],
                "depth_bin": key[5],
                "measurement_count": len(values["l2"]),
                "median_l2_norm": median(values["l2"]),
                "median_signed_robust_z": median(values["z"]),
                "median_abs_robust_z": median(values["abs_z"]),
                "median_relative_write_rate": median(values["relative_write"]),
                "baseline": "both_interface_none_history_same_model_family_layer_component",
                "scale": "median_absolute_deviation_with_epsilon_floor",
                "physical": True,
                "causal": False,
            }
        )
    return output


def aggregate_contrasts(
    contrast_rows: list[dict[str, Any]],
    direction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    directions = {
        (
            row["model"], row["family_id"], row["validation_partition"], row["contrast_type"],
            row["contrast_name"], row["history_interface"], row["component"], row["depth_bin"],
        ): row
        for row in direction_rows
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in contrast_rows:
        key = (
            row["model"], row["family_id"], validation_split(row["split"]), row["contrast_type"],
            row["contrast_name"], row["history_interface"], row["component"], row["depth_bin"],
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        direction = directions.get(key)
        output.append(
            {
                "schema_version": "92.0.0",
                "phase_id": "Phase418-ContrastRegionSummary",
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
                "median_dominant_layer_relative_delta": median([float(row["dominant_layer_relative_delta"]) for row in values]),
                "median_endpoint_cosine": median([
                    float(row["mean_endpoint_cosine"]) for row in values if row["mean_endpoint_cosine"] is not None
                ]),
                "near_length_case_count": sum(row["token_length_stratum"] in ("exact", "near_2") for row in values),
                "far_length_case_count": sum(row["token_length_stratum"] == "far" for row in values),
                "resultant_direction_concentration": (
                    float(direction["resultant_direction_concentration"]) if direction else None
                ),
                "mean_pairwise_direction_cosine": (
                    direction["mean_pairwise_direction_cosine"] if direction else None
                ),
                "composite_override_contrast": key[4] == "override",
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
    return output


def dominance_audit(region_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in region_rows:
        groups[(row["model"], row["family_id"], row["interface"], row["history_condition"], row["component"])].append(row)
    rows = []
    for key, values in sorted(groups.items()):
        absolute = max(values, key=lambda row: row["median_l2_norm"])
        standardized = max(values, key=lambda row: row["median_abs_robust_z"])
        relative = max(values, key=lambda row: row["median_relative_write_rate"])
        rows.append(
            {
                "model": key[0],
                "family_id": key[1],
                "interface": key[2],
                "history_condition": key[3],
                "component": key[4],
                "absolute_norm_dominant_depth": absolute["depth_bin"],
                "standardized_effect_dominant_depth": standardized["depth_bin"],
                "relative_write_dominant_depth": relative["depth_bin"],
                "absolute_and_standardized_depth_match": absolute["depth_bin"] == standardized["depth_bin"],
                "causal": False,
            }
        )
    return {
        "rows": rows,
        "cell_count": len(rows),
        "absolute_late_count": sum(row["absolute_norm_dominant_depth"] == "late" for row in rows),
        "standardized_late_count": sum(row["standardized_effect_dominant_depth"] == "late" for row in rows),
        "relative_write_late_count": sum(row["relative_write_dominant_depth"] == "late" for row in rows),
        "absolute_standardized_match_count": sum(row["absolute_and_standardized_depth_match"] for row in rows),
        "interpretation": "absolute norm depth is a scale audit; standardized and relative-write depths are descriptive condition effects",
    }


def length_audit(contrast_rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in contrast_rows if row["contrast_type"] != "interaction"]
    strata = Counter(row["token_length_stratum"] for row in paired)
    # Each case-level contrast is repeated across 15 component-depth cells.
    unique = {
        (row["model"], row["semantic_case_id"], row["contrast_type"], row["contrast_name"], row["history_interface"], row["prompt_token_count_delta"])
        for row in paired
    }
    unique_strata = Counter(
        "exact" if abs(item[-1]) == 0 else "near_2" if abs(item[-1]) <= 2 else "near_8" if abs(item[-1]) <= 8 else "far"
        for item in unique
    )
    return {
        "paired_contrast_region_row_count": len(paired),
        "region_row_strata": dict(sorted(strata.items())),
        "unique_paired_contrast_count": len(unique),
        "unique_pair_strata": dict(sorted(unique_strata.items())),
        "exact_or_near_2_fraction": (
            (unique_strata["exact"] + unique_strata["near_2"]) / len(unique) if unique else 0.0
        ),
        "literal_token_level_orthogonality_claim_authorized": False,
        "interface_history_mechanism_claim_authorized": False,
        "reason": "prompt length and role serialization remain part of each measured condition",
    }


def replication_audit(contrast_regions: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in contrast_regions:
        key = (
            row["model"], row["family_id"], row["contrast_type"], row["contrast_name"], row["history_interface"]
        )
        groups[key][row["validation_partition"]].append(row)
    rows = []
    for key, partitions in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        discovery = partitions.get("discovery", [])
        validation = partitions.get("non_discovery", [])
        if not discovery or not validation:
            continue
        left = max(discovery, key=lambda row: row["median_mean_relative_delta"])
        right = max(validation, key=lambda row: row["median_mean_relative_delta"])
        rows.append(
            {
                "model": key[0],
                "family_id": key[1],
                "contrast_type": key[2],
                "contrast_name": key[3],
                "history_interface": key[4],
                "discovery_top_component": left["component"],
                "discovery_top_depth": left["depth_bin"],
                "non_discovery_top_component": right["component"],
                "non_discovery_top_depth": right["depth_bin"],
                "component_match": left["component"] == right["component"],
                "depth_match": left["depth_bin"] == right["depth_bin"],
                "exact_region_match": (left["component"], left["depth_bin"]) == (right["component"], right["depth_bin"]),
                "discovery_score": left["median_mean_relative_delta"],
                "non_discovery_score": right["median_mean_relative_delta"],
                "causal": False,
            }
        )
    return {
        "rows": rows,
        "eligible_cell_count": len(rows),
        "component_match_count": sum(row["component_match"] for row in rows),
        "depth_match_count": sum(row["depth_match"] for row in rows),
        "exact_region_match_count": sum(row["exact_region_match"] for row in rows),
        "predictive_qualification_authorized": False,
        "reason": "descriptive top-region replication is not held-out behavior prediction",
    }


def cross_model_audit(replication: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in replication["rows"]:
        key = (row["family_id"], row["contrast_type"], row["contrast_name"], row["history_interface"])
        groups[key].append(row)
    rows = []
    for key, values in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        if len(values) != 3:
            continue
        regions = [(row["non_discovery_top_component"], row["non_discovery_top_depth"]) for row in values]
        depths = [region[1] for region in regions]
        components = [region[0] for region in regions]
        rows.append(
            {
                "family_id": key[0],
                "contrast_type": key[1],
                "contrast_name": key[2],
                "history_interface": key[3],
                "model_regions": {row["model"]: [row["non_discovery_top_component"], row["non_discovery_top_depth"]] for row in values},
                "all_three_component_match": len(set(components)) == 1,
                "all_three_depth_match": len(set(depths)) == 1,
                "all_three_exact_region_match": len(set(regions)) == 1,
                "cross_model_hidden_space_aligned": False,
                "causal": False,
            }
        )
    return {
        "rows": rows,
        "cell_count": len(rows),
        "all_three_component_match_count": sum(row["all_three_component_match"] for row in rows),
        "all_three_depth_match_count": sum(row["all_three_depth_match"] for row in rows),
        "all_three_exact_region_match_count": sum(row["all_three_exact_region_match"] for row in rows),
    }


def build_graph(model: str, regions: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in regions if row["model"] == model]
    max_z = max((row["median_abs_robust_z"] for row in rows), default=1.0)
    nodes = []
    node_ids = set()
    for row in rows:
        score = min(1.0, row["median_abs_robust_z"] / max(max_z, 1e-8))
        node_id = ":".join(
            ["phase418", model, row["family_id"], row["interface"], row["history_condition"], row["component"], row["depth_bin"]]
        )
        node_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": f"{row['family_id']} · {row['interface']} · {row['history_condition']} · {row['component']} · {row['depth_bin']}",
                "type": "event",
                "model": model,
                "family_id": row["family_id"],
                "interface": row["interface"],
                "history_condition": row["history_condition"],
                "component": row["component"],
                "depth_bin": row["depth_bin"],
                "score": score,
                "size": 0.16 + 0.84 * score,
                "color": FAMILY_COLORS[row["family_id"]],
                "position": [
                    (COMPONENTS.index(row["component"]) - 2) * 4.2 + (INTERFACES.index(row["interface"]) - 0.5) * 1.4,
                    DEPTHS.index(row["depth_bin"]) * 8.0 + score * 2.0,
                    FAMILIES.index(row["family_id"]) * 9.0 + (HISTORIES.index(row["history_condition"]) - 2) * 1.25,
                ],
                "median_l2_norm": row["median_l2_norm"],
                "median_abs_robust_z": row["median_abs_robust_z"],
                "median_relative_write_rate": row["median_relative_write_rate"],
                "measurement_count": row["measurement_count"],
                "composite_override_condition": row["history_condition"] == "override",
                "physical": True,
                "predictive": False,
                "causal": False,
                "show_label": score >= 0.78,
            }
        )
    edges = []
    for family in FAMILIES:
        for interface in INTERFACES:
            for history in HISTORIES:
                for component in COMPONENTS:
                    for left, right in zip(DEPTHS, DEPTHS[1:]):
                        source = f"phase418:{model}:{family}:{interface}:{history}:{component}:{left}"
                        target = f"phase418:{model}:{family}:{interface}:{history}:{component}:{right}"
                        if source in node_ids and target in node_ids:
                            edges.append({"id": f"{source}->{target}", "source": source, "target": target, "relation": "observed_depth_order", "causal": False})
                    for depth in DEPTHS:
                        for history_value in HISTORIES[1:]:
                            source = f"phase418:{model}:{family}:{interface}:none:{component}:{depth}"
                            target = f"phase418:{model}:{family}:{interface}:{history_value}:{component}:{depth}"
                            if source in node_ids and target in node_ids:
                                edges.append({"id": f"{source}->{target}", "source": source, "target": target, "relation": "registered_history_contrast", "causal": False})
        for history in HISTORIES:
            for component in COMPONENTS:
                for depth in DEPTHS:
                    source = f"phase418:{model}:{family}:chat:{history}:{component}:{depth}"
                    target = f"phase418:{model}:{family}:completion:{history}:{component}:{depth}"
                    if source in node_ids and target in node_ids:
                        edges.append({"id": f"{source}->{target}", "source": source, "target": target, "relation": "registered_interface_contrast", "causal": False})
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase418-InterfaceHistoryPhysicalAtlas",
        "title": f"Phase418 {model} 接口—历史动态物理图谱",
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {
            "source_condition_count": summary["model_condition_count"][model],
            "source_physical_row_count": summary["model_physical_row_count"][model],
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "visual_encoding": {
            "fixed_axes": ["component", "depth_bin", "family_id", "interface", "history_condition"],
            "node_size": "model_local_median_absolute_robust_z",
            "node_color": "formal_language_family",
            "edge_meaning": "registered paired contrast or architectural depth order; never causal",
        },
        "evidence_boundary": [
            "Nodes are reduced current-position prompt-prefill measurements, not full activation tensors.",
            "Chat and completion serializations have recorded token-length and role-marker differences.",
            "Override is a composite manipulation.",
            "High robust-z and directional consistency are descriptive and do not authorize mechanism or neuron claims.",
        ],
    }


def register_public_source() -> None:
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase418_interface_history_atlas",
        "label": "GPT5 Phase418 接口—历史动态物理图谱",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "description": "三模型在接口与历史配对条件下的提示前向稳健标准化物理差分。",
        "manifest_path": "/vis_data/phase418_interface_history_atlas/manifest.json",
        "manifest_schema": "phase418_interface_history_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase418_interface_history_atlas",
        "models": ["qwen3", "glm4", "deepseek7b"],
        "evidence_scope": "接口—历史配对观测差分；提示长度与角色序列化未正交，非功能、因果或神经元机制",
        "color": "#14b8a6",
    }
    items = registry["sources"]
    items = [item for item in items if item["id"] != source["id"]]
    items.append(source)
    registry["sources"] = items
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze() -> dict[str, Any]:
    all_cases: list[dict[str, Any]] = []
    all_physical: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    all_directions: list[dict[str, Any]] = []
    model_conditions = {}
    model_physical = {}
    model_contrasts = {}
    model_targets = {}
    for model in MODELS:
        root = OUT / "models" / model
        complete = read_json(root / "phase418_trace_complete.json")
        if not complete["all_conditions_pass"] or complete["condition_count"] != 400:
            raise RuntimeError(f"Phase418 trace incomplete for {model}: {complete}")
        cases = list(read_jsonl(root / "phase418_condition_rows.jsonl"))
        physical = list(read_jsonl(root / "phase418_prefill_physical_rows.jsonl"))
        contrasts = list(read_jsonl(root / "phase418_vector_contrast_rows.jsonl"))
        directions = list(read_jsonl(root / "phase418_direction_consistency_rows.jsonl"))
        all_cases.extend(cases)
        all_physical.extend(physical)
        all_contrasts.extend(contrasts)
        all_directions.extend(directions)
        model_conditions[model] = len(cases)
        model_physical[model] = len(physical)
        model_contrasts[model] = len(contrasts)
        model_targets[model] = sum(row["target_event_match"] for row in cases)

    behavior = behavior_cells(all_cases)
    standardized = standardized_regions(all_physical)
    contrast_regions = aggregate_contrasts(all_contrasts, all_directions)
    dominance = dominance_audit(standardized)
    lengths = length_audit(all_contrasts)
    replication = replication_audit(contrast_regions)
    cross_model = cross_model_audit(replication)
    summary = {
        "schema_version": "92.0.0",
        "phase_id": "Phase418-InterfaceHistoryAtlasSummary",
        "created_at": now(),
        "valid": True,
        "model_count": len(MODELS),
        "semantic_case_count": 120,
        "cross_model_aligned_semantic_case_count": 40,
        "condition_count": len(all_cases),
        "condition_pass_count": sum(row["condition_pass"] for row in all_cases),
        "target_event_match_count": sum(row["target_event_match"] for row in all_cases),
        "exact_answer_match_count": sum(row["exact_answer_match"] for row in all_cases),
        "right_censored_count": sum(row["right_censored"] for row in all_cases),
        "physical_row_count": len(all_physical),
        "vector_contrast_row_count": len(all_contrasts),
        "direction_consistency_row_count": len(all_directions),
        "behavior_cell_count": len(behavior),
        "functional_behavior_cell_qualified_count": sum(row["functional_cell_qualified"] for row in behavior),
        "standardized_region_cell_count": len(standardized),
        "contrast_region_cell_count": len(contrast_regions),
        "model_condition_count": model_conditions,
        "model_target_event_match_count": model_targets,
        "model_physical_row_count": model_physical,
        "model_vector_contrast_row_count": model_contrasts,
        "dominance_audit": {key: value for key, value in dominance.items() if key != "rows"},
        "length_audit": {key: value for key, value in lengths.items() if key != "rows"},
        "replication_audit": {key: value for key, value in replication.items() if key != "rows"},
        "cross_model_audit": {key: value for key, value in cross_model.items() if key != "rows"},
        "strict_mechanism_closure_count": 0,
        "strict_mechanism_denominator": 72,
        "authorization": {
            "publish_paired_physical_atlas": True,
            "publish_interface_history_mechanism": False,
            "publish_predictive_state": False,
            "run_causal_intervention_from_phase418_alone": False,
            "run_single_neuron_scan_from_phase418_alone": False,
        },
        "evidence_boundary": [
            "Natural computation is measured on a frozen formal-world denominator, not externally validated natural language.",
            "All-layer data are reduced current-position component measurements plus selected exact anchor vectors.",
            "Interface and history conditions are paired but not token-length orthogonal.",
            "Discovery/non-discovery region replication is descriptive, not held-out behavior prediction.",
            "Small-model agreement can reveal architecture-level regularities but cannot establish a universal language code.",
        ],
    }
    write_jsonl(OUT / "phase418_behavior_cells.jsonl", behavior)
    write_jsonl(OUT / "phase418_standardized_region_rows.jsonl", standardized)
    write_jsonl(OUT / "phase418_contrast_region_rows.jsonl", contrast_regions)
    write_json(OUT / "phase418_depth_dominance_audit.json", dominance)
    write_json(OUT / "phase418_length_position_audit.json", lengths)
    write_json(OUT / "phase418_holdout_replication.json", replication)
    write_json(OUT / "phase418_cross_model_layout.json", cross_model)
    write_json(OUT / "phase418_global_summary.json", summary)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for model in MODELS:
        graph = build_graph(model, standardized, summary)
        filename = f"phase418_{model}_interface_history_physical_atlas.json"
        write_json(PUBLIC / filename, graph)
        manifest_items.append(
            {
                "id": f"phase418_{model}_interface_history_physical_atlas",
                "label": f"Phase418 {model} 接口—历史动态物理图谱",
                "filename": filename,
                "model": model,
                "phase": 418,
                "evidence_scope": "paired interface-history prompt-prefill physical differences; non-causal",
            }
        )
    write_json(
        PUBLIC / "manifest.json",
        {
            "schema_version": "phase418_interface_history_atlas_manifest.v1",
            "generated_at": now(),
            "default_item_id": manifest_items[0]["id"],
            "items": manifest_items,
        },
    )
    write_json(PUBLIC / "phase418_global_summary.json", summary)
    register_public_source()
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, allow_nan=False))
