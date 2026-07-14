#!/usr/bin/env python3
"""Build the Phase416 observer-free prefill physical atlas.

The reducer intentionally uses only counts, means, bounded pair differences,
and exact region matches.  It does not train a probe or infer causal edges.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from phase416_dual_track_case_bank import (
    MODELS,
    OUT,
    PHASE_ID,
    SCHEMA_VERSION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase416_prefill_physical_trace import CORE_COMPONENTS


PUBLIC = Path(__file__).resolve().parents[2] / "frontend/public/vis_data/phase416_formal_prefill_atlas"
FAMILIES = ("knowledge_network", "reasoning", "grammar", "protocol_control")
ROLES = ("source", "query", "answer_start")
DEPTHS = ("early", "middle", "late")
FAMILY_COLORS = {
    "knowledge_network": "#22c55e",
    "reasoning": "#f59e0b",
    "grammar": "#38bdf8",
    "protocol_control": "#a78bfa",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def bounded_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left) + abs(right), 1e-8)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_difference(
    rows: list[tuple[tuple[str, str, str, str, str], float]],
    contrast_type: str,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for key, value in rows:
        buckets[key].append(value)
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "contrast_type": contrast_type,
            "model": key[0],
            "family_id": key[1],
            "component": key[2],
            "depth_bin": key[3],
            "position_role": key[4],
            "pair_cell_count": len(values),
            "mean_bounded_l2_difference": mean(values),
            "observer_id": None,
            "causal": False,
        }
        for key, values in sorted(buckets.items())
    ]


def build_graph(model: str, region_rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    model_rows = [row for row in region_rows if row["model"] == model]
    nodes = []
    node_ids: set[str] = set()
    component_index = {value: index for index, value in enumerate(CORE_COMPONENTS)}
    family_index = {value: index for index, value in enumerate(FAMILIES)}
    role_offset = {"source": -1.0, "query": 0.0, "answer_start": 1.0}
    depth_index = {"early": 0, "middle": 1, "late": 2}
    for row in model_rows:
        relative_norm = max(0.0, min(1.0, row["within_component_relative_norm"]))
        local_strength_offset = relative_norm - 0.5
        node_id = (
            f"phase416:{model}:{row['family_id']}:{row['component']}:"
            f"{row['depth_bin']}:{row['position_role']}"
        )
        node_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": (
                    f"{row['family_id']} · {row['component']} · "
                    f"{row['depth_bin']} · {row['position_role']}"
                ),
                "type": "cluster",
                "model": model,
                "family_id": row["family_id"],
                "component": row["component"],
                "depth_bin": row["depth_bin"],
                "position_role": row["position_role"],
                "score": relative_norm,
                "size": 0.10 + 0.90 * relative_norm,
                "color": FAMILY_COLORS[row["family_id"]],
                "position": [
                    (component_index[row["component"]] - 2) * 4.0
                    + local_strength_offset * 1.4,
                    depth_index[row["depth_bin"]] * 8.0
                    + local_strength_offset * 4.0,
                    family_index[row["family_id"]] * 5.0 + role_offset[row["position_role"]],
                ],
                "case_count": row["case_count"],
                "mean_l2_norm": row["mean_l2_norm"],
                "formal_behavior_qualified": row["formal_behavior_qualified"],
                "evidence_level": (
                    "observer_free_formal_conditioned_physical_distribution"
                    if row["formal_behavior_qualified"]
                    else "observer_free_unqualified_condition_physical_distribution"
                ),
                "observer_id": None,
                "natural_prefill": True,
                "generation_time": False,
                "causal": False,
                "show_label": bool(
                    row["dominant_region_for_component"] and row["position_role"] == "query"
                ),
            }
        )
    edges = []
    for family in FAMILIES:
        for component in CORE_COMPONENTS:
            for role in ROLES:
                for left, right in zip(DEPTHS, DEPTHS[1:]):
                    source = f"phase416:{model}:{family}:{component}:{left}:{role}"
                    target = f"phase416:{model}:{family}:{component}:{right}:{role}"
                    if source in node_ids and target in node_ids:
                        edges.append(
                            {
                                "id": f"{source}->{target}",
                                "source": source,
                                "target": target,
                                "relation": "observed_depth_continuity",
                                "causal": False,
                                "evidence_boundary": "aggregated prefill depth order only",
                            }
                        )
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": PHASE_ID,
        "title": f"Phase416 {model} 形式世界前向物理分布",
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {
            "source_case_count": summary["model_physical_case_count"][model],
            "source_physical_row_count": summary["model_physical_row_count"][model],
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "visual_encoding": {
            "fixed_axes": ["component", "depth_bin", "family_id", "position_role"],
            "local_displacement": "within_component_relative_norm",
            "node_size": "within_component_relative_norm",
            "cross_model_scale": "model_local_only_not_absolute_cross_model_magnitude",
        },
        "evidence_boundary": [
            "Observer-free reduced prefill measurements only.",
            "Family labels describe formal input conditions, not decoded internal semantics.",
            "Depth-continuity edges are observational and non-causal.",
            "KV-cached generation-time instrumentation did not qualify in Phase416.",
        ],
    }


def analyze() -> dict[str, Any]:
    qualification = read_json(OUT / "phase416_instrument_domain_qualification.json")
    case_meta: dict[tuple[str, str], dict[str, Any]] = {}
    value_index: dict[tuple[str, str, int, str, str], float] = {}
    aggregate: dict[tuple[str, str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "norm_sum": 0.0, "qualified_count": 0.0}
    )
    model_row_count: Counter[str] = Counter()
    model_case_count: dict[str, int] = {}
    model_layer_count: dict[str, int] = {}
    model_anchor_count: dict[str, int] = {}

    for model in MODELS:
        complete = read_json(OUT / "models" / model / "phase416_prefill_physical_complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid prefill physical collection for {model}")
        model_case_count[model] = complete["case_count"]
        model_layer_count[model] = complete["layer_count"]
        model_anchor_count[model] = complete["lossless_anchor_vector_count"]
        path = OUT / "models" / model / "phase416_prefill_physical_rows.jsonl"
        for row in read_rows(path):
            model_row_count[model] += 1
            if row["component"] not in CORE_COMPONENTS or row["position_role"] not in ROLES:
                continue
            key = (model, row["case_id"], int(row["layer"]), row["component"], row["position_role"])
            value_index[key] = float(row["l2_norm"])
            case_meta[(model, row["case_id"])] = row
            bucket_key = (
                model,
                row["family_id"],
                row["component"],
                row["depth_bin"],
                row["position_role"],
            )
            bucket = aggregate[bucket_key]
            bucket["count"] += 1
            bucket["norm_sum"] += float(row["l2_norm"])
            bucket["qualified_count"] += bool(row["formal_behavior_qualified"])

    region_rows = []
    component_max: dict[tuple[str, str, str], float] = defaultdict(float)
    for key, bucket in aggregate.items():
        value = bucket["norm_sum"] / bucket["count"]
        component_max[key[:3]] = max(component_max[key[:3]], value)
    dominant_key: dict[tuple[str, str, str], tuple[str, str]] = {}
    for key, bucket in aggregate.items():
        value = bucket["norm_sum"] / bucket["count"]
        group = key[:3]
        relative = value / max(component_max[group], 1e-8)
        current = dominant_key.get(group)
        if current is None:
            dominant_key[group] = (key[3], key[4])
        else:
            current_value = aggregate[(*group, current[0], current[1])]["norm_sum"] / aggregate[(*group, current[0], current[1])]["count"]
            if value > current_value:
                dominant_key[group] = (key[3], key[4])
        region_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "component": key[2],
                "depth_bin": key[3],
                "position_role": key[4],
                "case_count": int(bucket["count"]),
                "mean_l2_norm": value,
                "within_component_relative_norm": relative,
                "formal_behavior_qualified": bucket["qualified_count"] == bucket["count"],
                "dominant_region_for_component": False,
                "observer_id": None,
                "causal": False,
            }
        )
    for row in region_rows:
        row["dominant_region_for_component"] = (
            row["depth_bin"], row["position_role"]
        ) == dominant_key[(row["model"], row["family_id"], row["component"])]

    # Pair cases by frozen external factors.  These are activity differences,
    # not claims that the changed factor has been internally isolated.
    cases_by_surface: dict[tuple[str, str, int, str], list[str]] = defaultdict(list)
    cases_by_state_lexical: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for (model, case_id), meta in case_meta.items():
        cases_by_surface[(model, meta["mechanism_id"], int(meta["item_index"]), meta["split"])].append(case_id)
        cases_by_state_lexical[(model, meta["mechanism_id"], meta["template_id"])].append(case_id)

    surface_values = []
    for (model, _mechanism, _item, _split), case_ids in cases_by_surface.items():
        unique = sorted(set(case_ids))
        if len(unique) < 2:
            continue
        left, right = unique[0], unique[1]
        meta = case_meta[(model, left)]
        for layer in range(model_layer_count[model]):
            depth = "early" if layer / max(1, model_layer_count[model] - 1) < 1 / 3 else "middle" if layer / max(1, model_layer_count[model] - 1) < 2 / 3 else "late"
            for component in CORE_COMPONENTS:
                for role in ROLES:
                    left_value = value_index.get((model, left, layer, component, role))
                    right_value = value_index.get((model, right, layer, component, role))
                    if left_value is not None and right_value is not None:
                        surface_values.append(((model, meta["family_id"], component, depth, role), bounded_difference(left_value, right_value)))

    state_values = []
    for (model, _mechanism, _template), case_ids in cases_by_state_lexical.items():
        unique = sorted(set(case_ids), key=lambda case_id: int(case_meta[(model, case_id)]["item_index"]))
        if len(unique) < 2:
            continue
        left, right = unique[0], unique[-1]
        meta = case_meta[(model, left)]
        for layer in range(model_layer_count[model]):
            depth = "early" if layer / max(1, model_layer_count[model] - 1) < 1 / 3 else "middle" if layer / max(1, model_layer_count[model] - 1) < 2 / 3 else "late"
            for component in CORE_COMPONENTS:
                for role in ROLES:
                    left_value = value_index.get((model, left, layer, component, role))
                    right_value = value_index.get((model, right, layer, component, role))
                    if left_value is not None and right_value is not None:
                        state_values.append(((model, meta["family_id"], component, depth, role), bounded_difference(left_value, right_value)))

    role_values = []
    for (model, case_id), meta in case_meta.items():
        for layer in range(model_layer_count[model]):
            depth = "early" if layer / max(1, model_layer_count[model] - 1) < 1 / 3 else "middle" if layer / max(1, model_layer_count[model] - 1) < 2 / 3 else "late"
            for component in CORE_COMPONENTS:
                source = value_index.get((model, case_id, layer, component, "source"))
                query = value_index.get((model, case_id, layer, component, "query"))
                if source is not None and query is not None:
                    role_values.append(((model, meta["family_id"], component, depth, "query_vs_source"), bounded_difference(source, query)))

    surface_rows = aggregate_difference(surface_values, "surface_template_pair")
    state_rows = aggregate_difference(state_values, "state_plus_lexical_instance_pair")
    role_rows = aggregate_difference(role_values, "query_vs_source_role")

    dominant_rows = [row for row in region_rows if row["dominant_region_for_component"]]
    cross_model_consensus = []
    for family in FAMILIES:
        for component in CORE_COMPONENTS:
            values = [
                row for row in dominant_rows
                if row["family_id"] == family and row["component"] == component
            ]
            counts = Counter((row["depth_bin"], row["position_role"]) for row in values)
            best_region, best_count = counts.most_common(1)[0]
            cross_model_consensus.append(
                {
                    "family_id": family,
                    "component": component,
                    "dominant_region": {"depth_bin": best_region[0], "position_role": best_region[1]},
                    "matching_model_count": best_count,
                    "all_three_models_match": best_count == 3,
                    "at_least_two_models_match": best_count >= 2,
                    "causal": False,
                }
            )

    cross_family_reuse = []
    for model in MODELS:
        for component in CORE_COMPONENTS:
            values = [
                row for row in dominant_rows
                if row["model"] == model and row["component"] == component
            ]
            counts = Counter((row["depth_bin"], row["position_role"]) for row in values)
            region, count = counts.most_common(1)[0]
            cross_family_reuse.append(
                {
                    "model": model,
                    "component": component,
                    "dominant_region": {"depth_bin": region[0], "position_role": region[1]},
                    "matching_family_count": count,
                    "all_four_families_match": count == 4,
                    "layout_overlap_only": True,
                    "mechanism_reuse_proven": False,
                }
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase416-PrefillPhysicalAtlasSummary",
        "created_at": now(),
        "valid": all(model_case_count[model] == 55 for model in MODELS),
        "model_count": len(MODELS),
        "physical_case_count": sum(model_case_count.values()),
        "model_physical_case_count": model_case_count,
        "physical_row_count": sum(model_row_count.values()),
        "model_physical_row_count": dict(model_row_count),
        "lossless_anchor_vector_count": sum(model_anchor_count.values()),
        "model_lossless_anchor_vector_count": model_anchor_count,
        "region_cell_count": len(region_rows),
        "dominant_region_cell_count": len(dominant_rows),
        "surface_pair_measurement_count": len(surface_values),
        "state_lexical_pair_measurement_count": len(state_values),
        "query_source_role_measurement_count": len(role_values),
        "cross_model_component_family_cell_count": len(cross_model_consensus),
        "cross_model_at_least_two_layout_match_count": sum(row["at_least_two_models_match"] for row in cross_model_consensus),
        "cross_model_all_three_layout_match_count": sum(row["all_three_models_match"] for row in cross_model_consensus),
        "cross_family_layout_overlap_cell_count": len(cross_family_reuse),
        "cross_family_all_four_layout_match_count": sum(row["all_four_families_match"] for row in cross_family_reuse),
        "instrument_domain": qualification["results"],
        "unavailable_factor_axes": [
            "interface_contrast_not_present_in_frozen_55_case_denominator",
            "history_contrast_not_present_in_frozen_55_case_denominator",
            "generation_time_events_blocked_by_incremental_instrument_failure",
        ],
        "evidence_boundary": [
            "Family labels are external formal conditions, not observer-decoded internal semantics.",
            "Norm layout overlap is not a mechanism-reuse proof.",
            "State pairs also change lexical content and are not pure state interventions.",
            "No causal, neuron, stop, or generation-time path is promoted.",
        ],
        "authorization": {
            "publish_descriptive_prefill_atlas": True,
            "publish_generation_time_atlas": False,
            "publish_functional_mechanism_reuse": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
    }

    write_jsonl(OUT / "phase416_region_rows.jsonl", region_rows)
    write_jsonl(OUT / "phase416_surface_contrast_rows.jsonl", surface_rows)
    write_jsonl(OUT / "phase416_state_lexical_contrast_rows.jsonl", state_rows)
    write_jsonl(OUT / "phase416_role_contrast_rows.jsonl", role_rows)
    write_json(
        OUT / "phase416_reuse_difference_layout.json",
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "cross_model_consensus": cross_model_consensus,
            "cross_family_layout_overlap": cross_family_reuse,
            "mechanism_reuse_proven": False,
            "causal": False,
        },
    )
    write_json(OUT / "phase416_global_summary.json", summary)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for model in MODELS:
        graph = build_graph(model, region_rows, summary)
        filename = f"phase416_{model}_formal_prefill_physical_atlas.json"
        write_json(OUT / filename, graph)
        write_json(PUBLIC / filename, graph)
        manifest_items.append(
            {
                "id": f"phase416_{model}_formal_prefill_physical_atlas",
                "label": f"Phase416 {model} 形式世界前向物理图谱",
                "filename": filename,
                "model": model,
                "phase": 416,
                "evidence_scope": "observer-free prefill physical distribution; non-causal",
            }
        )
    manifest = {
        "schema_version": "phase416_prefill_physical_atlas_manifest.v1",
        "generated_at": now(),
        "default_item_id": manifest_items[0]["id"],
        "items": manifest_items,
    }
    write_json(OUT / "manifest.json", manifest)
    write_json(PUBLIC / "manifest.json", manifest)
    write_json(PUBLIC / "phase416_global_summary.json", summary)
    return summary


def main() -> None:
    summary = analyze()
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
