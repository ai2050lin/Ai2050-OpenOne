#!/usr/bin/env python3
"""Aggregate Phase417 native-generation traces into a non-causal atlas."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase417_native_generation_physical_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase417_native_generation_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_network", "reasoning", "grammar", "protocol_control")
COMPONENTS = (
    "layer_input",
    "attention_output",
    "mlp_output",
    "residual_increment",
    "layer_output",
)
DEPTHS = ("early", "middle", "late")
EXECUTION_PHASES = ("prompt_prefill", "cached_incremental")
FAMILY_COLORS = {
    "knowledge_network": "#22c55e",
    "reasoning": "#f59e0b",
    "grammar": "#38bdf8",
    "protocol_control": "#a78bfa",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bounded_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left) + abs(right), 1e-8)


def build_graph(model: str, region_rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in region_rows if row["model"] == model]
    component_index = {value: index for index, value in enumerate(COMPONENTS)}
    family_index = {value: index for index, value in enumerate(FAMILIES)}
    depth_index = {value: index for index, value in enumerate(DEPTHS)}
    phase_offset = {"prompt_prefill": -1.1, "cached_incremental": 1.1}
    nodes = []
    node_ids = set()
    for row in rows:
        relative = max(0.0, min(1.0, row["within_phase_component_relative_norm"]))
        local_offset = relative - 0.5
        transition_difference = row["prompt_to_cached_bounded_l2_difference"]
        transition_offset = (
            transition_difference if row["execution_phase"] == "cached_incremental" else 0.0
        )
        node_id = (
            f"phase417:{model}:{row['family_id']}:{row['component']}:"
            f"{row['depth_bin']}:{row['execution_phase']}"
        )
        node_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": (
                    f"{row['family_id']} · {row['component']} · "
                    f"{row['depth_bin']} · {row['execution_phase']}"
                ),
                "type": "event",
                "model": model,
                "family_id": row["family_id"],
                "component": row["component"],
                "depth_bin": row["depth_bin"],
                "execution_phase": row["execution_phase"],
                "score": relative,
                "size": 0.10 + 0.90 * relative + 2.0 * transition_offset,
                "color": FAMILY_COLORS[row["family_id"]],
                "position": [
                    (component_index[row["component"]] - 2) * 4.2
                    + local_offset * 1.3
                    + transition_offset * 5.0,
                    depth_index[row["depth_bin"]] * 8.0
                    + local_offset * 4.0
                    + transition_offset * 18.0,
                    family_index[row["family_id"]] * 6.0 + phase_offset[row["execution_phase"]],
                ],
                "measurement_count": row["measurement_count"],
                "case_count": row["case_count"],
                "mean_l2_norm": row["mean_l2_norm"],
                "prompt_to_cached_bounded_l2_difference": transition_difference,
                "dominant_depth_for_phase_component": row["dominant_depth_for_phase_component"],
                "formal_behavior_qualified": row["formal_behavior_qualified"],
                "observer_id": None,
                "physical": True,
                "generation_time": True,
                "predictive": False,
                "causal": False,
                "show_label": bool(row["dominant_depth_for_phase_component"]),
            }
        )

    edges = []
    for family in FAMILIES:
        for component in COMPONENTS:
            for phase in EXECUTION_PHASES:
                for left, right in zip(DEPTHS, DEPTHS[1:]):
                    source = f"phase417:{model}:{family}:{component}:{left}:{phase}"
                    target = f"phase417:{model}:{family}:{component}:{right}:{phase}"
                    if source in node_ids and target in node_ids:
                        edges.append(
                            {
                                "id": f"{source}->{target}",
                                "source": source,
                                "target": target,
                                "relation": "observed_depth_continuity",
                                "causal": False,
                            }
                        )
            for depth in DEPTHS:
                source = f"phase417:{model}:{family}:{component}:{depth}:prompt_prefill"
                target = f"phase417:{model}:{family}:{component}:{depth}:cached_incremental"
                if source in node_ids and target in node_ids:
                    edges.append(
                        {
                            "id": f"{source}->{target}",
                            "source": source,
                            "target": target,
                            "relation": "observed_native_generation_phase_transition",
                            "causal": False,
                        }
                    )
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase417-NativeGenerationPhysicalAtlas",
        "title": f"Phase417 {model} 原生生成物理事件图谱",
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {
            "source_case_count": summary["model_case_count"][model],
            "source_physical_row_count": summary["model_physical_row_count"][model],
            "source_native_call_count": summary["model_native_call_count"][model],
            "right_censored_case_count": summary["model_right_censored_count"][model],
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "visual_encoding": {
            "fixed_axes": ["component", "depth_bin", "family_id", "execution_phase"],
            "local_displacement": "model_family_component_phase_relative_norm",
            "node_size": "model_family_component_phase_relative_norm",
            "cached_phase_extra_displacement": "prompt_to_cached_bounded_l2_difference",
            "cross_model_scale": "model_local_only",
        },
        "evidence_boundary": [
            "Passive measurements from the model-owned native generation path.",
            "Prompt-prefill and cached-incremental nodes are execution phases, not semantic stages.",
            "Edges are observed call/depth order and are non-causal.",
            "Only the current prediction input position is reduced; historical KV states are not decoded.",
        ],
    }


def analyze() -> dict[str, Any]:
    aggregate: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "norm_sum": 0.0, "cases": set(), "qualified": 0}
    )
    values: dict[tuple[str, str, int, str, int], tuple[str, str, float]] = {}
    model_case_count: dict[str, int] = {}
    model_row_count: dict[str, int] = {}
    model_call_count: dict[str, int] = {}
    model_right_censored: dict[str, int] = {}
    all_case_rows = []
    all_physical_rows = 0

    for model in MODELS:
        root = OUT / "models" / model
        complete = read_json(root / "phase417_native_generation_complete.json")
        if not complete["native_generation_qualification_pass"]:
            raise RuntimeError(f"Native generation collector did not qualify for {model}")
        case_rows = list(read_jsonl(root / "phase417_native_generation_case_rows.jsonl"))
        all_case_rows.extend(case_rows)
        model_case_count[model] = complete["case_count"]
        model_row_count[model] = complete["physical_row_count"]
        model_call_count[model] = complete["physical_call_count"]
        model_right_censored[model] = complete["right_censored_count"]
        observed_rows = 0
        for row in read_jsonl(root / "phase417_generation_physical_rows.jsonl"):
            observed_rows += 1
            key = (
                model,
                row["family_id"],
                row["component"],
                row["depth_bin"],
                row["execution_phase"],
            )
            bucket = aggregate[key]
            bucket["count"] += 1
            bucket["norm_sum"] += float(row["l2_norm"])
            bucket["cases"].add(row["case_id"])
            bucket["qualified"] += int(row["formal_behavior_qualified"])
            values[(model, row["case_id"], int(row["layer"]), row["component"], int(row["prediction_step"]))] = (
                row["family_id"],
                row["depth_bin"],
                float(row["l2_norm"]),
            )
        if observed_rows != complete["physical_row_count"]:
            raise RuntimeError(f"Physical row mismatch for {model}: {observed_rows}")
        all_physical_rows += observed_rows

    component_max: dict[tuple[str, str, str, str], float] = defaultdict(float)
    for key, bucket in aggregate.items():
        mean_norm = bucket["norm_sum"] / bucket["count"]
        component_max[(key[0], key[1], key[2], key[4])] = max(
            component_max[(key[0], key[1], key[2], key[4])], mean_norm
        )
    dominant: dict[tuple[str, str, str, str], tuple[str, float]] = {}
    region_rows = []
    for key, bucket in sorted(aggregate.items()):
        mean_norm = bucket["norm_sum"] / bucket["count"]
        group = (key[0], key[1], key[2], key[4])
        if group not in dominant or mean_norm > dominant[group][1]:
            dominant[group] = (key[3], mean_norm)
        region_rows.append(
            {
                "schema_version": "90.0.0",
                "phase_id": "Phase417-NativeGenerationPhysicalAtlas",
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "component": key[2],
                "depth_bin": key[3],
                "execution_phase": key[4],
                "measurement_count": bucket["count"],
                "case_count": len(bucket["cases"]),
                "mean_l2_norm": mean_norm,
                "within_phase_component_relative_norm": mean_norm / max(component_max[group], 1e-8),
                "formal_behavior_qualified": bucket["qualified"] == bucket["count"],
                "dominant_depth_for_phase_component": False,
                "observer_id": None,
                "causal": False,
            }
        )
    for row in region_rows:
        group = (row["model"], row["family_id"], row["component"], row["execution_phase"])
        row["dominant_depth_for_phase_component"] = row["depth_bin"] == dominant[group][0]

    transition_buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    calls_by_case: dict[tuple[str, str], set[int]] = defaultdict(set)
    for model, case_id, _layer, _component, step in values:
        calls_by_case[(model, case_id)].add(step)
    for (model, case_id), calls in calls_by_case.items():
        for step in sorted(value for value in calls if value > 0):
            for layer_component in [key for key in values if key[0] == model and key[1] == case_id and key[4] == step]:
                _model, _case, layer, component, _step = layer_component
                baseline_key = (model, case_id, layer, component, 0)
                if baseline_key not in values:
                    continue
                family, depth, right = values[layer_component]
                left = values[baseline_key][2]
                transition_buckets[(model, family, component, depth)].append(
                    bounded_difference(left, right)
                )
    transition_rows = [
        {
            "schema_version": "90.0.0",
            "phase_id": "Phase417-NativeGenerationPhysicalAtlas",
            "created_at": now(),
            "model": key[0],
            "family_id": key[1],
            "component": key[2],
            "depth_bin": key[3],
            "measurement_count": len(items),
            "mean_prompt_to_cached_bounded_l2_difference": sum(items) / len(items),
            "contrast_mixes_generated_token_identity_and_cache_history": True,
            "observer_id": None,
            "causal": False,
        }
        for key, items in sorted(transition_buckets.items())
    ]
    transition_lookup = {
        (row["model"], row["family_id"], row["component"], row["depth_bin"]): row[
            "mean_prompt_to_cached_bounded_l2_difference"
        ]
        for row in transition_rows
    }
    for row in region_rows:
        row["prompt_to_cached_bounded_l2_difference"] = transition_lookup[
            (row["model"], row["family_id"], row["component"], row["depth_bin"])
        ]

    cross_model_rows = []
    for family in FAMILIES:
        for component in COMPONENTS:
            for phase in EXECUTION_PHASES:
                depths = [dominant[(model, family, component, phase)][0] for model in MODELS]
                most_common, count = Counter(depths).most_common(1)[0]
                cross_model_rows.append(
                    {
                        "family_id": family,
                        "component": component,
                        "execution_phase": phase,
                        "dominant_depth": most_common,
                        "matching_model_count": count,
                        "all_three_models_match": count == 3,
                        "at_least_two_models_match": count >= 2,
                        "causal": False,
                    }
                )

    total_cases = sum(model_case_count.values())
    total_calls = sum(model_call_count.values())
    prompt_call_count = total_cases
    cached_call_count = total_calls - prompt_call_count
    summary = {
        "schema_version": "90.0.0",
        "phase_id": "Phase417-NativeGenerationPhysicalAtlasSummary",
        "created_at": now(),
        "valid": True,
        "model_count": 3,
        "case_count": total_cases,
        "native_generation_qualification_pass_count": sum(
            row["native_generation_case_pass"] for row in all_case_rows
        ),
        "same_contract_zero_score_difference_case_count": sum(
            row["comparison"]["score_max_abs"] == 0.0 and row["comparison"]["score_js"] == 0.0
            for row in all_case_rows
        ),
        "physical_row_count": all_physical_rows,
        "native_call_count": total_calls,
        "prompt_prefill_call_count": prompt_call_count,
        "cached_incremental_call_count": cached_call_count,
        "right_censored_case_count": sum(model_right_censored.values()),
        "model_case_count": model_case_count,
        "model_physical_row_count": model_row_count,
        "model_native_call_count": model_call_count,
        "model_right_censored_count": model_right_censored,
        "region_cell_count": len(region_rows),
        "prompt_to_cached_transition_measurement_count": sum(
            row["measurement_count"] for row in transition_rows
        ),
        "cross_model_layout_cell_count": len(cross_model_rows),
        "cross_model_at_least_two_depth_match_count": sum(
            row["at_least_two_models_match"] for row in cross_model_rows
        ),
        "cross_model_all_three_depth_match_count": sum(
            row["all_three_models_match"] for row in cross_model_rows
        ),
        "authorization": {
            "publish_native_generation_physical_atlas": True,
            "publish_functional_generation_mechanism": False,
            "claim_cache_history_isolated": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "evidence_boundary": [
            "The pass is same-native-generation-contract hook noninterference, not independent implementation equivalence.",
            "Current-position physical summaries do not decode historical KV states.",
            "Prompt-to-cache differences mix generated token identity, context length, and cache history.",
            "No observer, prediction, causality, mechanism reuse, or neuron claim is promoted.",
        ],
    }
    write_jsonl(OUT / "phase417_generation_region_rows.jsonl", region_rows)
    write_jsonl(OUT / "phase417_prompt_to_cached_transition_rows.jsonl", transition_rows)
    write_json(OUT / "phase417_cross_model_layout.json", {"rows": cross_model_rows})
    write_json(OUT / "phase417_global_summary.json", summary)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        graph = build_graph(model, region_rows, summary)
        filename = f"phase417_{model}_native_generation_physical_atlas.json"
        write_json(PUBLIC / filename, graph)
        items.append(
            {
                "id": f"phase417_{model}_native_generation_physical_atlas",
                "label": f"Phase417 {model} 原生生成物理事件图谱",
                "filename": filename,
                "model": model,
                "phase": 417,
                "evidence_scope": "native-generation passive physical trace; non-causal",
            }
        )
    write_json(
        PUBLIC / "manifest.json",
        {
            "schema_version": "phase417_native_generation_atlas_manifest.v1",
            "generated_at": now(),
            "default_item_id": items[0]["id"],
            "items": items,
        },
    )
    write_json(PUBLIC / "phase417_global_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
