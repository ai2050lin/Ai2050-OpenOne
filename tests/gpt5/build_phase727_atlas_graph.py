#!/usr/bin/env python3
"""
Build atlas_graph_v1 data for the 3D mechanism atlas client from Phase 727.

The output is intentionally conservative: it records measured intervention
effects and component hints, but it does not claim generation closure unless
the generation metrics changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASE724_ROOT = Path("results/glm5_phase724_fruit_route_channel_group_drilldown")
PHASE725_ROOT = Path("results/glm5_phase725_fine_channel_category_route_scan")
PHASE727_ROOT = Path("results/glm5_phase727_category_fruit_cluster_intervention")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_category_channels(model_name: str, n: int = 4) -> list[dict[str, Any]]:
    path = PHASE725_ROOT / f"phase725_{model_name}_fine_channel_summary.json"
    if not path.exists():
        return []
    data = read_json(path)
    if model_name == "deepseek7b":
        return [{"layer": 20, "head": 17, "head_key": "L20H17", "channel": c} for c in [25, 30, 24, 23][:n]]

    out: list[dict[str, Any]] = []
    seen = set()
    rows = data.get("top_category_selective_channels", []) + data.get("top_harmful_channels", [])
    for row in rows:
        key = (int(row["layer"]), int(row["head"]), int(row["channel"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"layer": key[0], "head": key[1], "head_key": row.get("head_key", f"L{key[0]}H{key[1]}"), "channel": key[2]})
        if len(out) >= n:
            break
    return out


def load_fruit_groups(model_name: str, avoid_head_key: str | None = None, n: int = 2) -> list[dict[str, Any]]:
    path = PHASE724_ROOT / f"phase724_{model_name}_channel_group_summary.json"
    if not path.exists():
        return []
    data = read_json(path)
    out: list[dict[str, Any]] = []
    seen = set()
    rows = data.get("top_fruit_shared_channel_groups", []) + data.get("top_harmful_channel_groups", [])
    for row in rows:
        if avoid_head_key and row.get("head_key") == avoid_head_key:
            continue
        key = (int(row["layer"]), int(row["head"]), int(row["channel_start"]), int(row["channel_end"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "layer": key[0],
                "head": key[1],
                "head_key": row.get("head_key", f"L{key[0]}H{key[1]}"),
                "start": key[2],
                "end": key[3],
            }
        )
        if len(out) >= n:
            break
    return out


def metric_node_fields(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": stats.get("n"),
        "hit_rate": stats.get("hit_rate"),
        "mean_logprob_delta": stats.get("mean_logprob_delta"),
        "changed_rate_vs_baseline": stats.get("changed_rate_vs_baseline"),
        "hit_drop_rate_vs_baseline": stats.get("hit_drop_rate_vs_baseline"),
        "first_rank_delta": stats.get("first_rank_delta"),
    }


def relation_for_intervention(name: str, stats: dict[str, Any]) -> str:
    changed = float(stats.get("changed_rate_vs_baseline") or 0)
    delta = float(stats.get("mean_logprob_delta") or 0)
    if changed >= 0.05:
        return "changes_generation"
    if abs(delta) >= 0.02:
        return "supports_likelihood"
    return "weak_generation_effect"


def evidence_level(stats: dict[str, Any]) -> str:
    changed = float(stats.get("changed_rate_vs_baseline") or 0)
    delta = abs(float(stats.get("mean_logprob_delta") or 0))
    if changed >= 0.05:
        return "generation_changed"
    if delta >= 0.02:
        return "likelihood_only"
    return "weak_or_null"


def add_node(nodes: list[dict[str, Any]], seen: set[str], node: dict[str, Any]) -> None:
    node_id = node["id"]
    if node_id in seen:
        return
    seen.add(node_id)
    nodes.append(node)


def add_edge(edges: list[dict[str, Any]], source: str, target: str, relation: str, weight: float = 0.0, **extra: Any) -> None:
    edges.append({"source": source, "target": target, "relation": relation, "weight": weight, **extra})


def build_graph(summary: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    models = summary.get("models") or sorted(summary.get("by_model", {}).keys())
    phase = summary.get("phase", 727)

    for model_index, model in enumerate(models):
        model_node = f"{model}:model"
        task_node = f"{model}:task:fruit_category"
        phase_node = f"{model}:phase:{phase}"
        model_stats = summary["by_model"][model]
        lane_z = (model_index - (len(models) - 1) / 2) * 8

        add_node(
            nodes,
            seen_nodes,
            {
                "id": model_node,
                "type": "model",
                "label": model,
                "model": model,
                "position": [-12, 0, lane_z],
                "role": "tested_model",
            },
        )
        add_node(
            nodes,
            seen_nodes,
            {
                "id": phase_node,
                "type": "phase",
                "label": f"Phase {phase}",
                "model": model,
                "position": [-9, 3, lane_z],
                "role": "experiment_phase",
            },
        )
        add_node(
            nodes,
            seen_nodes,
            {
                "id": task_node,
                "type": "task",
                "label": "fruit category",
                "model": model,
                "position": [-6, 6, lane_z],
                "role": "category_task",
                "n": model_stats.get("n_cases"),
            },
        )
        add_edge(edges, model_node, phase_node, "contains")
        add_edge(edges, phase_node, task_node, "tested_by")

        category_channels = load_category_channels(model, 4)
        category_head = category_channels[0] if category_channels else {}
        avoid_head_key = category_head.get("head_key")
        fruit_groups = load_fruit_groups(model, avoid_head_key, 2)

        component_nodes: dict[str, list[str]] = {
            "category_single": [],
            "category_cluster": [],
            "category_full_head": [],
            "fruit_cluster": [],
            "category_plus_fruit_cluster": [],
        }

        if category_channels:
            head_id = f"{model}:head:L{category_channels[0]['layer']}H{category_channels[0]['head']}"
            add_node(
                nodes,
                seen_nodes,
                {
                    "id": head_id,
                    "type": "head",
                    "label": category_channels[0]["head_key"],
                    "model": model,
                    "layer": category_channels[0]["layer"],
                    "head": category_channels[0]["head"],
                    "role": "category_full_head_candidate",
                    "evidence_level": "generation_changed" if model == "deepseek7b" else "weak_or_null",
                },
            )
            component_nodes["category_full_head"].append(head_id)
            component_nodes["category_plus_fruit_cluster"].append(head_id)

            cluster_id = f"{model}:cluster:category:{category_channels[0]['head_key']}"
            add_node(
                nodes,
                seen_nodes,
                {
                    "id": cluster_id,
                    "type": "cluster",
                    "label": f"{category_channels[0]['head_key']} category cluster",
                    "model": model,
                    "layer": category_channels[0]["layer"],
                    "head": category_channels[0]["head"],
                    "channels": [c["channel"] for c in category_channels],
                    "role": "category_likelihood_support_cluster",
                    "evidence_level": "likelihood_only",
                },
            )
            component_nodes["category_cluster"].append(cluster_id)
            component_nodes["category_plus_fruit_cluster"].append(cluster_id)
            add_edge(edges, head_id, cluster_id, "contains", 0.4)

            for idx, channel in enumerate(category_channels):
                channel_id = f"{model}:channel:L{channel['layer']}H{channel['head']}C{channel['channel']}"
                add_node(
                    nodes,
                    seen_nodes,
                    {
                        "id": channel_id,
                        "type": "channel",
                        "label": f"L{channel['layer']}H{channel['head']}C{channel['channel']}",
                        "model": model,
                        "layer": channel["layer"],
                        "head": channel["head"],
                        "channel": channel["channel"],
                        "role": "category_channel_candidate",
                        "evidence_level": "likelihood_only",
                    },
                )
                add_edge(edges, cluster_id, channel_id, "contains", 0.2)
                if idx == 0:
                    component_nodes["category_single"].append(channel_id)

        for group in fruit_groups:
            group_id = f"{model}:cluster:fruit:L{group['layer']}H{group['head']}C{group['start']}-{group['end']}"
            add_node(
                nodes,
                seen_nodes,
                {
                    "id": group_id,
                    "type": "cluster",
                    "label": f"{group['head_key']} C{group['start']}-{group['end']}",
                    "model": model,
                    "layer": group["layer"],
                    "head": group["head"],
                    "channels": list(range(group["start"], group["end"])),
                    "role": "fruit_shared_support_cluster",
                    "evidence_level": "likelihood_only",
                },
            )
            component_nodes["fruit_cluster"].append(group_id)
            component_nodes["category_plus_fruit_cluster"].append(group_id)

        failure_id = f"{model}:failure:likelihood_without_generation"
        add_node(
            nodes,
            seen_nodes,
            {
                "id": failure_id,
                "type": "failure",
                "label": "likelihood only",
                "model": model,
                "position": [12, 16, lane_z],
                "role": "likelihood_shift_without_generation_closure",
                "evidence_level": "boundary",
            },
        )

        for intervention, stats in model_stats.get("by_intervention", {}).items():
            intervention_id = f"{model}:intervention:{intervention}"
            relation = relation_for_intervention(intervention, stats)
            weight = abs(float(stats.get("mean_logprob_delta") or 0)) + abs(float(stats.get("changed_rate_vs_baseline") or 0))
            add_node(
                nodes,
                seen_nodes,
                {
                    "id": intervention_id,
                    "type": "intervention",
                    "label": intervention,
                    "model": model,
                    "role": "intervention_result",
                    "evidence_level": evidence_level(stats),
                    "score": stats.get("mean_logprob_delta"),
                    **metric_node_fields(stats),
                },
            )
            add_edge(edges, task_node, intervention_id, "tested_by", weight, phase=phase)

            for component_id in component_nodes.get(intervention, []):
                add_edge(edges, component_id, intervention_id, relation, weight, phase=phase)

            if relation == "supports_likelihood":
                add_edge(edges, intervention_id, failure_id, "weak_generation_effect", weight, phase=phase)
            elif relation == "changes_generation":
                add_edge(edges, intervention_id, task_node, "changes_generation", weight, phase=phase)

    return {
        "schema_version": "atlas_graph_v1",
        "title": "Phase 727 Category/Fruit Route Cluster Intervention Atlas",
        "model_info": {
            "model": "cross_model",
            "models": models,
            "phase": phase,
            "timestamp": summary.get("timestamp"),
            "evidence_type": summary.get("evidence_type"),
        },
        "layout": {
            "x": "component offset + head/channel index",
            "y": "layer index",
            "z": "model lane",
        },
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "metrics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_phase": phase,
        },
        "source_files": [
            str(PHASE727_ROOT / "phase727_cross_model_summary.json"),
            str(PHASE724_ROOT),
            str(PHASE725_ROOT),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=PHASE727_ROOT / "phase727_cross_model_summary.json")
    parser.add_argument("--output", type=Path, default=PHASE727_ROOT / "phase727_atlas_graph.json")
    args = parser.parse_args()

    summary = read_json(args.summary)
    graph = build_graph(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}")


if __name__ == "__main__":
    main()
