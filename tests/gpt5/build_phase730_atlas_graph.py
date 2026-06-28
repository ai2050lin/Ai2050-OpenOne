#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASE730_ROOT = Path("results/glm5_phase730_downstream_node_cancellation")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_node(nodes: list[dict[str, Any]], seen: set[str], node: dict[str, Any]) -> None:
    if node["id"] in seen:
        return
    seen.add(node["id"])
    nodes.append(node)


def relation_for_condition(condition: str, row: dict[str, Any]) -> str:
    recovery = float(row.get("recovery_fraction_vs_upstream") or 0.0)
    changed = float(row.get("changed_rate_vs_baseline") or 0.0)
    hit_drop = float(row.get("hit_drop_rate_vs_baseline") or 0.0)
    if condition.endswith("cancel_top_layer_out") and recovery >= 0.95:
        return "carries_downstream_state"
    if changed > 0.05 or hit_drop > 0.05:
        return "generation_sensitive"
    if recovery >= 0.5:
        return "strong_mediator"
    if recovery >= 0.05:
        return "partial_mediator"
    if recovery < 0:
        return "negative_or_compensatory"
    return "weak_or_null"


def condition_target_site(intervention: str, cancel_name: str, site_maps: dict[str, Any]) -> str | None:
    site_map = site_maps.get(intervention, {})
    if cancel_name == "cancel_top_layer_out":
        return site_map.get("top_layer_out")
    if cancel_name == "cancel_top_mlp_out":
        return site_map.get("top_mlp_out")
    return None


def build_graph(summary: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    phase = int(summary.get("phase", 730))
    models = list(summary.get("models", []))

    for model_index, model in enumerate(models):
        lane_z = (model_index - (len(models) - 1) / 2) * 8
        model_payload = summary["by_model"][model]
        site_maps = model_payload.get("site_maps", {})
        by_condition = model_payload.get("by_condition", {})

        model_node = f"{model}:model"
        phase_node = f"{model}:phase:{phase}"
        boundary_node = f"{model}:boundary:mlp_not_full_bottleneck"
        add_node(
            nodes,
            seen,
            {
                "id": model_node,
                "type": "model",
                "label": model,
                "model": model,
                "position": [-14, 0, lane_z],
                "role": "tested_model",
            },
        )
        add_node(
            nodes,
            seen,
            {
                "id": phase_node,
                "type": "phase",
                "label": f"Phase {phase}",
                "model": model,
                "position": [-10, 3, lane_z],
                "role": "downstream_cancellation_phase",
                "evidence_level": "mediation_test",
            },
        )
        add_node(
            nodes,
            seen,
            {
                "id": boundary_node,
                "type": "failure",
                "label": "MLP is partial, not full bottleneck",
                "model": model,
                "position": [14, 24, lane_z],
                "role": "generation_closure_boundary",
                "evidence_level": "boundary",
            },
        )
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "weight": 0, "phase": phase})

        for intervention, site_map in site_maps.items():
            intervention_node = f"{model}:intervention:{intervention}"
            add_node(
                nodes,
                seen,
                {
                    "id": intervention_node,
                    "type": "intervention",
                    "label": intervention,
                    "model": model,
                    "role": "upstream_perturbation",
                    "evidence_level": "upstream_intervention",
                    "top_layer_out": site_map.get("top_layer_out"),
                    "top_layer_delta": site_map.get("top_layer_delta"),
                    "top_mlp_out": site_map.get("top_mlp_out"),
                    "top_mlp_delta": site_map.get("top_mlp_delta"),
                },
            )
            edges.append({"source": phase_node, "target": intervention_node, "relation": "tested_by", "weight": 1, "phase": phase})

            for site_key, site_type, x in (
                ("top_layer_out", "layer", 4),
                ("top_mlp_out", "cluster", 8),
            ):
                site = site_map.get(site_key)
                if not site:
                    continue
                layer = None
                if site.startswith("hidden_"):
                    layer = int(site.split("_", 1)[1])
                elif site.startswith("L") and "_" in site:
                    layer = int(site[1:].split("_", 1)[0])
                site_node = f"{model}:site:{intervention}:{site}"
                add_node(
                    nodes,
                    seen,
                    {
                        "id": site_node,
                        "type": site_type,
                        "label": site,
                        "model": model,
                        "layer": layer,
                        "role": site_key,
                        "evidence_level": "candidate_mediator",
                        "score": site_map.get("top_layer_delta" if site_key == "top_layer_out" else "top_mlp_delta"),
                        "position": [x, layer or 0, lane_z],
                    },
                )
                edges.append(
                    {
                        "source": intervention_node,
                        "target": site_node,
                        "relation": "candidate_of",
                        "weight": site_map.get("top_layer_delta" if site_key == "top_layer_out" else "top_mlp_delta") or 0,
                        "phase": phase,
                    }
                )

        for condition, row in by_condition.items():
            if condition == "baseline|baseline":
                continue
            intervention, cancel_name = condition.split("|", 1)
            condition_node = f"{model}:condition:{condition}"
            recovery = row.get("recovery_fraction_vs_upstream")
            relation = relation_for_condition(condition, row)
            add_node(
                nodes,
                seen,
                {
                    "id": condition_node,
                    "type": "intervention",
                    "label": condition,
                    "model": model,
                    "role": "cancellation_result",
                    "evidence_level": "mediation_test",
                    "mean_logprob_delta": row.get("mean_logprob_delta"),
                    "recovery_fraction_vs_upstream": recovery,
                    "hit_rate": row.get("hit_rate"),
                    "changed_rate_vs_baseline": row.get("changed_rate_vs_baseline"),
                    "hit_drop_rate_vs_baseline": row.get("hit_drop_rate_vs_baseline"),
                },
            )
            upstream_node = f"{model}:intervention:{intervention}"
            edges.append(
                {
                    "source": upstream_node,
                    "target": condition_node,
                    "relation": relation,
                    "weight": abs(float(row.get("mean_logprob_delta") or 0.0)) + abs(float(recovery or 0.0)),
                    "phase": phase,
                    "evidence": "downstream_cancellation",
                }
            )

            target_site = condition_target_site(intervention, cancel_name, site_maps)
            if target_site:
                site_node = f"{model}:site:{intervention}:{target_site}"
                edges.append(
                    {
                        "source": condition_node,
                        "target": site_node,
                        "relation": relation,
                        "weight": abs(float(recovery or 0.0)),
                        "phase": phase,
                        "evidence": "mediation",
                    }
                )

            if relation in {"partial_mediator", "negative_or_compensatory", "generation_sensitive", "weak_or_null"}:
                edges.append(
                    {
                        "source": condition_node,
                        "target": boundary_node,
                        "relation": "not_full_closure",
                        "weight": abs(float(row.get("mean_logprob_delta") or 0.0)),
                        "phase": phase,
                    }
                )

    return {
        "schema_version": "atlas_graph_v1",
        "title": "Phase 730 Downstream Propagation Node Cancellation Atlas",
        "model_info": {
            "model": "cross_model",
            "models": models,
            "phase": phase,
            "timestamp": summary.get("timestamp"),
            "evidence_type": summary.get("evidence_type"),
        },
        "layout": {
            "x": "intervention / cancellation site offset",
            "y": "layer index",
            "z": "model lane",
        },
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": phase},
        "source_files": [str(PHASE730_ROOT / "phase730_cross_model_summary.json")],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=PHASE730_ROOT / "phase730_cross_model_summary.json")
    parser.add_argument("--output", type=Path, default=PHASE730_ROOT / "phase730_atlas_graph.json")
    args = parser.parse_args()
    graph = build_graph(read_json(args.summary))
    args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}")


if __name__ == "__main__":
    main()
