#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASE729_ROOT = Path("results/glm5_phase729_full_head_vs_cluster_residual_propagation")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relation_for_site(site: dict[str, Any]) -> str:
    kind = site.get("site_kind")
    amp = float(site.get("mean_amplification_vs_source") or 0)
    if kind == "layer_out" and amp >= 1.5:
        return "upstream_of"
    if kind == "mlp_out":
        return "supports_likelihood"
    if kind == "attn_out":
        return "candidate_of"
    return "weak_generation_effect"


def add_node(nodes: list[dict[str, Any]], seen: set[str], node: dict[str, Any]) -> None:
    if node["id"] in seen:
        return
    seen.add(node["id"])
    nodes.append(node)


def build_graph(summary: dict[str, Any], max_sites_per_intervention: int) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    models = summary.get("models", [])
    phase = summary.get("phase", 729)

    for model_index, model in enumerate(models):
        lane_z = (model_index - (len(models) - 1) / 2) * 8
        model_node = f"{model}:model"
        phase_node = f"{model}:phase:{phase}"
        boundary_node = f"{model}:boundary:likelihood_generation_split"
        add_node(nodes, seen, {"id": model_node, "type": "model", "label": model, "model": model, "position": [-12, 0, lane_z], "role": "tested_model"})
        add_node(nodes, seen, {"id": phase_node, "type": "phase", "label": f"Phase {phase}", "model": model, "position": [-9, 3, lane_z], "role": "propagation_phase"})
        add_node(
            nodes,
            seen,
            {
                "id": boundary_node,
                "type": "failure",
                "label": "likelihood/generation split",
                "model": model,
                "position": [12, 22, lane_z],
                "role": "propagation_boundary",
                "evidence_level": "boundary",
            },
        )
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "weight": 0})

        model_payload = summary["by_model"][model]
        for intervention, rec in model_payload["by_intervention"].items():
            intervention_node = f"{model}:intervention:{intervention}"
            add_node(
                nodes,
                seen,
                {
                    "id": intervention_node,
                    "type": "intervention",
                    "label": intervention,
                    "model": model,
                    "role": "propagation_intervention",
                    "evidence_level": "residual_propagation",
                    "max_layer_out_amplification": rec.get("max_layer_out_amplification"),
                    "mean_mlp_component_vs_input": rec.get("mean_mlp_component_vs_input"),
                    "mean_attn_component_vs_input": rec.get("mean_attn_component_vs_input"),
                },
            )
            edges.append({"source": phase_node, "target": intervention_node, "relation": "tested_by", "weight": rec.get("max_layer_out_amplification") or 0, "phase": phase})

            top_sites = rec.get("top_sites_by_delta", [])[:max_sites_per_intervention]
            for site in top_sites:
                site_node = f"{model}:site:{intervention}:{site['site']}"
                add_node(
                    nodes,
                    seen,
                    {
                        "id": site_node,
                        "type": "layer" if site.get("site_kind") == "layer_out" else "cluster",
                        "label": site["site"],
                        "model": model,
                        "layer": site.get("layer"),
                        "role": f"{site.get('site_kind')}_propagation_site",
                        "evidence_level": "propagation_metric",
                        "score": site.get("mean_delta_norm"),
                        "mean_delta_norm": site.get("mean_delta_norm"),
                        "mean_amplification_vs_source": site.get("mean_amplification_vs_source"),
                        "mean_cos_with_final_delta": site.get("mean_cos_with_final_delta"),
                    },
                )
                edges.append(
                    {
                        "source": intervention_node,
                        "target": site_node,
                        "relation": relation_for_site(site),
                        "weight": site.get("mean_delta_norm") or 0,
                        "phase": phase,
                    }
                )
            if intervention in {"category_cluster", "category_plus_fruit_cluster"}:
                edges.append({"source": intervention_node, "target": boundary_node, "relation": "weak_generation_effect", "weight": rec.get("max_layer_out_amplification") or 0, "phase": phase})

    return {
        "schema_version": "atlas_graph_v1",
        "title": "Phase 729 Full-Head vs Channel-Cluster Residual Propagation Atlas",
        "model_info": {
            "model": "cross_model",
            "models": models,
            "phase": phase,
            "timestamp": summary.get("timestamp"),
            "evidence_type": summary.get("evidence_type"),
        },
        "layout": {
            "x": "intervention/site offset",
            "y": "layer index",
            "z": "model lane",
        },
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": phase},
        "source_files": [str(PHASE729_ROOT / "phase729_cross_model_summary.json")],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=PHASE729_ROOT / "phase729_cross_model_summary.json")
    parser.add_argument("--output", type=Path, default=PHASE729_ROOT / "phase729_atlas_graph.json")
    parser.add_argument("--max-sites-per-intervention", type=int, default=6)
    args = parser.parse_args()
    graph = build_graph(read_json(args.summary), args.max_sites_per_intervention)
    args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}")


if __name__ == "__main__":
    main()
