#!/usr/bin/env python3
"""Apply frozen gates and publish the Phase423 observer atlas."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase423_workspace_observer_qualification"
PUBLIC = ROOT / "frontend/public/vis_data/phase423_workspace_observer_qualification"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
COLORS = {"qwen3": "#22c55e", "glm4": "#f59e0b", "deepseek7b": "#38bdf8"}


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
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite scalar: {value}")
    return round(float(value), 10)


def aggregate_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    values = [row for row in rows if row["split"] == split]
    total = sum(int(row["total_intermediates"]) for row in values)
    eligible = sum(int(row["eligible_intermediates"]) for row in values)
    # Dataset summaries expose rates, so weight them by eligible intermediates.
    improved = sum(
        float(row["improved_fraction"]) * int(row["eligible_intermediates"])
        for row in values
    )
    jlens_mrr = sum(
        float(row["jlens_mrr"]) * int(row["eligible_intermediates"])
        for row in values
    )
    logit_mrr = sum(
        float(row["logit_lens_mrr"]) * int(row["eligible_intermediates"])
        for row in values
    )
    jlens_pass10 = sum(
        float(row["jlens_pass_at_10"]) * int(row["eligible_intermediates"])
        for row in values
    )
    logit_pass10 = sum(
        float(row["logit_lens_pass_at_10"]) * int(row["eligible_intermediates"])
        for row in values
    )
    return {
        "split": split,
        "dataset_count": len(values),
        "total_intermediates": total,
        "eligible_intermediates": eligible,
        "eligible_fraction": clean(eligible / total if total else 0.0),
        "improved_fraction": clean(improved / eligible if eligible else 0.0),
        "jlens_mrr": clean(jlens_mrr / eligible if eligible else 0.0),
        "logit_lens_mrr": clean(logit_mrr / eligible if eligible else 0.0),
        "mrr_ratio": clean(jlens_mrr / (logit_mrr + 1e-12)),
        "jlens_pass_at_10": clean(jlens_pass10 / eligible if eligible else 0.0),
        "logit_lens_pass_at_10": clean(logit_pass10 / eligible if eligible else 0.0),
        "pass_at_10_gain": clean((jlens_pass10 - logit_pass10) / eligible if eligible else 0.0),
    }


def dataset_pass(row: dict[str, Any], gates: dict[str, Any]) -> bool:
    return bool(
        float(row["eligible_fraction"]) >= gates["eligible_intermediate_fraction_min"]
        and float(row["improved_fraction"]) >= gates["improved_fraction_each_split_min"]
        and float(row["mrr_ratio"]) >= gates["mrr_ratio_each_split_min"]
        and float(row["pass_at_10_gain"])
        >= gates["pass_at_10_absolute_gain_each_split_min"]
    )


def rank_metrics(ranks: list[int]) -> dict[str, float]:
    return {
        "median_rank": clean(statistics.median(ranks)) if ranks else 0.0,
        "mrr": clean(statistics.fmean(1.0 / rank for rank in ranks)) if ranks else 0.0,
        "pass_at_10": clean(sum(rank <= 10 for rank in ranks) / len(ranks)) if ranks else 0.0,
    }


def fixed_layer_holdout_audit(
    rows: list[dict[str, Any]],
    fit: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Post-hoc diagnostic: select one layer on calibration, then freeze it."""
    source_layers = list(fit["source_layers"])
    datasets = sorted({row["dataset"] for row in rows})
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        calibration = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["split"] == "calibration"
            and row["eligible_single_token"]
        ]
        holdout = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["split"] == "holdout"
            and row["eligible_single_token"]
        ]
        selected: dict[str, int] = {}
        for observer in ("jlens", "logit_lens"):
            field = f"{observer}_ranks_by_layer"
            layer_mrr = [
                statistics.fmean(1.0 / int(row[field][index]) for row in calibration)
                for index in range(len(source_layers))
            ]
            selected[observer] = max(
                range(len(source_layers)), key=lambda index: layer_mrr[index]
            )
        j_index = selected["jlens"]
        l_index = selected["logit_lens"]
        j_ranks = [int(row["jlens_ranks_by_layer"][j_index]) for row in holdout]
        l_ranks = [int(row["logit_lens_ranks_by_layer"][l_index]) for row in holdout]
        j_metrics = rank_metrics(j_ranks)
        l_metrics = rank_metrics(l_ranks)
        total_holdout = sum(
            row["dataset"] == dataset and row["split"] == "holdout" for row in rows
        )
        eligible_fraction = len(holdout) / total_holdout if total_holdout else 0.0
        improved_fraction = (
            sum(j_rank < l_rank for j_rank, l_rank in zip(j_ranks, l_ranks)) / len(holdout)
            if holdout
            else 0.0
        )
        comparison = {
            "dataset": dataset,
            "calibration_selected_jlens_layer": source_layers[j_index],
            "calibration_selected_logit_lens_layer": source_layers[l_index],
            "holdout_total_intermediates": total_holdout,
            "holdout_eligible_intermediates": len(holdout),
            "eligible_fraction": clean(eligible_fraction),
            "improved_fraction": clean(improved_fraction),
            "jlens": j_metrics,
            "logit_lens": l_metrics,
            "mrr_ratio": clean(j_metrics["mrr"] / (l_metrics["mrr"] + 1e-12)),
            "pass_at_10_gain": clean(j_metrics["pass_at_10"] - l_metrics["pass_at_10"]),
        }
        comparison["holdout_gate_pass"] = dataset_pass(comparison, gates)
        results.append(comparison)
    passing = sum(row["holdout_gate_pass"] for row in results)
    return {
        "status": "posthoc_not_preregistered",
        "reason": (
            "The frozen primary metric selected the best layer per labeled item. "
            "This diagnostic instead selects one layer per dataset and observer on "
            "calibration, then evaluates that fixed layer on holdout."
        ),
        "changes_frozen_authorization": False,
        "datasets": results,
        "passing_holdout_dataset_count": passing,
        "diagnostic_holdout_pass": bool(
            fit["fit_reproducibility_gate_pass"]
            and passing >= gates["passing_evaluation_datasets_each_split_min"]
        ),
    }


def atlas_payload(model: str, title: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], boundary: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase423-WorkspaceObserverQualification",
        "title": title,
        "model": model,
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges)},
        "evidence_boundary": boundary,
    }


def stability_atlas(model: str, fit: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    edges = []
    rows = fit["matrix_stability"]
    for index, row in enumerate(rows):
        relative_depth = fit["relative_source_depths"][index]
        node_id = f"phase423:{model}:stability:l{row['layer']}"
        nodes.append(
            {
                "id": node_id,
                "label": f"L{row['layer']} · cos {row['matrix_cosine']:.3f}",
                "type": "observer_matrix_stability",
                "model": model,
                "layer": row["layer"],
                "relative_depth": relative_depth,
                **row,
                "score": row["matrix_cosine"],
                "size": 0.6,
                "color": COLORS[model],
                "position": [relative_depth * 10 - 5, (row["matrix_cosine"] - 0.5) * 5, 0.0],
                "show_label": True,
                "observer": True,
                "physical": False,
                "compute_edge": False,
                "causal": False,
            }
        )
        if index:
            edges.append(
                {
                    "id": f"phase423:{model}:stability:{index-1}:{index}",
                    "source": nodes[index - 1]["id"],
                    "target": node_id,
                    "type": "observer_depth_order",
                    "observer": True,
                    "compute_edge": False,
                    "causal": False,
                    "weight": 0.25,
                    "color": "#64748b",
                }
            )
    return atlas_payload(
        model,
        f"Phase423 {model} 雅可比观察器半样本稳定性",
        nodes,
        edges,
        [
            "Nodes compare two independently fitted 50-prompt observer matrices.",
            "Depth-order edges are display ordering only, not model compute edges.",
            "No workspace, mechanism, or causal claim is authorized by this graph.",
        ],
    )


def quality_atlas(model: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    edges = []
    dataset_index = {name: index for index, name in enumerate(sorted({row["dataset"] for row in evaluation["summaries"]}))}
    for row in evaluation["summaries"]:
        x = dataset_index[row["dataset"]] * 3.0 - 4.5
        z = -1.2 if row["split"] == "calibration" else 1.2
        for observer_name, label, offset, color in (
            ("logit_lens", "词表透镜", -0.55, "#94a3b8"),
            ("jlens", "雅可比透镜", 0.55, COLORS[model]),
        ):
            node_id = f"phase423:{model}:quality:{row['dataset']}:{row['split']}:{observer_name}"
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{label} · P@10 {row[f'{observer_name}_pass_at_10']:.3f}",
                    "type": "observer_quality",
                    "model": model,
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "observer_name": observer_name,
                    "pass_at_10": row[f"{observer_name}_pass_at_10"],
                    "mrr": row[f"{observer_name}_mrr"],
                    "eligible_fraction": row["eligible_fraction"],
                    "score": row[f"{observer_name}_pass_at_10"],
                    "size": 0.55,
                    "color": color,
                    "position": [x + offset, row[f"{observer_name}_pass_at_10"] * 6 - 2, z],
                    "show_label": True,
                    "observer": True,
                    "physical": False,
                    "compute_edge": False,
                    "causal": False,
                }
            )
        edges.append(
            {
                "id": f"phase423:{model}:comparison:{row['dataset']}:{row['split']}",
                "source": nodes[-2]["id"],
                "target": nodes[-1]["id"],
                "type": "observer_metric_comparison",
                "pass_at_10_gain": row["pass_at_10_gain"],
                "mrr_ratio": row["mrr_ratio"],
                "observer": True,
                "compute_edge": False,
                "causal": False,
                "weight": max(0.1, abs(row["pass_at_10_gain"])),
                "color": "#a78bfa" if row["pass_at_10_gain"] > 0 else "#ef4444",
            }
        )
    return atlas_payload(
        model,
        f"Phase423 {model} 雅可比透镜与词表透镜外部评价",
        nodes,
        edges,
        [
            "Comparison edges encode metric differences, not internal computation.",
            "Targets are external intermediate concept labels with single-token coverage checks.",
            "This is observer calibration and cannot establish a global workspace.",
        ],
    )


def register_source() -> None:
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase423_workspace_observer_qualification",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase423 全局工作空间观察器资格图谱",
        "description": "三模型精确雅可比透镜的半样本稳定性与外部中间概念读出资格；仅观察边。",
        "manifest_path": "/vis_data/phase423_workspace_observer_qualification/manifest.json",
        "manifest_schema": "phase423_workspace_observer_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase423_workspace_observer_qualification",
        "models": list(MODELS),
        "evidence_scope": "observer qualification only; no workspace, compute-edge, or causal claim",
        "color": "#a78bfa",
    }
    registry["sources"] = [item for item in registry["sources"] if item["id"] != source["id"]]
    registry["sources"].append(source)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def main() -> None:
    protocol = read_json(OUT / "phase423_protocol.json")
    gates = protocol["frozen_gates"]
    model_summaries = {}
    manifest_items = []
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        model_root = OUT / "models" / model
        fit = read_json(model_root / "phase423_fit_summary.json")
        evaluation = read_json(model_root / "phase423_observer_evaluation_summary.json")
        evaluation_rows = read_jsonl(
            model_root / "phase423_observer_evaluation_rows.jsonl"
        )
        split_summaries = {
            split: aggregate_split(evaluation["summaries"], split)
            for split in ("calibration", "holdout")
        }
        dataset_pass_counts = {
            split: sum(
                dataset_pass(row, gates)
                for row in evaluation["summaries"]
                if row["split"] == split
            )
            for split in ("calibration", "holdout")
        }
        aggregate_gates = {}
        for split, row in split_summaries.items():
            aggregate_gates[split] = bool(
                row["eligible_fraction"] >= gates["eligible_intermediate_fraction_min"]
                and row["improved_fraction"] >= gates["improved_fraction_each_split_min"]
                and row["mrr_ratio"] >= gates["mrr_ratio_each_split_min"]
                and row["pass_at_10_gain"] >= gates["pass_at_10_absolute_gain_each_split_min"]
                and dataset_pass_counts[split]
                >= gates["passing_evaluation_datasets_each_split_min"]
            )
        observer_qualified = bool(
            fit["fit_reproducibility_gate_pass"]
            and aggregate_gates["calibration"]
            and aggregate_gates["holdout"]
        )
        posthoc_fixed_layer = fixed_layer_holdout_audit(
            evaluation_rows, fit, gates
        )
        model_summaries[model] = {
            "fit_reproducibility_gate_pass": fit["fit_reproducibility_gate_pass"],
            "matrix_cosine_median": fit["matrix_cosine_median"],
            "matrix_cosine_min": fit["matrix_cosine_min"],
            "relative_difference_median": fit["relative_difference_median"],
            "split_summaries": split_summaries,
            "dataset_pass_counts": dataset_pass_counts,
            "aggregate_split_gates": aggregate_gates,
            "observer_qualified": observer_qualified,
            "posthoc_fixed_layer_holdout_audit": posthoc_fixed_layer,
            "workspace_claim_allowed": False,
            "compute_edge_claim_allowed": False,
            "causal_claim_allowed": False,
        }
        payloads = {
            "stability": stability_atlas(model, fit),
            "quality": quality_atlas(model, evaluation),
        }
        for suffix, payload in payloads.items():
            filename = f"phase423_{model}_{suffix}.json"
            write_json(PUBLIC / filename, payload)
            manifest_items.append(
                {
                    "id": f"phase423_{model}_{suffix}",
                    "label": payload["title"],
                    "filename": filename,
                    "model": model,
                    "phase": 423,
                    "evidence_scope": "observer qualification only; non-causal",
                }
            )

    qualified_models = [model for model, summary in model_summaries.items() if summary["observer_qualified"]]
    cross_model_gate = len(qualified_models) >= int(gates["cross_model_qualified_models_min"])
    posthoc_diagnostic_models = [
        model
        for model, summary in model_summaries.items()
        if summary["posthoc_fixed_layer_holdout_audit"]["diagnostic_holdout_pass"]
    ]
    global_summary = {
        "schema_version": "phase423_workspace_observer_global_summary.v1",
        "phase": 423,
        "generated_at": now(),
        "scientific_role": "observer_qualification",
        "fit_prompt_count": protocol["fit_contract"]["prompt_count"],
        "evaluation_item_count": protocol["evaluation_contract"]["total_items"],
        "models": model_summaries,
        "qualified_models": qualified_models,
        "qualified_model_count": len(qualified_models),
        "cross_model_observer_gate_pass": cross_model_gate,
        "next_workspace_function_phase_authorized": cross_model_gate,
        "posthoc_fixed_layer_diagnostic_models": posthoc_diagnostic_models,
        "posthoc_fixed_layer_cross_model_pass": len(posthoc_diagnostic_models)
        >= int(gates["cross_model_qualified_models_min"]),
        "posthoc_diagnostic_changes_frozen_authorization": False,
        "workspace_claim_allowed": False,
        "compute_edge_claim_allowed": False,
        "causal_claim_allowed": False,
        "strict_mechanism_closure_count": 0,
        "strict_mechanism_denominator": 72,
        "interpretation": (
            "Passing qualifies a model-specific observer for later experiments. "
            "It does not demonstrate J-space, broadcast, reuse, necessity, or causality."
        ),
    }
    write_json(OUT / "phase423_global_summary.json", global_summary)
    write_json(PUBLIC / "phase423_global_summary.json", global_summary)
    write_json(
        PUBLIC / "manifest.json",
        {
            "schema_version": "phase423_workspace_observer_manifest.v1",
            "generated_at": now(),
            "default_item_id": manifest_items[0]["id"],
            "items": manifest_items,
        },
    )
    register_source()
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
