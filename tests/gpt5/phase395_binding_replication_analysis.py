#!/usr/bin/env python3
"""Replicate the frozen Phase395 observational candidate without layer search."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("field_extraction", "entity_recency")
CONDITION_MAP = {
    "A_direct_lex_x": "A",
    "B_swapped_lex_x": "B",
    "C_direct_lex_y": "C",
    "D_swapped_lex_y": "D",
}
MIN_LAYER_OUTPUT_CONTRAST = 0.02
MIN_ATTENTION_OUTPUT_CONTRAST = 0.005
DISCOVERY_SUPPORT_FRACTION = 8 / 12


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def relative_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    numerator = float(torch.linalg.vector_norm(left - right).item())
    denominator = 0.5 * (
        float(torch.linalg.vector_norm(left).item())
        + float(torch.linalg.vector_norm(right).item())
    )
    return numerator / max(denominator, 1e-12)


def pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    coordinate = left["coordinate_names"].index("query_integrated")
    return {
        component: relative_delta(
            left["component_vectors"][component][0, coordinate],
            right["component_vectors"][component][0, coordinate],
        )
        for component in ("layer_input", "attention_output", "mlp_output", "layer_output")
    }


def analyze_cell(
    split: str,
    model: str,
    surface: str,
    layer: int,
    expected_groups: int,
) -> dict[str, Any]:
    case_path = OUT / f"protocol/private/phase395_{split}_cases.jsonl"
    collection = OUT / "collection" / split / "private/models" / model
    rows = [
        row for row in read_jsonl(case_path)
        if row["private_execution_model"] == model and row["task_surface_private"] == surface
    ]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["phase395_public_parallel_group_id"]][CONDITION_MAP[row["contrast_condition"]]] = row
    if len(grouped) != expected_groups or any(set(items) != {"A", "B", "C", "D"} for items in grouped.values()):
        raise RuntimeError(f"Invalid {split} groups for {model}/{surface}")

    metrics = []
    for group_id, cases in sorted(grouped.items()):
        payloads = {
            name: torch.load(
                collection / case["blind_case_id"] / f"layer_{layer:03d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, case in cases.items()
        }
        x = pair_metrics(payloads["A"], payloads["B"])
        y = pair_metrics(payloads["C"], payloads["D"])
        metrics.append({
            "parallel_group_id": group_id,
            **{f"min_{component}_relative_contrast": min(x[component], y[component]) for component in x},
        })

    keys = (
        "min_layer_input_relative_contrast",
        "min_attention_output_relative_contrast",
        "min_mlp_output_relative_contrast",
        "min_layer_output_relative_contrast",
    )
    medians = {key: median(row[key] for row in metrics) for key in keys}
    support_required = math.ceil(DISCOVERY_SUPPORT_FRACTION * expected_groups)
    support = sum(
        row["min_layer_output_relative_contrast"] >= MIN_LAYER_OUTPUT_CONTRAST
        and row["min_attention_output_relative_contrast"] >= MIN_ATTENTION_OUTPUT_CONTRAST
        for row in metrics
    )
    gate = (
        medians["min_layer_output_relative_contrast"] >= MIN_LAYER_OUTPUT_CONTRAST
        and medians["min_attention_output_relative_contrast"] >= MIN_ATTENTION_OUTPUT_CONTRAST
        and support >= support_required
    )
    return {
        "model": model,
        "task_surface": surface,
        "layer_index": layer,
        "group_count": len(metrics),
        **{f"median_{key}": value for key, value in medians.items()},
        "support_count": support,
        "support_required": support_required,
        "cell_gate_pass": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "physical_holdout"), required=True)
    args = parser.parse_args()
    split = args.split
    candidate = read_json(OUT / "phase395_discovery_candidate_freeze.json")["frozen_candidate"]
    expected_groups = 6
    expected_cases = expected_groups * len(SURFACES) * 4
    manifests = [read_json(OUT / "collection" / split / "models" / model / "manifest.json") for model in MODELS]
    if any(not item["valid"] or item["case_count"] != expected_cases for item in manifests):
        raise RuntimeError(f"Invalid Phase395 {split} collection")

    candidate_cells = []
    wrong_depth_cells = []
    for model in MODELS:
        layers = candidate["model_layers"][model]
        for surface in SURFACES:
            candidate_cells.append(analyze_cell(
                split, model, surface, layers["candidate_layer"], expected_groups,
            ))
            wrong_depth_cells.append(analyze_cell(
                split, model, surface, layers["wrong_depth_layer"], expected_groups,
            ))
    shared_gate = all(row["cell_gate_pass"] for row in candidate_cells)
    wrong_gate_count = sum(row["cell_gate_pass"] for row in wrong_depth_cells)
    payload = {
        "schema_version": "69.6.0",
        "phase_id": "Phase395-BindingReplicationAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "frozen_candidate_anchor_index": candidate["anchor_index"],
        "frozen_wrong_anchor_index": candidate["wrong_anchor_index"],
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "groups_per_surface_model": expected_groups,
            "case_count": expected_cases * len(MODELS),
            "cell_count": len(candidate_cells),
        },
        "frozen_thresholds": {
            "minimum_median_layer_output_relative_contrast": MIN_LAYER_OUTPUT_CONTRAST,
            "minimum_median_attention_output_relative_contrast": MIN_ATTENTION_OUTPUT_CONTRAST,
            "support_fraction_inherited_from_discovery": DISCOVERY_SUPPORT_FRACTION,
            "minimum_group_support": math.ceil(DISCOVERY_SUPPORT_FRACTION * expected_groups),
        },
        "candidate_cells": candidate_cells,
        "wrong_depth_cells": wrong_depth_cells,
        "results": {
            "crossmodel_crosssurface_replication_gate_pass": shared_gate,
            "candidate_cell_pass_count": sum(row["cell_gate_pass"] for row in candidate_cells),
            "wrong_depth_cell_pass_count": wrong_gate_count,
            "causal_binding_state_count": 0,
            "natural_necessity_count": 0,
        },
        "authorization": {
            "causal_intervention": split == "calibration" and shared_gate,
            "physical_holdout_collection": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "replicated_contrast_is_binding_rule": False,
            "wrong_depth_pass_invalidates_candidate": False,
            "observational_replication_is_causal": False,
        },
    }
    output_dir = OUT / f"{split}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"phase395_{split}_replication.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
