#!/usr/bin/env python3
"""Freeze a Phase395 query binding-contrast depth without learned predictors."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))


OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
COLLECTION = OUT / "collection/discovery"
CASES = OUT / "protocol/private/phase395_discovery_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("field_extraction", "entity_recency")
LAYER_COUNTS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
CONDITION_MAP = {
    "A_direct_lex_x": "A",
    "B_swapped_lex_x": "B",
    "C_direct_lex_y": "C",
    "D_swapped_lex_y": "D",
}
ANCHOR_COUNT = 8
MIN_LAYER_OUTPUT_CONTRAST = 0.02
MIN_ATTENTION_OUTPUT_CONTRAST = 0.005
MIN_SUPPORT = 8


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def relative_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    numerator = float(torch.linalg.vector_norm(left - right).item())
    denominator = 0.5 * (
        float(torch.linalg.vector_norm(left).item())
        + float(torch.linalg.vector_norm(right).item())
    )
    return numerator / max(denominator, 1e-12)


def layer_for_anchor(model: str, anchor: int) -> int:
    return round((anchor / (ANCHOR_COUNT - 1)) * (LAYER_COUNTS[model] - 1))


def layer_payload(model: str, case_id: str, layer: int) -> dict[str, Any]:
    return torch.load(
        COLLECTION / "private/models" / model / case_id / f"layer_{layer:03d}.pt",
        map_location="cpu", weights_only=False,
    )


def group_rows(model: str, surface: str) -> dict[str, dict[str, dict[str, Any]]]:
    rows = [
        row for row in read_jsonl(CASES)
        if row["private_execution_model"] == model and row["task_surface_private"] == surface
    ]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["phase395_public_parallel_group_id"]][CONDITION_MAP[row["contrast_condition"]]] = row
    if len(grouped) != 12 or any(set(items) != {"A", "B", "C", "D"} for items in grouped.values()):
        raise RuntimeError(f"Invalid Phase395 discovery groups for {model}/{surface}")
    return dict(sorted(grouped.items()))


def pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    coordinate = left["coordinate_names"].index("query_integrated")
    return {
        component: relative_delta(
            left["component_vectors"][component][0, coordinate],
            right["component_vectors"][component][0, coordinate],
        )
        for component in ("layer_input", "attention_output", "mlp_output", "layer_output")
    }


def analyze_cell(model: str, surface: str, anchor: int) -> dict[str, Any]:
    layer = layer_for_anchor(model, anchor)
    metrics = []
    for group_id, cases in group_rows(model, surface).items():
        payloads = {name: layer_payload(model, case["blind_case_id"], layer) for name, case in cases.items()}
        x = pair_metrics(payloads["A"], payloads["B"])
        y = pair_metrics(payloads["C"], payloads["D"])
        row = {
            "parallel_group_id": group_id,
            **{f"min_{component}_relative_contrast": min(x[component], y[component]) for component in x},
        }
        row["attention_increment_over_input"] = (
            row["min_layer_output_relative_contrast"]
            - row["min_layer_input_relative_contrast"]
        )
        metrics.append(row)
    medians = {
        key: median(row[key] for row in metrics)
        for key in (
            "min_layer_input_relative_contrast",
            "min_attention_output_relative_contrast",
            "min_mlp_output_relative_contrast",
            "min_layer_output_relative_contrast",
            "attention_increment_over_input",
        )
    }
    support = sum(
        row["min_layer_output_relative_contrast"] >= MIN_LAYER_OUTPUT_CONTRAST
        and row["min_attention_output_relative_contrast"] >= MIN_ATTENTION_OUTPUT_CONTRAST
        for row in metrics
    )
    gate = (
        medians["min_layer_output_relative_contrast"] >= MIN_LAYER_OUTPUT_CONTRAST
        and medians["min_attention_output_relative_contrast"] >= MIN_ATTENTION_OUTPUT_CONTRAST
        and support >= MIN_SUPPORT
    )
    return {
        "schema_version": "69.5.0",
        "phase_id": "Phase395-BindingDiscoveryAnalysis",
        "model": model,
        "task_surface": surface,
        "anchor_index": anchor,
        "anchor_fraction": anchor / (ANCHOR_COUNT - 1),
        "layer_index": layer,
        "group_count": len(metrics),
        **{f"median_{key}": value for key, value in medians.items()},
        "support_count": support,
        "support_required": MIN_SUPPORT,
        "cell_gate_pass": gate,
    }


def main() -> None:
    manifests = [read_json(COLLECTION / "models" / model / "manifest.json") for model in MODELS]
    if any(not item["valid"] or item["case_count"] != 96 for item in manifests):
        raise RuntimeError("Invalid Phase395 discovery collection")
    cells = [
        analyze_cell(model, surface, anchor)
        for anchor in range(ANCHOR_COUNT)
        for model in MODELS
        for surface in SURFACES
    ]
    anchor_rows = []
    for anchor in range(ANCHOR_COUNT):
        rows = [row for row in cells if row["anchor_index"] == anchor]
        anchor_rows.append(
            {
                "anchor_index": anchor,
                "anchor_fraction": anchor / (ANCHOR_COUNT - 1),
                "median_crosscell_attention_contrast": median(
                    row["median_min_attention_output_relative_contrast"] for row in rows
                ),
                "median_crosscell_layer_output_contrast": median(
                    row["median_min_layer_output_relative_contrast"] for row in rows
                ),
                "passing_cell_count": sum(row["cell_gate_pass"] for row in rows),
                "cell_denominator": len(rows),
            }
        )
    selected_anchor = max(
        range(ANCHOR_COUNT),
        key=lambda anchor: (
            anchor_rows[anchor]["passing_cell_count"],
            anchor_rows[anchor]["median_crosscell_attention_contrast"],
            anchor_rows[anchor]["median_crosscell_layer_output_contrast"],
        ),
    )
    selected_cells = [row for row in cells if row["anchor_index"] == selected_anchor]
    shared_gate = len(selected_cells) == len(MODELS) * len(SURFACES) and all(
        row["cell_gate_pass"] for row in selected_cells
    )
    wrong_anchor = 1 if selected_anchor >= ANCHOR_COUNT // 2 else ANCHOR_COUNT - 2
    candidate = {
        "anchor_index": selected_anchor,
        "anchor_fraction": selected_anchor / (ANCHOR_COUNT - 1),
        "wrong_anchor_index": wrong_anchor,
        "wrong_anchor_fraction": wrong_anchor / (ANCHOR_COUNT - 1),
        "model_layers": {
            model: {
                "candidate_layer": layer_for_anchor(model, selected_anchor),
                "wrong_depth_layer": layer_for_anchor(model, wrong_anchor),
            }
            for model in MODELS
        },
        "cells": selected_cells,
        "crossmodel_crosssurface_discovery_gate_pass": shared_gate,
        "causal_binding_claim": False,
    }
    summary = {
        "schema_version": "69.5.0",
        "phase_id": "Phase395-BindingDiscoveryAnalysis",
        "created_at": now(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "groups_per_surface_model": 12,
            "relative_anchor_count": ANCHOR_COUNT,
            "cell_count": len(cells),
        },
        "frozen_thresholds": {
            "minimum_median_layer_output_relative_contrast": MIN_LAYER_OUTPUT_CONTRAST,
            "minimum_median_attention_output_relative_contrast": MIN_ATTENTION_OUTPUT_CONTRAST,
            "minimum_group_support": MIN_SUPPORT,
        },
        "anchor_rows": anchor_rows,
        "frozen_candidate": candidate,
        "results": {
            "crossmodel_crosssurface_query_binding_contrast_candidate_count": int(shared_gate),
            "causal_binding_state_count": 0,
            "natural_necessity_count": 0,
        },
        "authorization": {
            "calibration_collection": shared_gate,
            "physical_holdout_collection": False,
            "causal_intervention": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "query_contrast_is_binding_rule": False,
            "same_token_multiset_removes_order_information": False,
            "observational_depth_is_causal_depth": False,
        },
    }
    write_jsonl(OUT / "discovery_analysis/phase395_binding_contrast_cells.jsonl", cells)
    write_json(OUT / "phase395_discovery_candidate_freeze.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
