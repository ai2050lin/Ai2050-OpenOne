#!/usr/bin/env python3
"""Post-hoc control diagnosis authorized by the frozen Phase1082 stop gate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1082_semantic_output_operation_world_finalize as analysis
import phase1082_semantic_output_operation_world_protocol as protocol


def selected(rows: list[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]


def decomposition(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    tensor = np.stack([
        np.stack([
            analysis.operation_profile(
                rows, operation, (world,), "confirmation", field
            )
            for world in protocol.WORLDS
        ])
        for operation in protocol.OPERATIONS
    ])
    grand = tensor.mean(axis=(0, 1), keepdims=True)
    operation_main = tensor.mean(axis=1, keepdims=True) - grand
    world_main = tensor.mean(axis=0, keepdims=True) - grand
    interaction = tensor - grand - operation_main - world_main
    energies = {
        "operation_main": float(np.square(operation_main).sum()),
        "world_main": float(np.square(world_main).sum()),
        "operation_world_interaction": float(np.square(interaction).sum()),
    }
    total = sum(energies.values())
    return {
        "energies": energies,
        "fractions": {
            key: value / total if total > analysis.EPSILON else None
            for key, value in energies.items()
        },
    }


def main() -> None:
    root = protocol.OUT_ROOT / "analysis"
    assignments = protocol.read_json(root / "exact_assignments.json")["rows"]
    metrics = {
        model: protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        for model in protocol.MODELS
    }
    factor = protocol.read_json(root / "factor_ratio_audit.json")
    by_model = {}
    for model in protocol.MODELS:
        content_within = selected(
            assignments,
            comparison="within_model_item_split",
            field="content_route",
            profile="operation_centered",
            source_model=model,
            target_model=model,
        )[0]
        duplicate_within = selected(
            assignments,
            comparison="within_model_item_split",
            field="duplicate_route",
            profile="operation_centered",
            source_model=model,
            target_model=model,
        )[0]
        heldout_content = selected(
            assignments,
            comparison="within_model_heldout_world",
            field="content_route",
            source_model=model,
            target_model=model,
        )
        heldout_duplicate = selected(
            assignments,
            comparison="within_model_heldout_world",
            field="duplicate_route",
            source_model=model,
            target_model=model,
        )
        cross_content = selected(
            assignments,
            comparison="within_model_directed_cross_world",
            field="content_route",
            source_model=model,
            target_model=model,
        )
        cross_duplicate = selected(
            assignments,
            comparison="within_model_directed_cross_world",
            field="duplicate_route",
            source_model=model,
            target_model=model,
        )
        duplicate_lookup = {
            (row["source_world"], row["target_world"]): row
            for row in cross_duplicate
        }
        advantages = [
            float(row["identity_mean_score"])
            - float(duplicate_lookup[
                (row["source_world"], row["target_world"])
            ]["identity_mean_score"])
            for row in cross_content
        ]
        content_energy = decomposition(metrics[model], "content_route")
        duplicate_energy = decomposition(metrics[model], "duplicate_route")
        by_model[model] = {
            "within_item_split": {
                "content_top1": content_within["top1_correct"],
                "duplicate_top1": duplicate_within["top1_correct"],
                "content_identity": content_within["identity_mean_score"],
                "duplicate_identity": duplicate_within["identity_mean_score"],
                "content_minus_duplicate": (
                    float(content_within["identity_mean_score"])
                    - float(duplicate_within["identity_mean_score"])
                ),
            },
            "heldout_world": {
                "content_top1": [row["top1_correct"] for row in heldout_content],
                "duplicate_top1": [row["top1_correct"] for row in heldout_duplicate],
                "content_identity_mean": float(np.mean([
                    row["identity_mean_score"] for row in heldout_content
                ])),
                "duplicate_identity_mean": float(np.mean([
                    row["identity_mean_score"] for row in heldout_duplicate
                ])),
            },
            "directed_cross_world": {
                "content_minus_duplicate": advantages,
                "mean_advantage": float(np.mean(advantages)),
                "positive_pair_count": sum(value > 0 for value in advantages),
                "preregistered_pass_pair_count": sum(value >= 0.05 for value in advantages),
            },
            "content_decomposition": content_energy,
            "duplicate_decomposition": duplicate_energy,
            "control_to_content": factor["by_model"][model],
        }

    content_operation = [
        row["content_decomposition"]["fractions"]["operation_main"]
        for row in by_model.values()
    ]
    duplicate_operation = [
        row["duplicate_decomposition"]["fractions"]["operation_main"]
        for row in by_model.values()
    ]
    result = {
        "schema_version": "phase1082_posthoc_control_diagnostic.v2",
        "phase": protocol.PHASE,
        "status": "posthoc_descriptive_no_new_model_calls",
        "by_model": by_model,
        "aggregate": {
            "content_operation_fraction_range": [
                min(content_operation), max(content_operation)
            ],
            "duplicate_operation_fraction_range": [
                min(duplicate_operation), max(duplicate_operation)
            ],
            "models_with_duplicate_8_of_8": [
                model for model, row in by_model.items()
                if row["within_item_split"]["duplicate_top1"] == 8
            ],
        },
        "diagnosis": {
            "primary": (
                "Operation-specific lexical and grammatical carriers remain in "
                "both active and duplicate panels. They can identify the operation "
                "without demonstrating content-conditioned computation."
            ),
            "secondary": (
                "Output vocabulary and shell responses exceed the content field; "
                "cross-model physical coordinates are not conserved."
            ),
            "required_next_protocol": (
                "Use the same carrier sentences and lexical inventory while a late, "
                "behavior-valid selector changes only the operation applied to them. "
                "Pre-register a carrier-only negative control and require operation "
                "identity to exceed it before any component or neuron selection."
            ),
        },
        "limits": [
            "This is post-hoc diagnosis and cannot upgrade prospective evidence.",
            "Energy fractions are descriptive, not causal or independent components.",
            "No new model inference, intervention, or neuron localization was performed.",
        ],
    }
    result["diagnostic_digest"] = protocol.digest(result)
    protocol.write_json(root / "posthoc_control_diagnostic.json", result)
    print({
        "phase": protocol.PHASE,
        "status": result["status"],
        "duplicate_8_of_8_models": result["aggregate"][
            "models_with_duplicate_8_of_8"
        ],
        "diagnostic_digest": result["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
