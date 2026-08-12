#!/usr/bin/env python3
"""Posthoc boundary diagnostic; never upgrades the frozen Phase1115 verdict."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1114_wordnet_contextual_hypernym_finalize as metrics
import phase1114_wordnet_contextual_hypernym_protocol as phase1114
import phase1115_wordnet_context_modulation_confirmation_protocol as phase1115


def phase_panel(protocol: Any) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in protocol.MODELS:
        rows = list(
            phase1114.read_jsonl(
                protocol.OUT_ROOT
                / "behavior"
                / model
                / "candidate_detail.jsonl"
            )
        )
        pairs = metrics.build_pairs(rows)
        concepts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pairs:
            concepts[row["concept_id"]].append(row)
        concept_signs = {}
        for concept_id, values in sorted(concepts.items()):
            finite = [row for row in values if row["finite"]]
            effects = [row["context_effect"] for row in finite]
            concept_signs[concept_id] = {
                "base": values[0]["base"],
                "split": values[0]["split"],
                "finite_fraction": len(finite) / len(values),
                "median_context_effect": statistics.median(effects)
                if effects
                else None,
                "positive_median": bool(
                    len(finite) == len(values)
                    and effects
                    and statistics.median(effects) > 0.0
                ),
                "all_templates_positive": bool(
                    len(finite) == len(values)
                    and all(effect > 0.0 for effect in effects)
                ),
            }
        finite_pairs = [row for row in pairs if row["finite"]]
        models[model] = {
            "pair_count": len(pairs),
            "finite_pair_count": len(finite_pairs),
            "positive_pair_count": sum(
                row["context_direction_hit"] for row in finite_pairs
            ),
            "context_direction_accuracy": sum(
                row["context_direction_hit"] for row in finite_pairs
            )
            / max(len(finite_pairs), 1),
            "positive_median_concept_count": sum(
                row["positive_median"] for row in concept_signs.values()
            ),
            "concept_count": len(concept_signs),
            "concept_signs": concept_signs,
        }
    return models


def shared_positive(models: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    concepts = set.intersection(
        *(set(models[name]["concept_signs"]) for name in names)
    )
    return sorted(
        concept_id
        for concept_id in concepts
        if all(
            models[name]["concept_signs"][concept_id]["positive_median"]
            for name in names
        )
    )


def main() -> None:
    panel1114 = phase_panel(phase1114)
    panel1115 = phase_panel(phase1115)
    qg1114 = shared_positive(panel1114, ("qwen3", "glm4"))
    qg1115 = shared_positive(panel1115, ("qwen3", "glm4"))
    frozen = phase1114.read_json(
        phase1115.OUT_ROOT / "analysis" / "final_summary.json"
    )
    thresholds = frozen["thresholds"]
    deficits: dict[str, Any] = {}
    for model in phase1115.MODELS:
        row = frozen["models"][model]
        min_split = min(
            value["context_direction_accuracy"]
            for value in row["pairs_by_split"].values()
        )
        min_template = min(
            value["context_direction_accuracy"]
            for value in row["pairs_by_template"].values()
        )
        deficits[model] = {
            "overall_minus_gate": row["overall_pairs"][
                "context_direction_accuracy"
            ]
            - thresholds["minimum_context_direction_accuracy"],
            "minimum_split_minus_gate": min_split
            - thresholds["minimum_split_context_direction_accuracy"],
            "minimum_template_minus_gate": min_template
            - thresholds["minimum_template_context_direction_accuracy"],
            "concept_fraction_minus_gate": row["overall_concepts"][
                "positive_median_fraction"
            ]
            - thresholds["minimum_concept_direction_accuracy"],
        }
    combined = {}
    for model in phase1115.MODELS:
        positive = (
            panel1114[model]["positive_pair_count"]
            + panel1115[model]["positive_pair_count"]
        )
        finite = (
            panel1114[model]["finite_pair_count"]
            + panel1115[model]["finite_pair_count"]
        )
        combined[model] = {
            "descriptive_positive_pair_count": positive,
            "descriptive_finite_pair_count": finite,
            "descriptive_context_direction_accuracy": positive / max(finite, 1),
            "positive_median_concept_count": (
                panel1114[model]["positive_median_concept_count"]
                + panel1115[model]["positive_median_concept_count"]
            ),
            "concept_count": (
                panel1114[model]["concept_count"]
                + panel1115[model]["concept_count"]
            ),
        }
    result = {
        "schema_version": "phase1115_posthoc_boundary_diagnostic.v1",
        "phase": phase1115.PHASE,
        "status": "posthoc_diagnostic_does_not_modify_frozen_verdict",
        "frozen_cross_model_confirmation": frozen[
            "cross_model_context_modulation_confirmed"
        ],
        "phase1114": {
            "models": panel1114,
            "qwen3_glm4_shared_positive_concepts": qg1114,
            "qwen3_glm4_shared_positive_count": len(qg1114),
        },
        "phase1115": {
            "models": panel1115,
            "qwen3_glm4_shared_positive_concepts": qg1115,
            "qwen3_glm4_shared_positive_count": len(qg1115),
            "qwen3_glm4_shared_positive_fraction": len(qg1115) / 21,
            "frozen_gate_deficits": deficits,
        },
        "combined_descriptive_only": combined,
        "interpretation_limit": (
            "Pooling phases or inspecting near-threshold deficits is descriptive only. "
            "It cannot rescue the failed Phase1115 preregistered confirmation."
        ),
    }
    result["diagnostic_digest"] = phase1114.digest(result)
    phase1114.write_json(
        phase1115.OUT_ROOT / "analysis" / "posthoc_boundary_diagnostic.json",
        result,
    )
    print(
        json.dumps(
            {
                "phase": phase1115.PHASE,
                "frozen_confirmation": result[
                    "frozen_cross_model_confirmation"
                ],
                "phase1114_qwen_glm_shared": len(qg1114),
                "phase1115_qwen_glm_shared": len(qg1115),
                "phase1115_qwen_glm_shared_fraction": len(qg1115) / 21,
                "combined_descriptive": combined,
                "diagnostic_digest": result["diagnostic_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
