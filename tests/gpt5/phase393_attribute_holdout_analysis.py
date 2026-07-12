#!/usr/bin/env python3
"""Analyze independent Phase393 attribute content transport and depth specificity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase393_attribute_content_holdout"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    models = []
    all_rows = []
    for model in MODELS:
        root = OUT / "collection" / model
        complete = read_json(root / "complete.json")
        rows = read_jsonl(root / "direction_rows.jsonl")
        if not complete["valid"] or len(rows) != 24:
            raise RuntimeError(f"Invalid Phase393 holdout for {model}")
        all_rows.extend(rows)
        scenarios = [{item["scenario"]: item for item in row["scenario_rows"]} for row in rows]
        generations = [{item["scenario"]: item for item in row["generation_rows"]} for row in rows]
        attribute = [items["donor_attributes_candidate_depth"]["normalized_margin_mediation"] for items in scenarios]
        structure = [items["donor_structure_candidate_depth"]["normalized_margin_mediation"] for items in scenarios]
        random = [items["donor_random_candidate_depth"]["normalized_margin_mediation"] for items in scenarios]
        wrong = [items["donor_attributes_wrong_depth"]["normalized_margin_mediation"] for items in scenarios]
        metrics = {
            "median_attribute_normalized_margin_mediation": median(attribute),
            "median_structure_mediation": median(structure),
            "median_random_mediation": median(random),
            "median_wrong_depth_attribute_mediation": median(wrong),
            "median_attribute_advantage_over_structure": median([a - b for a, b in zip(attribute, structure)]),
            "median_attribute_advantage_over_random": median([a - b for a, b in zip(attribute, random)]),
            "median_candidate_depth_advantage": median([a - b for a, b in zip(attribute, wrong)]),
            "positive_attribute_direction_rate": rate([value > 0 for value in attribute]),
            "attribute_answer_switch_rate": rate([
                items["donor_attributes_candidate_depth"]["strict_donor_target_switch"] for items in generations
            ]),
            "structure_answer_switch_rate": rate([
                items["donor_structure_candidate_depth"]["strict_donor_target_switch"] for items in generations
            ]),
            "random_answer_switch_rate": rate([
                items["donor_random_candidate_depth"]["strict_donor_target_switch"] for items in generations
            ]),
            "wrong_depth_attribute_switch_rate": rate([
                items["donor_attributes_wrong_depth"]["strict_donor_target_switch"] for items in generations
            ]),
        }
        transport = (
            metrics["median_attribute_normalized_margin_mediation"] >= 0.10
            and metrics["median_attribute_advantage_over_structure"] >= 0.05
            and metrics["median_attribute_advantage_over_random"] >= 0.05
            and metrics["positive_attribute_direction_rate"] >= 0.75
            and metrics["attribute_answer_switch_rate"] >= 0.75
        )
        depth_specific = transport and metrics["median_candidate_depth_advantage"] >= 0.05
        models.append(
            {
                "schema_version": "67.2.0",
                "phase_id": "Phase393-AttributeHoldoutAnalysis",
                "model": model,
                "direction_count": len(rows),
                **metrics,
                "attribute_content_transport_gate_pass": transport,
                "candidate_depth_specificity_gate_pass": depth_specific,
            }
        )
    shared_transport = all(row["attribute_content_transport_gate_pass"] for row in models)
    shared_depth = all(row["candidate_depth_specificity_gate_pass"] for row in models)
    summary = {
        "schema_version": "67.2.0",
        "phase_id": "Phase393-AttributeHoldoutAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "independent_group_count": 12,
            "direction_count": len(all_rows),
            "scenario_count": len(all_rows) * 6,
            "generation_count": len(all_rows) * 4,
            "phase392_group_overlap": 0,
        },
        "models": models,
        "results": {
            "models_passing_attribute_transport": sum(row["attribute_content_transport_gate_pass"] for row in models),
            "models_passing_depth_specificity": sum(row["candidate_depth_specificity_gate_pass"] for row in models),
            "crossmodel_attribute_content_transport_established": shared_transport,
            "crossmodel_candidate_depth_specificity_established": shared_depth,
            "multi_source_joint_path_established": False,
            "complete_field_extraction_path_established": False,
            "single_neuron_causal_path_count": 0,
            "language_encoding_closed": False,
        },
        "authorization": {
            "promote_attribute_content_transport_edge": shared_transport,
            "promote_depth_specific_specialized_path": shared_depth,
            "promote_multi_source_joint_path": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "attribute_transport_is_field_extraction_algorithm": False,
            "attribute_transport_is_multi_source_cooperation": False,
            "depth_nonspecific_transport_is_specialized_layer": False,
            "language_encoding_closed": False,
        },
    }
    (OUT / "phase393_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
