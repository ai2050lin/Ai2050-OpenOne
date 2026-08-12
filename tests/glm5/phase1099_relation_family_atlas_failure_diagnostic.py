#!/usr/bin/env python3
"""Post-hoc diagnostics for the frozen Phase1099 family-atlas failure."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1099_relation_family_atlas"
FINAL_PATH = RESULT_ROOT / "analysis" / "final_summary.json"
OUTPUT_PATH = RESULT_ROOT / "analysis" / "failure_diagnostic.json"
FAMILIES = (
    "physical_magnitude",
    "temporal_order",
    "spatial_order",
    "social_status",
    "epistemic_causal",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "diagnostic_digest"}
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    permutation_counts = Counter(tuple(row["best_permutation"]) for row in records)
    fixed_counts = {family: 0 for family in FAMILIES}
    for row in records:
        permutation = row["best_permutation"]
        for index, family in enumerate(FAMILIES):
            fixed_counts[family] += int(permutation[index] == index)
    count = len(records)
    return {
        "record_count": count,
        "passing_records": sum(bool(row["passed"]) for row in records),
        "identity_rank1_records": sum(row["identity_rank"] == 1 for row in records),
        "identity_score_median": median(row["identity_score"] for row in records),
        "permutation_margin_median": median(row["permutation_margin"] for row in records),
        "field_advantage_median": median(row["field_specificity_advantage"] for row in records),
        "family_fixed_rate_in_best_permutation": {
            family: fixed_counts[family] / count if count else 0.0 for family in FAMILIES
        },
        "most_common_best_permutations": [
            {"permutation": list(permutation), "count": frequency}
            for permutation, frequency in permutation_counts.most_common(5)
        ],
    }


def main() -> None:
    final = read_json(FINAL_PATH)
    models: dict[str, Any] = {}
    for model_name, model in final["models"].items():
        cohesion = model["cohesion"]
        primary = cohesion["relational_execution"]["mean_within_minus_between"]
        control_values = {
            field: values["mean_within_minus_between"]
            for field, values in cohesion.items()
            if field != "relational_execution"
        }
        strongest_control_field, strongest_control = max(control_values.items(), key=lambda row: row[1])
        models[model_name] = {
            "behavior_formal": model["behavior_formal"],
            "instrument_passed": model["instrument_passed"],
            "heldout_split": summarize_records(model["heldout_split_gate"]["records"]),
            "cross_language": summarize_records(model["cross_language_gate"]["records"]),
            "primary_cohesion": primary,
            "strongest_control_cohesion_field": strongest_control_field,
            "strongest_control_cohesion": strongest_control,
            "primary_minus_strongest_control_cohesion": primary - strongest_control,
            "shared_energy_median": model["energy"]["shared_median"],
            "differential_energy_median": model["energy"]["differential_median"],
        }

    cross_model: dict[str, Any] = {}
    for pair in final["cross_model"]["pairs"]:
        records = []
        for cell in pair["cells"]:
            records.extend((cell["forward"], cell["reverse"]))
        cross_model[f'{pair["left"]}->{pair["right"]}'] = summarize_records(records)

    formal_names = final["formal_models"]
    formal_heldout_records = [
        row
        for model_name in formal_names
        for row in final["models"][model_name]["heldout_split_gate"]["records"]
    ]
    formal_cross_language_records = [
        row
        for model_name in formal_names
        for row in final["models"][model_name]["cross_language_gate"]["records"]
    ]
    result: dict[str, Any] = {
        "schema_version": "phase1099_failure_diagnostic.v1",
        "phase": 1099,
        "evidence_status": "post_hoc_descriptive_only; does_not_upgrade_frozen_gates",
        "final_summary_digest": final["summary_digest"],
        "models": models,
        "formal_models_combined": {
            "heldout_split": summarize_records(formal_heldout_records),
            "cross_language": summarize_records(formal_cross_language_records),
            "mean_primary_minus_strongest_control_cohesion": mean(
                models[name]["primary_minus_strongest_control_cohesion"] for name in formal_names
            ),
        },
        "cross_model": cross_model,
        "interpretation_boundary": (
            "A high fixed rate in best permutations is only a candidate asymmetry. "
            "It cannot establish a semantic family because the registered relation-heldout, "
            "cross-language, cross-model, and matched-control gates failed."
        ),
    }
    result["diagnostic_digest"] = digest(result)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({
        "phase": 1099,
        "formal_heldout_rank1": result["formal_models_combined"]["heldout_split"]["identity_rank1_records"],
        "formal_cross_language_rank1": result["formal_models_combined"]["cross_language"]["identity_rank1_records"],
        "diagnostic_digest": result["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
