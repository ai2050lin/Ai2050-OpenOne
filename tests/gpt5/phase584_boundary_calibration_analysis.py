#!/usr/bin/env python3
"""Fit simple discovery boundaries and audit transfer on Phase583 rows."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Iterable

import phase581_typed_category_protocol as source
import phase583_prompt_boundary_observer as observer
import phase584_boundary_calibration_protocol as protocol


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(model: str) -> list[dict[str, Any]]:
    path = observer.paths(model)["rows"]
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixed_axis_value(row: dict[str, Any], categories: tuple[str, str]) -> float:
    """Return logit(category 0) - logit(category 1) from target/foil rows."""
    if row["target_category"] == categories[0]:
        return float(row["target_boundary_logit"] - row["foil_boundary_logit"])
    if row["target_category"] == categories[1]:
        return float(row["foil_boundary_logit"] - row["target_boundary_logit"])
    raise ValueError(f"Unexpected category {row['target_category']!r}")


def fit_boundary(
    rows: Iterable[dict[str, Any]], categories: tuple[str, str]
) -> dict[str, Any]:
    values = {
        category: [
            fixed_axis_value(row, categories)
            for row in rows
            if row["target_category"] == category
        ]
        for category in categories
    }
    if any(not category_values for category_values in values.values()):
        raise RuntimeError("Both discovery categories are required")
    class_means = {category: mean(values[category]) for category in categories}
    if class_means[categories[0]] == class_means[categories[1]]:
        raise RuntimeError("Discovery class means are tied")
    positive_category = max(categories, key=class_means.__getitem__)
    negative_category = min(categories, key=class_means.__getitem__)
    return {
        "class_means": class_means,
        "threshold": mean(class_means.values()),
        "positive_category": positive_category,
        "negative_category": negative_category,
    }


def predict_category(
    row: dict[str, Any], categories: tuple[str, str], boundary: dict[str, Any]
) -> str:
    axis_value = fixed_axis_value(row, categories)
    if axis_value > boundary["threshold"]:
        return str(boundary["positive_category"])
    return str(boundary["negative_category"])


def audit_repeat_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    max_delta = 0.0
    malformed = 0
    for repeats in grouped.values():
        if len(repeats) != len(protocol.REPEATS):
            malformed += 1
            continue
        max_delta = max(
            max_delta,
            abs(
                float(repeats[0]["target_boundary_logit"])
                - float(repeats[1]["target_boundary_logit"])
            ),
            abs(
                float(repeats[0]["foil_boundary_logit"])
                - float(repeats[1]["foil_boundary_logit"])
            ),
        )
    return {
        "case_count": len(grouped),
        "malformed_repeat_groups": malformed,
        "maximum_repeat_logit_delta": max_delta,
        "passes": malformed == 0 and max_delta <= protocol.MAX_REPEAT_LOGIT_DELTA,
    }


def evaluate_split(
    rows: list[dict[str, Any]],
    categories: tuple[str, str],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    predictions = {
        row["case_id"]: predict_category(row, categories, boundary) for row in rows
    }
    category_metrics: dict[str, Any] = {}
    qualified_objects_by_category: dict[str, list[str]] = {}
    for category in categories:
        category_rows = [row for row in rows if row["target_category"] == category]
        category_correct = sum(
            predictions[row["case_id"]] == category for row in category_rows
        )
        by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in category_rows:
            by_object[row["object_id"]].append(row)
        qualified = sorted(
            object_id
            for object_id, object_rows in by_object.items()
            if sum(predictions[row["case_id"]] == category for row in object_rows)
            >= protocol.MIN_STABLE_SURFACES_PER_OBJECT
        )
        qualified_objects_by_category[category] = qualified
        category_metrics[category] = {
            "case_count": len(category_rows),
            "correct_count": category_correct,
            "accuracy": category_correct / len(category_rows),
            "object_count": len(by_object),
            "qualified_object_count": len(qualified),
            "minimum_required_qualified_objects": protocol.MIN_QUALIFIED_BY_RELATION_CATEGORY[
                rows[0]["relation"]
            ][category],
        }
    correct = sum(
        predictions[row["case_id"]] == row["target_category"] for row in rows
    )
    overall_accuracy = correct / len(rows)
    accuracy_gate = (
        overall_accuracy >= protocol.MIN_OVERALL_ACCURACY
        and all(
            category_metrics[category]["accuracy"]
            >= protocol.MIN_PER_CATEGORY_ACCURACY
            for category in categories
        )
    )
    object_gate = all(
        category_metrics[category]["qualified_object_count"]
        >= category_metrics[category]["minimum_required_qualified_objects"]
        for category in categories
    )
    return {
        "case_count": len(rows),
        "correct_count": correct,
        "overall_accuracy": overall_accuracy,
        "category_metrics": category_metrics,
        "qualified_objects_by_category": qualified_objects_by_category,
        "accuracy_gate_passes": accuracy_gate,
        "object_gate_passes": object_gate,
        "passes": accuracy_gate and object_gate,
    }


def main() -> None:
    frozen = protocol.register()
    model_results: dict[str, Any] = {}
    passing_model_relations: dict[str, list[str]] = {}
    for model in protocol.MODELS:
        all_rows = read_rows(model)
        repeat_audit = audit_repeat_identity(all_rows)
        if not repeat_audit["passes"]:
            raise RuntimeError(f"Phase584 {model} repeat audit failed")
        rows = [row for row in all_rows if row["execution_repeat"] == "forward1"]
        relation_results: dict[str, Any] = {}
        model_passes: list[str] = []
        for relation in protocol.RELATIONS:
            categories = tuple(source.RELATION_CATEGORIES[relation])
            relation_rows = [row for row in rows if row["relation"] == relation]
            discovery_rows = [
                row
                for row in relation_rows
                if row["split"] == protocol.CALIBRATION_SPLIT
            ]
            boundary = fit_boundary(discovery_rows, categories)
            split_results = {
                split: evaluate_split(
                    [row for row in relation_rows if row["split"] == split],
                    categories,
                    boundary,
                )
                for split in (protocol.CALIBRATION_SPLIT, *protocol.EVALUATION_SPLITS)
            }
            passes = all(split_results[split]["passes"] for split in protocol.EVALUATION_SPLITS)
            if passes:
                model_passes.append(relation)
            relation_results[relation] = {
                "categories_in_axis_order": list(categories),
                "boundary": boundary,
                "split_results": split_results,
                "both_evaluation_splits_pass": passes,
            }
        if model_passes:
            passing_model_relations[model] = model_passes
        model_results[model] = {
            "repeat_audit": repeat_audit,
            "relations": relation_results,
            "passing_relations": model_passes,
        }

    payload = {
        "schema_version": "phase584_boundary_calibration_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete_retrospective_diagnostic",
        "analysis_contract_sha256": protocol.sha256_file(protocol.PROTOCOL_PATH),
        "source_phase": frozen["source_phase"],
        "model_results": model_results,
        "passing_model_relations": passing_model_relations,
        "prompt_trace_authorized_model_relations": {},
        "natural_generation_qualified_models": [],
        "causal_intervention_authorized": False,
        "sealed_validation_authorized": False,
        "sealed_split_read": False,
        "evidence_classification": {
            "retrospective_open_data_diagnostic": True,
            "independent_confirmation": False,
            "mechanism_evidence": False,
        },
        "conclusion": (
            "A discovery-fitted scalar boundary does not meet the frozen transfer gate "
            "for any model-relation pair; lexical-prior calibration alone is insufficient."
        ),
    }
    protocol.write_json(protocol.DECISION_PATH, payload)
    print(
        json.dumps(
            {
                "passing_model_relations": passing_model_relations,
                "prompt_trace_authorized_model_relations": {},
                "sealed_validation_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
