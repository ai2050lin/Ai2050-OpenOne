#!/usr/bin/env python3
"""Finalize frozen Phase1073 late-query operation-selection gates."""

from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1073_late_query_protocol as protocol


PRIMARY_METRICS = (
    "operation_contrast_relative_magnitude",
    "transitive_did_relative_magnitude",
    "key_copy_did_relative_magnitude",
    "task_did_cosine",
    "operation_lexical_reuse_cosine",
    "operation_answer_invariance_cosine",
)


def finite_values(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def mean(values: list[Any]) -> float | None:
    clean = finite_values(values)
    return sum(clean) / len(clean) if clean else None


def median(values: list[Any]) -> float | None:
    clean = sorted(finite_values(values))
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


def maximum(values: list[Any]) -> float | None:
    clean = finite_values(values)
    return max(clean) if clean else None


def minimum(values: list[Any]) -> float | None:
    clean = finite_values(values)
    return min(clean) if clean else None


def ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / max(denominator, 1e-12)


def cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def operation_profile(
    rows: list[dict[str, Any]],
    operation_condition: str,
    split: str | None = None,
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["operation_condition"] == operation_condition
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_OPERATION_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["operation_window_start"]
        and (split is None or row["split"] == split)
    ]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in selected:
        value = row["mean_operation_contrast_relative_magnitude"]
        if value is not None and math.isfinite(float(value)):
            grouped[(str(row["role"]), int(row["depth"]))].append(
                float(value)
            )
    return [
        sum(grouped[key]) / len(grouped[key])
        for key in sorted(grouped)
    ]


def binned_relation_profile(
    rows: list[dict[str, Any]],
    relation: str,
    bins: int = 11,
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["base_relation"] == relation
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_OPERATION_ROLES
    ]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in selected:
        value = row["mean_operation_contrast_relative_magnitude"]
        if value is None or not math.isfinite(float(value)):
            continue
        bucket = min(
            bins - 1,
            int(round(float(row["relative_depth"]) * (bins - 1))),
        )
        grouped[(str(row["role"]), bucket)].append(float(value))
    result = []
    for role in protocol.PRIMARY_OPERATION_ROLES:
        for bucket in range(bins):
            values = grouped.get((role, bucket), [])
            result.append(sum(values) / len(values) if values else 0.0)
    return result


def condition_evidence(
    model: str,
    condition: str,
    formal_behavior: dict[str, Any],
    calibration_behavior: dict[str, Any],
) -> dict[str, Any]:
    parsed = protocol.parse_condition(condition)
    formal_candidate = float(
        formal_behavior["candidate_first_token_accuracy"]
    )
    formal_semantic = float(
        formal_behavior["semantic_first_natural_rate"]
    )
    calibration_candidate = float(
        calibration_behavior["candidate_accuracy"]
    )
    calibration_semantic = float(
        calibration_behavior["semantic_first_rate"]
    )
    candidate_gap = abs(formal_candidate - calibration_candidate)
    semantic_gap = abs(formal_semantic - calibration_semantic)
    transfer = bool(
        candidate_gap
        <= protocol.GATES["calibration_formal_candidate_gap_max"]
        and semantic_gap
        <= protocol.GATES["calibration_formal_semantic_gap_max"]
    )
    return {
        "schema_version": "phase1073_condition_evidence.v1",
        "phase": protocol.PHASE,
        "model": model,
        "condition": condition,
        **parsed,
        "formal_candidate_accuracy": formal_candidate,
        "formal_semantic_first_rate": formal_semantic,
        "formal_behavior_gate_passed": bool(
            formal_behavior["formal_condition_behavior_gate_passed"]
        ),
        "calibration_candidate_accuracy": calibration_candidate,
        "calibration_semantic_first_rate": calibration_semantic,
        "calibration_behavior_gate_passed": bool(
            calibration_behavior["condition_behavior_gate_passed"]
        ),
        "candidate_transfer_gap": candidate_gap,
        "semantic_transfer_gap": semantic_gap,
        "exact_behavior_transfer_gate_passed": transfer,
    }


def operation_evidence(
    model: str,
    response_rows: list[dict[str, Any]],
    operation_condition: str,
) -> dict[str, Any]:
    relation, prompt_branch, alignment, order = (
        operation_condition.split("::")
    )
    primary = [
        row
        for row in response_rows
        if row["operation_condition"] == operation_condition
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_OPERATION_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["operation_window_start"]
    ]
    prebranch = [
        row["mean_operation_contrast_relative_magnitude"]
        for row in response_rows
        if row["operation_condition"] == operation_condition
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRE_BRANCH_HARD_NEGATIVE_ROLES
    ]
    embedding = [
        row["mean_operation_contrast_relative_magnitude"]
        for row in response_rows
        if row["operation_condition"] == operation_condition
        and row["conditioning"] == "all"
        and int(row["depth"]) == 0
    ]
    metrics = {
        metric: median([
            row[f"mean_{metric}"] for row in primary
        ])
        for metric in PRIMARY_METRICS
    }
    discovery = operation_profile(
        response_rows, operation_condition, "discovery"
    )
    confirmation = operation_profile(
        response_rows, operation_condition, "confirmation"
    )
    return {
        "schema_version": "phase1073_operation_evidence.v1",
        "phase": protocol.PHASE,
        "model": model,
        "operation_condition": operation_condition,
        "base_relation": relation,
        "prompt_branch": prompt_branch,
        "key_alignment": alignment,
        "evidence_order": order,
        "operation_metrics": metrics,
        "pre_branch_operation_contrast_max": maximum(prebranch),
        "embedding_operation_contrast_max": maximum(embedding),
        "discovery_confirmation_profile_cosine": cosine(
            discovery, confirmation
        ),
        "profile": operation_profile(
            response_rows, operation_condition
        ),
    }


def relation_gate(
    model: str,
    relation: str,
    condition_map: dict[str, dict[str, Any]],
    operation_map: dict[str, dict[str, Any]],
    numerical_passed: bool,
) -> dict[str, Any]:
    conditions = [
        condition_map[protocol.condition_key(
            relation, task, prompt, alignment, order
        )]
        for task in protocol.TASK_FAMILIES
        for prompt in protocol.PROMPT_BRANCHES
        for alignment in protocol.KEY_ALIGNMENTS
        for order in protocol.EVIDENCE_ORDERS
    ]
    operations = [
        operation_map[protocol.operation_condition_key(
            relation, prompt, alignment, order
        )]
        for prompt in protocol.PROMPT_BRANCHES
        for alignment in protocol.KEY_ALIGNMENTS
        for order in protocol.EVIDENCE_ORDERS
    ]
    congruent = [
        row for row in operations if row["key_alignment"] == "congruent"
    ]
    incongruent = [
        row for row in operations if row["key_alignment"] == "incongruent"
    ]
    operation_magnitude = median([
        row["operation_metrics"][
            "operation_contrast_relative_magnitude"
        ]
        for row in operations
    ])
    congruent_magnitude = median([
        row["operation_metrics"][
            "operation_contrast_relative_magnitude"
        ]
        for row in congruent
    ])
    incongruent_magnitude = median([
        row["operation_metrics"][
            "operation_contrast_relative_magnitude"
        ]
        for row in incongruent
    ])
    congruent_ratio = ratio(
        congruent_magnitude, incongruent_magnitude
    )
    lexical_reuse = median([
        row["operation_metrics"][
            "operation_lexical_reuse_cosine"
        ]
        for row in operations
    ])
    answer_invariance = median([
        row["operation_metrics"][
            "operation_answer_invariance_cosine"
        ]
        for row in operations
    ])
    transitive_did = median([
        row["operation_metrics"][
            "transitive_did_relative_magnitude"
        ]
        for row in operations
    ])
    key_copy_did = median([
        row["operation_metrics"][
            "key_copy_did_relative_magnitude"
        ]
        for row in operations
    ])
    task_did_cosine = median([
        row["operation_metrics"]["task_did_cosine"]
        for row in operations
    ])
    prebranch_max = maximum([
        row["pre_branch_operation_contrast_max"]
        for row in operations
    ])
    embedding_max = maximum([
        row["embedding_operation_contrast_max"]
        for row in operations
    ])
    split_reuse_min = minimum([
        row["discovery_confirmation_profile_cosine"]
        for row in operations
    ])

    order_reuse = {}
    for prompt in protocol.PROMPT_BRANCHES:
        for alignment in protocol.KEY_ALIGNMENTS:
            left = operation_map[
                protocol.operation_condition_key(
                    relation, prompt, alignment, "switch_first"
                )
            ]["profile"]
            right = operation_map[
                protocol.operation_condition_key(
                    relation, prompt, alignment, "anchor_first"
                )
            ]["profile"]
            order_reuse[f"{prompt}::{alignment}"] = cosine(left, right)

    prompt_reuse = {}
    for alignment in protocol.KEY_ALIGNMENTS:
        for order in protocol.EVIDENCE_ORDERS:
            left = operation_map[
                protocol.operation_condition_key(
                    relation, "natural", alignment, order
                )
            ]["profile"]
            right = operation_map[
                protocol.operation_condition_key(
                    relation, "explicit", alignment, order
                )
            ]["profile"]
            prompt_reuse[f"{alignment}::{order}"] = cosine(left, right)

    alignment_reuse = {}
    for prompt in protocol.PROMPT_BRANCHES:
        for order in protocol.EVIDENCE_ORDERS:
            left = operation_map[
                protocol.operation_condition_key(
                    relation, prompt, "congruent", order
                )
            ]["profile"]
            right = operation_map[
                protocol.operation_condition_key(
                    relation, prompt, "incongruent", order
                )
            ]["profile"]
            alignment_reuse[f"{prompt}::{order}"] = cosine(left, right)

    per_task_candidate = {
        task: mean([
            row["formal_candidate_accuracy"]
            for row in conditions
            if row["task_family"] == task
        ])
        for task in protocol.TASK_FAMILIES
    }
    per_alignment_candidate = {
        alignment: mean([
            row["formal_candidate_accuracy"]
            for row in conditions
            if row["key_alignment"] == alignment
        ])
        for alignment in protocol.KEY_ALIGNMENTS
    }
    checks = {
        "numerical": numerical_passed,
        "formal_behavior": all(
            row["formal_behavior_gate_passed"] for row in conditions
        ),
        "calibration_behavior": all(
            row["calibration_behavior_gate_passed"]
            for row in conditions
        ),
        "exact_behavior_transfer": all(
            row["exact_behavior_transfer_gate_passed"]
            for row in conditions
        ),
        "per_task_behavior": all(
            value is not None
            and value
            >= protocol.GATES["per_task_candidate_accuracy_min"]
            for value in per_task_candidate.values()
        ),
        "per_alignment_behavior": all(
            value is not None
            and value
            >= protocol.GATES["per_alignment_candidate_accuracy_min"]
            for value in per_alignment_candidate.values()
        ),
        "pre_branch_hard_control": (
            prebranch_max is not None
            and prebranch_max
            <= protocol.GATES["pre_branch_operation_contrast_max"]
        ),
        "embedding_control": (
            embedding_max is not None
            and embedding_max
            <= protocol.GATES["embedding_operation_contrast_max"]
        ),
        "operation_signal": (
            operation_magnitude is not None
            and operation_magnitude
            >= protocol.GATES[
                "operation_contrast_relative_magnitude_min"
            ]
        ),
        "congruent_output_identity_signal": (
            congruent_magnitude is not None
            and congruent_magnitude
            >= protocol.GATES[
                "congruent_operation_relative_magnitude_min"
            ]
        ),
        "congruent_incongruent_ratio": (
            congruent_ratio is not None
            and congruent_ratio
            >= protocol.GATES[
                "congruent_to_incongruent_magnitude_ratio_min"
            ]
        ),
        "lexical_reuse": (
            lexical_reuse is not None
            and lexical_reuse
            >= protocol.GATES[
                "operation_lexical_reuse_cosine_min"
            ]
        ),
        "answer_invariance": (
            answer_invariance is not None
            and answer_invariance
            >= protocol.GATES[
                "operation_answer_invariance_cosine_min"
            ]
        ),
        "split_reuse": (
            split_reuse_min is not None
            and split_reuse_min
            >= protocol.GATES[
                "operation_discovery_confirmation_cosine_min"
            ]
        ),
        "order_reuse": (
            minimum(list(order_reuse.values())) is not None
            and minimum(list(order_reuse.values()))
            >= protocol.GATES["operation_order_profile_cosine_min"]
        ),
        "prompt_reuse": (
            minimum(list(prompt_reuse.values())) is not None
            and minimum(list(prompt_reuse.values()))
            >= protocol.GATES["operation_prompt_profile_cosine_min"]
        ),
        "alignment_reuse": (
            minimum(list(alignment_reuse.values())) is not None
            and minimum(list(alignment_reuse.values()))
            >= protocol.GATES[
                "operation_alignment_profile_cosine_min"
            ]
        ),
    }
    return {
        "schema_version": "phase1073_relation_gate.v1",
        "phase": protocol.PHASE,
        "model": model,
        "base_relation": relation,
        "formal_candidate_accuracy_by_task": per_task_candidate,
        "formal_candidate_accuracy_by_alignment": (
            per_alignment_candidate
        ),
        "operation_contrast_relative_magnitude": operation_magnitude,
        "congruent_operation_relative_magnitude": congruent_magnitude,
        "incongruent_operation_relative_magnitude": incongruent_magnitude,
        "congruent_to_incongruent_magnitude_ratio": congruent_ratio,
        "transitive_did_relative_magnitude": transitive_did,
        "key_copy_did_relative_magnitude": key_copy_did,
        "transitive_to_key_copy_did_ratio": ratio(
            transitive_did, key_copy_did
        ),
        "task_did_cosine": task_did_cosine,
        "operation_lexical_reuse_cosine": lexical_reuse,
        "operation_answer_invariance_cosine": answer_invariance,
        "pre_branch_operation_contrast_max": prebranch_max,
        "embedding_operation_contrast_max": embedding_max,
        "split_profile_cosine_min": split_reuse_min,
        "order_profile_cosines": order_reuse,
        "prompt_profile_cosines": prompt_reuse,
        "alignment_profile_cosines": alignment_reuse,
        "checks": checks,
        "relation_gate_passed": all(checks.values()),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    calibration = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "analysis"
        / "calibration_summary.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1073 protocol audit failed")
    if (
        calibration["protocol_digest"]
        != prereg["calibration_protocol_digest"]
    ):
        raise RuntimeError("Phase1073 calibration digest drift")
    calibration_map = {
        (row["model"], row["condition"]): row
        for row in calibration["condition_rows"]
    }

    condition_rows = []
    operation_rows = []
    relation_rows = []
    model_rows = []
    binned_profiles: dict[tuple[str, str], list[float]] = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(atlas / "summary.json")
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"Phase1073 digest drift: {model}")
        response_rows = protocol.read_jsonl(
            atlas / "response_metrics.jsonl"
        )
        numerical_passed = bool(
            float(summary["candidate_finite_rate"])
            >= protocol.GATES["candidate_finite_rate_min"]
            and float(summary["residual_metric_finite_rate"])
            >= protocol.GATES["internal_finite_rate_min"]
        )
        evidence_map = {}
        for condition in protocol.RELATION_NAMES:
            row = condition_evidence(
                model,
                condition,
                summary["conditions"][condition],
                calibration_map[(model, condition)],
            )
            evidence_map[condition] = row
            condition_rows.append(row)
        operation_map = {}
        for operation_condition in protocol.OPERATION_CONDITIONS:
            row = operation_evidence(
                model, response_rows, operation_condition
            )
            operation_map[operation_condition] = row
            operation_rows.append(row)

        selected_relations = []
        for relation in protocol.BASE_RELATIONS:
            row = relation_gate(
                model,
                relation,
                evidence_map,
                operation_map,
                numerical_passed,
            )
            relation_rows.append(row)
            if row["relation_gate_passed"]:
                selected_relations.append(relation)
            binned_profiles[(model, relation)] = (
                binned_relation_profile(response_rows, relation)
            )
        model_passed = bool(
            numerical_passed
            and len(selected_relations)
            >= protocol.GATES["minimum_strong_relations_per_model"]
        )
        model_rows.append({
            "schema_version": "phase1073_model_gate.v1",
            "phase": protocol.PHASE,
            "model": model,
            "candidate_finite_rate": float(
                summary["candidate_finite_rate"]
            ),
            "residual_metric_finite_rate": float(
                summary["residual_metric_finite_rate"]
            ),
            "numerical_gate_passed": numerical_passed,
            "selected_relations": selected_relations,
            "selected_relation_count": len(selected_relations),
            "model_gate_passed": model_passed,
        })
        del response_rows

    cross_model_rows = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        relation_cosines = {
            relation: cosine(
                binned_profiles[(left, relation)],
                binned_profiles[(right, relation)],
            )
            for relation in protocol.BASE_RELATIONS
        }
        cross_model_rows.append({
            "schema_version": "phase1073_cross_model_profile.v1",
            "phase": protocol.PHASE,
            "left_model": left,
            "right_model": right,
            "relation_profile_cosines": relation_cosines,
            "median_relation_profile_cosine": median(
                list(relation_cosines.values())
            ),
        })

    selected_models = [
        row["model"] for row in model_rows if row["model_gate_passed"]
    ]
    should_continue = bool(
        len(selected_models)
        >= protocol.GATES["minimum_repeated_models"]
    )
    analysis = protocol.OUT_ROOT / "analysis"
    protocol.write_jsonl(
        analysis / "condition_evidence.jsonl", condition_rows
    )
    protocol.write_jsonl(
        analysis / "operation_evidence.jsonl", operation_rows
    )
    protocol.write_jsonl(
        analysis / "relation_gates.jsonl", relation_rows
    )
    protocol.write_jsonl(analysis / "model_gates.jsonl", model_rows)
    protocol.write_jsonl(
        analysis / "cross_model_profiles.jsonl", cross_model_rows
    )
    atlas_summary = {
        "schema_version": "phase1073_atlas_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "calibration_protocol_digest": prereg[
            "calibration_protocol_digest"
        ],
        "condition_evidence_count": len(condition_rows),
        "operation_evidence_count": len(operation_rows),
        "relation_gate_count": len(relation_rows),
        "relation_gate_pass_count": sum(
            row["relation_gate_passed"] for row in relation_rows
        ),
        "model_gates": model_rows,
        "cross_model_profiles": cross_model_rows,
        "selected_models": selected_models,
        "repeated_model_count": len(selected_models),
        "automatic_next_authorized": should_continue,
        "claim": (
            "The atlas tests whether a late task cue changes the "
            "factorial relation-path interaction under an identical "
            "evidence prefix. It does not identify a complete language "
            "algorithm or a minimal physical circuit."
        ),
    }
    protocol.write_json(analysis / "atlas_summary.json", atlas_summary)
    automatic = {
        "schema_version": "phase1073_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selected_models": selected_models,
        "repeated_model_count": len(selected_models),
        "should_continue_automatically": should_continue,
        "next_phase": 1074 if should_continue else None,
        "route": (
            "localize_repeated_late_query_operation_components"
            if should_continue
            else "stop_at_late_query_operation_selection"
        ),
        "reason": (
            "At least two FP16 models passed all frozen behavior, "
            "negative-control, operation-signal, and reuse gates."
            if should_continue
            else "The frozen cross-model operation-selection gate was "
            "not met; retain the descriptive atlas without promoting a "
            "component-level mechanism claim."
        ),
    }
    protocol.write_json(analysis / "automatic_next.json", automatic)
    print(
        "Phase1073 finalized: "
        f"selected_models={selected_models} "
        f"automatic={should_continue}"
    )


if __name__ == "__main__":
    main()
