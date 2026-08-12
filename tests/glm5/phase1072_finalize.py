#!/usr/bin/env python3
"""Finalize frozen Phase1072 bidirectional and task-control gates."""

from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1072_bidirectional_pattern_protocol as protocol


PRIMARY_METRICS = (
    "process_did_relative_magnitude",
    "process_lexical_reuse_cosine",
    "process_answer_invariance_cosine",
    "surface_relative_magnitude",
    "process_answer_absolute_cosine",
)


def finite_values(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def median(values: list[Any]) -> float | None:
    clean = sorted(finite_values(values))
    if not clean:
        return None
    width = len(clean)
    middle = width // 2
    if width % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


def maximum(values: list[Any]) -> float | None:
    clean = finite_values(values)
    return max(clean) if clean else None


def cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def profile(
    rows: list[dict[str, Any]],
    condition: str,
    split: str | None = None,
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["relation"] == condition
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
        and (split is None or row["split"] == split)
    ]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in selected:
        value = row["mean_process_did_relative_magnitude"]
        if value is not None and math.isfinite(float(value)):
            grouped[(str(row["role"]), int(row["depth"]))].append(
                float(value)
            )
    return [
        sum(grouped[key]) / len(grouped[key])
        for key in sorted(grouped)
    ]


def binned_profile(
    rows: list[dict[str, Any]],
    base_relation: str,
    bins: int = 11,
) -> list[float]:
    selected = [
        row
        for row in rows
        if protocol.parse_condition(row["relation"])[
            "base_relation"
        ] == base_relation
        and protocol.parse_condition(row["relation"])[
            "task_family"
        ] == "transitive"
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
    ]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in selected:
        value = row["mean_process_did_relative_magnitude"]
        if value is None or not math.isfinite(float(value)):
            continue
        bucket = min(
            bins - 1,
            int(round(float(row["relative_depth"]) * (bins - 1))),
        )
        grouped[(str(row["role"]), bucket)].append(float(value))
    result = []
    for role in protocol.PRIMARY_PROCESS_ROLES:
        for bucket in range(bins):
            values = grouped.get((role, bucket), [])
            result.append(
                sum(values) / len(values) if values else 0.0
            )
    return result


def condition_evidence(
    response_rows: list[dict[str, Any]],
    readout_rows: list[dict[str, Any]],
    condition: str,
    formal_behavior: dict[str, Any],
    calibration_behavior: dict[str, Any],
) -> dict[str, Any]:
    parsed = protocol.parse_condition(condition)
    primary = [
        row
        for row in response_rows
        if row["relation"] == condition
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
    ]
    hard_negative = [
        row["mean_process_did_relative_magnitude"]
        for row in response_rows
        if row["relation"] == condition
        and row["conditioning"] == "all"
        and row["role"] in protocol.HARD_NEGATIVE_ROLES
    ]
    embedding = [
        row["mean_process_did_relative_magnitude"]
        for row in response_rows
        if row["relation"] == condition
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
        and int(row["depth"]) == 0
    ]
    late_readout = [
        row["absolute_process_to_answer_readout_ratio"]
        for row in readout_rows
        if row["relation"] == condition
        and row["conditioning"] == "all"
        and float(row["relative_depth"])
        >= protocol.GATES["late_depth_start"]
    ]
    metrics = {
        metric: median([
            row[f"mean_{metric}"] for row in primary
        ])
        for metric in PRIMARY_METRICS
    }
    discovery = profile(
        response_rows, condition, split="discovery"
    )
    confirmation = profile(
        response_rows, condition, split="confirmation"
    )
    candidate = float(
        formal_behavior["candidate_first_token_accuracy"]
    )
    semantic = float(
        formal_behavior["semantic_first_natural_rate"]
    )
    calibration_candidate = float(
        calibration_behavior["candidate_accuracy"]
    )
    calibration_semantic = float(
        calibration_behavior["semantic_first_rate"]
    )
    formal_behavior_passed = bool(
        formal_behavior["strong_behavior_gate_passed"]
    )
    calibration_behavior_passed = bool(
        calibration_behavior[
            "condition_behavior_gate_passed"
        ]
    )
    candidate_gap = abs(candidate - calibration_candidate)
    semantic_gap = abs(semantic - calibration_semantic)
    transfer_passed = bool(
        candidate_gap
        <= protocol.GATES[
            "calibration_formal_candidate_gap_max"
        ]
        and semantic_gap
        <= protocol.GATES[
            "calibration_formal_semantic_gap_max"
        ]
    )
    return {
        "schema_version": "phase1072_condition_evidence.v1",
        "phase": protocol.PHASE,
        "condition": condition,
        **parsed,
        "formal_candidate_accuracy": candidate,
        "formal_semantic_first_rate": semantic,
        "formal_behavior_gate_passed": formal_behavior_passed,
        "calibration_candidate_accuracy": calibration_candidate,
        "calibration_semantic_first_rate": calibration_semantic,
        "calibration_behavior_gate_passed": (
            calibration_behavior_passed
        ),
        "candidate_transfer_gap": candidate_gap,
        "semantic_transfer_gap": semantic_gap,
        "exact_behavior_transfer_gate_passed": transfer_passed,
        "post_evidence_metrics": metrics,
        "hard_negative_process_did_max": maximum(hard_negative),
        "embedding_process_did_max": maximum(embedding),
        "discovery_confirmation_profile_cosine": cosine(
            discovery, confirmation
        ),
        "late_process_to_answer_readout_ratio": median(
            late_readout
        ),
        "profile": profile(response_rows, condition),
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / max(denominator, 1e-12)


def minimum(values: list[Any]) -> float | None:
    clean = finite_values(values)
    return min(clean) if clean else None


def relation_gate(
    model: str,
    base_relation: str,
    evidence_by_condition: dict[str, dict[str, Any]],
    numerical_passed: bool,
) -> dict[str, Any]:
    target_conditions = [
        protocol.condition_key(
            base_relation, "transitive", prompt, order
        )
        for prompt in protocol.PROMPT_BRANCHES
        for order in protocol.EVIDENCE_ORDERS
    ]
    control_conditions = [
        protocol.condition_key(
            base_relation,
            "direct_key_control",
            prompt,
            order,
        )
        for prompt in protocol.PROMPT_BRANCHES
        for order in protocol.EVIDENCE_ORDERS
    ]
    targets = [
        evidence_by_condition[value]
        for value in target_conditions
    ]
    controls = [
        evidence_by_condition[value]
        for value in control_conditions
    ]
    target_did = median([
        row["post_evidence_metrics"][
            "process_did_relative_magnitude"
        ]
        for row in targets
    ])
    control_did = median([
        row["post_evidence_metrics"][
            "process_did_relative_magnitude"
        ]
        for row in controls
    ])
    task_ratio = ratio(target_did, control_did)
    task_gap = (
        target_did - control_did
        if target_did is not None and control_did is not None
        else None
    )

    order_reuse = {}
    for prompt in protocol.PROMPT_BRANCHES:
        left = evidence_by_condition[
            protocol.condition_key(
                base_relation,
                "transitive",
                prompt,
                "switch_first",
            )
        ]["profile"]
        right = evidence_by_condition[
            protocol.condition_key(
                base_relation,
                "transitive",
                prompt,
                "anchor_first",
            )
        ]["profile"]
        order_reuse[prompt] = cosine(left, right)

    prompt_reuse = {}
    for order in protocol.EVIDENCE_ORDERS:
        left = evidence_by_condition[
            protocol.condition_key(
                base_relation,
                "transitive",
                "natural",
                order,
            )
        ]["profile"]
        right = evidence_by_condition[
            protocol.condition_key(
                base_relation,
                "transitive",
                "explicit",
                order,
            )
        ]["profile"]
        prompt_reuse[order] = cosine(left, right)

    target_lexical = median([
        row["post_evidence_metrics"][
            "process_lexical_reuse_cosine"
        ]
        for row in targets
    ])
    target_answer = median([
        row["post_evidence_metrics"][
            "process_answer_invariance_cosine"
        ]
        for row in targets
    ])
    target_surface = median([
        row["post_evidence_metrics"][
            "surface_relative_magnitude"
        ]
        for row in targets
    ])
    process_answer_overlap = median([
        row["post_evidence_metrics"][
            "process_answer_absolute_cosine"
        ]
        for row in targets
    ])
    hard_negative_max = maximum([
        row["hard_negative_process_did_max"]
        for row in targets + controls
    ])
    embedding_max = maximum([
        row["embedding_process_did_max"] for row in targets
    ])
    split_reuse_min = minimum([
        row["discovery_confirmation_profile_cosine"]
        for row in targets
    ])
    readout_ratio = median([
        row["late_process_to_answer_readout_ratio"]
        for row in targets
    ])
    checks = {
        "numerical": numerical_passed,
        "target_formal_behavior": all(
            row["formal_behavior_gate_passed"]
            for row in targets
        ),
        "control_formal_behavior": all(
            row["formal_behavior_gate_passed"]
            for row in controls
        ),
        "target_calibration_behavior": all(
            row["calibration_behavior_gate_passed"]
            for row in targets
        ),
        "control_calibration_behavior": all(
            row["calibration_behavior_gate_passed"]
            for row in controls
        ),
        "exact_behavior_transfer": all(
            row["exact_behavior_transfer_gate_passed"]
            for row in targets + controls
        ),
        "hard_negative": (
            hard_negative_max is not None
            and hard_negative_max
            <= protocol.GATES["hard_negative_process_did_max"]
        ),
        "embedding_control": (
            embedding_max is not None
            and embedding_max
            <= protocol.GATES[
                "embedding_process_did_relative_magnitude_max"
            ]
        ),
        "target_process_signal": (
            target_did is not None
            and target_did
            >= protocol.GATES[
                "target_process_did_relative_magnitude_min"
            ]
        ),
        "lexical_reuse": (
            target_lexical is not None
            and target_lexical
            >= protocol.GATES[
                "process_lexical_reuse_cosine_min"
            ]
        ),
        "answer_invariance": (
            target_answer is not None
            and target_answer
            >= protocol.GATES[
                "process_answer_invariance_cosine_min"
            ]
        ),
        "split_reuse": (
            split_reuse_min is not None
            and split_reuse_min
            >= protocol.GATES[
                "process_discovery_confirmation_profile_cosine_min"
            ]
        ),
        "bidirectional_order_reuse": (
            minimum(list(order_reuse.values())) is not None
            and minimum(list(order_reuse.values()))
            >= protocol.GATES[
                "bidirectional_order_profile_cosine_min"
            ]
        ),
        "natural_explicit_reuse": (
            minimum(list(prompt_reuse.values())) is not None
            and minimum(list(prompt_reuse.values()))
            >= protocol.GATES[
                "natural_explicit_profile_cosine_min"
            ]
        ),
        "task_specificity_ratio": (
            task_ratio is not None
            and task_ratio
            >= protocol.GATES[
                "target_control_process_ratio_min"
            ]
        ),
        "task_specificity_gap": (
            task_gap is not None
            and task_gap
            >= protocol.GATES[
                "target_control_process_gap_min"
            ]
        ),
        "process_answer_readout_separation": (
            readout_ratio is not None
            and readout_ratio
            <= protocol.GATES[
                "process_to_answer_readout_ratio_max"
            ]
        ),
    }
    return {
        "schema_version": "phase1072_relation_gate.v1",
        "phase": protocol.PHASE,
        "model": model,
        "base_relation": base_relation,
        "target_process_did": target_did,
        "control_process_did": control_did,
        "target_control_process_ratio": task_ratio,
        "target_control_process_gap": task_gap,
        "target_lexical_reuse_cosine": target_lexical,
        "target_answer_invariance_cosine": target_answer,
        "target_surface_relative_magnitude": target_surface,
        "target_process_answer_absolute_cosine": (
            process_answer_overlap
        ),
        "hard_negative_process_did_max": hard_negative_max,
        "embedding_process_did_max": embedding_max,
        "split_profile_cosine_min": split_reuse_min,
        "order_profile_cosines": order_reuse,
        "prompt_profile_cosines": prompt_reuse,
        "late_process_to_answer_readout_ratio": readout_ratio,
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
        raise RuntimeError("Phase1072 protocol audit failed")
    if (
        calibration["protocol_digest"]
        != prereg["calibration_protocol_digest"]
    ):
        raise RuntimeError("Phase1072 calibration digest drift")
    calibration_map = {
        (row["model"], row["condition"]): row
        for row in calibration["condition_rows"]
    }

    condition_rows = []
    relation_rows = []
    model_rows = []
    binned_profiles: dict[tuple[str, str], list[float]] = {}
    model_response_rows: dict[str, list[dict[str, Any]]] = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(atlas / "summary.json")
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"Phase1072 digest drift: {model}")
        response_rows = protocol.read_jsonl(
            atlas / "response_metrics.jsonl"
        )
        readout_rows = protocol.read_jsonl(
            atlas / "local_readout_metrics.jsonl"
        )
        model_response_rows[model] = response_rows
        numerical_passed = bool(
            float(summary["candidate_finite_rate"])
            >= protocol.GATES["candidate_finite_rate_min"]
            and float(summary["residual_metric_finite_rate"])
            >= protocol.GATES["internal_finite_rate_min"]
            and float(summary["internal_readout_finite_rate"])
            >= protocol.GATES["internal_finite_rate_min"]
        )
        evidence_map = {}
        for condition in protocol.RELATION_NAMES:
            row = condition_evidence(
                response_rows,
                readout_rows,
                condition,
                summary["relations"][condition],
                calibration_map[(model, condition)],
            )
            row["model"] = model
            evidence_map[condition] = row
            condition_rows.append(row)

        selected_relations = []
        for base_relation in protocol.BASE_RELATIONS:
            row = relation_gate(
                model,
                base_relation,
                evidence_map,
                numerical_passed,
            )
            relation_rows.append(row)
            if row["relation_gate_passed"]:
                selected_relations.append(base_relation)
            binned_profiles[(model, base_relation)] = (
                binned_profile(response_rows, base_relation)
            )
        model_passed = bool(
            numerical_passed
            and len(selected_relations)
            >= protocol.GATES[
                "minimum_strong_relations_per_model"
            ]
        )
        model_rows.append({
            "schema_version": "phase1072_model_gate.v1",
            "phase": protocol.PHASE,
            "model": model,
            "candidate_finite_rate": float(
                summary["candidate_finite_rate"]
            ),
            "residual_metric_finite_rate": float(
                summary["residual_metric_finite_rate"]
            ),
            "internal_readout_finite_rate": float(
                summary["internal_readout_finite_rate"]
            ),
            "numerical_gate_passed": numerical_passed,
            "selected_relations": selected_relations,
            "selected_relation_count": len(selected_relations),
            "model_gate_passed": model_passed,
        })

    cross_model_rows = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        cross_model_rows.append({
            "schema_version": (
                "phase1072_cross_model_profile.v1"
            ),
            "phase": protocol.PHASE,
            "left_model": left,
            "right_model": right,
            "per_relation_binned_profile_cosine": {
                relation: cosine(
                    binned_profiles[(left, relation)],
                    binned_profiles[(right, relation)],
                )
                for relation in protocol.BASE_RELATIONS
            },
            "interpretation": (
                "Descriptive relative-depth agreement only; this is "
                "not component or neuron homology."
            ),
        })

    selected_models = [
        row["model"]
        for row in model_rows
        if row["model_gate_passed"]
    ]
    should_continue = bool(
        len(selected_models)
        >= protocol.GATES["minimum_repeated_models"]
    )
    automatic = {
        "schema_version": "phase1072_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": should_continue,
        "selected_models": selected_models,
        "repeated_model_count": len(selected_models),
        "route": (
            "authorize_phase1073_component_localization"
            if should_continue
            else "stop_at_bidirectional_pattern_specificity"
        ),
        "next_phase": 1073 if should_continue else None,
        "rationale": (
            "Authorization requires at least two models with at least "
            "two relations passing exact prompt transfer, dual-order "
            "hard controls, natural/explicit and order reuse, matched "
            "task specificity, numerical stability, lexical/answer "
            "reuse, and process/answer separation."
        ),
    }
    summary = {
        "schema_version": "phase1072_atlas_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "calibration_protocol_digest": prereg[
            "calibration_protocol_digest"
        ],
        "condition_evidence": condition_rows,
        "relation_gates": relation_rows,
        "model_gates": model_rows,
        "cross_model_profiles": cross_model_rows,
        "automatic_next": automatic,
        "claim_limits": prereg["frozen_claim_limits"],
    }
    analysis = protocol.OUT_ROOT / "analysis"
    protocol.write_jsonl(
        analysis / "condition_evidence.jsonl", condition_rows
    )
    protocol.write_jsonl(
        analysis / "relation_gates.jsonl", relation_rows
    )
    protocol.write_jsonl(
        analysis / "model_gates.jsonl", model_rows
    )
    protocol.write_jsonl(
        analysis / "cross_model_profiles.jsonl",
        cross_model_rows,
    )
    protocol.write_json(
        analysis / "automatic_next.json", automatic
    )
    protocol.write_json(
        analysis / "atlas_summary.json", summary
    )
    print(
        f"Phase1072 finalized: selected_models={selected_models} "
        f"continue={should_continue}"
    )


if __name__ == "__main__":
    main()
