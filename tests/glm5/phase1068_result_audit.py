#!/usr/bin/env python3
"""Audit Phase1068 outputs and compute descriptive negative controls."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1068_reasoning_generalization_protocol as protocol


RESULT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1068_reasoning_generalization"
)
MODELS = tuple(protocol.MODELS)
RELATIONS = tuple(protocol.RELATION_NAMES)


def strict_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def read_json(path: Path) -> Any:
    return strict_loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = strict_loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def mean_or_none(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def label_first_match(text: str, labels: list[str]) -> bool:
    """Post-hoc format-tolerant audit; never replaces the frozen exact gate."""
    normalized = text.strip().lower()
    normalized = re.sub(r"^[\s*_`#\\]+", "", normalized)
    normalized = re.sub(r"^boxed\s*\{\s*", "", normalized)
    normalized = re.sub(r"^answer\s*:\s*", "", normalized)
    normalized = re.sub(r"^[\s*_`#\\{(\[]+", "", normalized)
    return any(
        re.match(
            rf"^{re.escape(str(label).strip().lower())}(?=$|[^a-z])",
            normalized,
        )
        is not None
        for label in labels
    )


def response_profile(
    rows: list[dict[str, Any]],
    relation: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_depth: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        value = row["mean_semantic_relative_magnitude"]
        if (
            row["bucket_id"] != f"relation:{relation}"
            or row["role"] != "answer_boundary"
            or value is None
        ):
            continue
        by_depth[float(row["relative_depth"])].append(float(value))
    depths = np.array(sorted(by_depth), dtype=np.float64)
    values = np.array(
        [sum(by_depth[depth]) / len(by_depth[depth]) for depth in depths],
        dtype=np.float64,
    )
    return depths, values


def profile_cosine(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> float | None:
    if not len(left[0]) or not len(right[0]):
        return None
    grid = np.linspace(0.0, 1.0, 41)
    return cosine(
        np.interp(grid, left[0], left[1]),
        np.interp(grid, right[0], right[1]),
    )


def model_descriptives(
    model: str,
    prereg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    atlas = RESULT_ROOT / "atlas" / model
    summary = read_json(atlas / "summary.json")
    candidates = read_jsonl(atlas / "candidate_behavior.jsonl")
    natural = read_jsonl(atlas / "natural_generation_audit.jsonl")
    responses = read_jsonl(atlas / "response_metrics.jsonl")
    directions = read_jsonl(atlas / "cross_template_directions.jsonl")

    candidate_by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    natural_by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidate_by_relation[row["relation"]].append(row)
    for row in natural:
        natural_by_relation[row["relation"]].append(row)

    relation_rows = {}
    for relation in RELATIONS:
        relation_candidates = candidate_by_relation[relation]
        relation_natural = natural_by_relation[relation]
        strict_count = sum(bool(row["exact"]) for row in relation_natural)
        label_first_count = sum(
            label_first_match(
                str(row["generated_text"]),
                [str(value) for value in row["acceptable_labels"]],
            )
            for row in relation_natural
        )
        relation_rows[relation] = {
            "candidate_case_count": len(relation_candidates),
            "candidate_accuracy_recomputed": mean_or_none(
                [
                    1.0 if row["candidate_hit"] else 0.0
                    for row in relation_candidates
                ]
            ),
            "nonfinite_candidate_count": sum(
                bool(row["nonfinite_candidate"])
                for row in relation_candidates
            ),
            "natural_case_count": len(relation_natural),
            "natural_strict_exact_rate": (
                strict_count / len(relation_natural)
                if relation_natural
                else None
            ),
            "natural_label_first_rate_posthoc": (
                label_first_count / len(relation_natural)
                if relation_natural
                else None
            ),
            "natural_terminated_rate": (
                sum(bool(row["terminated"]) for row in relation_natural)
                / len(relation_natural)
                if relation_natural
                else None
            ),
            "strong_behavior_gate_passed": bool(
                summary["relations"][relation][
                    "strong_behavior_gate_passed"
                ]
            ),
        }

    query_rows = {}
    for query_type in protocol.QUERY_TYPES:
        selected = [
            row for row in candidates
            if row["query_type"] == query_type
        ]
        query_rows[query_type] = {
            "case_count": len(selected),
            "candidate_accuracy": mean_or_none(
                [1.0 if row["candidate_hit"] else 0.0 for row in selected]
            ),
        }

    task_rows = {}
    for task_kind in ("direct", "transitive"):
        selected = [
            row for row in candidates
            if row["task_kind"] == task_kind
        ]
        task_rows[task_kind] = {
            "case_count": len(selected),
            "candidate_accuracy": mean_or_none(
                [1.0 if row["candidate_hit"] else 0.0 for row in selected]
            ),
        }

    layout_rows = {}
    for layout in protocol.LAYOUTS:
        selected = [row for row in candidates if row["layout"] == layout]
        layout_rows[layout] = {
            "case_count": len(selected),
            "candidate_accuracy": mean_or_none(
                [1.0 if row["candidate_hit"] else 0.0 for row in selected]
            ),
        }

    late_relation_reuse = {}
    for relation in RELATIONS:
        late_response = [
            row for row in responses
            if row["bucket_id"] == f"relation:{relation}"
            and row["role"] == "answer_boundary"
            and float(row["relative_depth"]) >= 0.65
        ]
        late_direction = [
            row for row in directions
            if row["bucket_id"].startswith(
                f"relation_query:{relation}:"
            )
            and float(row["relative_depth"]) >= 0.65
            and int(row["discovery_pair_count"]) >= 20
            and int(row["confirmation_pair_count"]) >= 20
        ]
        late_relation_reuse[relation] = {
            "late_answer_lexical_branch_semantic_cosine_median": (
                median_or_none([
                    float(row["mean_surface_branch_semantic_cosine"])
                    for row in late_response
                    if row["mean_surface_branch_semantic_cosine"]
                    is not None
                ])
            ),
            "late_answer_interaction_relative_magnitude_median": (
                median_or_none([
                    float(row["mean_interaction_relative_magnitude"])
                    for row in late_response
                    if row["mean_interaction_relative_magnitude"]
                    is not None
                ])
            ),
            "late_cross_template_mean_direction_cosine_median": (
                median_or_none([
                    float(row[
                        "discovery_confirmation_direction_cosine"
                    ])
                    for row in late_direction
                    if row[
                        "discovery_confirmation_direction_cosine"
                    ] is not None
                ])
            ),
            "late_discovery_individual_direction_consistency_median": (
                median_or_none([
                    float(row["discovery_direction_consistency"])
                    for row in late_direction
                    if row["discovery_direction_consistency"] is not None
                ])
            ),
            "late_confirmation_individual_direction_consistency_median": (
                median_or_none([
                    float(row["confirmation_direction_consistency"])
                    for row in late_direction
                    if row["confirmation_direction_consistency"] is not None
                ])
            ),
        }

    nonfinite_cases = [
        {
            "semantic_case_index": int(row["semantic_case_index"]),
            "unit_id": row["unit_id"],
            "relation": row["relation"],
            "chain_length": int(row["chain_length"]),
            "query_type": row["query_type"],
            "layout": row["layout"],
            "state": row["state"],
        }
        for row in candidates
        if row["nonfinite_candidate"]
    ]

    archive = np.load(
        atlas / "answer_directions.fp16.npz",
        allow_pickle=False,
    )
    mean_directions = archive["mean_directions"]
    direction_counts = archive["direction_counts"]

    checks = {
        "summary_protocol_digest_matches": (
            summary["protocol_digest"] == prereg["protocol_digest"]
        ),
        "summary_case_count_is_2400": (
            int(summary["case_count"])
            == int(prereg["case_count_per_model"])
            == len(candidates)
        ),
        "natural_case_count_is_400": (
            len(natural)
            == len(RELATIONS) * int(prereg["natural_audit_per_relation"])
        ),
        "identity_maximum_is_zero": (
            float(summary["identity_maximum"]) == 0.0
        ),
        "fp16_parameters_only": (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        ),
        "relation_summary_recomputed": all(
            math.isclose(
                relation_rows[relation]["candidate_accuracy_recomputed"],
                float(summary["relations"][relation][
                    "candidate_first_token_accuracy"
                ]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                relation_rows[relation]["natural_strict_exact_rate"],
                float(summary["relations"][relation][
                    "natural_audit_exact_rate"
                ]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for relation in RELATIONS
        ),
        "npz_direction_dtype_is_fp16": (
            mean_directions.dtype == np.float16
        ),
        "npz_direction_values_finite": bool(
            np.isfinite(mean_directions).all()
        ),
        "npz_direction_counts_nonnegative": bool(
            (direction_counts >= 0).all()
        ),
    }

    return {
        "summary": {
            "n_layers": int(summary["model_info"]["n_layers"]),
            "d_model": int(summary["model_info"]["d_model"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "identity_maximum": float(summary["identity_maximum"]),
            "nonfinite_candidate_count": int(
                summary["nonfinite_candidate_count"]
            ),
        },
        "relations": relation_rows,
        "query_asymmetry": query_rows,
        "task_kind": task_rows,
        "layout": layout_rows,
        "late_relative_reuse": late_relation_reuse,
        "nonfinite_cases": nonfinite_cases,
        "row_counts": {
            "candidate_behavior": len(candidates),
            "natural_generation_audit": len(natural),
            "response_metrics": len(responses),
            "cross_template_directions": len(directions),
        },
    }, checks


def profile_controls() -> dict[str, Any]:
    metrics = {
        model: read_jsonl(
            RESULT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        for model in MODELS
    }
    profiles = {
        (model, relation): response_profile(
            metrics[model], relation
        )
        for model in MODELS
        for relation in RELATIONS
    }

    within_model_rows = []
    for model in MODELS:
        for left_relation, right_relation in itertools.combinations(
            RELATIONS, 2
        ):
            within_model_rows.append({
                "model": model,
                "left_relation": left_relation,
                "right_relation": right_relation,
                "profile_cosine": profile_cosine(
                    profiles[(model, left_relation)],
                    profiles[(model, right_relation)],
                ),
            })

    cross_model_rows = []
    pair_summaries = []
    for left_model, right_model in itertools.combinations(MODELS, 2):
        matched = []
        mismatched = []
        for left_relation in RELATIONS:
            for right_relation in RELATIONS:
                value = profile_cosine(
                    profiles[(left_model, left_relation)],
                    profiles[(right_model, right_relation)],
                )
                cross_model_rows.append({
                    "left_model": left_model,
                    "right_model": right_model,
                    "left_relation": left_relation,
                    "right_relation": right_relation,
                    "relation_matched": left_relation == right_relation,
                    "profile_cosine": value,
                })
                if value is not None:
                    if left_relation == right_relation:
                        matched.append(value)
                    else:
                        mismatched.append(value)
        matched_median = median_or_none(matched)
        mismatched_median = median_or_none(mismatched)
        pair_summaries.append({
            "left_model": left_model,
            "right_model": right_model,
            "matched_relation_profile_cosine_median": matched_median,
            "mismatched_relation_profile_cosine_median": mismatched_median,
            "matched_minus_mismatched_median": (
                matched_median - mismatched_median
                if matched_median is not None
                and mismatched_median is not None
                else None
            ),
        })

    return {
        "within_model_cross_relation_profiles": within_model_rows,
        "cross_model_all_relation_pairs": cross_model_rows,
        "cross_model_matched_vs_mismatched": pair_summaries,
        "interpretation_limit": (
            "A high matched-relation profile cosine is relation-specific "
            "only if it exceeds mismatched-relation controls."
        ),
    }


def main() -> None:
    prereg_path = RESULT_ROOT / "protocol" / "preregistration.json"
    prereg = read_json(prereg_path)
    digest_payload = dict(prereg)
    claimed_digest = digest_payload.pop("protocol_digest")
    recomputed_digest = protocol.digest(digest_payload)

    all_json_values = []
    json_file_count = 0
    jsonl_row_count = 0
    for path in RESULT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            all_json_values.append(read_json(path))
            json_file_count += 1
        elif path.suffix == ".jsonl":
            rows = read_jsonl(path)
            all_json_values.extend(rows)
            jsonl_row_count += len(rows)

    models = {}
    model_checks = {}
    for model in MODELS:
        models[model], model_checks[model] = model_descriptives(
            model, prereg
        )

    automatic_next = read_json(
        RESULT_ROOT / "analysis" / "automatic_next.json"
    )
    aggregate = read_json(RESULT_ROOT / "aggregate.json")
    global_checks = {
        "protocol_digest_recomputed": claimed_digest == recomputed_digest,
        "all_protocol_model_audits_passed": all(
            bool(prereg["model_audits"][model]["all_checks_passed"])
            for model in MODELS
        ),
        "all_json_numbers_finite_or_null": all(
            finite_tree(value) for value in all_json_values
        ),
        "aggregate_protocol_digest_matches": (
            aggregate["protocol_digest"] == claimed_digest
        ),
        "automatic_next_is_frozen_stop": (
            automatic_next["should_continue_automatically"] is False
            and automatic_next["selected_relations"] == []
            and automatic_next["route"]
            == "stop_and_repair_reasoning_behavior_protocol"
        ),
    }
    all_checks = list(global_checks.values()) + [
        value
        for checks in model_checks.values()
        for value in checks.values()
    ]

    audit = {
        "schema_version": "phase1068_integrity_audit.v1",
        "phase": 1068,
        "protocol_digest": claimed_digest,
        "recomputed_protocol_digest": recomputed_digest,
        "all_integrity_checks_passed": all(all_checks),
        "global_checks": global_checks,
        "model_checks": model_checks,
        "models": models,
        "profile_negative_controls": profile_controls(),
        "automatic_next": automatic_next,
        "strict_json_inventory": {
            "json_file_count": json_file_count,
            "jsonl_row_count": jsonl_row_count,
        },
        "numerical_warning": {
            "models_with_nonfinite_candidate_rows": [
                model for model in MODELS
                if models[model]["summary"][
                    "nonfinite_candidate_count"
                ] > 0
            ],
            "nonfinite_rows_are_excluded_from_valid_pairs": True,
            "nonfinite_rows_are_not_silently_imputed": True,
        },
        "posthoc_metrics_do_not_change_preregistered_gates": True,
        "source_hashes": {
            path.name: sha256(path)
            for path in (
                ROOT
                / "tests"
                / "glm5"
                / "phase1068_reasoning_generalization_protocol.py",
                ROOT
                / "tests"
                / "glm5"
                / "phase1068_reasoning_generalization_scan.py",
                ROOT
                / "tests"
                / "glm5"
                / "phase1068_finalize.py",
                prereg_path,
            )
        },
    }
    protocol.write_json(
        RESULT_ROOT / "analysis" / "integrity_audit.json",
        audit,
    )
    print(json.dumps({
        "phase": 1068,
        "all_integrity_checks_passed": (
            audit["all_integrity_checks_passed"]
        ),
        "automatic_next": automatic_next,
        "nonfinite_models": audit["numerical_warning"][
            "models_with_nonfinite_candidate_rows"
        ],
        "profile_controls": audit["profile_negative_controls"][
            "cross_model_matched_vs_mismatched"
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
