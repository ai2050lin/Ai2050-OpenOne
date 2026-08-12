#!/usr/bin/env python3
"""Descriptive, explicitly post-hoc diagnostics for Phase1070."""

from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1070_process_answer_protocol as protocol


PROCESS_FIELDS = (
    "mean_process_did_relative_magnitude",
    "mean_process_lexical_reuse_cosine",
    "mean_process_answer_invariance_cosine",
    "mean_process_answer_absolute_cosine",
)
ANSWER_FIELDS = (
    "mean_answer_relative_magnitude",
    "mean_answer_lexical_reuse_cosine",
    "mean_answer_path_invariance_cosine",
)


def median(values: list[Any]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(statistics.median(clean)) if clean else None


def mean(values: list[Any]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(statistics.fmean(clean)) if clean else None


def aggregate_fields(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, float | None]:
    return {
        field: median([row[field] for row in rows])
        for field in fields
    }


def path_behavior(
    summary: dict[str, Any],
) -> dict[str, Any]:
    relations = {}
    for relation, values in summary["relations"].items():
        paths = values["by_path"]
        shortcut = paths["shortcut_only"][
            "candidate_accuracy"
        ]
        transitive = paths["transitive_only"][
            "candidate_accuracy"
        ]
        duplicated = paths["duplicated_direct"][
            "candidate_accuracy"
        ]
        direct_bridge = paths["direct_plus_bridge"][
            "candidate_accuracy"
        ]
        natural_shortcut = paths["shortcut_only"][
            "semantic_first_natural_rate"
        ]
        natural_transitive = paths["transitive_only"][
            "semantic_first_natural_rate"
        ]
        natural_duplicated = paths["duplicated_direct"][
            "semantic_first_natural_rate"
        ]
        natural_direct_bridge = paths["direct_plus_bridge"][
            "semantic_first_natural_rate"
        ]
        relations[relation] = {
            "candidate_accuracy_by_path": {
                path: row["candidate_accuracy"]
                for path, row in paths.items()
            },
            "semantic_first_natural_rate_by_path": {
                path: row["semantic_first_natural_rate"]
                for path, row in paths.items()
            },
            "candidate_transitive_minus_shortcut": (
                transitive - shortcut
            ),
            "candidate_control_switch_gap": (
                direct_bridge - duplicated
            ),
            "candidate_behavior_difference_in_differences": (
                (transitive - shortcut)
                - (direct_bridge - duplicated)
            ),
            "natural_transitive_minus_shortcut": (
                natural_transitive - natural_shortcut
            ),
            "natural_control_switch_gap": (
                natural_direct_bridge - natural_duplicated
            ),
            "natural_behavior_difference_in_differences": (
                (natural_transitive - natural_shortcut)
                - (
                    natural_direct_bridge
                    - natural_duplicated
                )
            ),
        }
    return relations


def first_coverage_drop(
    rows: list[dict[str, Any]],
    count_field: str,
    threshold: float = 0.995,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["conditioning"] == "all"
    ]
    maximum = max(int(row[count_field]) for row in selected)
    by_depth: dict[float, list[int]] = defaultdict(list)
    for row in selected:
        by_depth[float(row["relative_depth"])].append(
            int(row[count_field])
        )
    depth_rows = []
    for depth, counts in sorted(by_depth.items()):
        coverage = (
            sum(counts) / (len(counts) * maximum)
            if maximum else 0.0
        )
        depth_rows.append({
            "relative_depth": depth,
            "coverage": coverage,
            "minimum_count": min(counts),
            "maximum_count": max(counts),
        })
    first = next(
        (
            row for row in depth_rows
            if row["coverage"] < threshold
        ),
        None,
    )
    return {
        "expected_count_per_cell": maximum,
        "first_depth_below_0_995": first,
        "final_depth": depth_rows[-1],
    }


def depth_role_peak(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["conditioning"] == "all"
        and row[
            "mean_process_did_relative_magnitude"
        ] is not None
    ]
    grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in selected:
        grouped[(
            float(row["relative_depth"]),
            str(row["role"]),
        )].append(
            float(row[
                "mean_process_did_relative_magnitude"
            ])
        )
    summaries = [
        {
            "relative_depth": depth,
            "role": role,
            "mean_process_did_relative_magnitude": (
                statistics.fmean(values)
            ),
        }
        for (depth, role), values in grouped.items()
    ]
    peak = max(
        summaries,
        key=lambda row: row[
            "mean_process_did_relative_magnitude"
        ],
    )
    by_role = {}
    for role in protocol.CAPTURE_ROLES:
        values = [
            row
            for row in summaries
            if row["role"] == role
        ]
        by_role[role] = max(
            values,
            key=lambda row: row[
                "mean_process_did_relative_magnitude"
            ],
        )
    return {
        "global_peak": peak,
        "peak_by_role": by_role,
        "note": (
            "Peak locations are post-hoc descriptions, not frozen "
            "localization gates."
        ),
    }


def model_diagnostics(model: str) -> dict[str, Any]:
    atlas = protocol.OUT_ROOT / "atlas" / model
    summary = protocol.read_json(atlas / "summary.json")
    responses = protocol.read_jsonl(
        atlas / "response_metrics.jsonl"
    )
    readouts = protocol.read_jsonl(
        atlas / "local_readout_metrics.jsonl"
    )
    process_window = [
        row
        for row in responses
        if row["role"] in protocol.PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
    ]
    all_process = [
        row for row in process_window
        if row["conditioning"] == "all"
    ]
    conditioned_process = [
        row for row in process_window
        if row["conditioning"] == "behavior_conditioned"
    ]
    all_answer = [
        row for row in process_window
        if row["conditioning"] == "all"
    ]
    conditioned_answer = [
        row for row in process_window
        if row["conditioning"] == "behavior_conditioned"
    ]
    late_readout = [
        row
        for row in readouts
        if row["conditioning"] == "all"
        and float(row["relative_depth"])
        >= protocol.GATES["late_depth_start"]
    ]
    conditioned_late_readout = [
        row
        for row in readouts
        if row["conditioning"] == "behavior_conditioned"
        and float(row["relative_depth"])
        >= protocol.GATES["late_depth_start"]
    ]
    all_process_fields = aggregate_fields(
        all_process, PROCESS_FIELDS
    )
    conditioned_process_fields = aggregate_fields(
        conditioned_process, PROCESS_FIELDS
    )
    selection_delta = {
        field: (
            conditioned_process_fields[field]
            - all_process_fields[field]
            if conditioned_process_fields[field] is not None
            and all_process_fields[field] is not None
            else None
        )
        for field in PROCESS_FIELDS
    }
    all_answer_fields = aggregate_fields(
        all_answer, ANSWER_FIELDS
    )
    conditioned_answer_fields = aggregate_fields(
        conditioned_answer, ANSWER_FIELDS
    )
    answer_selection_delta = {
        field: (
            conditioned_answer_fields[field]
            - all_answer_fields[field]
            if conditioned_answer_fields[field] is not None
            and all_answer_fields[field] is not None
            else None
        )
        for field in ANSWER_FIELDS
    }
    readout_summary = {
        "late_matched_answer_positive_rate": median([
            row["matched_answer_positive_rate"]
            for row in late_readout
        ]),
        "late_mismatched_answer_positive_rate": median([
            row["mismatched_answer_positive_rate"]
            for row in late_readout
        ]),
        "late_positive_rate_gap": median([
            row["positive_rate_gap"]
            for row in late_readout
        ]),
        "late_process_to_answer_readout_ratio": median([
            row["absolute_process_to_answer_readout_ratio"]
            for row in late_readout
        ]),
        "conditioned_late_positive_rate_gap": median([
            row["positive_rate_gap"]
            for row in conditioned_late_readout
        ]),
        "conditioned_late_process_to_answer_readout_ratio": median([
            row["absolute_process_to_answer_readout_ratio"]
            for row in conditioned_late_readout
        ]),
    }
    role_window_metrics = {}
    for role in protocol.CAPTURE_ROLES:
        role_rows = [
            row for row in all_process
            if row["role"] == role
        ]
        role_window_metrics[role] = aggregate_fields(
            role_rows, PROCESS_FIELDS
        )
    boundary_rows = [
        row for row in all_process
        if row["role"] == "answer_boundary"
    ]
    lower_rows = [
        row
        for row in responses
        if row["conditioning"] == "all"
        and row["role"] == "lower_edge"
    ]
    return {
        "model": model,
        "behavior_by_relation": path_behavior(summary),
        "all_pair_process_window": all_process_fields,
        "behavior_conditioned_process_window": (
            conditioned_process_fields
        ),
        "behavior_conditioning_process_delta": selection_delta,
        "all_pair_answer_window": all_answer_fields,
        "behavior_conditioned_answer_window": (
            conditioned_answer_fields
        ),
        "behavior_conditioning_answer_delta": (
            answer_selection_delta
        ),
        "late_local_readout": readout_summary,
        "process_window_by_role": role_window_metrics,
        "causally_exposed_answer_boundary": aggregate_fields(
            boundary_rows, PROCESS_FIELDS
        ),
        "causal_mask_sanity_control": {
            "lower_edge_process_did_maximum": max(
                float(row[
                    "mean_process_did_relative_magnitude"
                ])
                for row in lower_rows
                if row[
                    "mean_process_did_relative_magnitude"
                ] is not None
            ),
            "explanation": (
                "Across frozen layouts, lower_edge never occurs after both "
                "the switch manipulation and the anchor manipulation. An "
                "exactly zero two-factor DiD is therefore a causal-order "
                "sanity control, not evidence that the lower premise is "
                "generally unused."
            ),
        },
        "posthoc_process_peak": depth_role_peak(responses),
        "numerical_coverage": {
            "process_did": first_coverage_drop(
                responses,
                "process_did_relative_magnitude_count",
            ),
            "process_lexical_reuse": first_coverage_drop(
                responses,
                "process_lexical_reuse_cosine_count",
            ),
            "answer_relative_magnitude": first_coverage_drop(
                responses,
                "answer_relative_magnitude_count",
            ),
        },
        "model_summary_finite_rates": {
            "candidate": summary["candidate_finite_rate"],
            "residual_metric": summary[
                "residual_metric_finite_rate"
            ],
            "internal_readout": summary[
                "internal_readout_finite_rate"
            ],
        },
    }


def main() -> None:
    relation_evidence = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "relation_evidence.jsonl"
    )
    model_gates = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "model_gates.jsonl"
    )
    diagnostics = {
        model: model_diagnostics(model)
        for model in protocol.MODELS
    }
    failed_checks = defaultdict(int)
    for row in relation_evidence:
        if not row["behavior_gate_passed"]:
            failed_checks["behavior_gate"] += 1
        if not row["numerical_gate_passed"]:
            failed_checks["numerical_gate"] += 1
        for split in protocol.SPLITS:
            for check, passed in row[split]["checks"].items():
                if not passed:
                    failed_checks[
                        f"{split}:{check}"
                    ] += 1
        for check, passed in row["profile_checks"].items():
            if not passed:
                failed_checks[f"profile:{check}"] += 1
    payload = {
        "schema_version": "phase1070_posthoc_diagnostics.v1",
        "phase": protocol.PHASE,
        "status": (
            "Descriptive post-hoc analysis. It cannot change the frozen "
            "automatic-next decision or authorize component localization."
        ),
        "models": diagnostics,
        "failed_gate_counts_across_15_model_relations": dict(
            sorted(failed_checks.items())
        ),
        "model_gates": model_gates,
        "hypothesis_status": {
            "relative_context_conditioning": (
                "Supported locally: the matched process DiD is exactly "
                "zero at embeddings and nonzero after contextual layers. "
                "This does not establish a universal relative-code law."
            ),
            "answer_invariant_process_reuse": (
                "Partially supported: process DiD directions reuse across "
                "answer identities more than across held-out lexical "
                "realizations."
            ),
            "lexically_invariant_process_rule": (
                "Not established: frozen lexical-reuse gates fail broadly, "
                "especially on confirmation wording."
            ),
            "brain_optimality_and_homology": (
                "Not tested. No Phase1070 measurement can infer biological "
                "homology, evolutionary optimality, or energy efficiency."
            ),
            "word_ecological_niches": (
                "Not directly tested. Unique token readout coordinates are "
                "compatible with, but do not prove, stable ecological "
                "niches."
            ),
            "complete_internal_language_rule": (
                "Not established. The atlas isolates a repeated process-"
                "sensitive interaction and a stronger answer field, not a "
                "complete grammar/logic/knowledge mechanism."
            ),
            "causal_exposure_aware_measurement": (
                "Required next: the frozen pooled process-role gate mixed "
                "positions before evidence with positions after evidence. "
                "Answer-boundary post-hoc evidence is materially stronger, "
                "but cannot retroactively change the frozen decision."
            ),
        },
    }
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "posthoc_diagnostics.json",
        payload,
    )
    print({
        "phase": protocol.PHASE,
        "failed_gate_counts": payload[
            "failed_gate_counts_across_15_model_relations"
        ],
        "models": {
            model: {
                "process": row["all_pair_process_window"],
                "readout": row["late_local_readout"],
                "finite": row["model_summary_finite_rates"],
            }
            for model, row in diagnostics.items()
        },
    })


if __name__ == "__main__":
    main()
