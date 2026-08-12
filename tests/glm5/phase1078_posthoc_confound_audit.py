#!/usr/bin/env python3
"""Posthoc Phase1078 audit for local claim-token and output-axis confounds.

This diagnostic is explicitly not a preregistered evidence gate.  It asks how
much family retrieval remains after excluding the changed claim-predicate
position and, separately, the answer boundary.
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1078_shared_shell_pattern_atlas_protocol as protocol
import phase1078_shared_shell_pattern_atlas_finalize as finalizer


ROLE_VARIANTS = {
    "full": tuple(protocol.CAPTURE_ROLES),
    "without_claim_predicate": tuple(
        role
        for role in protocol.CAPTURE_ROLES
        if role != "claim_predicate"
    ),
    "without_claim_predicate_or_answer_boundary": (
        "evidence_anchor",
        "claim_subject",
        "decision_cue",
    ),
    "source_roles_only": (
        "evidence_anchor",
        "claim_subject",
    ),
}


def build_profile(
    rows: list[dict[str, Any]],
    *,
    family: str,
    split: str,
    roles: tuple[str, ...],
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and row["split"] == split
        and row["conditioning"] == "all_finite"
    ]
    channels = []
    for metric in finalizer.PROFILE_METRICS:
        values = []
        for component in finalizer.COMPONENTS:
            for role in roles:
                values.append(finalizer.interpolate_channel(
                    selected,
                    component,
                    role,
                    metric,
                ))
        channels.append(
            finalizer.channel_normalize(np.concatenate(values))
        )
    return np.concatenate(channels).astype(np.float64)


def assignment_collection(
    profiles: dict[tuple[str, str, str], np.ndarray],
    *,
    variant: str,
    centered: bool,
) -> list[dict[str, Any]]:
    rows = []
    comparisons = []
    for model in protocol.MODELS:
        comparisons.append((
            "within_model_discovery_to_confirmation",
            model,
            "discovery",
            model,
            "confirmation",
        ))
    for left, right in itertools.combinations(protocol.MODELS, 2):
        comparisons.extend([
            (
                "cross_model_confirmation",
                left,
                "confirmation",
                right,
                "confirmation",
            ),
            (
                "cross_model_confirmation",
                right,
                "confirmation",
                left,
                "confirmation",
            ),
        ])
    for comparison, sm, ss, tm, ts in comparisons:
        matrix = finalizer.similarity_matrix(
            profiles,
            source_model=sm,
            source_split=ss,
            target_model=tm,
            target_split=ts,
        )
        result = finalizer.exact_assignment_test(matrix)
        rows.append({
            "role_variant": variant,
            "profile": "family_centered" if centered else "raw",
            "comparison": comparison,
            "source_model": sm,
            "target_model": tm,
            "top1_correct": result["top1_correct"],
            "family_count": result["family_count"],
            "exact_upper_tail_p": result["exact_upper_tail_p"],
            "minimum_margin_over_best_other": result[
                "minimum_margin_over_best_other"
            ],
            "family_rows": result["rows"],
        })
    return rows


def role_dominance(
    metrics_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_model = {}
    claim_peak_count = 0
    pair_count = 0
    for model, rows in metrics_by_model.items():
        by_family = {}
        for family in protocol.FAMILIES:
            selected = [
                row
                for row in rows
                if row["conditioning"] == "all_finite"
                and row["split"] == "confirmation"
                and row["family"] == family
                and row["mean_truth_relative_magnitude"] is not None
            ]
            role_values: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                role_values[str(row["role"])].append(
                    float(row["mean_truth_relative_magnitude"])
                )
            totals = {
                role: float(np.sum(values))
                for role, values in role_values.items()
            }
            total = sum(totals.values())
            shares = {
                role: value / total if total else 0.0
                for role, value in totals.items()
            }
            peak = max(totals, key=totals.get)
            other = max(
                value
                for role, value in totals.items()
                if role != "claim_predicate"
            )
            pair_count += 1
            claim_peak_count += int(peak == "claim_predicate")
            by_family[family] = {
                "peak_role": peak,
                "role_response_share": shares,
                "claim_predicate_to_best_other_total_ratio": (
                    totals["claim_predicate"] / other
                    if other > finalizer.EPSILON
                    else None
                ),
            }
        by_model[model] = by_family
    return {
        "by_model": by_model,
        "claim_predicate_peak_count": claim_peak_count,
        "model_family_pair_count": pair_count,
        "claim_predicate_peak_rate": (
            claim_peak_count / pair_count if pair_count else None
        ),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    metrics_by_model = {
        model: protocol.read_jsonl(
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "response_metrics.jsonl"
        )
        for model in protocol.MODELS
    }
    assignment_rows = []
    for variant, roles in ROLE_VARIANTS.items():
        profiles = {}
        for model in protocol.MODELS:
            for family in protocol.FAMILIES:
                for split in protocol.SPLITS:
                    profiles[(model, family, split)] = build_profile(
                        metrics_by_model[model],
                        family=family,
                        split=split,
                        roles=roles,
                    )
        centered_profiles = {}
        for model in protocol.MODELS:
            for split in protocol.SPLITS:
                matrix = np.stack([
                    profiles[(model, family, split)]
                    for family in protocol.FAMILIES
                ])
                mean = matrix.mean(axis=0)
                for index, family in enumerate(protocol.FAMILIES):
                    centered_profiles[(model, family, split)] = (
                        matrix[index] - mean
                    )
        assignment_rows.extend(assignment_collection(
            profiles,
            variant=variant,
            centered=False,
        ))
        assignment_rows.extend(assignment_collection(
            centered_profiles,
            variant=variant,
            centered=True,
        ))

    summary = {}
    for variant in ROLE_VARIANTS:
        summary[variant] = {}
        for profile in ("raw", "family_centered"):
            selected = [
                row
                for row in assignment_rows
                if row["role_variant"] == variant
                and row["profile"] == profile
            ]
            within = [
                row for row in selected
                if row["comparison"]
                == "within_model_discovery_to_confirmation"
            ]
            cross = [
                row for row in selected
                if row["comparison"] == "cross_model_confirmation"
            ]
            summary[variant][profile] = {
                "within_top1_mean": float(np.mean([
                    row["top1_correct"] for row in within
                ])),
                "within_all_exact_p_minimum": all(
                    row["exact_upper_tail_p"]
                    == 1.0 / 40320.0
                    for row in within
                ),
                "cross_top1_mean": float(np.mean([
                    row["top1_correct"] for row in cross
                ])),
                "cross_top1_min": min(
                    row["top1_correct"] for row in cross
                ),
                "cross_top1_max": max(
                    row["top1_correct"] for row in cross
                ),
                "cross_exact_p_le_0_01_count": sum(
                    row["exact_upper_tail_p"] <= 0.01
                    for row in cross
                ),
                "cross_comparison_count": len(cross),
            }

    payload = {
        "schema_version": "phase1078_posthoc_confound_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "preregistered_evidence_gate": False,
        "role_variants": {
            key: list(value)
            for key, value in ROLE_VARIANTS.items()
        },
        "role_dominance": role_dominance(metrics_by_model),
        "assignment_summary": summary,
        "assignment_rows": assignment_rows,
        "interpretation": (
            "This audit measures robustness to removing the locally changed "
            "claim predicate and the common answer boundary. Persistence "
            "supports a distributed family response signature; collapse "
            "would indicate that retrieval was dominated by local lexical "
            "or output-protocol geometry. Neither outcome establishes a "
            "causal mechanism."
        ),
    }
    payload["audit_digest"] = protocol.digest(payload)
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "posthoc_confound_audit.json",
        payload,
    )
    print({
        "phase": protocol.PHASE,
        "role_dominance": payload["role_dominance"][
            "claim_predicate_peak_rate"
        ],
        "assignment_summary": summary,
        "audit_digest": payload["audit_digest"],
    })


if __name__ == "__main__":
    main()
