#!/usr/bin/env python3
"""Finalize Phase1075 held-out internal evidence and automatic route."""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1075_relation_polarity_protocol as protocol


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return float("nan")
    a = left[finite].astype(np.float64)
    b = right[finite].astype(np.float64)
    denominator = float(
        np.linalg.norm(a) * np.linalg.norm(b)
    )
    if denominator <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def profile(
    rows: list[dict[str, Any]],
    field: str,
    *,
    split: str | None = None,
    path: str | None = None,
    role: str = "answer_boundary",
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["role"] == role
        and (split is None or row["split"] == split)
        and (path is None or row["path"] == path)
        and row.get(field) is not None
    ]
    by_depth: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        by_depth[int(row["depth"])].append(float(row[field]))
    if not by_depth:
        return np.asarray([], dtype=np.float64)
    return np.asarray([
        float(np.mean(by_depth[depth]))
        for depth in range(max(by_depth) + 1)
    ])


def resample(values: np.ndarray, width: int = 21) -> np.ndarray:
    if values.size == 0:
        return np.full(width, np.nan, dtype=np.float64)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, width)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full(width, np.nan, dtype=np.float64)
    return np.interp(target, source[finite], values[finite])


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return float("inf") if numerator > 0.0 else 1.0
    return numerator / denominator


def routing_evidence(
    npz_path: Path,
    relation: str,
    top_count: int,
) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        sums = data["sums"]
        counts = data["counts"]
        positives = data["positive_counts"]
        relations = [str(value) for value in data["relations"]]
        splits = [str(value) for value in data["splits"]]
        destinations = [
            str(value) for value in data["destinations"]
        ]
        source_pairs = [
            str(value) for value in data["source_pairs"]
        ]
        metrics = [str(value) for value in data["metrics"]]
        conditionings = [
            str(value) for value in data["conditionings"]
        ]
    r = relations.index(relation)
    discovery = splits.index("discovery")
    confirmation = splits.index("confirmation")
    destination = destinations.index("answer_boundary")
    fact = source_pairs.index("fact")
    null = source_pairs.index("null_control")
    metric = metrics.index("attention_mass")
    conditioning = conditionings.index("all")

    def aggregate(split_slot: int, pair_slot: int):
        selected_sums = sums[
            r,
            split_slot,
            :,
            :,
            :,
            :,
            destination,
            pair_slot,
            metric,
            conditioning,
        ]
        selected_counts = counts[
            r,
            split_slot,
            :,
            :,
            :,
            :,
            destination,
            pair_slot,
            metric,
            conditioning,
        ]
        selected_positive = positives[
            r,
            split_slot,
            :,
            :,
            :,
            :,
            destination,
            pair_slot,
            metric,
            conditioning,
        ]
        total_sums = selected_sums.sum(axis=(0, 1))
        total_counts = selected_counts.sum(axis=(0, 1))
        total_positive = selected_positive.sum(axis=(0, 1))
        means = np.divide(
            total_sums,
            total_counts,
            out=np.full_like(total_sums, np.nan, dtype=np.float64),
            where=total_counts > 0,
        )
        positive_fraction = np.divide(
            total_positive,
            total_counts,
            out=np.full_like(total_sums, np.nan, dtype=np.float64),
            where=total_counts > 0,
        )
        return means, positive_fraction, total_counts, total_positive

    discovery_mean, discovery_pos, _, _ = aggregate(
        discovery, fact
    )
    ranked = []
    for depth in range(discovery_mean.shape[0]):
        for head in range(discovery_mean.shape[1]):
            mean_value = float(discovery_mean[depth, head])
            pos_value = float(discovery_pos[depth, head])
            if not (
                math.isfinite(mean_value)
                and math.isfinite(pos_value)
            ):
                continue
            ranked.append((
                mean_value,
                pos_value,
                -depth,
                -head,
                depth,
                head,
            ))
    ranked.sort(reverse=True)
    selected_heads = [
        (item[-2], item[-1])
        for item in ranked[:top_count]
    ]
    confirmation_mean, confirmation_pos, confirmation_count, (
        confirmation_positive
    ) = aggregate(confirmation, fact)
    null_mean, _, _, _ = aggregate(confirmation, null)
    selected_confirmation = np.asarray([
        confirmation_mean[depth, head]
        for depth, head in selected_heads
    ])
    selected_null = np.asarray([
        null_mean[depth, head]
        for depth, head in selected_heads
    ])
    total_count = sum(
        int(confirmation_count[depth, head])
        for depth, head in selected_heads
    )
    total_positive = sum(
        int(confirmation_positive[depth, head])
        for depth, head in selected_heads
    )
    confirmation_positive_fraction = (
        total_positive / total_count if total_count else 0.0
    )
    fact_abs = float(np.nanmean(np.abs(selected_confirmation)))
    null_abs = float(np.nanmean(np.abs(selected_null)))
    return {
        "selected_head_count": len(selected_heads),
        "selected_heads": [
            {
                "depth": depth + 1,
                "head": head,
                "discovery_mean": float(
                    discovery_mean[depth, head]
                ),
                "discovery_positive_fraction": float(
                    discovery_pos[depth, head]
                ),
                "confirmation_mean": float(
                    confirmation_mean[depth, head]
                ),
                "confirmation_positive_fraction": float(
                    confirmation_pos[depth, head]
                ),
                "confirmation_null_mean": float(
                    null_mean[depth, head]
                ),
            }
            for depth, head in selected_heads
        ],
        "confirmation_positive_fraction": (
            confirmation_positive_fraction
        ),
        "confirmation_fact_abs_mean": fact_abs,
        "confirmation_null_abs_mean": null_abs,
        "confirmation_fact_to_null_ratio": ratio(
            fact_abs, null_abs
        ),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    decision = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_decision.json"
    )
    internal_prereg = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "internal_preregistration.json"
    )
    analysis_dir = protocol.OUT_ROOT / "analysis"
    if not decision["should_run_internal_mapping"]:
        payload = {
            "schema_version": "phase1075_automatic_next.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "should_continue_automatically": False,
            "route": "stop_at_heldout_behavior",
            "reason": decision["reason"],
            "repeated_internal_relations": [],
        }
        protocol.write_json(
            analysis_dir / "automatic_next.json", payload
        )
        protocol.write_json(
            analysis_dir / "final_summary.json",
            {
                "schema_version": "phase1075_final_summary.v1",
                "phase": protocol.PHASE,
                "behavior_decision": decision,
                "internal_mapping_ran": False,
                "automatic_next": payload,
            },
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    gates = internal_prereg["claim_gates"]
    top_count = int(
        internal_prereg["candidate_selection"][
            "top_heads_per_relation_model"
        ]
    )
    evidence_rows = []
    routing_rows = []
    local_profiles: dict[
        tuple[str, str], np.ndarray
    ] = {}
    for model in decision["selected_models"]:
        summary_path = (
            protocol.OUT_ROOT / "internal" / model / "summary.json"
        )
        if not summary_path.exists():
            continue
        summary = protocol.read_json(summary_path)
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "internal"
            / model
            / "unit_metrics.jsonl"
        )
        for relation in summary["authorized_relations"]:
            relation_rows = [
                row for row in rows if row["relation"] == relation
            ]
            local_discovery = profile(
                relation_rows,
                "local_selection_separation",
                split="discovery",
            )
            local_confirmation = profile(
                relation_rows,
                "local_selection_separation",
                split="confirmation",
            )
            raw_discovery = profile(
                relation_rows,
                "raw_interaction_relative_magnitude",
                split="discovery",
            )
            raw_confirmation = profile(
                relation_rows,
                "raw_interaction_relative_magnitude",
                split="confirmation",
            )
            local_direct = profile(
                relation_rows,
                "local_selection_separation",
                path="direct",
            )
            local_transitive = profile(
                relation_rows,
                "local_selection_separation",
                path="transitive",
            )
            late_confirmation = [
                row
                for row in relation_rows
                if row["split"] == "confirmation"
                and row["role"] == "answer_boundary"
                and float(row["relative_depth"]) >= 0.75
                and row["local_selection_separation"] is not None
            ]
            local_positive = float(np.mean([
                float(row["local_selection_separation"]) > 0.0
                for row in late_confirmation
            ])) if late_confirmation else 0.0
            prebranch_values = [
                float(row["raw_interaction_relative_magnitude"])
                for row in relation_rows
                if row["role"] in protocol.PRE_BRANCH_ROLES
                and row["raw_interaction_relative_magnitude"]
                is not None
            ]
            prebranch_max = (
                max(prebranch_values) if prebranch_values else float("inf")
            )
            routing = routing_evidence(
                protocol.OUT_ROOT
                / "internal"
                / model
                / "routing_aggregates.npz",
                relation,
                top_count,
            )
            routing_checks = {
                "attention_confirmation_positive": (
                    routing[
                        "confirmation_positive_fraction"
                    ]
                    >= gates[
                        "attention_confirmation_positive_min"
                    ]
                ),
                "attention_fact_to_null_ratio": (
                    routing[
                        "confirmation_fact_to_null_ratio"
                    ]
                    >= gates[
                        "attention_fact_to_null_ratio_min"
                    ]
                ),
            }
            checks = {
                "internal_finite_rate": (
                    summary["residual_metric_finite_rate"]
                    >= gates["internal_finite_rate_min"]
                    and summary["routing_metric_finite_rate"]
                    >= gates["internal_finite_rate_min"]
                ),
                "prebranch_interaction": (
                    prebranch_max
                    <= gates[
                        "prebranch_interaction_relative_max"
                    ]
                ),
                "local_confirmation_positive": (
                    local_positive
                    >= gates[
                        "local_selection_confirmation_positive_min"
                    ]
                ),
                "local_split_profile": (
                    cosine(local_discovery, local_confirmation)
                    >= gates[
                        "local_selection_split_profile_cosine_min"
                    ]
                ),
                "local_path_profile": (
                    cosine(local_direct, local_transitive)
                    >= gates[
                        "local_selection_path_profile_cosine_min"
                    ]
                ),
                "raw_split_profile": (
                    cosine(raw_discovery, raw_confirmation)
                    >= gates[
                        "raw_interaction_split_profile_cosine_min"
                    ]
                ),
                **routing_checks,
            }
            evidence = {
                "schema_version": (
                    "phase1075_model_relation_internal_evidence.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                "relation": relation,
                "prebranch_raw_interaction_max": prebranch_max,
                "late_confirmation_local_positive_fraction": (
                    local_positive
                ),
                "local_discovery_confirmation_cosine": cosine(
                    local_discovery, local_confirmation
                ),
                "local_direct_transitive_cosine": cosine(
                    local_direct, local_transitive
                ),
                "raw_discovery_confirmation_cosine": cosine(
                    raw_discovery, raw_confirmation
                ),
                "local_confirmation_peak": (
                    float(np.nanmax(local_confirmation))
                    if local_confirmation.size
                    else None
                ),
                "raw_confirmation_peak": (
                    float(np.nanmax(raw_confirmation))
                    if raw_confirmation.size
                    else None
                ),
                "routing": {
                    key: value
                    for key, value in routing.items()
                    if key != "selected_heads"
                },
                "gate_checks": checks,
                "model_relation_internal_gate_passed": all(
                    checks.values()
                ),
            }
            evidence_rows.append(evidence)
            routing_rows.append({
                "schema_version": (
                    "phase1075_routing_candidate_confirmation.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                "relation": relation,
                **routing,
            })
            local_profiles[(model, relation)] = resample(
                local_confirmation
            )

    cross_model_rows = []
    repeated_internal_relations = []
    for relation in decision["selected_relations"]:
        passing_models = [
            row["model"]
            for row in evidence_rows
            if row["relation"] == relation
            and row["model_relation_internal_gate_passed"]
        ]
        pair_cosines = []
        for left, right in itertools.combinations(
            sorted(passing_models), 2
        ):
            value = cosine(
                local_profiles[(left, relation)],
                local_profiles[(right, relation)],
            )
            pair_cosines.append({
                "left_model": left,
                "right_model": right,
                "local_confirmation_profile_cosine": value,
            })
        minimum_cosine = (
            min(
                row["local_confirmation_profile_cosine"]
                for row in pair_cosines
            )
            if pair_cosines
            else float("nan")
        )
        repeated = bool(
            len(passing_models)
            >= gates["minimum_internal_models_per_relation"]
            and pair_cosines
            and minimum_cosine
            >= gates["cross_model_local_profile_cosine_min"]
        )
        if repeated:
            repeated_internal_relations.append(relation)
        cross_model_rows.append({
            "schema_version": (
                "phase1075_cross_model_internal_evidence.v1"
            ),
            "phase": protocol.PHASE,
            "relation": relation,
            "passing_models": passing_models,
            "pair_profile_cosines": pair_cosines,
            "minimum_pair_profile_cosine": (
                minimum_cosine
                if math.isfinite(minimum_cosine)
                else None
            ),
            "cross_model_internal_gate_passed": repeated,
        })

    should_continue = bool(repeated_internal_relations)
    automatic_next = {
        "schema_version": "phase1075_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "internal_preregistration_digest": internal_prereg[
            "internal_preregistration_digest"
        ],
        "should_continue_automatically": should_continue,
        "route": (
            "freeze_targeted_causal_validation"
            if should_continue
            else "stop_at_heldout_internal_atlas"
        ),
        "reason": (
            "At least one relation repeated its held-out local-selection "
            "and discovery-to-confirmation routing evidence in two or "
            "more models."
            if should_continue
            else (
                "No relation passed the complete held-out internal gate "
                "in two models with a repeated normalized-depth profile."
            )
        ),
        "repeated_internal_relations": repeated_internal_relations,
    }
    protocol.write_jsonl(
        analysis_dir / "model_relation_internal_evidence.jsonl",
        evidence_rows,
    )
    protocol.write_jsonl(
        analysis_dir / "routing_candidate_confirmation.jsonl",
        routing_rows,
    )
    protocol.write_jsonl(
        analysis_dir / "cross_model_internal_evidence.jsonl",
        cross_model_rows,
    )
    protocol.write_json(
        analysis_dir / "automatic_next.json", automatic_next
    )
    final_summary = {
        "schema_version": "phase1075_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_decision": decision,
        "internal_mapping_ran": True,
        "model_relation_internal_evidence": evidence_rows,
        "cross_model_internal_evidence": cross_model_rows,
        "automatic_next": automatic_next,
    }
    protocol.write_json(
        analysis_dir / "final_summary.json", final_summary
    )
    print(json.dumps(automatic_next, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
