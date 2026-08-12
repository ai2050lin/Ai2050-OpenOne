#!/usr/bin/env python3
"""Finalize Phase1080 as a descriptive atlas, never causal evidence."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1079_output_orthogonal_pattern_finalize as atlas_math
import phase1080_natural_relevance_atlas_protocol as protocol


EPSILON = 1e-12
FIELD_COLUMNS = {
    "relevance": "mean_relevance_relative_magnitude",
    "presence": "mean_presence_relative_magnitude",
    "total": "mean_total_relative_magnitude",
    "infer_answer": "mean_infer_answer_relative_magnitude",
    "direct_answer": "mean_direct_answer_relative_magnitude",
}

# Reuse only the preregistered profile interpolation and exact finite-label
# permutation machinery.  These are measurement utilities, not a theory.
atlas_math.protocol = protocol
atlas_math.FIELD_COLUMNS = FIELD_COLUMNS


def safe_median(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def assignment(**kwargs: Any) -> dict[str, Any]:
    row = atlas_math.assignment_record(**kwargs)
    row["schema_version"] = "phase1080_assignment.v1"
    row["phase"] = protocol.PHASE
    return row


def find_assignment(
    rows: list[dict[str, Any]], **criteria: Any
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(selected) != 1:
        raise RuntimeError(f"assignment lookup is not unique: {criteria}")
    return selected[0]


def assignment_passes(
    row: dict[str, Any], threshold_p: float, minimum_top1: int
) -> bool:
    return (
        float(row["exact_upper_tail_p"]) <= threshold_p
        and int(row["top1_correct"]) >= minimum_top1
    )


def family_correct(row: dict[str, Any], family: str) -> bool:
    return bool(next(
        value["correct"] for value in row["rows"]
        if value["family"] == family
    ))


def behavior_models(
    summaries: dict[str, dict[str, Any]], family: str
) -> list[str]:
    candidate_threshold = float(
        protocol.EVIDENCE_THRESHOLDS[
            "candidate_accuracy_for_behavior_annotation"
        ]
    )
    generation_threshold = float(
        protocol.EVIDENCE_THRESHOLDS["generation_first_accuracy"]
    )
    passing = []
    for model_name, summary in summaries.items():
        split_passes = []
        for split in protocol.SPLITS:
            rows = summary["behavior_summary"][family][split]
            candidates = [
                rows[branch]["candidate_accuracy"]
                for branch in protocol.BRANCHES
            ]
            infer_generation = rows["infer"][
                "generation_semantic_first_accuracy"
            ]
            split_passes.append(
                all(
                    value is not None
                    and float(value) >= candidate_threshold
                    for value in candidates
                )
                and infer_generation is not None
                and float(infer_generation) >= generation_threshold
            )
        if all(split_passes):
            passing.append(model_name)
    return passing


def factor_ratios(
    metrics_by_model: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name, rows in metrics_by_model.items():
        family_rows: dict[str, Any] = {}
        pooled: list[float] = []
        for family in protocol.FAMILIES:
            ratios: list[float] = []
            for row in rows:
                if (
                    row["conditioning"] != "all_finite"
                    or row["family"] != family
                ):
                    continue
                relevance = row["mean_relevance_relative_magnitude"]
                surface = row["mean_surface"]
                shell = row["mean_shell"]
                if (
                    relevance is None or float(relevance) <= EPSILON
                    or surface is None or shell is None
                ):
                    continue
                ratio = max(float(surface), float(shell)) / float(relevance)
                if math.isfinite(ratio):
                    ratios.append(ratio)
                    pooled.append(ratio)
            family_rows[family] = {
                "median_max_control_to_relevance": safe_median(ratios),
                "observation_count": len(ratios),
            }
        by_model[model_name] = {
            "families": family_rows,
            "pooled_median_max_control_to_relevance": safe_median(pooled),
        }
    return {
        "schema_version": "phase1080_factor_ratios.v1",
        "phase": protocol.PHASE,
        "by_model": by_model,
    }


def top_regions(
    metrics_by_model: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_name, rows in metrics_by_model.items():
        for family in protocol.FAMILIES:
            for field, column in FIELD_COLUMNS.items():
                candidates = [
                    row
                    for row in rows
                    if row["conditioning"] == "all_finite"
                    and row["family"] == family
                    and row["split"] == "confirmation"
                    and row[column] is not None
                ]
                candidates.sort(key=lambda row: float(row[column]), reverse=True)
                for rank, row in enumerate(candidates[:5], 1):
                    output.append({
                        "schema_version": "phase1080_top_region.v1",
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "family": family,
                        "field": field,
                        "rank": rank,
                        "component": row["component"],
                        "depth": row["depth"],
                        "relative_depth": row["relative_depth"],
                        "role": row["role"],
                        "mean_relative_magnitude": float(row[column]),
                    })
    return output


def heldout_audit(
    metrics_by_model: dict[str, list[dict[str, Any]]],
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name, rows in metrics_by_model.items():
        values = atlas_math.profile_bank(
            rows,
            protocol.FAMILIES,
            "confirmation",
            "relevance",
            roles=("answer_boundary",),
            centered=True,
        )
        heldout_index = protocol.FAMILIES.index(protocol.HELDOUT_FAMILY)
        similarities = [
            (family, float(values[heldout_index] @ values[index]))
            for index, family in enumerate(protocol.FAMILIES)
            if index != heldout_index
        ]
        similarities.sort(key=lambda value: value[1], reverse=True)
        peak = next(
            row for row in regions
            if row["model"] == model_name
            and row["family"] == protocol.HELDOUT_FAMILY
            and row["field"] == "relevance"
            and row["rank"] == 1
        )
        nearest_pass = similarities[0][0] == "contrast_conjunction"
        peak_pass = (
            float(peak["relative_depth"]) >= 0.4
            and peak["component"] in {"attention_output", "mlp_output"}
        )
        by_model[model_name] = {
            "nearest_base_family": similarities[0][0],
            "nearest_similarity": similarities[0][1],
            "all_base_similarities": [
                {"family": family, "similarity": similarity}
                for family, similarity in similarities
            ],
            "relevance_peak": peak,
            "nearest_prediction_passed": nearest_pass,
            "peak_prediction_passed": peak_pass,
            "joint_prediction_passed": nearest_pass and peak_pass,
        }
    return {
        "schema_version": "phase1080_heldout_audit.v1",
        "phase": protocol.PHASE,
        "heldout_family": protocol.HELDOUT_FAMILY,
        "predicted_nearest_family": "contrast_conjunction",
        "by_model": by_model,
        "passing_models": [
            model for model, row in by_model.items()
            if row["joint_prediction_passed"]
        ],
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    metrics_by_model = {
        model: protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        for model in protocol.MODELS
    }
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    threshold_p = float(
        prereg["evidence_thresholds"]["permutation_p_max"]
    )
    minimum = int(
        prereg["evidence_thresholds"]["minimum_repeated_models_or_pairs"]
    )
    minimum_top1 = int(
        prereg["evidence_thresholds"]["minimum_base_family_top1"]
    )
    families = protocol.BASE_FAMILIES

    assignments: list[dict[str, Any]] = []
    for model_name, rows in metrics_by_model.items():
        for field in FIELD_COLUMNS:
            for centered in (False, True):
                profile = "family_centered" if centered else "raw"
                assignments.append(assignment(
                    comparison="within_model_discovery_to_confirmation",
                    field=field,
                    profile=profile,
                    source_model=model_name,
                    target_model=model_name,
                    families=families,
                    source_values=atlas_math.profile_bank(
                        rows, families, "discovery", field, centered=centered
                    ),
                    target_values=atlas_math.profile_bank(
                        rows, families, "confirmation", field, centered=centered
                    ),
                ))
        for centered in (False, True):
            profile = "intermediate_family_centered" if centered else "intermediate_raw"
            assignments.append(assignment(
                comparison="within_model_intermediate_discovery_to_confirmation",
                field="relevance",
                profile=profile,
                source_model=model_name,
                target_model=model_name,
                families=families,
                source_values=atlas_math.profile_bank(
                    rows,
                    families,
                    "discovery",
                    "relevance",
                    roles=protocol.INTERMEDIATE_ROLES,
                    centered=centered,
                ),
                target_values=atlas_math.profile_bank(
                    rows,
                    families,
                    "confirmation",
                    "relevance",
                    roles=protocol.INTERMEDIATE_ROLES,
                    centered=centered,
                ),
            ))

    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            for field in ("relevance", "presence", "total"):
                for centered in (False, True):
                    profile = "family_centered" if centered else "raw"
                    assignments.append(assignment(
                        comparison="cross_model_confirmation",
                        field=field,
                        profile=profile,
                        source_model=source_model,
                        target_model=target_model,
                        families=families,
                        source_values=atlas_math.profile_bank(
                            metrics_by_model[source_model],
                            families,
                            "confirmation",
                            field,
                            centered=centered,
                        ),
                        target_values=atlas_math.profile_bank(
                            metrics_by_model[target_model],
                            families,
                            "confirmation",
                            field,
                            centered=centered,
                        ),
                    ))

    factors = factor_ratios(metrics_by_model)
    regions = top_regions(metrics_by_model)
    heldout = heldout_audit(metrics_by_model, regions)

    p1_models: list[str] = []
    p3_models: list[str] = []
    p4_models: list[str] = []
    retrieval_by_model: dict[str, Any] = {}
    for model in protocol.MODELS:
        relevance = find_assignment(
            assignments,
            comparison="within_model_discovery_to_confirmation",
            field="relevance",
            profile="family_centered",
            source_model=model,
        )
        presence = find_assignment(
            assignments,
            comparison="within_model_discovery_to_confirmation",
            field="presence",
            profile="family_centered",
            source_model=model,
        )
        intermediate = find_assignment(
            assignments,
            comparison="within_model_intermediate_discovery_to_confirmation",
            field="relevance",
            profile="intermediate_family_centered",
            source_model=model,
        )
        relevance_pass = assignment_passes(relevance, threshold_p, minimum_top1)
        presence_gain = int(relevance["top1_correct"]) - int(
            presence["top1_correct"]
        )
        intermediate_pass = assignment_passes(
            intermediate, threshold_p, minimum_top1
        )
        if relevance_pass:
            p1_models.append(model)
        if (
            relevance_pass
            and presence_gain >= int(
                prereg["evidence_thresholds"][
                    "minimum_relevance_over_presence_gain"
                ]
            )
        ):
            p3_models.append(model)
        if intermediate_pass:
            p4_models.append(model)
        retrieval_by_model[model] = {
            "relevance_top1": relevance["top1_correct"],
            "relevance_p": relevance["exact_upper_tail_p"],
            "presence_top1": presence["top1_correct"],
            "presence_p": presence["exact_upper_tail_p"],
            "relevance_minus_presence_top1": presence_gain,
            "intermediate_relevance_top1": intermediate["top1_correct"],
            "intermediate_relevance_p": intermediate["exact_upper_tail_p"],
        }

    p2_pairs = [
        f"{row['source_model']}__{row['target_model']}"
        for row in assignments
        if row["comparison"] == "cross_model_confirmation"
        and row["field"] == "relevance"
        and row["profile"] == "family_centered"
        and assignment_passes(row, threshold_p, minimum_top1)
    ]
    p6_models = [
        model for model, summary in summaries.items()
        if float(summary["pre_branch_global_max_abs"])
        <= float(prereg["evidence_thresholds"]["pre_branch_tolerance"])
    ]
    p7_models = [
        model for model, row in factors["by_model"].items()
        if row["pooled_median_max_control_to_relevance"] is not None
        and float(row["pooled_median_max_control_to_relevance"])
        <= float(prereg["evidence_thresholds"][
            "maximum_control_to_relevance_ratio"
        ])
    ]
    behavior_by_family = {
        family: behavior_models(summaries, family)
        for family in protocol.FAMILIES
    }
    p8_families = [
        family for family in protocol.BASE_FAMILIES
        if len(behavior_by_family[family]) >= minimum
    ]

    prediction_audit = {
        "schema_version": "phase1080_prediction_audit.v1",
        "phase": protocol.PHASE,
        "predictions": {
            "P1": {
                "passed": len(p1_models) >= minimum,
                "passing_models": p1_models,
                "by_model": retrieval_by_model,
            },
            "P2": {
                "passed": len(p2_pairs) >= minimum,
                "passing_directed_pairs": p2_pairs,
            },
            "P3": {
                "passed": len(p3_models) >= minimum,
                "passing_models": p3_models,
                "by_model": retrieval_by_model,
            },
            "P4": {
                "passed": len(p4_models) >= minimum,
                "passing_models": p4_models,
                "by_model": retrieval_by_model,
            },
            "P5": {
                "passed": len(heldout["passing_models"]) >= minimum,
                "passing_models": heldout["passing_models"],
            },
            "P6": {
                "passed": len(p6_models) == len(protocol.MODELS),
                "passing_models": p6_models,
                "by_model": {
                    model: summaries[model]["pre_branch_global_max_abs"]
                    for model in protocol.MODELS
                },
            },
            "P7": {
                "passed": len(p7_models) >= minimum,
                "passing_models": p7_models,
                "by_model": {
                    model: factors["by_model"][model][
                        "pooled_median_max_control_to_relevance"
                    ]
                    for model in protocol.MODELS
                },
            },
            "P8": {
                "passed": len(p8_families) >= int(
                    prereg["evidence_thresholds"]["minimum_behavior_families"]
                ),
                "passing_families": p8_families,
                "behavior_models_by_family": behavior_by_family,
            },
        },
    }
    prediction_audit["passed_count"] = sum(
        int(row["passed"])
        for row in prediction_audit["predictions"].values()
    )
    prediction_audit["prediction_digest"] = protocol.digest(prediction_audit)

    family_evidence: dict[str, Any] = {}
    evidence_rows: list[dict[str, Any]] = []
    for family in protocol.FAMILIES:
        if family == protocol.HELDOUT_FAMILY:
            within_hits = []
            cross_hits = []
            source_hits = []
        else:
            within_hits = []
            source_hits = []
            for model in protocol.MODELS:
                row = find_assignment(
                    assignments,
                    comparison="within_model_discovery_to_confirmation",
                    field="relevance",
                    profile="family_centered",
                    source_model=model,
                )
                if (
                    assignment_passes(row, threshold_p, minimum_top1)
                    and family_correct(row, family)
                ):
                    within_hits.append(model)
                source_row = find_assignment(
                    assignments,
                    comparison="within_model_intermediate_discovery_to_confirmation",
                    field="relevance",
                    profile="intermediate_family_centered",
                    source_model=model,
                )
                if (
                    assignment_passes(source_row, threshold_p, minimum_top1)
                    and family_correct(source_row, family)
                ):
                    source_hits.append(model)
            cross_hits = [
                f"{row['source_model']}__{row['target_model']}"
                for row in assignments
                if row["comparison"] == "cross_model_confirmation"
                and row["field"] == "relevance"
                and row["profile"] == "family_centered"
                and assignment_passes(row, threshold_p, minimum_top1)
                and family_correct(row, family)
            ]
        behavior = behavior_by_family[family]
        l1 = len(within_hits) >= minimum
        l2 = l1 and len(cross_hits) >= minimum
        l3 = l2 and len(source_hits) >= minimum
        l4 = l3 and len(behavior) >= minimum
        highest = "L4" if l4 else "L3" if l3 else "L2" if l2 \
            else "L1" if l1 else "L0"
        if family == protocol.HELDOUT_FAMILY:
            highest = "L0"
        row = {
            "highest_evidence_level": highest,
            "within_model_relevance_hits": within_hits,
            "cross_model_relevance_hits": cross_hits,
            "intermediate_relevance_hits": source_hits,
            "behavior_annotation_models": behavior,
            "descriptive_status": (
                "repeated_natural_relevance_topology"
                if highest in {"L2", "L3", "L4"}
                else "mapped_without_required_repetition"
            ),
            "causal_status": "not_tested",
            "retained_in_atlas": True,
        }
        family_evidence[family] = row
        evidence_rows.append({
            "schema_version": "phase1080_family_evidence.v1",
            "phase": protocol.PHASE,
            "family": family,
            **row,
        })

    l3_base_count = sum(
        family_evidence[family]["highest_evidence_level"] in {"L3", "L4"}
        for family in protocol.BASE_FAMILIES
    )
    empirical_continue = (
        all(
            prediction_audit["predictions"][key]["passed"]
            for key in tuple(f"P{index}" for index in range(1, 9))
        )
        and l3_base_count >= 5
    )
    automatic_next = {
        "schema_version": "phase1080_automatic_next.v1",
        "phase": protocol.PHASE,
        "continue": empirical_continue,
        "integrity_audit_pending": True,
        "required_predictions": [f"P{index}" for index in range(1, 9)],
        "l3_base_family_count": l3_base_count,
        "reason": (
            "The frozen empirical gate passed; integrity audit remains required."
            if empirical_continue
            else "The frozen empirical gate failed. Preserve the atlas and do not select heads or neurons."
        ),
        "recommended_next_task": (
            prereg["automatic_next"]["next_task_if_passed"]
            if empirical_continue
            else prereg["automatic_next"]["stop_if_failed"]
        ),
    }
    automatic_next["decision_digest"] = protocol.digest(automatic_next)

    hypothesis_audit = {
        "language_as_pattern_collection": {
            "status": "compatible_not_exhaustive",
            "reason": "Nine operational families do not prove a complete language basis.",
        },
        "relative_encoding": {
            "status": "tested_as_output_matched_conditional_differences",
            "reason": (
                "Target cue identity and answer are fixed in direct-decoy; this tests natural relevance wording, not an invariant latent code."
            ),
        },
        "reuse_plus_minimal_difference": {
            "status": "reuse_described_minimality_unmeasured",
            "reason": (
                "Cross-answer, surface, split, and model repetition describe reuse; no minimum-code theorem is tested."
            ),
        },
        "efficient_or_optimal_distribution": {
            "status": "unsupported",
            "reason": "No training, capacity, energy, robustness, or compression optimum is compared.",
        },
        "unique_word_ecological_niche": {
            "status": "family_level_only",
            "reason": "Rare semantics is mapped as a family; complete token-specific niches are not recovered.",
        },
        "joint_style_logic_grammar_selection": {
            "status": "not_factorially_identified",
            "reason": "The current factors do not fully cross style, logic, grammar, and content.",
        },
        "small_model_roughness": {
            "status": "live_limit_not_explanation",
            "reason": "Scale, tokenizer, architecture, and training data remain confounded.",
        },
    }

    analysis_root = protocol.OUT_ROOT / "analysis"
    assignment_payload = {
        "schema_version": "phase1080_assignment_collection.v1",
        "phase": protocol.PHASE,
        "rows": assignments,
    }
    assignment_payload["assignment_digest"] = protocol.digest(assignment_payload)
    protocol.write_json(analysis_root / "exact_assignments.json", assignment_payload)
    protocol.write_json(analysis_root / "factor_ratios.json", factors)
    protocol.write_jsonl(analysis_root / "top_regions.jsonl", regions)
    protocol.write_json(analysis_root / "heldout_prediction.json", heldout)
    protocol.write_json(analysis_root / "prediction_audit.json", prediction_audit)
    protocol.write_jsonl(
        analysis_root / "family_evidence_ledger.jsonl", evidence_rows
    )
    protocol.write_json(analysis_root / "automatic_next.json", automatic_next)

    final = {
        "schema_version": "phase1080_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "case_count_total": sum(int(row["case_count"]) for row in summaries.values()),
        "unit_count_total": sum(int(row["unit_count"]) for row in summaries.values()),
        "primary_population": prereg["primary_population"],
        "secondary_population": prereg["secondary_population"],
        "model_summaries": summaries,
        "family_evidence": family_evidence,
        "exact_assignment_summary": [{
            "comparison": row["comparison"],
            "field": row["field"],
            "profile": row["profile"],
            "source_model": row["source_model"],
            "target_model": row["target_model"],
            "top1_correct": row["top1_correct"],
            "family_count": row["family_count"],
            "exact_upper_tail_p": row["exact_upper_tail_p"],
            "identity_margin_over_best_other": row[
                "identity_margin_over_best_other"
            ],
        } for row in assignments],
        "factor_ratios": factors,
        "heldout_prediction": heldout,
        "prospective_prediction_audit": prediction_audit,
        "hypothesis_audit": hypothesis_audit,
        "mechanism_status": {
            family: {
                "observed": (
                    "Output-matched target-cue relevance, target-cue presence, total availability, answer, surface, and shell response fields."
                ),
                "descriptive_evidence": row["highest_evidence_level"],
                "not_established": (
                    "No necessary or sufficient component, transport edge, minimal code, complete family algorithm, or physical optimum."
                ),
            }
            for family, row in family_evidence.items()
        },
        "mathematical_status": {
            "current_tools_sufficient_for": [
                "conditional forward differences",
                "normalized-depth descriptive topology",
                "exact finite-label permutation retrieval",
                "split and cross-model repetition audits",
            ],
            "not_yet_recovered": [
                "a complete language pattern ontology",
                "a predictive component transition law",
                "causal transport routes",
                "minimality, efficiency, or optimality",
                "brain-model homology",
            ],
            "new_mathematics_needed_now": False,
            "reason": (
                "The present bottleneck is empirical identification and control quality. No result yet forces a new mathematical primitive."
            ),
        },
        "hard_limits": list(prereg["interpretation_limits"]) + [
            "Relevant-versus-unrelated wording can itself form a generic discourse-control field.",
            "All branches retain the task evidence, so direct is not computation-free.",
            "Exact global retrieval can coexist with individual family confusion.",
            "Behavior-conditioned maps are secondary because correctness changes the population.",
        ],
        "automatic_next": automatic_next,
    }
    final["summary_digest"] = protocol.digest(final)
    protocol.write_json(analysis_root / "final_summary.json", final)
    print({
        "phase": protocol.PHASE,
        "status": "finalized",
        "case_count_total": final["case_count_total"],
        "passed_predictions": prediction_audit["passed_count"],
        "automatic_continue": empirical_continue,
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
