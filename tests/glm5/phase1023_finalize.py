#!/usr/bin/env python3
"""Aggregate Phase1023 without promoting observations into mechanisms."""

from __future__ import annotations

import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1023_ecological_niche_protocol as protocol


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return protocol.read_jsonl(path)


def passes_niche(
    metrics: dict[str, Any],
    gates: dict[str, Any],
) -> bool:
    for split in ("discovery", "confirmation"):
        joint = metrics[split]["joint"]
        if (
            joint["within_family_top1"]
            < float(gates["within_family_identity_top1"])
            or joint["all_concept_top1"]
            < float(gates["all_concept_identity_top1"])
            or joint["same_vs_shifted_margin"]
            < float(gates["same_vs_shifted_cosine_margin"])
        ):
            return False
    return (
        metrics["strict_family_transfer"]["accuracy"]
        >= float(gates["family_transfer_accuracy"])
    )


def depth_bin(relative_depth: float) -> int:
    return min(9, max(0, int(relative_depth * 10)))


def signature(
    rows: list[dict[str, Any]],
    component: str,
) -> set[str]:
    return {
        f"{component}|{row['role']}|b{depth_bin(row['relative_depth'])}"
        for row in rows
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def ability_repeat(
    model: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    discovery = [
        row for row in candidates
        if row["prompt_split"] == "discovery"
        and row["consistency_excess"] >= 0.10
        and row["magnitude_ratio"] > 1.0
    ]
    confirmation_lookup = {
        (row["role"], row["depth"]): row
        for row in candidates
        if row["prompt_split"] == "confirmation"
    }
    repeated = []
    for row in discovery:
        confirmation = confirmation_lookup.get((row["role"], row["depth"]))
        if (
            confirmation is not None
            and confirmation["consistency_excess"] >= 0.10
            and confirmation["magnitude_ratio"] > 1.0
        ):
            repeated.append({
                "model": model,
                "role": row["role"],
                "depth": row["depth"],
                "relative_depth": row["relative_depth"],
                "discovery_consistency_excess": row[
                    "consistency_excess"
                ],
                "confirmation_consistency_excess": confirmation[
                    "consistency_excess"
                ],
                "discovery_magnitude_ratio": row["magnitude_ratio"],
                "confirmation_magnitude_ratio": confirmation[
                    "magnitude_ratio"
                ],
            })
    return {
        "discovery_candidate_count": len(discovery),
        "confirmed_candidate_count": len(repeated),
        "confirmed_candidates": repeated,
    }


def generation_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    translation = [
        row for row in rows if row["family"] == "translation"
    ]
    repeated_single_symbol = [
        row
        for row in translation
        if re.fullmatch(r"\s*(.)\1{7,}\s*", row["generated_text"], re.S)
    ]
    return {
        "translation_count": len(translation),
        "truncation_rate": (
            sum(
                row["error_class"] == "truncated_error"
                for row in translation
            )
            / len(translation)
            if translation else 0.0
        ),
        "repeated_single_symbol_rate": (
            len(repeated_single_symbol) / len(translation)
            if translation else 0.0
        ),
        "repeated_single_symbol_examples": [
            {
                "case_key": row["case_key"],
                "generated_text": row["generated_text"][:80],
            }
            for row in repeated_single_symbol[:5]
        ],
        "precision_behavior_stable": (
            len(repeated_single_symbol) / max(len(translation), 1) < 0.05
        ),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    pairing = protocol.read_json(
        protocol.OUT_ROOT / "pairing" / "summary.json"
    )
    gates = prereg["ecological_niche_gates"]
    model_rows = {}
    signatures = {}
    ability = {}
    behavior = {}

    for model in protocol.MODELS:
        ecology_root = protocol.OUT_ROOT / "ecology" / model
        scan_summary = protocol.read_json(ecology_root / "summary.json")
        residual = read_jsonl(ecology_root / "residual_metrics.jsonl")
        heads = read_jsonl(ecology_root / "attention_head_metrics.jsonl")
        neurons = read_jsonl(ecology_root / "mlp_neuron_candidates.jsonl")
        selected = scan_summary["selected_layers"]
        frozen_residual = [
            row for row in residual
            if int(row["depth"]) in selected[row["role"]]
        ]
        confirmed_residual = [
            row for row in frozen_residual
            if passes_niche(row["metrics"], gates)
        ]
        confirmed_heads = [
            row for row in heads
            if passes_niche(row["metrics"], gates)
        ]
        confirmed_neurons = [
            row for row in neurons if row["confirmation_repeated"]
        ]
        pattern_summary = scan_summary["language_pattern_scan"]
        confirmed_patterns = []
        for family, rows in pattern_summary[
            "selected_discovery_layers"
        ].items():
            metric_name = (
                "identity_top1"
                if family == "rare_definition" else "accuracy"
            )
            margin_name = (
                "same_vs_shifted_margin"
                if family == "rare_definition"
                else "true_vs_best_wrong_margin"
            )
            for row in rows:
                if (
                    row["confirmation"][metric_name]
                    > row["confirmation"]["chance"]
                    and row["confirmation"][margin_name] > 0
                ):
                    confirmed_patterns.append({
                        "family": family,
                        **row,
                    })
        model_rows[model] = {
            "selected_residual_region_count": len(frozen_residual),
            "confirmed_residual_region_count": len(confirmed_residual),
            "confirmed_attention_head_count": len(confirmed_heads),
            "discovery_selected_mlp_neuron_count": len(neurons),
            "confirmed_mlp_neuron_count": len(confirmed_neurons),
            "confirmed_language_pattern_region_count": len(
                confirmed_patterns
            ),
            "confirmed_residual_regions": [
                {
                    "role": row["role"],
                    "depth": row["depth"],
                    "relative_depth": row["relative_depth"],
                    "discovery_joint": row["metrics"]["discovery"]["joint"],
                    "confirmation_joint": row["metrics"][
                        "confirmation"
                    ]["joint"],
                    "family_transfer": row["metrics"][
                        "strict_family_transfer"
                    ],
                }
                for row in confirmed_residual
            ],
            "confirmed_attention_heads": [
                {
                    "role": row["role"],
                    "depth": row["depth"],
                    "relative_depth": row["relative_depth"],
                    "head": row["head"],
                    "discovery_joint": row["metrics"]["discovery"]["joint"],
                    "confirmation_joint": row["metrics"][
                        "confirmation"
                    ]["joint"],
                    "family_transfer": row["metrics"][
                        "strict_family_transfer"
                    ],
                }
                for row in confirmed_heads
            ],
            "confirmed_mlp_neurons": confirmed_neurons,
            "confirmed_language_pattern_regions": confirmed_patterns,
            "language_pattern_claim_limit": pattern_summary[
                "claim_limit"
            ],
        }
        signatures[model] = (
            signature(confirmed_residual, "residual")
            | signature(confirmed_heads, "attention_head")
            | signature(confirmed_neurons, "mlp_neuron")
            | {
                (
                    f"pattern_{row['family']}|pre_output|"
                    f"b{depth_bin(row['relative_depth'])}"
                )
                for row in confirmed_patterns
            }
        )

        ability_path = ecology_root / "ability_candidates.jsonl"
        if ability_path.exists():
            ability[model] = ability_repeat(
                model,
                read_jsonl(ability_path),
            )
        else:
            ability[model] = {
                "discovery_candidate_count": 0,
                "confirmed_candidate_count": 0,
                "confirmed_candidates": [],
                "not_authorized": True,
            }
        behavior_summary = protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )
        behavior_rows = read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        behavior[model] = {
            "translation": behavior_summary[
                "translation_nonidentity"
            ],
            "classification": behavior_summary["family"][
                "classification"
            ],
            "rare_definition": behavior_summary["family"][
                "rare_definition"
            ],
            "punctuation": behavior_summary["family"]["punctuation"],
            "connector": behavior_summary["family"]["connector"],
            "rare_term": behavior_summary["rare_term"],
            "generation_diagnostics": generation_diagnostics(
                behavior_rows
            ),
        }

    pairwise_similarity = {}
    for left, right in itertools.combinations(protocol.MODELS, 2):
        pairwise_similarity[f"{left}|{right}"] = {
            "functional_region_jaccard": jaccard(
                signatures[left],
                signatures[right],
            ),
            "shared_signatures": sorted(
                signatures[left] & signatures[right]
            ),
            "left_only": sorted(signatures[left] - signatures[right]),
            "right_only": sorted(signatures[right] - signatures[left]),
        }

    ability_successful_models = [
        model
        for model, row in ability.items()
        if row["confirmed_candidate_count"] > 0
    ]
    shared_ability_bins = []
    for left, right in itertools.combinations(
        ability_successful_models,
        2,
    ):
        left_bins = {
            (
                row["role"],
                depth_bin(row["relative_depth"]),
            )
            for row in ability[left]["confirmed_candidates"]
        }
        right_bins = {
            (
                row["role"],
                depth_bin(row["relative_depth"]),
            )
            for row in ability[right]["confirmed_candidates"]
        }
        for role, bin_index in sorted(left_bins & right_bins):
            shared_ability_bins.append({
                "models": [left, right],
                "role": role,
                "relative_depth_bin": bin_index,
            })
    causal_authorized = bool(
        len(ability_successful_models) >= 2
        and shared_ability_bins
        and pairing["two_successful_models_ability_authorized"]
    )

    confirmed_model_count = sum(
        row["confirmed_residual_region_count"] > 0
        or row["confirmed_attention_head_count"] > 0
        or row["confirmed_mlp_neuron_count"] > 0
        for row in model_rows.values()
    )
    conclusion = {
        "stable_family_or_niche_structure_repeated_in_models": (
            confirmed_model_count
        ),
        "causal_mechanism_established": False,
        "near_optimality_established": False,
        "brain_homology_established": False,
        "single_word_storage_cell_established": False,
        "automatic_causal_followup_authorized": causal_authorized,
        "next_action": (
            "run preregistered local causal validation on shared "
            "output-preceding ability regions"
            if causal_authorized else
            "expand balanced ecological atlas and refine component-level "
            "negative controls; do not launch causal closure automatically"
        ),
    }
    result = {
        "schema_version": "phase1023_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "principle": prereg["principle"],
        "behavior": behavior,
        "pairing": pairing,
        "ecological_atlas": model_rows,
        "cross_model_functional_regions": pairwise_similarity,
        "ability_fork": {
            "models": ability,
            "successful_models": ability_successful_models,
            "shared_relative_depth_bins": shared_ability_bins,
            "causal_followup_authorized": causal_authorized,
        },
        "hypothesis_status": {
            "plasticity_near_optimality": (
                "plausible motivation; not tested by this phase"
            ),
            "reuse_efficiency_tradeoff": (
                "repeated distributed response can support reuse; optimality "
                "requires efficiency and alternative-architecture evidence"
            ),
            "relative_concept_coding": (
                "tested observationally by same-concept versus balanced "
                "within-family shifted controls"
            ),
            "language_as_pattern_collection": (
                "useful decomposition hypothesis; not an exhaustive theory"
            ),
            "unique_word_ecological_niche": (
                "tested as cross-context identity retrieval, not as a "
                "dedicated neuron or immutable coordinate"
            ),
            "small_model_roughness": (
                "handled by three-model comparison; scaling remains untested"
            ),
        },
        "claim_limits": prereg["claim_limits"],
        "conclusion": conclusion,
    }
    write_path = protocol.OUT_ROOT / "final" / "summary.json"
    protocol.write_json(write_path, result)
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "automatic_next_action.json",
        {
            "phase": protocol.PHASE,
            "authorized": causal_authorized,
            "reason": conclusion["next_action"],
        },
    )
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
