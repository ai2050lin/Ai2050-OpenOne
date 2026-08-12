#!/usr/bin/env python3
"""Finalize Phase1022 without promoting measurements to mechanism laws."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1022_ability_family_protocol as protocol


GROUPS = ("whole", "head", "key", "value")
ROLE_INDEX = {
    role: index for index, role in enumerate(protocol.INTERNAL_ROLES)
}
TIMELINE_ROLES = (
    "pre_from_source",
    "output1_from_pre",
    "output2_from_pre",
    "outputlast_from_pre",
)
MODEL_PAIRS = (
    ("qwen3", "glm4"),
    ("qwen3", "deepseek7b"),
    ("glm4", "deepseek7b"),
)


def finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def cosine_arrays(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum(
        "...d,...d->...", left.astype(np.float64), right.astype(np.float64)
    )
    denominator = np.sqrt(
        np.einsum("...d,...d->...", left, left, dtype=np.float64)
        * np.einsum("...d,...d->...", right, right, dtype=np.float64)
    )
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = denominator > 1e-12
    result[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return result


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def event_ids(metadata: dict[str, Any], group: str) -> list[str]:
    if group == "whole":
        return list(metadata["whole_event_ids"])
    if group == "head":
        return list(metadata["head_event_ids"])
    return [
        value.replace("kv_cache", f"{group}_cache")
        for value in metadata["kv_event_ids"]
    ]


def event_fields(
    event_id: str,
    *,
    group: str,
    n_layers: int,
) -> dict[str, Any]:
    depth_match = re.search(r"\.d(\d+)", event_id)
    depth = int(depth_match.group(1)) if depth_match else 0
    if group == "whole":
        component = event_id.split(".d", 1)[0]
    elif group == "head":
        component = "attention_head_pre_o_proj"
    else:
        component = f"{group}_cache"
    head_match = re.search(r"\.h(\d+)", event_id)
    return {
        "event_id": event_id,
        "component_group": group,
        "component": component,
        "depth": depth,
        "relative_depth": float(depth / max(n_layers, 1)),
        "depth_decile": min(9, int(10 * depth / max(n_layers, 1))),
        "head": int(head_match.group(1)) if head_match else None,
    }


def controls_at(
    ss: dict[str, np.ndarray],
    ff: dict[str, np.ndarray],
    group: str,
    metric: str,
    role_index: int,
) -> np.ndarray:
    left = ss[f"{group}_{metric}"][role_index]
    right = ff[f"{group}_{metric}"][role_index]
    return np.fmax(left, right)


def ability_candidates_for_model(
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = protocol.OUT_ROOT / "internal_scan" / model
    metadata = protocol.read_json(root / "events.json")
    n_layers = int(metadata["n_layers"])
    panels = {
        (pair_type, split): load_npz(
            root / "ability" / pair_type / f"{split}.npz"
        )
        for pair_type in (
            "success_failure",
            "success_success",
            "failure_failure",
        )
        for split in protocol.SPLITS
    }
    rows = []
    diagnostics = {}
    for group in GROUPS:
        ids = event_ids(metadata, group)
        for role in protocol.INTERNAL_ROLES:
            role_index = ROLE_INDEX[role]
            discovery = panels[("success_failure", "discovery")]
            confirmation = panels[("success_failure", "confirmation")]
            discovery_ss = panels[("success_success", "discovery")]
            discovery_ff = panels[("failure_failure", "discovery")]
            confirmation_ss = panels[("success_success", "confirmation")]
            confirmation_ff = panels[("failure_failure", "confirmation")]
            magnitude_d = discovery[
                f"{group}_mean_normalized_magnitude"
            ][role_index]
            magnitude_c = confirmation[
                f"{group}_mean_normalized_magnitude"
            ][role_index]
            consistency_d = discovery[
                f"{group}_direction_consistency"
            ][role_index]
            consistency_c = confirmation[
                f"{group}_direction_consistency"
            ][role_index]
            control_magnitude_d = controls_at(
                discovery_ss,
                discovery_ff,
                group,
                "mean_normalized_magnitude",
                role_index,
            )
            control_magnitude_c = controls_at(
                confirmation_ss,
                confirmation_ff,
                group,
                "mean_normalized_magnitude",
                role_index,
            )
            control_consistency_d = controls_at(
                discovery_ss,
                discovery_ff,
                group,
                "direction_consistency",
                role_index,
            )
            control_consistency_c = controls_at(
                confirmation_ss,
                confirmation_ff,
                group,
                "direction_consistency",
                role_index,
            )
            direction_d = discovery[f"{group}_mean_direction"][role_index]
            direction_c = confirmation[f"{group}_mean_direction"][role_index]
            holdout = cosine_arrays(direction_d, direction_c)
            discovery_excess = magnitude_d - control_magnitude_d
            confirmation_excess = magnitude_c - control_magnitude_c
            valid = (
                np.isfinite(discovery_excess)
                & np.isfinite(confirmation_excess)
                & np.isfinite(consistency_d)
                & np.isfinite(consistency_c)
                & np.isfinite(control_consistency_d)
                & np.isfinite(control_consistency_c)
                & np.isfinite(holdout)
            )
            repeated = (
                valid
                & (discovery_excess > 0)
                & (confirmation_excess > 0)
                & (consistency_d > control_consistency_d)
                & (consistency_c > control_consistency_c)
                & (holdout > 0)
            )
            diagnostics[f"{group}|{role}"] = {
                "event_count": len(ids),
                "valid_count": int(valid.sum()),
                "repeated_excess_count": int(repeated.sum()),
            }
            for event_index in np.flatnonzero(repeated):
                fields = event_fields(
                    ids[int(event_index)],
                    group=group,
                    n_layers=n_layers,
                )
                rows.append({
                    "schema_version": "phase1022_ability_candidate.v1",
                    "phase": protocol.PHASE,
                    "model": model,
                    "role": role,
                    "event_index": int(event_index),
                    **fields,
                    "discovery_magnitude": finite(
                        magnitude_d[event_index]
                    ),
                    "confirmation_magnitude": finite(
                        magnitude_c[event_index]
                    ),
                    "discovery_control_magnitude": finite(
                        control_magnitude_d[event_index]
                    ),
                    "confirmation_control_magnitude": finite(
                        control_magnitude_c[event_index]
                    ),
                    "discovery_magnitude_excess": finite(
                        discovery_excess[event_index]
                    ),
                    "confirmation_magnitude_excess": finite(
                        confirmation_excess[event_index]
                    ),
                    "discovery_consistency": finite(
                        consistency_d[event_index]
                    ),
                    "confirmation_consistency": finite(
                        consistency_c[event_index]
                    ),
                    "discovery_control_consistency": finite(
                        control_consistency_d[event_index]
                    ),
                    "confirmation_control_consistency": finite(
                        control_consistency_c[event_index]
                    ),
                    "holdout_direction_cosine": finite(
                        holdout[event_index]
                    ),
                    "measurement_status": (
                        "repeated_success_failure_excess_not_causal"
                    ),
                })
    return rows, diagnostics


def family_candidates_for_model(
    model: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root = protocol.OUT_ROOT / "internal_scan" / model
    metadata = protocol.read_json(root / "events.json")
    n_layers = int(metadata["n_layers"])
    panels = {
        (category, split): load_npz(
            root / "family" / category / f"{split}.npz"
        )
        for category in protocol.CATEGORIES
        for split in protocol.SPLITS
    }
    rows = []
    reuse_rows = []
    diagnostics = {}
    for group in GROUPS:
        ids = event_ids(metadata, group)
        category_direction_d = {
            category: panels[(category, "discovery")][
                f"{group}_mean_direction"
            ]
            for category in protocol.CATEGORIES
        }
        category_direction_c = {
            category: panels[(category, "confirmation")][
                f"{group}_mean_direction"
            ]
            for category in protocol.CATEGORIES
        }
        for role in protocol.INTERNAL_ROLES:
            role_index = ROLE_INDEX[role]
            candidate_masks = {}
            for category in protocol.CATEGORIES:
                discovery = panels[(category, "discovery")]
                confirmation = panels[(category, "confirmation")]
                direction_d = category_direction_d[category][role_index]
                direction_c = category_direction_c[category][role_index]
                diagonal = cosine_arrays(direction_d, direction_c)
                off = np.stack([
                    cosine_arrays(
                        direction_d,
                        category_direction_c[other][role_index],
                    )
                    for other in protocol.CATEGORIES
                    if other != category
                ])
                off_median = np.nanmedian(off, axis=0)
                off_max = np.nanmax(off, axis=0)
                consistency_d = discovery[
                    f"{group}_direction_consistency"
                ][role_index]
                consistency_c = confirmation[
                    f"{group}_direction_consistency"
                ][role_index]
                magnitude_d = discovery[
                    f"{group}_mean_normalized_magnitude"
                ][role_index]
                magnitude_c = confirmation[
                    f"{group}_mean_normalized_magnitude"
                ][role_index]
                finite_opportunity = (
                    np.isfinite(diagonal)
                    & np.isfinite(off_max)
                    & np.isfinite(consistency_d)
                    & np.isfinite(consistency_c)
                )
                repeated = (
                    finite_opportunity
                    & (diagonal > 0)
                    & (diagonal > off_max)
                    & (consistency_d > 0)
                    & (consistency_c > 0)
                )
                diagnostic_key = f"{group}|{role}|{category}"
                opportunity_count = int(finite_opportunity.sum())
                repeated_count = int(repeated.sum())
                diagnostics[diagnostic_key] = {
                    "opportunity_count": opportunity_count,
                    "same_family_top1_count": repeated_count,
                    "same_family_top1_rate": (
                        repeated_count / opportunity_count
                        if opportunity_count else None
                    ),
                    "exchangeable_reference_rate": 1.0 / len(
                        protocol.CATEGORIES
                    ),
                }
                candidate_masks[category] = repeated
                for event_index in np.flatnonzero(repeated):
                    fields = event_fields(
                        ids[int(event_index)],
                        group=group,
                        n_layers=n_layers,
                    )
                    rows.append({
                        "schema_version": "phase1022_family_candidate.v1",
                        "phase": protocol.PHASE,
                        "model": model,
                        "category": category,
                        "role": role,
                        "event_index": int(event_index),
                        **fields,
                        "holdout_same_family_cosine": finite(
                            diagonal[event_index]
                        ),
                        "holdout_other_family_median_cosine": finite(
                            off_median[event_index]
                        ),
                        "holdout_other_family_max_cosine": finite(
                            off_max[event_index]
                        ),
                        "discovery_consistency": finite(
                            consistency_d[event_index]
                        ),
                        "confirmation_consistency": finite(
                            consistency_c[event_index]
                        ),
                        "discovery_magnitude": finite(
                            magnitude_d[event_index]
                        ),
                        "confirmation_magnitude": finite(
                            magnitude_c[event_index]
                        ),
                        "measurement_status": (
                            "heldout_family_relative_signature_not_causal"
                        ),
                    })
            mask_stack = np.stack([
                candidate_masks[category]
                for category in protocol.CATEGORIES
            ])
            reuse_count = mask_stack.sum(axis=0)
            for event_index in np.flatnonzero(reuse_count >= 2):
                categories = [
                    category
                    for category_index, category in enumerate(
                        protocol.CATEGORIES
                    )
                    if mask_stack[category_index, event_index]
                ]
                fields = event_fields(
                    ids[int(event_index)],
                    group=group,
                    n_layers=n_layers,
                )
                reuse_rows.append({
                    "schema_version": "phase1022_family_reuse_event.v1",
                    "phase": protocol.PHASE,
                    "model": model,
                    "role": role,
                    "event_index": int(event_index),
                    **fields,
                    "repeated_category_count": len(categories),
                    "categories": categories,
                    "interpretation": (
                        "shared physical event with category-specific "
                        "directions; efficiency or optimality is not proven"
                    ),
                })
    return rows, reuse_rows, diagnostics


def depth_profiles(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    group: str,
    role_index: int,
) -> dict[str, np.ndarray]:
    values = arrays[f"{group}_mean_normalized_magnitude"][role_index]
    n_layers = int(metadata["n_layers"])
    grid = np.linspace(0.0, 1.0, 21)
    result = {}
    ids = event_ids(metadata, group)
    if group == "whole":
        components = ("residual", "attention_output", "mlp_output")
        for component in components:
            selected = [
                (event_fields(
                    event_id,
                    group=group,
                    n_layers=n_layers,
                )["relative_depth"], float(values[index]))
                for index, event_id in enumerate(ids)
                if event_id.startswith(component + ".")
                and np.isfinite(values[index])
            ]
            if len(selected) >= 2:
                selected.sort()
                result[component] = np.interp(
                    grid,
                    [row[0] for row in selected],
                    [row[1] for row in selected],
                ).astype(np.float32)
    elif group == "head":
        by_depth: dict[int, list[float]] = defaultdict(list)
        for index, event_id in enumerate(ids):
            if np.isfinite(values[index]):
                depth = event_fields(
                    event_id,
                    group=group,
                    n_layers=n_layers,
                )["depth"]
                by_depth[depth].append(float(values[index]))
        selected = sorted(
            (depth / n_layers, float(np.mean(rows)))
            for depth, rows in by_depth.items()
        )
        if len(selected) >= 2:
            result["attention_head_mean"] = np.interp(
                grid,
                [row[0] for row in selected],
                [row[1] for row in selected],
            ).astype(np.float32)
    else:
        selected = [
            (index + 1) / n_layers
            for index in range(len(values))
            if np.isfinite(values[index])
        ]
        selected_values = [
            float(value) for value in values if np.isfinite(value)
        ]
        if len(selected) >= 2:
            result[f"{group}_cache"] = np.interp(
                grid, selected, selected_values
            ).astype(np.float32)
    return result


def timeline_profiles() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    for model in protocol.MODELS:
        root = protocol.OUT_ROOT / "internal_scan" / model
        metadata = protocol.read_json(root / "events.json")
        for group_name in (
            "qwen_glm_success_ds_failure",
            "all_success",
            "all_failure",
        ):
            for split in protocol.SPLITS:
                path = root / "timeline" / group_name / f"{split}.npz"
                if not path.exists():
                    continue
                arrays = load_npz(path)
                for role_index, contrast in enumerate(TIMELINE_ROLES):
                    for group in GROUPS:
                        for component, profile in depth_profiles(
                            arrays, metadata, group, role_index
                        ).items():
                            profiles[(
                                model,
                                group_name,
                                split,
                                contrast,
                                component,
                            )] = profile

    rows = []
    for group_name, split, contrast, component in sorted({
        key[1:] for key in profiles
    }):
        for left_model, right_model in MODEL_PAIRS:
            left = profiles.get((
                left_model, group_name, split, contrast, component
            ))
            right = profiles.get((
                right_model, group_name, split, contrast, component
            ))
            if left is None or right is None:
                continue
            rows.append({
                "schema_version": "phase1022_timeline_profile_comparison.v1",
                "phase": protocol.PHASE,
                "left_model": left_model,
                "right_model": right_model,
                "group": group_name,
                "split": split,
                "contrast": contrast,
                "component": component,
                "profile_cosine": cosine(left, right),
                "comparison_rule": (
                    "normalized scalar relative-depth profile; no hidden "
                    "vectors or neuron coordinates were aligned"
                ),
            })

    within = []
    for model in protocol.MODELS:
        for group_name in (
            "qwen_glm_success_ds_failure",
            "all_success",
            "all_failure",
        ):
            for contrast in TIMELINE_ROLES:
                components = {
                    key[4]
                    for key in profiles
                    if key[:4] == (
                        model, group_name, "discovery", contrast
                    )
                }
                for component in sorted(components):
                    left = profiles.get((
                        model,
                        group_name,
                        "discovery",
                        contrast,
                        component,
                    ))
                    right = profiles.get((
                        model,
                        group_name,
                        "confirmation",
                        contrast,
                        component,
                    ))
                    if left is None or right is None:
                        continue
                    within.append({
                        "model": model,
                        "group": group_name,
                        "contrast": contrast,
                        "component": component,
                        "discovery_confirmation_profile_cosine": cosine(
                            left, right
                        ),
                    })
    return rows, {
        "profile_count": len(profiles),
        "cross_model_comparison_count": len(rows),
        "within_model_holdout": within,
    }


def behavior_detail(
    pairing: dict[str, Any],
) -> dict[str, Any]:
    rare_rows = []
    auxiliary_rows = []
    for model in protocol.MODELS:
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        for row in rows:
            if row["family"] == "rare_definition":
                rare_rows.append({
                    "model": model,
                    "term": row["concept_id"],
                    "split": row["split"],
                    "template": row["template"],
                    "semantic_hit": row["semantic_hit"],
                    "generated_text": row["generated_text"],
                    "accepted_outputs": row["accepted_outputs"],
                })
            elif row["family"] in ("punctuation", "connector"):
                auxiliary_rows.append({
                    "model": model,
                    "family": row["family"],
                    "case_key": row["case_key"],
                    "semantic_hit": row["semantic_hit"],
                    "generated_text": row["generated_text"],
                    "accepted_outputs": row["accepted_outputs"],
                })
    taotie = [
        row for row in rare_rows if row["term"] == "饕餮"
    ]
    return {
        "behavior_gates": pairing["behavior_gates"],
        "taotie_rows": taotie,
        "rare_row_count": len(rare_rows),
        "auxiliary_row_count": len(auxiliary_rows),
        "rare_rows": rare_rows,
        "auxiliary_rows": auxiliary_rows,
    }


def functional_keys(
    rows: list[dict[str, Any]],
    model: str,
) -> set[tuple[str, str, int]]:
    return {
        (
            row["component"],
            row["role"],
            int(row["depth_decile"]),
        )
        for row in rows
        if row["model"] == model
        and row["role"] == "pre_output"
        and row["holdout_direction_cosine"] is not None
        and row["holdout_direction_cosine"] > 0
    }


def automatic_decision(
    ability_rows: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    pairing: dict[str, Any],
) -> dict[str, Any]:
    qwen = functional_keys(ability_rows, "qwen3")
    glm = functional_keys(ability_rows, "glm4")
    shared = sorted(qwen & glm)
    timeline_by_key = {
        (
            row["left_model"],
            row["right_model"],
            row["group"],
            row["split"],
            row["contrast"],
            row["component"],
        ): row["profile_cosine"]
        for row in timeline_rows
    }
    profile_support = []
    for split in protocol.SPLITS:
        for contrast in ("pre_from_source", "output1_from_pre"):
            components = (
                "residual",
                "attention_output",
                "mlp_output",
                "attention_head_mean",
                "key_cache",
                "value_cache",
            )
            for component in components:
                qg = timeline_by_key.get((
                    "qwen3",
                    "glm4",
                    "qwen_glm_success_ds_failure",
                    split,
                    contrast,
                    component,
                ))
                qd = timeline_by_key.get((
                    "qwen3",
                    "deepseek7b",
                    "qwen_glm_success_ds_failure",
                    split,
                    contrast,
                    component,
                ))
                gd = timeline_by_key.get((
                    "glm4",
                    "deepseek7b",
                    "qwen_glm_success_ds_failure",
                    split,
                    contrast,
                    component,
                ))
                if (
                    qg is not None
                    and qd is not None
                    and gd is not None
                    and qg > qd
                    and qg > gd
                    and qg > 0
                ):
                    advantage = qg - max(qd, gd)
                    profile_support.append({
                        "split": split,
                        "contrast": contrast,
                        "component": component,
                        "qwen_glm_cosine": qg,
                        "qwen_ds_cosine": qd,
                        "glm_ds_cosine": gd,
                        "qwen_glm_advantage": advantage,
                    })
    # The ordering-only rule is audited, but the profiles are so smooth that
    # tiny ordering differences can be meaningless.  A post-hoc safety veto
    # may reject continuation but can never create positive evidence.
    material_profile_support = [
        row for row in profile_support
        if row["qwen_glm_advantage"] >= 0.05
    ]
    supported_components = {
        (row["contrast"], row["component"])
        for row in material_profile_support
    }
    repeated_profile_support = [
        key for key in supported_components
        if {
            row["split"]
            for row in material_profile_support
            if (row["contrast"], row["component"]) == key
        } == set(protocol.SPLITS)
    ]
    original_ordering_rule_passed = bool(
        pairing["translation_internal_authorized"]
        and shared
        and profile_support
    )
    start_targeted = bool(
        original_ordering_rule_passed
        and repeated_profile_support
    )
    profile_values = [
        row["profile_cosine"]
        for row in timeline_rows
        if row["profile_cosine"] is not None
    ]
    return {
        "translation_internal_authorized": pairing[
            "translation_internal_authorized"
        ],
        "qwen_pre_output_functional_keys": len(qwen),
        "glm_pre_output_functional_keys": len(glm),
        "shared_relative_region_keys": [
            {
                "component": row[0],
                "role": row[1],
                "depth_decile": row[2],
            }
            for row in shared
        ],
        "timeline_profile_support": profile_support,
        "material_timeline_profile_support": material_profile_support,
        "repeated_profile_support": [
            {"contrast": row[0], "component": row[1]}
            for row in repeated_profile_support
        ],
        "original_ordering_rule_passed": original_ordering_rule_passed,
        "posthoc_saturation_safety_veto": bool(
            original_ordering_rule_passed and not repeated_profile_support
        ),
        "timeline_profile_cosine_median": (
            float(np.median(profile_values)) if profile_values else None
        ),
        "maximum_qwen_glm_advantage": (
            max(
                (row["qwen_glm_advantage"] for row in profile_support),
                default=None,
            )
        ),
        "start_targeted_confirmation": start_targeted,
        "decision_rule": (
            "The preregistered ordering rule is reported.  Because all-model "
            "profiles saturated near one, a post-hoc safety audit may only "
            "veto continuation.  It cannot turn a result positive; a new "
            "sensitive metric must be preregistered before targeted testing."
        ),
    }


def main() -> None:
    pairing = protocol.read_json(
        protocol.OUT_ROOT / "pairing" / "summary.json"
    )
    analysis_root = protocol.OUT_ROOT / "analysis"
    ability_rows = []
    ability_diagnostics = {}
    family_rows = []
    reuse_rows = []
    family_diagnostics = {}
    for model in protocol.MODELS:
        model_ability, diagnostics = ability_candidates_for_model(model)
        ability_rows.extend(model_ability)
        ability_diagnostics[model] = diagnostics
        (
            model_family,
            model_reuse,
            model_family_diagnostics,
        ) = family_candidates_for_model(model)
        family_rows.extend(model_family)
        reuse_rows.extend(model_reuse)
        family_diagnostics[model] = model_family_diagnostics

    timeline_rows, timeline_summary = timeline_profiles()
    behavior = behavior_detail(pairing)
    automatic = automatic_decision(
        ability_rows, timeline_rows, pairing
    )
    protocol.write_jsonl(
        analysis_root / "ability_candidates.jsonl", ability_rows
    )
    protocol.write_jsonl(
        analysis_root / "family_candidates.jsonl", family_rows
    )
    protocol.write_jsonl(
        analysis_root / "family_physical_reuse.jsonl", reuse_rows
    )
    protocol.write_jsonl(
        analysis_root / "timeline_cross_model_profiles.jsonl",
        timeline_rows,
    )
    protocol.write_jsonl(
        analysis_root / "rare_definition_behavior.jsonl",
        behavior["rare_rows"],
    )
    protocol.write_jsonl(
        analysis_root / "symbol_connector_behavior.jsonl",
        behavior["auxiliary_rows"],
    )
    protocol.write_json(
        analysis_root / "automatic_continuation.json", automatic
    )

    pre_output_ability = [
        row for row in ability_rows if row["role"] == "pre_output"
    ]
    pre_output_family = [
        row for row in family_rows if row["role"] == "pre_output"
    ]
    summary = {
        "schema_version": "phase1022_analysis_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": pairing["protocol_digest"],
        "pairing_digest": pairing["pairing_digest"],
        "behavior_gates": pairing["behavior_gates"],
        "ability_candidate_count": len(ability_rows),
        "pre_output_ability_candidate_count": len(pre_output_ability),
        "ability_by_model_component_role": dict(Counter(
            f"{row['model']}|{row['component']}|{row['role']}"
            for row in ability_rows
        )),
        "ability_diagnostics": ability_diagnostics,
        "family_candidate_count": len(family_rows),
        "pre_output_family_candidate_count": len(pre_output_family),
        "family_by_model_category_role": dict(Counter(
            f"{row['model']}|{row['category']}|{row['role']}"
            for row in family_rows
        )),
        "family_reuse_event_count": len(reuse_rows),
        "family_diagnostics": family_diagnostics,
        "family_reuse_by_model_role": dict(Counter(
            f"{row['model']}|{row['role']}" for row in reuse_rows
        )),
        "timeline": timeline_summary,
        "taotie_behavior": behavior["taotie_rows"],
        "automatic_continuation": automatic,
        "theory_status": {
            "relative_differential_reuse": (
                "Supported only if held-out family directions and physical "
                "reuse survive; still observational."
            ),
            "near_optimality": (
                "Not tested.  Reuse alone cannot establish an efficiency "
                "optimum without training/compression/perturbation comparisons."
            ),
            "biological_equivalence": (
                "Not tested.  Similar functional pressure does not establish "
                "homology with the brain."
            ),
            "small_model_roughness": (
                "A plausible scope limitation, not a license to explain away "
                "negative results."
            ),
        },
        "claim_limits": [
            "Ability candidates are matched observational differences.",
            "Family candidates can still contain lexical and tokenization effects.",
            "Timeline output roles contain consequences after token emission.",
            "Cross-model comparisons use scalar relative-depth profiles only.",
            "No causal necessity, sufficiency, optimality, or global language law is claimed.",
        ],
    }
    protocol.write_json(analysis_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
