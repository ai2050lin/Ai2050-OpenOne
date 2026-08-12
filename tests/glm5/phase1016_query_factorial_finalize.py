#!/usr/bin/env python3
"""Finalize Phase1016 with registered and observation-led result layers."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1016_query_factorial_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1016_query_factorial_scan import (
    ANALYSIS_CONTRASTS,
    CAPTURE_ROLES,
    DIRECTION_CONTRASTS,
    DIRECTION_MODES,
)


FORMAL_ROOT = OUT_ROOT / "formal_scan"
ANALYSIS_ROOT = OUT_ROOT / "analysis"
KEY_ROLES = ("query_operator", "answer_boundary")
EPSILON = 1e-12


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def nanmedian(values: np.ndarray) -> float | None:
    if not np.isfinite(values).any():
        return None
    return finite(float(np.nanmedian(values)))


def aggregate(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, group in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, values)}
        item.update({
            "n": len(group),
            "factorial_candidate_all_hit_rate": float(np.mean([
                row["factorial_candidate_all_hit"] for row in group
            ])),
            "mean_factorial_candidate_hit_count": float(np.mean([
                row["factorial_candidate_hit_count"] for row in group
            ])),
            "semantic_switch_pair_hit_l0_rate": float(np.mean([
                row["semantic_switch_pair_hit_l0"] for row in group
            ])),
            "semantic_switch_pair_hit_l1_rate": float(np.mean([
                row["semantic_switch_pair_hit_l1"] for row in group
            ])),
        })
        result.append(item)
    return result


def event_arrays(
    model_name: str,
    panel_root: Path,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    events = read_jsonl(FORMAL_ROOT / model_name / "events.jsonl")
    response = np.load(panel_root / "response_scalars.npz")
    direction = np.load(panel_root / "direction_metrics.npz")
    normalized = response["normalized_magnitude"]
    contrast_names = response["contrast_names"].tolist()
    role_names = response["role_names"].tolist()
    raw_index = DIRECTION_MODES.index("raw")
    canonical_index = DIRECTION_MODES.index("canonical")
    semantic_index = DIRECTION_CONTRASTS.index("S")
    raw = np.concatenate([
        direction["whole_consistency"][raw_index, semantic_index],
        direction["head_consistency"][raw_index, semantic_index],
    ], axis=1)
    canonical = np.concatenate([
        direction["whole_consistency"][canonical_index, semantic_index],
        direction["head_consistency"][canonical_index, semantic_index],
    ], axis=1)
    lexical_alignment = np.concatenate([
        direction["whole_lexical_alignment"],
        direction["head_lexical_alignment"],
    ], axis=1)
    if normalized.shape[3] != len(events):
        raise RuntimeError(f"{panel_root}: event count drift")
    return (
        events,
        normalized,
        raw,
        canonical,
        lexical_alignment,
        contrast_names,
        role_names,
    )


def panel_metric_rows(
    *,
    model_name: str,
    family: str,
    template: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panel_root = (
        FORMAL_ROOT / model_name / family / f"template_{template}"
    )
    summary = read_json(panel_root / "summary.json")
    units = read_jsonl(panel_root / "units.jsonl")
    (
        events,
        response,
        raw_consistency,
        canonical_consistency,
        lexical_alignment,
        contrast_names,
        role_names,
    ) = event_arrays(model_name, panel_root)
    contrast_index = {
        name: contrast_names.index(name)
        for name in ANALYSIS_CONTRASTS
    }
    role_index = {name: role_names.index(name) for name in role_names}
    candidate_mask = np.asarray([
        row["factorial_candidate_all_hit"] for row in units
    ], dtype=bool)
    failed_mask = ~candidate_mask
    rows = []
    for role in KEY_ROLES:
        r = role_index[role]
        s = response[:, contrast_index["S"], r]
        lexical = response[:, contrast_index["L"], r]
        interaction = response[:, contrast_index["SL"], r]
        order = response[:, contrast_index["O"], r]
        entity = response[:, contrast_index["E"], r]
        identity = response[:, contrast_index["I"], r]
        prevalence = np.mean(s > lexical, axis=0)
        semantic_median = np.median(s, axis=0)
        lexical_median = np.median(lexical, axis=0)
        semantic_minus_lexical = np.median(s - lexical, axis=0)
        interaction_ratio = np.median(
            interaction / np.maximum(s, EPSILON),
            axis=0,
        )
        order_median = np.median(order, axis=0)
        entity_median = np.median(entity, axis=0)
        identity_max = np.max(identity, axis=0)
        if candidate_mask.any():
            qualified_median = np.median(s[candidate_mask], axis=0)
        else:
            qualified_median = np.full(s.shape[1], np.nan)
        if failed_mask.any():
            failed_median = np.median(s[failed_mask], axis=0)
        else:
            failed_median = np.full(s.shape[1], np.nan)
        for event_index, event in enumerate(events):
            raw_value = float(raw_consistency[r, event_index])
            canonical_value = float(
                canonical_consistency[r, event_index]
            )
            alignment_value = float(
                lexical_alignment[r, event_index]
            )
            registered = bool(
                math.isfinite(canonical_value)
                and math.isfinite(raw_value)
                and math.isfinite(alignment_value)
                and canonical_value >= 0.45
                and canonical_value - raw_value >= 0.20
                and alignment_value >= 0.40
                and prevalence[event_index] >= 0.70
                and semantic_minus_lexical[event_index] > 0
                and identity_max[event_index] <= 1e-6
            )
            observed = bool(
                math.isfinite(raw_value)
                and math.isfinite(alignment_value)
                and raw_value >= 0.45
                and alignment_value >= 0.40
                and prevalence[event_index] >= 0.70
                and semantic_minus_lexical[event_index] > 0
                and identity_max[event_index] <= 1e-6
            )
            rows.append({
                "schema_version": (
                    "phase1016_factorial_event_role_metric.v1"
                ),
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model_name,
                "family": family,
                "template": int(template),
                "split": summary["split"],
                "event_index": int(event_index),
                "event_id": event["event_id"],
                "component": event["component"],
                "depth": int(event["depth"]),
                "relative_depth": float(event["relative_depth"]),
                "head": event["head"],
                "role": role,
                "unit_count": len(units),
                "candidate_qualified_n": int(candidate_mask.sum()),
                "semantic_median": float(
                    semantic_median[event_index]
                ),
                "lexical_median": float(
                    lexical_median[event_index]
                ),
                "semantic_minus_lexical_median": float(
                    semantic_minus_lexical[event_index]
                ),
                "semantic_over_lexical_prevalence": float(
                    prevalence[event_index]
                ),
                "interaction_over_semantic_median": float(
                    interaction_ratio[event_index]
                ),
                "order_control_median": float(
                    order_median[event_index]
                ),
                "entity_control_median": float(
                    entity_median[event_index]
                ),
                "control_envelope_median": float(max(
                    order_median[event_index],
                    entity_median[event_index],
                )),
                "identity_maximum": float(identity_max[event_index]),
                "raw_semantic_direction_consistency": finite(raw_value),
                "fact_signed_direction_consistency": finite(
                    canonical_value
                ),
                "fact_signed_orientation_gain": finite(
                    canonical_value - raw_value
                ),
                "lexical_family_direction_alignment": finite(
                    alignment_value
                ),
                "candidate_qualified_semantic_median": finite(
                    qualified_median[event_index]
                ),
                "behavior_failed_semantic_median": finite(
                    failed_median[event_index]
                ),
                "candidate_minus_failed_semantic_median": finite(
                    qualified_median[event_index]
                    - failed_median[event_index]
                ),
                "registered_fact_signed_candidate": registered,
                "observation_fixed_orientation_candidate": observed,
                "claim_level": (
                    "registered_descriptive_response"
                    if registered
                    else (
                        "posthoc_descriptive_response"
                        if observed
                        else "continuous_measurement_only"
                    )
                ),
            })

    chain_rows = []
    for component in sorted({event["component"] for event in events}):
        event_indices = [
            index for index, event in enumerate(events)
            if event["component"] == component
        ]
        for role in role_names:
            r = role_index[role]
            s = response[
                :,
                contrast_index["S"],
                r,
                :,
            ][:, event_indices]
            lexical = response[
                :,
                contrast_index["L"],
                r,
                :,
            ][:, event_indices]
            chain_rows.append({
                "schema_version": (
                    "phase1016_panel_role_component_summary.v1"
                ),
                "phase": PHASE,
                "model": model_name,
                "family": family,
                "template": int(template),
                "split": summary["split"],
                "component": component,
                "role": role,
                "event_count": len(event_indices),
                "semantic_median": float(np.median(s)),
                "lexical_median": float(np.median(lexical)),
                "semantic_nonzero_rate": float(np.mean(s > 0)),
                "lexical_nonzero_rate": float(np.mean(lexical > 0)),
            })
    return rows, chain_rows


def repeated_rows(
    metric_rows: list[dict[str, Any]],
    candidate_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    discovery = defaultdict(list)
    confirmation = defaultdict(list)
    for row in metric_rows:
        if not row[candidate_key]:
            continue
        key = (row["model"], row["event_id"], row["role"])
        target = discovery if row["split"] == "discovery" else confirmation
        target[key].append(row)
    cores = []
    confirmed = []
    keys = sorted(set(discovery) | set(confirmation))
    for key in keys:
        drows = discovery.get(key, [])
        crows = confirmation.get(key, [])
        dpanels = {
            (row["family"], row["template"]) for row in drows
        }
        cpanels = {
            (row["family"], row["template"]) for row in crows
        }
        dfamilies = {row["family"] for row in drows}
        cfamilies = {row["family"] for row in crows}
        is_core = len(dpanels) >= 4 and len(dfamilies) >= 2
        is_confirmed = (
            is_core
            and len(cpanels) >= 2
            and len(cfamilies) >= 2
        )
        if not is_core:
            continue
        exemplar = drows[0]
        payload = {
            "schema_version": "phase1016_repeated_event_role.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "candidate_definition": candidate_key,
            "model": exemplar["model"],
            "event_id": exemplar["event_id"],
            "component": exemplar["component"],
            "depth": exemplar["depth"],
            "relative_depth": exemplar["relative_depth"],
            "head": exemplar["head"],
            "role": exemplar["role"],
            "discovery_panel_count": len(dpanels),
            "discovery_family_count": len(dfamilies),
            "discovery_families": sorted(dfamilies),
            "confirmation_panel_count": len(cpanels),
            "confirmation_family_count": len(cfamilies),
            "confirmation_families": sorted(cfamilies),
            "heldout_confirmed": bool(is_confirmed),
            "candidate_qualified_observation_count": int(sum(
                row["candidate_qualified_n"]
                for row in drows + crows
            )),
            "median_raw_consistency": nanmedian(np.asarray([
                row["raw_semantic_direction_consistency"]
                for row in drows + crows
            ], dtype=float)),
            "median_fact_signed_consistency": nanmedian(np.asarray([
                row["fact_signed_direction_consistency"]
                for row in drows + crows
            ], dtype=float)),
            "median_lexical_alignment": nanmedian(np.asarray([
                row["lexical_family_direction_alignment"]
                for row in drows + crows
            ], dtype=float)),
            "median_semantic_over_lexical_prevalence": float(
                np.median([
                    row["semantic_over_lexical_prevalence"]
                    for row in drows + crows
                ])
            ),
            "median_semantic_minus_lexical": float(np.median([
                row["semantic_minus_lexical_median"]
                for row in drows + crows
            ])),
            "median_candidate_minus_failed": nanmedian(np.asarray([
                row["candidate_minus_failed_semantic_median"]
                for row in drows + crows
            ], dtype=float)),
            "evidence_level": (
                "heldout_repeated_response"
                if is_confirmed
                else "discovery_repeated_response"
            ),
            "causal_edge_claim": False,
        }
        cores.append(payload)
        if is_confirmed:
            confirmed.append(payload)
    return cores, confirmed


def cross_model_regions(
    confirmed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in confirmed:
        depth_bin = min(9, int(float(row["relative_depth"]) * 10))
        grouped[(
            row["component"],
            row["role"],
            depth_bin,
        )].append(row)
    result = []
    for (component, role, depth_bin), group in sorted(grouped.items()):
        models = sorted({row["model"] for row in group})
        if len(models) < 2:
            continue
        result.append({
            "schema_version": "phase1016_cross_model_region.v1",
            "phase": PHASE,
            "component": component,
            "role": role,
            "relative_depth_bin": depth_bin,
            "relative_depth_interval": [
                depth_bin / 10,
                (depth_bin + 1) / 10,
            ],
            "model_count": len(models),
            "models": models,
            "physical_event_count": len(group),
            "event_count_by_model": dict(Counter(
                row["model"] for row in group
            )),
            "functional_homology_only": True,
            "coordinate_identity_claim": False,
        })
    return result


def threshold_sensitivity(
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for raw_min in (0.30, 0.45, 0.60):
        for alignment_min in (0.20, 0.40, 0.60):
            for prevalence_min in (0.60, 0.70, 0.80):
                selected = []
                for row in metric_rows:
                    raw = row["raw_semantic_direction_consistency"]
                    align = row["lexical_family_direction_alignment"]
                    if (
                        raw is not None
                        and align is not None
                        and raw >= raw_min
                        and align >= alignment_min
                        and row["semantic_over_lexical_prevalence"]
                        >= prevalence_min
                        and row["semantic_minus_lexical_median"] > 0
                        and row["identity_maximum"] <= 1e-6
                    ):
                        selected.append({
                            **row,
                            "observation_fixed_orientation_candidate": True,
                        })
                _cores, confirmed = repeated_rows(
                    selected,
                    "observation_fixed_orientation_candidate",
                )
                result.append({
                    "schema_version": (
                        "phase1016_threshold_sensitivity.v1"
                    ),
                    "raw_consistency_min": raw_min,
                    "lexical_alignment_min": alignment_min,
                    "semantic_prevalence_min": prevalence_min,
                    "panel_event_role_count": len(selected),
                    "heldout_confirmed_core_count": len(confirmed),
                })
    return result


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    if int(prereg["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("Phase1016 protocol revision drift")
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    chain_rows = []
    behavior_rows = []
    model_summaries = {}
    calibration = {}
    for model_name in MODELS:
        model_summary = read_json(
            FORMAL_ROOT / model_name / "summary.json"
        )
        if model_summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model_name}: formal digest drift")
        model_summaries[model_name] = model_summary
        calibration[model_name] = read_json(
            OUT_ROOT
            / "behavior_calibration"
            / model_name
            / "selection.json"
        )
        for family in FAMILIES:
            for template in range(4):
                panel_rows, panel_chain = panel_metric_rows(
                    model_name=model_name,
                    family=family,
                    template=template,
                )
                metric_rows.extend(panel_rows)
                chain_rows.extend(panel_chain)
                behavior_rows.extend(read_jsonl(
                    FORMAL_ROOT
                    / model_name
                    / family
                    / f"template_{template}"
                    / "units.jsonl"
                ))

    write_jsonl(ANALYSIS_ROOT / "event_role_metrics.jsonl", metric_rows)
    write_jsonl(
        ANALYSIS_ROOT / "panel_role_component_summary.jsonl",
        chain_rows,
    )
    behavior_summary = aggregate(
        behavior_rows,
        ("model", "split", "family"),
    )
    write_jsonl(ANALYSIS_ROOT / "behavior_summary.jsonl", behavior_summary)

    registered_cores, registered_confirmed = repeated_rows(
        metric_rows,
        "registered_fact_signed_candidate",
    )
    observed_cores, observed_confirmed = repeated_rows(
        metric_rows,
        "observation_fixed_orientation_candidate",
    )
    write_jsonl(
        ANALYSIS_ROOT / "registered_discovery_cores.jsonl",
        registered_cores,
    )
    write_jsonl(
        ANALYSIS_ROOT / "registered_heldout_confirmed.jsonl",
        registered_confirmed,
    )
    write_jsonl(
        ANALYSIS_ROOT / "observed_discovery_cores.jsonl",
        observed_cores,
    )
    write_jsonl(
        ANALYSIS_ROOT / "observed_heldout_confirmed.jsonl",
        observed_confirmed,
    )
    cross_regions = cross_model_regions(observed_confirmed)
    write_jsonl(
        ANALYSIS_ROOT / "observed_cross_model_regions.jsonl",
        cross_regions,
    )
    sensitivity = threshold_sensitivity(metric_rows)
    write_jsonl(
        ANALYSIS_ROOT / "threshold_sensitivity.jsonl",
        sensitivity,
    )

    observed_by_model = Counter(
        row["model"] for row in observed_confirmed
    )
    observed_by_component = Counter(
        (row["model"], row["component"])
        for row in observed_confirmed
    )
    observed_by_role = Counter(
        (row["model"], row["role"])
        for row in observed_confirmed
    )
    top_observed = sorted(
        observed_confirmed,
        key=lambda row: (
            row["confirmation_family_count"],
            row["confirmation_panel_count"],
            row["discovery_family_count"],
            row["discovery_panel_count"],
            row["median_lexical_alignment"] or -math.inf,
            row["median_raw_consistency"] or -math.inf,
        ),
        reverse=True,
    )[:50]
    max_registered_canonical = max(
        (
            row["fact_signed_direction_consistency"]
            for row in metric_rows
            if row["fact_signed_direction_consistency"] is not None
        ),
        default=None,
    )
    max_registered_gain = max(
        (
            row["fact_signed_orientation_gain"]
            for row in metric_rows
            if row["fact_signed_orientation_gain"] is not None
        ),
        default=None,
    )
    summary = {
        "schema_version": "phase1016_factorial_analysis_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "model_count": len(MODELS),
        "panel_count": sum(
            row["panel_count"] for row in model_summaries.values()
        ),
        "unit_count": len(behavior_rows),
        "singleton_forward_count": sum(
            row["singleton_forward_count"]
            for row in model_summaries.values()
        ),
        "event_role_metric_count": len(metric_rows),
        "identity_maximum": max(
            row["identity_maximum"]
            for row in model_summaries.values()
        ),
        "semantic_causal_prefix_maximum": max(
            row["semantic_causal_prefix_maximum"]
            for row in model_summaries.values()
        ),
        "prompt_mode_by_model": {
            model: calibration[model]["selected_prompt_mode"]
            for model in MODELS
        },
        "calibration_generation_accuracy_by_model": {
            model: next(
                row["generation_first_word_accuracy"]
                for row in calibration[model]["mode_summaries"]
                if row["prompt_mode"]
                == calibration[model]["selected_prompt_mode"]
            )
            for model in MODELS
        },
        "factorial_candidate_all_hit_by_model": {
            model: int(
                model_summaries[model][
                    "factorial_candidate_all_hit_count"
                ]
            )
            for model in MODELS
        },
        "registered_result": {
            "panel_candidate_count": int(sum(
                row["registered_fact_signed_candidate"]
                for row in metric_rows
            )),
            "discovery_core_count": len(registered_cores),
            "heldout_confirmed_core_count": len(registered_confirmed),
            "maximum_fact_signed_consistency": max_registered_canonical,
            "maximum_fact_signed_orientation_gain": max_registered_gain,
            "status": (
                "SUPPORTED"
                if registered_confirmed
                else "NOT_SUPPORTED"
            ),
            "interpretation": (
                "The preregistered fact-bit sign-reversal hypothesis is "
                "separate from observation-led fixed-orientation mapping."
            ),
        },
        "observation_led_result": {
            "panel_candidate_count": int(sum(
                row["observation_fixed_orientation_candidate"]
                for row in metric_rows
            )),
            "discovery_core_count": len(observed_cores),
            "heldout_confirmed_core_count": len(observed_confirmed),
            "heldout_confirmed_by_model": dict(observed_by_model),
            "heldout_confirmed_by_model_component": {
                ":".join(key): value
                for key, value in observed_by_component.items()
            },
            "heldout_confirmed_by_model_role": {
                ":".join(key): value
                for key, value in observed_by_role.items()
            },
            "cross_model_region_count": len(cross_regions),
            "claim": (
                "Repeated semantic-versus-synonym response topology only; "
                "not a causal path or language equation."
            ),
        },
        "behavior_summary": behavior_summary,
        "top_observed_heldout_cores": top_observed,
        "automatic_continuation_assessment": {
            "registered_relative_sign_gate_passed": bool(
                registered_confirmed
            ),
            "observation_repetition_gate_passed": bool(
                observed_confirmed
            ),
            "behavior_is_uniformly_qualified": all(
                model_summaries[model][
                    "factorial_candidate_all_hit_count"
                ] >= 240
                for model in MODELS
            ),
            "next_action": (
                "Run a targeted behavior-stratified rescan of a small frozen "
                "set only if observation-led repeated cores are present; do "
                "not begin causal closure or neuron claims."
            ),
        },
        "hard_limits": [
            "The fact-bit sign formula was not supported.",
            "Fixed-orientation consistency can still contain query-token "
            "semantics and is not proof of answer transport.",
            "Color heldout labels are approximate shades, not exact synonyms.",
            "Behavior qualification is weak for GLM4 and especially DS7B.",
            "Behavior-stratified scalar magnitude was persisted, but "
            "behavior-stratified direction vectors were not.",
            "No intervention, edge, neuron, or global mechanism claim is "
            "licensed by this phase.",
        ],
    }
    write_json(ANALYSIS_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
