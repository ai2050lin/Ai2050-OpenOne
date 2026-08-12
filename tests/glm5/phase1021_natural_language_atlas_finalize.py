#!/usr/bin/env python3
"""Finalize Phase1021 natural behavior and repeated differential atlas."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1018_language_pattern_finalize as base
import phase1020_language_operation_finalize as old_finalize
import phase1020_language_operation_protocol as old_protocol
import phase1021_natural_language_atlas_protocol as protocol


ANALYSIS_ROOT = protocol.OUT_ROOT / "analysis"
DISCOVERY_CONFIRMATION = 0.40
CROSS_ITEM = 0.30
PREVALENCE = 0.50


def finite_median(values: np.ndarray) -> float | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = np.asarray([
        row["candidate_margin"] for row in rows
    ], dtype=np.float64)
    return {
        "count": len(rows),
        "exact_accuracy": (
            float(np.mean([row["exact_hit"] for row in rows]))
            if rows else None
        ),
        "semantic_accuracy": (
            float(np.mean([row["semantic_hit"] for row in rows]))
            if rows else None
        ),
        "mean_semantic_score": (
            float(np.mean([row["semantic_score"] for row in rows]))
            if rows else None
        ),
        "first_token_accuracy": (
            float(np.mean([row["first_token_hit"] for row in rows]))
            if rows else None
        ),
        "median_hidden_foil_margin": finite_median(margins),
    }


def behavior_analysis() -> tuple[
    list[dict[str, Any]], dict[str, list[dict[str, Any]]]
]:
    rows_out = []
    all_rows = {}
    for model in protocol.MODELS:
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        all_rows[model] = rows
        grouping_specs = [
            (
                "family",
                lambda row: row["family"],
            ),
            (
                "subgroup",
                lambda row: f"{row['family']}:{row['subgroup']}",
            ),
            (
                "task_kind",
                lambda row: f"{row['family']}:{row['task_kind']}",
            ),
            (
                "split",
                lambda row: f"{row['family']}:{row['split']}",
            ),
        ]
        for group_type, key_fn in grouping_specs:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[key_fn(row)].append(row)
            for group, subset in sorted(grouped.items()):
                rows_out.append({
                    "schema_version": "phase1021_behavior_group.v1",
                    "phase": protocol.PHASE,
                    "model": model,
                    "group_type": group_type,
                    "group": group,
                    **metrics(subset),
                })

        directions: dict[
            tuple[str, str, str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in rows:
            if (
                row["family"] == "multilingual_operation"
                and row.get("source_language")
                and row.get("target_language")
            ):
                directions[
                    (
                        row["task_kind"],
                        row["source_language"],
                        row["target_language"],
                        row["split"],
                    )
                ].append(row)
        for key, subset in sorted(directions.items()):
            task, source, target, split = key
            rows_out.append({
                "schema_version": "phase1021_behavior_group.v1",
                "phase": protocol.PHASE,
                "model": model,
                "group_type": "operation_direction_split",
                "group": f"{task}:{source}_{target}:{split}",
                "task_kind": task,
                "source_language": source,
                "target_language": target,
                "split": split,
                **metrics(subset),
            })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "behavior_groups.jsonl", rows_out
    )
    return rows_out, all_rows


def special_pattern_analysis(
    all_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rare_rows = []
    symbol_rows = []
    for model, rows in all_rows.items():
        rare_grouped: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in rows:
            if row["family"] == "rare_definition":
                rare_grouped[(row["term"], row["split"])].append(row)
        for (term, split), subset in sorted(rare_grouped.items()):
            rare_rows.append({
                "schema_version": "phase1021_rare_term_behavior.v1",
                "phase": protocol.PHASE,
                "model": model,
                "term": term,
                "split": split,
                "examples": [
                    {
                        "generated": row["cleaned_output"],
                        "gold": row["gold"],
                        "hit": row["semantic_hit"],
                    }
                    for row in subset[:4]
                ],
                **metrics(subset),
            })
        for family in ("punctuation_next", "contrast_relation"):
            grouped: dict[
                tuple[str, str], list[dict[str, Any]]
            ] = defaultdict(list)
            for row in rows:
                if row["family"] == family:
                    grouped[(row["subgroup"], row["split"])].append(row)
            for (subgroup, split), subset in sorted(grouped.items()):
                symbol_rows.append({
                    "schema_version": (
                        "phase1021_symbol_relation_behavior.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model,
                    "family": family,
                    "subgroup": subgroup,
                    "split": split,
                    "most_common_outputs": Counter(
                        row["cleaned_output"] for row in subset
                    ).most_common(12),
                    **metrics(subset),
                })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "rare_term_behavior.jsonl", rare_rows
    )
    protocol.write_jsonl(
        ANALYSIS_ROOT / "symbol_relation_behavior.jsonl", symbol_rows
    )
    return rare_rows, symbol_rows


def panel_catalog() -> dict[
    tuple[str, str, str, str], dict[str, Any]
]:
    catalog = {}
    for model in protocol.MODELS:
        model_root = protocol.OUT_ROOT / "formal_scan" / model
        summary_path = model_root / "summary.json"
        if not summary_path.exists():
            continue
        summary = protocol.read_json(summary_path)
        for family in summary.get("eligible_families", []):
            family_root = model_root / family
            if not family_root.exists():
                continue
            for item_root in sorted(
                path for path in family_root.iterdir() if path.is_dir()
            ):
                for split in protocol.SPLITS:
                    root = item_root / split
                    if not (root / "summary.json").exists():
                        continue
                    catalog[(model, family, item_root.name, split)] = {
                        "root": root,
                        "summary": protocol.read_json(
                            root / "summary.json"
                        ),
                        **base.load_panel_metrics(root),
                        "directions": None,
                    }
    return catalog


def panel_directions(panel: dict[str, Any]) -> dict[str, np.ndarray]:
    if panel["directions"] is None:
        panel["directions"] = base.load_directions(panel["root"])
    return panel["directions"]


def confirmed_panels(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], np.ndarray],
]:
    rows = []
    masks = {}
    keys = sorted({
        key[:3] for key in catalog
        if (key[0], key[1], key[2], "discovery") in catalog
        and (key[0], key[1], key[2], "confirmation") in catalog
    })
    for model, family, item_id in keys:
        discovery = catalog[(model, family, item_id, "discovery")]
        confirmation = catalog[
            (model, family, item_id, "confirmation")
        ]
        cosine = base.direction_cosine(
            panel_directions(discovery),
            panel_directions(confirmation),
        )
        confirmed = (
            discovery["candidate"]
            & confirmation["candidate"]
            & (cosine >= DISCOVERY_CONFIRMATION)
        )
        masks[(model, family, item_id)] = confirmed
        rows.append({
            "schema_version": "phase1021_confirmed_panel.v1",
            "phase": protocol.PHASE,
            "model": model,
            "family": family,
            "item_id": item_id,
            "subgroup": discovery["summary"]["subgroup"],
            "discovery_unit_count": discovery["summary"]["unit_count"],
            "confirmation_unit_count": (
                confirmation["summary"]["unit_count"]
            ),
            "confirmed_event_role_count": int(confirmed.sum()),
            "median_discovery_confirmation_cosine": finite_median(
                cosine[confirmed]
            ),
            "median_discovery_magnitude": finite_median(
                discovery["magnitude"][confirmed]
            ),
            "median_confirmation_magnitude": finite_median(
                confirmation["magnitude"][confirmed]
            ),
            "confirmed_count_by_role": {
                role: int(confirmed[index].sum())
                for index, role in enumerate(protocol.CAPTURE_ROLES)
            },
            "claim_limit": (
                "A confirmed event is a repeated natural-prompt "
                "differential, not a causal edge."
            ),
        })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "confirmed_panels.jsonl", rows
    )
    return rows, masks


def combined_panel_direction(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]],
    model: str,
    family: str,
    item_id: str,
) -> dict[str, np.ndarray]:
    discovery = panel_directions(
        catalog[(model, family, item_id, "discovery")]
    )
    confirmation = panel_directions(
        catalog[(model, family, item_id, "confirmation")]
    )
    return old_finalize.combine_split_directions(
        discovery, confirmation
    )


def masked_comparison(
    *,
    model: str,
    comparison: str,
    left: str,
    right: str,
    left_direction: dict[str, np.ndarray],
    right_direction: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    cosine = base.direction_cosine(left_direction, right_direction)
    return {
        "schema_version": "phase1021_direction_comparison.v1",
        "phase": protocol.PHASE,
        "model": model,
        "comparison": comparison,
        "left": left,
        "right": right,
        "shared_candidate_count": int(mask.sum()),
        "median_cosine": finite_median(cosine[mask]),
        "median_cosine_by_role": {
            role: finite_median(cosine[index][mask[index]])
            for index, role in enumerate(protocol.CAPTURE_ROLES)
        },
        "shared_candidate_count_by_role": {
            role: int(mask[index].sum())
            for index, role in enumerate(protocol.CAPTURE_ROLES)
        },
    }


def natural_comparisons(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]],
    masks: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    family = "multilingual_operation"
    for model in protocol.MODELS:
        item_ids = {
            item_id
            for key_model, key_family, item_id, _ in catalog
            if key_model == model and key_family == family
        }
        for source, target in protocol.LANGUAGE_DIRECTIONS:
            direction = f"{source}_{target}"
            mode = f"mode_{direction}"
            irrelevant = f"irrelevant_{direction}"
            content = f"content_{direction}"
            if mode in item_ids and irrelevant in item_ids:
                mask = (
                    masks[(model, family, mode)]
                    & masks[(model, family, irrelevant)]
                )
                rows.append(masked_comparison(
                    model=model,
                    comparison="mode_vs_irrelevant",
                    left=mode,
                    right=irrelevant,
                    left_direction=combined_panel_direction(
                        catalog, model, family, mode
                    ),
                    right_direction=combined_panel_direction(
                        catalog, model, family, irrelevant
                    ),
                    mask=mask,
                ))
            if mode in item_ids and content in item_ids:
                mask = (
                    masks[(model, family, mode)]
                    & masks[(model, family, content)]
                )
                rows.append(masked_comparison(
                    model=model,
                    comparison="mode_vs_content",
                    left=mode,
                    right=content,
                    left_direction=combined_panel_direction(
                        catalog, model, family, mode
                    ),
                    right_direction=combined_panel_direction(
                        catalog, model, family, content
                    ),
                    mask=mask,
                ))

        reverse_pairs = (
            ("en_zh", "zh_en"),
            ("en_fr", "fr_en"),
            ("zh_fr", "fr_zh"),
        )
        for left_direction_name, right_direction_name in reverse_pairs:
            left = f"mode_{left_direction_name}"
            right = f"mode_{right_direction_name}"
            if left not in item_ids or right not in item_ids:
                continue
            mask = (
                masks[(model, family, left)]
                & masks[(model, family, right)]
            )
            rows.append(masked_comparison(
                model=model,
                comparison="natural_reverse_direction",
                left=left,
                right=right,
                left_direction=combined_panel_direction(
                    catalog, model, family, left
                ),
                right_direction=combined_panel_direction(
                    catalog, model, family, right
                ),
                mask=mask,
            ))
    protocol.write_jsonl(
        ANALYSIS_ROOT / "natural_direction_comparisons.jsonl", rows
    )
    return rows


def old_phase1020_aggregate(
    model: str,
    direction: str,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    old_root = old_protocol.OUT_ROOT / "formal_scan" / model
    if not old_root.exists():
        return None
    items = [
        item_id
        for item_id, spec in old_protocol.TRANSLATION_ITEMS.items()
        if spec["profile"] == "full"
        and spec["direction"] == direction
        and spec["scan_eligible"]
    ]
    panels_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confirmed_masks = []
    valid_items = []
    for item_id in items:
        panel_values = {}
        for split in old_protocol.SPLITS:
            root = (
                old_root
                / "translation_mode"
                / item_id
                / split
            )
            if not (root / "summary.json").exists():
                panel_values = {}
                break
            panel_values[split] = {
                "root": root,
                **base.load_panel_metrics(root),
                "directions": base.load_directions(root),
            }
        if len(panel_values) != 2:
            continue
        cosine = base.direction_cosine(
            panel_values["discovery"]["directions"],
            panel_values["confirmation"]["directions"],
        )
        confirmed = (
            panel_values["discovery"]["candidate"]
            & panel_values["confirmation"]["candidate"]
            & (cosine >= DISCOVERY_CONFIRMATION)
        )
        for split in old_protocol.SPLITS:
            panels_by_split[split].append(panel_values[split])
        confirmed_masks.append(confirmed)
        valid_items.append(item_id)
    if not valid_items:
        return None

    split_directions = {}
    split_counts = {}
    split_consistency = {}
    for split in old_protocol.SPLITS:
        (
            split_directions[split],
            split_counts[split],
            split_consistency[split],
        ) = old_finalize.aggregate_panels(
            panels_by_split[split], confirmed_masks
        )
    holdout_cosine = base.direction_cosine(
        split_directions["discovery"],
        split_directions["confirmation"],
    )
    prevalence = np.minimum(
        split_counts["discovery"],
        split_counts["confirmation"],
    ) / len(valid_items)
    stable = (
        (prevalence >= PREVALENCE)
        & (
            np.minimum(
                split_consistency["discovery"],
                split_consistency["confirmation"],
            ) >= CROSS_ITEM
        )
        & (holdout_cosine >= DISCOVERY_CONFIRMATION)
    )
    combined = old_finalize.combine_split_directions(
        split_directions["discovery"],
        split_directions["confirmation"],
    )
    return combined, stable


def cross_protocol_comparisons(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]],
    masks: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    family = "multilingual_operation"
    for model in protocol.MODELS:
        for direction in ("en_zh", "zh_en"):
            natural_item = f"mode_{direction}"
            key = (model, family, natural_item)
            if key not in masks:
                continue
            old_value = old_phase1020_aggregate(model, direction)
            if old_value is None:
                continue
            old_direction, old_mask = old_value
            mask = masks[key] & old_mask
            rows.append(masked_comparison(
                model=model,
                comparison="phase1020_candidate_vs_phase1021_natural",
                left=f"phase1020.full.{direction}",
                right=f"phase1021.{natural_item}",
                left_direction=old_direction,
                right_direction=combined_panel_direction(
                    catalog, model, family, natural_item
                ),
                mask=mask,
            ))
    protocol.write_jsonl(
        ANALYSIS_ROOT / "cross_protocol_comparisons.jsonl", rows
    )
    return rows


def repeated_physical_events(
    masks: dict[tuple[str, str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    distribution = defaultdict(lambda: {
        "confirmed": 0,
        "denominator": 0,
    })
    for (model, family, item_id), mask in sorted(masks.items()):
        events = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "formal_scan"
            / model
            / "events.jsonl"
        )
        for role_index, role in enumerate(protocol.CAPTURE_ROLES):
            for event in events:
                key = (
                    model,
                    family,
                    item_id.split("_", 1)[0],
                    role,
                    event["component"],
                )
                distribution[key]["denominator"] += 1
                if not mask[role_index, int(event["event_index"])]:
                    continue
                distribution[key]["confirmed"] += 1
                rows.append({
                    "schema_version": (
                        "phase1021_repeated_physical_event.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model,
                    "family": family,
                    "item_id": item_id,
                    "role": role,
                    **{
                        key: value
                        for key, value in event.items()
                        if key not in {"phase", "schema_version"}
                    },
                    "evidence_status": (
                        "descriptive_discovery_confirmation_repeat"
                    ),
                })
    distribution_rows = []
    for key, value in sorted(distribution.items()):
        model, family, track, role, component = key
        denominator = value["denominator"]
        distribution_rows.append({
            "schema_version": (
                "phase1021_repeated_event_distribution.v1"
            ),
            "phase": protocol.PHASE,
            "model": model,
            "family": family,
            "track": track,
            "role": role,
            "component": component,
            "confirmed_count": value["confirmed"],
            "candidate_denominator": denominator,
            "confirmed_rate": (
                value["confirmed"] / denominator
                if denominator else None
            ),
        })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "repeated_physical_events.jsonl", rows
    )
    protocol.write_jsonl(
        ANALYSIS_ROOT / "repeated_event_distribution.jsonl",
        distribution_rows,
    )
    return rows, distribution_rows


def automatic_continuation(
    gate: dict[str, Any],
    comparisons: list[dict[str, Any]],
    cross_protocol: list[dict[str, Any]],
) -> dict[str, Any]:
    natural_models = gate["tracks"]["natural_translation"][
        "passing_models"
    ]
    operation_models = gate["tracks"]["operation_factor"][
        "passing_models"
    ]
    cross_models = sorted({
        row["model"]
        for row in cross_protocol
        if row["shared_candidate_count"] > 0
        and (row["median_cosine"] or -1.0) >= 0.30
    })
    negative_rows = [
        row
        for row in comparisons
        if row["comparison"] == "mode_vs_irrelevant"
    ]
    negative_separation_models = sorted({
        row["model"]
        for row in negative_rows
        if row["shared_candidate_count"] > 0
        and abs(row["median_cosine"] or 0.0) <= 0.50
    })
    qualifying = sorted(
        set(natural_models)
        & set(operation_models)
        & set(cross_models)
        & set(negative_separation_models)
    )
    start = len(qualifying) >= 2
    result = {
        "schema_version": "phase1021_automatic_continuation.v1",
        "phase": protocol.PHASE,
        "natural_translation_models": natural_models,
        "operation_factor_models": operation_models,
        "cross_protocol_repeat_models": cross_models,
        "negative_separation_models": negative_separation_models,
        "qualifying_models": qualifying,
        "start_targeted_causal_validation": start,
        "decision": (
            "authorize a separately preregistered local causal test"
            if start
            else (
                "do not start causal validation; preserve the natural "
                "descriptive atlas and repair the unresolved operation "
                "factor"
            )
        ),
    }
    protocol.write_json(
        ANALYSIS_ROOT / "automatic_continuation.json", result
    )
    return result


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    behavior_rows, all_behavior = behavior_analysis()
    rare_rows, symbol_rows = special_pattern_analysis(all_behavior)
    gate = protocol.read_json(ANALYSIS_ROOT / "scan_gate.json")
    catalog = panel_catalog()
    confirmed_rows, masks = confirmed_panels(catalog)
    comparison_rows = natural_comparisons(catalog, masks)
    cross_protocol_rows = cross_protocol_comparisons(catalog, masks)
    repeated_rows, distribution_rows = repeated_physical_events(masks)
    continuation = automatic_continuation(
        gate, comparison_rows, cross_protocol_rows
    )
    claim_ledger = {
        "schema_version": "phase1021_claim_ledger.v1",
        "phase": protocol.PHASE,
        "supported": [
            "Candidate-free natural generation was measured before scanning.",
            "English, Chinese, and French directions were tested.",
            "Discovery and confirmation operation concepts are disjoint.",
            "Translate-versus-classify and target-language contrasts are logically valid.",
            "Repeated physical events are reported with component denominators.",
        ],
        "not_supported_without_later_work": [
            "The three role locations are three discrete processing layers.",
            "A repeated natural differential is a translation algorithm.",
            "A stable attention head is necessary or sufficient.",
            "Rare-word, punctuation, and contrast mechanisms exist when behavior gates fail.",
            "One mathematical law explains all four pattern families.",
        ],
        "formula_status": (
            "Phase1021 equations remain post-observation measurement "
            "definitions, not a fitted language law."
        ),
    }
    protocol.write_json(
        ANALYSIS_ROOT / "claim_ledger.json", claim_ledger
    )
    scan_summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT
            / "formal_scan"
            / model
            / "summary.json"
        )
        for model in protocol.MODELS
    }
    summary = {
        "schema_version": "phase1021_analysis_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": gate["protocol_digest"],
        "scan_gate": gate,
        "automatic_continuation": continuation,
        "counts": {
            "behavior_group_rows": len(behavior_rows),
            "rare_term_rows": len(rare_rows),
            "symbol_relation_rows": len(symbol_rows),
            "confirmed_panel_rows": len(confirmed_rows),
            "natural_comparison_rows": len(comparison_rows),
            "cross_protocol_rows": len(cross_protocol_rows),
            "repeated_physical_event_rows": len(repeated_rows),
            "repeated_distribution_rows": len(distribution_rows),
        },
        "model_scan_summaries": scan_summaries,
        "claim_ledger": claim_ledger,
    }
    protocol.write_json(ANALYSIS_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
