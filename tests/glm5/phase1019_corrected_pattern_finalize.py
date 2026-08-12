#!/usr/bin/env python3
"""Finalize Phase1019 and quantify rare-term lexical necessity."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1018_language_pattern_finalize as engine
import phase1019_corrected_pattern_protocol as protocol


def configure_engine() -> None:
    engine.ANALYSIS_ROOT = protocol.OUT_ROOT / "analysis"
    engine.CAPTURE_ROLES = protocol.CAPTURE_ROLES
    engine.FAMILIES = protocol.FAMILIES
    engine.MODELS = protocol.MODELS
    engine.OUT_ROOT = protocol.OUT_ROOT
    engine.PHASE = protocol.PHASE
    engine.PROTOCOL_REVISION = protocol.PROTOCOL_REVISION


def rare_lexical_rows() -> list[dict[str, Any]]:
    item_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "item_summary.jsonl"
    )
    item_by_key = {
        (row["model"], row["family"], row["item_id"]): row
        for row in item_rows
    }
    result = []
    for model in protocol.MODELS:
        for base_item in protocol.RARE_ITEMS:
            actual_id = base_item
            masked_id = f"{base_item}__masked"
            actual_summary = item_by_key[
                (model, "rare_semantics", actual_id)
            ]
            masked_summary = item_by_key[
                (model, "rare_semantics", masked_id)
            ]
            split_values = {}
            stable_mask = None
            for split in ("discovery", "confirmation"):
                actual_root = (
                    protocol.OUT_ROOT
                    / "formal_scan"
                    / model
                    / "rare_semantics"
                    / actual_id
                    / split
                )
                masked_root = (
                    protocol.OUT_ROOT
                    / "formal_scan"
                    / model
                    / "rare_semantics"
                    / masked_id
                    / split
                )
                actual_metrics = engine.load_panel_metrics(actual_root)
                masked_metrics = engine.load_panel_metrics(masked_root)
                joint = (
                    actual_metrics["candidate"]
                    & masked_metrics["candidate"]
                )
                stable_mask = (
                    joint if stable_mask is None else stable_mask & joint
                )
                cosine = engine.direction_cosine(
                    engine.load_directions(actual_root),
                    engine.load_directions(masked_root),
                )
                split_values[split] = {
                    "joint": joint,
                    "cosine": cosine,
                }
            assert stable_mask is not None
            stable_cosines = []
            role_values = {}
            for role_index, role in enumerate(protocol.CAPTURE_ROLES):
                role_mask = stable_mask[role_index]
                discovery_cos = split_values["discovery"]["cosine"][
                    role_index
                ][role_mask]
                confirmation_cos = split_values["confirmation"]["cosine"][
                    role_index
                ][role_mask]
                combined = np.concatenate(
                    (discovery_cos, confirmation_cos)
                )
                combined = combined[np.isfinite(combined)]
                stable_cosines.extend(combined.tolist())
                role_values[role] = {
                    "stable_joint_event_count": int(role_mask.sum()),
                    "median_actual_masked_direction_cosine": (
                        float(np.median(combined))
                        if combined.size
                        else None
                    ),
                }
            actual_accuracy = actual_summary["behavior"][
                "candidate_accuracy"
            ]
            masked_accuracy = masked_summary["behavior"][
                "candidate_accuracy"
            ]
            result.append({
                "schema_version": (
                    "phase1019_rare_lexical_necessity.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                "base_item_id": base_item,
                "actual_candidate_accuracy": actual_accuracy,
                "masked_candidate_accuracy": masked_accuracy,
                "actual_minus_masked_accuracy": (
                    actual_accuracy - masked_accuracy
                ),
                "actual_first_token_accuracy": actual_summary[
                    "behavior"
                ]["first_token_accuracy"],
                "masked_first_token_accuracy": masked_summary[
                    "behavior"
                ]["first_token_accuracy"],
                "stable_joint_event_role_count": int(stable_mask.sum()),
                "median_actual_masked_direction_cosine": (
                    float(np.median(stable_cosines))
                    if stable_cosines
                    else None
                ),
                "by_role": role_values,
                "claim_boundary": (
                    "Carrier participation control; not a full lexical "
                    "meaning mechanism."
                ),
            })
    return result


def rare_depth_trajectory() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows = []
    for model in protocol.MODELS:
        events = protocol.read_jsonl(
            protocol.OUT_ROOT / "formal_scan" / model / "events.jsonl"
        )
        residual_indices = [
            index
            for index, event in enumerate(events)
            if event["component"] == "residual"
        ]
        for base_item in protocol.RARE_ITEMS:
            for split in ("discovery", "confirmation"):
                roots = {
                    condition: (
                        protocol.OUT_ROOT
                        / "formal_scan"
                        / model
                        / "rare_semantics"
                        / (
                            base_item
                            if condition == "actual"
                            else f"{base_item}__masked"
                        )
                        / split
                    )
                    for condition in ("actual", "masked")
                }
                cosine = engine.direction_cosine(
                    engine.load_directions(roots["actual"]),
                    engine.load_directions(roots["masked"]),
                )
                for role_index, role in enumerate(
                    protocol.CAPTURE_ROLES
                ):
                    for event_index in residual_indices:
                        event = events[event_index]
                        value = cosine[role_index, event_index]
                        rows.append({
                            "schema_version": (
                                "phase1019_rare_depth_trajectory.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model,
                            "base_item_id": base_item,
                            "split": split,
                            "role": role,
                            "depth": int(event["depth"]),
                            "relative_depth": float(
                                event["relative_depth"]
                            ),
                            "actual_masked_direction_cosine": (
                                float(value)
                                if np.isfinite(value)
                                else None
                            ),
                        })
    grouped: dict[tuple[str, str, int], list[float]] = {}
    relative_depths = {}
    for row in rows:
        value = row["actual_masked_direction_cosine"]
        if value is None:
            continue
        key = (row["model"], row["role"], row["depth"])
        grouped.setdefault(key, []).append(value)
        relative_depths[key] = row["relative_depth"]
    summary = []
    for key, values in sorted(grouped.items()):
        summary.append({
            "schema_version": (
                "phase1019_rare_depth_trajectory_summary.v1"
            ),
            "phase": protocol.PHASE,
            "model": key[0],
            "role": key[1],
            "depth": key[2],
            "relative_depth": relative_depths[key],
            "observation_count": len(values),
            "median_actual_masked_direction_cosine": float(
                np.median(values)
            ),
            "mean_actual_masked_direction_cosine": float(
                np.mean(values)
            ),
        })
    return rows, summary


def rewrite_phase1019_summary(
    base_summary: dict[str, Any],
    lexical_rows: list[dict[str, Any]],
    depth_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    lexical_by_model = {}
    for model in protocol.MODELS:
        rows = [row for row in lexical_rows if row["model"] == model]
        gains = np.asarray([
            row["actual_minus_masked_accuracy"] for row in rows
        ])
        cosines = np.asarray([
            row["median_actual_masked_direction_cosine"]
            for row in rows
            if row["median_actual_masked_direction_cosine"] is not None
        ])
        lexical_by_model[model] = {
            "item_count": len(rows),
            "mean_actual_minus_masked_accuracy": float(np.mean(gains)),
            "median_actual_minus_masked_accuracy": float(
                np.median(gains)
            ),
            "item_gain_ge_0_10_count": int((gains >= 0.10).sum()),
            "median_actual_masked_direction_cosine": (
                float(np.median(cosines)) if cosines.size else None
            ),
        }
    lexical_gate_models = [
        model for model, values in lexical_by_model.items()
        if values["median_actual_minus_masked_accuracy"] >= 0.10
    ]
    automatic = base_summary["automatic_continuation"]
    automatic["schema_version"] = (
        "phase1019_automatic_continuation.v1"
    )
    for row in automatic["by_family"]:
        if row["family"] == "rare_semantics":
            row["lexical_necessity_gate_models"] = lexical_gate_models
            row["start_targeted_causal_test"] = (
                row["start_targeted_causal_test"]
                and len(lexical_gate_models) >= 2
            )
            row["decision"] = (
                "start preregistered targeted causal test"
                if row["start_targeted_causal_test"]
                else "stop at descriptive atlas"
            )
    automatic["any_targeted_causal_test_started"] = any(
        row["start_targeted_causal_test"]
        for row in automatic["by_family"]
    )
    claim_ledger = {
        "schema_version": "phase1019_claim_ledger.v1",
        "phase": protocol.PHASE,
        "supported": [
            "All discovery and confirmation prompts are text-disjoint.",
            "Punctuation and contrast are measured before their target "
            "mark or connector is inserted into the carrier.",
            "Translation behavior exceeds 0.70 in Qwen3 and GLM4.",
            "Actual rare terms and paired masked carriers are directly "
            "comparable under identical context branches.",
        ],
        "requires_numeric_interpretation": {
            "rare_lexical_necessity": lexical_by_model,
            "automatic_continuation": automatic,
        },
        "not_supported_without_later_work": [
            "A repeated response is causally necessary or sufficient.",
            "The rare-term differential equals the word's complete meaning.",
            "Punctuation or contrast is solved when behavior is near chance.",
            "A shared translation direction is a complete translation rule.",
            "The four language-pattern families share one mechanism.",
        ],
        "formula_status": (
            "Phase1019 equations remain post-observation measurement "
            "definitions, not a fitted language law."
        ),
    }
    base_summary["schema_version"] = "phase1019_analysis_summary.v1"
    base_summary["counts"]["rare_lexical_necessity_rows"] = len(
        lexical_rows
    )
    base_summary["counts"]["rare_depth_summary_rows"] = len(
        depth_summary
    )
    base_summary["rare_lexical_necessity"] = lexical_by_model
    base_summary["rare_depth_minimum_cosine"] = {
        model: {
            role: min(
                (
                    row
                    for row in depth_summary
                    if row["model"] == model and row["role"] == role
                ),
                key=lambda row: row[
                    "median_actual_masked_direction_cosine"
                ],
                default=None,
            )
            for role in protocol.CAPTURE_ROLES
        }
        for model in protocol.MODELS
    }
    base_summary["automatic_continuation"] = automatic
    base_summary["claim_ledger"] = claim_ledger
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_continuation.json",
        automatic,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "claim_ledger.json",
        claim_ledger,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "summary.json",
        base_summary,
    )
    return base_summary


def finalize() -> dict[str, Any]:
    configure_engine()
    base_summary = engine.finalize()
    lexical_rows = rare_lexical_rows()
    depth_rows, depth_summary = rare_depth_trajectory()
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "rare_lexical_necessity.jsonl",
        lexical_rows,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "rare_depth_trajectory.jsonl",
        depth_rows,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "rare_depth_trajectory_summary.jsonl",
        depth_summary,
    )
    summary = rewrite_phase1019_summary(
        base_summary, lexical_rows, depth_summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    finalize()
