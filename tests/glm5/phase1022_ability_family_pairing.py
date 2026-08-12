#!/usr/bin/env python3
"""Freeze Phase1022 matched comparisons after all behavior runs finish."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1022_ability_family_protocol as protocol


MAX_ABILITY_PAIRS_PER_SPLIT = 96
FAMILY_DIRECTIONS = ("en_zh", "zh_en", "fr_en")
FAMILY_TEMPLATE = 0


def behavior_rows(model_name: str) -> list[dict[str, Any]]:
    path = protocol.OUT_ROOT / "behavior" / model_name / "formal.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing behavior file: {path}")
    return protocol.read_jsonl(path)


def eligible_translation(row: dict[str, Any]) -> bool:
    return bool(
        row["family"] == "translation"
        and not row["surface_identity"]
        and int(row["generated_token_count"]) > 0
    )


def stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["source_language"],
        row["target_language"],
        int(row["template"]),
        row["category"],
    )


def pair_cost(left: dict[str, Any], right: dict[str, Any]) -> tuple[Any, ...]:
    return (
        abs(int(left["source_token_count"]) - int(right["source_token_count"])),
        abs(
            int(left["generated_token_count"])
            - int(right["generated_token_count"])
        ),
        abs(
            int(left["prompt_token_count"])
            - int(right["prompt_token_count"])
        ),
        right["case_key"],
    )


def greedy_cross_pairs(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    available = {row["case_key"]: row for row in right_rows}
    pairs = []
    for left in sorted(left_rows, key=lambda row: row["case_key"]):
        candidates = [
            row
            for key, row in available.items()
            if key != left["case_key"]
        ]
        if not candidates:
            break
        right = min(candidates, key=lambda row: pair_cost(left, row))
        pairs.append((left, right))
        del available[right["case_key"]]
    return pairs


def same_outcome_pairs(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    available = {row["case_key"]: row for row in rows}
    pairs = []
    for left_key in sorted(tuple(available)):
        if left_key not in available:
            continue
        left = available.pop(left_key)
        candidates = list(available.values())
        if not candidates:
            break
        right = min(candidates, key=lambda row: pair_cost(left, row))
        pairs.append((left, right))
        del available[right["case_key"]]
    return pairs


def round_robin_cap(
    grouped: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]],
    limit: int,
) -> list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]:
    for values in grouped.values():
        values.sort(key=lambda pair: (pair[0]["case_key"], pair[1]["case_key"]))
    keys = sorted(grouped)
    result = []
    cursor = 0
    while len(result) < limit:
        added = False
        for key in keys:
            values = grouped[key]
            if cursor < len(values) and len(result) < limit:
                left, right = values[cursor]
                result.append((key, left, right))
                added = True
        if not added:
            break
        cursor += 1
    return result


def pair_row(
    model_name: str,
    pair_type: str,
    pair_index: int,
    key: tuple[Any, ...],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "phase1022_matched_pair.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "model": model_name,
        "pair_id": f"{model_name}.{pair_type}.{pair_index:04d}",
        "pair_type": pair_type,
        "split": key[0],
        "source_language": key[1],
        "target_language": key[2],
        "template": int(key[3]),
        "category": key[4],
        "left_case_key": left["case_key"],
        "right_case_key": right["case_key"],
        "left_hit": bool(left["semantic_hit"]),
        "right_hit": bool(right["semantic_hit"]),
        "left_source_tokens": int(left["source_token_count"]),
        "right_source_tokens": int(right["source_token_count"]),
        "left_generated_tokens": int(left["generated_token_count"]),
        "right_generated_tokens": int(right["generated_token_count"]),
        "left_prompt_tokens": int(left["prompt_token_count"]),
        "right_prompt_tokens": int(right["prompt_token_count"]),
        "source_token_gap": abs(
            int(left["source_token_count"])
            - int(right["source_token_count"])
        ),
        "generated_token_gap": abs(
            int(left["generated_token_count"])
            - int(right["generated_token_count"])
        ),
        "prompt_token_gap": abs(
            int(left["prompt_token_count"])
            - int(right["prompt_token_count"])
        ),
    }


def build_model_pairs(
    model_name: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if eligible_translation(row):
            grouped[stratum(row)].append(row)

    by_type: dict[str, dict[tuple[Any, ...], list[Any]]] = {
        "success_failure": defaultdict(list),
        "success_success": defaultdict(list),
        "failure_failure": defaultdict(list),
    }
    for key, values in grouped.items():
        success = [row for row in values if row["semantic_hit"]]
        failure = [row for row in values if not row["semantic_hit"]]
        by_type["success_failure"][key] = greedy_cross_pairs(success, failure)
        by_type["success_success"][key] = same_outcome_pairs(success)
        by_type["failure_failure"][key] = same_outcome_pairs(failure)

    result = []
    for split in protocol.SPLITS:
        selected_by_type = {}
        for pair_type, values in by_type.items():
            split_values = {
                key: pairs
                for key, pairs in values.items()
                if key[0] == split and pairs
            }
            limit = MAX_ABILITY_PAIRS_PER_SPLIT
            selected_by_type[pair_type] = round_robin_cap(
                split_values, limit
            )
        sf_count = len(selected_by_type["success_failure"])
        for control in ("success_success", "failure_failure"):
            selected_by_type[control] = selected_by_type[control][:sf_count]
        for pair_type in (
            "success_failure",
            "success_success",
            "failure_failure",
        ):
            for key, left, right in selected_by_type[pair_type]:
                result.append(pair_row(
                    model_name,
                    pair_type,
                    len(result),
                    key,
                    left,
                    right,
                ))
    return result


def behavior_accuracy(
    rows: list[dict[str, Any]], family: str
) -> float:
    values = [
        row["semantic_hit"]
        for row in rows
        if row["family"] == family
        and (
            family != "translation"
            or not row["surface_identity"]
        )
    ]
    return float(np.mean(values)) if values else 0.0


def build_family_pairs(
    model_name: str,
    rows: list[dict[str, Any]],
    family_keys: set[str],
) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["case_key"] in family_keys
        and int(row["generated_token_count"]) > 0
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(
            row["split"],
            row["source_language"],
            row["target_language"],
        )].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        for left in sorted(values, key=lambda row: row["case_key"]):
            candidates = [
                row for row in values
                if row["category"] != left["category"]
            ]
            if not candidates:
                continue
            right = min(candidates, key=lambda row: pair_cost(left, row))
            result.append({
                "schema_version": "phase1022_family_pair.v1",
                "phase": protocol.PHASE,
                "protocol_revision": protocol.PROTOCOL_REVISION,
                "model": model_name,
                "pair_id": f"{model_name}.family.{len(result):04d}",
                "pair_type": "family_vs_other",
                "split": key[0],
                "source_language": key[1],
                "target_language": key[2],
                "template": int(left["template"]),
                "category": left["category"],
                "other_category": right["category"],
                "left_case_key": left["case_key"],
                "right_case_key": right["case_key"],
                "left_hit": bool(left["semantic_hit"]),
                "right_hit": bool(right["semantic_hit"]),
                "source_token_gap": abs(
                    int(left["source_token_count"])
                    - int(right["source_token_count"])
                ),
                "generated_token_gap": abs(
                    int(left["generated_token_count"])
                    - int(right["generated_token_count"])
                ),
                "prompt_token_gap": abs(
                    int(left["prompt_token_count"])
                    - int(right["prompt_token_count"])
                ),
            })
    return result


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    common_cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "common_cases.jsonl"
    )
    rows_by_model = {
        model: behavior_rows(model) for model in protocol.MODELS
    }
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model}: behavior/protocol digest mismatch")

    pairing_root = protocol.OUT_ROOT / "pairing"
    all_pairs = {}
    pair_counts = {}
    for model, rows in rows_by_model.items():
        pairs = build_model_pairs(model, rows)
        all_pairs[model] = pairs
        protocol.write_jsonl(
            pairing_root / f"ability_pairs.{model}.jsonl",
            pairs,
        )
        pair_counts[model] = dict(Counter(
            f"{row['pair_type']}|{row['split']}" for row in pairs
        ))

    by_key = {
        model: {row["case_key"]: row for row in rows}
        for model, rows in rows_by_model.items()
    }
    common_eligible = sorted(
        set(by_key["qwen3"])
        & set(by_key["glm4"])
        & set(by_key["deepseek7b"])
    )
    cross_model = []
    for case_key in common_eligible:
        values = {model: by_key[model][case_key] for model in protocol.MODELS}
        if not all(eligible_translation(row) for row in values.values()):
            continue
        outcomes = {
            model: bool(row["semantic_hit"]) for model, row in values.items()
        }
        if outcomes == {
            "qwen3": True,
            "glm4": True,
            "deepseek7b": False,
        }:
            group = "qwen_glm_success_ds_failure"
        elif all(outcomes.values()):
            group = "all_success"
        elif not any(outcomes.values()):
            group = "all_failure"
        else:
            group = "other_discordant"
        base = values["qwen3"]
        cross_model.append({
            "schema_version": "phase1022_cross_model_case.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "case_key": case_key,
            "group": group,
            "split": base["split"],
            "source_language": base["source_language"],
            "target_language": base["target_language"],
            "template": int(base["template"]),
            "category": base["category"],
            "outcomes": outcomes,
            "generated_token_counts": {
                model: int(row["generated_token_count"])
                for model, row in values.items()
            },
        })
    protocol.write_jsonl(pairing_root / "cross_model_cases.jsonl", cross_model)

    family_cases = []
    common_by_key = {row["case_key"]: row for row in common_cases}
    for row in common_cases:
        direction = f"{row['source_language']}_{row['target_language']}"
        if (
            row["family"] == "translation"
            and int(row["template"]) == FAMILY_TEMPLATE
            and direction in FAMILY_DIRECTIONS
            and not row["surface_identity"]
        ):
            family_cases.append({
                "schema_version": "phase1022_family_case.v1",
                "phase": protocol.PHASE,
                "protocol_revision": protocol.PROTOCOL_REVISION,
                "case_key": row["case_key"],
                "split": row["split"],
                "direction": direction,
                "category": row["category"],
                "concept_id": row["concept_id"],
                "template": int(row["template"]),
            })
    if any(row["case_key"] not in common_by_key for row in family_cases):
        raise RuntimeError("family case drift")
    protocol.write_jsonl(pairing_root / "family_cases.jsonl", family_cases)
    family_keys = {row["case_key"] for row in family_cases}
    family_pair_counts = {}
    for model, rows in rows_by_model.items():
        pairs = build_family_pairs(model, rows, family_keys)
        protocol.write_jsonl(
            pairing_root / f"family_pairs.{model}.jsonl",
            pairs,
        )
        family_pair_counts[model] = dict(Counter(
            f"{row['split']}|{row['category']}" for row in pairs
        ))

    gates = {}
    thresholds = prereg["behavior_gates"]
    for family, threshold_key in (
        ("translation", "translation_two_model_accuracy"),
        ("classification", "classification_two_model_accuracy"),
        ("rare_definition", "rare_two_model_accuracy"),
        ("punctuation", "punctuation_two_model_accuracy"),
        ("connector", "connector_two_model_accuracy"),
    ):
        threshold = float(thresholds[threshold_key])
        accuracies = {
            model: behavior_accuracy(rows, family)
            for model, rows in rows_by_model.items()
        }
        passing = [
            model for model, accuracy in accuracies.items()
            if accuracy >= threshold
        ]
        gates[family] = {
            "threshold": threshold,
            "accuracies": accuracies,
            "passing_models": passing,
            "two_model_pass": len(passing) >= 2,
        }

    ability_split_counts = {
        model: {
            split: sum(
                row["pair_type"] == "success_failure"
                and row["split"] == split
                for row in pairs
            )
            for split in protocol.SPLITS
        }
        for model, pairs in all_pairs.items()
    }
    qualified_pair_models = [
        model
        for model, counts in ability_split_counts.items()
        if min(counts.values()) >= 24
    ]
    translation_internal_authorized = bool(
        gates["translation"]["two_model_pass"]
        and len(qualified_pair_models) >= 2
    )

    behavior_identity = {
        model: protocol.digest([{
            "case_key": row["case_key"],
            "semantic_hit": row["semantic_hit"],
            "generated_token_ids": row["generated_token_ids"],
        } for row in rows])
        for model, rows in rows_by_model.items()
    }
    pairing_identity = {
        "protocol_digest": prereg["protocol_digest"],
        "behavior_digests": behavior_identity,
        "pair_counts": pair_counts,
        "family_pair_counts": family_pair_counts,
        "family_case_keys": [row["case_key"] for row in family_cases],
        "cross_model_case_keys": [
            (row["case_key"], row["group"]) for row in cross_model
        ],
    }
    pairing_digest = protocol.digest(pairing_identity)
    audit = {
        "unique_pair_ids": all(
            len(rows) == len({row["pair_id"] for row in rows})
            for rows in all_pairs.values()
        ),
        "success_failure_orientation": all(
            row["left_hit"] and not row["right_hit"]
            for rows in all_pairs.values()
            for row in rows
            if row["pair_type"] == "success_failure"
        ),
        "family_case_unique": (
            len(family_cases)
            == len({row["case_key"] for row in family_cases})
        ),
        "family_split_counts": dict(Counter(
            row["split"] for row in family_cases
        )),
        "family_category_counts": dict(Counter(
            f"{row['split']}|{row['category']}" for row in family_cases
        )),
        "family_pair_counts": family_pair_counts,
    }
    audit["all_checks_passed"] = bool(
        audit["unique_pair_ids"]
        and audit["success_failure_orientation"]
        and audit["family_case_unique"]
        and all(
            count >= 9 for count in audit["family_category_counts"].values()
        )
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"pairing audit failed: {audit}")

    summary = {
        "schema_version": "phase1022_pairing_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "pairing_digest": pairing_digest,
        "behavior_digests": behavior_identity,
        "behavior_gates": gates,
        "pair_counts": pair_counts,
        "family_pair_counts": family_pair_counts,
        "ability_success_failure_by_split": ability_split_counts,
        "qualified_pair_models": qualified_pair_models,
        "translation_internal_authorized": translation_internal_authorized,
        "family_case_count": len(family_cases),
        "cross_model_group_counts": dict(Counter(
            f"{row['group']}|{row['split']}" for row in cross_model
        )),
        "audit": audit,
        "claim_limit": (
            "Success/failure matching is observational.  It does not identify "
            "a causal ability variable, and cross-model coordinates remain "
            "unaligned."
        ),
    }
    protocol.write_json(pairing_root / "summary.json", summary)
    protocol.write_json(pairing_root / "audit.json", audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
