#!/usr/bin/env python3
"""Freeze Phase1023 exact semantic-success/error pairs and controls."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1023_ecological_niche_protocol as protocol


PAIR_TYPES = (
    "semantic_success_error",
    "semantic_success_success",
    "semantic_error_error",
)
TRUE_ERROR_CLASSES = {"semantic_error", "language_error"}


def behavior_rows(model: str) -> list[dict[str, Any]]:
    return protocol.read_jsonl(
        protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
    )


def translation_accuracy(rows: list[dict[str, Any]]) -> float:
    values = [
        row["semantic_hit"]
        for row in rows
        if row["family"] == "translation"
        and not row["surface_identity"]
    ]
    return sum(values) / len(values) if values else 0.0


def family_accuracy(rows: list[dict[str, Any]], family: str) -> float:
    values = [
        row["semantic_hit"] for row in rows if row["family"] == family
    ]
    return sum(values) / len(values) if values else 0.0


def stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["prompt_split"],
        row["source_language"],
        row["target_language"],
        row["category"],
        int(row["source_token_count"]),
        int(row["minimum_target_token_count"]),
        int(row["prompt_token_count"]),
    )


def cross_pairs(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left = sorted(left, key=lambda row: row["case_key"])
    right = sorted(right, key=lambda row: row["case_key"])
    count = min(len(left), len(right))
    return list(zip(left[:count], right[:count]))


def same_pairs(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = sorted(rows, key=lambda row: row["case_key"])
    return [
        (rows[index], rows[index + 1])
        for index in range(0, len(rows) - 1, 2)
    ]


def pair_record(
    *,
    model: str,
    pair_type: str,
    index: int,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "phase1023_strict_pair.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "model": model,
        "pair_id": f"{model}.{pair_type}.{index:04d}",
        "pair_type": pair_type,
        "prompt_split": left["prompt_split"],
        "source_language": left["source_language"],
        "target_language": left["target_language"],
        "category": left["category"],
        "source_token_count": int(left["source_token_count"]),
        "minimum_target_token_count": int(
            left["minimum_target_token_count"]
        ),
        "prompt_token_count": int(left["prompt_token_count"]),
        "left_case_key": left["case_key"],
        "right_case_key": right["case_key"],
        "left_error_class": left["error_class"],
        "right_error_class": right["error_class"],
        "left_semantic_hit": bool(left["semantic_hit"]),
        "right_semantic_hit": bool(right["semantic_hit"]),
        "generated_token_gap": abs(
            int(left["generated_token_count"])
            - int(right["generated_token_count"])
        ),
    }


def build_pairs(
    model: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    translation = [
        row for row in rows
        if row["family"] == "translation"
        and not row["surface_identity"]
    ]
    grouped = defaultdict(lambda: {"success": [], "error": []})
    for row in translation:
        if row["semantic_hit"]:
            grouped[stratum(row)]["success"].append(row)
        elif row["error_class"] in TRUE_ERROR_CLASSES:
            grouped[stratum(row)]["error"].append(row)

    raw: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        pair_type: [] for pair_type in PAIR_TYPES
    }
    for key in sorted(grouped):
        success = grouped[key]["success"]
        error = grouped[key]["error"]
        raw["semantic_success_error"].extend(cross_pairs(success, error))
        raw["semantic_success_success"].extend(same_pairs(success))
        raw["semantic_error_error"].extend(same_pairs(error))

    result = []
    for prompt_split in protocol.PROMPT_SPLITS:
        by_type = {
            pair_type: [
                pair for pair in pairs
                if pair[0]["prompt_split"] == prompt_split
            ]
            for pair_type, pairs in raw.items()
        }
        target_count = len(by_type["semantic_success_error"])
        for control in (
            "semantic_success_success",
            "semantic_error_error",
        ):
            by_type[control] = by_type[control][:target_count]
        for pair_type in PAIR_TYPES:
            for left, right in by_type[pair_type]:
                result.append(pair_record(
                    model=model,
                    pair_type=pair_type,
                    index=len(result),
                    left=left,
                    right=right,
                ))
    return result


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    rows_by_model = {
        model: behavior_rows(model) for model in protocol.MODELS
    }
    thresholds = prereg["behavior_gates"]
    behavior_gates = {}
    for family, threshold_key in (
        ("translation", "translation_semantic_two_model_accuracy"),
        ("classification", "classification_two_model_accuracy"),
        ("rare_definition", "rare_definition_two_model_accuracy"),
        ("punctuation", "punctuation_two_model_accuracy"),
        ("connector", "connector_two_model_accuracy"),
    ):
        accuracies = {
            model: (
                translation_accuracy(rows)
                if family == "translation"
                else family_accuracy(rows, family)
            )
            for model, rows in rows_by_model.items()
        }
        threshold = float(thresholds[threshold_key])
        passing = [
            model for model, value in accuracies.items()
            if value >= threshold
        ]
        behavior_gates[family] = {
            "threshold": threshold,
            "accuracies": accuracies,
            "passing_models": passing,
            "two_model_pass": len(passing) >= 2,
        }

    pairing_root = protocol.OUT_ROOT / "pairing"
    all_pairs = {}
    pair_counts = {}
    model_authorized = {}
    minimum = int(
        prereg["ability_pairing"]["minimum_pairs_per_split"]
    )
    for model, rows in rows_by_model.items():
        pairs = build_pairs(model, rows)
        all_pairs[model] = pairs
        protocol.write_jsonl(
            pairing_root / f"ability_pairs.{model}.jsonl",
            pairs,
        )
        counts = Counter(
            f"{row['pair_type']}|{row['prompt_split']}" for row in pairs
        )
        pair_counts[model] = dict(counts)
        model_authorized[model] = bool(
            all(
                counts[f"semantic_success_error|{split}"] >= minimum
                for split in protocol.PROMPT_SPLITS
            )
            and all(
                counts[f"{control}|{split}"] >= minimum
                for control in (
                    "semantic_success_success",
                    "semantic_error_error",
                )
                for split in protocol.PROMPT_SPLITS
            )
        )

    frozen = {
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_gates": behavior_gates,
        "pairs": all_pairs,
    }
    pairing_digest = protocol.digest(frozen)
    summary = {
        "schema_version": "phase1023_pairing_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "pairing_digest": pairing_digest,
        "behavior_gates": behavior_gates,
        "pair_counts": pair_counts,
        "ability_scan_authorized_by_model": model_authorized,
        "two_successful_models_ability_authorized": (
            model_authorized.get("qwen3", False)
            and model_authorized.get("glm4", False)
        ),
    }
    protocol.write_json(pairing_root / "summary.json", summary)

    audit = {
        "protocol_digest_match": (
            prereg["protocol_digest"] == frozen["protocol_digest"]
        ),
        "pair_ids_unique": all(
            len(rows) == len({row["pair_id"] for row in rows})
            for rows in all_pairs.values()
        ),
        "exact_strata": all(
            row["prompt_token_count"] >= 1
            and row["source_token_count"] >= 1
            and row["minimum_target_token_count"] >= 1
            for rows in all_pairs.values()
            for row in rows
        ),
        "success_error_labels_valid": all(
            (
                row["left_semantic_hit"]
                and not row["right_semantic_hit"]
                and row["right_error_class"] in TRUE_ERROR_CLASSES
            )
            for rows in all_pairs.values()
            for row in rows
            if row["pair_type"] == "semantic_success_error"
        ),
        "format_echo_truncation_excluded": all(
            row["right_error_class"] not in {
                "format_success",
                "echo_error",
                "truncated_error",
            }
            for rows in all_pairs.values()
            for row in rows
            if row["pair_type"] == "semantic_success_error"
        ),
    }
    audit["all_checks_passed"] = all(audit.values())
    protocol.write_json(pairing_root / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

