#!/usr/bin/env python3
"""Describe why Phase403 discovery failed without changing its gates."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase403_predictive_state_analysis import (  # noqa: E402
    consensus_fingerprint,
    expected_fingerprint,
    fingerprint_surface_pass,
)
from phase403_predictive_state_protocol import (  # noqa: E402
    FAMILIES,
    MODELS,
    OUT,
    QUERIES,
    STATE_VARIANTS,
    SURFACE_REPLICAS,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def rate(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def base_group_pass(rows: list[dict[str, Any]], family: str) -> bool:
    for state_variant in STATE_VARIANTS:
        selected = [
            row
            for row in rows
            if row["state_variant_private"] == state_variant
            and row["operation_context_private"] == "base"
        ]
        surface_passes = 0
        for surface in SURFACE_REPLICAS:
            surface_rows = [
                row
                for row in selected
                if row["surface_id_private"] == surface["surface_id"]
            ]
            surface_passes += int(fingerprint_surface_pass(surface_rows, family))
        if surface_passes < 3:
            return False
        if consensus_fingerprint(selected, family) != expected_fingerprint(
            selected, family
        ):
            return False
    left = [
        row
        for row in rows
        if row["state_variant_private"] == 0
        and row["operation_context_private"] == "base"
    ]
    right = [
        row
        for row in rows
        if row["state_variant_private"] == 1
        and row["operation_context_private"] == "base"
    ]
    return sum(
        a != b
        for a, b in zip(
            expected_fingerprint(left, family),
            expected_fingerprint(right, family),
            strict=True,
        )
    ) >= 2


def main() -> None:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        path = OUT / "behavior/discovery/private" / model / "rows.jsonl"
        all_rows[model] = read_jsonl(path)

    context_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    base_group_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for model, rows in all_rows.items():
        for family in FAMILIES:
            family_rows = [row for row in rows if row["family_id"] == family]
            contexts = sorted(
                {row["operation_context_private"] for row in family_rows}
            )
            for context in contexts:
                selected = [
                    row
                    for row in family_rows
                    if row["operation_context_private"] == context
                ]
                correct = sum(row["semantic_correct"] for row in selected)
                context_rows.append(
                    {
                        "model": model,
                        "family_id": family,
                        "operation_context": context,
                        "case_count": len(selected),
                        "semantic_correct_count": correct,
                        "semantic_correct_rate": rate(correct, len(selected)),
                    }
                )
            for query in QUERIES[family]:
                selected = [
                    row
                    for row in family_rows
                    if row["future_query_private"] == query
                ]
                correct = sum(row["semantic_correct"] for row in selected)
                query_rows.append(
                    {
                        "model": model,
                        "family_id": family,
                        "future_query": query,
                        "case_count": len(selected),
                        "semantic_correct_count": correct,
                        "semantic_correct_rate": rate(correct, len(selected)),
                    }
                )
            for axis in ("lexical", "syntax", "order"):
                for level in (0, 1):
                    selected = [
                        row
                        for row in family_rows
                        if row["surface_id_private"]
                        in {
                            surface["surface_id"]
                            for surface in SURFACE_REPLICAS
                            if surface[axis] == level
                        }
                    ]
                    correct = sum(row["semantic_correct"] for row in selected)
                    surface_rows.append(
                        {
                            "model": model,
                            "family_id": family,
                            "surface_axis": axis,
                            "surface_level": level,
                            "case_count": len(selected),
                            "semantic_correct_count": correct,
                            "semantic_correct_rate": rate(correct, len(selected)),
                        }
                    )

            by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in family_rows:
                by_group[row["anonymous_parallel_group_id"]].append(row)
            for group_id, selected in sorted(by_group.items()):
                base_selected = [
                    row
                    for row in selected
                    if row["operation_context_private"] == "base"
                ]
                correct = sum(row["semantic_correct"] for row in base_selected)
                base_group_rows.append(
                    {
                        "model": model,
                        "family_id": family,
                        "anonymous_parallel_group_id": group_id,
                        "base_case_count": len(base_selected),
                        "base_semantic_correct_count": correct,
                        "base_group_pass": base_group_pass(selected, family),
                    }
                )

            lookup = {
                (
                    row["anonymous_parallel_group_id"],
                    row["state_variant_private"],
                    row["surface_id_private"],
                    row["future_query_private"],
                    row["operation_context_private"],
                ): row
                for row in family_rows
            }
            update_contexts = sorted(
                context for context in contexts if context != "base"
            )
            for context in update_contexts:
                counts: Counter[str] = Counter()
                for row in family_rows:
                    if row["operation_context_private"] != context:
                        continue
                    base = lookup[
                        (
                            row["anonymous_parallel_group_id"],
                            row["state_variant_private"],
                            row["surface_id_private"],
                            row["future_query_private"],
                            "base",
                        )
                    ]
                    if (
                        row["expected_canonical_private"]
                        == base["expected_canonical_private"]
                    ):
                        counts["non_informative_same_target"] += 1
                    elif row["semantic_correct"]:
                        counts["correct_changed_target"] += 1
                    elif (
                        row["predicted_canonical_private"]
                        == base["expected_canonical_private"]
                    ):
                        counts["update_ignored_base_answer"] += 1
                    elif row["predicted_canonical_private"] is None:
                        counts["unresolved"] += 1
                    else:
                        counts["other_wrong_answer"] += 1
                informative = sum(
                    counts[key]
                    for key in (
                        "correct_changed_target",
                        "update_ignored_base_answer",
                        "unresolved",
                        "other_wrong_answer",
                    )
                )
                update_rows.append(
                    {
                        "model": model,
                        "family_id": family,
                        "operation_context": context,
                        "case_count": sum(counts.values()),
                        "informative_changed_target_count": informative,
                        **dict(counts),
                        "correct_changed_target_rate": rate(
                            counts["correct_changed_target"], informative
                        ),
                        "update_ignored_rate": rate(
                            counts["update_ignored_base_answer"], informative
                        ),
                    }
                )

    base_summary = []
    for model in MODELS:
        for family in FAMILIES:
            selected = [
                row
                for row in base_group_rows
                if row["model"] == model and row["family_id"] == family
            ]
            base_summary.append(
                {
                    "model": model,
                    "family_id": family,
                    "base_group_pass_count": sum(
                        row["base_group_pass"] for row in selected
                    ),
                    "base_group_count": len(selected),
                }
            )
    crossmodel_base_groups = []
    for family in FAMILIES:
        group_ids = {
            row["anonymous_parallel_group_id"]
            for row in base_group_rows
            if row["family_id"] == family
        }
        for group_id in sorted(group_ids):
            selected = [
                row
                for row in base_group_rows
                if row["family_id"] == family
                and row["anonymous_parallel_group_id"] == group_id
            ]
            if len(selected) == len(MODELS) and all(
                row["base_group_pass"] for row in selected
            ):
                crossmodel_base_groups.append(
                    {
                        "family_id": family,
                        "anonymous_parallel_group_id": group_id,
                    }
                )

    payload = {
        "schema_version": "77.3.0",
        "phase_id": "Phase403-FailureDiagnostic",
        "created_at": now(),
        "formal_discovery_case_count": sum(len(rows) for rows in all_rows.values()),
        "formal_discovery_semantic_correct_count": sum(
            row["semantic_correct"]
            for rows in all_rows.values()
            for row in rows
        ),
        "base_group_summary": base_summary,
        "crossmodel_base_group_count_by_family": dict(
            Counter(row["family_id"] for row in crossmodel_base_groups)
        ),
        "update_diagnostics": update_rows,
        "surface_axis_rows": surface_rows,
        "query_rows": query_rows,
        "context_rows": context_rows,
        "interpretation_boundary": {
            "base_only_counts_are_post_failure_descriptive": True,
            "base_only_counts_can_authorize_phase403_calibration": False,
            "ignored_update_is_proven_internal_state_persistence": False,
            "new_direct_endpoint_protocol_requires_fresh_groups": True,
        },
    }
    write_json(OUT / "phase403_failure_diagnostic.json", payload)
    write_jsonl(OUT / "diagnostic/phase403_context_rows.jsonl", context_rows)
    write_jsonl(OUT / "diagnostic/phase403_query_rows.jsonl", query_rows)
    write_jsonl(OUT / "diagnostic/phase403_surface_axis_rows.jsonl", surface_rows)
    write_jsonl(OUT / "diagnostic/phase403_base_group_rows.jsonl", base_group_rows)
    write_jsonl(OUT / "diagnostic/phase403_update_rows.jsonl", update_rows)
    write_jsonl(
        OUT / "diagnostic/phase403_crossmodel_base_groups.jsonl",
        crossmodel_base_groups,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
