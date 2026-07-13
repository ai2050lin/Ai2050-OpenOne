#!/usr/bin/env python3
"""Publish compact Phase405 failure axes without prompt text."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase405_natural_future_state"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def summarize(
    rows: list[dict[str, Any]], dimensions: tuple[str, ...]
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in dimensions)].append(row)
    result = []
    for key, selected in sorted(buckets.items(), key=lambda item: str(item[0])):
        natural_top = Counter(
            (row["global_top_token_text_private"] or "<nonfinite>").replace("\n", "\\n")
            for row in selected
        )
        candidate_errors = Counter(
            (
                row["target_private"],
                row["predicted_candidate_private"] or "<nonfinite>",
            )
            for row in selected
            if not row["finite_candidate_correct"]
        )
        result.append(
            {
                **dict(zip(dimensions, key)),
                "case_count": len(selected),
                "finite_candidate_correct_count": sum(
                    row["finite_candidate_correct"] for row in selected
                ),
                "natural_top_target_count": sum(
                    row["global_top_is_target_token"] for row in selected
                ),
                "natural_top_in_candidate_set_count": sum(
                    row["global_top_in_candidate_set"] for row in selected
                ),
                "dominant_natural_top_tokens": [
                    {"token_text": token, "count": count}
                    for token, count in natural_top.most_common(6)
                ],
                "dominant_candidate_errors": [
                    {
                        "target": target,
                        "prediction": prediction,
                        "count": count,
                    }
                    for (target, prediction), count in candidate_errors.most_common(6)
                ],
            }
        )
    return result


def main() -> None:
    rows = []
    for model in MODELS:
        rows.extend(
            read_jsonl(
                OUT / "collection/discovery/private" / model / "rows.jsonl"
            )
        )
    normalized = []
    for row in rows:
        copy = dict(row)
        copy.update(
            {
                "lexical_axis": row["surface_axes_private"]["lexical"],
                "syntax_axis": row["surface_axes_private"]["syntax"],
                "order_axis": row["surface_axes_private"]["order"],
            }
        )
        normalized.append(copy)

    dimensions = {
        "model_family_query": ("model", "family_id", "future_query_private"),
        "model_family_state": ("model", "family_id", "state_id_private"),
        "model_family_surface": ("model", "family_id", "surface_id_private"),
        "model_family_syntax": ("model", "family_id", "syntax_axis"),
        "model_family_order": ("model", "family_id", "order_axis"),
    }
    axis_rows = []
    for axis_name, keys in dimensions.items():
        axis_rows.extend(
            {"axis_name": axis_name, **row}
            for row in summarize(normalized, keys)
        )

    analysis = read_json(OUT / "phase405_discovery_analysis.json")
    grammar_natural = {
        row["model"]: row["natural_top_correct_count"]
        for row in analysis["model_family_rows"]
        if row["family_id"] == "grammar_constraint"
    }
    payload = {
        "schema_version": "79.3.0",
        "phase_id": "Phase405-FailureDiagnostic",
        "created_at": now(),
        "case_count": len(normalized),
        "finite_candidate_correct_count": sum(
            row["finite_candidate_correct"] for row in normalized
        ),
        "natural_top_target_count": sum(
            row["global_top_is_target_token"] for row in normalized
        ),
        "candidate_correct_but_natural_top_wrong_count": sum(
            row["finite_candidate_correct"]
            and not row["global_top_is_target_token"]
            for row in normalized
        ),
        "nonfinite_global_logit_case_count": sum(
            not row["global_logits_valid"] for row in normalized
        ),
        "grammar_natural_top_target_count_by_model": grammar_natural,
        "model_family_group_pass_count": {
            f"{row['model']}:{row['family_id']}": row["natural_group_pass_count"]
            for row in analysis["model_family_rows"]
        },
        "crossmodel_candidate_family_count": len(
            analysis["crossmodel_candidate_families"]
        ),
        "axis_summary_row_count": len(axis_rows),
        "interpretation_boundary": {
            "models_lack_language_behavior": False,
            "current_finite_natural_future_panel_defines_stable_equivalence": False,
            "natural_completion_interface_is_semantically_neutral": False,
            "physical_mapping_authorized": False,
            "next_algorithm_must_model_branch_condition_and_response_together": True,
        },
    }
    write_json(OUT / "phase405_failure_diagnostic.json", payload)
    write_jsonl(OUT / "analysis/phase405_failure_axes.jsonl", axis_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
