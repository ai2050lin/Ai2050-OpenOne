#!/usr/bin/env python3
"""Decompose Phase404 failures without changing its frozen gates."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase404_direct_predictive_state"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def summarize(rows: list[dict[str, Any]], dimensions: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[dimension] for dimension in dimensions)].append(row)
    result = []
    for key, selected in sorted(buckets.items(), key=lambda item: str(item[0])):
        errors = Counter(
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
                "global_top_target_count": sum(
                    row["global_top_is_target_token"] for row in selected
                ),
                "nonfinite_candidate_logit_count": sum(
                    not row.get("candidate_logits_valid", True) for row in selected
                ),
                "dominant_candidate_errors": [
                    {
                        "target": target,
                        "prediction": prediction,
                        "count": count,
                    }
                    for (target, prediction), count in errors.most_common(8)
                ],
            }
        )
    return result


def main() -> None:
    rows = []
    for model in MODELS:
        path = OUT / "collection/discovery/private" / model / "rows.jsonl"
        rows.extend(read_jsonl(path))

    normalized = []
    for row in rows:
        copy = dict(row)
        axes = row["surface_axes_private"]
        copy["lexical_axis"] = axes["lexical"]
        copy["syntax_axis"] = axes["syntax"]
        copy["order_axis"] = axes["order"]
        normalized.append(copy)

    dimensions = {
        "model_family_query": ("model", "family_id", "future_query_private"),
        "model_family_state": ("model", "family_id", "state_id_private"),
        "model_family_surface": ("model", "family_id", "surface_id_private"),
        "model_family_lexical": ("model", "family_id", "lexical_axis"),
        "model_family_syntax": ("model", "family_id", "syntax_axis"),
        "model_family_order": ("model", "family_id", "order_axis"),
    }
    axis_rows = []
    for axis_name, keys in dimensions.items():
        for item in summarize(normalized, keys):
            axis_rows.append({"axis_name": axis_name, **item})

    candidate_correct = sum(row["finite_candidate_correct"] for row in normalized)
    global_correct = sum(row["global_top_is_target_token"] for row in normalized)
    candidate_only = sum(
        row["finite_candidate_correct"] and not row["global_top_is_target_token"]
        for row in normalized
    )
    payload = {
        "schema_version": "78.3.0",
        "phase_id": "Phase404-FailureDiagnostic",
        "created_at": now(),
        "case_count": len(normalized),
        "finite_candidate_correct_count": candidate_correct,
        "global_top_target_count": global_correct,
        "finite_candidate_correct_but_global_top_wrong_count": candidate_only,
        "nonfinite_candidate_logit_count": sum(
            not row.get("candidate_logits_valid", True) for row in normalized
        ),
        "axis_summary_row_count": len(axis_rows),
        "diagnostic_conclusion": {
            "phase404_gate_may_be_relaxed": False,
            "finite_candidate_readout_equals_natural_generation": False,
            "next_test": "natural_unfinished_future_branches_without_choose_or_return_instruction",
            "physical_mapping_open": False,
        },
    }
    write_json(OUT / "phase404_failure_diagnostic.json", payload)
    write_jsonl(OUT / "analysis/phase404_failure_axes.jsonl", axis_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
