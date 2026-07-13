#!/usr/bin/env python3
"""Compare independent Phase401 batch=1 and batch=8 pilot executions."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase401_local_edge_graph"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = (
    "possession_relation",
    "role_filling",
    "coreference_resolution",
    "field_extraction",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def main() -> None:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        roots = {
            size: OUT
            / "behavior/batch_pilot/private"
            / f"batch_{size}"
            / model
            / "rows.jsonl"
            for size in (1, 8)
        }
        if not all(path.is_file() for path in roots.values()):
            raise FileNotFoundError(f"Incomplete Phase401 batch pilot for {model}")
        by_size = {
            size: {row["blind_case_id"]: row for row in read_jsonl(path)}
            for size, path in roots.items()
        }
        if set(by_size[1]) != set(by_size[8]):
            raise RuntimeError(f"Phase401 batch pilot denominator mismatch for {model}")
        for case_id in sorted(by_size[1]):
            one, eight = by_size[1][case_id], by_size[8][case_id]
            fields = {
                "token_sequence_match": one["effective_generated_token_ids"]
                == eight["effective_generated_token_ids"],
                "semantic_correct_match": one["semantic_correct"]
                == eight["semantic_correct"],
                "semantic_span_match": (
                    one["semantic_start_step"],
                    one["semantic_completion_step"],
                    one["semantic_answer_text"],
                )
                == (
                    eight["semantic_start_step"],
                    eight["semantic_completion_step"],
                    eight["semantic_answer_text"],
                ),
                "format_prefix_match": one["format_prefix_text"]
                == eight["format_prefix_text"],
                "format_suffix_match": one["format_suffix_text"]
                == eight["format_suffix_text"],
                "stop_match": (one["stop_step"], one["stop_kind"])
                == (eight["stop_step"], eight["stop_kind"]),
            }
            rows.append(
                {
                    "schema_version": "75.3.0",
                    "phase_id": "Phase401-BatchSensitivityAudit",
                    "created_at": now(),
                    "model": model,
                    "blind_case_id": case_id,
                    "task_surface_private": one["task_surface_private"],
                    **fields,
                    "all_observed_fields_match": all(fields.values()),
                    "formal_denominator": False,
                }
            )
    differences = [row for row in rows if not row["all_observed_fields_match"]]
    counts = Counter((row["model"], row["task_surface_private"]) for row in rows)
    diff_counts = Counter(
        (row["model"], row["task_surface_private"]) for row in differences
    )
    payload = {
        "schema_version": "75.3.0",
        "phase_id": "Phase401-BatchSensitivityAudit",
        "created_at": now(),
        "case_count": len(rows),
        "all_observed_fields_match_count": len(rows) - len(differences),
        "batch_sensitive_case_count": len(differences),
        "token_sequence_difference_count": sum(
            not row["token_sequence_match"] for row in rows
        ),
        "semantic_correctness_difference_count": sum(
            not row["semantic_correct_match"] for row in rows
        ),
        "semantic_span_difference_count": sum(
            not row["semantic_span_match"] for row in rows
        ),
        "format_prefix_difference_count": sum(
            not row["format_prefix_match"] for row in rows
        ),
        "format_suffix_difference_count": sum(
            not row["format_suffix_match"] for row in rows
        ),
        "stop_difference_count": sum(not row["stop_match"] for row in rows),
        "cells": [
            {
                "model": model,
                "task_surface": surface,
                "case_count": counts[(model, surface)],
                "batch_sensitive_case_count": diff_counts[(model, surface)],
            }
            for model in MODELS
            for surface in SURFACES
        ],
        "result": {
            "batch_shape_is_empirically_invariant_on_pilot": not differences,
            "batch_shape_remains_part_of_measurement_contract": True,
            "pilot_rows_enter_formal_mechanism_denominator": False,
        },
        "claim_boundary": {
            "no_pilot_difference_proves_global_batch_invariance": False,
            "batch_difference_is_a_semantic_latent_state": False,
        },
    }
    write_jsonl(OUT / "behavior/batch_pilot/private/comparison_rows.jsonl", rows)
    write_jsonl(OUT / "behavior/batch_pilot/private/difference_rows.jsonl", differences)
    write_json(OUT / "phase401_batch_sensitivity_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
