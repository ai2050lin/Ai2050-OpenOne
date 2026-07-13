#!/usr/bin/env python3
"""Summarize Phase406 interface, parser, sequence, and stopping failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase406_conditioned_sequence_protocol import MODELS, OUT  # noqa: E402


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


def main(stage: str) -> None:
    rows = []
    for model in MODELS:
        path = OUT / "analysis" / stage / "private" / model / "semantic_rows.jsonl"
        if path.is_file():
            rows.extend(read_jsonl(path))

    axes = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model"],
                row["family_id"],
                row["interface_private"],
                row["future_query_private"],
            )
        ].append(row)
    for (model, family, interface, query), selected in sorted(grouped.items()):
        axes.append(
            {
                "schema_version": "80.3.0",
                "phase_id": "Phase406-FailureDiagnostic",
                "stage": stage,
                "model": model,
                "family_id": family,
                "interface": interface,
                "future_query": query,
                "case_count": len(selected),
                "first_step_candidate_correct_count": sum(
                    row["first_step_candidate_correct"] for row in selected
                ),
                "first_step_global_top_is_target_count": sum(
                    row["first_step_global_top_is_target"] for row in selected
                ),
                "semantic_parse_count": sum(
                    row["semantic_label_private"] is not None for row in selected
                ),
                "short_sequence_semantic_correct_count": sum(
                    row["short_sequence_semantic_correct"] for row in selected
                ),
                "sequence_stop_or_boundary_after_semantic_count": sum(
                    row["sequence_stop_or_boundary_after_semantic"]
                    for row in selected
                ),
                "candidate_correct_sequence_wrong_count": sum(
                    row["first_step_candidate_correct"]
                    and not row["short_sequence_semantic_correct"]
                    for row in selected
                ),
                "first_vocab_wrong_sequence_correct_count": sum(
                    not row["first_step_global_top_is_target"]
                    and row["short_sequence_semantic_correct"]
                    for row in selected
                ),
                "semantic_correct_without_stop_or_boundary_count": sum(
                    row["short_sequence_semantic_correct"]
                    and not row["sequence_stop_or_boundary_after_semantic"]
                    for row in selected
                ),
            }
        )

    disagreement_units = 0
    unit_count = 0
    unit_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        unit_groups[
            (
                row["model"],
                row["family_id"],
                row["anonymous_parallel_group_id"],
                row["state_id_private"],
                row["future_query_private"],
                row["interface_private"],
            )
        ].append(row)
    for selected in unit_groups.values():
        unit_count += 1
        labels = {
            row["semantic_label_private"]
            for row in selected
            if row["semantic_label_private"] is not None
        }
        disagreement_units += int(len(labels) > 1)

    parse_methods = Counter(row["semantic_parse_method"] for row in rows)
    payload = {
        "schema_version": "80.3.0",
        "phase_id": "Phase406-FailureDiagnostic",
        "created_at": now(),
        "stage": stage,
        "case_count": len(rows),
        "first_step_candidate_correct_count": sum(
            row["first_step_candidate_correct"] for row in rows
        ),
        "first_step_global_top_is_target_count": sum(
            row["first_step_global_top_is_target"] for row in rows
        ),
        "semantic_parse_count": sum(
            row["semantic_label_private"] is not None for row in rows
        ),
        "short_sequence_semantic_correct_count": sum(
            row["short_sequence_semantic_correct"] for row in rows
        ),
        "sequence_stop_or_boundary_after_semantic_count": sum(
            row["sequence_stop_or_boundary_after_semantic"] for row in rows
        ),
        "candidate_correct_sequence_wrong_count": sum(
            row["first_step_candidate_correct"]
            and not row["short_sequence_semantic_correct"]
            for row in rows
        ),
        "first_vocab_wrong_sequence_correct_count": sum(
            not row["first_step_global_top_is_target"]
            and row["short_sequence_semantic_correct"]
            for row in rows
        ),
        "semantic_correct_without_stop_or_boundary_count": sum(
            row["short_sequence_semantic_correct"]
            and not row["sequence_stop_or_boundary_after_semantic"]
            for row in rows
        ),
        "surface_disagreement_unit_count": disagreement_units,
        "surface_unit_count": unit_count,
        "parse_method_counts": dict(sorted(parse_methods.items())),
        "nonfinite_any_generated_step_case_count": sum(
            not row["all_generated_step_logits_valid"] for row in rows
        ),
        "failure_axes_path": "analysis/phase406_failure_axes.jsonl",
        "claim_boundary": {
            "semantic_parser_is_conservative_observer": True,
            "parse_failure_is_model_semantic_failure": False,
            "surface_disagreement_is_internal_mechanism": False,
        },
    }
    write_jsonl(OUT / "analysis" / "phase406_failure_axes.jsonl", axes)
    write_json(OUT / f"phase406_{stage}_failure_diagnostic.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("discovery", "calibration", "behavioral_holdout"),
        required=True,
    )
    args = parser.parse_args()
    main(args.stage)
