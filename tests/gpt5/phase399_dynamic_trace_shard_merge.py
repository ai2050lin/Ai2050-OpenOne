#!/usr/bin/env python3
"""Merge short-lived Phase399 trace shards into the frozen stage denominator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
STAGES = ("discovery", "calibration", "physical_holdout")
EXPECTED_GROUPS = {"discovery": 30, "calibration": 15, "physical_holdout": 15}


def read_json(path: Path) -> Any:
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


def main(model: str, stage: str) -> None:
    root = OUT / "dynamic_trace" / stage / "private/models" / model
    shard_roots = sorted((root / "shards").glob("shard_*"))
    if not shard_roots:
        raise RuntimeError(f"No Phase399 shards for {model}/{stage}")
    summaries = [read_json(path / "complete.json") for path in shard_roots]
    if any(not item["valid"] for item in summaries):
        raise RuntimeError(f"Invalid Phase399 shard for {model}/{stage}")
    group_ids = [group for item in summaries for group in item["selected_group_ids"]]
    expected_groups = EXPECTED_GROUPS[stage]
    if len(group_ids) != expected_groups or len(set(group_ids)) != expected_groups:
        raise RuntimeError(
            f"Phase399 shard group mismatch for {model}/{stage}: "
            f"{len(group_ids)}/{len(set(group_ids))} != {expected_groups}"
        )
    event_rows = [
        row
        for path in shard_roots
        for row in read_jsonl(path / "event_trajectory_rows.jsonl")
    ]
    group_rows = [
        row
        for path in shard_roots
        for row in read_jsonl(path / "group_audit_rows.jsonl")
    ]
    if len(group_rows) != expected_groups:
        raise RuntimeError(f"Phase399 merged audit mismatch for {model}/{stage}")
    write_jsonl(root / "event_trajectory_rows.jsonl", event_rows)
    write_jsonl(root / "group_audit_rows.jsonl", group_rows)
    complete = {
        "schema_version": "73.4.1",
        "phase_id": "Phase399-DynamicTraceShardMerge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "stage": stage,
        "execution_mode": "short_lived_three_group_shards",
        "shard_count": len(shard_roots),
        "case_count": expected_groups * 16,
        "group_count": expected_groups,
        "layer_count": summaries[0]["layer_count"],
        "event_trajectory_row_count": len(event_rows),
        "quality_group_count": sum(row["quality_gate_pass"] for row in group_rows),
        "max_block_relative_error": max(
            row["max_block_relative_error"] for row in group_rows
        ),
        "max_attention_replay_relative_error": max(
            row["max_attention_replay_relative_error"] for row in group_rows
        ),
        "max_probability_sum_error": max(
            row["max_probability_sum_error"] for row in group_rows
        ),
        "group_ids": sorted(group_ids),
        "valid": all(row["quality_gate_pass"] for row in group_rows),
    }
    write_json(root / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    main(args.model, args.stage)
