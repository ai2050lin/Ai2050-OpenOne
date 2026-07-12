#!/usr/bin/env python3
"""Collect Phase380 exact full-layer traces after the common behavior gate."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase379_decision_aligned_trace import decision_input, trace_batch  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
CASES = OUT / "private/phase380_qualified_trace_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def process(model: str, batch_size: int) -> dict[str, Any]:
    gate = read_json(OUT / "phase380_behavior_analysis_summary.json")
    if not gate["authorization"]["run_exact_trace"]:
        raise RuntimeError("Phase380 behavior gate did not authorize tracing")
    cases = [
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
    ]
    loaded = None
    rows = []
    try:
        loaded = load_probe_model(model)
        prepared = [(case, *decision_input(loaded, case)) for case in cases]
        buckets: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
        for row in prepared:
            buckets[len(row[1])].append(row)
        destination = OUT / "trace/private/models" / model / "cases"
        completed = 0
        for _length, bucket in sorted(buckets.items()):
            for start in range(0, len(bucket), batch_size):
                selected = bucket[start : start + batch_size]
                rows.extend(
                    trace_batch(
                        loaded,
                        [row[0] for row in selected],
                        [row[1] for row in selected],
                        [row[2] for row in selected],
                        destination,
                        artifact_root=OUT,
                    )
                )
                completed += len(selected)
                print(
                    f"[{model}] Phase380 exact trace {completed}/{len(cases)}",
                    flush=True,
                )
        metadata_path = OUT / "trace/models" / model / "phase380_trace_rows.jsonl"
        write_jsonl(metadata_path, rows)
        summary = {
            "schema_version": "53.3.0",
            "phase_id": "Phase380-ExactTrace",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "case_count": len(rows),
            "layer_count": len(get_layers(loaded.model)),
            "exact_event_vector_count": len(rows)
            * len(get_layers(loaded.model))
            * 4
            * 3,
            "baseline_replay_match_count": sum(
                row["baseline_replay_matches_observed_target_token"] for row in rows
            ),
            "baseline_replay_mismatch_count": sum(
                not row["baseline_replay_matches_observed_target_token"] for row in rows
            ),
            "semantic_labels_available_to_trace": False,
            "top_k_used": False,
            "valid": len(rows) == len(cases),
        }
        write_json(OUT / "trace/models" / model / "complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(process(args.model, args.batch_size), ensure_ascii=False, indent=2))
