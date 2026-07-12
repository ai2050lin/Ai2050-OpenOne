#!/usr/bin/env python3
"""Collect exact full-layer decision traces for the frozen Phase381 groups."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase379_decision_aligned_trace import decision_input, trace_batch  # noqa: E402
from phase381_joint_state_case_bank import read_jsonl, write_json, write_jsonl  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
CASES = OUT / "private/phase381_qualified_trace_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def process(model: str, batch_size: int) -> dict[str, Any]:
    gate = read_json(OUT / "phase381_behavior_analysis_final_summary.json")
    if not gate["authorization"]["run_exact_trace"]:
        raise RuntimeError("Phase381 final behavior gate did not authorize tracing")
    cases = [
        row for row in read_jsonl(CASES) if row["private_execution_model"] == model
    ]
    if len(cases) != 96:
        raise RuntimeError(f"Expected 96 Phase381 trace cases for {model}, got {len(cases)}")
    loaded = None
    rows: list[dict[str, Any]] = []
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
                batch_rows = trace_batch(
                    loaded,
                    [row[0] for row in selected],
                    [row[1] for row in selected],
                    [row[2] for row in selected],
                    destination,
                    artifact_root=OUT,
                )
                for metadata, selected_row in zip(batch_rows, selected, strict=True):
                    metadata["schema_version"] = "54.3.0"
                    metadata["phase_id"] = "Phase381-ExactTrace"
                    metadata["phase381_split"] = selected_row[0]["phase381_split"]
                rows.extend(batch_rows)
                completed += len(selected)
                print(f"[{model}] Phase381 exact trace {completed}/96", flush=True)
        write_jsonl(OUT / "trace/models" / model / "phase381_trace_rows.jsonl", rows)
        layer_count = len(get_layers(loaded.model))
        summary = {
            "schema_version": "54.3.0",
            "phase_id": "Phase381-ExactTrace",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "case_count": len(rows),
            "layer_count": layer_count,
            "exact_event_vector_count": len(rows) * layer_count * 4 * 3,
            "baseline_replay_match_count": sum(
                row["baseline_replay_matches_observed_target_token"] for row in rows
            ),
            "baseline_replay_mismatch_count": sum(
                not row["baseline_replay_matches_observed_target_token"] for row in rows
            ),
            "semantic_labels_available_to_trace": False,
            "top_k_used": False,
            "valid": len(rows) == 96,
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
