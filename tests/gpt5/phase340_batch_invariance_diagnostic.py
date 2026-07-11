#!/usr/bin/env python3
"""Check whether GLM4 Phase340 baseline behavior depends on rollout batching."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase339_cross_task_boundary_audit import generate_batch  # noqa: E402
from phase340_cross_task_protocol_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def run(model: str = "glm4", round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    cases = [row for row in read_jsonl(root / "phase340_registered_cases.jsonl") if row["model"] == model]
    batch_rows = {
        row["case_id"]: row
        for row in read_jsonl(root / "models" / model / "phase340_rollout_rows.jsonl")
    }
    rows: list[dict[str, Any]] = []
    loaded = None
    try:
        loaded = load_probe_model(model)
        for index, case in enumerate(cases):
            single = generate_batch(loaded, [case], None, None, 16)[0]
            batched = batch_rows[case["case_id"]]
            rows.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "model": model,
                "case_id": case["case_id"], "mechanism_id": case["mechanism_id"],
                "split": case["split"], "template_id": case["template_id"],
                "target": case["target"],
                "batch6_text": batched["answer_head_text"],
                "batch6_correct": batched["answer_head_semantic_correct"],
                "batch1_text": single["answer_head_text"],
                "batch1_correct": single["answer_head_semantic_correct"],
                "text_equal": batched["answer_head_text"] == single["answer_head_text"],
                "correctness_equal": batched["answer_head_semantic_correct"] == single["answer_head_semantic_correct"],
                "internal_intervention": False,
            })
            if (index + 1) % 54 == 0 or index + 1 == len(cases):
                print(f"[{model} batch1] {index + 1}/{len(cases)}", flush=True)
        summary = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "case_count": len(rows),
            "batch6_accuracy": round(sum(row["batch6_correct"] for row in rows) / len(rows), 7),
            "batch1_accuracy": round(sum(row["batch1_correct"] for row in rows) / len(rows), 7),
            "text_invariance_rate": round(sum(row["text_equal"] for row in rows) / len(rows), 7),
            "correctness_invariance_rate": round(sum(row["correctness_equal"] for row in rows) / len(rows), 7),
            "failure_recovered_count": sum(not row["batch6_correct"] and row["batch1_correct"] for row in rows),
            "success_lost_count": sum(row["batch6_correct"] and not row["batch1_correct"] for row in rows),
            "batch_invariant": all(row["text_equal"] for row in rows),
            "internal_intervention": False,
        }
        write_jsonl(root / "models" / model / "phase340_batch_invariance_rows.jsonl", rows)
        write_json(root / "models" / model / "phase340_batch_invariance_summary.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm4", choices=("glm4",))
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
