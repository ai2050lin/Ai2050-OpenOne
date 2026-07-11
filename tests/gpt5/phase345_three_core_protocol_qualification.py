#!/usr/bin/env python3
"""Run Phase345 baseline qualification on the official single-case path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase338_block_causal_screen import score_cases  # noqa: E402
from phase339_cross_task_boundary_audit import generate_batch  # noqa: E402
from phase345_three_core_protocol_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")


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


def base(case: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "model": model, "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "family_id": case["family_id"], "mechanism_id": case["mechanism_id"],
        "task_class": case["task_class"], "split": case["split"],
        "template_id": case["template_id"], "target": case["target"],
        "condition": "baseline", "execution_mode": "b1_left_cache0",
        "internal_intervention": False,
    }


def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    cases = [row for row in read_jsonl(root / "phase345_registered_cases.jsonl") if row["model"] == model]
    phrase_rows = []
    rollout_rows = []
    loaded = None
    try:
        loaded = load_probe_model(model)
        for index, case in enumerate(cases):
            score = score_cases(loaded, [case], None, None)[0]
            rollout = generate_batch(loaded, [case], None, None, 24)[0]
            valid = all(math.isfinite(float(value)) for value in score.values())
            phrase_rows.append({
                **base(case, model),
                **{
                    key: round(float(value), 7) if math.isfinite(float(value)) else None
                    for key, value in score.items()
                },
                "score_valid": valid,
            })
            rollout_rows.append({**base(case, model), **rollout})
            if (index + 1) % 72 == 0 or index + 1 == len(cases):
                print(f"[{model}] {index + 1}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        write_jsonl(model_root / "phase345_phrase_rows.jsonl", phrase_rows)
        write_jsonl(model_root / "phase345_rollout_rows.jsonl", rollout_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "registered_case_count": len(cases),
            "phrase_row_count": len(phrase_rows), "rollout_row_count": len(rollout_rows),
            "invalid_phrase_row_count": sum(not row["score_valid"] for row in phrase_rows),
            "valid": len(cases) == 864 and len(phrase_rows) == 864 and len(rollout_rows) == 864,
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
