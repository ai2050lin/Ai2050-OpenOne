#!/usr/bin/env python3
"""Run Phase402 formal behavior at the frozen batch=1 execution shape."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase401_behavior_qualification import generate_batch  # noqa: E402
from phase402_behavior_protocol import (  # noqa: E402
    CANDIDATE_GROUPS_PER_SURFACE,
    CONDITIONS,
)
from phase402_multiparent_protocol import (  # noqa: E402
    FROZEN_DTYPES,
    MODELS,
    OUT,
    SURFACES,
)


SOURCE = OUT / "protocol/private/phase402_candidate_cases.jsonl"
EXPECTED_PER_MODEL = len(SURFACES) * CANDIDATE_GROUPS_PER_SURFACE * len(CONDITIONS)


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


@torch.inference_mode()
def run(model: str) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(SOURCE)
        if row["private_execution_model"] == model
    ]
    if len(cases) != EXPECTED_PER_MODEL:
        raise RuntimeError(
            f"Expected {EXPECTED_PER_MODEL} Phase402 cases for {model}, got {len(cases)}"
        )
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPES[model]:
            raise RuntimeError(
                f"Phase402 dtype mismatch for {model}: {runtime_dtype}"
            )
        for index, case in enumerate(cases, 1):
            generated, execution = generate_batch(loaded, [case], 10)
            result = generated[0]
            rows.append(
                {
                    "schema_version": "76.2.0",
                    "phase_id": "Phase402-BehaviorQualification",
                    "created_at": now(),
                    "model": model,
                    "runtime_dtype": runtime_dtype,
                    "batch_size": 1,
                    "attention_implementation": "eager",
                    "use_cache": True,
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_parallel_group_id": case[
                        "anonymous_parallel_group_id"
                    ],
                    "anonymous_condition_slot": case[
                        "anonymous_condition_slot"
                    ],
                    "candidate_split_private": case["candidate_split_private"],
                    "selection_priority_private": case[
                        "selection_priority_private"
                    ],
                    "group_priority": case["group_priority"],
                    "task_surface_private": case["task_surface_private"],
                    "target_private": case["target"],
                    "prompt_token_count": case["prompt_token_count"],
                    "unpadding_contract_pass": execution[
                        "formal_single_case_unpadded"
                    ],
                    **result,
                }
            )
            if index % 128 == 0 or index == len(cases):
                print(
                    f"[{model}/phase402] {index}/{len(cases)} "
                    f"semantic={sum(row['semantic_correct'] for row in rows)} "
                    f"exact={sum(row['exact_format_match'] for row in rows)}",
                    flush=True,
                )
            if index % 256 == 0:
                gc.collect()

        counts = Counter(row["task_surface_private"] for row in rows)
        correct = Counter(
            row["task_surface_private"] for row in rows if row["semantic_correct"]
        )
        resolved = Counter(
            row["task_surface_private"]
            for row in rows
            if row["semantic_span_resolved"]
        )
        exact = Counter(
            row["task_surface_private"] for row in rows if row["exact_format_match"]
        )
        payload = {
            "schema_version": "76.2.0",
            "phase_id": "Phase402-BehaviorQualification",
            "created_at": now(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "batch_size": 1,
            "max_new_tokens": 10,
            "case_count": len(rows),
            "semantic_correct_count": sum(row["semantic_correct"] for row in rows),
            "semantic_span_resolved_count": sum(
                row["semantic_span_resolved"] for row in rows
            ),
            "exact_format_match_count": sum(
                row["exact_format_match"] for row in rows
            ),
            "stop_observed_count": sum(row["stop_observed"] for row in rows),
            "unpadding_contract_pass": all(
                row["unpadding_contract_pass"] for row in rows
            ),
            "surfaces": [
                {
                    "task_surface": surface,
                    "case_count": counts[surface],
                    "semantic_correct_count": correct[surface],
                    "semantic_span_resolved_count": resolved[surface],
                    "exact_format_match_count": exact[surface],
                }
                for surface in SURFACES
            ],
            "valid": len(rows) == EXPECTED_PER_MODEL
            and all(row["unpadding_contract_pass"] for row in rows),
            "claim_boundary": {
                "behavior_success_is_a_parent_set": False,
                "format_mismatch_is_semantic_failure": False,
            },
        }
        write_jsonl(OUT / "behavior/private" / model / "rows.jsonl", rows)
        write_json(OUT / "behavior" / model / "complete.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    run(args.model)
