#!/usr/bin/env python3
"""Repeat frozen Phase407 cases at batch one before formal collection."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase407_event_horizon_collection import (  # noqa: E402
    configure_determinism,
    generate_case,
    read_jsonl,
    write_json,
)
from phase407_event_horizon_protocol import (  # noqa: E402
    FAMILIES,
    FROZEN_DTYPES,
    MODELS,
    OUT,
)


SOURCE = OUT / "protocol/private/phase407_all_cases.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace_signature(row: dict[str, Any]) -> dict[str, Any]:
    steps = row["step_ledger_private"]
    return {
        "generated_token_ids_private": row["generated_token_ids_private"],
        "eos_step_private": row["eos_step_private"],
        "step_validity": [step["logits_valid"] for step in steps],
        "raw_argmax_token_ids": [
            step["top_k_private"][0]["token_id_private"] for step in steps
        ],
        "target_score_valid": row["target_completion_score_private"]["valid"],
        "foil_score_valid": row["foil_completion_score_private"]["valid"],
        "target_sum_logprob": row["target_completion_score_private"]["sum_logprob"],
        "foil_sum_logprob": row["foil_completion_score_private"]["sum_logprob"],
    }


def digest_signature(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@torch.inference_mode()
def run(model_key: str) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(SOURCE)
        if row["private_execution_model"] == model_key
        and row["execution_qualification_case"]
    ]
    loaded = None
    records = []
    try:
        configure_determinism()
        loaded = load_probe_model(model_key)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace(
            "torch.", ""
        )
        if runtime_dtype != FROZEN_DTYPES[model_key]:
            raise RuntimeError(
                f"Phase407 qualification dtype mismatch {model_key}: {runtime_dtype}"
            )
        for index, case in enumerate(cases, 1):
            first = generate_case(loaded, case)
            second = generate_case(loaded, case)
            first_signature = trace_signature(first)
            second_signature = trace_signature(second)
            exact_repeat = first_signature == second_signature
            records.append(
                {
                    "blind_case_id": case["blind_case_id"],
                    "family_id": case["family_id"],
                    "interface_private": case["interface_private"],
                    "history_mode_private": case["history_mode_private"],
                    "first_signature_sha256": digest_signature(first_signature),
                    "second_signature_sha256": digest_signature(second_signature),
                    "exact_repeat": exact_repeat,
                    "generated_token_ids_equal": first[
                        "generated_token_ids_private"
                    ]
                    == second["generated_token_ids_private"],
                    "nonfinite_pattern_equal": [
                        step["logits_valid"]
                        for step in first["step_ledger_private"]
                    ]
                    == [
                        step["logits_valid"]
                        for step in second["step_ledger_private"]
                    ],
                }
            )
            if index % 4 == 0 or index == len(cases):
                print(
                    f"[{model_key}/phase407/qualification] {index}/{len(cases)}",
                    flush=True,
                )
        family_counts = {
            family: sum(row["family_id"] == family for row in records)
            for family in FAMILIES
        }
        payload = {
            "schema_version": "81.2.0",
            "phase_id": "Phase407-ExecutionQualification",
            "created_at": now(),
            "model": model_key,
            "runtime_dtype": runtime_dtype,
            "case_count": len(records),
            "repeat_count_per_case": 2,
            "batch_size": 1,
            "family_case_counts": family_counts,
            "exact_repeat_count": sum(row["exact_repeat"] for row in records),
            "generated_token_repeat_count": sum(
                row["generated_token_ids_equal"] for row in records
            ),
            "nonfinite_pattern_repeat_count": sum(
                row["nonfinite_pattern_equal"] for row in records
            ),
            "valid": len(records) == 24
            and all(count == 8 for count in family_counts.values())
            and all(row["exact_repeat"] for row in records),
            "semantic_outputs_inspected_for_protocol_revision": False,
            "records": records,
        }
        write_json(OUT / "qualification" / f"{model_key}_complete.json", payload)
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
