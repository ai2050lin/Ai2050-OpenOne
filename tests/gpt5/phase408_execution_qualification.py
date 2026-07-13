#!/usr/bin/env python3
"""Qualify deterministic and finite Phase408 execution before discovery."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase407_event_horizon_collection import configure_determinism  # noqa: E402
from phase408_partition_interface_collection import (  # noqa: E402
    generate_case,
    read_json,
    read_jsonl,
    write_json,
)
from phase408_partition_interface_protocol import (  # noqa: E402
    FAMILIES,
    MODELS,
    OUT,
    QUALIFICATION_CASE_COUNT,
)


SOURCE = OUT / "protocol/private/phase408_all_cases.jsonl"
DTYPE_CANDIDATES = {
    "qwen3": ("float16",),
    "glm4": ("float16", "bfloat16"),
    "deepseek7b": ("bfloat16",),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_token_ids_private": row["generated_token_ids_private"],
        "eos_step_private": row["eos_step_private"],
        "raw_step_validity": [
            step["raw_logits_valid"] for step in row["step_ledger_private"]
        ],
        "processed_step_validity": [
            step["processed_scores_valid"] for step in row["step_ledger_private"]
        ],
        "raw_argmax_token_ids": [
            step["top_k_private"][0]["token_id_private"]
            for step in row["step_ledger_private"]
        ],
        "generated_token_logprobs": [
            step["generated_token_logprob"] for step in row["step_ledger_private"]
        ],
    }


def digest_signature(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@torch.inference_mode()
def run_dtype(model_key: str, dtype: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = None
    prior_dtype = os.environ.get("PROBE_TORCH_DTYPE")
    records = []
    try:
        os.environ["PROBE_TORCH_DTYPE"] = dtype
        configure_determinism()
        loaded = load_probe_model(model_key)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace(
            "torch.", ""
        )
        if runtime_dtype != dtype:
            raise RuntimeError(
                f"Phase408 qualification dtype mismatch {model_key}: "
                f"{runtime_dtype} != {dtype}"
            )
        for index, case in enumerate(cases, 1):
            first = generate_case(loaded, case)
            second = generate_case(loaded, case)
            first_signature = trace_signature(first)
            second_signature = trace_signature(second)
            exact_repeat = first_signature == second_signature
            first_finite = (
                first["all_generated_raw_logits_valid"]
                and first["all_generated_processed_scores_valid"]
            )
            second_finite = (
                second["all_generated_raw_logits_valid"]
                and second["all_generated_processed_scores_valid"]
            )
            records.append(
                {
                    "blind_case_id": case["blind_case_id"],
                    "family_id": case["family_id"],
                    "interface_private": case["interface_private"],
                    "first_signature_sha256": digest_signature(first_signature),
                    "second_signature_sha256": digest_signature(second_signature),
                    "exact_repeat": exact_repeat,
                    "first_all_finite": first_finite,
                    "second_all_finite": second_finite,
                    "H48_repeat": first["H48_right_edge_reached"]
                    == second["H48_right_edge_reached"],
                }
            )
            if index % 4 == 0 or index == len(cases):
                print(
                    f"[{model_key}/phase408/qualification/{dtype}] "
                    f"{index}/{len(cases)}",
                    flush=True,
                )
        family_counts = {
            family: sum(row["family_id"] == family for row in records)
            for family in FAMILIES
        }
        return {
            "runtime_dtype": runtime_dtype,
            "case_count": len(records),
            "repeat_count_per_case": 2,
            "family_case_counts": family_counts,
            "exact_repeat_count": sum(row["exact_repeat"] for row in records),
            "all_finite_case_count": sum(
                row["first_all_finite"] and row["second_all_finite"]
                for row in records
            ),
            "H48_repeat_count": sum(row["H48_repeat"] for row in records),
            "valid": len(records) == QUALIFICATION_CASE_COUNT
            and all(row["exact_repeat"] for row in records)
            and all(row["first_all_finite"] for row in records)
            and all(row["second_all_finite"] for row in records)
            and all(row["H48_repeat"] for row in records),
            "records": records,
        }
    finally:
        release_loaded(loaded)
        if prior_dtype is None:
            os.environ.pop("PROBE_TORCH_DTYPE", None)
        else:
            os.environ["PROBE_TORCH_DTYPE"] = prior_dtype
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_dtype(model_key: str, audits: dict[str, dict[str, Any]]) -> str | None:
    if model_key == "glm4":
        if audits.get("bfloat16", {}).get("valid"):
            return "bfloat16"
        if audits.get("float16", {}).get("valid"):
            return "float16"
        return None
    for dtype in DTYPE_CANDIDATES[model_key]:
        if audits[dtype]["valid"]:
            return dtype
    return None


def run(model_key: str) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(SOURCE)
        if row["private_execution_model"] == model_key
        and row["execution_qualification_case"]
    ]
    if len(cases) != QUALIFICATION_CASE_COUNT:
        raise RuntimeError(
            f"Phase408 qualification denominator {model_key}: {len(cases)}"
        )
    audits = {}
    for dtype in DTYPE_CANDIDATES[model_key]:
        audit_path = OUT / "qualification/private" / f"{model_key}_{dtype}.json"
        if audit_path.is_file():
            audit = read_json(audit_path)
        else:
            audit = run_dtype(model_key, dtype, cases)
            write_json(audit_path, audit)
        audits[dtype] = audit
    selected = select_dtype(model_key, audits)
    payload = {
        "schema_version": "82.2.0",
        "phase_id": "Phase408-ExecutionQualification",
        "created_at": now(),
        "model": model_key,
        "case_count": len(cases),
        "repeat_count_per_case": 2,
        "batch_size": 1,
        "dtype_selection_rule": (
            "glm4_prefers_all_finite_exact_bfloat16_then_float16; "
            "other_models_use_their_single_frozen_candidate"
        ),
        "dtype_audits": audits,
        "selected_runtime_dtype": selected,
        "valid": selected is not None,
        "semantic_outputs_inspected_for_protocol_revision": False,
    }
    write_json(OUT / "qualification" / f"{model_key}_complete.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    run(args.model)
