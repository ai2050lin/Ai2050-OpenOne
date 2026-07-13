#!/usr/bin/env python3
"""Collect Phase403 natural-future responses in the frozen model order."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase401_behavior_qualification import generate_batch  # noqa: E402
from phase403_predictive_state_protocol import (  # noqa: E402
    FAMILIES,
    FROZEN_DTYPES,
    MODELS,
    OUT,
)


SOURCE = OUT / "protocol/private/phase403_all_cases.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold().strip())


def read_json(path: Path) -> dict[str, Any]:
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


def authorized_families(split: str) -> tuple[str, ...]:
    if split == "discovery":
        return FAMILIES
    if split == "calibration":
        gate = read_json(OUT / "phase403_discovery_analysis.json")
        return tuple(gate["crossmodel_candidate_families"])
    if split == "behavioral_holdout":
        gate = read_json(OUT / "phase403_calibration_analysis.json")
        return tuple(gate["crossmodel_candidate_families"])
    raise KeyError(split)


def stage_context_kind(split: str) -> set[str]:
    return {"base", "single"} if split in {"discovery", "calibration"} else {"composition"}


def detect_candidate(text: str, candidates: list[str], mapping: dict[str, str]) -> tuple[str | None, str | None, bool]:
    value = normalize(text)
    detected: list[str] = []
    for candidate in candidates:
        pattern = re.compile(rf"(?<!\w){re.escape(normalize(candidate))}(?!\w)")
        if pattern.search(value):
            detected.append(candidate)
    canonical = {mapping[item] for item in detected if item in mapping}
    if len(canonical) != 1:
        return None, None, len(canonical) > 1
    selected = next(iter(canonical))
    raw = next(item for item in detected if mapping.get(item) == selected)
    return raw, selected, False


@torch.inference_mode()
def run(model: str, split: str) -> dict[str, Any]:
    families = authorized_families(split)
    context_kinds = stage_context_kind(split)
    all_cases = read_jsonl(SOURCE)
    cases = [
        row
        for row in all_cases
        if row["private_execution_model"] == model
        and row["candidate_split_private"] == split
        and row["family_id"] in families
        and row["context_kind_private"] in context_kinds
    ]
    expected = sum(
        1
        for row in all_cases
        if row["private_execution_model"] == model
        and row["candidate_split_private"] == split
        and row["family_id"] in families
        and row["context_kind_private"] in context_kinds
    )
    if len(cases) != expected:
        raise RuntimeError(f"Phase403 selection mismatch: {len(cases)} != {expected}")
    if not cases:
        payload = {
            "schema_version": "77.1.0",
            "phase_id": "Phase403-BehaviorCollection",
            "created_at": now(),
            "model": model,
            "split": split,
            "authorized_families": [],
            "case_count": 0,
            "valid": True,
            "stopped_by_prior_gate": True,
        }
        write_json(OUT / "behavior" / split / model / "complete.json", payload)
        return payload

    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPES[model]:
            raise RuntimeError(
                f"Phase403 dtype mismatch for {model}: {runtime_dtype}"
            )
        for index, case in enumerate(cases, 1):
            generated, execution = generate_batch(loaded, [case], 6)
            result = generated[0]
            predicted_raw, predicted_canonical, ambiguous = detect_candidate(
                result["generated_text_before_stop"],
                case["candidate_answers_private"],
                case["answer_to_canonical_private"],
            )
            semantic_correct = bool(
                result["semantic_correct"]
                and predicted_canonical == case["expected_canonical_private"]
            )
            rows.append(
                {
                    "schema_version": "77.1.0",
                    "phase_id": "Phase403-BehaviorCollection",
                    "created_at": now(),
                    "model": model,
                    "runtime_dtype": runtime_dtype,
                    "split": split,
                    "batch_size": 1,
                    "attention_implementation": "eager",
                    "use_cache": True,
                    "blind_case_id": case["blind_case_id"],
                    "family_id": case["family_id"],
                    "anonymous_parallel_group_id": case[
                        "anonymous_parallel_group_id"
                    ],
                    "group_priority": case["group_priority"],
                    "state_variant_private": case["state_variant_private"],
                    "surface_id_private": case["surface_id_private"],
                    "operation_context_private": case[
                        "operation_context_private"
                    ],
                    "context_kind_private": case["context_kind_private"],
                    "future_query_private": case["future_query_private"],
                    "future_query_role_private": case[
                        "future_query_role_private"
                    ],
                    "expected_canonical_private": case[
                        "expected_canonical_private"
                    ],
                    "predicted_answer_private": predicted_raw,
                    "predicted_canonical_private": predicted_canonical,
                    "candidate_ambiguous_private": ambiguous,
                    "abstract_state_private": case["abstract_state_private"],
                    "target_private": case["target"],
                    "prompt_token_count": case["prompt_token_count"],
                    "unpadding_contract_pass": execution[
                        "formal_single_case_unpadded"
                    ],
                    **result,
                    "semantic_correct": semantic_correct,
                }
            )
            if index % 128 == 0 or index == len(cases):
                print(
                    f"[{model}/phase403/{split}] {index}/{len(cases)} "
                    f"semantic={sum(row['semantic_correct'] for row in rows)}",
                    flush=True,
                )
            if index % 256 == 0:
                gc.collect()

        family_counts = Counter(row["family_id"] for row in rows)
        family_correct = Counter(
            row["family_id"] for row in rows if row["semantic_correct"]
        )
        payload = {
            "schema_version": "77.1.0",
            "phase_id": "Phase403-BehaviorCollection",
            "created_at": now(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "split": split,
            "authorized_families": list(families),
            "context_kinds": sorted(context_kinds),
            "batch_size": 1,
            "max_new_tokens": 6,
            "case_count": len(rows),
            "semantic_correct_count": sum(row["semantic_correct"] for row in rows),
            "semantic_span_resolved_count": sum(
                row["semantic_span_resolved"] for row in rows
            ),
            "ambiguous_candidate_count": sum(
                row["candidate_ambiguous_private"] for row in rows
            ),
            "unpadding_contract_pass": all(
                row["unpadding_contract_pass"] for row in rows
            ),
            "families": [
                {
                    "family_id": family,
                    "case_count": family_counts[family],
                    "semantic_correct_count": family_correct[family],
                }
                for family in families
            ],
            "valid": len(rows) == expected
            and all(row["unpadding_contract_pass"] for row in rows),
            "claim_boundary": {
                "semantic_success_is_internal_state_evidence": False,
                "finite_future_panel_is_all_future_behavior": False,
            },
        }
        write_jsonl(
            OUT / "behavior" / split / "private" / model / "rows.jsonl", rows
        )
        write_json(OUT / "behavior" / split / model / "complete.json", payload)
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
    parser.add_argument(
        "--split",
        choices=("discovery", "calibration", "behavioral_holdout"),
        required=True,
    )
    args = parser.parse_args()
    run(args.model, args.split)
