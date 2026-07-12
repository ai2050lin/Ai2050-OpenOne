#!/usr/bin/env python3
"""Freeze qualified Phase371C calibration cases for Phase377 replication."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase376_decision_time_alignment_audit import first_target_step  # noqa: E402


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PHASE376 = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
OUT = ROOT / "tests/gpt5/result/phase377_decision_aligned_calibration"
EXECUTION_CASES = (
    PHASE371
    / "phase371c_case_bank/private/phase371c_nonphysical_execution_cases.jsonl"
)
BEHAVIOR = PHASE371 / "phase371c_behavior_qualification/private/models"
DISCOVERY_PROTOCOL = PHASE376 / "phase376_intervention_protocol.json"
DISCOVERY_SUMMARY = (
    PHASE376 / "phase376_intervention/phase376_intervention_summary.json"
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    discovery = read_json(DISCOVERY_SUMMARY)
    if not discovery["authorization"]["open_calibration"]:
        raise RuntimeError("Phase376 did not authorize calibration")
    winning = [
        row
        for row in discovery["cross_model_rows"]
        if row["heterogeneous_level2_winner_flip_pass"]
    ]
    expected = {
        ("entity_recency", "late", "residual_current"),
        ("entity_recency", "late", "residual_source_query_current"),
        ("relation_binding", "late", "residual_current"),
        ("relation_binding", "late", "residual_source_query_current"),
    }
    found = {
        (row["mechanism_id"], row["relative_depth"], row["template"])
        for row in winning
    }
    if found != expected:
        raise RuntimeError(f"Unexpected calibration candidates: {found}")

    raw_cases = [
        row
        for row in read_jsonl(EXECUTION_CASES)
        if row["phase371c_split"] == "sealed_calibration"
        and row["mechanism_id"] in {"relation_binding", "entity_recency"}
    ]
    case_by_id = {row["blind_case_id"]: row for row in raw_cases}
    behavior_by_model = {
        model: [
            row
            for row in read_jsonl(BEHAVIOR / model / "phase371c_behavior_rows.jsonl")
            if row["phase371c_split"] == "sealed_calibration"
            and row["mechanism_id"] in {"relation_binding", "entity_recency"}
        ]
        for model in MODELS
    }
    qualification: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for model, rows in behavior_by_model.items():
        for row in rows:
            if row["strict_behavior_correct"]:
                qualification[(row["mechanism_id"], row["semantic_group_id"])][model].add(
                    row["contrast_condition"].split("_", 1)[0]
                )
    qualified_groups = {
        key
        for key, model_conditions in qualification.items()
        if set(model_conditions) == set(MODELS)
        and all(values == {"A", "B", "C", "D"} for values in model_conditions.values())
    }
    counts: dict[str, int] = defaultdict(int)
    for mechanism, _group in qualified_groups:
        counts[mechanism] += 1
    if counts != {"relation_binding": 6, "entity_recency": 5}:
        raise RuntimeError(f"Unexpected qualified calibration groups: {dict(counts)}")

    selected_rows = []
    decision_distributions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        for behavior in behavior_by_model[model]:
            key = (behavior["mechanism_id"], behavior["semantic_group_id"])
            if key not in qualified_groups:
                continue
            case = case_by_id[behavior["blind_case_id"]]
            step = first_target_step(
                tokenizer, behavior["generated_token_ids"], behavior["target_aliases"]
            )
            if step is None:
                raise RuntimeError(f"No calibration decision step: {behavior['blind_case_id']}")
            decision_distributions[model][str(step)] += 1
            selected_rows.append(
                {
                    **case,
                    "generated_text": behavior["generated_text"],
                    "generated_token_ids": behavior["generated_token_ids"],
                    "strict_behavior_correct": behavior["strict_behavior_correct"],
                    "target_decision_step": step,
                    "phase377_scope": "qualified_calibration",
                }
            )
    if len(selected_rows) != 132:
        raise RuntimeError(f"Expected 132 calibration cases, got {len(selected_rows)}")
    private = OUT / "private/phase377_calibration_cases.jsonl"
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    discovery_protocol = read_json(DISCOVERY_PROTOCOL)
    protocol = {
        "schema_version": "50.0.0",
        "phase_id": "Phase377-CalibrationProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "independently_replicate_phase376_late_residual_content_transfer",
        "scope": {
            "models": list(MODELS),
            "execution_order": list(MODELS),
            "mechanisms": ["relation_binding", "entity_recency"],
            "qualified_group_counts": dict(sorted(counts.items())),
            "case_count": len(selected_rows),
            "case_count_per_model": 44,
            "relative_depth": "late",
            "templates": ["residual_current", "residual_source_query_current"],
            "physical_opened": False,
            "new_prompt_generation": False,
        },
        "transfer_pairs": discovery_protocol["transfer_pairs"],
        "batched_conditions": discovery_protocol["batched_conditions"],
        "primary_readout": discovery_protocol["primary_readout"],
        "frozen_numeric_gates": {
            **discovery_protocol["frozen_numeric_gates"],
            "minimum_independent_groups_per_model_mechanism_template": 4,
        },
        "decision_step_distributions": {
            model: dict(values) for model, values in decision_distributions.items()
        },
        "candidate_contract": winning,
        "input_hashes": {
            "discovery_protocol": sha256(DISCOVERY_PROTOCOL),
            "discovery_summary": sha256(DISCOVERY_SUMMARY),
            "execution_cases": sha256(EXECUTION_CASES),
            **{
                f"behavior_{model}": sha256(
                    BEHAVIOR / model / "phase371c_behavior_rows.jsonl"
                )
                for model in MODELS
            },
        },
        "claim_boundary": {
            "calibration_replication_is_physical_holdout": False,
            "late_residual_carrier_is_upstream_encoding_rule": False,
            "winner_transfer_is_full_generation_sufficiency": False,
            "language_mechanism_claimed": False,
        },
        "authorization": {
            "run_calibration_interventions": True,
            "open_physical_before_calibration_merge": False,
            "single_neuron_scan": False,
        },
    }
    write_json(OUT / "phase377_calibration_protocol.json", protocol)
    summary = {
        "schema_version": "50.0.0",
        "phase_id": "Phase377-CalibrationCases",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "case_count": len(selected_rows),
        "case_count_per_model": 44,
        "qualified_group_counts": dict(sorted(counts.items())),
        "private_case_hash": sha256(private),
        "physical_case_count_loaded": 0,
    }
    write_json(OUT / "phase377_calibration_case_summary.json", summary)
    print(json.dumps({"protocol": protocol, "cases": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
