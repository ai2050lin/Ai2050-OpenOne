#!/usr/bin/env python3
"""Freeze qualified Phase388 groups, token roles, layers, and target decision prefixes."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


P388 = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
P386_ROWS = (
    ROOT
    / "tests/gpt5/result/phase386_multitime_relation_atlas"
    / "phase386_physical_candidate_rows.jsonl"
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def subsequence_starts(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    ]


def phrase_end_positions(tokenizer: Any, prompt_ids: list[int], text: str) -> list[int]:
    for phrase in (f" to {text}", f" {text}", text):
        phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        starts = subsequence_starts(prompt_ids, phrase_ids)
        if starts:
            return [start + len(phrase_ids) - 1 for start in starts]
    raise RuntimeError(f"Could not locate token role for {text!r}")


def clean_token_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def target_span(tokenizer: Any, token_ids: list[int], target: str) -> tuple[int, int]:
    clean_target = clean_token_text(target)
    for index, token_id in enumerate(token_ids):
        piece = clean_token_text(tokenizer.decode([token_id], skip_special_tokens=True))
        if piece == clean_target or (clean_target and clean_target in piece):
            return index, index + 1
    candidates: list[tuple[int, int]] = []
    for width in range(2, min(7, len(token_ids) + 1)):
        for start in range(len(token_ids) - width + 1):
            end = start + width
            piece = clean_token_text(
                tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
            )
            if piece == clean_target or (clean_target and clean_target in piece):
                candidates.append((start, end))
        if candidates:
            return min(candidates, key=lambda item: (item[0], item[1]))
    raise RuntimeError(f"Could not locate target {target!r} in generated tokens")


def layer_contract() -> dict[str, dict[str, int]]:
    rows = read_jsonl(P386_ROWS)
    upstream = next(
        row
        for row in rows
        if row.get("physical_predictive_relation_path_gate_pass")
        and row["mechanism_id"] == "relation_binding"
        and row["vector_family"] == "attention_head_state"
        and row["source_coordinate"] == "source_encoded"
        and row["target_coordinate"] == "query_integrated"
    )
    terminal = next(
        row
        for row in rows
        if row.get("physical_predictive_relation_path_gate_pass")
        and row["mechanism_id"] == "relation_binding"
        and row["vector_family"] == "attention_output"
        and row["source_coordinate"] == "target_encoded"
        and row["target_coordinate"] == "post_decision_next_token"
    )
    return {
        model: {
            "candidate_layer": int(upstream["model_layers"][model]),
            "terminal_control_layer": int(terminal["model_layers"][model]),
        }
        for model in MODELS
    }


def main() -> None:
    candidates = read_jsonl(
        P388 / "protocol/private/phase388_candidate_execution_cases.jsonl"
    )
    behavior_rows = {
        model: read_jsonl(P388 / "behavior/private" / model / "rows.jsonl")
        for model in MODELS
    }
    by_case = {
        row["blind_case_id"]: row
        for rows in behavior_rows.values()
        for row in rows
    }
    tokenizers = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in candidates:
        grouped[case["parallel_group_id"]].append(case)
    qualified_groups: list[tuple[int, str, list[dict[str, Any]]]] = []
    excluded: list[dict[str, Any]] = []
    prepared_by_id: dict[str, dict[str, Any]] = {}
    for group_id, cases in grouped.items():
        priority = int(cases[0]["group_priority"])
        reasons: list[str] = []
        prepared: list[dict[str, Any]] = []
        for case in cases:
            behavior = by_case.get(case["blind_case_id"])
            if behavior is None or not behavior["strict_behavior_correct"]:
                reasons.append(f"behavior_failure:{case['private_execution_model']}:{case['condition']}")
                continue
            model = case["private_execution_model"]
            tokenizer = tokenizers[model]
            prompt_ids = tokenizer(case["prompt"], add_special_tokens=False)["input_ids"]
            try:
                source_positions = phrase_end_positions(
                    tokenizer, prompt_ids, case["source_entity"]
                )
                wrong_positions = phrase_end_positions(
                    tokenizer, prompt_ids, case["wrong_source_entity"]
                )
                if len(source_positions) < 2 or len(wrong_positions) != 1:
                    raise RuntimeError(
                        f"role occurrence mismatch source={source_positions} wrong={wrong_positions}"
                    )
                generated_ids = [int(item) for item in behavior["generated_token_ids"]]
                start, end = target_span(tokenizer, generated_ids, case["target"])
                if start >= len(generated_ids) or end <= start:
                    raise RuntimeError("empty target span")
                prepared.append(
                    {
                        **case,
                        "prompt_token_ids_private": prompt_ids,
                        "source_position_private": source_positions[0],
                        "query_position_private": source_positions[-1],
                        "wrong_source_position_private": wrong_positions[0],
                        "natural_generated_token_ids_private": generated_ids,
                        "target_decision_prefix_token_ids_private": generated_ids[:start],
                        "target_token_ids_private": generated_ids[start:end],
                        "target_first_token_id_private": generated_ids[start],
                        "target_span_start_private": start,
                        "target_span_end_private": end,
                    }
                )
            except RuntimeError as error:
                reasons.append(f"position_or_target_failure:{model}:{case['condition']}:{error}")
        if reasons or len(prepared) != 6:
            excluded.append(
                {
                    "parallel_group_id": group_id,
                    "group_priority": priority,
                    "reasons": sorted(set(reasons)),
                }
            )
        else:
            qualified_groups.append((priority, group_id, prepared))
            for row in prepared:
                prepared_by_id[row["blind_case_id"]] = row

    qualified_groups.sort()
    if len(qualified_groups) < 18:
        raise RuntimeError(
            f"Phase388 needs 18 three-model paired groups, found {len(qualified_groups)}"
        )
    instrument_ids = [item[1] for item in qualified_groups[:2]]
    causal_ids = [item[1] for item in qualified_groups[2:18]]
    selected_ids = set(instrument_ids + causal_ids)
    selected = [
        row
        for _priority, group_id, rows in qualified_groups
        if group_id in selected_ids
        for row in rows
    ]
    layers = layer_contract()
    for row in selected:
        row.update(layers[row["private_execution_model"]])
        row["phase388_split"] = (
            "instrument_audit"
            if row["parallel_group_id"] in instrument_ids
            else "causal_test"
        )

    write_jsonl(
        P388 / "protocol/private/phase388_frozen_intervention_cases.jsonl",
        selected,
    )
    summary = {
        "schema_version": "62.2.0",
        "phase_id": "Phase388-InterventionFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "candidate_group_count": 24,
            "three_model_paired_behavior_and_position_qualified_group_count": len(
                qualified_groups
            ),
            "instrument_group_count": len(instrument_ids),
            "causal_test_group_count": len(causal_ids),
            "instrument_direction_count": len(instrument_ids) * 2 * len(MODELS),
            "causal_test_direction_count": len(causal_ids) * 2 * len(MODELS),
            "frozen_case_count": len(selected),
        },
        "instrument_group_ids_private": instrument_ids,
        "causal_test_group_ids_private": causal_ids,
        "selected_group_checksum": sha256_ids(instrument_ids + causal_ids),
        "excluded_groups": excluded,
        "model_layers": layers,
        "runtime_contract": {
            "execution_batch_size": 1,
            "actual_incremental_target_prefix_replay": True,
            "target_may_occur_after_model_specific_format_prefix": True,
            "source_query_and_wrong_source_token_roles_frozen_before_intervention": True,
        },
        "authorization": {
            "run_instrument_audit": True,
            "run_causal_test_before_instrument_pass": False,
            "replace_group_after_intervention_starts": False,
            "reuse_phase386_physical_holdout": False,
            "run_single_neuron_scan": False,
        },
    }
    write_json(P388 / "phase388_intervention_freeze.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
