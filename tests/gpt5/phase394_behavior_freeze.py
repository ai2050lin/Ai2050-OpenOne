#!/usr/bin/env python3
"""Freeze Phase394 behavior eligibility without replacing failed groups."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase379_global_layout_protocol import first_target_step  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase394_binding_separation"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("field_extraction", "relation_qa", "entity_recency")
CONDITIONS = (
    "A_direct_lex_x",
    "B_swapped_lex_x",
    "C_direct_lex_y",
    "D_swapped_lex_y",
)
SPLIT_COUNTS = {"discovery": 12, "calibration": 6, "physical_holdout": 6}
SALT = "phase394-formal-binding-freeze-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


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


def main() -> None:
    cases = read_jsonl(OUT / "protocol/private/phase394_candidate_cases.jsonl")
    case_by_id = {row["blind_case_id"]: row for row in cases}
    if len(case_by_id) != 864:
        raise RuntimeError(f"Expected 864 Phase394 cases, got {len(case_by_id)}")

    behavior: list[dict[str, Any]] = []
    for model in MODELS:
        complete = read_json(OUT / "behavior" / model / "complete.json")
        if not complete["valid"] or complete["execution_batch_size"] != 1:
            raise RuntimeError(f"Invalid Phase394 behavior output for {model}")
        behavior.extend(read_jsonl(OUT / "behavior/private" / model / "rows.jsonl"))
    if len(behavior) != 864:
        raise RuntimeError(f"Expected 864 Phase394 behavior rows, got {len(behavior)}")

    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )

    enriched: list[dict[str, Any]] = []
    for row in behavior:
        case = case_by_id[row["blind_case_id"]]
        step = first_target_step(
            tokenizers[row["model"]],
            row["generated_token_ids"],
            case["target_aliases"],
        )
        has_next = step is not None and step + 1 < len(row["generated_token_ids"])
        generated_lower = row["generated_text"].lower()
        rejected = case["distractors"][0].lower()
        enriched.append(
            {
                **row,
                "target_decision_step_private": step,
                "post_decision_next_token_available": has_next,
                "formal_behavior_qualified": bool(row["strict_behavior_correct"] and has_next),
                "generated_contains_rejected_value": rejected in generated_lower,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["anonymous_parallel_group_id"]].append(row)
    surface_by_group = {
        group_id: rows[0]["task_surface_private"] for group_id, rows in grouped.items()
    }
    qualified_groups = {
        group_id
        for group_id, rows in grouped.items()
        if len(rows) == len(MODELS) * len(CONDITIONS)
        and {row["model"] for row in rows} == set(MODELS)
        and all(row["formal_behavior_qualified"] for row in rows)
    }
    qualified_counts = Counter(surface_by_group[group_id] for group_id in qualified_groups)
    eligible_surfaces = [surface for surface in SURFACES if qualified_counts[surface] == 24]

    selected: dict[str, dict[str, list[str]]] = {}
    assignment: dict[str, tuple[str, str]] = {}
    for surface in eligible_surfaces:
        ordered = sorted(
            (group_id for group_id in qualified_groups if surface_by_group[group_id] == surface),
            key=lambda group_id: digest(f"{SALT}:{surface}:{group_id}"),
        )
        selected[surface] = {}
        cursor = 0
        for split, count in SPLIT_COUNTS.items():
            split_groups = ordered[cursor : cursor + count]
            selected[surface][split] = split_groups
            for group_id in split_groups:
                assignment[group_id] = (split, "p394g_" + digest(f"{SALT}:{group_id}", 24))
            cursor += count

    selected_cases: list[dict[str, Any]] = []
    for row in enriched:
        source_group = row["anonymous_parallel_group_id"]
        if source_group not in assignment:
            continue
        split, public_group = assignment[source_group]
        case = case_by_id[row["blind_case_id"]]
        selected_cases.append(
            {
                **case,
                "schema_version": "68.2.0",
                "phase_id": "Phase394-FrozenDenominator",
                "phase394_split": split,
                "phase394_public_parallel_group_id": public_group,
                "generated_text_private": row["generated_text"],
                "generated_token_ids": row["generated_token_ids"],
                "target_decision_step": int(row["target_decision_step_private"]),
                "strict_behavior_correct": True,
                "post_decision_next_token_available": True,
                "semantic_labels_available_to_collection": False,
            }
        )
    private = OUT / "protocol/private"
    write_jsonl(private / "phase394_frozen_execution_cases.jsonl", selected_cases)
    for split in SPLIT_COUNTS:
        write_jsonl(
            private / f"phase394_{split}_cases.jsonl",
            [row for row in selected_cases if row["phase394_split"] == split],
        )

    condition_rows = []
    for model in MODELS:
        for surface in SURFACES:
            for condition in CONDITIONS:
                rows = [
                    row for row in enriched
                    if row["model"] == model
                    and row["task_surface_private"] == surface
                    and row["contrast_condition_private"] == condition
                ]
                condition_rows.append(
                    {
                        "model": model,
                        "task_surface": surface,
                        "condition": condition,
                        "case_count": len(rows),
                        "strict_correct_count": sum(row["strict_behavior_correct"] for row in rows),
                        "rejected_value_output_count": sum(row["generated_contains_rejected_value"] for row in rows),
                    }
                )
    write_jsonl(OUT / "phase394_behavior_condition_rows.jsonl", condition_rows)

    summary = {
        "schema_version": "68.2.0",
        "phase_id": "Phase394-BehaviorFreeze",
        "created_at": now(),
        "denominator": {
            "candidate_case_count": len(behavior),
            "candidate_parallel_group_count": len(grouped),
            "qualified_parallel_group_count": len(qualified_groups),
            "eligible_surface_count": len(eligible_surfaces),
            "selected_case_count": len(selected_cases),
        },
        "model_results": {
            model: {
                "strict_correct_count": sum(
                    row["strict_behavior_correct"] for row in enriched if row["model"] == model
                ),
                "case_count": sum(row["model"] == model for row in enriched),
                "direct_condition_strict_count": sum(
                    row["strict_behavior_correct"]
                    for row in enriched
                    if row["model"] == model
                    and row["contrast_condition_private"].startswith(("A_", "C_"))
                ),
                "swapped_condition_strict_count": sum(
                    row["strict_behavior_correct"]
                    for row in enriched
                    if row["model"] == model
                    and row["contrast_condition_private"].startswith(("B_", "D_"))
                ),
                "swapped_rejected_value_output_count": sum(
                    row["generated_contains_rejected_value"]
                    for row in enriched
                    if row["model"] == model
                    and row["contrast_condition_private"].startswith(("B_", "D_"))
                ),
            }
            for model in MODELS
        },
        "surface_gates": [
            {
                "task_surface": surface,
                "candidate_group_count": 24,
                "qualified_group_count": qualified_counts[surface],
                "eligible": surface in eligible_surfaces,
                "exclusion_reason": (
                    None if surface in eligible_surfaces
                    else "not_all_24_three_model_four_condition_groups_qualified"
                ),
            }
            for surface in SURFACES
        ],
        "eligible_surfaces": eligible_surfaces,
        "selected_groups_private": selected,
        "results": {
            "formal_pointer_interface_crossmodel_qualified": bool(eligible_surfaces),
            "internal_event_collection_authorized": bool(eligible_surfaces),
            "formal_binding_state_candidate_count": 0,
            "natural_language_binding_rejected": False,
        },
        "authorization": {
            "run_internal_event_collection": bool(eligible_surfaces),
            "run_formal_binding_causal_intervention": False,
            "run_formal_to_natural_transfer": False,
            "design_independent_natural_binding_denominator": True,
            "run_single_neuron_scan": False,
        },
        "claim_boundary": {
            "formal_pointer_failure_means_no_binding_state": False,
            "formal_pointer_failure_means_artificial_interface_not_shared": True,
            "natural_language_binding_was_tested": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase394_behavior_freeze_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
