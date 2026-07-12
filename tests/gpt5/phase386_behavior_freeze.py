#!/usr/bin/env python3
"""Freeze Phase386 discovery, calibration, and sealed physical groups."""

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


PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    "relation_binding",
    "entity_recency",
    "number_agreement",
    "target_vs_wrong",
    "missing_condition_control",
    "field_extraction",
)
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
SPLIT_COUNTS = {"discovery": 8, "calibration": 4, "physical_holdout": 4}
REQUIRED_GROUPS = sum(SPLIT_COUNTS.values())
SALT = "phase386-multitime-fresh-groups-v1"


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
    amendment = read_json(PHASE_ROOT / "phase386_behavior_budget_amendment.json")
    if not amendment["pilot_cohort_retired"] or amendment["replacement_max_new_tokens"] != 24:
        raise RuntimeError("Phase386 behavior budget amendment is not frozen")
    execution = {
        row["blind_case_id"]: row
        for row in read_jsonl(
            PHASE_ROOT / "protocol/private/phase386_candidate_execution_cases.jsonl"
        )
    }
    behavior: list[dict[str, Any]] = []
    for model in MODELS:
        complete = read_json(PHASE_ROOT / "behavior/models" / model / "complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase386 behavior output for {model}")
        behavior.extend(
            read_jsonl(
                PHASE_ROOT
                / "behavior/private/models"
                / model
                / "phase386_behavior_rows.jsonl"
            )
        )
    if len(behavior) != 2880:
        raise RuntimeError(f"Expected 2880 behavior rows, got {len(behavior)}")

    tokenizers: dict[str, Any] = {}
    try:
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
            case = execution[row["blind_case_id"]]
            step = first_target_step(
                tokenizers[row["model"]],
                row["generated_token_ids"],
                case["target_aliases"],
            )
            has_next = step is not None and step + 1 < len(row["generated_token_ids"])
            enriched.append(
                {
                    **row,
                    "target_decision_step_private": step,
                    "post_decision_next_token_available": has_next,
                    "multitime_behavior_qualified": bool(
                        row["strict_behavior_correct"] and has_next
                    ),
                }
            )
    finally:
        tokenizers.clear()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["anonymous_parallel_group_id"]].append(row)
    mechanism_by_group = {
        group_id: rows[0]["mechanism_id_private"]
        for group_id, rows in grouped.items()
    }
    qualified_groups = {
        group_id
        for group_id, rows in grouped.items()
        if len(rows) == len(MODELS) * len(CONDITIONS)
        and {row["model"] for row in rows} == set(MODELS)
        and all(row["multitime_behavior_qualified"] for row in rows)
        and all(
            sum(
                row["model"] == model
                and row["contrast_condition_private"] == condition
                for row in rows
            )
            == 1
            for model in MODELS
            for condition in CONDITIONS
        )
    }
    qualified_counts = Counter(
        mechanism_by_group[group_id] for group_id in qualified_groups
    )
    eligible_mechanisms = [
        mechanism
        for mechanism in MECHANISMS
        if qualified_counts[mechanism] >= REQUIRED_GROUPS
    ]
    selected_by_mechanism: dict[str, dict[str, list[str]]] = {}
    assignment: dict[str, tuple[str, str]] = {}
    for mechanism in eligible_mechanisms:
        ordered = sorted(
            (
                group_id
                for group_id in qualified_groups
                if mechanism_by_group[group_id] == mechanism
            ),
            key=lambda group_id: digest(f"{SALT}:{mechanism}:{group_id}"),
        )[:REQUIRED_GROUPS]
        selected_by_mechanism[mechanism] = {}
        cursor = 0
        for split, count in SPLIT_COUNTS.items():
            groups = ordered[cursor : cursor + count]
            selected_by_mechanism[mechanism][split] = groups
            for group_id in groups:
                assignment[group_id] = (
                    split,
                    "p386g_" + digest(f"{SALT}:{group_id}", 24),
                )
            cursor += count

    selected_cases: list[dict[str, Any]] = []
    for row in enriched:
        source_group = row["anonymous_parallel_group_id"]
        if source_group not in assignment:
            continue
        split, public_group = assignment[source_group]
        base = execution[row["blind_case_id"]]
        selected_cases.append(
            {
                **base,
                "schema_version": "60.2.0",
                "phase_id": "Phase386-FrozenDenominator",
                "phase386_split": split,
                "phase386_public_parallel_group_id": public_group,
                "generated_text_private": row["generated_text"],
                "generated_token_ids": row["generated_token_ids"],
                "target_decision_step": int(row["target_decision_step_private"]),
                "strict_behavior_correct": True,
                "post_decision_next_token_available": True,
                "behavior_max_new_tokens": 24,
                "semantic_labels_available_to_collection": False,
            }
        )

    expected_selected = len(eligible_mechanisms) * REQUIRED_GROUPS * 4 * 3
    if len(selected_cases) != expected_selected:
        raise RuntimeError(
            f"Phase386 selected case mismatch: {len(selected_cases)} != {expected_selected}"
        )
    private = PHASE_ROOT / "protocol/private"
    write_jsonl(private / "phase386_frozen_execution_cases.jsonl", selected_cases)
    for split in SPLIT_COUNTS:
        write_jsonl(
            private / f"phase386_{split}_cases.jsonl",
            [row for row in selected_cases if row["phase386_split"] == split],
        )
    instrument_groups = {
        selected_by_mechanism[mechanism]["discovery"][0]
        for mechanism in eligible_mechanisms
    }
    instrument_cases = [
        {
            **row,
            "phase386_collection_stage": "instrument_audit",
            "instrument_group_reused_in_discovery": True,
        }
        for row in selected_cases
        if row["anonymous_parallel_group_id"] in instrument_groups
    ]
    write_jsonl(
        private / "phase386_instrument_audit_cases.jsonl",
        instrument_cases,
    )
    write_jsonl(
        PHASE_ROOT / "protocol/phase386_frozen_blind_groups.jsonl",
        [
            {
                "schema_version": "60.2.0",
                "phase_id": "Phase386-FrozenDenominator",
                "public_parallel_group_id": public_group,
                "split": split,
                "all_three_models_all_four_conditions_correct": True,
                "post_decision_next_token_available": True,
                "semantic_label_exported": False,
            }
            for _source_group, (split, public_group) in sorted(assignment.items())
        ],
    )
    gates = [
        {
            "mechanism_id": mechanism,
            "qualified_parallel_group_count": qualified_counts[mechanism],
            "minimum_required": REQUIRED_GROUPS,
            "eligible": mechanism in eligible_mechanisms,
            "exclusion_reason": (
                None
                if mechanism in eligible_mechanisms
                else "fewer_than_16_three_model_four_condition_groups"
            ),
        }
        for mechanism in MECHANISMS
    ]
    summary = {
        "schema_version": "60.2.0",
        "phase_id": "Phase386-BehaviorFreeze",
        "created_at": now(),
        "denominator": {
            "candidate_case_count": len(behavior),
            "candidate_parallel_group_count": len(grouped),
            "multitime_qualified_parallel_group_count": len(qualified_groups),
            "eligible_mechanism_count": len(eligible_mechanisms),
            "selected_parallel_group_count": len(assignment),
            "selected_case_count": len(selected_cases),
            "instrument_audit_parallel_group_count": len(instrument_groups),
            "instrument_audit_case_count": len(instrument_cases),
            "split_group_counts_per_mechanism": SPLIT_COUNTS,
        },
        "gates": gates,
        "eligible_mechanisms": eligible_mechanisms,
        "selected_groups_private": selected_by_mechanism,
        "behavior_budget": {
            "retired_pilot_max_new_tokens": 8,
            "frozen_max_new_tokens": 24,
            "all_models_rerun": True,
            "failed_cases_selectively_replaced": False,
        },
        "internal_collection_started_before_freeze": False,
        "physical_holdout_opened": False,
        "authorization": {
            "run_instrument_audit": bool(eligible_mechanisms),
            "run_discovery_before_instrument_audit": False,
            "run_calibration_before_relation_freeze": False,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
        },
        "claim_boundary": {
            "excluded_mechanism_may_be_replaced": False,
            "behavior_qualified_mechanism_is_internal_path": False,
        },
    }
    write_json(PHASE_ROOT / "phase386_behavior_freeze_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
