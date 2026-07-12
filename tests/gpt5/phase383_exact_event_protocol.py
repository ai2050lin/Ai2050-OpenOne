#!/usr/bin/env python3
"""Freeze the Phase383 decision-aligned exact-component event denominator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
OUT = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    "relation_binding",
    "entity_recency",
    "number_agreement",
    "target_vs_wrong",
)
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
SPLIT_GROUP_COUNTS = {
    "instrument_audit": 1,
    "discovery": 3,
    "calibration": 2,
    "physical_holdout": 1,
}
RUNTIME_DTYPE_BY_MODEL = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}
SALT = "phase383-balanced-replay-qualified-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def replay_qualified_groups(
    cases: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    single_by_case: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        path = (
            OUT
            / "qualification/private/models"
            / model
            / "phase383_single_path_rows.jsonl"
        )
        if not path.is_file():
            raise RuntimeError(
                f"Run phase383_single_path_qualification.py first: missing {path}"
            )
        for row in read_jsonl(path):
            single_by_case[row["blind_case_id"]] = row

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["anonymous_parallel_group_id"]].append(case)

    qualified: dict[str, list[str]] = defaultdict(list)
    for group_id, rows in sorted(grouped.items()):
        mechanism = rows[0]["mechanism_id"]
        models = Counter(row["private_execution_model"] for row in rows)
        conditions = Counter(row["contrast_condition"] for row in rows)
        complete = (
            len(rows) == len(MODELS) * len(CONDITIONS)
            and models == Counter({model: len(CONDITIONS) for model in MODELS})
            and conditions == Counter({condition: len(MODELS) for condition in CONDITIONS})
            and all(row["strict_behavior_correct"] for row in rows)
            and all(
                single_by_case.get(row["blind_case_id"], {}).get(
                    "single_path_qualified", False
                )
                for row in rows
            )
        )
        if complete:
            qualified[mechanism].append(group_id)
    return qualified, single_by_case


def main() -> None:
    source_summary = read_json(SOURCE / "phase380_residual_validation_summary.json")
    cases = read_jsonl(SOURCE / "private/phase380_qualified_trace_cases.jsonl")
    qualified, single_by_case = replay_qualified_groups(cases)
    required = sum(SPLIT_GROUP_COUNTS.values())
    if set(qualified) != set(MECHANISMS):
        raise RuntimeError(f"Mechanism mismatch: {sorted(qualified)}")
    if any(len(qualified[mechanism]) < required for mechanism in MECHANISMS):
        raise RuntimeError(
            "Phase383 needs seven replay-qualified groups per mechanism: "
            + repr({key: len(value) for key, value in qualified.items()})
        )

    assignment: dict[str, tuple[str, str, int]] = {}
    selected_groups: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        ordered = sorted(
            qualified[mechanism],
            key=lambda group: digest(f"{SALT}:{mechanism}:{group}", 64),
        )[:required]
        cursor = 0
        for split, count in SPLIT_GROUP_COUNTS.items():
            for split_index, group_id in enumerate(ordered[cursor : cursor + count]):
                public_group_id = "p383g_" + digest(f"{SALT}:{group_id}")
                assignment[group_id] = (split, public_group_id, split_index)
                selected_groups.append(
                    {
                        "schema_version": "57.0.0",
                        "phase_id": "Phase383-Protocol",
                        "public_parallel_group_id": public_group_id,
                        "phase383_split": split,
                        "mechanism_id_private": mechanism,
                        "split_index_private": split_index,
                        "source_parallel_group_id_private": group_id,
                    }
                )
            cursor += count

    selected_cases = []
    public_rows = []
    for case in cases:
        source_group = case["anonymous_parallel_group_id"]
        if source_group not in assignment:
            continue
        split, public_group_id, _split_index = assignment[source_group]
        selected = dict(case)
        selected.update(
            {
                "schema_version": "57.0.0",
                "phase_id": "Phase383-Protocol",
                "phase383_split": split,
                "phase383_public_parallel_group_id": public_group_id,
                "phase383_case_id": "p383c_"
                + digest(f"{SALT}:{case['blind_case_id']}", 28),
                "source_phase380_case_id": case["blind_case_id"],
            }
        )
        selected_cases.append(selected)
        public_rows.append(
            {
                "schema_version": "57.0.0",
                "phase_id": "Phase383-Protocol",
                "phase383_case_id": selected["phase383_case_id"],
                "phase383_public_parallel_group_id": public_group_id,
                "phase383_split": split,
                "anonymous_model_id": case["anonymous_model_id"],
                "anonymous_condition_slot": case["anonymous_condition_slot"],
                "semantic_decision_aligned": True,
                "strict_behavior_correct": True,
                "baseline_replay_qualified": True,
            }
        )

    expected_case_count = required * len(MECHANISMS) * len(MODELS) * len(CONDITIONS)
    if len(selected_cases) != expected_case_count:
        raise RuntimeError(
            f"Expected {expected_case_count} Phase383 cases, got {len(selected_cases)}"
        )
    counts = Counter(
        (row["phase383_split"], row["mechanism_id"], row["private_execution_model"])
        for row in selected_cases
    )
    expected_counts = {
        (split, mechanism, model): group_count * len(CONDITIONS)
        for split, group_count in SPLIT_GROUP_COUNTS.items()
        for mechanism in MECHANISMS
        for model in MODELS
    }
    if counts != Counter(expected_counts):
        raise RuntimeError("Phase383 split/model/mechanism balance failed")

    private_root = OUT / "protocol/private"
    write_jsonl(private_root / "phase383_execution_cases.jsonl", selected_cases)
    write_jsonl(private_root / "phase383_group_key.jsonl", selected_groups)
    for split in SPLIT_GROUP_COUNTS:
        write_jsonl(
            private_root / f"phase383_{split}_cases.jsonl",
            [row for row in selected_cases if row["phase383_split"] == split],
        )
    write_jsonl(OUT / "protocol/phase383_blind_case_registry.jsonl", public_rows)

    protocol = {
        "schema_version": "57.0.0",
        "phase_id": "Phase383-Protocol",
        "created_at": now(),
        "objective": (
            "Build a semantic-decision-aligned exact component-event graph before "
            "any new state intervention."
        ),
        "source_denominator": {
            "source_phase": "Phase380",
            "source_registered_parallel_group_count": source_summary["denominator"][
                "registered_parallel_group_count"
            ],
            "source_replay_qualified_groups_by_mechanism": source_summary[
                "denominator"
            ]["replay_qualified_groups_by_mechanism"],
            "single_path_qualified_case_count_by_model": {
                model: sum(
                    row["model"] == model and row["single_path_qualified"]
                    for row in single_by_case.values()
                )
                for model in MODELS
            },
            "single_path_qualified_groups_by_mechanism": {
                mechanism: len(qualified[mechanism]) for mechanism in MECHANISMS
            },
            "source_batch_contract_reused": False,
            "single_sample_requalification_completed": True,
            "retrospective_algorithm_development_denominator": True,
            "independent_scientific_confirmation_denominator": False,
        },
        "frozen_denominator": {
            "models": list(MODELS),
            "mechanisms": list(MECHANISMS),
            "conditions": list(CONDITIONS),
            "split_group_counts_per_mechanism": SPLIT_GROUP_COUNTS,
            "balanced_group_count_per_mechanism": required,
            "parallel_group_count": required * len(MECHANISMS),
            "case_count": len(selected_cases),
            "case_count_by_split": dict(
                Counter(row["phase383_split"] for row in selected_cases)
            ),
        },
        "instrument_contract": {
            "semantic_time": "target_decision",
            "runtime_dtype_by_model": RUNTIME_DTYPE_BY_MODEL,
            "runtime_dtype_is_scientific_contract": True,
            "receiver_roles": [
                "source",
                "query",
                "answer_start",
                "current_generation",
            ],
            "exact_replayable_families": [
                "attention_head_source_write",
                "mlp_channel_write",
                "attention_residual_merge",
                "mlp_residual_merge",
                "layer_residual_transition",
                "vocabulary_state",
            ],
            "top_k_used": False,
            "component_relative_error_max": 0.01,
            "probability_sum_error_max": 0.01,
            "threshold_source": "Phase358-366 frozen conservation protocol",
        },
        "stage_gates": {
            "instrument_audit": "all three models and all component gates must pass",
            "discovery": "opens only after instrument audit",
            "calibration": "opens only after a signed event map and thresholds are frozen",
            "physical_holdout": "remains unopened in descriptive event-map development",
            "causal_intervention": "not authorized by this protocol",
        },
        "claim_boundary": {
            "seven_balanced_groups_replace_requested_twenty_four": True,
            "reason": (
                "number_agreement has only seven single-path-qualified parallel groups; "
                "the denominator is capped before analysis."
            ),
            "phase380_batch_replay_qualification_is_phase383_qualification": False,
            "public_architecture_backbone_complete": False,
            "upstream_functional_layout_discovered": False,
            "language_encoding_closed": False,
        },
        "authorization": {
            "instrument_audit_collection": True,
            "discovery_collection": False,
            "calibration_collection": False,
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
    }
    qualification_summary = {
        "schema_version": "57.0.2",
        "phase_id": "Phase383-SinglePathQualificationAnalysis",
        "created_at": now(),
        "denominator": {
            "candidate_case_count": len(single_by_case),
            "candidate_parallel_group_count": sum(len(values) for values in qualified.values())
            + sum(
                1
                for group_id in {
                    row["anonymous_parallel_group_id"] for row in cases
                }
                if all(
                    group_id not in values for values in qualified.values()
                )
            ),
            "single_path_qualified_case_count_by_model": protocol[
                "source_denominator"
            ]["single_path_qualified_case_count_by_model"],
            "single_path_qualified_groups_by_mechanism": protocol[
                "source_denominator"
            ]["single_path_qualified_groups_by_mechanism"],
        },
        "frozen_runtime": {
            "execution_batch_size": 1,
            "output_attentions": True,
            "runtime_dtype_by_model": RUNTIME_DTYPE_BY_MODEL,
        },
        "results": {
            "phase380_batch_qualified_groups_reused_without_check": False,
            "balanced_group_cap": required,
            "all_four_mechanisms_retain_balanced_denominator": True,
        },
        "claim_boundary": {
            "single_path_qualification_is_language_mechanism": False,
            "batch_path_and_single_path_are_interchangeable": False,
        },
    }
    write_json(OUT / "phase383_single_path_qualification_summary.json", qualification_summary)
    write_json(OUT / "phase383_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
