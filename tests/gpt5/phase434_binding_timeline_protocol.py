#!/usr/bin/env python3
"""Freeze the Phase434 relation-binding timeline protocol.

The protocol keeps the Phase433 result immutable.  It uses a mixed paired design:
record order and role mapping are crossed within each candidate semantic group,
while lexical aliases are balanced between independent groups.  This preserves a
large independent denominator without pretending that a Cartesian expansion of
the same vocabulary creates new evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase434_binding_timeline"
PHASE_ID = "Phase434-BindingTimelineProtocol"
SCHEMA_VERSION = "phase434_binding_timeline.v1"
TRACE_SCHEMA_VERSION = "phase434_binding_timeline_trace.v1"

MODELS = ("qwen3", "glm4", "deepseek7b")
DTYPES = {"qwen3": "float16", "glm4": "bfloat16", "deepseek7b": "bfloat16"}
LANGUAGE_MODEL = "qwen3"
ROLES = ("a", "b")
TIMINGS = (
    "before_records",
    "after_role_a",
    "after_role_b",
    "after_records",
    "near_query",
)
RECORD_ORDERS = ("ab", "ba")
MAPPINGS = ("direct", "swapped")
BEHAVIOR_SPLITS = ("behavior_discovery", "behavior_holdout")
PHYSICAL_SPLIT = "physical_calibration"
SEALED_SPLIT = "sealed_physical_holdout"
STRESS_SPLIT = "conflict_stress"
GROUPS_BY_SPLIT = {
    "behavior_discovery": 96,
    "behavior_holdout": 192,
    "physical_calibration": 192,
    "sealed_physical_holdout": 192,
    "conflict_stress": 96,
}

ROLE_ALIAS_PAIRS = (
    ("role-ax", "role-by"),
    ("role-cz", "role-dw"),
    ("role-ev", "role-fu"),
    ("role-gt", "role-hs"),
)
CUE_ALIAS_PAIRS = (
    ("cue-ax", "cue-by"),
    ("cue-cz", "cue-dw"),
    ("cue-ev", "cue-fu"),
    ("cue-gt", "cue-hs"),
)
NEUTRAL_CUE = "cue-00"

BLOCKS = (
    {
        "block_id": "language_action_binding_timeline_candidate",
        "family_id": "language_action",
        "mechanism_id": "relation_binding_formation_timeline",
        "candidate": True,
        "matched_control_block_id": "language_action_binding_timeline_control",
    },
    {
        "block_id": "language_action_binding_timeline_control",
        "family_id": "language_action",
        "mechanism_id": "timing_invariant_stable_result_control",
        "candidate": False,
        "matched_control_block_id": "language_action_binding_timeline_candidate",
    },
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def factor_assignment(index: int) -> dict[str, Any]:
    """Return a balanced 128-cell lexical/control assignment."""

    cell = index % 128
    role_alias_index = cell % 4
    cue_alias_index = (cell // 4) % 4
    baseline_order = RECORD_ORDERS[(cell // 16) % 2]
    baseline_mapping = MAPPINGS[(cell // 32) % 2]
    parity = cell.bit_count() % 2
    control_target = "source_2" if parity else "source_1"
    query_execution_order = RECORD_ORDERS[
        parity
    ]
    return {
        "role_alias_index": role_alias_index,
        "cue_alias_index": cue_alias_index,
        "role_aliases": {
            "a": ROLE_ALIAS_PAIRS[role_alias_index][0],
            "b": ROLE_ALIAS_PAIRS[role_alias_index][1],
        },
        "cue_aliases": {
            "a": CUE_ALIAS_PAIRS[cue_alias_index][0],
            "b": CUE_ALIAS_PAIRS[cue_alias_index][1],
        },
        "control_target_source": control_target,
        "baseline_record_order": baseline_order,
        "baseline_mapping": baseline_mapping,
        "query_execution_order": query_execution_order,
        "factor_cell": cell,
        "replicate_index": index // 128,
    }


def build_group(block: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    split_order = (*BEHAVIOR_SPLITS, PHYSICAL_SPLIT, SEALED_SPLIT, STRESS_SPLIT)
    split_index = split_order.index(split)
    serial = 4340000 + split_index * 10000 + index
    shared = f"unit-{serial:07d}-shared"
    source_1 = f"{shared}-path-red-zone-sun"
    source_2 = f"{shared}-path-blu-area-ice"
    factors = factor_assignment(index)
    pair_id = f"phase434__{split}__pair_{index:03d}"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        **block,
        "split": split,
        "pipeline_sealed": split == SEALED_SPLIT,
        "stress_only": split == STRESS_SPLIT,
        "contract_variant": "no_examples_five_slot_binding_timeline",
        "group_index": index,
        "paired_group_id": pair_id,
        "semantic_group_id": f"{pair_id}__{block['block_id']}",
        "source_1": source_1,
        "source_2": source_2,
        "shared_stem": shared,
        **factors,
    }


def build_groups() -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        split: [] for split in GROUPS_BY_SPLIT
    }
    for split, count in GROUPS_BY_SPLIT.items():
        blocks = BLOCKS if split != STRESS_SPLIT else (BLOCKS[0],)
        for block in blocks:
            rows[split].extend(build_group(block, split, index) for index in range(count))
    return rows


def conditions_per_group(candidate: bool, split: str) -> int:
    if split == STRESS_SPLIT:
        return len(ROLES) * 2
    if candidate:
        return len(RECORD_ORDERS) * len(MAPPINGS) * len(ROLES) * len(TIMINGS)
    return len(ROLES) * len(TIMINGS)


def denominator_audit(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter((row["block_id"], row["split"]) for values in rows.values() for row in values)
    expected_group_counts = {
        split: GROUPS_BY_SPLIT[split] * (1 if split == STRESS_SPLIT else len(BLOCKS))
        for split in GROUPS_BY_SPLIT
    }
    condition_counts = {}
    for split, values in rows.items():
        condition_counts[split] = sum(
            conditions_per_group(bool(row["candidate"]), split) for row in values
        )
    all_rows = [row for values in rows.values() for row in values]
    paired_vocab = all(
        len(row["source_1"]) == len(row["source_2"])
        and row["source_1"].startswith(row["shared_stem"])
        and row["source_2"].startswith(row["shared_stem"])
        for row in all_rows
    )
    split_vocab = {
        split: {row["shared_stem"] for row in values}
        for split, values in rows.items()
    }
    disjoint = all(
        not split_vocab[left].intersection(split_vocab[right])
        for left_index, left in enumerate(split_vocab)
        for right in list(split_vocab)[left_index + 1 :]
    )
    balance = {}
    for split, values in rows.items():
        candidate = [row for row in values if row["candidate"]]
        balance[split] = {
            "role_alias": dict(Counter(row["role_alias_index"] for row in candidate)),
            "cue_alias": dict(Counter(row["cue_alias_index"] for row in candidate)),
            "baseline_order": dict(Counter(row["baseline_record_order"] for row in candidate)),
            "baseline_mapping": dict(Counter(row["baseline_mapping"] for row in candidate)),
            "control_target": dict(Counter(row["control_target_source"] for row in values if not row["candidate"])),
        }
    valid = bool(
        all(len(values) == expected_group_counts[split] for split, values in rows.items())
        and len({row["semantic_group_id"] for row in all_rows}) == len(all_rows)
        and all(not row["pipeline_sealed"] for split, values in rows.items() if split != SEALED_SPLIT for row in values)
        and all(row["pipeline_sealed"] for row in rows[SEALED_SPLIT])
        and paired_vocab
        and disjoint
        and condition_counts["behavior_discovery"] == 4800
        and condition_counts["behavior_holdout"] == 9600
        and condition_counts[PHYSICAL_SPLIT] == 9600
        and condition_counts[SEALED_SPLIT] == 9600
        and condition_counts[STRESS_SPLIT] == 384
    )
    return {
        "valid": valid,
        "design": "candidate paired full order-mapping crossing; lexical aliases balanced between independent groups",
        "groups_by_split": {key: len(value) for key, value in rows.items()},
        "independent_groups_per_block": dict(GROUPS_BY_SPLIT),
        "conditions_by_split_per_model": condition_counts,
        "behavior_open_conditions_per_model": condition_counts["behavior_discovery"] + condition_counts["behavior_holdout"] + condition_counts[STRESS_SPLIT],
        "three_model_behavior_open_conditions": (
            condition_counts["behavior_discovery"] + condition_counts["behavior_holdout"] + condition_counts[STRESS_SPLIT]
        ) * len(MODELS),
        "physical_open_conditions_per_eligible_model": condition_counts[PHYSICAL_SPLIT],
        "sealed_conditions_qwen_if_unlocked": condition_counts[SEALED_SPLIT],
        "surface_length_matched": paired_vocab,
        "vocabulary_disjoint_across_splits": disjoint,
        "factor_balance": balance,
        "counts": {"::".join(key): value for key, value in sorted(counts.items())},
    }


def implementation_hashes() -> dict[str, str | None]:
    names = (
        "phase434_binding_timeline_protocol.py",
        "phase434_binding_timeline_collect.py",
        "phase434_binding_timeline_analysis.py",
        "test_phase434_binding_timeline.py",
    )
    return {
        name: (
            hashlib.sha256((ROOT / "tests/gpt5" / name).read_bytes()).hexdigest()
            if (ROOT / "tests/gpt5" / name).exists()
            else None
        )
        for name in names
    }


def freeze() -> dict[str, Any]:
    rows = build_groups()
    audit = denominator_audit(rows)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    for split, values in rows.items():
        if split == SEALED_SPLIT:
            path = OUT / "sealed/phase434_groups_sealed.jsonl"
        else:
            path = OUT / f"phase434_groups_{split}.jsonl"
        write_jsonl(path, values)
    commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": SEALED_SPLIT,
        "sealed_group_count": len(rows[SEALED_SPLIT]),
        "sealed_condition_count": audit["sealed_conditions_qwen_if_unlocked"],
        "sealed_group_rows_sha256": digest_rows(rows[SEALED_SPLIT]),
        "read_requires_open_gate": True,
        "open_analysis_must_not_import_sealed_rows": True,
    }
    write_json(OUT / "phase434_sealed_commitment.json", commitment)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models_in_execution_order": list(MODELS),
        "execution_dtypes": DTYPES,
        "language_interpretation_model": LANGUAGE_MODEL,
        "timings": list(TIMINGS),
        "record_orders": list(RECORD_ORDERS),
        "mappings": list(MAPPINGS),
        "behavior_splits": list(BEHAVIOR_SPLITS),
        "physical_split": PHYSICAL_SPLIT,
        "sealed_split": SEALED_SPLIT,
        "stress_split": STRESS_SPLIT,
        "denominator_audit": audit,
        "rows_sha256": {
            split: digest_rows(values) for split, values in rows.items() if split != SEALED_SPLIT
        },
        "sealed_commitment": commitment,
        "event_contract": {
            "minimum_common_prefix_tokens_each_model": 2,
            "minimum_post_divergence_tokens_each_event": 2,
            "minimum_post_divergence_mismatches": 2,
            "same_first_token_required": True,
            "same_surface_length_required": True,
            "complete_sequence_behavior_is_primary": True,
            "all_five_selector_slots_always_present": True,
            "exactly_one_active_selector_slot": True,
        },
        "behavior_contract": {
            "behavior_before_physical": True,
            "discovery_groups_per_block": GROUPS_BY_SPLIT["behavior_discovery"],
            "holdout_groups_per_block": GROUPS_BY_SPLIT["behavior_holdout"],
            "late_timings": ["after_records", "near_query"],
            "stress_nonblocking": True,
            "model_specific_graph_allowed": True,
            "cross_model_claim_requires_models": 2,
        },
        "physical_contract": {
            "behavior_qualified_models_only": True,
            "positions": [
                "selector_slot_end",
                "role_a_result_end",
                "role_b_result_end",
                "after_records_end",
                "question_end",
                "instruction_end",
                "assistant_boundary",
                "prompt_terminal",
                "teacher_branch_boundary",
            ],
            "state_sketch_dimensions": 16,
            "state_sketch_seed": 4340715,
            "state_sketch_is_output_label_blind": True,
            "equivariance_metric": "same-source paired sketch distance versus different-source paired sketch distance",
            "physical_internal_discovery_groups": 96,
            "physical_internal_holdout_groups": 96,
            "unknown_hidden_permutation_is_not_assumed": True,
            "no_head_channel_neuron_scan": True,
        },
        "numeric_gates": {
            "token_contract_valid_fraction_min": 1.0,
            "late_behavior_group_lcb_min": 0.95,
            "control_all_timing_group_lcb_min": 0.95,
            "physical_group_coverage_fraction_min": 1.0,
            "physical_finite_fraction_min": 1.0,
            "binding_geometry_effect_min": 0.05,
            "binding_geometry_holdout_direction_agreement_min": 0.80,
            "candidate_control_specificity_min": 0.05,
            "hook_hidden_state_max_abs_error": 0.001,
        },
        "gate_order": [
            "G0_natural_behavior_qualification",
            "G1_token_position_cache_component_identity",
            "G2_label_blind_binding_geometry",
            "G3_binding_time_replication",
            "G4_source_specific_transport",
            "G5_complete_event_holdout_prediction",
            "G6_matched_control_specificity",
            "G7_sealed_replication",
            "G8_optional_aggregate_causal",
        ],
        "source_transport_is_not_inferred_from_state_geometry": True,
        "sealed_unlock_requires_G0_through_G6": True,
        "causal_unlock_requires_sealed_pass": True,
        "implementation_hashes": implementation_hashes(),
    }
    write_json(OUT / "phase434_protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
