#!/usr/bin/env python3
"""Freeze the Phase426 exact-position role-timing denominator.

The active role tag is placed either immediately before or immediately after
the registered source.  A same-length neutral tag occupies the other slot, so
the source and query token indices are identical while the source's causal
past differs.  This is a new denominator; no Phase425 prompt is reused.
"""

from __future__ import annotations

import argparse
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


PHASE_ID = "Phase426-ExactPositionRoleProtocol"
SCHEMA_VERSION = "phase426_exact_position_role.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
OUT = ROOT / "tests/gpt5/result/phase426_exact_position_role_validation"
BLOCKS = (
    {
        "block_id": "language_action_translate_role_swap",
        "family_id": "language_action",
        "mechanism_id": "translate",
        "candidate": True,
        "matched_control_block_id": "language_action_stable_mapping_control",
    },
    {
        "block_id": "language_action_stable_mapping_control",
        "family_id": "language_action",
        "mechanism_id": "translate_role_neutral_control",
        "candidate": False,
        "matched_control_block_id": "language_action_translate_role_swap",
    },
    {
        "block_id": "syntax_relative_clause_role_swap",
        "family_id": "syntax_structure",
        "mechanism_id": "relative_clause_role",
        "candidate": True,
        "matched_control_block_id": "syntax_relative_clause_surface_control",
    },
    {
        "block_id": "syntax_relative_clause_surface_control",
        "family_id": "syntax_structure",
        "mechanism_id": "relative_clause_role_neutral_control",
        "candidate": False,
        "matched_control_block_id": "syntax_relative_clause_role_swap",
    },
)
SPLITS = (
    "discovery",
    "calibration",
    "behavior_holdout",
    "sealed_physical_holdout",
)
INTERFACES = ("direct", "result_field")
HISTORIES = ("bare", "worked_example")
ROLES = ("a", "b")
TIMINGS = ("early_role", "late_role")
GROUPS_PER_BLOCK_SPLIT = 32
LEXICAL_REPLICAS_PER_GROUP = 2
INSTRUMENT_GROUPS_PER_BLOCK = 2
ACTIVE_TAG = {"a": "Routing tag: rolea.", "b": "Routing tag: roleb."}
NEUTRAL_TAG = "Routing tag: rolen."


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suffix(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    value = index
    output = ""
    while True:
        output = alphabet[value % 26] + output
        value = value // 26 - 1
        if value < 0:
            return output


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"Phase426 requires a fast tokenizer for {model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def continuation_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    ]


def branch_contract(
    target_ids: list[int], opposite_ids: list[int]
) -> tuple[list[int], int, int]:
    if not target_ids or not opposite_ids or target_ids == opposite_ids:
        raise ValueError("identical_or_empty_branch")
    common = 0
    while (
        common < len(target_ids)
        and common < len(opposite_ids)
        and target_ids[common] == opposite_ids[common]
    ):
        common += 1
    if common >= len(target_ids) or common >= len(opposite_ids):
        raise ValueError("prefix_branch")
    return target_ids[:common], target_ids[common], opposite_ids[common]


def span_inside(prompt: str, container: str, value: str) -> tuple[int, int]:
    container_start = prompt.find(container)
    if container_start < 0:
        raise RuntimeError(f"Missing container {container!r}")
    value_start = prompt.find(value, container_start, container_start + len(container))
    if value_start < 0:
        raise RuntimeError(f"Missing value {value!r} in {container!r}")
    return value_start, value_start + len(value)


def token_positions(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if not positions:
        raise RuntimeError(f"Empty token span {span}")
    return positions


def history_text(family_id: str, history: str) -> str:
    if history == "bare":
        return ""
    if family_id == "language_action":
        return (
            "Worked example: under rolea, sample item coddemo maps to lumdemo. "
            "Recorded result: lumdemo."
        )
    return (
        "Worked example: in 'Cira greeted Alen, who thanked Boro', rolea selects "
        "Alen. Recorded result: Alen."
    )


def query_tail(interface: str, answer_kind: str) -> tuple[str, str]:
    if interface == "direct":
        return (
            f"Question: Return the selected {answer_kind}.",
            f"Answer with only the exact {answer_kind}:",
        )
    return (
        f"Query: Store the selected {answer_kind} in the result field.",
        "Result:",
    )


def lexical_index(pair: dict[str, Any]) -> int:
    block_number = next(
        index for index, block in enumerate(BLOCKS) if block["block_id"] == pair["block_id"]
    )
    stage_offset = 4000 if pair["instrument"] else 0
    return stage_offset + block_number * 800 + int(pair["pair_index"])


def translation_payload(
    pair: dict[str, Any], role: str, interface: str, history: str, timing: str
) -> dict[str, Any]:
    key = suffix(lexical_index(pair) + 700)
    focus = f"cod{key}"
    target_a = f"lum{key}"
    target_b = f"ner{key}"
    if pair["candidate"]:
        reference = (
            f"Dictionary: item {focus} maps to {target_a} under rolea and to "
            f"{target_b} under roleb."
        )
        target = target_a if role == "a" else target_b
        opposite = target_b if role == "a" else target_a
        role_changes_correct_event = True
    else:
        reference = (
            f"Dictionary: item {focus} maps to {target_a} under both rolea and "
            f"roleb. Item {target_b} is an unrelated distractor."
        )
        target, opposite = target_a, target_b
        role_changes_correct_event = False
    source_line = f"Source item: {focus}."
    query_line, tail = query_tail(interface, "item")
    return assemble_prompt(
        pair,
        role,
        interface,
        history,
        timing,
        reference,
        source_line,
        focus,
        query_line,
        tail,
        target,
        opposite,
        role_changes_correct_event,
    )


def syntax_payload(
    pair: dict[str, Any], role: str, interface: str, history: str, timing: str
) -> dict[str, Any]:
    key = suffix(lexical_index(pair) + 2900).capitalize()
    anchor = f"Ari{key}"
    other = f"Beno{key}"
    neutral = f"Calo{key}"
    context = f"Context: {neutral} greeted {anchor}, who thanked {other}."
    if pair["candidate"]:
        rule = (
            "Selection rule: rolea selects the person inside the relative clause "
            "who acts; roleb selects the person acted on."
        )
        target = anchor if role == "a" else other
        opposite = other if role == "a" else anchor
        role_changes_correct_event = True
    else:
        rule = (
            "Selection rule: both rolea and roleb select the person inside the "
            "relative clause who acts; the other person is a distractor."
        )
        target, opposite = anchor, other
        role_changes_correct_event = False
    reference = context + "\n" + rule
    source_line = f"Source person: {anchor}."
    query_line, tail = query_tail(interface, "person")
    return assemble_prompt(
        pair,
        role,
        interface,
        history,
        timing,
        reference,
        source_line,
        anchor,
        query_line,
        tail,
        target,
        opposite,
        role_changes_correct_event,
    )


def assemble_prompt(
    pair: dict[str, Any],
    role: str,
    interface: str,
    history: str,
    timing: str,
    reference: str,
    source_line: str,
    source: str,
    query_line: str,
    tail: str,
    target: str,
    opposite: str,
    role_changes_correct_event: bool,
) -> dict[str, Any]:
    active = ACTIVE_TAG[role]
    before, after = (
        (active, NEUTRAL_TAG) if timing == "early_role" else (NEUTRAL_TAG, active)
    )
    history_line = history_text(pair["family_id"], history)
    prompt = "\n".join(
        part
        for part in (
            history_line,
            reference,
            before,
            source_line,
            after,
            query_line,
            tail,
        )
        if part
    )
    return {
        "prompt": prompt,
        "target": target,
        "opposite": opposite,
        "source": source,
        "source_line": source_line,
        "query_line": query_line,
        "active_tag": active,
        "neutral_tag": NEUTRAL_TAG,
        "active_tag_slot": "before_source" if timing == "early_role" else "after_source",
        "role_changes_correct_event": role_changes_correct_event,
    }


def pair_rows(instrument: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if instrument:
        split_specs = (("instrument", INSTRUMENT_GROUPS_PER_BLOCK),)
    else:
        split_specs = tuple((split, GROUPS_PER_BLOCK_SPLIT) for split in SPLITS)
    for block in BLOCKS:
        pair_index = 0
        for split, group_count in split_specs:
            for group_in_split in range(group_count):
                group_id = (
                    f"phase426__{block['block_id']}__{split}__group_{group_in_split:03d}"
                )
                for replica in range(LEXICAL_REPLICAS_PER_GROUP):
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE_ID,
                            **block,
                            "pair_id": f"{group_id}__lexical_{replica}",
                            "pair_index": pair_index,
                            "split": split,
                            "pipeline_sealed": split == "sealed_physical_holdout",
                            "instrument": instrument,
                            "replica_group_id": group_id,
                            "lexical_replica": replica,
                            "condition_count_per_model": 16,
                            "strict_human_double_blind": False,
                        }
                    )
                    pair_index += 1
    return rows


def register_condition(
    model: str,
    tokenizer: Any,
    pair: dict[str, Any],
    role: str,
    interface: str,
    history: str,
    timing: str,
) -> dict[str, Any]:
    payload = (
        translation_payload(pair, role, interface, history, timing)
        if pair["family_id"] == "language_action"
        else syntax_payload(pair, role, interface, history, timing)
    )
    prompt = payload["prompt"]
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    target_ids = continuation_ids(tokenizer, payload["target"])
    opposite_ids = continuation_ids(tokenizer, payload["opposite"])
    prefix, target_branch, opposite_branch = branch_contract(target_ids, opposite_ids)
    source_positions = token_positions(
        offsets, span_inside(prompt, payload["source_line"], payload["source"])
    )
    query_start = prompt.find(payload["query_line"])
    query_positions = token_positions(
        offsets, (query_start, query_start + len(payload["query_line"]))
    )
    active_start = prompt.find(payload["active_tag"])
    neutral_start = prompt.find(payload["neutral_tag"])
    active_positions = token_positions(
        offsets, (active_start, active_start + len(payload["active_tag"]))
    )
    neutral_positions = token_positions(
        offsets, (neutral_start, neutral_start + len(payload["neutral_tag"]))
    )
    executed = [*prompt_ids, *prefix]
    key = f"r{role}__i{interface}__h{history}__t{timing}"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": model,
        "condition_id": f"{pair['pair_id']}__{key}__{model}",
        "condition_key": key,
        "pair_id": pair["pair_id"],
        "pair_index": pair["pair_index"],
        "replica_group_id": pair["replica_group_id"],
        "lexical_replica": pair["lexical_replica"],
        "block_id": pair["block_id"],
        "family_id": pair["family_id"],
        "mechanism_id": pair["mechanism_id"],
        "candidate": pair["candidate"],
        "matched_control_block_id": pair["matched_control_block_id"],
        "split": pair["split"],
        "pipeline_sealed": pair["pipeline_sealed"],
        "instrument": pair["instrument"],
        "role": role,
        "interface": interface,
        "history": history,
        "timing": timing,
        "active_tag_slot": payload["active_tag_slot"],
        "role_changes_correct_event": payload["role_changes_correct_event"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "base_prompt_token_count": len(prompt_ids),
        "common_branch_prefix_token_ids": prefix,
        "executed_token_count": len(executed),
        "executed_token_ids_sha256": hashlib.sha256(
            json.dumps(executed, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "prediction_position": len(executed) - 1,
        "source_positions": source_positions,
        "query_positions": query_positions,
        "instruction_control_positions": source_positions,
        "active_role_positions": active_positions,
        "neutral_role_positions": neutral_positions,
        "source_token_count": len(source_positions),
        "query_token_count": len(query_positions),
        "instruction_control_token_count": len(source_positions),
        "target": payload["target"],
        "opposite_target": payload["opposite"],
        "target_sequence_token_ids": target_ids,
        "opposite_sequence_token_ids": opposite_ids,
        "common_target_prefix_length": len(prefix),
        "target_branch_token_id": target_branch,
        "opposite_branch_token_id": opposite_branch,
        "target_word_count": 1,
        "target_absent_from_prompt": payload["target"] not in prompt,
        "negative_control": not pair["candidate"],
        "exact_position_counterfactual_axis": "role_tag_before_vs_after_source",
        "physical": True,
        "observer_overlay": True,
        "predictive": False,
        "causal": False,
    }


def register_rows(
    pairs: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for model, tokenizer in tokenizers.items():
            for role in ROLES:
                for interface in INTERFACES:
                    for history in HISTORIES:
                        for timing in TIMINGS:
                            rows.append(
                                register_condition(
                                    model,
                                    tokenizer,
                                    pair,
                                    role,
                                    interface,
                                    history,
                                    timing,
                                )
                            )
    return rows


def position_mismatch_count(rows: list[dict[str, Any]]) -> int:
    by_counterfactual: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        by_counterfactual[
            (
                row["pair_id"],
                row["model"],
                row["role"],
                row["interface"],
                row["history"],
            )
        ].append(row)
    mismatches = 0
    for pair_rows in by_counterfactual.values():
        if len(pair_rows) != 2:
            mismatches += 1
            continue
        left, right = pair_rows
        invariant = (
            left["source_positions"] == right["source_positions"]
            and left["query_positions"] == right["query_positions"]
            and left["prediction_position"] == right["prediction_position"]
            and left["executed_token_count"] == right["executed_token_count"]
            and left["base_prompt_token_count"] == right["base_prompt_token_count"]
        )
        mismatches += int(not invariant)
    return mismatches


def validate(
    formal_pairs: list[dict[str, Any]],
    instrument_pairs: list[dict[str, Any]],
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    instrument_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    formal_rows = [*open_rows, *sealed_rows]
    all_rows = [*formal_rows, *instrument_rows]
    formal_groups = {row["replica_group_id"] for row in formal_pairs}
    instrument_groups = {row["replica_group_id"] for row in instrument_pairs}
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_pair_counts: Counter[str] = Counter()
    for pair in [*formal_pairs, *instrument_pairs]:
        group_splits[pair["replica_group_id"]].add(pair["split"])
        group_pair_counts[pair["replica_group_id"]] += 1
    model_formal = Counter(row["model"] for row in formal_rows)
    split_groups = Counter(
        (row["block_id"], row["split"])
        for row in formal_pairs
        if row["lexical_replica"] == 0
    )
    valid = bool(
        len(formal_pairs) == 1024
        and len(instrument_pairs) == 16
        and len(formal_groups) == 512
        and len(instrument_groups) == 8
        and len(formal_rows) == 49152
        and len(open_rows) == 36864
        and len(sealed_rows) == 12288
        and len(instrument_rows) == 768
        and all(value == 16384 for value in model_formal.values())
        and all(value == GROUPS_PER_BLOCK_SPLIT for value in split_groups.values())
        and all(value == 2 for value in group_pair_counts.values())
        and all(len(value) == 1 for value in group_splits.values())
        and len({row["condition_id"] for row in all_rows}) == len(all_rows)
        and position_mismatch_count(all_rows) == 0
        and all(row["target_sequence_token_ids"] for row in all_rows)
        and all(row["opposite_sequence_token_ids"] for row in all_rows)
        and all(row["source_positions"] for row in all_rows)
        and all(row["query_positions"] for row in all_rows)
    )
    return {
        "valid": valid,
        "block_count": len(BLOCKS),
        "formal_pair_count": len(formal_pairs),
        "formal_replica_group_count": len(formal_groups),
        "formal_condition_count": len(formal_rows),
        "open_condition_count": len(open_rows),
        "sealed_condition_count": len(sealed_rows),
        "instrument_pair_count": len(instrument_pairs),
        "instrument_replica_group_count": len(instrument_groups),
        "instrument_condition_count": len(instrument_rows),
        "conditions_per_pair_per_model": 16,
        "conditions_per_group_per_model": 32,
        "position_counterfactual_mismatch_count": position_mismatch_count(all_rows),
        "replica_group_split_leak_count": sum(
            len(value) != 1 for value in group_splits.values()
        ),
        "formal_condition_counts_by_model": dict(sorted(model_formal.items())),
        "formal_group_counts_by_block_split": {
            f"{block}::{split}": count
            for (block, split), count in sorted(split_groups.items())
        },
    }


def freeze() -> dict[str, Any]:
    protocol_path = OUT / "phase426_protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if (
            protocol.get("schema_version") == SCHEMA_VERSION
            and protocol.get("validation", {}).get("valid")
        ):
            return protocol
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    formal_pairs = pair_rows(instrument=False)
    instrument_pairs = pair_rows(instrument=True)
    formal_rows = register_rows(formal_pairs, tokenizers)
    instrument_rows = register_rows(instrument_pairs, tokenizers)
    open_rows = [row for row in formal_rows if not row["pipeline_sealed"]]
    sealed_rows = [row for row in formal_rows if row["pipeline_sealed"]]
    validation = validate(
        formal_pairs,
        instrument_pairs,
        open_rows,
        sealed_rows,
        instrument_rows,
    )
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))

    write_jsonl(OUT / "phase426_registered_pairs.jsonl", formal_pairs)
    write_jsonl(OUT / "phase426_instrument_pairs.jsonl", instrument_pairs)
    write_jsonl(OUT / "phase426_registered_conditions_open.jsonl", open_rows)
    write_jsonl(OUT / "phase426_instrument_conditions.jsonl", instrument_rows)
    sealed_path = OUT / "sealed" / "phase426_registered_conditions_sealed.jsonl"
    write_jsonl(sealed_path, sealed_rows)
    sealed_commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": "sealed_physical_holdout",
        "sealed_condition_count": len(sealed_rows),
        "sealed_condition_rows_sha256": digest_rows(sealed_rows),
        "sealed_prompt_hashes_sha256": hashlib.sha256(
            "\n".join(sorted(row["prompt_sha256"] for row in sealed_rows)).encode(
                "utf-8"
            )
        ).hexdigest(),
        "strict_human_double_blind": False,
        "pipeline_sealed_until_gate_freeze": True,
    }
    write_json(OUT / "phase426_sealed_commitment.json", sealed_commitment)

    implementations = (
        ROOT / "tests/gpt5/phase426_exact_position_protocol.py",
        ROOT / "tests/gpt5/phase426_exact_position_collect.py",
        ROOT / "tests/gpt5/phase426_exact_position_analysis.py",
    )
    missing = [str(path) for path in implementations if not path.exists()]
    if missing:
        raise RuntimeError(f"Create Phase426 implementations before freeze: {missing}")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "objective": (
            "Test whether a reusable role component exceeds matched controls and is "
            "carried from an exactly position-matched causal source into full events."
        ),
        "models": list(MODELS),
        "blocks": list(BLOCKS),
        "interfaces": list(INTERFACES),
        "histories": list(HISTORIES),
        "roles": list(ROLES),
        "timings": list(TIMINGS),
        "execution_dtype_by_model": {
            "qwen3": "float16",
            "glm4": "bfloat16",
            "deepseek7b": "bfloat16",
        },
        "split_contract": {
            "groups_per_block_per_split": GROUPS_PER_BLOCK_SPLIT,
            "lexical_replicas_per_group": LEXICAL_REPLICAS_PER_GROUP,
            "conditions_per_pair_per_model": 16,
            "conditions_per_group_per_model": 32,
            "formal_independent_group_count": 512,
            "instrument_independent_group_count": 8,
            "split_unit": "lexical_replica_group",
        },
        "exact_position_contract": {
            "active_tag_early": "active tag before source; neutral tag after source",
            "active_tag_late": "neutral tag before source; active tag after source",
            "source_positions_equal": True,
            "query_positions_equal": True,
            "executed_lengths_equal": True,
            "causal_interpretation": (
                "the late active tag cannot causally alter the earlier source state; "
                "both timings can alter the later query state"
            ),
            "same_position_does_not_imply_same_state": True,
        },
        "registered_thresholds": {
            "component_ledger_relative_error_max": 0.01,
            "role_covariance_min": 0.20,
            "conditional_covariance_min": 0.20,
            "replica_signal_ratio_min": 1.0,
            "exact_position_specificity_min": 0.0,
            "candidate_minus_control_min": 0.0,
            "positive_group_fraction_min": 2 / 3,
            "source_to_write_identity_alignment_min": 0.20,
            "teacher_sequence_correct_fraction_min": 0.75,
            "natural_target_fraction_min": 0.75,
            "prediction_r2_min": 0.0,
            "prediction_delta_r2_min": 0.05,
            "prediction_mae_gain_min": 0.0,
            "cross_model_replication_min": 2,
            "natural_generation_max_new_tokens": 8,
        },
        "geometry_contract": {
            "radial_distance": "abs(norm(x)-norm(y))/(0.5*(norm(x)+norm(y))+eps)",
            "angular_distance": "1-cosine(x,y)",
            "normalized_delta_range": "[0,2] up to epsilon rounding",
            "role_component_and_role_dominance_are_separate": True,
            "role_dominance_is_descriptive_only": True,
            "identity_map_is_primary": True,
            "learned_three_dimensional_map_pre_registered": False,
        },
        "prediction_contract": {
            "fit_split": "discovery",
            "gate_splits": ["calibration", "behavior_holdout"],
            "baseline_features": [
                "executed_token_count_mean",
                "target_sequence_token_count_mean",
            ],
            "physical_features": [
                "formation_exact_specificity",
                "transport_exact_specificity",
                "competition_specificity",
            ],
            "physical_feature_count_max": 3,
            "ridge_alpha": 1.0,
            "teacher_and_natural_accounts_are_separate": True,
            "sealed_split_used_for_selection": False,
        },
        "evidence_contract": {
            "instrument_groups_update_thresholds": False,
            "matched_negative_controls_required": True,
            "candidate_must_exceed_control": True,
            "attention_source_write_is_legal_compute_quantity": True,
            "legal_compute_quantity_is_not_causal_edge": True,
            "strict_human_double_blind": False,
            "sealed_read_allowed_before_open_gate": False,
            "causal_claim_allowed_before_sealed_pass": False,
            "full_neuron_scan_allowed": False,
            "single_aggregate_block_test_only_after_sealed_pass": True,
        },
        "stop_rules": [
            "candidate_control_role_covariance_not_positive_closes_function_specific_role",
            "exact_position_transport_failure_closes_role_transport_candidate",
            "formation_to_transport_identity_failure_prevents_complete_path",
            "either_teacher_or_natural_event_prediction_failure_keeps_sealed_closed",
            "one_model_only_is_model_specific",
            "no_threshold_layer_feature_or_sample_rescue_on_this_denominator",
        ],
        "implementation_commitments": {
            path.name: digest_file(path) for path in implementations
        },
        "sealed_commitment": sealed_commitment,
        "validation": validation,
    }
    write_json(protocol_path, protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-frozen", action="store_true")
    parser.parse_args()
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
