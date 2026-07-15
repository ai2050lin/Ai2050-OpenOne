#!/usr/bin/env python3
"""Freeze the independent Phase427 dual-route behavior denominator."""

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


PHASE_ID = "Phase427-DualRouteBehaviorProtocol"
SCHEMA_VERSION = "phase427_dual_route_behavior.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
OUT = ROOT / "tests/gpt5/result/phase427_dual_route_behavior"
BLOCKS = (
    {
        "block_id": "language_action_dual_route_candidate",
        "family_id": "language_action",
        "mechanism_id": "dual_route_lookup",
        "candidate": True,
        "matched_control_block_id": "language_action_stable_result_control",
    },
    {
        "block_id": "language_action_stable_result_control",
        "family_id": "language_action",
        "mechanism_id": "stable_result_control",
        "candidate": False,
        "matched_control_block_id": "language_action_dual_route_candidate",
    },
    {
        "block_id": "syntax_agent_patient_dual_route_candidate",
        "family_id": "syntax_structure",
        "mechanism_id": "agent_patient_dual_route",
        "candidate": True,
        "matched_control_block_id": "syntax_marked_anchor_control",
    },
    {
        "block_id": "syntax_marked_anchor_control",
        "family_id": "syntax_structure",
        "mechanism_id": "marked_anchor_control",
        "candidate": False,
        "matched_control_block_id": "syntax_agent_patient_dual_route_candidate",
    },
)
SPLITS = (
    "behavior_discovery",
    "behavior_calibration",
    "behavior_holdout",
    "sealed_behavior_holdout",
)
OPEN_SPLITS = SPLITS[:3]
INTERFACES = ("direct", "result_field")
HISTORIES = ("bare", "balanced_examples")
ROLES = ("a", "b")
ROUTE_MODES = ("none", "source_only", "query_only", "consistent", "conflict")
SCORABLE_CANDIDATE_ROUTES = ("source_only", "query_only", "consistent")
GROUPS_PER_BLOCK_SPLIT = 64
INSTRUMENT_GROUPS_PER_BLOCK = 2
ACTIVE_TAG = {"a": "Routing tag: rolea.", "b": "Routing tag: roleb."}
NEUTRAL_TAG = "Routing tag: rolen."
VERBS = ("praised", "helped", "thanked", "called", "visited", "greeted")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_chat(tokenizer: Any, model: str, content: str) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], **kwargs
    )
    if model == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return rendered


def interface_history(group_index: int) -> tuple[str, str]:
    combinations = (
        ("direct", "bare"),
        ("direct", "balanced_examples"),
        ("result_field", "bare"),
        ("result_field", "balanced_examples"),
    )
    return combinations[group_index % len(combinations)]


def route_tags(route_mode: str, role: str) -> tuple[str, str, str, str]:
    active = ACTIVE_TAG[role]
    opposite_role = "b" if role == "a" else "a"
    opposite = ACTIVE_TAG[opposite_role]
    if route_mode == "none":
        return NEUTRAL_TAG, NEUTRAL_TAG, "none", "none"
    if route_mode == "source_only":
        return active, NEUTRAL_TAG, role, "none"
    if route_mode == "query_only":
        return NEUTRAL_TAG, active, "none", role
    if route_mode == "consistent":
        return active, active, role, role
    if route_mode == "conflict":
        return active, opposite, role, opposite_role
    raise ValueError(route_mode)


def history_text(family_id: str, candidate: bool, history: str) -> str:
    if history == "bare":
        return ""
    if family_id == "language_action" and candidate:
        return (
            "Balanced examples: Before rolea, record Role A result A900001 and "
            "Role B result B900001, after rolen, gives A900001. Before rolen, "
            "the same record, after roleb, gives B900001. Examples end."
        )
    if family_id == "language_action":
        return (
            "Balanced examples: A stable-result record with result A900002 and "
            "decoy B900002 gives A900002 under either rolea or roleb. Examples end."
        )
    if candidate:
        return (
            "Balanced examples: In 'Nora901 praised Liam901', before rolea and "
            "after rolen gives Nora901. Before rolen and after roleb gives Liam901. "
            "Examples end."
        )
    return (
        "Balanced examples: In a marked-anchor record, either rolea or roleb returns "
        "the marked person and not the distractor. Examples end."
    )


def response_text(interface: str, target: str) -> str:
    if interface == "direct":
        return target
    return json.dumps({"result": target}, ensure_ascii=True, separators=(",", ":"))


def build_group(block: dict[str, Any], split: str, group_index: int) -> dict[str, Any]:
    block_index = next(
        index for index, value in enumerate(BLOCKS) if value["block_id"] == block["block_id"]
    )
    split_index = SPLITS.index(split) if split in SPLITS else 9
    serial = block_index * 100000 + split_index * 10000 + group_index
    suffix = f"{serial:06d}"
    interface, history = interface_history(group_index)
    first = f"A{suffix}"
    second = f"B{suffix}"
    swap = group_index % 2 == 1
    if block["family_id"] == "language_action":
        source = f"S{suffix}"
        if block["candidate"]:
            role_targets = {
                "a": second if swap else first,
                "b": first if swap else second,
            }
            record = (
                f"Record: source {source}; Role A result {role_targets['a']}; "
                f"Role B result {role_targets['b']}."
            )
            task = (
                "Task: rolea selects Role A result; roleb selects Role B result; "
                "rolen carries no role signal."
            )
            stable_target = None
            decoy = None
        else:
            stable_target = second if swap else first
            decoy = first if swap else second
            role_targets = {"a": stable_target, "b": stable_target}
            record = (
                f"Record: source {source}; Stable result {stable_target}; Decoy {decoy}."
            )
            task = (
                "Task: rolea and roleb both select Stable result; rolen carries no "
                "role signal. Never select Decoy."
            )
    else:
        left = f"Nora{suffix}"
        right = f"Liam{suffix}"
        agent, patient = (right, left) if swap else (left, right)
        verb = VERBS[group_index % len(VERBS)]
        source = f"{agent} {verb} {patient}"
        if block["candidate"]:
            role_targets = {"a": agent, "b": patient}
            record = f"Sentence record: {source}."
            task = (
                "Task: rolea selects the agent before the verb; roleb selects the "
                "patient after the verb; rolen carries no role signal."
            )
            stable_target = None
            decoy = None
        else:
            stable_target = patient if swap else agent
            decoy = agent if swap else patient
            role_targets = {"a": stable_target, "b": stable_target}
            record = (
                f"Sentence record: {source}. Marked person {stable_target}; "
                f"Distractor {decoy}."
            )
            task = (
                "Task: rolea and roleb both select Marked person; rolen carries no "
                "role signal. Never select Distractor."
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        **block,
        "split": split,
        "pipeline_sealed": split == "sealed_behavior_holdout",
        "instrument": split == "instrument",
        "group_index": group_index,
        "semantic_group_id": (
            f"phase427__{block['block_id']}__{split}__group_{group_index:03d}"
        ),
        "interface": interface,
        "history": history,
        "source": source,
        "record": record,
        "task": task,
        "history_text": history_text(block["family_id"], block["candidate"], history),
        "role_targets": role_targets,
        "stable_target": stable_target,
        "decoy": decoy,
    }


def group_rows(instrument: bool) -> list[dict[str, Any]]:
    rows = []
    split_specs = (
        (("instrument", INSTRUMENT_GROUPS_PER_BLOCK),)
        if instrument
        else tuple((split, GROUPS_PER_BLOCK_SPLIT) for split in SPLITS)
    )
    for block in BLOCKS:
        for split, count in split_specs:
            rows.extend(build_group(block, split, index) for index in range(count))
    return rows


def content_for(group: dict[str, Any], role: str, route_mode: str) -> dict[str, Any]:
    before, after, source_role, query_role = route_tags(route_mode, role)
    role_targets = group["role_targets"]
    if group["candidate"]:
        source_target = role_targets[role]
        query_target = role_targets["b" if role == "a" else "a"] if route_mode == "conflict" else role_targets[role]
        target = source_target
        opposite = query_target if route_mode == "conflict" else role_targets["b" if role == "a" else "a"]
        normative = route_mode in SCORABLE_CANDIDATE_ROUTES
    else:
        source_target = group["stable_target"]
        query_target = group["stable_target"]
        target = group["stable_target"]
        opposite = group["decoy"]
        normative = True
    if group["interface"] == "direct":
        output_contract = "Output exactly the selected code and then stop."
    else:
        output_contract = (
            'Output exactly {"result":"SELECTED_CODE"} with SELECTED_CODE replaced '
            "by the selected code, and then stop."
        )
    before_line = f"Before-source routing slot: {before}"
    after_line = f"After-source routing slot: {after}"
    query_line = "Question: Which result or person is selected?"
    content = "\n".join(
        part
        for part in (
            "Follow this deterministic routing task.",
            group["task"],
            group["history_text"],
            before_line,
            group["record"],
            after_line,
            query_line,
            output_contract,
        )
        if part
    )
    return {
        "content": content,
        "before_tag": before,
        "after_tag": after,
        "before_line": before_line,
        "after_line": after_line,
        "query_line": query_line,
        "source_role": source_role,
        "query_role": query_role,
        "source_route_target": source_target,
        "query_route_target": query_target,
        "target": target,
        "opposite": opposite,
        "normative_target": normative,
        "target_text": response_text(group["interface"], target),
        "opposite_text": response_text(group["interface"], opposite),
    }


def token_positions(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]


def locate_span(rendered: str, line: str, value: str, use_last: bool = False) -> tuple[int, int]:
    line_start = rendered.rfind(line) if use_last else rendered.find(line)
    if line_start < 0:
        raise RuntimeError(f"Line not found in rendered prompt: {line}")
    value_start = rendered.find(value, line_start, line_start + len(line))
    if value_start < 0:
        raise RuntimeError(f"Value not found in rendered prompt line: {value}")
    return value_start, value_start + len(value)


def register_condition(
    group: dict[str, Any], role: str, route_mode: str, model: str, tokenizer: Any
) -> dict[str, Any]:
    payload = content_for(group, role, route_mode)
    rendered = render_chat(tokenizer, model, payload["content"])
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    source_start, source_end = locate_span(rendered, group["record"], group["source"])
    query_start, query_end = locate_span(
        rendered, payload["query_line"], payload["query_line"], use_last=True
    )
    before_start, before_end = locate_span(
        rendered, payload["before_line"], payload["before_tag"], use_last=True
    )
    after_start, after_end = locate_span(
        rendered, payload["after_line"], payload["after_tag"], use_last=True
    )
    target_ids = [
        int(value)
        for value in tokenizer(payload["target_text"], add_special_tokens=False)[
            "input_ids"
        ]
    ]
    opposite_ids = [
        int(value)
        for value in tokenizer(payload["opposite_text"], add_special_tokens=False)[
            "input_ids"
        ]
    ]
    condition_key = f"r{role}__route_{route_mode}"
    condition_id = f"{group['semantic_group_id']}__{condition_key}__{model}"
    prompt_id_bytes = ",".join(str(value) for value in prompt_ids).encode("ascii")
    return {
        **{key: value for key, value in group.items() if key not in {"role_targets", "history_text", "task", "record"}},
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": model,
        "role": role,
        "route_mode": route_mode,
        "condition_key": condition_key,
        "condition_id": condition_id,
        "content_prompt": payload["content"],
        "rendered_prompt": rendered,
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": hashlib.sha256(prompt_id_bytes).hexdigest(),
        "prompt_token_count": len(prompt_ids),
        "source_positions": token_positions(offsets, source_start, source_end),
        "query_positions": token_positions(offsets, query_start, query_end),
        "before_tag_positions": token_positions(offsets, before_start, before_end),
        "after_tag_positions": token_positions(offsets, after_start, after_end),
        "source_role": payload["source_role"],
        "query_role": payload["query_role"],
        "source_route_target": payload["source_route_target"],
        "query_route_target": payload["query_route_target"],
        "target": payload["target"],
        "opposite_target": payload["opposite"],
        "target_text": payload["target_text"],
        "opposite_text": payload["opposite_text"],
        "target_sequence_token_ids": target_ids,
        "opposite_sequence_token_ids": opposite_ids,
        "normative_target": payload["normative_target"],
        "descriptive_conflict_only": bool(group["candidate"] and route_mode == "conflict"),
        "descriptive_none_only": bool(group["candidate"] and route_mode == "none"),
        "natural_generation_max_new_tokens": 24 if group["interface"] == "direct" else 32,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
    }


def register(groups: list[dict[str, Any]], tokenizers: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        register_condition(group, role, route_mode, model, tokenizers[model])
        for group in groups
        for model in MODELS
        for role in ROLES
        for route_mode in ROUTE_MODES
    ]


def route_position_mismatch_count(rows: list[dict[str, Any]]) -> int:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["semantic_group_id"], row["model"], row["role"])].append(row)
    mismatches = 0
    for values in grouped.values():
        if len(values) != len(ROUTE_MODES):
            mismatches += 1
            continue
        reference = values[0]
        invariant = all(
            row["source_positions"] == reference["source_positions"]
            and row["query_positions"] == reference["query_positions"]
            and row["before_tag_positions"] == reference["before_tag_positions"]
            and row["after_tag_positions"] == reference["after_tag_positions"]
            and row["prompt_token_count"] == reference["prompt_token_count"]
            for row in values[1:]
        )
        mismatches += int(not invariant)
    return mismatches


def validate(
    formal_groups: list[dict[str, Any]],
    instrument_groups: list[dict[str, Any]],
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    instrument_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    formal_rows = [*open_rows, *sealed_rows]
    all_rows = [*formal_rows, *instrument_rows]
    group_splits: dict[str, set[str]] = defaultdict(set)
    for group in [*formal_groups, *instrument_groups]:
        group_splits[group["semantic_group_id"]].add(group["split"])
    split_counts = Counter((row["block_id"], row["split"]) for row in formal_groups)
    model_counts = Counter(row["model"] for row in formal_rows)
    interface_history_counts = Counter(
        (row["block_id"], row["split"], row["interface"], row["history"])
        for row in formal_groups
    )
    expected_formal_groups = len(BLOCKS) * len(SPLITS) * GROUPS_PER_BLOCK_SPLIT
    expected_open = len(BLOCKS) * len(OPEN_SPLITS) * GROUPS_PER_BLOCK_SPLIT * len(MODELS) * len(ROLES) * len(ROUTE_MODES)
    expected_sealed = len(BLOCKS) * GROUPS_PER_BLOCK_SPLIT * len(MODELS) * len(ROLES) * len(ROUTE_MODES)
    expected_instrument = len(BLOCKS) * INSTRUMENT_GROUPS_PER_BLOCK * len(MODELS) * len(ROLES) * len(ROUTE_MODES)
    valid = bool(
        len(formal_groups) == expected_formal_groups
        and len(instrument_groups) == len(BLOCKS) * INSTRUMENT_GROUPS_PER_BLOCK
        and len(open_rows) == expected_open
        and len(sealed_rows) == expected_sealed
        and len(instrument_rows) == expected_instrument
        and len({row["condition_id"] for row in all_rows}) == len(all_rows)
        and all(value == GROUPS_PER_BLOCK_SPLIT for value in split_counts.values())
        and all(value == 16 for value in interface_history_counts.values())
        and all(value == expected_formal_groups * len(ROLES) * len(ROUTE_MODES) for value in model_counts.values())
        and all(len(value) == 1 for value in group_splits.values())
        and route_position_mismatch_count(all_rows) == 0
        and all(row["source_positions"] and row["query_positions"] for row in all_rows)
        and all(row["target_sequence_token_ids"] and row["opposite_sequence_token_ids"] for row in all_rows)
        and all(len(row["target_sequence_token_ids"]) == len(row["opposite_sequence_token_ids"]) for row in all_rows)
        and all(
            row["normative_target"]
            == (not row["candidate"] or row["route_mode"] in SCORABLE_CANDIDATE_ROUTES)
            for row in all_rows
        )
    )
    return {
        "valid": valid,
        "formal_group_count": len(formal_groups),
        "instrument_group_count": len(instrument_groups),
        "formal_condition_count": len(formal_rows),
        "open_condition_count": len(open_rows),
        "sealed_condition_count": len(sealed_rows),
        "instrument_condition_count": len(instrument_rows),
        "conditions_per_group_per_model": len(ROLES) * len(ROUTE_MODES),
        "route_position_mismatch_count": route_position_mismatch_count(all_rows),
        "group_split_leak_count": sum(len(value) != 1 for value in group_splits.values()),
        "formal_condition_counts_by_model": dict(sorted(model_counts.items())),
        "formal_group_counts_by_block_split": {
            f"{block}::{split}": value
            for (block, split), value in sorted(split_counts.items())
        },
        "interface_history_group_counts": {
            f"{block}::{split}::{interface}::{history}": value
            for (block, split, interface, history), value in sorted(interface_history_counts.items())
        },
    }


def freeze() -> dict[str, Any]:
    protocol_path = OUT / "phase427_protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol.get("schema_version") == SCHEMA_VERSION and protocol.get("validation", {}).get("valid"):
            return protocol
    implementations = (
        ROOT / "tests/gpt5/phase427_dual_route_protocol.py",
        ROOT / "tests/gpt5/phase427_behavior_collect.py",
        ROOT / "tests/gpt5/phase427_behavior_analysis.py",
    )
    missing = [str(path) for path in implementations if not path.exists()]
    if missing:
        raise RuntimeError(f"Create Phase427 implementations before freeze: {missing}")
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    formal_groups = group_rows(instrument=False)
    instrument_groups = group_rows(instrument=True)
    formal_rows = register(formal_groups, tokenizers)
    instrument_rows = register(instrument_groups, tokenizers)
    open_rows = [row for row in formal_rows if not row["pipeline_sealed"]]
    sealed_rows = [row for row in formal_rows if row["pipeline_sealed"]]
    validation = validate(
        formal_groups, instrument_groups, open_rows, sealed_rows, instrument_rows
    )
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    write_jsonl(OUT / "phase427_registered_groups.jsonl", formal_groups)
    write_jsonl(OUT / "phase427_instrument_groups.jsonl", instrument_groups)
    write_jsonl(OUT / "phase427_registered_conditions_open.jsonl", open_rows)
    write_jsonl(OUT / "phase427_instrument_conditions.jsonl", instrument_rows)
    sealed_path = OUT / "sealed" / "phase427_registered_conditions_sealed.jsonl"
    write_jsonl(sealed_path, sealed_rows)
    sealed_commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": "sealed_behavior_holdout",
        "sealed_condition_count": len(sealed_rows),
        "sealed_condition_rows_sha256": digest_rows(sealed_rows),
        "sealed_prompt_hashes_sha256": hashlib.sha256(
            "\n".join(sorted(row["prompt_sha256"] for row in sealed_rows)).encode("utf-8")
        ).hexdigest(),
        "pipeline_sealed_until_open_behavior_gate": True,
        "strict_human_double_blind": False,
    }
    write_json(OUT / "phase427_sealed_commitment.json", sealed_commitment)
    thresholds = {
        "teacher_sequence_correct_fraction_min": 0.80,
        "teacher_sequence_margin_median_min": 0.0,
        "natural_target_first_fraction_min": 0.70,
        "natural_opposite_first_fraction_max": 0.10,
        "natural_revision_fraction_max": 0.05,
        "natural_boundary_fraction_min": 0.70,
        "natural_stop_fraction_min": 0.70,
        "natural_censoring_fraction_max": 0.25,
        "cross_model_replication_min": 2,
        "groups_per_block_split": GROUPS_PER_BLOCK_SPLIT,
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "objective": (
            "Qualify complete behavior before physical hooks and separate source "
            "prebinding from query-side gating on an independent denominator."
        ),
        "models": list(MODELS),
        "blocks": list(BLOCKS),
        "splits": list(SPLITS),
        "interfaces": list(INTERFACES),
        "histories": list(HISTORIES),
        "roles": list(ROLES),
        "route_modes": list(ROUTE_MODES),
        "scorable_candidate_routes": list(SCORABLE_CANDIDATE_ROUTES),
        "execution_dtype_by_model": {
            "qwen3": "float16",
            "glm4": "bfloat16",
            "deepseek7b": "bfloat16",
        },
        "split_contract": {
            "groups_per_block_per_split": GROUPS_PER_BLOCK_SPLIT,
            "conditions_per_group_per_model": len(ROLES) * len(ROUTE_MODES),
            "formal_independent_group_count": len(formal_groups),
            "open_condition_count": len(open_rows),
            "sealed_condition_count": len(sealed_rows),
            "instrument_condition_count": len(instrument_rows),
            "interface_history_balanced_between_groups": True,
            "split_unit": "semantic_group",
        },
        "route_contract": {
            "none": "neutral before source and neutral before query; descriptive only for candidates",
            "source_only": "active role before source and neutral after source",
            "query_only": "neutral before source and active role after source",
            "consistent": "same active role in both slots",
            "conflict": "opposite active roles in the two slots; descriptive route preference only",
            "conflict_has_normative_candidate_target": False,
            "same_source_query_and_slot_positions_across_routes": True,
        },
        "behavior_contract": {
            "chat_interface": "concise chat; Qwen thinking disabled; DeepSeek empty thinking prefill",
            "teacher_target": "full exact response versus full opposite response",
            "natural_generation_tokens": {"direct": 24, "result_field": 32},
            "teacher_and_natural_accounts_separate": True,
            "right_censoring_explicit": True,
            "gate_splits": ["behavior_calibration", "behavior_holdout"],
            "block_qualification": (
                "candidate and matched control both pass for the consistent route "
                "and at least one single route on both gate splits"
            ),
            "independent_unit": "semantic_group; two role conditions are paired measurements",
        },
        "registered_thresholds": thresholds,
        "future_physical_contract": {
            "physical_hooks_before_behavior_qualification": False,
            "independent_physical_denominator_required": True,
            "exact_finite_difference_attention_decomposition_primary": True,
            "full_hidden_rank_r_map_prohibited": True,
            "learned_map_only_in_preregistered_low_dimensional_basis": True,
            "maximum_summary_rank": 3,
            "sealed_physical_and_causal_stages_not_registered_yet": True,
        },
        "evidence_contract": {
            "instrument_updates_thresholds": False,
            "candidate_conflict_and_none_are_descriptive": True,
            "pipeline_sealed": True,
            "strict_human_double_blind": False,
            "causal_claim": False,
            "head_channel_neuron_scan_allowed": False,
        },
        "stop_rules": [
            "open_behavior_failure_keeps_sealed_behavior_closed",
            "sealed_behavior_failure_prevents_physical_registration",
            "one_model_only_is_model_specific",
            "conflict_preference_never_counts_as_behavior_correct",
            "no_threshold_prompt_sample_or_window_rescue_on_this_denominator",
            "no_physical_hook_until_behavior_gate_passes",
        ],
        "implementation_commitments": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in implementations
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
