#!/usr/bin/env python3
"""Freeze Phase429 observer qualification and typed dual-route denominators."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


PHASE_ID = "Phase429-TypedRouteProtocol"
SCHEMA_VERSION = "phase429_typed_route.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
OUT = ROOT / "tests/gpt5/result/phase429_typed_route"
INTERFACES = ("direct_item", "short_code", "result_field", "forced_choice")
INTERFACE_SIMPLICITY = INTERFACES
OBSERVER_BLOCKS = (
    {
        "block_id": "observer_semantic_lookup",
        "family_id": "knowledge_semantics",
        "mechanism_id": "observer_interface_qualification",
    },
    {
        "block_id": "observer_syntax_readout",
        "family_id": "syntax_structure",
        "mechanism_id": "observer_interface_qualification",
    },
)
BEHAVIOR_BLOCKS = (
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
OBSERVER_SPLITS = ("interface_calibration", "interface_holdout")
BEHAVIOR_SPLITS = (
    "behavior_calibration",
    "behavior_holdout",
    "sealed_physical_holdout",
)
OPEN_BEHAVIOR_SPLITS = BEHAVIOR_SPLITS[:2]
CONTRACT_VARIANTS = ("no_examples", "fully_crossed_examples")
ROLES = ("a", "b")
ROUTE_MODES = ("none", "source_only", "query_only", "consistent", "conflict")
SCORABLE_ROUTES = ("source_only", "query_only", "consistent")
GROUPS_PER_BLOCK_SPLIT = 96
INSTRUMENT_GROUPS_PER_BLOCK = 2
ACTIVE_TAG = {"a": "selector-alpha", "b": "selector-beta"}
NEUTRAL_TAG = "selector-neutral"
VERBS = ("praised", "helped", "thanked", "called", "visited", "greeted")


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


def interface_payload(
    interface: str, first: str, second: str, semantic_target: str
) -> dict[str, str | int]:
    if semantic_target not in {first, second}:
        raise ValueError(f"Target {semantic_target!r} is not a registered option")
    if interface == "direct_item":
        return {
            "contract": "Output exactly the selected item and then stop.",
            "target": semantic_target,
            "opposite": second if semantic_target == first else first,
            "max_new_tokens": 12,
        }
    if interface == "short_code":
        target = "alpha" if semantic_target == first else "beta"
        return {
            "contract": (
                f"Code alpha means {first}; code beta means {second}. "
                "Output exactly alpha or beta and then stop."
            ),
            "target": target,
            "opposite": "beta" if target == "alpha" else "alpha",
            "max_new_tokens": 8,
        }
    if interface == "result_field":
        target = json.dumps(
            {"result": semantic_target}, ensure_ascii=True, separators=(",", ":")
        )
        opposite_item = second if semantic_target == first else first
        return {
            "contract": (
                'Output exactly {"result":"SELECTED_ITEM"}, replacing '
                "SELECTED_ITEM with the selected item, and then stop."
            ),
            "target": target,
            "opposite": json.dumps(
                {"result": opposite_item},
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "max_new_tokens": 20,
        }
    if interface == "forced_choice":
        target = "option1" if semantic_target == first else "option2"
        return {
            "contract": (
                f"Option option1 means {first}; option option2 means {second}. "
                "Output exactly option1 or option2 and then stop."
            ),
            "target": target,
            "opposite": "option2" if target == "option1" else "option1",
            "max_new_tokens": 8,
        }
    raise ValueError(interface)


def observer_group(block: dict[str, str], split: str, index: int) -> dict[str, Any]:
    block_index = next(i for i, item in enumerate(OBSERVER_BLOCKS) if item == block)
    split_index = OBSERVER_SPLITS.index(split) if split in OBSERVER_SPLITS else 9
    serial = block_index * 100000 + split_index * 10000 + index
    suffix = f"{serial:06d}"
    first, second = f"X{suffix}", f"Y{suffix}"
    target_first = index % 2 == 0
    target = first if target_first else second
    if block["block_id"] == "observer_semantic_lookup":
        item_a, item_b = f"object-C{suffix}", f"object-D{suffix}"
        query_item = item_a if target_first else item_b
        record = (
            f"Reference: {item_a} has label {first}; {item_b} has label {second}."
        )
        task = f"Question: What label belongs to {query_item}?"
    else:
        verb = VERBS[index % len(VERBS)]
        sentence = f"{first} {verb} {second}"
        role = "actor" if target_first else "receiver"
        record = f"Sentence: {sentence}."
        task = f"Question: Which item is the {role} in the sentence?"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        **block,
        "split": split,
        "instrument": split == "instrument_observer",
        "pipeline_sealed": False,
        "group_index": index,
        "semantic_group_id": (
            f"phase429__{block['block_id']}__{split}__group_{index:03d}"
        ),
        "first_item": first,
        "second_item": second,
        "semantic_target": target,
        "semantic_opposite": second if target == first else first,
        "record": record,
        "task": task,
    }


def register_observer_condition(
    group: dict[str, Any], interface: str, model: str, tokenizer: Any
) -> dict[str, Any]:
    output = interface_payload(
        interface, group["first_item"], group["second_item"], group["semantic_target"]
    )
    content = "\n".join(
        (
            "Solve this deterministic readout task.",
            group["record"],
            group["task"],
            str(output["contract"]),
        )
    )
    rendered = render_chat(tokenizer, model, content)
    prompt_ids = [
        int(value)
        for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]
    target_ids = [
        int(value)
        for value in tokenizer(str(output["target"]), add_special_tokens=False)[
            "input_ids"
        ]
    ]
    opposite_ids = [
        int(value)
        for value in tokenizer(str(output["opposite"]), add_special_tokens=False)[
            "input_ids"
        ]
    ]
    condition_id = f"{group['semantic_group_id']}__{interface}__{model}"
    return {
        **{key: value for key, value in group.items() if key not in {"record", "task"}},
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "stage_kind": "observer_interface",
        "model": model,
        "condition_id": condition_id,
        "interface": interface,
        "contract_variant": "observer_no_examples",
        "role": "observer",
        "route_mode": "observer",
        "content_prompt": content,
        "rendered_prompt": rendered,
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": hashlib.sha256(
            ",".join(str(value) for value in prompt_ids).encode("ascii")
        ).hexdigest(),
        "prompt_token_count": len(prompt_ids),
        "target": str(output["target"]),
        "opposite_target": str(output["opposite"]),
        "semantic_target": group["semantic_target"],
        "semantic_opposite": group["semantic_opposite"],
        "target_sequence_token_ids": target_ids,
        "opposite_sequence_token_ids": opposite_ids,
        "natural_generation_max_new_tokens": int(output["max_new_tokens"]),
        "normative_target": True,
        "candidate": False,
        "matched_control_block_id": None,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
    }


def observer_groups(instrument: bool) -> list[dict[str, Any]]:
    rows = []
    splits = (
        (("instrument_observer", INSTRUMENT_GROUPS_PER_BLOCK),)
        if instrument
        else tuple((split, GROUPS_PER_BLOCK_SPLIT) for split in OBSERVER_SPLITS)
    )
    for block in OBSERVER_BLOCKS:
        for split, count in splits:
            rows.extend(observer_group(block, split, index) for index in range(count))
    return rows


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


def crossed_history(family_id: str, candidate: bool, index: int) -> dict[str, Any]:
    cells = [("a", "source"), ("a", "query"), ("b", "source"), ("b", "query")]
    rotation = index % len(cells)
    cells = cells[rotation:] + cells[:rotation]
    demos = []
    for demo_index, (role, route) in enumerate(cells):
        serial = 900000 + (index % 96) * 10 + demo_index
        first, second = f"X{serial:06d}", f"Y{serial:06d}"
        active = ACTIVE_TAG[role]
        before = active if route == "source" else NEUTRAL_TAG
        after = NEUTRAL_TAG if route == "source" else active
        if family_id == "language_action":
            if candidate:
                selected = first if role == "a" else second
                record = f"role-A result {first}; role-B result {second}"
            else:
                selected = first if demo_index % 2 == 0 else second
                decoy = second if selected == first else first
                record = f"stable result {selected}; decoy {decoy}"
        else:
            verb = VERBS[demo_index % len(VERBS)]
            record = f"sentence {first} {verb} {second}"
            if candidate:
                selected = first if role == "a" else second
            else:
                selected = first if demo_index % 2 == 0 else second
                record += f"; marked item {selected}"
        demos.append(
            f"Demo {demo_index + 1}: before {before}; {record}; after {after}; "
            f"answer {selected}."
        )
    return {
        "history_text": "Fully crossed demonstrations:\n" + "\n".join(demos),
        "demonstration_cells": [f"{role}:{route}" for role, route in cells],
    }


def behavior_group(
    block: dict[str, Any], split: str, contract_variant: str, index: int
) -> dict[str, Any]:
    block_index = next(i for i, item in enumerate(BEHAVIOR_BLOCKS) if item == block)
    split_index = BEHAVIOR_SPLITS.index(split) if split in BEHAVIOR_SPLITS else 9
    contract_index = CONTRACT_VARIANTS.index(contract_variant)
    serial = block_index * 200000 + split_index * 20000 + contract_index * 10000 + index
    suffix = f"{serial:06d}"
    first, second = f"X{suffix}", f"Y{suffix}"
    swap = index % 2 == 1
    if block["family_id"] == "language_action":
        source = f"source-Z{suffix}"
        if block["candidate"]:
            role_targets = {"a": second if swap else first, "b": first if swap else second}
            record = (
                f"Record: source {source}; role-A result {role_targets['a']}; "
                f"role-B result {role_targets['b']}."
            )
            task = (
                "Selector-alpha selects role-A result; selector-beta selects role-B "
                "result; selector-neutral carries no selector signal."
            )
            stable_target = None
            decoy = None
        else:
            stable_target = second if swap else first
            decoy = first if swap else second
            role_targets = {"a": stable_target, "b": stable_target}
            record = (
                f"Record: source {source}; stable result {stable_target}; decoy {decoy}."
            )
            task = (
                "Both selector-alpha and selector-beta select stable result; "
                "selector-neutral also leaves stable result unchanged. Never select decoy."
            )
    else:
        verb = VERBS[index % len(VERBS)]
        agent, patient = (second, first) if swap else (first, second)
        source = f"{agent} {verb} {patient}"
        if block["candidate"]:
            role_targets = {"a": agent, "b": patient}
            record = f"Sentence record: {source}."
            task = (
                "Selector-alpha selects the actor before the verb; selector-beta "
                "selects the receiver after the verb; selector-neutral carries no selector signal."
            )
            stable_target = None
            decoy = None
        else:
            stable_target = agent if (index // 2) % 2 == 0 else patient
            decoy = patient if stable_target == agent else agent
            role_targets = {"a": stable_target, "b": stable_target}
            record = (
                f"Sentence record: {source}. Marked item {stable_target}; distractor {decoy}."
            )
            task = (
                "Both selector-alpha and selector-beta select marked item; "
                "selector-neutral also leaves marked item unchanged. Never select distractor."
            )
    history = (
        {"history_text": "", "demonstration_cells": []}
        if contract_variant == "no_examples"
        else crossed_history(block["family_id"], bool(block["candidate"]), index)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        **block,
        "split": split,
        "instrument": split == "instrument_behavior",
        "pipeline_sealed": split == "sealed_physical_holdout",
        "contract_variant": contract_variant,
        "group_index": index,
        "semantic_group_id": (
            f"phase429__{block['block_id']}__{contract_variant}__{split}__group_{index:03d}"
        ),
        "source": source,
        "first_item": first,
        "second_item": second,
        "record": record,
        "task": task,
        "role_targets": role_targets,
        "stable_target": stable_target,
        "decoy": decoy,
        **history,
    }


def behavior_groups(instrument: bool) -> list[dict[str, Any]]:
    rows = []
    splits = (
        (("instrument_behavior", INSTRUMENT_GROUPS_PER_BLOCK),)
        if instrument
        else tuple((split, GROUPS_PER_BLOCK_SPLIT) for split in BEHAVIOR_SPLITS)
    )
    for block in BEHAVIOR_BLOCKS:
        for contract in CONTRACT_VARIANTS:
            for split, count in splits:
                rows.extend(behavior_group(block, split, contract, index) for index in range(count))
    return rows


def validate_groups(
    observer_formal: list[dict[str, Any]],
    observer_instrument: list[dict[str, Any]],
    behavior_formal: list[dict[str, Any]],
    behavior_instrument: list[dict[str, Any]],
) -> dict[str, Any]:
    observer_counts = Counter((row["block_id"], row["split"]) for row in observer_formal)
    behavior_counts = Counter(
        (row["block_id"], row["contract_variant"], row["split"])
        for row in behavior_formal
    )
    all_groups = [*observer_formal, *observer_instrument, *behavior_formal, *behavior_instrument]
    crossed = [row for row in behavior_formal if row["contract_variant"] == "fully_crossed_examples"]
    bare = [row for row in behavior_formal if row["contract_variant"] == "no_examples"]
    expected_cells = {"a:source", "a:query", "b:source", "b:query"}
    valid = bool(
        len(observer_formal) == len(OBSERVER_BLOCKS) * len(OBSERVER_SPLITS) * GROUPS_PER_BLOCK_SPLIT
        and len(observer_instrument) == len(OBSERVER_BLOCKS) * INSTRUMENT_GROUPS_PER_BLOCK
        and len(behavior_formal) == len(BEHAVIOR_BLOCKS) * len(CONTRACT_VARIANTS) * len(BEHAVIOR_SPLITS) * GROUPS_PER_BLOCK_SPLIT
        and len(behavior_instrument) == len(BEHAVIOR_BLOCKS) * len(CONTRACT_VARIANTS) * INSTRUMENT_GROUPS_PER_BLOCK
        and len({row["semantic_group_id"] for row in all_groups}) == len(all_groups)
        and all(value == GROUPS_PER_BLOCK_SPLIT for value in observer_counts.values())
        and all(value == GROUPS_PER_BLOCK_SPLIT for value in behavior_counts.values())
        and all(set(row["demonstration_cells"]) == expected_cells for row in crossed)
        and all(not row["demonstration_cells"] for row in bare)
        and all(
            row["first_item"] != row["second_item"]
            and set(row["role_targets"].values()).issubset({row["first_item"], row["second_item"]})
            for row in behavior_formal
        )
    )
    return {
        "valid": valid,
        "observer_formal_group_count": len(observer_formal),
        "observer_instrument_group_count": len(observer_instrument),
        "behavior_formal_group_count": len(behavior_formal),
        "behavior_open_group_count": sum(not row["pipeline_sealed"] for row in behavior_formal),
        "behavior_sealed_group_count": sum(row["pipeline_sealed"] for row in behavior_formal),
        "behavior_instrument_group_count": len(behavior_instrument),
        "groups_per_block_split_contract": GROUPS_PER_BLOCK_SPLIT,
        "observer_group_counts": {
            "::".join(key): value for key, value in sorted(observer_counts.items())
        },
        "behavior_group_counts": {
            "::".join(key): value for key, value in sorted(behavior_counts.items())
        },
        "fully_crossed_cell_count_per_group": 4,
        "role_route_cells": sorted(expected_cells),
    }


def freeze() -> dict[str, Any]:
    protocol_path = OUT / "phase429_protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol.get("schema_version") == SCHEMA_VERSION and protocol.get("validation", {}).get("valid"):
            return protocol
    implementations = (
        ROOT / "tests/gpt5/phase429_typed_route_protocol.py",
        ROOT / "tests/gpt5/phase429_typed_route_collect.py",
        ROOT / "tests/gpt5/phase429_typed_route_analysis.py",
    )
    missing = [str(path) for path in implementations if not path.exists()]
    if missing:
        raise RuntimeError(f"Create Phase429 implementations before freeze: {missing}")
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    observer_formal = observer_groups(instrument=False)
    observer_instrument = observer_groups(instrument=True)
    observer_rows = [
        register_observer_condition(group, interface, model, tokenizers[model])
        for group in observer_formal
        for interface in INTERFACES
        for model in MODELS
    ]
    observer_instrument_rows = [
        register_observer_condition(group, interface, model, tokenizers[model])
        for group in observer_instrument
        for interface in INTERFACES
        for model in MODELS
    ]
    behavior_formal = behavior_groups(instrument=False)
    behavior_instrument = behavior_groups(instrument=True)
    validation = validate_groups(
        observer_formal, observer_instrument, behavior_formal, behavior_instrument
    )
    validation.update(
        {
            "observer_formal_condition_count": len(observer_rows),
            "observer_instrument_condition_count": len(observer_instrument_rows),
            "behavior_open_selected_condition_count_per_model": (
                len(BEHAVIOR_BLOCKS)
                * len(CONTRACT_VARIANTS)
                * len(OPEN_BEHAVIOR_SPLITS)
                * GROUPS_PER_BLOCK_SPLIT
                * len(ROLES)
                * len(ROUTE_MODES)
            ),
            "behavior_sealed_selected_condition_count_per_model": (
                len(BEHAVIOR_BLOCKS)
                * len(CONTRACT_VARIANTS)
                * GROUPS_PER_BLOCK_SPLIT
                * len(ROLES)
                * len(ROUTE_MODES)
            ),
        }
    )
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    write_jsonl(OUT / "phase429_observer_groups.jsonl", observer_formal)
    write_jsonl(OUT / "phase429_observer_conditions.jsonl", observer_rows)
    write_jsonl(OUT / "phase429_observer_instrument_conditions.jsonl", observer_instrument_rows)
    write_jsonl(
        OUT / "phase429_behavior_groups_open.jsonl",
        [row for row in behavior_formal if not row["pipeline_sealed"]],
    )
    sealed_groups = [row for row in behavior_formal if row["pipeline_sealed"]]
    write_jsonl(OUT / "sealed" / "phase429_behavior_groups_sealed.jsonl", sealed_groups)
    write_jsonl(OUT / "phase429_behavior_instrument_groups.jsonl", behavior_instrument)
    sealed_commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_group_count": len(sealed_groups),
        "sealed_group_rows_sha256": digest_rows(sealed_groups),
        "pipeline_sealed_until_open_physical_prediction_gate": True,
        "strict_human_double_blind": False,
    }
    write_json(OUT / "phase429_sealed_commitment.json", sealed_commitment)
    thresholds = {
        "confidence_level": 0.95,
        "confidence_method": "Wilson score on independent semantic-group all/any events",
        "observer": {
            "groups_per_block_split": GROUPS_PER_BLOCK_SPLIT,
            "teacher_all_lcb_min": 0.75,
            "target_first_lcb_min": 0.65,
            "opposite_first_ucb_max": 0.15,
            "event_coverage_lcb_min": 0.80,
            "teacher_margin_median_min": 0.0,
        },
        "content": {
            "groups_per_block_split": GROUPS_PER_BLOCK_SPLIT,
            "teacher_all_lcb_min": 0.70,
            "target_first_all_lcb_min": 0.60,
            "opposite_first_any_ucb_max": 0.20,
            "event_coverage_all_lcb_min": 0.70,
            "teacher_margin_median_min": 0.0,
        },
        "interface": {"valid_all_lcb_min": 0.70},
        "revision": {"revision_any_ucb_max": 0.10},
        "boundary": {"boundary_all_lcb_min": 0.70},
        "termination": {"stop_all_lcb_min": 0.70, "censor_any_ucb_max": 0.25},
        "specificity_effect_min": 0.15,
        "cross_model_replication_min": 2,
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models": list(MODELS),
        "observer_blocks": list(OBSERVER_BLOCKS),
        "behavior_blocks": list(BEHAVIOR_BLOCKS),
        "interfaces": list(INTERFACES),
        "interface_simplicity_order": list(INTERFACE_SIMPLICITY),
        "observer_splits": list(OBSERVER_SPLITS),
        "behavior_splits": list(BEHAVIOR_SPLITS),
        "contract_variants": list(CONTRACT_VARIANTS),
        "roles": list(ROLES),
        "route_modes": list(ROUTE_MODES),
        "scorable_routes": list(SCORABLE_ROUTES),
        "execution_dtype_by_model": {
            "qwen3": "float16",
            "glm4": "bfloat16",
            "deepseek7b": "bfloat16",
        },
        "selection_contract": {
            "selection_split": "interface_calibration",
            "qualification_split": "interface_holdout",
            "selection_primary": "minimum block target-first Wilson lower bound",
            "selection_secondary": "minimum block teacher-correct Wilson lower bound",
            "selection_tertiary": "maximum block opposite-first Wilson upper bound",
            "tie_break": "shortest least-structured interface by frozen order",
            "behavior_holdout_reselection": False,
            "unqualified_model_behavior_collection": False,
        },
        "typed_thresholds": thresholds,
        "physical_contract": {
            "content_gate_can_authorize_without_termination_gate": True,
            "candidate_and_matched_control_required": True,
            "calibration_and_holdout_required": True,
            "architecture_components_only": [
                "residual_state",
                "query_projection",
                "key_projection",
                "value_projection",
                "attention_write",
                "mlp_write",
                "output_projection",
            ],
            "head_channel_neuron_scan_allowed": False,
            "full_hidden_rank_fit_allowed": False,
            "sealed_read_requires_physical_reconstruction_and_prediction": True,
        },
        "evidence_contract": {
            "observer_and_behavior_denominators_independent": True,
            "no_examples_and_fully_crossed_are_independent_denominators": True,
            "candidate_none_and_conflict_descriptive": True,
            "content_interface_revision_boundary_termination_separate": True,
            "strict_human_double_blind": False,
            "causal_claim": False,
        },
        "stop_rules": [
            "no qualified interface closes that model before route collection",
            "no threshold prompt window interface or sample rescue after freeze",
            "candidate content failure prevents physical collection",
            "matched-control content failure prevents specificity claim",
            "one-model-only is model-specific",
            "termination failure does not erase qualified content evidence",
            "sealed groups remain unread until open physical prediction passes",
            "no head channel or neuron scan in Phase429",
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
