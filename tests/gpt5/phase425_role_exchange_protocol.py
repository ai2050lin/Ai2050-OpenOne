#!/usr/bin/env python3
"""Freeze the Phase425 matched-lexical role-exchange denominator."""

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


PHASE_ID = "Phase425-RoleExchangeProtocol"
SCHEMA_VERSION = "phase425_role_exchange.v2"
MODELS = ("qwen3", "glm4", "deepseek7b")
OUT = ROOT / "tests/gpt5/result/phase425_role_exchange_validation"
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
SPLIT_BY_PAIR_INDEX = (
    (0, 24, "discovery"),
    (24, 48, "calibration"),
    (48, 72, "behavior_holdout"),
    (72, 96, "sealed_physical_holdout"),
)
INTERFACES = ("direct", "result_field")
HISTORIES = ("bare", "worked_example")
ROLES = ("a", "b")


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
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suffix(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return alphabet[(index // 26) % 26] + alphabet[index % 26]


def split_for_pair(index: int) -> str:
    for start, end, split in SPLIT_BY_PAIR_INDEX:
        if start <= index < end:
            return split
    raise ValueError(index)


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"Phase425 requires a fast tokenizer for {model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def continuation_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer(" " + text, add_special_tokens=False)["input_ids"]]


def branch_contract(tokenizer: Any, target: str, opposite: str) -> tuple[list[int], int, int]:
    target_ids = continuation_ids(tokenizer, target)
    opposite_ids = continuation_ids(tokenizer, opposite)
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
    return target_ids[:common], int(target_ids[common]), int(opposite_ids[common])


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


def history_text(block_id: str, history: str) -> str:
    if history == "bare":
        return ""
    if block_id.startswith("language_action"):
        return (
            "Completed unrelated example: the sample codeword coddemo maps to demoresult. "
            "Recorded result: demoresult.\n"
        )
    return (
        "Completed unrelated example: Cira greeted Alen, who thanked Boro. "
        "Recorded answer: Alen.\n"
    )


def answer_tail(interface: str, answer_kind: str) -> tuple[str, str]:
    if interface == "direct":
        return (
            f"Question: Who or what is the correct {answer_kind}?",
            f"Answer with only the exact {answer_kind}:",
        )
    return (
        f"Query: Select the correct {answer_kind} for the result field.",
        "Result:",
    )


def translation_prompt(
    block_id: str,
    pair_index: int,
    role: str,
    interface: str,
    history: str,
) -> dict[str, str]:
    key = suffix(pair_index + (0 if "role_swap" in block_id else 130))
    focus = f"cod{key}"
    target_a = f"lum{key}"
    target_b = f"ner{key}"
    neutral = f"aux{key}"
    demo_a = f"dema{key}"
    demo_b = f"demb{key}"
    if "role_swap" in block_id:
        reference = (
            f"Reference table: the focus codeword {focus} has Luma form {target_a} "
            f"and Neri form {target_b}. The neutral codeword {neutral} has Luma form "
            f"{demo_a} and Neri form {demo_b}."
        )
        target = target_a if role == "a" else target_b
        opposite = target_b if role == "a" else target_a
        role_changes_correct_event = True
    else:
        reference = (
            f"Reference table: the focus codeword {focus} has the same form {target_a} "
            f"in both Luma and Neri. The codeword {target_b} is an unrelated distractor. "
            f"The neutral codeword {neutral} has form {demo_a}."
        )
        target = target_a
        opposite = target_b
        role_changes_correct_event = False
    role_name = "Luma" if role == "a" else "Neri"
    focus_line = f"Focus codeword: {focus}."
    control_line = f"Position control codeword: {neutral}."
    query_line, tail = answer_tail(interface, "codeword")
    prompt = "\n".join(
        part
        for part in (
            history_text(block_id, history).rstrip("\n"),
            reference,
            f"Active target language: {role_name}.",
            focus_line,
            control_line,
            query_line,
            tail,
        )
        if part
    )
    return {
        "prompt": prompt,
        "target": target,
        "opposite": opposite,
        "focus": focus,
        "control": neutral,
        "focus_line": focus_line,
        "control_line": control_line,
        "query_line": query_line,
        "role_label": role_name,
        "role_line": f"Active target language: {role_name}.",
        "reference_line": reference,
        "role_changes_correct_event": role_changes_correct_event,
    }


def relative_prompt(
    block_id: str,
    pair_index: int,
    role: str,
    interface: str,
    history: str,
) -> dict[str, str]:
    key = suffix(pair_index + (260 if "role_swap" in block_id else 390))
    anchor = f"Ari{key.capitalize()}"
    other = f"Beno{key.capitalize()}"
    neutral = f"Calo{key.capitalize()}"
    if "role_swap" in block_id:
        if role == "a":
            context = f"Context: {neutral} greeted {anchor}, who thanked {other}."
            target, opposite = anchor, other
        else:
            context = f"Context: {neutral} greeted {anchor}, whom {other} thanked."
            target, opposite = other, anchor
        role_changes_correct_event = True
    else:
        if role == "a":
            context = f"Context: {neutral} greeted {anchor}, who thanked {other}."
        else:
            context = f"Context: {anchor}, who thanked {other}, was greeted by {neutral}."
        target, opposite = anchor, other
        role_changes_correct_event = False
    focus_line = f"Focus person: {anchor}."
    control_line = f"Position control person: {neutral}."
    if interface == "direct":
        query_line = "Question: Who did the thanking?"
        tail = "Answer with only the exact name:"
    else:
        query_line = "Query: Select the person who performed the thanking."
        tail = "Result:"
    prompt = "\n".join(
        part
        for part in (
            history_text(block_id, history).rstrip("\n"),
            context,
            focus_line,
            control_line,
            query_line,
            tail,
        )
        if part
    )
    return {
        "prompt": prompt,
        "target": target,
        "opposite": opposite,
        "focus": anchor,
        "control": neutral,
        "focus_line": focus_line,
        "control_line": control_line,
        "query_line": query_line,
        "role_label": "relative_subject" if role == "a" else "relative_object_or_surface",
        "role_line": context,
        "reference_line": context,
        "role_changes_correct_event": role_changes_correct_event,
    }


def pair_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        for pair_index in range(96):
            split = split_for_pair(pair_index)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "block_id": block["block_id"],
                    "family_id": block["family_id"],
                    "mechanism_id": block["mechanism_id"],
                    "candidate": block["candidate"],
                    "matched_control_block_id": block["matched_control_block_id"],
                    "pair_id": f"phase425__{block['block_id']}__{pair_index:03d}",
                    "pair_index": pair_index,
                    "split": split,
                    "pipeline_sealed": split == "sealed_physical_holdout",
                    "replica_group_id": f"{block['block_id']}__replica_{pair_index // 2:03d}",
                    "lexical_replica": pair_index % 2,
                    "condition_count_per_model": 8,
                    "strict_human_double_blind": False,
                }
            )
    return rows


def register_condition(
    model: str,
    tokenizer: Any,
    pair: dict[str, Any],
    role: str,
    interface: str,
    history: str,
) -> dict[str, Any]:
    if pair["family_id"] == "language_action":
        payload = translation_prompt(
            pair["block_id"], pair["pair_index"], role, interface, history
        )
    else:
        payload = relative_prompt(
            pair["block_id"], pair["pair_index"], role, interface, history
        )
    prompt = payload["prompt"]
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    prefix, target_branch, opposite_branch = branch_contract(
        tokenizer, payload["target"], payload["opposite"]
    )
    source_positions = token_positions(
        offsets, span_inside(prompt, payload["focus_line"], payload["focus"])
    )
    control_positions = token_positions(
        offsets, span_inside(prompt, payload["control_line"], payload["control"])
    )
    query_start = prompt.find(payload["query_line"])
    query_positions = token_positions(
        offsets, (query_start, query_start + len(payload["query_line"]))
    )
    role_start = prompt.find(payload["role_line"])
    role_positions = token_positions(
        offsets, (role_start, role_start + len(payload["role_line"]))
    )
    executed_ids = [*prompt_ids, *prefix]
    condition_key = f"r{role}__i{interface}__h{history}"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": model,
        "condition_id": f"{pair['pair_id']}__{condition_key}__{model}",
        "condition_key": condition_key,
        "pair_id": pair["pair_id"],
        "pair_index": pair["pair_index"],
        "replica_group_id": pair["replica_group_id"],
        "lexical_replica": pair["lexical_replica"],
        "block_id": pair["block_id"],
        "family_id": pair["family_id"],
        "mechanism_id": pair["mechanism_id"],
        "candidate": pair["candidate"],
        "split": pair["split"],
        "pipeline_sealed": pair["pipeline_sealed"],
        "strict_human_double_blind": False,
        "role": role,
        "interface": interface,
        "history": history,
        "role_label": payload["role_label"],
        "role_changes_correct_event": payload["role_changes_correct_event"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "base_prompt_token_count": len(prompt_ids),
        "common_branch_prefix_token_ids": prefix,
        "executed_token_count": len(executed_ids),
        "executed_token_ids_sha256": hashlib.sha256(
            json.dumps(executed_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "prediction_position": len(executed_ids) - 1,
        "source_positions": source_positions,
        "query_positions": query_positions,
        "instruction_control_positions": control_positions,
        "role_positions": role_positions,
        "source_token_count": len(source_positions),
        "query_token_count": len(query_positions),
        "instruction_control_token_count": len(control_positions),
        "target": payload["target"],
        "opposite_target": payload["opposite"],
        "target_branch_token_id": target_branch,
        "opposite_branch_token_id": opposite_branch,
        "target_word_count": 1,
        "target_absent_from_prompt": payload["target"] not in prompt,
        "negative_control": not pair["candidate"],
        "open_set_control": False,
        "physical": True,
        "observer_overlay": True,
        "predictive": False,
        "causal": False,
    }


def validate(
    pairs: list[dict[str, Any]],
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [*open_rows, *sealed_rows]
    split_pairs = Counter(row["split"] for row in pairs)
    block_pairs = Counter(row["block_id"] for row in pairs)
    model_conditions = Counter(row["model"] for row in rows)
    block_conditions = Counter(row["block_id"] for row in rows)
    split_conditions = Counter(row["split"] for row in rows)
    replica_splits: dict[str, set[str]] = defaultdict(set)
    replica_counts: Counter[str] = Counter()
    for pair in pairs:
        replica_splits[pair["replica_group_id"]].add(pair["split"])
        replica_counts[pair["replica_group_id"]] += 1
    valid = bool(
        len(pairs) == 384
        and len(rows) == 9216
        and len(open_rows) == 6912
        and len(sealed_rows) == 2304
        and all(value == 96 for value in block_pairs.values())
        and all(value == 96 for value in split_pairs.values())
        and all(value == 3072 for value in model_conditions.values())
        and all(value == 2304 for value in block_conditions.values())
        and all(value == 2304 for value in split_conditions.values())
        and all(len(value) == 1 for value in replica_splits.values())
        and all(value == 2 for value in replica_counts.values())
        and len({row["condition_id"] for row in rows}) == len(rows)
        and all(row["source_positions"] for row in rows)
        and all(row["query_positions"] for row in rows)
        and all(row["instruction_control_positions"] for row in rows)
    )
    return {
        "valid": valid,
        "block_count": len(block_pairs),
        "pair_count": len(pairs),
        "condition_count": len(rows),
        "open_condition_count": len(open_rows),
        "sealed_condition_count": len(sealed_rows),
        "pair_counts_by_block": dict(sorted(block_pairs.items())),
        "pair_counts_by_split": dict(sorted(split_pairs.items())),
        "condition_counts_by_model": dict(sorted(model_conditions.items())),
        "condition_counts_by_block": dict(sorted(block_conditions.items())),
        "condition_counts_by_split": dict(sorted(split_conditions.items())),
        "replica_group_count": len(replica_counts),
        "replica_group_split_leak_count": sum(
            len(value) != 1 for value in replica_splits.values()
        ),
    }


def freeze() -> dict[str, Any]:
    protocol_path = OUT / "phase425_protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if (
            protocol.get("schema_version") == SCHEMA_VERSION
            and protocol.get("validation", {}).get("valid")
        ):
            return protocol
    pairs = pair_rows()
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for pair in pairs:
        destination = sealed_rows if pair["pipeline_sealed"] else open_rows
        for model, tokenizer in tokenizers.items():
            for role in ROLES:
                for interface in INTERFACES:
                    for history in HISTORIES:
                        destination.append(
                            register_condition(
                                model, tokenizer, pair, role, interface, history
                            )
                        )
    validation = validate(pairs, open_rows, sealed_rows)
    if not validation["valid"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    write_jsonl(OUT / "phase425_registered_pairs.jsonl", pairs)
    write_jsonl(OUT / "phase425_registered_conditions_open.jsonl", open_rows)
    write_jsonl(OUT / "sealed" / "phase425_registered_conditions_sealed.jsonl", sealed_rows)
    sealed_commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": "sealed_physical_holdout",
        "sealed_condition_count": len(sealed_rows),
        "sealed_condition_rows_sha256": digest_rows(sealed_rows),
        "sealed_prompt_hashes_sha256": hashlib.sha256(
            "\n".join(sorted(row["prompt_sha256"] for row in sealed_rows)).encode("utf-8")
        ).hexdigest(),
        "human_double_blind": False,
        "pipeline_sealed_until_gate_freeze": True,
    }
    write_json(OUT / "phase425_sealed_commitment.json", sealed_commitment)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "objective": (
            "Distinguish matched-literal functional-role formation from lexical, position, "
            "interface and history separation before any sealed or causal promotion."
        ),
        "models": list(MODELS),
        "blocks": list(BLOCKS),
        "interfaces": list(INTERFACES),
        "histories": list(HISTORIES),
        "roles": list(ROLES),
        "execution_dtype_by_model": {
            "qwen3": "float16",
            "glm4": "bfloat16",
            "deepseek7b": "bfloat16",
        },
        "registered_thresholds": {
            "behavior_correct_fraction_min": 0.75,
            "signal_positive_fraction_min": 2 / 3,
            "signal_median_min": 0.0,
            "role_delta_coherence_min": 0.20,
            "prediction_r2_min": 0.0,
            "prediction_delta_r2_min": 0.05,
            "prediction_mae_gain_min": 0.0,
            "cross_model_replication_min": 2,
            "component_ledger_relative_error_max": 0.01,
        },
        "split_contract": {
            "discovery_pairs_per_block": 24,
            "calibration_pairs_per_block": 24,
            "behavior_holdout_pairs_per_block": 24,
            "sealed_physical_holdout_pairs_per_block": 24,
            "conditions_per_pair_per_model": 8,
            "replica_pairs_per_group": 2,
            "split_unit": "lexical_replica_group",
        },
        "prediction_contract": {
            "baseline_features": [
                "executed_token_count_mean",
                "source_token_count_mean",
                "query_token_count_mean",
            ],
            "physical_features": [
                "formation_specificity",
                "transport_specificity",
                "competition_specificity",
            ],
            "ridge_alpha": 1.0,
            "fit_split": "discovery",
            "gate_splits": ["calibration", "behavior_holdout"],
            "sealed_split_used_for_selection": False,
        },
        "role_signal_contract": {
            "functional_specificity": (
                "role contrast minus the larger matched-position or interface-history contrast"
            ),
            "role_dominance": "role contrast minus lexical-main-effect contrast",
            "strict_path_specificity": (
                "minimum of functional specificity and role dominance"
            ),
            "cross_lexical_covariance": (
                "cosine agreement of role-difference vectors across two lexical replicas"
            ),
            "interpretation": (
                "functional specificity is evidence for a reusable role component; "
                "positive role dominance is the stronger Phase425 sealed-unlock requirement"
            ),
        },
        "evidence_contract": {
            "attention_source_write_is_legal_compute_quantity": True,
            "event_order_graph_edges_are_compute_edges": False,
            "competition_is_observer": True,
            "strict_human_double_blind": False,
            "pipeline_sealed": True,
            "causal_claim_allowed_before_sealed_pass": False,
            "head_channel_neuron_scan_allowed": False,
        },
        "pair_rows_sha256": digest_rows(pairs),
        "open_condition_rows_sha256": digest_rows(open_rows),
        "implementation_commitments": {
            path.name: digest_file(path)
            for path in (
                Path(__file__),
                ROOT / "tests/gpt5/phase425_role_exchange_collect.py",
                ROOT / "tests/gpt5/phase425_role_exchange_analysis.py",
            )
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
    protocol = freeze()
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
