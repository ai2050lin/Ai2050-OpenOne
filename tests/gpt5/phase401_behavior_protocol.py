#!/usr/bin/env python3
"""Freeze a fresh Phase401 four-surface denominator and independent batch pilot."""

from __future__ import annotations

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
from phase386_multitime_protocol import NAMES, OBJECTS  # noqa: E402
from phase390_role_mapping import prompt_token_ids  # noqa: E402
from phase397_multitask_protocol import ROLE_VALUES  # noqa: E402
from phase398_joint_factorial_protocol import (  # noqa: E402
    AXES,
    LEVELS,
    lexical_audit,
    order_audit,
    parse_condition,
    query_audit,
    relation_audit,
)
from phase399_dynamic_binding_protocol import compatible_pool, lexical_items  # noqa: E402
from phase400_dynamic_protocol import render  # noqa: E402
from phase401_local_edge_protocol import (  # noqa: E402
    MODELS,
    OUT,
    SPLIT_CANDIDATE_COUNTS,
    SURFACES,
)


CONDITIONS = tuple(
    f"{axis}_R{relation}_O{order}_Q{query}"
    for axis in AXES
    for relation in LEVELS
    for order in LEVELS
    for query in LEVELS
)
CANDIDATE_GROUPS_PER_SURFACE = sum(SPLIT_CANDIDATE_COUNTS.values())
PILOT_GROUPS_PER_SURFACE = 1
FAMILY_BY_SURFACE = {
    "possession_relation": "content_knowledge",
    "role_filling": "language_action",
    "coreference_resolution": "reasoning_constraint",
    "field_extraction": "content_knowledge",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


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


def values_for(surface: str) -> tuple[str, ...]:
    return tuple(ROLE_VALUES if surface == "role_filling" else OBJECTS)


def split_for(priority: int) -> str:
    cursor = 0
    for split, count in SPLIT_CANDIDATE_COUNTS.items():
        if priority < cursor + count:
            return split
        cursor += count
    raise ValueError(f"Invalid Phase401 group priority: {priority}")


def previous_signatures() -> set[tuple[str, str, str, str, str]]:
    paths = (
        ROOT / "tests/gpt5/result/phase398_joint_binding/protocol/private/phase398_candidate_cases.jsonl",
        ROOT / "tests/gpt5/result/phase399_dynamic_binding/protocol/private/phase399_candidate_cases.jsonl",
        ROOT / "tests/gpt5/result/phase400_partial_order/protocol/private/phase400_candidate_cases.jsonl",
    )
    signatures: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["private_execution_model"] != "qwen3":
                continue
            if row["anonymous_condition_slot"] not in {"X_R0_O0_Q0", "Y_R0_O0_Q0"}:
                continue
            slots = row["semantic_slot_fragments_private"]
            signatures.add(
                (
                    row["task_surface_private"],
                    slots["entity_a"],
                    slots["entity_b"],
                    slots["value_a"],
                    slots["value_b"],
                )
            )
    return signatures


def main() -> None:
    frozen_contract = OUT / "phase401_local_edge_protocol.json"
    if not frozen_contract.is_file():
        raise FileNotFoundError("Freeze phase401_local_edge_protocol.json first")
    contract_hash = digest(frozen_contract.read_text(encoding="utf-8"))
    created_at = now()
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    prior = previous_signatures()
    entity_pool = compatible_pool(tuple(NAMES), tokenizers)
    value_pools = {
        surface: compatible_pool(values_for(surface), tokenizers) for surface in SURFACES
    }
    if len(entity_pool) < 8 or any(len(pool) < 8 for pool in value_pools.values()):
        raise RuntimeError("Insufficient Phase401 cross-model token-width pools")

    rows: list[dict[str, Any]] = []
    pilot_rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    pilot_groups: list[dict[str, Any]] = []
    prompt_hashes: set[tuple[str, str]] = set()
    signatures = set(prior)
    for surface_index, surface in enumerate(SURFACES):
        accepted = 0
        pilot_accepted = 0
        seed = 141000 + 4000 * surface_index
        rejects: Counter[str] = Counter()
        total_needed = CANDIDATE_GROUPS_PER_SURFACE + PILOT_GROUPS_PER_SURFACE
        while accepted + pilot_accepted < total_needed and seed < 240000:
            try:
                entity_a, entity_b, v1, v2, v3, v4 = lexical_items(
                    surface, seed, entity_pool, value_pools[surface]
                )
            except ValueError:
                seed += 1
                continue
            pair_signatures = {
                (surface, entity_a, entity_b, v1, v2),
                (surface, entity_a, entity_b, v3, v4),
            }
            if signatures.intersection(pair_signatures):
                rejects["prior_or_within_phase_signature_overlap"] += 1
                seed += 1
                continue
            per_model: dict[str, dict[str, Any]] = {}
            valid = True
            for model in MODELS:
                tokenizer = tokenizers[model]
                cases = {
                    condition: render(
                        tokenizer,
                        model,
                        surface,
                        condition,
                        entity_a,
                        entity_b,
                        *((v1, v2) if condition.startswith("X_") else (v3, v4)),
                    )
                    for condition in CONDITIONS
                }
                audits: list[dict[str, Any]] = []
                for axis in AXES:
                    for order in LEVELS:
                        for query in LEVELS:
                            audits.append(
                                relation_audit(
                                    tokenizer,
                                    cases[f"{axis}_R0_O{order}_Q{query}"],
                                    cases[f"{axis}_R1_O{order}_Q{query}"],
                                )
                            )
                    for relation in LEVELS:
                        for query in LEVELS:
                            audits.append(
                                order_audit(
                                    tokenizer,
                                    cases[f"{axis}_R{relation}_O0_Q{query}"],
                                    cases[f"{axis}_R{relation}_O1_Q{query}"],
                                )
                            )
                        for order in LEVELS:
                            audits.append(
                                query_audit(
                                    tokenizer,
                                    cases[f"{axis}_R{relation}_O{order}_Q0"],
                                    cases[f"{axis}_R{relation}_O{order}_Q1"],
                                )
                            )
                for relation in LEVELS:
                    for order in LEVELS:
                        for query in LEVELS:
                            audits.append(
                                lexical_audit(
                                    tokenizer,
                                    cases[f"X_R{relation}_O{order}_Q{query}"],
                                    cases[f"Y_R{relation}_O{order}_Q{query}"],
                                )
                            )
                invalid = next((item for item in audits if not item["valid"]), None)
                if invalid is not None:
                    rejects[f"{model}:{invalid['reason']}"] += 1
                    valid = False
                    break
                target_ids = {
                    tokenizer(" " + cases[condition]["target"], add_special_tokens=False)[
                        "input_ids"
                    ][0]
                    for condition in CONDITIONS
                }
                if len(target_ids) != 4:
                    rejects[f"{model}:target_first_token_not_four_way_distinct"] += 1
                    valid = False
                    break
                per_model[model] = cases
            if not valid:
                seed += 1
                continue

            signatures.update(pair_signatures)
            is_pilot = accepted >= CANDIDATE_GROUPS_PER_SURFACE
            if is_pilot:
                group_priority = pilot_accepted
                group_id = "p401bp_" + digest(f"phase401-batch-pilot:{surface}:{seed}", 24)
                split = "batch_sensitivity_pilot"
                pilot_groups.append(
                    {
                        "schema_version": "75.1.0",
                        "phase_id": "Phase401-BehaviorProtocol",
                        "anonymous_parallel_group_id": group_id,
                        "task_surface": surface,
                        "candidate_split": split,
                        "group_priority": group_priority,
                        "formal_denominator": False,
                    }
                )
            else:
                group_priority = accepted
                group_id = "p401g_" + digest(f"phase401:{surface}:{seed}", 24)
                split = split_for(accepted)
                groups.append(
                    {
                        "schema_version": "75.1.0",
                        "phase_id": "Phase401-BehaviorProtocol",
                        "anonymous_parallel_group_id": group_id,
                        "task_surface": surface,
                        "candidate_split": split,
                        "group_priority": group_priority,
                        "condition_count": len(CONDITIONS),
                        "fresh_against_phase398_to_phase400_pair_signatures": True,
                        "formal_denominator": True,
                    }
                )
            selection_priority = digest(
                f"phase401-selection:{surface}:{split}:{group_id}", 24
            )
            target_rows = pilot_rows if is_pilot else rows
            for model in MODELS:
                tokenizer = tokenizers[model]
                for condition in CONDITIONS:
                    item = per_model[model][condition]
                    prompt_hash = digest(item["prompt"])
                    if (model, prompt_hash) in prompt_hashes:
                        raise RuntimeError("Duplicate Phase401 prompt within model")
                    prompt_hashes.add((model, prompt_hash))
                    target_rows.append(
                        {
                            "schema_version": "75.1.0",
                            "phase_id": "Phase401-BehaviorProtocol",
                            "created_at": created_at,
                            "frozen_local_edge_protocol_sha256": contract_hash,
                            "private_execution_model": model,
                            "anonymous_model_id": "p401m_" + digest(model, 12),
                            "blind_case_id": "p401c_"
                            + digest(f"{model}:{group_id}:{condition}", 26),
                            "anonymous_parallel_group_id": group_id,
                            "anonymous_group_id": "p401s_" + digest(surface, 12),
                            "anonymous_condition_slot": condition,
                            "candidate_split_private": split,
                            "selection_priority_private": selection_priority,
                            "group_priority": group_priority,
                            "family_id": FAMILY_BY_SURFACE[surface],
                            "mechanism_id": surface,
                            "semantic_group_id": f"p401_private_{surface}_{seed:05d}",
                            "contrast_condition": condition,
                            "task_surface_private": surface,
                            "prompt": item["prompt"],
                            "raw_prompt": item["raw_prompt"],
                            "source_fragment": item["source_fragment"],
                            "query_fragment": item["query_fragment"],
                            "clause_fragments_private": item["clause_fragments_private"],
                            "tokenization_add_special_tokens": item[
                                "tokenization_add_special_tokens"
                            ],
                            "interface": item["interface"],
                            "answer_phase": item["answer_phase"],
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                            "axis_private": item["axis_private"],
                            "relation_level_private": item["relation_level_private"],
                            "order_level_private": item["order_level_private"],
                            "query_level_private": item["query_level_private"],
                            "semantic_role_fragments_private": item[
                                "semantic_role_fragments_private"
                            ],
                            "semantic_slot_fragments_private": item[
                                "semantic_slot_fragments_private"
                            ],
                            "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                            "formal_denominator": not is_pilot,
                        }
                    )
            if is_pilot:
                pilot_accepted += 1
            else:
                accepted += 1
            print(
                f"[phase401/{surface}] formal={accepted}/{CANDIDATE_GROUPS_PER_SURFACE} "
                f"pilot={pilot_accepted}/{PILOT_GROUPS_PER_SURFACE} seed={seed}",
                flush=True,
            )
            seed += 1
        if accepted != CANDIDATE_GROUPS_PER_SURFACE or pilot_accepted != PILOT_GROUPS_PER_SURFACE:
            raise RuntimeError(
                f"Could not freeze Phase401 groups for {surface}: formal={accepted}, "
                f"pilot={pilot_accepted}; rejects={dict(rejects)}"
            )

    expected = len(SURFACES) * CANDIDATE_GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    expected_pilot = len(SURFACES) * PILOT_GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected or len(pilot_rows) != expected_pilot:
        raise RuntimeError(
            f"Invalid Phase401 row counts formal={len(rows)}/{expected}, "
            f"pilot={len(pilot_rows)}/{expected_pilot}"
        )
    write_jsonl(OUT / "protocol/private/phase401_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/private/phase401_batch_pilot_cases.jsonl", pilot_rows)
    write_jsonl(OUT / "protocol/phase401_blind_group_registry.jsonl", groups)
    write_jsonl(OUT / "protocol/phase401_batch_pilot_group_registry.jsonl", pilot_groups)
    payload = {
        "schema_version": "75.1.0",
        "phase_id": "Phase401-BehaviorProtocol",
        "created_at": created_at,
        "frozen_local_edge_protocol_sha256": contract_hash,
        "denominator": {
            "task_surfaces": list(SURFACES),
            "candidate_groups_per_surface": CANDIDATE_GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "candidate_split_group_counts": SPLIT_CANDIDATE_COUNTS,
        },
        "batch_sensitivity_pilot": {
            "group_count": len(pilot_groups),
            "case_count_per_execution_shape": len(pilot_rows),
            "formal_denominator_overlap": False,
        },
        "authorization": {
            "run_batch_sensitivity_pilot": True,
            "run_formal_behavior": True,
            "run_internal_before_behavior_freeze": False,
            "run_head_channel_or_neuron_scan": False,
        },
        "claim_boundary": {
            "fresh_behavior_is_a_local_edge": False,
            "batch_invariance_is_assumed": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase401_behavior_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
