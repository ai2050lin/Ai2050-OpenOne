#!/usr/bin/env python3
"""Phase 1000 orthogonal protocol for entity-value binding.

The protocol separates four factors:

1. entity identity at each fact slot;
2. value identity at each fact slot;
3. entity-value binding induced by their slot co-occurrence;
4. queried entity identity.

Entity-swap pairs change only the two fact entity tokens. Color tokens and
their positions remain fixed, while the correct answer changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS


PHASE = 1000
MODEL = "qwen3"
COLORS = ("red", "blue", "green", "yellow")
NAMES = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Jack", "Kelly", "Paul", "Ruby", "Sam", "Blake", "Leo", "Will",
    "Iris", "Liam", "Maya", "Nora", "Oscar", "Quinn", "Tina", "Uma",
)
TEMPLATE_COUNT = 4
FORMAL_WORLDS = 128
SMOKE_WORLDS = 4
CASES_PER_WORLD = TEMPLATE_COUNT * 2 * 2 * 2 * 2
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1000_factorial_binding_scpg"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temp.replace(path)


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def split_for_world(world: int, n_worlds: int) -> str:
    if n_worlds < FORMAL_WORLDS:
        return "smoke"
    if world < 64:
        return "discovery"
    if world < 96:
        return "validation"
    return "holdout"


def render_user_prompt(
    template: int,
    first_entity: str,
    first_color: str,
    second_entity: str,
    second_color: str,
    query_entity: str,
) -> str:
    if template == 0:
        return (
            f"Records: {first_entity} carries the {first_color} marker. "
            f"{second_entity} carries the {second_color} marker.\n"
            f"Question: What color marker does {query_entity} carry?\n"
            "Answer with exactly one color word."
        )
    if template == 1:
        return (
            f"Assignments: the marker assigned to {first_entity} is {first_color}; "
            f"the marker assigned to {second_entity} is {second_color}.\n"
            f"Question: Which marker color is assigned to {query_entity}?\n"
            "Reply with one color word only."
        )
    if template == 2:
        return (
            f"Registry: {first_entity} has a {first_color} marker, while "
            f"{second_entity} has a {second_color} marker.\n"
            f"Based only on the registry, name the marker color for {query_entity}. "
            "Use one word."
        )
    if template == 3:
        return (
            f"For this task, {first_entity}'s marker color is {first_color}, whereas "
            f"{second_entity}'s marker color is {second_color}.\n"
            f"What is the marker color belonging to {query_entity}? "
            "Respond using only the color."
        )
    raise ValueError(template)


def render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def unique_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def world_specs(n_worlds: int) -> list[dict[str, Any]]:
    rng = random.Random(1000_20260723)
    name_pairs = list(combinations(NAMES, 2))
    rng.shuffle(name_pairs)
    color_pairs = list(combinations(COLORS, 2))
    rows = []
    for world in range(n_worlds):
        rows.append(
            {
                "world": world,
                "world_id": f"w{world:03d}",
                "split": split_for_world(world, n_worlds),
                "base_entities": list(name_pairs[world]),
                "base_colors": list(color_pairs[world % len(color_pairs)]),
            }
        )
    return rows


def factor_pair_ids(
    world_id: str,
    template: int,
    display_order: int,
    entity_swap: int,
    value_swap: int,
    query_role: int,
) -> dict[str, str]:
    return {
        "entity": (
            f"entity.{world_id}.t{template}.o{display_order}."
            f"v{value_swap}.q{query_role}"
        ),
        "value": (
            f"value.{world_id}.t{template}.o{display_order}."
            f"e{entity_swap}.q{query_role}"
        ),
        "query": (
            f"query.{world_id}.t{template}.o{display_order}."
            f"e{entity_swap}.v{value_swap}"
        ),
    }


def build_cases(
    tokenizer, n_worlds: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt_name_ids = {name: unique_token_id(tokenizer, " " + name) for name in NAMES}
    prompt_color_ids = {color: unique_token_id(tokenizer, " " + color) for color in COLORS}
    candidate_ids = {color: unique_token_id(tokenizer, color) for color in COLORS}
    if len(set(prompt_name_ids.values())) != len(prompt_name_ids):
        raise RuntimeError("name token IDs are not unique")
    if len(set(prompt_color_ids.values())) != len(prompt_color_ids):
        raise RuntimeError("prompt color token IDs are not unique")
    if len(set(candidate_ids.values())) != len(candidate_ids):
        raise RuntimeError("candidate token IDs are not unique")

    cases: list[dict[str, Any]] = []
    factor_groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        factor: defaultdict(list) for factor in ("entity", "value", "query")
    }
    token_lengths: dict[int, set[int]] = defaultdict(set)

    for world in world_specs(n_worlds):
        base_entities = world["base_entities"]
        base_colors = world["base_colors"]
        for template in range(TEMPLATE_COUNT):
            for display_order in (0, 1):
                for entity_swap in (0, 1):
                    for value_swap in (0, 1):
                        for query_role in (0, 1):
                            slot_entities = (
                                list(base_entities)
                                if entity_swap == 0
                                else [base_entities[1], base_entities[0]]
                            )
                            slot_colors = (
                                list(base_colors)
                                if value_swap == 0
                                else [base_colors[1], base_colors[0]]
                            )
                            query_entity = base_entities[query_role]
                            query_slot = slot_entities.index(query_entity)
                            gold = slot_colors[query_slot]
                            foil = slot_colors[1 - query_slot]
                            first_slot, second_slot = (
                                (0, 1) if display_order == 0 else (1, 0)
                            )
                            prompt = render_user_prompt(
                                template,
                                slot_entities[first_slot],
                                slot_colors[first_slot],
                                slot_entities[second_slot],
                                slot_colors[second_slot],
                                query_entity,
                            )
                            rendered = render_chat(tokenizer, prompt)
                            ids = [
                                int(value)
                                for value in tokenizer.encode(
                                    rendered, add_special_tokens=False
                                )
                            ]

                            entity_fact_positions: dict[str, int] = {}
                            for entity in base_entities:
                                positions = positions_of(ids, prompt_name_ids[entity])
                                expected = 2 if entity == query_entity else 1
                                if len(positions) != expected:
                                    raise RuntimeError(
                                        f"entity role drift: {world['world_id']}/"
                                        f"t{template}/e{entity_swap}/q{query_role}/"
                                        f"{entity}/{positions}"
                                    )
                                entity_fact_positions[entity] = positions[0]
                            query_positions = positions_of(
                                ids, prompt_name_ids[query_entity]
                            )
                            color_positions = {
                                color: positions_of(ids, prompt_color_ids[color])
                                for color in base_colors
                            }
                            if any(len(values) != 1 for values in color_positions.values()):
                                raise RuntimeError(
                                    f"color role drift: {world['world_id']}/"
                                    f"t{template}/{color_positions}"
                                )
                            for color, candidate_id in candidate_ids.items():
                                extended = tokenizer.encode(
                                    rendered + color, add_special_tokens=False
                                )
                                if extended != ids + [candidate_id]:
                                    raise RuntimeError(
                                        f"candidate boundary drift: "
                                        f"{world['world_id']}/t{template}/{color}"
                                    )

                            pair_ids = factor_pair_ids(
                                world["world_id"],
                                template,
                                display_order,
                                entity_swap,
                                value_swap,
                                query_role,
                            )
                            record_id = (
                                f"{world['world_id']}.t{template}.o{display_order}."
                                f"e{entity_swap}.v{value_swap}.q{query_role}"
                            )
                            row = {
                                "schema_version": "phase1000_case.v1",
                                "phase": PHASE,
                                "model": MODEL,
                                "record_id": record_id,
                                "world": world["world"],
                                "world_id": world["world_id"],
                                "split": world["split"],
                                "template": template,
                                "display_order": display_order,
                                "entity_swap": entity_swap,
                                "value_swap": value_swap,
                                "query_role": query_role,
                                "base_entities": base_entities,
                                "base_colors": base_colors,
                                "slot_entities": slot_entities,
                                "slot_colors": slot_colors,
                                "query_entity": query_entity,
                                "query_slot": query_slot,
                                "gold": gold,
                                "foil": foil,
                                "prompt": prompt,
                                "rendered_prompt": rendered,
                                "prompt_sha256": sha256_text(prompt),
                                "rendered_prompt_sha256": sha256_text(rendered),
                                "input_ids": ids,
                                "input_token_count": len(ids),
                                "candidate_token_ids": candidate_ids,
                                "factor_pair_ids": pair_ids,
                                "role_positions": {
                                    "slot0_entity": entity_fact_positions[
                                        slot_entities[0]
                                    ],
                                    "slot0_color": color_positions[
                                        slot_colors[0]
                                    ][0],
                                    "slot1_entity": entity_fact_positions[
                                        slot_entities[1]
                                    ],
                                    "slot1_color": color_positions[
                                        slot_colors[1]
                                    ][0],
                                    "query_name": query_positions[-1],
                                    "answer_boundary": len(ids) - 1,
                                },
                            }
                            token_lengths[template].add(len(ids))
                            cases.append(row)
                            for factor, pair_id in pair_ids.items():
                                factor_groups[factor][pair_id].append(row)

    expected_cases = n_worlds * CASES_PER_WORLD
    if len(cases) != expected_cases:
        raise RuntimeError(f"case count drift: {len(cases)} != {expected_cases}")
    if any(len(lengths) != 1 for lengths in token_lengths.values()):
        raise RuntimeError(f"template token lengths drift: {token_lengths}")

    expected_changed_roles = {
        "entity": ("slot0_entity", "slot1_entity"),
        "value": ("slot0_color", "slot1_color"),
        "query": ("query_name",),
    }
    factor_arm = {
        "entity": "entity_swap",
        "value": "value_swap",
        "query": "query_role",
    }
    pair_rows: list[dict[str, Any]] = []
    pair_audits: list[dict[str, Any]] = []
    for factor in ("entity", "value", "query"):
        for pair_id, rows in sorted(factor_groups[factor].items()):
            arm_field = factor_arm[factor]
            rows = sorted(rows, key=lambda row: row[arm_field])
            if len(rows) != 2 or [row[arm_field] for row in rows] != [0, 1]:
                raise RuntimeError(f"pair arm drift: {pair_id}/{len(rows)}")
            arm0, arm1 = rows
            if len(arm0["input_ids"]) != len(arm1["input_ids"]):
                raise RuntimeError(f"pair length drift: {pair_id}")
            changed = [
                index
                for index, (left, right) in enumerate(
                    zip(arm0["input_ids"], arm1["input_ids"])
                )
                if left != right
            ]
            expected_changed = sorted(
                arm0["role_positions"][role]
                for role in expected_changed_roles[factor]
            )
            if changed != expected_changed:
                raise RuntimeError(
                    f"changed position drift: {pair_id}/{changed}/{expected_changed}"
                )
            same_multiset = Counter(arm0["input_ids"]) == Counter(arm1["input_ids"])
            if factor in ("entity", "value") and not same_multiset:
                raise RuntimeError(f"token multiset drift: {pair_id}")
            if arm0["gold"] != arm1["foil"] or arm0["foil"] != arm1["gold"]:
                raise RuntimeError(f"answer swap drift: {pair_id}")
            pair_rows.append(
                {
                    "schema_version": "phase1000_factor_pair.v1",
                    "phase": PHASE,
                    "model": MODEL,
                    "pair_id": pair_id,
                    "factor": factor,
                    "split": arm0["split"],
                    "world_id": arm0["world_id"],
                    "template": arm0["template"],
                    "display_order": arm0["display_order"],
                    "arm_field": arm_field,
                    "arm0_record_id": arm0["record_id"],
                    "arm1_record_id": arm1["record_id"],
                    "arm0_gold": arm0["gold"],
                    "arm1_gold": arm1["gold"],
                    "changed_positions": changed,
                    "same_token_multiset": same_multiset,
                }
            )
            pair_audits.append(
                {
                    "pair_id": pair_id,
                    "factor": factor,
                    "changed_positions": changed,
                    "same_length": True,
                    "same_token_multiset": same_multiset,
                    "answer_swapped": True,
                }
            )

    expected_pairs_per_factor = expected_cases // 2
    pair_counts = Counter(row["factor"] for row in pair_rows)
    if any(
        pair_counts[factor] != expected_pairs_per_factor
        for factor in ("entity", "value", "query")
    ):
        raise RuntimeError(f"pair count drift: {pair_counts}")

    summary = {
        "schema_version": "phase1000_protocol.v1",
        "phase": PHASE,
        "model": MODEL,
        "world_count": n_worlds,
        "case_count": len(cases),
        "cases_per_world": CASES_PER_WORLD,
        "pair_count": len(pair_rows),
        "pair_counts_by_factor": dict(pair_counts),
        "template_count": TEMPLATE_COUNT,
        "token_lengths_by_template": {
            str(template): next(iter(lengths))
            for template, lengths in token_lengths.items()
        },
        "candidate_token_ids": candidate_ids,
        "prompt_color_token_ids": prompt_color_ids,
        "prompt_name_token_ids": prompt_name_ids,
        "split_counts": dict(Counter(row["split"] for row in cases)),
        "factor_counts": {
            factor: dict(Counter(str(row[factor]) for row in cases))
            for factor in (
                "template",
                "display_order",
                "entity_swap",
                "value_swap",
                "query_role",
            )
        },
        "entity_pairs_keep_color_tokens_and_positions_fixed": True,
        "entity_and_value_pairs_preserve_token_multiset": True,
        "case_manifest_sha256": sha256_text(
            "\n".join(canonical(row) for row in cases) + "\n"
        ),
        "pair_manifest_sha256": sha256_text(
            "\n".join(canonical(row) for row in pair_rows) + "\n"
        ),
        "pair_audit_sha256": sha256_text(canonical(pair_audits)),
        "cpu_protocol_pass": True,
    }
    return cases, pair_rows, summary


def audit_protocol(
    cases: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    case_by_id = {row["record_id"]: row for row in cases}
    failures: list[str] = []
    for pair in pairs:
        arm0 = case_by_id[pair["arm0_record_id"]]
        arm1 = case_by_id[pair["arm1_record_id"]]
        changed = [
            index
            for index, (left, right) in enumerate(
                zip(arm0["input_ids"], arm1["input_ids"])
            )
            if left != right
        ]
        if changed != pair["changed_positions"]:
            failures.append(f"changed/{pair['pair_id']}")
        if pair["factor"] == "entity":
            for role in ("slot0_color", "slot1_color", "query_name"):
                position = arm0["role_positions"][role]
                if arm0["input_ids"][position] != arm1["input_ids"][position]:
                    failures.append(f"entity_fixed/{pair['pair_id']}/{role}")
        if arm0["gold"] != arm1["foil"] or arm0["foil"] != arm1["gold"]:
            failures.append(f"answer/{pair['pair_id']}")
    return {
        "schema_version": "phase1000_protocol_audit.v1",
        "phase": PHASE,
        "case_count": len(cases),
        "pair_count": len(pairs),
        "factor_pair_counts": dict(Counter(row["factor"] for row in pairs)),
        "failures": failures,
        "passed": (
            not failures
            and summary["cpu_protocol_pass"]
            and summary["case_count"] == len(cases)
            and summary["pair_count"] == len(pairs)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()
    n_worlds = SMOKE_WORLDS if args.scope == "smoke" else FORMAL_WORLDS
    output_root = OUT_ROOT / ("smoke" if args.scope == "smoke" else "protocol")
    tokenizer = load_tokenizer()
    cases, pairs, summary = build_cases(tokenizer, n_worlds)
    audit = audit_protocol(cases, pairs, summary)
    if not audit["passed"]:
        raise RuntimeError(f"protocol audit failed: {audit}")
    write_jsonl(output_root / "cases.jsonl", cases)
    write_jsonl(output_root / "factor_pairs.jsonl", pairs)
    write_json(output_root / "protocol.json", summary)
    write_json(output_root / "audit.json", audit)
    print(
        json.dumps(
            {
                "passed": True,
                "scope": args.scope,
                "case_count": len(cases),
                "pair_count": len(pairs),
                "pair_counts_by_factor": summary["pair_counts_by_factor"],
                "output": str(output_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
