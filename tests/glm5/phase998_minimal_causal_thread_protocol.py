#!/usr/bin/env python3
"""Phase 998 protocol for the minimal counterfactual causal-thread test.

The protocol creates matched prompts in which both entities and both colors
are always present.  Only the entity-color binding changes between arms.  The
token multiset is therefore identical inside each counterfactual pair.
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


PHASE = 998
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
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase998_minimal_causal_thread"


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


def world_specs(n_worlds: int) -> list[dict[str, Any]]:
    rng = random.Random(998_20260723)
    name_pairs = list(combinations(NAMES, 2))
    rng.shuffle(name_pairs)
    color_pairs = list(combinations(COLORS, 2))
    rows = []
    for world in range(n_worlds):
        entity_a, entity_b = name_pairs[world]
        color_a, color_b = color_pairs[world % len(color_pairs)]
        rows.append(
            {
                "world": world,
                "world_id": f"w{world:03d}",
                "split": split_for_world(world, n_worlds),
                "entities": [entity_a, entity_b],
                "base_colors": [color_a, color_b],
            }
        )
    return rows


def unique_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def build_cases(tokenizer, n_worlds: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_lengths: dict[int, set[int]] = defaultdict(set)

    for world in world_specs(n_worlds):
        entities = world["entities"]
        base_colors = world["base_colors"]
        for template in range(TEMPLATE_COUNT):
            for order in (0, 1):
                for query_role in (0, 1):
                    pair_id = (
                        f"{world['world_id']}.t{template}.o{order}.q{query_role}"
                    )
                    for arm in (0, 1):
                        assigned = (
                            list(base_colors)
                            if arm == 0
                            else [base_colors[1], base_colors[0]]
                        )
                        first_role, second_role = ((0, 1) if order == 0 else (1, 0))
                        prompt = render_user_prompt(
                            template,
                            entities[first_role],
                            assigned[first_role],
                            entities[second_role],
                            assigned[second_role],
                            entities[query_role],
                        )
                        rendered = render_chat(tokenizer, prompt)
                        ids = [
                            int(value)
                            for value in tokenizer.encode(rendered, add_special_tokens=False)
                        ]
                        gold = assigned[query_role]
                        foil = assigned[1 - query_role]
                        query_name_positions = positions_of(
                            ids, prompt_name_ids[entities[query_role]]
                        )
                        other_name_positions = positions_of(
                            ids, prompt_name_ids[entities[1 - query_role]]
                        )
                        source_color_positions = positions_of(ids, prompt_color_ids[gold])
                        foil_color_positions = positions_of(ids, prompt_color_ids[foil])
                        if len(query_name_positions) != 2:
                            raise RuntimeError(
                                f"query name role drift: {pair_id}/arm{arm}/"
                                f"{query_name_positions}"
                            )
                        if len(other_name_positions) != 1:
                            raise RuntimeError(
                                f"other name role drift: {pair_id}/arm{arm}/"
                                f"{other_name_positions}"
                            )
                        if len(source_color_positions) != 1 or len(foil_color_positions) != 1:
                            raise RuntimeError(
                                f"color role drift: {pair_id}/arm{arm}/"
                                f"{source_color_positions}/{foil_color_positions}"
                            )
                        for color, candidate_id in candidate_ids.items():
                            extended = tokenizer.encode(
                                rendered + color, add_special_tokens=False
                            )
                            if extended != ids + [candidate_id]:
                                raise RuntimeError(
                                    f"candidate boundary drift: {pair_id}/{color}"
                                )
                        row = {
                            "schema_version": "phase998_case.v1",
                            "phase": PHASE,
                            "model": MODEL,
                            "record_id": f"{pair_id}.a{arm}",
                            "pair_id": pair_id,
                            "world": world["world"],
                            "world_id": world["world_id"],
                            "split": world["split"],
                            "template": template,
                            "order": order,
                            "query_role": query_role,
                            "arm": arm,
                            "entities": entities,
                            "base_colors": base_colors,
                            "assigned_colors": assigned,
                            "query_entity": entities[query_role],
                            "gold": gold,
                            "foil": foil,
                            "contrast": f"{base_colors[query_role]}->{base_colors[1-query_role]}",
                            "prompt": prompt,
                            "rendered_prompt": rendered,
                            "prompt_sha256": sha256_text(prompt),
                            "rendered_prompt_sha256": sha256_text(rendered),
                            "input_ids": ids,
                            "input_token_count": len(ids),
                            "candidate_token_ids": candidate_ids,
                            "role_positions": {
                                "source_entity": query_name_positions[0],
                                "source_color": source_color_positions[0],
                                "foil_entity": other_name_positions[0],
                                "foil_color": foil_color_positions[0],
                                "query_name": query_name_positions[-1],
                                "answer_boundary": len(ids) - 1,
                            },
                        }
                        token_lengths[template].add(len(ids))
                        pair_rows[pair_id].append(row)
                        cases.append(row)

    expected = n_worlds * TEMPLATE_COUNT * 2 * 2 * 2
    if len(cases) != expected or len(pair_rows) * 2 != expected:
        raise RuntimeError(f"case count drift: {len(cases)} vs {expected}")

    pair_audits = []
    for pair_id, rows in pair_rows.items():
        rows = sorted(rows, key=lambda row: row["arm"])
        if [row["arm"] for row in rows] != [0, 1]:
            raise RuntimeError(f"pair arm drift: {pair_id}")
        ids_a, ids_b = rows[0]["input_ids"], rows[1]["input_ids"]
        if len(ids_a) != len(ids_b):
            raise RuntimeError(f"pair length drift: {pair_id}")
        if Counter(ids_a) != Counter(ids_b):
            raise RuntimeError(f"pair token multiset drift: {pair_id}")
        changed = [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b]
        expected_changed = sorted(
            [
                rows[0]["role_positions"]["source_color"],
                rows[0]["role_positions"]["foil_color"],
            ]
        )
        if changed != expected_changed:
            raise RuntimeError(
                f"pair changed positions drift: {pair_id}: {changed}/{expected_changed}"
            )
        pair_audits.append(
            {
                "pair_id": pair_id,
                "changed_positions": changed,
                "same_length": True,
                "same_token_multiset": True,
            }
        )

    if any(len(lengths) != 1 for lengths in token_lengths.values()):
        raise RuntimeError(f"template token lengths drift: {token_lengths}")

    summary = {
        "schema_version": "phase998_protocol.v1",
        "phase": PHASE,
        "model": MODEL,
        "world_count": n_worlds,
        "case_count": len(cases),
        "pair_count": len(pair_rows),
        "template_count": TEMPLATE_COUNT,
        "token_lengths_by_template": {
            str(template): next(iter(lengths))
            for template, lengths in token_lengths.items()
        },
        "candidate_token_ids": candidate_ids,
        "prompt_color_token_ids": prompt_color_ids,
        "prompt_name_token_ids": prompt_name_ids,
        "all_pairs_same_token_multiset": True,
        "all_pairs_only_two_color_positions_changed": True,
        "split_counts": dict(Counter(row["split"] for row in cases)),
        "factor_counts": {
            factor: dict(Counter(str(row[factor]) for row in cases))
            for factor in ("template", "order", "query_role", "arm")
        },
        "case_manifest_sha256": sha256_text(
            "\n".join(canonical(row) for row in cases) + "\n"
        ),
        "pair_audit_sha256": sha256_text(canonical(pair_audits)),
        "cpu_protocol_pass": True,
    }
    return cases, summary


def audit_cases(cases: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        pair_groups[row["pair_id"]].append(row)
    failures = []
    for pair_id, rows in pair_groups.items():
        if len(rows) != 2:
            failures.append(f"{pair_id}: row_count={len(rows)}")
            continue
        a, b = sorted(rows, key=lambda row: row["arm"])
        if Counter(a["input_ids"]) != Counter(b["input_ids"]):
            failures.append(f"{pair_id}: token_multiset")
        if a["gold"] != b["foil"] or a["foil"] != b["gold"]:
            failures.append(f"{pair_id}: semantic_swap")
        for role in ("source_entity", "query_name", "answer_boundary"):
            if a["role_positions"][role] != b["role_positions"][role]:
                failures.append(f"{pair_id}: role_alignment/{role}")
    return {
        "schema_version": "phase998_protocol_audit.v1",
        "phase": PHASE,
        "case_count": len(cases),
        "pair_count": len(pair_groups),
        "failures": failures,
        "passed": not failures
        and summary["cpu_protocol_pass"]
        and summary["case_count"] == len(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()
    n_worlds = SMOKE_WORLDS if args.scope == "smoke" else FORMAL_WORLDS
    out = OUT_ROOT / ("smoke" if args.scope == "smoke" else "protocol")
    tokenizer = load_tokenizer()
    cases, summary = build_cases(tokenizer, n_worlds)
    audit = audit_cases(cases, summary)
    if not audit["passed"]:
        raise RuntimeError(f"protocol audit failed: {audit}")
    write_jsonl(out / "cases.jsonl", cases)
    write_json(out / "protocol.json", summary)
    write_json(out / "audit.json", audit)
    print(
        json.dumps(
            {
                "passed": True,
                "scope": args.scope,
                "case_count": len(cases),
                "pair_count": summary["pair_count"],
                "output": str(out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
