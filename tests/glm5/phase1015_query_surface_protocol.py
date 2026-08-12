#!/usr/bin/env python3
"""Freeze the Phase1015 query-surface and full-chain mapping protocol."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1009_crossfamily_response_protocol import (
    PromptBuilder,
    canonical,
    digest,
    read_jsonl,
    role_token_positions,
    write_json,
    write_jsonl,
)


PHASE = 1015
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = (
    "comparison",
    "negation",
    "semantic_role",
    "attribute_binding",
    "spatial_relation",
)
OUTPUT_MODE = "entity"
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
SURFACES_BY_SPLIT = {
    "discovery": (0, 1, 2),
    "confirmation": (3, 4, 5),
}
SURFACE_METADATA = {
    0: {
        "name": "direct_primary",
        "class": "natural_direct",
        "balanced_inventory": False,
    },
    1: {
        "name": "interrogative_synonym",
        "class": "natural_synonym",
        "balanced_inventory": False,
    },
    2: {
        "name": "ordered_first_second",
        "class": "balanced_inventory",
        "balanced_inventory": True,
    },
    3: {
        "name": "relative_synonym",
        "class": "natural_synonym_heldout",
        "balanced_inventory": False,
    },
    4: {
        "name": "negated_antonym",
        "class": "negated_heldout",
        "balanced_inventory": False,
    },
    5: {
        "name": "ordered_former_latter",
        "class": "balanced_inventory_heldout",
        "balanced_inventory": True,
    },
}
NAME_POOLS = {
    "discovery": (
        ("Paul", "Peter", "Philip", "Ray", "Robert", "Ryan"),
        ("Sam", "Scott", "Sean", "Simon", "Stephen", "Steve"),
        ("Thomas", "Tim", "Victor", "Walter", "William", "Zach"),
    ),
    "confirmation": (
        ("Adam", "Brian", "Colin", "Daniel", "Eric", "Frank"),
        ("George", "Henry", "Ian", "Jack", "James", "Jason"),
        ("Jeff", "John", "Kevin", "Mark", "Martin", "Mike"),
    ),
}
PRIMARY_LABELS = {
    "comparison": ("highest", "lowest"),
    "negation": ("present", "missing"),
    "semantic_role": ("agent", "patient"),
    "attribute_binding": ("red", "blue"),
    "spatial_relation": ("left", "right"),
}
SYNONYM_LABELS = {
    "comparison": ("tallest", "shortest"),
    "negation": ("listed", "absent"),
    "semantic_role": ("actor", "recipient"),
    "attribute_binding": ("crimson", "azure"),
    "spatial_relation": ("western", "eastern"),
}
HELDOUT_LABELS = {
    "comparison": ("upper", "lower"),
    "negation": ("available", "absent"),
    "semantic_role": ("source", "receiver"),
    "attribute_binding": ("ruby", "navy"),
    "spatial_relation": ("west", "east"),
}
NATURAL_STATES = ("base", "F", "Q", "FQ", "E", "N", "L")
PAIR_OPERATIONS = ("F", "Q", "FQ", "E", "N", "L", "I")
ANALYSIS_OPERATIONS = PAIR_OPERATIONS + ("X",)
CAPTURE_ROLES = (
    "fact_source",
    "fact_relation",
    "fact_target",
    "lexical_control",
    "query_anchor",
    "query_operator",
    "answer_boundary",
)
WORLDS_PER_POOL_TEMPLATE = 4
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1015_query_surface_chain_atlas"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bits_for_state(world_index: int, state: str) -> tuple[int, int]:
    fact_bit = world_index & 1
    query_bit = (world_index >> 1) & 1
    if state in {"F", "FQ"}:
        fact_bit ^= 1
    if state in {"Q", "FQ"}:
        query_bit ^= 1
    return fact_bit, query_bit


def factor_signs(world_index: int) -> dict[str, int | None]:
    fact_bit = world_index & 1
    query_bit = (world_index >> 1) & 1
    return {
        "F": 1 if fact_bit == 0 else -1,
        "Q": 1 if query_bit == 0 else -1,
        "FQ": None,
        "E": None,
        "N": None,
        "L": 1,
        "I": None,
        "X": None,
    }


def add_plain_token(builder: PromptBuilder, value: str) -> None:
    builder.add(" " + value)


def add_control_tag(builder: PromptBuilder, state: str) -> None:
    builder.add("\nReference tag:")
    builder.mark("lexical_control", "quartz" if state == "L" else "clear")
    builder.add(".")


def add_comparison_facts(
    builder: PromptBuilder,
    template: int,
    names: list[str],
    fact_bit: int,
    state: str,
) -> tuple[str, str]:
    a, b, c, d, e, f = names
    high, low = (a, c) if fact_bit == 0 else (c, a)
    nuisance_high, nuisance_low = d, e
    if state == "E":
        nuisance_high, nuisance_low = e, f
    elif state == "N":
        nuisance_high, nuisance_low = e, d

    def nuisance() -> None:
        builder.add("Background:")
        add_plain_token(builder, nuisance_high)
        builder.add(" is above")
        add_plain_token(builder, nuisance_low)
        builder.add(". ")

    if state == "O":
        nuisance()
    if template == 0:
        builder.add("Height fact 1:")
        builder.mark("fact_source", high)
        builder.add(" is")
        builder.mark("fact_relation", "above")
        add_plain_token(builder, b)
        builder.add(". Height fact 2:")
        add_plain_token(builder, b)
        builder.add(" is above")
        builder.mark("fact_target", low)
        builder.add(". ")
    elif template == 1:
        builder.add("Rank note 1:")
        builder.mark("fact_source", high)
        builder.mark("fact_relation", "leads")
        add_plain_token(builder, b)
        builder.add(". Rank note 2:")
        add_plain_token(builder, b)
        builder.add(" outranks")
        builder.mark("fact_target", low)
        builder.add(". ")
    elif template == 2:
        builder.add("Ordering 1:")
        add_plain_token(builder, b)
        builder.add(" stands")
        builder.mark("fact_relation", "below")
        builder.mark("fact_source", high)
        builder.add(". Ordering 2:")
        builder.mark("fact_target", low)
        builder.add(" stands below")
        add_plain_token(builder, b)
        builder.add(". ")
    else:
        builder.add("Measure 1 places")
        add_plain_token(builder, b)
        builder.mark("fact_relation", "under")
        builder.mark("fact_source", high)
        builder.add(". Measure 2 places")
        builder.mark("fact_target", low)
        builder.add(" under")
        add_plain_token(builder, b)
        builder.add(". ")
    if state != "O":
        nuisance()
    return high, low


def add_negation_facts(
    builder: PromptBuilder,
    template: int,
    names: list[str],
    fact_bit: int,
    state: str,
) -> tuple[str, str]:
    a, b, c, _, e, _ = names
    status_a = "present" if fact_bit == 0 else "missing"
    status_b = "missing" if fact_bit == 0 else "present"
    nuisance_name = e if state == "E" else c
    nuisance_status = "missing" if state == "N" else "present"

    def nuisance() -> None:
        builder.add("Background:")
        add_plain_token(builder, nuisance_name)
        builder.add(" is")
        add_plain_token(builder, nuisance_status)
        builder.add(". ")

    if state == "O":
        nuisance()
    if template == 0:
        builder.add("Status 1:")
        builder.mark("fact_source", a)
        builder.add(" is")
        builder.mark("fact_relation", status_a)
        builder.add(". Status 2:")
        builder.mark("fact_target", b)
        builder.add(" is")
        add_plain_token(builder, status_b)
        builder.add(". ")
    elif template == 1:
        builder.add("Roster 1 lists")
        builder.mark("fact_source", a)
        builder.add(" as")
        builder.mark("fact_relation", status_a)
        builder.add(". Roster 2 lists")
        builder.mark("fact_target", b)
        builder.add(" as")
        add_plain_token(builder, status_b)
        builder.add(". ")
    elif template == 2:
        builder.mark("fact_relation", status_a)
        builder.add(" in record 1:")
        builder.mark("fact_source", a)
        builder.add(".")
        add_plain_token(builder, status_b)
        builder.add(" in record 2:")
        builder.mark("fact_target", b)
        builder.add(". ")
    else:
        builder.add("Entry 1 marks")
        builder.mark("fact_relation", status_a)
        builder.add(":")
        builder.mark("fact_source", a)
        builder.add(". Entry 2 marks")
        add_plain_token(builder, status_b)
        builder.add(":")
        builder.mark("fact_target", b)
        builder.add(". ")
    if state != "O":
        nuisance()
    present = a if status_a == "present" else b
    missing = b if present == a else a
    return present, missing


def add_role_facts(
    builder: PromptBuilder,
    template: int,
    names: list[str],
    fact_bit: int,
    state: str,
) -> tuple[str, str]:
    a, b, c, d, e, f = names
    agent, patient = (a, b) if fact_bit == 0 else (b, a)
    nuisance_agent, nuisance_patient = c, d
    if state == "E":
        nuisance_agent, nuisance_patient = e, f
    elif state == "N":
        nuisance_agent, nuisance_patient = d, c
    verbs = ("helped", "guided", "thanked", "called")
    verb = verbs[template]

    def nuisance() -> None:
        builder.add("Background event:")
        add_plain_token(builder, nuisance_agent)
        builder.add(" helped")
        add_plain_token(builder, nuisance_patient)
        builder.add(". ")

    if state == "O":
        nuisance()
    builder.add("Focal event:")
    if template < 2:
        builder.mark("fact_source", agent)
        builder.mark("fact_relation", verb)
        builder.mark("fact_target", patient)
    else:
        builder.mark("fact_target", patient)
        builder.add(" was")
        builder.mark("fact_relation", verb)
        builder.add(" by")
        builder.mark("fact_source", agent)
    builder.add(". ")
    if state != "O":
        nuisance()
    return agent, patient


def add_attribute_facts(
    builder: PromptBuilder,
    template: int,
    names: list[str],
    fact_bit: int,
    state: str,
) -> tuple[str, str]:
    a, b, c, _, e, _ = names
    color_a = "red" if fact_bit == 0 else "blue"
    color_b = "blue" if fact_bit == 0 else "red"
    nuisance_name = e if state == "E" else c
    nuisance_color = "blue" if state == "N" else "red"

    def nuisance() -> None:
        builder.add("Background badge:")
        add_plain_token(builder, nuisance_name)
        builder.add(" wears")
        add_plain_token(builder, nuisance_color)
        builder.add(". ")

    if state == "O":
        nuisance()
    if template == 0:
        builder.add("Badge 1:")
        builder.mark("fact_source", a)
        builder.add(" wears")
        builder.mark("fact_relation", color_a)
        builder.add(". Badge 2:")
        builder.mark("fact_target", b)
        builder.add(" wears")
        add_plain_token(builder, color_b)
        builder.add(". ")
    elif template == 1:
        builder.add("Color note 1 gives")
        builder.mark("fact_source", a)
        builder.mark("fact_relation", color_a)
        builder.add(". Color note 2 gives")
        builder.mark("fact_target", b)
        add_plain_token(builder, color_b)
        builder.add(". ")
    elif template == 2:
        builder.mark("fact_relation", color_a)
        builder.add(" is assigned in record 1 to")
        builder.mark("fact_source", a)
        builder.add(".")
        add_plain_token(builder, color_b)
        builder.add(" is assigned in record 2 to")
        builder.mark("fact_target", b)
        builder.add(". ")
    else:
        builder.add("Record 1 assigns")
        builder.mark("fact_relation", color_a)
        builder.add(" to")
        builder.mark("fact_source", a)
        builder.add(". Record 2 assigns")
        add_plain_token(builder, color_b)
        builder.add(" to")
        builder.mark("fact_target", b)
        builder.add(". ")
    if state != "O":
        nuisance()
    red = a if color_a == "red" else b
    blue = b if red == a else a
    return red, blue


def add_spatial_facts(
    builder: PromptBuilder,
    template: int,
    names: list[str],
    fact_bit: int,
    state: str,
) -> tuple[str, str]:
    a, b, c, d, e, f = names
    left, right = (a, b) if fact_bit == 0 else (b, a)
    nuisance_left, nuisance_right = c, d
    if state == "E":
        nuisance_left, nuisance_right = e, f
    elif state == "N":
        nuisance_left, nuisance_right = d, c

    def nuisance() -> None:
        builder.add("Background position:")
        add_plain_token(builder, nuisance_left)
        builder.add(" is left of")
        add_plain_token(builder, nuisance_right)
        builder.add(". ")

    if state == "O":
        nuisance()
    if template < 2:
        builder.add("Focal position:")
        builder.mark("fact_source", left)
        builder.add(" is")
        builder.mark("fact_relation", "left")
        builder.add(" of")
        builder.mark("fact_target", right)
        builder.add(". ")
    else:
        builder.add("Focal position:")
        builder.mark("fact_target", right)
        builder.add(" is")
        builder.mark("fact_relation", "right")
        builder.add(" of")
        builder.mark("fact_source", left)
        builder.add(". ")
    if state != "O":
        nuisance()
    return left, right


def add_facts(
    builder: PromptBuilder,
    family: str,
    template: int,
    names: list[str],
    fact_bit: int,
    state: str,
) -> tuple[str, str]:
    functions = {
        "comparison": add_comparison_facts,
        "negation": add_negation_facts,
        "semantic_role": add_role_facts,
        "attribute_binding": add_attribute_facts,
        "spatial_relation": add_spatial_facts,
    }
    return functions[family](builder, template, names, fact_bit, state)


def add_query(
    builder: PromptBuilder,
    *,
    family: str,
    surface: int,
    query_bit: int,
    names: list[str],
) -> None:
    a, b, c, *_ = names
    builder.add("\nCandidates:")
    builder.mark("query_anchor", a)
    add_plain_token(builder, b)
    add_plain_token(builder, c)
    builder.add(".")
    primary = PRIMARY_LABELS[family]
    synonym = SYNONYM_LABELS[family]
    heldout = HELDOUT_LABELS[family]
    if surface == 0:
        builder.add("\nReturn the")
        builder.mark("query_operator", primary[query_bit])
        builder.add(" candidate.")
    elif surface == 1:
        builder.add("\nWhich candidate is")
        builder.mark("query_operator", synonym[query_bit])
        builder.add("? Return that name.")
    elif surface == 2:
        builder.add(
            f"\nThe ordered properties are {primary[0]} and "
            f"{primary[1]}. Use the"
        )
        builder.mark(
            "query_operator",
            "first" if query_bit == 0 else "second",
        )
        builder.add(" property and return its candidate.")
    elif surface == 3:
        builder.add("\nSelect the candidate on the")
        builder.mark("query_operator", heldout[query_bit])
        builder.add(" side of the focal relation.")
    elif surface == 4:
        builder.add("\nReturn the candidate that is not")
        builder.mark("query_operator", primary[query_bit ^ 1])
        builder.add(".")
    elif surface == 5:
        builder.add(
            f"\nRead the ordered pair {primary[0]}, {primary[1]}. "
            "Return the candidate matching the"
        )
        builder.mark(
            "query_operator",
            "former" if query_bit == 0 else "latter",
        )
        builder.add(" item.")
    else:
        raise KeyError(surface)
    builder.add(
        "\nReply with exactly one listed person's name and no other text."
    )


def render_case(
    *,
    family: str,
    template: int,
    surface: int,
    names: list[str],
    state: str,
    world_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    fact_bit, query_bit = bits_for_state(world_index, state)
    builder = PromptBuilder()
    relation_zero, relation_one = add_facts(
        builder,
        family,
        template,
        names,
        fact_bit,
        state,
    )
    add_control_tag(builder, state)
    add_query(
        builder,
        family=family,
        surface=surface,
        query_bit=query_bit,
        names=names,
    )
    prompt, spans = builder.finish()
    gold = relation_zero if query_bit == 0 else relation_one
    foil = relation_one if query_bit == 0 else relation_zero
    return prompt, spans, gold, foil


def boundary_token_id(tokenizer, rendered: str, label: str) -> int:
    base = tokenizer.encode(rendered, add_special_tokens=False)
    extended = tokenizer.encode(
        rendered + " " + label,
        add_special_tokens=False,
    )
    if extended[:len(base)] != base or len(extended) != len(base) + 1:
        raise RuntimeError(
            f"answer {label!r} is not one token at assistant boundary"
        )
    return int(extended[-1])


def build_case(
    *,
    tokenizer,
    model_name: str,
    family: str,
    split: str,
    template: int,
    surface: int,
    name_pool: int,
    world_index: int,
    unit_id: str,
    state: str,
    names: list[str],
) -> dict[str, Any]:
    raw_prompt, spans, gold, foil = render_case(
        family=family,
        template=template,
        surface=surface,
        names=names,
        state=state,
        world_index=world_index,
    )
    rendered = render_chat(tokenizer, model_name, raw_prompt)
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    positions = role_token_positions(
        tokenizer,
        rendered,
        raw_prompt,
        spans,
    )
    positions["answer_boundary"] = len(input_ids) - 1
    if set(positions) != set(CAPTURE_ROLES):
        raise RuntimeError(
            f"{family}/surface{surface}: role drift "
            f"{sorted(positions)}"
        )
    candidate_ids = {
        label: boundary_token_id(tokenizer, rendered, label)
        for label in names
    }
    if len(set(candidate_ids.values())) != len(candidate_ids):
        raise RuntimeError("candidate token collision")
    fact_bit, query_bit = bits_for_state(world_index, state)
    return {
        "schema_version": "phase1015_query_surface_case.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "family": family,
        "output_mode": OUTPUT_MODE,
        "split": split,
        "template": int(template),
        "query_surface": int(surface),
        "query_surface_name": SURFACE_METADATA[surface]["name"],
        "query_surface_class": SURFACE_METADATA[surface]["class"],
        "balanced_query_inventory": bool(
            SURFACE_METADATA[surface]["balanced_inventory"]
        ),
        "name_pool": int(name_pool),
        "world_index": int(world_index),
        "unit_id": unit_id,
        "record_id": f"{unit_id}.{state}",
        "state": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(positions[role]) for role in CAPTURE_ROLES
        },
        "gold": gold,
        "foil": foil,
        "candidate_labels": list(names),
        "candidate_token_ids": candidate_ids,
        "answer_text": " " + gold,
        "natural_gold_text": gold,
        "answer_token_ids": [candidate_ids[gold]],
        "semantic_step": 0,
        "protocol_prefix_ids": [],
        "fact_bit": int(fact_bit),
        "query_bit": int(query_bit),
        "answer_slot_bit": int(fact_bit ^ query_bit),
        "explicit_response_map_present": False,
    }


def edit_positions(base: list[int], variant: list[int]) -> list[int]:
    if len(base) != len(variant):
        raise RuntimeError(
            f"counterbalanced length drift {len(base)} != {len(variant)}"
        )
    return [
        index
        for index, (left, right) in enumerate(zip(base, variant))
        if left != right
    ]


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    cases: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for family in FAMILIES:
        for split in SPLITS:
            for surface in SURFACES_BY_SPLIT[split]:
                for template in TEMPLATES_BY_SPLIT[split]:
                    for name_pool, pool in enumerate(NAME_POOLS[split]):
                        for world_index in range(
                            WORLDS_PER_POOL_TEMPLATE
                        ):
                            names = list(pool)
                            shift = world_index % len(names)
                            names = names[shift:] + names[:shift]
                            unit_id = (
                                f"{model_name}.{family}."
                                f"{split}.s{surface}.t{template}."
                                f"p{name_pool}.w{world_index}"
                            )
                            state_cases = {}
                            case_ids = {}
                            for state in NATURAL_STATES:
                                case = build_case(
                                    tokenizer=tokenizer,
                                    model_name=model_name,
                                    family=family,
                                    split=split,
                                    template=template,
                                    surface=surface,
                                    name_pool=name_pool,
                                    world_index=world_index,
                                    unit_id=unit_id,
                                    state=state,
                                    names=names,
                                )
                                state_cases[state] = case
                                case_ids[state] = case["record_id"]
                            base = state_cases["base"]
                            lengths = {
                                state: len(case["input_ids"])
                                for state, case in state_cases.items()
                            }
                            if len(set(lengths.values())) != 1:
                                raise RuntimeError(
                                    f"{unit_id}: state length drift {lengths}"
                                )
                            for role in (
                                "lexical_control",
                                "query_anchor",
                                "query_operator",
                                "answer_boundary",
                            ):
                                if not all(
                                    case["role_positions"][role]
                                    == base["role_positions"][role]
                                    for case in state_cases.values()
                                ):
                                    raise RuntimeError(
                                        f"{unit_id}: {role} position drift"
                                    )
                            for operation in ("F", "Q"):
                                if (
                                    state_cases[operation]["gold"]
                                    == base["gold"]
                                ):
                                    raise RuntimeError(
                                        f"{unit_id}: {operation} did not "
                                        "flip answer"
                                    )
                            for operation in (
                                "FQ",
                                "E",
                                "N",
                                "L",
                            ):
                                if (
                                    state_cases[operation]["gold"]
                                    != base["gold"]
                                ):
                                    raise RuntimeError(
                                        f"{unit_id}: {operation} answer "
                                        "invariance failed"
                                    )
                            edits = {
                                operation: edit_positions(
                                    base["input_ids"],
                                    (
                                        base["input_ids"]
                                        if operation == "I"
                                        else state_cases[operation][
                                            "input_ids"
                                        ]
                                    ),
                                )
                                for operation in PAIR_OPERATIONS
                            }
                            query_position = base["role_positions"][
                                "query_operator"
                            ]
                            if edits["Q"] != [query_position]:
                                raise RuntimeError(
                                    f"{unit_id}: Q edit is not isolated "
                                    f"{edits['Q']} != [{query_position}]"
                                )
                            if len(edits["L"]) != 1:
                                raise RuntimeError(
                                    f"{unit_id}: lexical control edit "
                                    f"count {len(edits['L'])}"
                                )
                            if any(
                                base["input_ids"][index]
                                != state_cases["Q"]["input_ids"][index]
                                for index in range(query_position)
                            ):
                                raise RuntimeError(
                                    f"{unit_id}: Q changed causal prefix"
                                )
                            cases.extend(state_cases.values())
                            units.append({
                                "schema_version": (
                                    "phase1015_query_surface_unit.v1"
                                ),
                                "phase": PHASE,
                                "protocol_revision": PROTOCOL_REVISION,
                                "model": model_name,
                                "family": family,
                                "output_mode": OUTPUT_MODE,
                                "split": split,
                                "template": int(template),
                                "query_surface": int(surface),
                                "query_surface_name": (
                                    SURFACE_METADATA[surface]["name"]
                                ),
                                "query_surface_class": (
                                    SURFACE_METADATA[surface]["class"]
                                ),
                                "balanced_query_inventory": bool(
                                    SURFACE_METADATA[surface][
                                        "balanced_inventory"
                                    ]
                                ),
                                "name_pool": int(name_pool),
                                "world_index": int(world_index),
                                "base_fact_bit": world_index & 1,
                                "base_query_bit": (
                                    (world_index >> 1) & 1
                                ),
                                "counterbalance_cell": (
                                    f"f{world_index & 1}"
                                    f"q{(world_index >> 1) & 1}"
                                ),
                                "unit_id": unit_id,
                                "case_ids": case_ids,
                                "canonical_factor_signs": factor_signs(
                                    world_index
                                ),
                                "edit_positions": edits,
                                "edit_counts": {
                                    key: len(value)
                                    for key, value in edits.items()
                                },
                                "operation_pairs": {
                                    operation: {
                                        "base": case_ids["base"],
                                        "variant": (
                                            case_ids["base"]
                                            if operation == "I"
                                            else case_ids[operation]
                                        ),
                                    }
                                    for operation in PAIR_OPERATIONS
                                },
                            })
    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "units.jsonl", units)
    summary = {
        "schema_version": "phase1015_query_surface_model_protocol.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(units),
        "equal_state_length_unit_count": len(units),
        "single_token_query_edit_unit_count": sum(
            row["edit_counts"]["Q"] == 1 for row in units
        ),
        "single_token_lexical_control_unit_count": sum(
            row["edit_counts"]["L"] == 1 for row in units
        ),
        "balanced_inventory_unit_count": sum(
            row["balanced_query_inventory"] for row in units
        ),
        "counterbalance_counts": dict(Counter(
            row["counterbalance_cell"] for row in units
        )),
        "family_counts": dict(Counter(
            row["family"] for row in units
        )),
        "surface_counts": dict(Counter(
            str(row["query_surface"]) for row in units
        )),
        "split_counts": dict(Counter(
            row["split"] for row in units
        )),
        "all_answers_single_token_at_boundary": True,
        "explicit_response_map_case_count": 0,
    }
    write_json(model_root / "summary.json", summary)
    return summary


def main() -> None:
    summaries = [build_model(model) for model in MODELS]
    protocol = {
        "schema_version": "phase1015_query_surface_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "principle": (
            "discover stable repeated relative-difference structure "
            "before proposing a mechanism formula"
        ),
        "families": list(FAMILIES),
        "output_mode": OUTPUT_MODE,
        "splits": list(SPLITS),
        "templates_by_split": {
            key: list(value)
            for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "surfaces_by_split": {
            key: list(value)
            for key, value in SURFACES_BY_SPLIT.items()
        },
        "surface_metadata": SURFACE_METADATA,
        "capture_roles": list(CAPTURE_ROLES),
        "states": list(NATURAL_STATES),
        "pair_operations": list(PAIR_OPERATIONS),
        "analysis_operations": list(ANALYSIS_OPERATIONS),
        "target_operations": ["F", "Q"],
        "models": list(MODELS),
        "model_summaries": summaries,
        "preregistered_claim_limits": [
            "cross-surface recurrence weakens a token-specific shortcut "
            "but does not prove an abstract query variable",
            "ordered-inventory surfaces are diagnostic controls, not "
            "evidence of a natural internal response map",
            "ordered role co-response is not a transport edge",
            "direction and membership are separate evidence axes",
            "operational thresholds are rulers, not mechanism formulas",
        ],
    }
    protocol["preregistration_digest"] = digest(canonical(protocol))
    write_json(OUT_ROOT / "protocol" / "protocol.json", protocol)
    frozen = read_jsonl(
        OUT_ROOT / "protocol" / MODELS[0] / "units.jsonl"
    )
    print(json.dumps({
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "digest": protocol["preregistration_digest"],
        "units_per_model": len(frozen),
        "cases_per_model": summaries[0]["case_count"],
        "models": summaries,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
