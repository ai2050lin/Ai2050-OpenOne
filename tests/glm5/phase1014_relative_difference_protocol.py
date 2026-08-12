#!/usr/bin/env python3
"""Freeze a counterbalanced protocol for relative-difference reuse mapping."""

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
    read_json,
    read_jsonl,
    role_token_positions,
    write_json,
    write_jsonl,
)


PHASE = 1014
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = (
    "comparison",
    "negation",
    "semantic_role",
    "attribute_binding",
    "spatial_relation",
)
OUTPUT_MODES = ("entity", "property", "binary")
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
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
NATURAL_STATES = ("base", "F", "Q", "FQ", "E", "O", "N", "L")
PAIR_OPERATIONS = ("F", "Q", "FQ", "E", "O", "N", "L", "I")
ANALYSIS_OPERATIONS = PAIR_OPERATIONS + ("X",)
WORLDS_PER_POOL_TEMPLATE = 4
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)
PROPERTY_LABELS = {
    "comparison": ("highest", "lowest"),
    "negation": ("present", "missing"),
    "semantic_role": ("agent", "patient"),
    "attribute_binding": ("red", "blue"),
    "spatial_relation": ("left", "right"),
}


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
        "O": None,
        "N": None,
        "L": 1,
        "I": None,
        "X": None,
    }


def add_instruction(builder: PromptBuilder, output_mode: str) -> None:
    if output_mode == "entity":
        builder.add(
            "\nReply with exactly one listed person's name and no other text."
        )
    elif output_mode == "property":
        builder.add("\nReply with exactly one property word and no other text.")
    else:
        builder.add("\nReply with exactly yes or no and no other text.")


def add_control_tag(builder: PromptBuilder, state: str) -> None:
    builder.add("\nReference tag:")
    builder.mark("control_tag", "quartz" if state == "L" else "clear")
    builder.add(".")


def add_comparison_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    high: str,
    low: str,
    high_role: str,
    low_role: str,
) -> None:
    if template == 0:
        builder.add(f"Height fact {index}:")
        builder.mark(high_role, high)
        builder.add(" is above")
        builder.mark(low_role, low)
    elif template == 1:
        builder.add(f"Rank note {index}:")
        builder.mark(high_role, high)
        builder.add(" outranks")
        builder.mark(low_role, low)
    elif template == 2:
        builder.add(f"Ordering {index}:")
        builder.mark(low_role, low)
        builder.add(" stands below")
        builder.mark(high_role, high)
    else:
        builder.add(f"Measure {index} places")
        builder.mark(low_role, low)
        builder.add(" under")
        builder.mark(high_role, high)
    builder.add(". ")


def render_comparison(
    template: int,
    names: list[str],
    state: str,
    world_index: int,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    fact_bit, query_bit = bits_for_state(world_index, state)
    high, low = (a, c) if fact_bit == 0 else (c, a)
    chain = (
        ((a, b), (b, c))
        if fact_bit == 0
        else ((c, b), (b, a))
    )
    nuisance = (d, e)
    if state == "E":
        nuisance = (e, f)
    elif state == "N":
        nuisance = (e, d)
    facts = [
        (*chain[0], "focal_high_0", "focal_low_0"),
        (*chain[1], "focal_high_1", "focal_low_1"),
        (*nuisance, "nuisance_high", "nuisance_low"),
    ]
    order = (0, 1, 2) if state != "O" else (2, 1, 0)
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, 1):
        add_comparison_fact(
            builder,
            template,
            display_index,
            *facts[fact_index],
        )
    add_control_tag(builder, state)
    if output_mode == "entity":
        requested = "highest" if query_bit == 0 else "lowest"
        builder.add("\nAmong")
        for index, name in enumerate((a, b, c)):
            builder.mark(f"query_entity_{index}", name)
            builder.add("," if index < 2 else "")
        builder.add(" return the")
        builder.mark("query_operator", requested)
        builder.add(" person.")
        gold = high if query_bit == 0 else low
        foil = low if query_bit == 0 else high
    else:
        target = a if query_bit == 0 else c
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return")
            builder.mark("query_operator", "highest")
            builder.add(" or lowest.")
            gold = "highest" if target == high else "lowest"
            foil = "lowest" if gold == "highest" else "highest"
        else:
            builder.add(",")
            builder.mark("query_operator", "is")
            builder.add(" that person highest?")
            gold = "yes" if target == high else "no"
            foil = "no" if gold == "yes" else "yes"
    add_instruction(builder, output_mode)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def add_negation_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    entity: str,
    marker: str,
    entity_role: str,
    marker_role: str,
) -> None:
    if template == 0:
        builder.add(f"Status {index}:")
        builder.mark(entity_role, entity)
        builder.add(" is")
        builder.mark(marker_role, marker)
    elif template == 1:
        builder.add(f"Roster {index} lists")
        builder.mark(entity_role, entity)
        builder.add(" as")
        builder.mark(marker_role, marker)
    elif template == 2:
        builder.mark(marker_role, marker)
        builder.add(f" in record {index}:")
        builder.mark(entity_role, entity)
    else:
        builder.add(f"Entry {index} marks")
        builder.mark(marker_role, marker)
        builder.add(":")
        builder.mark(entity_role, entity)
    builder.add(". ")


def render_negation(
    template: int,
    names: list[str],
    state: str,
    world_index: int,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    fact_bit, query_bit = bits_for_state(world_index, state)
    status = {
        a: "present" if fact_bit == 0 else "missing",
        b: "missing" if fact_bit == 0 else "present",
    }
    nuisance_entity = c
    nuisance_marker = "present"
    if state == "E":
        nuisance_entity = e
    elif state == "N":
        nuisance_marker = "missing"
    facts = [
        (a, status[a], "focal_entity_0", "focal_marker_0"),
        (b, status[b], "focal_entity_1", "focal_marker_1"),
        (
            nuisance_entity,
            nuisance_marker,
            "nuisance_entity",
            "nuisance_marker",
        ),
    ]
    order = (0, 1, 2) if state != "O" else (2, 1, 0)
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, 1):
        add_negation_fact(
            builder,
            template,
            display_index,
            *facts[fact_index],
        )
    add_control_tag(builder, state)
    if output_mode == "entity":
        requested = "present" if query_bit == 0 else "missing"
        builder.add("\nBetween")
        builder.mark("query_entity_0", a)
        builder.add(" and")
        builder.mark("query_entity_1", b)
        builder.add(", return the")
        builder.mark("query_operator", requested)
        builder.add(" person.")
        gold = a if status[a] == requested else b
        foil = b if gold == a else a
    else:
        target = a if query_bit == 0 else b
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return")
            builder.mark("query_operator", "present")
            builder.add(" or missing.")
            gold = status[target]
            foil = "missing" if gold == "present" else "present"
        else:
            builder.add(",")
            builder.mark("query_operator", "is")
            builder.add(" that person present?")
            gold = "yes" if status[target] == "present" else "no"
            foil = "no" if gold == "yes" else "yes"
    add_instruction(builder, output_mode)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def add_role_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    agent: str,
    patient: str,
    agent_role: str,
    patient_role: str,
) -> None:
    verbs = ("helped", "guided", "thanked", "called")
    verb = verbs[template]
    if template < 2:
        builder.add(f"Event {index}:")
        builder.mark(agent_role, agent)
        builder.add(f" {verb}")
        builder.mark(patient_role, patient)
    else:
        builder.add(f"Event {index}:")
        builder.mark(patient_role, patient)
        builder.add(f" was {verb} by")
        builder.mark(agent_role, agent)
    builder.add(". ")


def render_semantic_role(
    template: int,
    names: list[str],
    state: str,
    world_index: int,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    fact_bit, query_bit = bits_for_state(world_index, state)
    agent, patient = (a, b) if fact_bit == 0 else (b, a)
    nuisance = (c, d)
    if state == "E":
        nuisance = (e, f)
    elif state == "N":
        nuisance = (d, c)
    facts = [
        (agent, patient, "focal_agent", "focal_patient"),
        (*nuisance, "nuisance_agent", "nuisance_patient"),
    ]
    order = (0, 1) if state != "O" else (1, 0)
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, 1):
        add_role_fact(
            builder,
            template,
            display_index,
            *facts[fact_index],
        )
    add_control_tag(builder, state)
    if output_mode == "entity":
        requested = "agent" if query_bit == 0 else "patient"
        builder.add("\nIn Event 1, return the")
        builder.mark("query_operator", requested)
        builder.add(".")
        gold = agent if query_bit == 0 else patient
        foil = patient if query_bit == 0 else agent
    else:
        target = a if query_bit == 0 else b
        role = "agent" if target == agent else "patient"
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return")
            builder.mark("query_operator", "agent")
            builder.add(" or patient.")
            gold = role
            foil = "patient" if gold == "agent" else "agent"
        else:
            builder.add(",")
            builder.mark("query_operator", "is")
            builder.add(" that person the agent in Event 1?")
            gold = "yes" if role == "agent" else "no"
            foil = "no" if gold == "yes" else "yes"
    add_instruction(builder, output_mode)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def add_attribute_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    entity: str,
    color: str,
    entity_role: str,
    color_role: str,
) -> None:
    if template == 0:
        builder.add(f"Badge {index}:")
        builder.mark(entity_role, entity)
        builder.add(" wears")
        builder.mark(color_role, color)
    elif template == 1:
        builder.add(f"Color note {index} gives")
        builder.mark(entity_role, entity)
        builder.mark(color_role, color)
    elif template == 2:
        builder.mark(color_role, color)
        builder.add(f" is assigned in record {index} to")
        builder.mark(entity_role, entity)
    else:
        builder.add(f"Record {index} assigns")
        builder.mark(color_role, color)
        builder.add(" to")
        builder.mark(entity_role, entity)
    builder.add(". ")


def render_attribute_binding(
    template: int,
    names: list[str],
    state: str,
    world_index: int,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    fact_bit, query_bit = bits_for_state(world_index, state)
    colors = {
        a: "red" if fact_bit == 0 else "blue",
        b: "blue" if fact_bit == 0 else "red",
    }
    nuisance_entity = c
    nuisance_color = "red"
    if state == "E":
        nuisance_entity = e
    elif state == "N":
        nuisance_color = "blue"
    facts = [
        (a, colors[a], "focal_entity_0", "focal_color_0"),
        (b, colors[b], "focal_entity_1", "focal_color_1"),
        (
            nuisance_entity,
            nuisance_color,
            "nuisance_entity",
            "nuisance_color",
        ),
    ]
    order = (0, 1, 2) if state != "O" else (2, 1, 0)
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, 1):
        add_attribute_fact(
            builder,
            template,
            display_index,
            *facts[fact_index],
        )
    add_control_tag(builder, state)
    if output_mode == "entity":
        requested = "red" if query_bit == 0 else "blue"
        builder.add("\nBetween")
        builder.mark("query_entity_0", a)
        builder.add(" and")
        builder.mark("query_entity_1", b)
        builder.add(", return the")
        builder.mark("query_operator", requested)
        builder.add(" person.")
        gold = a if colors[a] == requested else b
        foil = b if gold == a else a
    else:
        target = a if query_bit == 0 else b
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return")
            builder.mark("query_operator", "red")
            builder.add(" or blue.")
            gold = colors[target]
            foil = "blue" if gold == "red" else "red"
        else:
            builder.add(",")
            builder.mark("query_operator", "is")
            builder.add(" that person's badge red?")
            gold = "yes" if colors[target] == "red" else "no"
            foil = "no" if gold == "yes" else "yes"
    add_instruction(builder, output_mode)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def add_spatial_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    left: str,
    right: str,
    left_role: str,
    right_role: str,
) -> None:
    if template == 0:
        builder.add(f"Position {index}:")
        builder.mark(left_role, left)
        builder.add(" is left of")
        builder.mark(right_role, right)
    elif template == 1:
        builder.add(f"Layout {index} puts")
        builder.mark(left_role, left)
        builder.add(" left of")
        builder.mark(right_role, right)
    elif template == 2:
        builder.add(f"Position {index}:")
        builder.mark(right_role, right)
        builder.add(" is right of")
        builder.mark(left_role, left)
    else:
        builder.add(f"Layout {index} puts")
        builder.mark(right_role, right)
        builder.add(" right of")
        builder.mark(left_role, left)
    builder.add(". ")


def render_spatial_relation(
    template: int,
    names: list[str],
    state: str,
    world_index: int,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    fact_bit, query_bit = bits_for_state(world_index, state)
    left, right = (a, b) if fact_bit == 0 else (b, a)
    nuisance = (c, d)
    if state == "E":
        nuisance = (e, f)
    elif state == "N":
        nuisance = (d, c)
    facts = [
        (left, right, "focal_left", "focal_right"),
        (*nuisance, "nuisance_left", "nuisance_right"),
    ]
    order = (0, 1) if state != "O" else (1, 0)
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, 1):
        add_spatial_fact(
            builder,
            template,
            display_index,
            *facts[fact_index],
        )
    add_control_tag(builder, state)
    if output_mode == "entity":
        requested = "left" if query_bit == 0 else "right"
        builder.add("\nIn Position 1, return the")
        builder.mark("query_operator", requested)
        builder.add(" person.")
        gold = left if query_bit == 0 else right
        foil = right if query_bit == 0 else left
    else:
        target = a if query_bit == 0 else b
        relation = "left" if target == left else "right"
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return")
            builder.mark("query_operator", "left")
            builder.add(" or right.")
            gold = relation
            foil = "right" if gold == "left" else "left"
        else:
            builder.add(",")
            builder.mark("query_operator", "is")
            builder.add(" that person left in Position 1?")
            gold = "yes" if relation == "left" else "no"
            foil = "no" if gold == "yes" else "yes"
    add_instruction(builder, output_mode)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def render_family(
    family: str,
    template: int,
    names: list[str],
    state: str,
    world_index: int,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    functions = {
        "comparison": render_comparison,
        "negation": render_negation,
        "semantic_role": render_semantic_role,
        "attribute_binding": render_attribute_binding,
        "spatial_relation": render_spatial_relation,
    }
    return functions[family](
        template,
        names,
        state,
        world_index,
        output_mode,
    )


def candidate_labels(
    family: str,
    output_mode: str,
    names: list[str],
) -> list[str]:
    if output_mode == "entity":
        return list(names)
    if output_mode == "property":
        return list(PROPERTY_LABELS[family])
    return ["yes", "no"]


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


def role_class(role: str) -> str:
    if role == "answer_boundary":
        return role
    if role == "control_tag":
        return "lexical_control"
    if role.startswith("query_"):
        return "query"
    if role.startswith("nuisance_"):
        return "nuisance"
    if any(
        marker in role
        for marker in (
            "marker",
            "color",
            "operator",
        )
    ):
        return "focal_operator"
    return "focal_source"


def build_case(
    *,
    tokenizer,
    model_name: str,
    family: str,
    output_mode: str,
    split: str,
    template: int,
    name_pool: int,
    world_index: int,
    unit_id: str,
    state: str,
    names: list[str],
) -> dict[str, Any]:
    raw_prompt, spans, gold, foil = render_family(
        family,
        template,
        names,
        state,
        world_index,
        output_mode,
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
    roles = {
        role: role_class(role)
        for role in positions
    }
    labels = candidate_labels(family, output_mode, names)
    candidate_ids = {
        label: boundary_token_id(tokenizer, rendered, label)
        for label in labels
    }
    if len(set(candidate_ids.values())) != len(candidate_ids):
        raise RuntimeError("candidate token collision")
    fact_bit, query_bit = bits_for_state(world_index, state)
    return {
        "schema_version": "phase1014_relative_case.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "family": family,
        "output_mode": output_mode,
        "split": split,
        "template": int(template),
        "name_pool": int(name_pool),
        "world_index": int(world_index),
        "unit_id": unit_id,
        "record_id": f"{unit_id}.{state}",
        "state": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(position)
            for role, position in positions.items()
        },
        "role_classes": roles,
        "gold": gold,
        "foil": foil,
        "candidate_labels": labels,
        "candidate_token_ids": candidate_ids,
        "answer_text": " " + gold,
        "natural_gold_text": gold,
        "answer_token_ids": [candidate_ids[gold]],
        "semantic_step": 0,
        "protocol_prefix_ids": [],
        "fact_bit": fact_bit,
        "query_bit": query_bit,
        "answer_slot_bit": fact_bit ^ query_bit,
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
        for output_mode in OUTPUT_MODES:
            for split in SPLITS:
                for template in TEMPLATES_BY_SPLIT[split]:
                    for name_pool, pool in enumerate(NAME_POOLS[split]):
                        for world_index in range(
                            WORLDS_PER_POOL_TEMPLATE
                        ):
                            names = list(pool)
                            shift = world_index % len(names)
                            names = names[shift:] + names[:shift]
                            unit_id = (
                                f"{model_name}.{family}.{output_mode}."
                                f"{split}.t{template}.p{name_pool}."
                                f"w{world_index}"
                            )
                            state_cases = {}
                            case_ids = {}
                            for state in NATURAL_STATES:
                                case = build_case(
                                    tokenizer=tokenizer,
                                    model_name=model_name,
                                    family=family,
                                    output_mode=output_mode,
                                    split=split,
                                    template=template,
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
                            if not all(
                                case["role_positions"]["answer_boundary"]
                                == base["role_positions"]["answer_boundary"]
                                for case in state_cases.values()
                            ):
                                raise RuntimeError(
                                    f"{unit_id}: boundary position drift"
                                )
                            base_gold = base["gold"]
                            for operation in ("F", "Q"):
                                if state_cases[operation]["gold"] == base_gold:
                                    raise RuntimeError(
                                        f"{unit_id}: {operation} did not "
                                        "flip answer"
                                    )
                            for operation in (
                                "FQ",
                                "E",
                                "O",
                                "N",
                                "L",
                            ):
                                if state_cases[operation]["gold"] != base_gold:
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
                            if len(edits["L"]) != 1:
                                raise RuntimeError(
                                    f"{unit_id}: lexical control edit "
                                    f"count {len(edits['L'])}"
                                )
                            cases.extend(state_cases.values())
                            signs = factor_signs(world_index)
                            units.append({
                                "schema_version": (
                                    "phase1014_relative_unit.v1"
                                ),
                                "phase": PHASE,
                                "protocol_revision": PROTOCOL_REVISION,
                                "model": model_name,
                                "family": family,
                                "output_mode": output_mode,
                                "split": split,
                                "template": int(template),
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
                                "canonical_factor_signs": signs,
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
        "schema_version": "phase1014_relative_model_protocol.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(units),
        "equal_state_length_unit_count": len(units),
        "single_position_lexical_control_unit_count": sum(
            row["edit_counts"]["L"] == 1 for row in units
        ),
        "counterbalance_counts": dict(Counter(
            row["counterbalance_cell"] for row in units
        )),
        "panel_counts": dict(Counter(
            f"{row['family']}:{row['output_mode']}"
            for row in units
        )),
        "all_answers_single_token_at_boundary": True,
        "explicit_response_map_case_count": 0,
    }
    write_json(model_root / "summary.json", summary)
    return summary


def main() -> None:
    summaries = [build_model(model) for model in MODELS]
    protocol = {
        "schema_version": "phase1014_relative_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "principle": (
            "map stable relative-difference reuse before proposing a "
            "mechanism formula or causal edge"
        ),
        "models_in_required_execution_order": list(MODELS),
        "families": list(FAMILIES),
        "output_modes": list(OUTPUT_MODES),
        "splits": list(SPLITS),
        "templates_by_split": {
            split: list(values)
            for split, values in TEMPLATES_BY_SPLIT.items()
        },
        "name_pools": {
            split: [list(pool) for pool in pools]
            for split, pools in NAME_POOLS.items()
        },
        "natural_states": list(NATURAL_STATES),
        "pair_operations": list(PAIR_OPERATIONS),
        "analysis_operations": list(ANALYSIS_OPERATIONS),
        "counterbalance": {
            "worlds": ["f0q0", "f1q0", "f0q1", "f1q1"],
            "F": "toggle fact bit; orient differences to fact 0->1",
            "Q": "toggle query bit; orient differences to query 0->1",
            "FQ": "toggle both bits; correct answer remains invariant",
            "answer_slot": "fact_bit XOR query_bit",
            "purpose": (
                "counterbalance answer direction and lexical transition "
                "while preserving an abstract factor orientation"
            ),
        },
        "controls": {
            "E": "nuisance entity replacement",
            "O": "fact order change",
            "N": "nuisance fact change",
            "L": (
                "single-position irrelevant lexical replacement "
                "clear->quartz"
            ),
            "I": "identity repeat",
        },
        "selection_contract": {
            "discovery_selects": True,
            "confirmation_never_selects": True,
            "all_model_states_use_singleton_forward": True,
            "no_fixed_causal_success_rate": True,
            "no_edge_is_inferred_from_temporal_order": True,
            "formulas_are_measurement_definitions_only": True,
        },
        "claim_limits": [
            "canonical direction recurrence is a relative-difference "
            "candidate, not proof of a semantic axis",
            "cross-panel reuse is not transport or causality",
            "same head number across models has no physical meaning",
            "the protocol tests five controlled language-pattern families, "
            "not open language",
        ],
        "model_summaries": summaries,
    }
    protocol["preregistration_digest"] = digest(protocol)
    write_json(OUT_ROOT / "protocol" / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
