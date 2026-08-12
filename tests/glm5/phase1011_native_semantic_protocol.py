#!/usr/bin/env python3
"""Freeze Phase1011 native-output tasks without an explicit response map."""
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
from phase1008_global_response_atlas_protocol import NAME_POOLS
from phase1009_crossfamily_response_protocol import (
    PromptBuilder,
    comparison_fact,
    negation_fact,
    role_token_positions,
    semantic_role_fact,
    canonical,
    digest,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PHASE = 1011
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("comparison", "negation", "semantic_role")
OUTPUT_MODES = ("entity", "property", "binary")
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
NATURAL_STATES = ("base", "F", "Q", "FQ", "E", "O", "N", "S")
PAIR_OPERATIONS = ("F", "Q", "FQ", "E", "O", "N", "S", "I")
ANALYSIS_OPERATIONS = PAIR_OPERATIONS + ("X",)
TIME_STAGES = ("prompt", "after_answer")
WORLDS_PER_POOL_TEMPLATE = 4
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1011_native_semantic_atlas"
)
PROPERTY_LABELS = {
    "comparison": ("highest", "lowest"),
    "negation": ("present", "missing"),
    "semantic_role": ("agent", "patient"),
}


FACT_ROLES = {
    "comparison": {
        "chain_left": "focal_source",
        "chain_bridge_0": "focal_bridge",
        "chain_bridge_1": "focal_bridge",
        "chain_right": "focal_source",
        "nuisance_left": "nuisance",
        "nuisance_right": "nuisance",
    },
    "negation": {
        "focal_entity_0": "focal_source",
        "focal_marker_0": "focal_operator",
        "focal_entity_1": "focal_source",
        "focal_marker_1": "focal_operator",
        "nuisance_entity": "nuisance",
        "nuisance_marker": "nuisance",
    },
    "semantic_role": {
        "focal_agent": "focal_source",
        "focal_patient": "focal_source",
        "nuisance_agent": "nuisance",
        "nuisance_patient": "nuisance",
    },
}


def role_classes(family: str, output_mode: str) -> dict[str, str]:
    result = dict(FACT_ROLES[family])
    if output_mode == "entity":
        if family == "comparison":
            result.update({
                "query_entity_0": "query_anchor",
                "query_entity_1": "query_anchor",
                "query_entity_2": "query_anchor",
                "query_operator": "query_operator",
            })
        elif family == "negation":
            result.update({
                "query_entity_0": "query_anchor",
                "query_entity_1": "query_anchor",
                "query_operator": "query_operator",
            })
        else:
            result.update({
                "query_anchor": "query_anchor",
                "query_operator": "query_operator",
            })
    else:
        result.update({
            "query_target": "query_anchor",
            "query_operator": "query_operator",
        })
    result["answer_boundary"] = "answer_boundary"
    return result


def add_instruction(builder: PromptBuilder, output_mode: str) -> None:
    if output_mode == "entity":
        builder.add(
            "\nReply with exactly one listed person's name and no other text."
        )
    elif output_mode == "property":
        builder.add("\nReply with exactly one property word and no other text.")
    else:
        builder.add("\nReply with exactly yes or no and no other text.")


def render_comparison(
    template: int,
    names: list[str],
    state: str,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    reversed_chain = state in {"F", "FQ"}
    paraphrase = state == "S"
    if reversed_chain:
        chain = [
            (c, b, "chain_left", "chain_bridge_0"),
            (b, a, "chain_bridge_1", "chain_right"),
        ]
        high, low = c, a
    else:
        chain = [
            (a, b, "chain_left", "chain_bridge_0"),
            (b, c, "chain_bridge_1", "chain_right"),
        ]
        high, low = a, c
    nuisance = (d, e)
    if state == "E":
        nuisance = (e, f)
    elif state == "N":
        nuisance = (e, d)
    facts = [
        (*chain[0], False),
        (*chain[1], False),
        (nuisance[0], nuisance[1], "nuisance_left", "nuisance_right", True),
    ]
    order = [0, 1, 2] if state != "O" else [2, 1, 0]
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, start=1):
        left, right, left_role, right_role, _ = facts[fact_index]
        comparison_fact(
            builder,
            template,
            display_index,
            left,
            right,
            left_role,
            right_role,
            paraphrase,
        )
    query_flip = state in {"Q", "FQ"}
    if output_mode == "entity":
        operator = "lowest" if query_flip else "highest"
        builder.add("\nAmong")
        for index, name in enumerate((a, b, c)):
            builder.mark(f"query_entity_{index}", name)
            builder.add("," if index < 2 else "")
        builder.add(" return the")
        builder.mark("query_operator", operator)
        builder.add(" person.")
        gold = low if query_flip else high
        foil = high if query_flip else low
    else:
        target = c if query_flip else a
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return that person's")
            builder.mark("query_operator", "rank")
            builder.add(" as highest or lowest among the three focal people.")
            gold = "highest" if target == high else "lowest"
            foil = "lowest" if gold == "highest" else "highest"
        else:
            builder.add(",")
            builder.mark("query_operator", "is")
            builder.add(" that person the highest among the three focal people?")
            gold = "yes" if target == high else "no"
            foil = "no" if gold == "yes" else "yes"
    add_instruction(builder, output_mode)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def render_negation(
    template: int,
    names: list[str],
    state: str,
    output_mode: str,
    world_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, _, _ = names
    markers = ["present", "missing"]
    if state in {"F", "FQ"}:
        markers.reverse()
    query_flip = state in {"Q", "FQ"}
    nuisance_name = d if state == "E" else c
    nuisance_marker = "present" if world_index % 2 == 0 else "missing"
    if state == "N":
        nuisance_marker = (
            "missing" if nuisance_marker == "present" else "present"
        )
    facts = [
        (a, markers[0], "focal_entity_0", "focal_marker_0"),
        (b, markers[1], "focal_entity_1", "focal_marker_1"),
        (nuisance_name, nuisance_marker, "nuisance_entity", "nuisance_marker"),
    ]
    order = [0, 1, 2] if state != "O" else [2, 1, 0]
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, start=1):
        negation_fact(
            builder,
            template,
            display_index,
            *facts[fact_index],
            paraphrase=state == "S",
        )
    if output_mode == "entity":
        query_marker = "missing" if query_flip else "present"
        builder.add("\nBetween")
        builder.mark("query_entity_0", a)
        builder.add(" and")
        builder.mark("query_entity_1", b)
        builder.add(", return the person whose")
        builder.mark("query_operator", "status")
        builder.add(f" is {query_marker}.")
        gold = a if markers[0] == query_marker else b
        foil = b if gold == a else a
    else:
        target = b if query_flip else a
        status = markers[1] if target == b else markers[0]
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return the person's")
            builder.mark("query_operator", "status")
            builder.add(" as present or missing.")
            gold = status
            foil = "missing" if gold == "present" else "present"
        else:
            builder.add(",")
            builder.mark("query_operator", "is")
            builder.add(" that person present?")
            gold = "yes" if status == "present" else "no"
            foil = "no" if gold == "yes" else "yes"
    add_instruction(builder, output_mode)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def render_semantic_role(
    template: int,
    names: list[str],
    state: str,
    output_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    focal = (b, a) if state in {"F", "FQ"} else (a, b)
    nuisance = (e, f) if state == "E" else (c, d)
    if state == "N":
        nuisance = (d, c)
    facts = [
        (focal[0], focal[1], "focal_agent", "focal_patient"),
        (nuisance[0], nuisance[1], "nuisance_agent", "nuisance_patient"),
    ]
    order = [0, 1] if state != "O" else [1, 0]
    builder = PromptBuilder()
    for display_index, fact_index in enumerate(order, start=1):
        semantic_role_fact(
            builder,
            template,
            display_index,
            *facts[fact_index],
            passive=state == "S",
        )
    query_flip = state in {"Q", "FQ"}
    if output_mode == "entity":
        query_operator = "patient" if query_flip else "agent"
        builder.add("\nReturn the")
        builder.mark("query_operator", query_operator)
        if query_operator == "agent":
            builder.add(" who helped")
            builder.mark("query_anchor", focal[1])
            gold, foil = focal[0], focal[1]
        else:
            builder.add(" whom")
            builder.mark("query_anchor", focal[0])
            builder.add(" helped")
            gold, foil = focal[1], focal[0]
        builder.add(".")
    else:
        target = b if query_flip else a
        role = "agent" if target == focal[0] else "patient"
        builder.add("\nFor")
        builder.mark("query_target", target)
        if output_mode == "property":
            builder.add(", return that person's")
            builder.mark("query_operator", "role")
            builder.add(" as agent or patient in Event 1.")
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


def render_family(
    family: str,
    output_mode: str,
    template: int,
    names: list[str],
    state: str,
    world_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    if family == "comparison":
        return render_comparison(template, names, state, output_mode)
    if family == "negation":
        return render_negation(
            template,
            names,
            state,
            output_mode,
            world_index,
        )
    if family == "semantic_role":
        return render_semantic_role(template, names, state, output_mode)
    raise KeyError(family)


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
        output_mode,
        template,
        names,
        state,
        world_index,
    )
    if "response map" in raw_prompt.lower() or " maps to " in raw_prompt.lower():
        raise RuntimeError("explicit response-map text entered native protocol")
    rendered = render_chat(tokenizer, model_name, raw_prompt)
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    classes = role_classes(family, output_mode)
    positions = role_token_positions(
        tokenizer,
        rendered,
        raw_prompt,
        spans,
    )
    positions["answer_boundary"] = len(input_ids) - 1
    if set(positions) != set(classes):
        raise RuntimeError(
            f"{family}/{output_mode}/{state}: role coverage drift "
            f"{set(positions) ^ set(classes)}"
        )
    labels = candidate_labels(family, output_mode, names)
    candidate_ids = {
        label: boundary_token_id(tokenizer, rendered, label)
        for label in labels
    }
    if len(set(candidate_ids.values())) != len(candidate_ids):
        raise RuntimeError("candidate token collision")
    answer_id = candidate_ids[gold]
    return {
        "schema_version": "phase1011_native_case.v1",
        "phase": PHASE,
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
        "operation": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(positions[role]) for role in classes
        },
        "role_classes": classes,
        "gold": gold,
        "foil": foil,
        "candidate_labels": labels,
        "candidate_token_ids": candidate_ids,
        "answer_text": " " + gold,
        "natural_gold_text": gold,
        "answer_token_ids": [answer_id],
        "semantic_step": 0,
        "protocol_prefix_ids": [],
        "explicit_response_map_present": False,
    }


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    units = []
    for family in FAMILIES:
        for output_mode in OUTPUT_MODES:
            for split in SPLITS:
                for template in TEMPLATES_BY_SPLIT[split]:
                    for name_pool, pool in enumerate(NAME_POOLS[split]):
                        for world_index in range(WORLDS_PER_POOL_TEMPLATE):
                            names = list(pool)
                            shift = world_index % len(names)
                            names = names[shift:] + names[:shift]
                            unit_id = (
                                f"{model_name}.{family}.{output_mode}."
                                f"{split}.t{template}.p{name_pool}.w{world_index}"
                            )
                            case_ids = {}
                            state_cases = {}
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
                                cases.append(case)
                                case_ids[state] = case["record_id"]
                                state_cases[state] = case
                            base_gold = state_cases["base"]["gold"]
                            for operation in ("F", "Q"):
                                if state_cases[operation]["gold"] == base_gold:
                                    raise RuntimeError(
                                        f"{unit_id}: {operation} did not flip label"
                                    )
                            for operation in ("FQ", "E", "O", "N", "S"):
                                if state_cases[operation]["gold"] != base_gold:
                                    raise RuntimeError(
                                        f"{unit_id}: {operation} label invariant failed"
                                    )
                            units.append({
                                "schema_version": "phase1011_native_unit.v1",
                                "phase": PHASE,
                                "model": model_name,
                                "family": family,
                                "output_mode": output_mode,
                                "split": split,
                                "template": int(template),
                                "name_pool": int(name_pool),
                                "world_index": int(world_index),
                                "unit_id": unit_id,
                                "case_ids": case_ids,
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
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(units),
        "panel_counts": dict(Counter(
            f"{row['family']}:{row['output_mode']}" for row in units
        )),
        "all_answers_single_token_at_boundary": True,
        "explicit_response_map_case_count": int(sum(
            row["explicit_response_map_present"] for row in cases
        )),
    }
    write_json(model_root / "summary.json", summary)
    return summary


def main() -> None:
    summaries = [build_model(model) for model in MODELS]
    protocol = {
        "schema_version": "phase1011_native_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "principle": (
            "discover native-output response structures without assuming "
            "the Phase1010 L30 route or fitting a mechanism formula"
        ),
        "models_in_required_execution_order": list(MODELS),
        "families": list(FAMILIES),
        "output_modes": list(OUTPUT_MODES),
        "natural_states": list(NATURAL_STATES),
        "pair_operations": list(PAIR_OPERATIONS),
        "analysis_operations": list(ANALYSIS_OPERATIONS),
        "time_stages": list(TIME_STAGES),
        "templates_by_split": {
            key: list(value)
            for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "output_contract": {
            "entity": "natural person name from the task facts",
            "property": (
                "family-native property: rank, status, or semantic role"
            ),
            "binary": "shared natural yes/no proposition judgment",
            "teacher_forced_candidate_surface": (
                "one leading-space token at the assistant boundary"
            ),
            "free_rollout_surface": (
                "decode generated content and compare its first alphabetic "
                "word case-insensitively; do not equate tokenizer boundary "
                "variants with semantic failure"
            ),
            "explicit_response_map": False,
        },
        "selection_contract": {
            "atlas_nonblocking": True,
            "minimum_behavior_qualified_pairs_per_split": 8,
            "historical_frozen_heads_are_validation_only": True,
            "no_phase1011_candidate_can_be_selected_on_confirmation": True,
            "no_fixed_90_percent_behavior_gate": True,
        },
        "claim_limits": (
            "response repetition is not transport; relative depth is not "
            "cross-model physical homology; a frozen-head effect is not a "
            "rule source; formulas are measurement definitions only"
        ),
        "model_summaries": summaries,
    }
    protocol["preregistration_digest"] = digest(protocol)
    write_json(OUT_ROOT / "protocol" / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
