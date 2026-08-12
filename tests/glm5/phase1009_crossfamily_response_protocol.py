#!/usr/bin/env python3
"""Freeze the Phase1009 cross-family response-atlas protocol.

This phase tests whether structures discovered in Phase1008 repeat across
language families. It does not assume a shared decision field, a transport
path, or a mechanism formula. The factorial measurements are descriptive.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1006_autoregressive_temporal_aggregation_protocol import ANSWER_PREFIX
from phase1008_global_response_atlas_protocol import NAME_POOLS


PHASE = 1009
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("comparison", "negation", "semantic_role")
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
NATURAL_STATES = ("base", "F", "Q", "FQ", "E", "O", "N", "S")
PAIR_OPERATIONS = ("F", "Q", "FQ", "E", "O", "N", "S", "I")
ANALYSIS_OPERATIONS = PAIR_OPERATIONS + ("X",)
TIME_STAGES = ("prompt", "semantic0", "function0", "termination")
WORLDS_PER_POOL_TEMPLATE = 4
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1009_crossfamily_response_atlas"
)

ROLE_CLASSES = {
    "comparison": {
        "chain_left": "focal_source",
        "chain_bridge_0": "focal_bridge",
        "chain_bridge_1": "focal_bridge",
        "chain_right": "focal_source",
        "nuisance_left": "nuisance",
        "nuisance_right": "nuisance",
        "query_entity_0": "query_anchor",
        "query_entity_1": "query_anchor",
        "query_entity_2": "query_anchor",
        "query_operator": "query_operator",
        "answer_boundary": "answer_boundary",
    },
    "negation": {
        "focal_entity_0": "focal_source",
        "focal_marker_0": "focal_operator",
        "focal_entity_1": "focal_source",
        "focal_marker_1": "focal_operator",
        "nuisance_entity": "nuisance",
        "nuisance_marker": "nuisance",
        "query_entity_0": "query_anchor",
        "query_entity_1": "query_anchor",
        "query_operator": "query_operator",
        "answer_boundary": "answer_boundary",
    },
    "semantic_role": {
        "focal_agent": "focal_source",
        "focal_patient": "focal_source",
        "nuisance_agent": "nuisance",
        "nuisance_patient": "nuisance",
        "query_anchor": "query_anchor",
        "query_operator": "query_operator",
        "answer_boundary": "answer_boundary",
    },
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def one_token_id(tokenizer, text: str) -> int:
    values = tokenizer.encode(text, add_special_tokens=False)
    if len(values) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {values}")
    return int(values[0])


def rotated(values: tuple[Any, ...], amount: int) -> list[Any]:
    offset = amount % len(values)
    return list(values[offset:] + values[:offset])


@dataclass
class PromptBuilder:
    parts: list[str] = field(default_factory=list)
    spans: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    length: int = 0

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def mark(self, role: str, token: str) -> None:
        if role in self.spans:
            raise RuntimeError(f"duplicate prompt role {role}")
        segment = " " + token
        start = self.length
        self.add(segment)
        self.spans[role] = (start, self.length, segment)

    def finish(self) -> tuple[str, dict[str, tuple[int, int, str]]]:
        return "".join(self.parts), dict(self.spans)


def answer_text(model_name: str, name: str) -> str:
    return f"{ANSWER_PREFIX[model_name]}{name} done"


def answer_protocol(tokenizer, model_name: str, split: str) -> dict[str, Any]:
    names = [
        name
        for pool in NAME_POOLS[split]
        for name in pool
    ]
    answers = {
        name: [
            int(value)
            for value in tokenizer.encode(
                answer_text(model_name, name),
                add_special_tokens=False,
            )
        ]
        for name in names
    }
    widths = {len(ids) for ids in answers.values()}
    if len(widths) != 1:
        raise RuntimeError(f"{model_name}/{split}: answer width drift {widths}")
    width = next(iter(widths))
    varying = [
        index
        for index in range(width)
        if len({ids[index] for ids in answers.values()}) > 1
    ]
    if len(varying) != 1:
        raise RuntimeError(
            f"{model_name}/{split}: expected one semantic step, got {varying}"
        )
    semantic_step = varying[0]
    prefixes = {tuple(ids[:semantic_step]) for ids in answers.values()}
    suffixes = {tuple(ids[semantic_step + 1:]) for ids in answers.values()}
    if len(prefixes) != 1 or len(suffixes) != 1:
        raise RuntimeError(f"{model_name}/{split}: answer framing drift")
    suffix = list(next(iter(suffixes)))
    if len(suffix) != 1:
        raise RuntimeError(
            f"{model_name}/{split}: expected one function token, got {suffix}"
        )
    done_id = one_token_id(tokenizer, " done")
    if suffix[0] != done_id:
        raise RuntimeError(
            f"{model_name}/{split}: function suffix is not ' done'"
        )
    return {
        "answers": answers,
        "semantic_step": semantic_step,
        "function_step": semantic_step + 1,
        "protocol_prefix_ids": list(next(iter(prefixes))),
        "function_token_id": int(done_id),
        "candidate_name_ids": {
            name: int(ids[semantic_step])
            for name, ids in answers.items()
        },
    }


def add_instruction(builder: PromptBuilder) -> None:
    builder.add(
        "\nReply exactly as Answer: NAME done. Replace NAME with one listed "
        "person. Add nothing else."
    )


def comparison_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    left: str,
    right: str,
    left_role: str,
    right_role: str,
    paraphrase: bool,
) -> None:
    if not paraphrase:
        prefixes = (
            f"Rank fact {index}:",
            f"Height record {index} says",
            f"Comparison {index}:",
            f"Order statement {index} places",
        )
        middles = (
            " is taller than",
            " ranks above",
            " exceeds",
            " higher than",
        )
        builder.add(prefixes[template])
        builder.mark(left_role, left)
        builder.add(middles[template])
        builder.mark(right_role, right)
        builder.add(". ")
        return
    prefixes = (
        f"Rank fact {index}: compared with",
        f"Height record {index} puts",
        f"Comparison {index}:",
        f"Order statement {index} places",
    )
    middles = (
        ",",
        " below",
        " is lower than",
        " lower than",
    )
    builder.add(prefixes[template])
    builder.mark(right_role, right)
    builder.add(middles[template])
    builder.mark(left_role, left)
    if template == 0:
        builder.add(" is taller.")
    else:
        builder.add(". ")


def render_comparison(
    template: int,
    names: list[str],
    state: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    reversed_chain = state in {"F", "FQ"}
    query_high = state not in {"Q", "FQ"}
    paraphrase = state == "S"
    if reversed_chain:
        chain = [
            (c, b, "chain_left", "chain_bridge_0"),
            (b, a, "chain_bridge_1", "chain_right"),
        ]
        left_answer, right_answer = c, a
    else:
        chain = [
            (a, b, "chain_left", "chain_bridge_0"),
            (b, c, "chain_bridge_1", "chain_right"),
        ]
        left_answer, right_answer = a, c
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
    order = [0, 1, 2]
    if state == "O":
        order = [2, 1, 0]
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
    operator = "highest" if query_high else "lowest"
    builder.add("\nAmong")
    for index, name in enumerate((a, b, c)):
        builder.mark(f"query_entity_{index}", name)
        builder.add("," if index < 2 else "")
    builder.add(" return the")
    builder.mark("query_operator", operator)
    builder.add(" person.")
    add_instruction(builder)
    prompt, spans = builder.finish()
    gold = left_answer if query_high else right_answer
    foil = right_answer if query_high else left_answer
    return prompt, spans, gold, foil


def negation_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    name: str,
    marker: str,
    entity_role: str,
    marker_role: str,
    paraphrase: bool,
) -> None:
    if not paraphrase:
        prefixes = (
            f"Status fact {index}:",
            f"Record {index} marks",
            f"Signal entry {index} lists",
            f"Truth row {index} gives",
        )
        builder.add(prefixes[template])
        builder.mark(entity_role, name)
        builder.add(" as" if template in (2, 3) else "")
        builder.mark(marker_role, marker)
        builder.add(". ")
        return
    prefixes = (
        f"Status fact {index} lists",
        f"In record {index}, the status for",
        f"For signal entry {index},",
        f"According to truth row {index},",
    )
    builder.add(prefixes[template])
    builder.mark(entity_role, name)
    builder.add(" is")
    builder.mark(marker_role, marker)
    builder.add(". ")


def render_negation(
    template: int,
    names: list[str],
    state: str,
    world_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, _, _ = names
    markers = ["present", "absent"]
    if state in {"F", "FQ"}:
        markers.reverse()
    query_marker = "absent" if state in {"Q", "FQ"} else "present"
    nuisance_name = d if state == "E" else c
    nuisance_marker = "present" if world_index % 2 == 0 else "absent"
    if state == "N":
        nuisance_marker = (
            "absent" if nuisance_marker == "present" else "present"
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
    builder.add("\nBetween")
    builder.mark("query_entity_0", a)
    builder.add(" and")
    builder.mark("query_entity_1", b)
    builder.add(", return the person whose status is")
    builder.mark("query_operator", query_marker)
    builder.add(".")
    add_instruction(builder)
    prompt, spans = builder.finish()
    gold = a if markers[0] == query_marker else b
    foil = b if gold == a else a
    return prompt, spans, gold, foil


def semantic_role_fact(
    builder: PromptBuilder,
    template: int,
    index: int,
    agent: str,
    patient: str,
    agent_role: str,
    patient_role: str,
    passive: bool,
) -> None:
    prefixes = (
        f"Event {index}:",
        f"Report {index} says",
        f"According to record {index},",
        f"In episode {index},",
    )
    builder.add(prefixes[template])
    if passive:
        builder.mark(patient_role, patient)
        builder.add(" was helped by")
        builder.mark(agent_role, agent)
    else:
        builder.mark(agent_role, agent)
        builder.add(" helped")
        builder.mark(patient_role, patient)
    builder.add(". ")


def render_semantic_role(
    template: int,
    names: list[str],
    state: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    a, b, c, d, e, f = names
    focal = (b, a) if state in {"F", "FQ"} else (a, b)
    nuisance = (e, f) if state == "E" else (c, d)
    if state == "N":
        nuisance = (d, c)
    query_operator = "patient" if state in {"Q", "FQ"} else "agent"
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
    add_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, gold, foil


def render_family(
    family: str,
    template: int,
    names: list[str],
    state: str,
    world_index: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, str]:
    if family == "comparison":
        return render_comparison(template, names, state)
    if family == "negation":
        return render_negation(template, names, state, world_index)
    if family == "semantic_role":
        return render_semantic_role(template, names, state)
    raise KeyError(family)


def role_token_positions(
    tokenizer,
    rendered: str,
    raw_prompt: str,
    spans: dict[str, tuple[int, int, str]],
) -> dict[str, int]:
    raw_start = rendered.index(raw_prompt)
    result = {}
    full_ids = tokenizer.encode(rendered, add_special_tokens=False)
    for role, (start, end, marked_text) in spans.items():
        before_ids = tokenizer.encode(
            rendered[:raw_start + start],
            add_special_tokens=False,
        )
        through_ids = tokenizer.encode(
            rendered[:raw_start + end],
            add_special_tokens=False,
        )
        if through_ids[:len(before_ids)] != before_ids:
            raise RuntimeError(f"{role}: marked token prefix drift")
        added = through_ids[len(before_ids):]
        if len(added) != 1:
            raise RuntimeError(
                f"{role}: {marked_text!r} mapped to {added}, expected one token"
            )
        position = len(before_ids)
        if int(full_ids[position]) != int(added[0]):
            raise RuntimeError(f"{role}: full-sequence token position drift")
        result[role] = position
    return result


def build_case(
    *,
    tokenizer,
    model_name: str,
    family: str,
    split: str,
    template: int,
    name_pool: int,
    world_index: int,
    unit_id: str,
    state: str,
    names: list[str],
    answer: dict[str, Any],
) -> dict[str, Any]:
    raw_prompt, spans, gold, foil = render_family(
        family,
        template,
        names,
        state,
        world_index,
    )
    rendered = render_chat(tokenizer, model_name, raw_prompt)
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    observed_positions = role_token_positions(
        tokenizer,
        rendered,
        raw_prompt,
        spans,
    )
    observed_positions["answer_boundary"] = len(input_ids) - 1
    expected_roles = tuple(ROLE_CLASSES[family])
    if set(observed_positions) != set(expected_roles):
        raise RuntimeError(
            f"{family}/{state}: role set {tuple(observed_positions)} "
            f"!= {expected_roles}"
        )
    role_positions = {
        role: int(observed_positions[role])
        for role in expected_roles
    }
    answer_ids = list(answer["answers"][gold])
    extended = [
        int(value)
        for value in tokenizer.encode(
            rendered + answer_text(model_name, gold),
            add_special_tokens=False,
        )
    ]
    if extended != input_ids + answer_ids:
        raise RuntimeError(
            f"{model_name}/{family}/{state}: answer boundary drift"
        )
    pool_names = list(NAME_POOLS[split][name_pool])
    candidate_name_ids = {
        name: int(answer["candidate_name_ids"][name])
        for name in pool_names
    }
    return {
        "schema_version": "phase1009_case.v1",
        "phase": PHASE,
        "model": model_name,
        "family": family,
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
        "role_positions": role_positions,
        "role_classes": ROLE_CLASSES[family],
        "gold": gold,
        "foil": foil,
        "candidate_names": pool_names,
        "candidate_name_ids": candidate_name_ids,
        "answer_text": answer_text(model_name, gold),
        "answer_token_ids": answer_ids,
        "semantic_step": int(answer["semantic_step"]),
        "function_step": int(answer["function_step"]),
        "protocol_prefix_ids": list(answer["protocol_prefix_ids"]),
        "function_token_id": int(answer["function_token_id"]),
    }


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    all_names = tuple(
        name
        for split in SPLITS
        for pool in NAME_POOLS[split]
        for name in pool
    )
    name_ids = {
        name: one_token_id(tokenizer, " " + name)
        for name in all_names
    }
    if len(set(name_ids.values())) != len(name_ids):
        raise RuntimeError(f"{model_name}: name token collision")
    lexical_audit = {
        token: one_token_id(tokenizer, " " + token)
        for token in (
            "highest",
            "lowest",
            "present",
            "absent",
            "agent",
            "patient",
            "done",
        )
    }
    answer_by_split = {
        split: answer_protocol(tokenizer, model_name, split)
        for split in SPLITS
    }
    cases: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    widths: dict[tuple[str, str, int, str], set[int]] = defaultdict(set)
    for family in FAMILIES:
        for split in SPLITS:
            for template in TEMPLATES_BY_SPLIT[split]:
                for pool_index, pool in enumerate(NAME_POOLS[split]):
                    for world_index in range(WORLDS_PER_POOL_TEMPLATE):
                        names = rotated(pool, world_index)
                        unit_id = (
                            f"{model_name}.{family}.{split[0]}t{template}."
                            f"p{pool_index}.w{world_index}"
                        )
                        case_ids = {}
                        gold_by_state = {}
                        for state in NATURAL_STATES:
                            case = build_case(
                                tokenizer=tokenizer,
                                model_name=model_name,
                                family=family,
                                split=split,
                                template=template,
                                name_pool=pool_index,
                                world_index=world_index,
                                unit_id=unit_id,
                                state=state,
                                names=names,
                                answer=answer_by_split[split],
                            )
                            cases.append(case)
                            case_ids[state] = case["record_id"]
                            gold_by_state[state] = case["gold"]
                            widths[(family, split, template, state)].add(
                                len(case["input_ids"])
                            )
                        if gold_by_state["FQ"] != gold_by_state["base"]:
                            raise RuntimeError(
                                f"{unit_id}: FQ must preserve base answer"
                            )
                        for invariant in ("E", "O", "N", "S"):
                            if gold_by_state[invariant] != gold_by_state["base"]:
                                raise RuntimeError(
                                    f"{unit_id}: {invariant} answer drift"
                                )
                        for changing in ("F", "Q"):
                            if gold_by_state[changing] == gold_by_state["base"]:
                                raise RuntimeError(
                                    f"{unit_id}: {changing} failed to flip answer"
                                )
                        units.append({
                            "schema_version": "phase1009_unit.v1",
                            "phase": PHASE,
                            "model": model_name,
                            "family": family,
                            "split": split,
                            "template": int(template),
                            "name_pool": int(pool_index),
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
    width_audit = {
        ".".join((family, split, f"t{template}", state)): sorted(values)
        for (family, split, template, state), values in sorted(widths.items())
    }
    if any(len(values) != 1 for values in width_audit.values()):
        raise RuntimeError(f"{model_name}: prompt width drift {width_audit}")
    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "units.jsonl", units)
    summary = {
        "schema_version": "phase1009_model_protocol.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(units),
        "pair_count": len(units) * len(PAIR_OPERATIONS),
        "family_unit_counts": {
            family: sum(unit["family"] == family for unit in units)
            for family in FAMILIES
        },
        "split_unit_counts": {
            split: sum(unit["split"] == split for unit in units)
            for split in SPLITS
        },
        "prompt_widths": width_audit,
        "tokenizer_audit": {
            "single_token_name_count": len(name_ids),
            "name_collisions": 0,
            "lexical_tokens": lexical_audit,
        },
        "answer_protocol": answer_by_split,
    }
    write_json(model_root / "summary.json", summary)
    return summary


def build_protocol() -> dict[str, Any]:
    summaries = [build_model(model_name) for model_name in MODELS]
    payload = {
        "schema_version": "phase1009_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Cross-language-family dynamic role response atlas without "
            "mechanism precommitment"
        ),
        "models_in_required_execution_order": list(MODELS),
        "families": list(FAMILIES),
        "splits": list(SPLITS),
        "templates_by_split": {
            split: list(values)
            for split, values in TEMPLATES_BY_SPLIT.items()
        },
        "natural_states": list(NATURAL_STATES),
        "pair_operations": list(PAIR_OPERATIONS),
        "analysis_operations": list(ANALYSIS_OPERATIONS),
        "operation_schema": {
            "F": (
                "family-specific world/fact flip; answer changes while the "
                "query type is fixed"
            ),
            "Q": (
                "family-specific query-role flip; answer changes while the "
                "world/facts are fixed"
            ),
            "FQ": (
                "combined fact and query flip; the final answer equals base "
                "although internal conditions differ"
            ),
            "E": (
                "rename only an irrelevant entity; output remains unchanged"
            ),
            "O": "reorder facts without changing their content or answer",
            "N": (
                "change only an irrelevant fact; output remains unchanged"
            ),
            "S": (
                "semantic-preserving syntax/paraphrase transformation; "
                "output remains unchanged"
            ),
            "I": "repeat the identical base input as the numerical floor",
            "X": (
                "descriptive second difference "
                "h(FQ)-h(F)-h(Q)+h(base); not a natural input or mechanism"
            ),
        },
        "family_tasks": {
            "comparison": (
                "three-entity transitive ordering with highest/lowest query"
            ),
            "negation": (
                "two focal truth states with present/absent query"
            ),
            "semantic_role": (
                "agent/patient retrieval under active/passive syntax"
            ),
        },
        "role_classes": ROLE_CLASSES,
        "time_stages": list(TIME_STAGES),
        "components": (
            "residual_depth_0_to_L",
            "attention_output_layer_1_to_L",
            "mlp_output_layer_1_to_L",
        ),
        "discovery_contract": {
            "primary_question": (
                "which response shapes repeat across names, templates, "
                "families, and models"
            ),
            "forbidden_claims": (
                "co-response is not transport; X is not reasoning; "
                "late concentration is not a shared decision mechanism"
            ),
            "family_specific_first": (
                "discover and confirm within each family before any "
                "cross-family matching"
            ),
            "cross_family_match": (
                "compare stage, role class, component, normalized depth, "
                "trajectory shape, and operation profile; never raw head "
                "or neuron coordinates"
            ),
            "dimension_reduction": (
                "PCA/UMAP/t-SNE cannot select or validate candidates"
            ),
        },
        "qualification_contract": {
            "semantic": (
                "correct name must win within the frozen six-name panel; "
                "full-vocabulary top1 and natural exact rollout are recorded "
                "separately"
            ),
            "non_blocking": (
                "failed cells remain in the atlas with explicit validity "
                "flags and do not erase other observations"
            ),
        },
        "measurement_contract": {
            "response": "Delta h = h(operation)-h(base)",
            "normalized_response": (
                "||Delta h|| divided by mean paired state norm"
            ),
            "direction_consistency": (
                "norm of the mean unit direction; descriptive only"
            ),
            "interaction": "X = h(FQ)-h(F)-h(Q)+h(base)",
            "identity": "I must be exactly zero under deterministic forward",
        },
        "selection_rule": {
            "within_family_discovery_and_confirmation": True,
            "minimum_qualified_pairs_per_split": 8,
            "minimum_name_pools": 2,
            "minimum_templates": 2,
            "trajectory_peak_fraction": 0.90,
            "cross_family_minimum_families": 2,
            "cross_model_minimum_models": 2,
        },
        "causal_policy": {
            "when": (
                "only after a motif repeats in discovery and confirmation, "
                "then across at least two families or two models"
            ),
            "effect": (
                "causal success or failure updates an evidence axis and "
                "never deletes the descriptive atlas"
            ),
        },
        "model_summaries": summaries,
    }
    payload["preregistration_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "protocol.json", payload)
    return payload


def main() -> None:
    protocol = build_protocol()
    print(json.dumps({
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "preregistration_digest": protocol["preregistration_digest"],
        "families": list(FAMILIES),
        "model_cases": {
            row["model"]: row["case_count"]
            for row in protocol["model_summaries"]
        },
        "model_units": {
            row["model"]: row["unit_count"]
            for row in protocol["model_summaries"]
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
