#!/usr/bin/env python3
"""Freeze the corrected Phase1019 held-out language-pattern protocol."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import (
    CAPTURE_ROLES,
    FACTORIAL_STATES,
    MODELS,
    PROMPT_MODES,
    RARE_ITEMS,
    STATES,
    TRANSLATION_ITEMS,
    WORLDS,
    SpanBuilder,
    add_answer_instruction,
    add_choices,
    canonical,
    continuation_token_ids,
    digest,
    ordered_choices,
    read_json,
    read_jsonl,
    render_chat,
    token_spans,
    tokenizer_for,
    write_json,
    write_jsonl,
)


PHASE = 1019
PROTOCOL_REVISION = 1
FAMILIES = ("rare_semantics", "punctuation", "translation", "contrast")
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1019_corrected_language_pattern_atlas"
)


PUNCTUATION_ITEMS = {
    "question_archive": {
        "subgroup": "statement_question",
        "text": "The archive is open",
        "labels": (".", "?"),
    },
    "question_train": {
        "subgroup": "statement_question",
        "text": "The last train has arrived",
        "labels": (".", "?"),
    },
    "question_meeting": {
        "subgroup": "statement_question",
        "text": "The meeting starts at noon",
        "labels": (".", "?"),
    },
    "question_key": {
        "subgroup": "statement_question",
        "text": "This key opens the cabinet",
        "labels": (".", "?"),
    },
    "colon_sensor": {
        "subgroup": "separation_explanation",
        "text": "The cause was clear | the sensor failed",
        "labels": (".", ":"),
    },
    "colon_delay": {
        "subgroup": "separation_explanation",
        "text": "There was one reason for the delay | the bridge was closed",
        "labels": (".", ":"),
    },
    "colon_result": {
        "subgroup": "separation_explanation",
        "text": "The result was immediate | the alarm sounded",
        "labels": (".", ":"),
    },
    "colon_goal": {
        "subgroup": "separation_explanation",
        "text": "She had one goal | finish the manuscript",
        "labels": (".", ":"),
    },
    "semicolon_storm": {
        "subgroup": "weak_strong_boundary",
        "text": "The storm ended | the roads remained flooded",
        "labels": (",", ";"),
    },
    "semicolon_server": {
        "subgroup": "weak_strong_boundary",
        "text": "The server restarted | the queued jobs resumed",
        "labels": (",", ";"),
    },
    "semicolon_sun": {
        "subgroup": "weak_strong_boundary",
        "text": "The sun set | the air remained warm",
        "labels": (",", ";"),
    },
    "semicolon_bell": {
        "subgroup": "weak_strong_boundary",
        "text": "The bell rang | the audience took their seats",
        "labels": (",", ";"),
    },
}


PUNCTUATION_CUES = {
    "statement_question": (
        ("The utterance states information", "The speaker asserts a fact"),
        ("The utterance asks for information", "The speaker requests an answer"),
    ),
    "separation_explanation": (
        (
            "Treat the two parts as separate complete statements",
            "The second part stands independently",
        ),
        (
            "The second part explains the first",
            "The first part introduces its explanation",
        ),
    ),
    "weak_strong_boundary": (
        (
            "Use a light internal pause",
            "The parts form one tightly linked flow",
        ),
        (
            "Use a stronger boundary between related clauses",
            "The second clause is related but more independent",
        ),
    ),
}


CONTRAST_ITEMS = {
    "room": (
        "The room was small",
        "it felt cramped",
        "it felt comfortable",
    ),
    "exam": (
        "The exam was difficult",
        "Mira struggled throughout",
        "Mira remained calm",
    ),
    "weather": (
        "The sky was cloudy",
        "the afternoon stayed gloomy",
        "the afternoon stayed warm",
    ),
    "engine": (
        "The engine was old",
        "it broke down often",
        "it ran quietly",
    ),
    "book": (
        "The book was long",
        "the reading took weeks",
        "the argument stayed clear",
    ),
    "road": (
        "The road was narrow",
        "traffic moved slowly",
        "traffic moved quickly",
    ),
    "meal": (
        "The meal was simple",
        "the flavors were plain",
        "the guests were delighted",
    ),
    "team": (
        "The team was inexperienced",
        "it lost the match",
        "it won the match",
    ),
    "phone": (
        "The phone was inexpensive",
        "the camera was basic",
        "the camera was excellent",
    ),
    "house": (
        "The house was remote",
        "the internet was unreliable",
        "the internet was reliable",
    ),
    "lecture": (
        "The lecture was technical",
        "the audience was confused",
        "the audience followed it",
    ),
    "garden": (
        "The garden was tiny",
        "it contained few species",
        "it contained many species",
    ),
}


TRANSLATION_OPERATORS = {
    0: (
        ("copy", "repeat"),
        ("preserve", "echo"),
        ("retain", "reproduce"),
        ("keep", "mirror"),
    ),
    1: (
        ("translate", "convert"),
        ("render", "translate"),
        ("convert", "render"),
        ("translate", "rewrite"),
    ),
}


def state_factors(state: str) -> tuple[int, int]:
    if state == "identity":
        return 0, 0
    return int(state[1]), int(state[-1])


def rare_item_ids() -> tuple[str, ...]:
    actual = tuple(RARE_ITEMS)
    masked = tuple(f"{item_id}__masked" for item_id in RARE_ITEMS)
    return actual + masked


def item_ids(family: str) -> tuple[str, ...]:
    if family == "rare_semantics":
        return rare_item_ids()
    if family == "punctuation":
        return tuple(PUNCTUATION_ITEMS)
    if family == "translation":
        return tuple(TRANSLATION_ITEMS)
    if family == "contrast":
        return tuple(CONTRAST_ITEMS)
    raise KeyError(family)


def render_rare(
    item_id: str,
    split: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    masked = item_id.endswith("__masked")
    base_item = item_id.removesuffix("__masked")
    spec = RARE_ITEMS[base_item]
    carrier = "[unknown term]" if masked else spec["carrier"]
    cue = spec[split][branch][lexical]
    choices = ordered_choices(spec["labels"], world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    if template == 0:
        builder.add(" context:")
        builder.mark("context_anchor", cue)
        builder.add(". Focus term:")
        builder.mark("carrier", carrier)
        builder.add(" .")
        builder.mark("operator", "Interpret")
        builder.mark("query_anchor", "reading")
    elif template == 1:
        builder.add(" record. The clue is")
        builder.mark("context_anchor", cue)
        builder.add("; the lexical item is")
        builder.mark("carrier", carrier)
        builder.add(" .")
        builder.mark("operator", "Classify")
        builder.mark("query_anchor", "sense")
    elif template == 2:
        builder.add(" usage note:")
        builder.mark("context_anchor", cue)
        builder.add(". Term under review:")
        builder.mark("carrier", carrier)
        builder.add(" .")
        builder.mark("operator", "Resolve")
        builder.mark("query_anchor", "meaning")
    else:
        builder.add(" reading task. Nearby wording:")
        builder.mark("context_anchor", cue)
        builder.add(". The expression is")
        builder.mark("carrier", carrier)
        builder.add(" .")
        builder.mark("operator", "Choose")
        builder.mark("query_anchor", "interpretation")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, spec["labels"][branch], {
        "subgroup": (
            "rare_word_masked" if masked else "rare_word_actual"
        ),
        "carrier_text": carrier,
        "branch_labels": list(spec["labels"]),
        "base_item_id": base_item,
        "carrier_condition": "masked" if masked else "actual",
        "cue": cue,
    }


def render_punctuation(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = PUNCTUATION_ITEMS[item_id]
    cue = PUNCTUATION_CUES[spec["subgroup"]][branch][lexical]
    choices = ordered_choices(spec["labels"], world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    if template == 0:
        builder.add(" unpunctuated text:")
        builder.mark("carrier", spec["text"])
        builder.add(". Intended use:")
        builder.mark("context_anchor", cue)
        builder.add(".")
        builder.mark("operator", "Select")
        builder.mark("query_anchor", "missing boundary mark")
    elif template == 1:
        builder.add(" boundary task. Text without its target mark:")
        builder.mark("carrier", spec["text"])
        builder.add(". Relation clue:")
        builder.mark("context_anchor", cue)
        builder.add(".")
        builder.mark("operator", "Choose")
        builder.mark("query_anchor", "punctuation")
    elif template == 2:
        builder.add(" editing record. Unfilled text:")
        builder.mark("carrier", spec["text"])
        builder.add(". Writer intention:")
        builder.mark("context_anchor", cue)
        builder.add(".")
        builder.mark("operator", "Supply")
        builder.mark("query_anchor", "boundary symbol")
    else:
        builder.add(" copyediting case. The target slot is blank in:")
        builder.mark("carrier", spec["text"])
        builder.add(". Functional description:")
        builder.mark("context_anchor", cue)
        builder.add(".")
        builder.mark("operator", "Return")
        builder.mark("query_anchor", "best mark")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, spec["labels"][branch], {
        "subgroup": spec["subgroup"],
        "carrier_text": spec["text"],
        "branch_labels": list(spec["labels"]),
        "target_inserted_in_carrier": False,
        "cue": cue,
    }


def render_translation(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    spec = TRANSLATION_ITEMS[item_id]
    choices = ordered_choices(spec["labels"], world)
    operator = TRANSLATION_OPERATORS[branch][template][lexical]
    output_language = (
        spec["source_language"] if branch == 0
        else spec["target_language"]
    )
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    if template == 0:
        builder.add(" source:")
        builder.mark("carrier", spec["source"])
        builder.add(".")
        builder.mark("operator", operator)
        builder.add(" it into")
        builder.mark("context_anchor", output_language)
        builder.add(".")
        builder.mark("query_anchor", "output")
    elif template == 1:
        builder.add(" language entry:")
        builder.mark("carrier", spec["source"])
        builder.add(". Requested operation:")
        builder.mark("operator", operator)
        builder.add(". Requested language:")
        builder.mark("context_anchor", output_language)
        builder.add(".")
        builder.mark("query_anchor", "result")
    elif template == 2:
        builder.add(" input expression:")
        builder.mark("carrier", spec["source"])
        builder.add(". Rendering rule:")
        builder.mark("operator", operator)
        builder.add(". Destination language:")
        builder.mark("context_anchor", output_language)
        builder.add(".")
        builder.mark("query_anchor", "rendering")
    else:
        builder.add(" translator record. Received text:")
        builder.mark("carrier", spec["source"])
        builder.add(". Instruction:")
        builder.mark("operator", operator)
        builder.add(". Output language:")
        builder.mark("context_anchor", output_language)
        builder.add(".")
        builder.mark("query_anchor", "answer")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, spec["labels"][branch], {
        "subgroup": spec["direction"],
        "carrier_text": spec["source"],
        "branch_labels": list(spec["labels"]),
        "concept": spec["concept"],
        "source_language": spec["source_language"],
        "target_language": spec["target_language"],
        "operator_surface": operator,
    }


def render_contrast(
    item_id: str,
    template: int,
    world: int,
    branch: int,
    lexical: int,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    left, additive, contrastive = CONTRAST_ITEMS[item_id]
    right = additive if branch == 0 else contrastive
    labels = (
        ("and", "but")
        if lexical == 0
        else ("additionally", "however")
    )
    choices = ordered_choices(labels, world)
    builder = SpanBuilder()
    builder.mark("prefix_anchor", "Pattern")
    if template == 0:
        builder.add(" first clause:")
        builder.mark("carrier", left)
        builder.add(". Second clause:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Select")
        builder.mark("query_anchor", "connector")
    elif template == 1:
        builder.add(" relation task. Clause A:")
        builder.mark("carrier", left)
        builder.add(". Clause B:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Choose")
        builder.mark("query_anchor", "linking word")
    elif template == 2:
        builder.add(" discourse pair. Earlier clause:")
        builder.mark("carrier", left)
        builder.add(". Following clause:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Supply")
        builder.mark("query_anchor", "relation marker")
    else:
        builder.add(" composition record. Statement one:")
        builder.mark("carrier", left)
        builder.add(". Statement two:")
        builder.mark("context_anchor", right)
        builder.add(".")
        builder.mark("operator", "Return")
        builder.mark("query_anchor", "best transition")
    add_choices(builder, choices, world, template)
    add_answer_instruction(builder)
    prompt, spans = builder.finish()
    return prompt, spans, labels[branch], {
        "subgroup": "additive_vs_contrast",
        "carrier_text": left,
        "branch_labels": list(labels),
        "target_inserted_in_carrier": False,
        "right_clause": right,
    }


def render_case(
    *,
    family: str,
    item_id: str,
    split: str,
    template: int,
    world: int,
    state: str,
) -> tuple[str, dict[str, tuple[int, int, str]], str, dict[str, Any]]:
    branch, lexical = state_factors(state)
    if family == "rare_semantics":
        return render_rare(
            item_id, split, template, world, branch, lexical
        )
    if family == "punctuation":
        return render_punctuation(
            item_id, template, world, branch, lexical
        )
    if family == "translation":
        return render_translation(
            item_id, template, world, branch, lexical
        )
    if family == "contrast":
        return render_contrast(
            item_id, template, world, branch, lexical
        )
    raise KeyError(family)


def build_case(
    *,
    tokenizer,
    model_name: str,
    prompt_mode: str,
    family: str,
    item_id: str,
    split: str,
    template: int,
    world: int,
    unit_id: str,
    state: str,
) -> dict[str, Any]:
    raw_prompt, spans, gold, metadata = render_case(
        family=family,
        item_id=item_id,
        split=split,
        template=template,
        world=world,
        state=state,
    )
    rendered = (
        raw_prompt
        if prompt_mode == "raw"
        else render_chat(tokenizer, model_name, raw_prompt)
    )
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    located = token_spans(tokenizer, rendered, raw_prompt, spans)
    positions = {
        "prefix_anchor": located["prefix_anchor"][1],
        "carrier_start": located["carrier"][0],
        "carrier_end": located["carrier"][1],
        "context_anchor": located["context_anchor"][1],
        "operator": located["operator"][1],
        "query_anchor": located["query_anchor"][1],
        "answer_boundary": len(input_ids) - 1,
    }
    labels = tuple(metadata["branch_labels"])
    candidate_ids = {
        label: continuation_token_ids(tokenizer, rendered, label)
        for label in labels
    }
    foil = labels[1] if gold == labels[0] else labels[0]
    return {
        "schema_version": "phase1019_language_pattern_case.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "family": family,
        "item_id": item_id,
        "split": split,
        "template": int(template),
        "world": int(world),
        "unit_id": unit_id,
        "record_id": f"{unit_id}.{state}",
        "state": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(positions[role]) for role in CAPTURE_ROLES
        },
        "carrier_token_count": (
            located["carrier"][1] - located["carrier"][0] + 1
        ),
        "gold": gold,
        "foil": foil,
        "candidate_labels": list(labels),
        "candidate_token_ids": candidate_ids,
        "candidate_first_token_ids": {
            label: values[0] for label, values in candidate_ids.items()
        },
        **metadata,
    }


def audit_unit(
    unit: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = {
        state: by_id[unit["record_ids"][state]] for state in STATES
    }
    identity_exact = (
        cases["identity"]["input_ids"] == cases["b0_l0"]["input_ids"]
    )
    prefix_stable = all(
        case["input_ids"][
            case["role_positions"]["prefix_anchor"]
        ]
        == cases["b0_l0"]["input_ids"][
            cases["b0_l0"]["role_positions"]["prefix_anchor"]
        ]
        for case in cases.values()
    )
    candidate_unique = all(
        len({
            tuple(values)
            for values in case["candidate_token_ids"].values()
        }) == 2
        for case in cases.values()
    )
    carrier_prefix_stable = []
    for left_name, right_name in (
        ("b0_l0", "b1_l0"),
        ("b0_l1", "b1_l1"),
    ):
        left = cases[left_name]
        right = cases[right_name]
        left_end = left["role_positions"]["carrier_end"]
        right_end = right["role_positions"]["carrier_end"]
        carrier_prefix_stable.append(
            left_end == right_end
            and left["input_ids"][:left_end + 1]
            == right["input_ids"][:right_end + 1]
        )
    return {
        "schema_version": "phase1019_protocol_unit_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": unit["model"],
        "prompt_mode": unit["prompt_mode"],
        "family": unit["family"],
        "item_id": unit["item_id"],
        "split": unit["split"],
        "template": unit["template"],
        "world": unit["world"],
        "unit_id": unit["unit_id"],
        "identity_exact": identity_exact,
        "prefix_token_stable": prefix_stable,
        "candidate_ids_unique": candidate_unique,
        "branch_preserves_causal_prefix": prefix_stable,
        "branch_preserves_carrier_prefix": all(carrier_prefix_stable),
        "carrier_prefix_expected": unit["family"] != "rare_semantics",
    }


def build_protocol() -> dict[str, Any]:
    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "schema_version": "phase1019_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "prompt_modes": list(PROMPT_MODES),
        "families": list(FAMILIES),
        "items_per_family": {
            family: len(item_ids(family)) for family in FAMILIES
        },
        "splits": list(SPLITS),
        "templates_by_split": {
            key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "worlds": list(WORLDS),
        "states": list(STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "corrections_from_phase1018": [
            "All discovery and confirmation prompts must be text-disjoint.",
            "Punctuation and contrast targets are absent from the carrier.",
            "Rare terms have paired masked-carrier controls.",
            "Translation uses distinct operation words in all four templates.",
        ],
        "primary_descriptive_thresholds": {
            "direction_consistency": 0.45,
            "surface_alignment": 0.40,
            "minimum_normalized_magnitude": 1e-4,
        },
        "threshold_grid": {
            "direction_consistency": [0.30, 0.45, 0.60],
            "surface_alignment": [0.20, 0.40, 0.60],
        },
        "primary_questions": [
            "Which branch responses survive genuinely held-out phrasing?",
            "Does a real rare term improve behavior over its masked control?",
            "Does a real rare term alter the context-branch direction?",
            "Do target-absent punctuation and contrast prompts yield repeated downstream structure?",
            "Does translation show a shared operation direction across concepts?",
        ],
        "claim_limits": [
            "The protocol maps response structure, not causal necessity.",
            "Explicit task cues may still support shortcuts.",
            "A masked-word difference is lexical participation, not full word meaning.",
            "No universal equation is assumed across the four families.",
        ],
        "automatic_continuation_rule": {
            "causal_followup_requires": [
                "candidate accuracy >= 0.70 in at least two models",
                "independent discovery-confirmation response repeat",
                "matched-minus-mismatched gap >= 0.15 in at least two models",
                "for rare words, actual-minus-masked accuracy >= 0.10 in at least two models",
            ],
            "otherwise": "retain descriptive atlas and redesign the task",
        },
    }
    preregistration["protocol_digest"] = digest(preregistration)
    write_json(protocol_root / "preregistration.json", preregistration)

    global_summary = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": preregistration["protocol_digest"],
        "models": {},
    }
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        model_summary = {}
        for prompt_mode in PROMPT_MODES:
            cases = []
            units = []
            for family in FAMILIES:
                for item_id in item_ids(family):
                    for split in SPLITS:
                        for template in TEMPLATES_BY_SPLIT[split]:
                            for world in WORLDS:
                                unit_id = (
                                    f"p{PHASE}.{model_name}.{prompt_mode}."
                                    f"{family}.{item_id}.{split}.t{template}."
                                    f"w{world}"
                                )
                                record_ids = {}
                                for state in STATES:
                                    try:
                                        case = build_case(
                                            tokenizer=tokenizer,
                                            model_name=model_name,
                                            prompt_mode=prompt_mode,
                                            family=family,
                                            item_id=item_id,
                                            split=split,
                                            template=template,
                                            world=world,
                                            unit_id=unit_id,
                                            state=state,
                                        )
                                    except Exception as error:
                                        raise RuntimeError(
                                            f"case build failed: {unit_id}."
                                            f"{state}"
                                        ) from error
                                    cases.append(case)
                                    record_ids[state] = case["record_id"]
                                units.append({
                                    "schema_version": (
                                        "phase1019_pattern_unit.v1"
                                    ),
                                    "phase": PHASE,
                                    "protocol_revision": PROTOCOL_REVISION,
                                    "model": model_name,
                                    "prompt_mode": prompt_mode,
                                    "family": family,
                                    "item_id": item_id,
                                    "split": split,
                                    "template": int(template),
                                    "world": int(world),
                                    "unit_id": unit_id,
                                    "record_ids": record_ids,
                                })
            by_id = {case["record_id"]: case for case in cases}
            audits = [audit_unit(unit, by_id) for unit in units]
            if not all(
                row["identity_exact"]
                and row["prefix_token_stable"]
                and row["candidate_ids_unique"]
                and (
                    row["branch_preserves_carrier_prefix"]
                    if row["carrier_prefix_expected"]
                    else True
                )
                for row in audits
            ):
                raise RuntimeError(
                    f"protocol audit failed for {model_name}/{prompt_mode}"
                )
            split_prompts: dict[
                tuple[str, str], dict[str, set[str]]
            ] = {}
            for family in FAMILIES:
                for item_id in item_ids(family):
                    split_prompts[(family, item_id)] = {
                        split: {
                            row["raw_prompt"]
                            for row in cases
                            if row["family"] == family
                            and row["item_id"] == item_id
                            and row["split"] == split
                        }
                        for split in SPLITS
                    }
            overlap = {
                f"{family}:{item_id}": len(
                    values["discovery"] & values["confirmation"]
                )
                for (family, item_id), values in split_prompts.items()
            }
            if any(overlap.values()):
                raise RuntimeError(
                    f"held-out text overlap for {model_name}/{prompt_mode}"
                )
            write_jsonl(
                protocol_root / f"cases.{model_name}.{prompt_mode}.jsonl",
                cases,
            )
            write_jsonl(
                protocol_root / f"units.{model_name}.{prompt_mode}.jsonl",
                units,
            )
            write_jsonl(
                protocol_root / f"audit.{model_name}.{prompt_mode}.jsonl",
                audits,
            )
            model_summary[prompt_mode] = {
                "case_count": len(cases),
                "unit_count": len(units),
                "cases_by_family": dict(Counter(
                    row["family"] for row in cases
                )),
                "units_by_family": dict(Counter(
                    row["family"] for row in units
                )),
                "rare_actual_token_counts": dict(Counter(
                    row["carrier_token_count"]
                    for row in cases
                    if row["family"] == "rare_semantics"
                    and row["carrier_condition"] == "actual"
                )),
                "rare_masked_token_counts": dict(Counter(
                    row["carrier_token_count"]
                    for row in cases
                    if row["family"] == "rare_semantics"
                    and row["carrier_condition"] == "masked"
                )),
                "exact_split_overlap_count": sum(overlap.values()),
                "all_identity_exact": all(
                    row["identity_exact"] for row in audits
                ),
                "all_post_carrier_branches_preserve_carrier": all(
                    row["branch_preserves_carrier_prefix"]
                    for row in audits
                    if row["carrier_prefix_expected"]
                ),
            }
        global_summary["models"][model_name] = model_summary
        del tokenizer
    write_json(protocol_root / "summary.json", global_summary)
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))
    return global_summary


if __name__ == "__main__":
    build_protocol()
