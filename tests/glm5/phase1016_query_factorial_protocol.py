#!/usr/bin/env python3
"""Freeze the Phase1016 query-semantics x lexical-family protocol."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1009_crossfamily_response_protocol import (
    PromptBuilder,
    canonical,
    digest,
    role_token_positions,
    write_json,
    write_jsonl,
)


PHASE = 1016
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
PROMPT_MODES = ("raw", "native_chat")
FAMILIES = (
    "comparison",
    "negation",
    "semantic_role",
    "attribute_binding",
    "spatial_relation",
)
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
LEXICAL_LABELS = {
    "discovery": {
        "comparison": (
            ("highest", "lowest"),
            ("tallest", "shortest"),
        ),
        "negation": (
            ("present", "missing"),
            ("listed", "absent"),
        ),
        "semantic_role": (
            ("agent", "patient"),
            ("actor", "recipient"),
        ),
        "attribute_binding": (
            ("red", "blue"),
            ("crimson", "azure"),
        ),
        "spatial_relation": (
            ("left", "right"),
            ("western", "eastern"),
        ),
    },
    "confirmation": {
        "comparison": (
            ("tallest", "shortest"),
            ("upper", "lower"),
        ),
        "negation": (
            ("listed", "absent"),
            ("available", "unavailable"),
        ),
        "semantic_role": (
            ("actor", "recipient"),
            ("source", "receiver"),
        ),
        "attribute_binding": (
            ("crimson", "azure"),
            ("ruby", "navy"),
        ),
        "spatial_relation": (
            ("western", "eastern"),
            ("west", "east"),
        ),
    },
}
FACTORIAL_STATES = ("s0_l0", "s1_l0", "s0_l1", "s1_l1")
CONTROL_STATES = ("order_control", "entity_control", "identity")
STATES = FACTORIAL_STATES + CONTROL_STATES
CAPTURE_ROLES = (
    "focal_source",
    "focal_relation",
    "focal_target",
    "background_source",
    "background_relation",
    "background_target",
    "query_anchor",
    "query_operator",
    "answer_boundary",
)
WORLDS_PER_POOL_TEMPLATE = 8
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1016_query_factorial_atlas"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def state_factors(state: str) -> tuple[int, int]:
    if state.startswith("s"):
        return int(state[1]), int(state[4])
    return 0, 0


def add_comparison_fact(
    builder: PromptBuilder,
    template: int,
    source: str,
    target: str,
    prefix: str,
    label: str,
) -> None:
    if template == 0:
        builder.add(label + " statement:")
        builder.mark(prefix + "_source", source)
        builder.add(" is")
        builder.mark(prefix + "_relation", "higher")
        builder.add(" than")
        builder.mark(prefix + "_target", target)
        builder.add(".")
    elif template == 1:
        builder.add(label + " record:")
        builder.mark(prefix + "_source", source)
        builder.add(" ranks")
        builder.mark(prefix + "_relation", "above")
        builder.mark(prefix + "_target", target)
        builder.add(".")
    elif template == 2:
        builder.add(label + " relation:")
        builder.mark(prefix + "_source", source)
        builder.add(" stands")
        builder.mark(prefix + "_relation", "above")
        builder.mark(prefix + "_target", target)
        builder.add(".")
    else:
        builder.add(label + " fact:")
        builder.mark(prefix + "_target", target)
        builder.add(" is")
        builder.mark(prefix + "_relation", "below")
        builder.mark(prefix + "_source", source)
        builder.add(".")


def add_negation_fact(
    builder: PromptBuilder,
    template: int,
    source: str,
    target: str,
    prefix: str,
    label: str,
) -> None:
    descriptors = (
        ("present", "missing"),
        ("listed", "absent"),
        ("available", "unavailable"),
        ("included", "excluded"),
    )
    positive, negative = descriptors[template]
    builder.add(label + " status:")
    if template == 3:
        builder.mark(prefix + "_target", target)
        builder.add(" is " + negative + ";")
        builder.mark(prefix + "_source", source)
        builder.add(" is")
        builder.mark(prefix + "_relation", positive)
        builder.add(".")
    else:
        builder.mark(prefix + "_source", source)
        builder.add(" is")
        builder.mark(prefix + "_relation", positive)
        builder.add(";")
        builder.mark(prefix + "_target", target)
        builder.add(" is " + negative + ".")


def add_semantic_role_fact(
    builder: PromptBuilder,
    template: int,
    source: str,
    target: str,
    prefix: str,
    label: str,
) -> None:
    verbs = ("sent", "helped", "guided", "called")
    verb = verbs[template]
    builder.add(label + " event:")
    builder.mark(prefix + "_source", source)
    builder.mark(prefix + "_relation", verb)
    if template == 0:
        builder.add(" a signal to")
    else:
        builder.add("")
    builder.mark(prefix + "_target", target)
    builder.add(".")


def add_attribute_fact(
    builder: PromptBuilder,
    template: int,
    source: str,
    target: str,
    prefix: str,
    label: str,
) -> None:
    colors = (
        ("red", "blue"),
        ("crimson", "azure"),
        ("ruby", "navy"),
        ("rose", "cyan"),
    )
    positive, negative = colors[template]
    builder.add(label + " colors:")
    if template == 3:
        builder.mark(prefix + "_target", target)
        builder.add(" is " + negative + ";")
        builder.mark(prefix + "_source", source)
        builder.add(" is")
        builder.mark(prefix + "_relation", positive)
        builder.add(".")
    else:
        builder.mark(prefix + "_source", source)
        builder.add(" is")
        builder.mark(prefix + "_relation", positive)
        builder.add(";")
        builder.mark(prefix + "_target", target)
        builder.add(" is " + negative + ".")


def add_spatial_fact(
    builder: PromptBuilder,
    template: int,
    source: str,
    target: str,
    prefix: str,
    label: str,
) -> None:
    relations = ("left", "west", "western", "west")
    relation = relations[template]
    builder.add(label + " layout:")
    builder.mark(prefix + "_source", source)
    builder.add(" is")
    builder.mark(prefix + "_relation", relation)
    builder.add(" of")
    builder.mark(prefix + "_target", target)
    builder.add(".")


FACT_BUILDERS: dict[str, Callable[..., None]] = {
    "comparison": add_comparison_fact,
    "negation": add_negation_fact,
    "semantic_role": add_semantic_role_fact,
    "attribute_binding": add_attribute_fact,
    "spatial_relation": add_spatial_fact,
}


def add_query(
    builder: PromptBuilder,
    template: int,
    candidates: tuple[str, str],
    operator: str,
) -> None:
    first, second = candidates
    builder.add("\nCandidates: " + first + " " + second + ".")
    anchors = ("focal", "main", "target", "relevant")
    builder.add("\nQuestion about")
    builder.mark("query_anchor", anchors[template])
    builder.add(": which candidate is")
    builder.mark("query_operator", operator)
    builder.add("?")
    builder.add(
        "\nReply with exactly one candidate name and no other text."
        "\nAnswer:"
    )


def render_case(
    *,
    family: str,
    split: str,
    template: int,
    names: list[str],
    world_index: int,
    state: str,
) -> tuple[
    str,
    dict[str, tuple[int, int, str]],
    str,
    str,
    dict[str, Any],
]:
    fact_bit = world_index & 1
    candidate_order_bit = (world_index >> 1) & 1
    semantic_bit, lexical_bit = state_factors(state)
    a, b, c, d, e, f = names
    relation_zero, relation_one = ((a, b) if fact_bit == 0 else (b, a))
    background_zero, background_one = (
        (e, f) if state == "entity_control" else (c, d)
    )
    labels = LEXICAL_LABELS[split][family]
    operator = labels[lexical_bit][semantic_bit]

    builder = PromptBuilder()
    fact_builder = FACT_BUILDERS[family]

    def focal() -> None:
        fact_builder(
            builder,
            template,
            relation_zero,
            relation_one,
            "focal",
            "Focal",
        )

    def background() -> None:
        fact_builder(
            builder,
            template,
            background_zero,
            background_one,
            "background",
            "Background",
        )

    if state == "order_control":
        background()
        builder.add("\n")
        focal()
    else:
        focal()
        builder.add("\n")
        background()

    candidates = (
        (relation_one, relation_zero)
        if candidate_order_bit
        else (relation_zero, relation_one)
    )
    add_query(builder, template, candidates, operator)
    raw_prompt, spans = builder.finish()
    gold = relation_zero if semantic_bit == 0 else relation_one
    foil = relation_one if semantic_bit == 0 else relation_zero
    metadata = {
        "fact_bit": int(fact_bit),
        "candidate_order_bit": int(candidate_order_bit),
        "semantic_bit": int(semantic_bit),
        "lexical_bit": int(lexical_bit),
        "query_operator": operator,
        "lexical_pair": list(labels[lexical_bit]),
        "canonical_semantic_sign": 1 if fact_bit == 0 else -1,
    }
    return raw_prompt, spans, gold, foil, metadata


def render_mode(
    tokenizer,
    model_name: str,
    prompt_mode: str,
    raw_prompt: str,
) -> str:
    if prompt_mode == "raw":
        return raw_prompt
    if prompt_mode == "native_chat":
        return render_chat(tokenizer, model_name, raw_prompt)
    raise KeyError(prompt_mode)


def boundary_token_id(tokenizer, rendered: str, label: str) -> int:
    base = tokenizer.encode(rendered, add_special_tokens=False)
    extended = tokenizer.encode(
        rendered + " " + label,
        add_special_tokens=False,
    )
    if extended[:len(base)] != base or len(extended) != len(base) + 1:
        raise RuntimeError(
            f"answer {label!r} is not one token at answer boundary"
        )
    return int(extended[-1])


def build_case(
    *,
    tokenizer,
    model_name: str,
    prompt_mode: str,
    family: str,
    split: str,
    template: int,
    name_pool: int,
    world_index: int,
    unit_id: str,
    state: str,
    names: list[str],
) -> dict[str, Any]:
    raw_prompt, spans, gold, foil, metadata = render_case(
        family=family,
        split=split,
        template=template,
        names=names,
        world_index=world_index,
        state=state,
    )
    rendered = render_mode(
        tokenizer,
        model_name,
        prompt_mode,
        raw_prompt,
    )
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
            f"{family}/{state}: role drift {sorted(positions)}"
        )
    candidate_ids = {
        label: boundary_token_id(tokenizer, rendered, label)
        for label in names[:2]
    }
    candidate_ids.update({
        label: boundary_token_id(tokenizer, rendered, label)
        for label in names[2:]
        if label in {gold, foil}
    })
    candidate_ids = {
        gold: boundary_token_id(tokenizer, rendered, gold),
        foil: boundary_token_id(tokenizer, rendered, foil),
    }
    if candidate_ids[gold] == candidate_ids[foil]:
        raise RuntimeError("candidate boundary token collision")
    return {
        "schema_version": "phase1016_query_factorial_case.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "family": family,
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
            role: int(positions[role]) for role in CAPTURE_ROLES
        },
        "gold": gold,
        "foil": foil,
        "candidate_labels": [gold, foil],
        "candidate_token_ids": candidate_ids,
        "answer_text": " " + gold,
        "natural_gold_text": gold,
        "answer_token_ids": [candidate_ids[gold]],
        "semantic_step": 0,
        "protocol_prefix_ids": [],
        "explicit_response_map_present": False,
        **metadata,
    }


def edit_positions(left: list[int], right: list[int]) -> list[int]:
    if len(left) != len(right):
        raise RuntimeError(
            f"token length drift {len(left)} != {len(right)}"
        )
    return [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if a != b
    ]


def build_model_mode(
    model_name: str,
    prompt_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    units = []
    for family in FAMILIES:
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
                            f"{model_name}.{prompt_mode}.{family}."
                            f"{split}.t{template}.p{name_pool}."
                            f"w{world_index}"
                        )
                        by_state = {}
                        for state in STATES:
                            case = build_case(
                                tokenizer=tokenizer,
                                model_name=model_name,
                                prompt_mode=prompt_mode,
                                family=family,
                                split=split,
                                template=template,
                                name_pool=name_pool,
                                world_index=world_index,
                                unit_id=unit_id,
                                state=state,
                                names=names,
                            )
                            by_state[state] = case
                            cases.append(case)
                        base = by_state["s0_l0"]
                        identity = by_state["identity"]
                        if identity["input_ids"] != base["input_ids"]:
                            raise RuntimeError(f"{unit_id}: identity drift")
                        query_position = base["role_positions"]["query_operator"]
                        factorial_edits = {}
                        for left, right in (
                            ("s0_l0", "s1_l0"),
                            ("s0_l1", "s1_l1"),
                            ("s0_l0", "s0_l1"),
                            ("s1_l0", "s1_l1"),
                        ):
                            edits = edit_positions(
                                by_state[left]["input_ids"],
                                by_state[right]["input_ids"],
                            )
                            if edits != [query_position]:
                                raise RuntimeError(
                                    f"{unit_id}: {left}/{right} "
                                    f"edits={edits}, query={query_position}"
                                )
                            factorial_edits[f"{left}:{right}"] = edits
                        lengths = {
                            len(case["input_ids"])
                            for case in by_state.values()
                        }
                        if len(lengths) != 1:
                            raise RuntimeError(
                                f"{unit_id}: state length drift {lengths}"
                            )
                        units.append({
                            "schema_version": (
                                "phase1016_query_factorial_unit.v1"
                            ),
                            "phase": PHASE,
                            "protocol_revision": PROTOCOL_REVISION,
                            "model": model_name,
                            "prompt_mode": prompt_mode,
                            "family": family,
                            "split": split,
                            "template": int(template),
                            "name_pool": int(name_pool),
                            "world_index": int(world_index),
                            "unit_id": unit_id,
                            "record_ids": {
                                state: by_state[state]["record_id"]
                                for state in STATES
                            },
                            "query_operator_position": int(query_position),
                            "answer_boundary_position": int(
                                base["role_positions"]["answer_boundary"]
                            ),
                            "canonical_semantic_sign": int(
                                base["canonical_semantic_sign"]
                            ),
                            "factorial_edit_positions": factorial_edits,
                            "token_count": len(base["input_ids"]),
                        })
    return cases, units


def preregistration() -> dict[str, Any]:
    return {
        "schema_version": "phase1016_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "primary_goal": (
            "Map repeated internal response structure before proposing a "
            "mechanism equation."
        ),
        "factorial_design": {
            "semantic_axis": "binary query meaning at one token slot",
            "lexical_axis": (
                "semantic-preserving synonym family at the same token slot"
            ),
            "cells": list(FACTORIAL_STATES),
            "semantic_contrast": (
                "0.5*((h_s1_l0-h_s0_l0)+(h_s1_l1-h_s0_l1))"
            ),
            "lexical_contrast": (
                "0.5*((h_s0_l1-h_s0_l0)+(h_s1_l1-h_s1_l0))"
            ),
            "interaction_contrast": (
                "h_s1_l1-h_s1_l0-h_s0_l1+h_s0_l0"
            ),
            "interpretation": (
                "These are experimental contrasts, not a language-mechanism "
                "formula."
            ),
        },
        "controls": {
            "order_control": "same facts in reversed line order",
            "entity_control": "background-only entity replacement",
            "identity": "exact repeated forward",
        },
        "prompt_mode_calibration": {
            "modes": list(PROMPT_MODES),
            "selection_split": "discovery calibration subset only",
            "primary_metric": "generated first content word accuracy",
            "tie_breakers": [
                "candidate-pair accuracy",
                "mean candidate margin",
                "full-vocabulary next-token accuracy",
                "native_chat on exact tie",
            ],
            "confirmation_data_forbidden": True,
            "revision_note": (
                "Revision 2 treats leading chat-template whitespace as output "
                "formatting rather than a semantic error. Full-vocabulary "
                "next-token accuracy remains a reported diagnostic."
            ),
        },
        "capture": {
            "components": [
                "residual_stream",
                "attention_output",
                "mlp_output",
                "all_real_attention_heads_pre_o_proj",
            ],
            "roles": list(CAPTURE_ROLES),
            "raw_hidden_tensor_persistence": False,
            "precision": "bf16",
        },
        "analysis": {
            "primary_population": "all protocol units",
            "secondary_populations": [
                "all_four_candidate_predictions_correct",
                "all_four_full_vocabulary_predictions_correct",
            ],
            "selection_uses_behavior": False,
            "discovery_only_selection": True,
            "confirmation_is_held_out": True,
            "continuous_outputs_required": [
                "normalized_magnitude",
                "canonical_direction_consistency",
                "raw_direction_consistency",
                "lexical_family_direction_alignment",
                "semantic_over_lexical_prevalence",
                "semantic_minus_lexical_median",
                "interaction_ratio",
            ],
            "candidate_thresholds": {
                "canonical_direction_consistency_min": 0.45,
                "orientation_gain_min": 0.20,
                "lexical_family_alignment_min": 0.40,
                "semantic_over_lexical_prevalence_min": 0.70,
                "identity_max": 1e-6,
            },
            "repeated_core": {
                "discovery_panel_min": 4,
                "discovery_family_min": 2,
                "confirmation_panel_min": 2,
                "confirmation_family_min": 2,
            },
        },
        "automatic_continuation_gate": {
            "required_before_neuron_or_patch_followup": [
                "heldout repeated component-depth-role structure",
                "positive same-slot semantic-versus-lexical separation",
                "sufficient behavior-qualified observations",
            ],
            "failed_gate_action": (
                "diagnose behavior or measurement; do not force a closure test"
            ),
        },
        "claim_limits": [
            "Response is not causal transport.",
            "Canonical alignment is compatible with relative differential "
            "coding but does not prove it.",
            "Repeated physical components need not carry one invariant vector.",
            "No global language equation is tested in this phase.",
        ],
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "protocol").mkdir(parents=True, exist_ok=True)
    all_cases = []
    all_units = []
    for model_name in MODELS:
        for prompt_mode in PROMPT_MODES:
            cases, units = build_model_mode(model_name, prompt_mode)
            all_cases.extend(cases)
            all_units.extend(units)
            write_jsonl(
                OUT_ROOT
                / "protocol"
                / f"cases.{model_name}.{prompt_mode}.jsonl",
                cases,
            )
            write_jsonl(
                OUT_ROOT
                / "protocol"
                / f"units.{model_name}.{prompt_mode}.jsonl",
                units,
            )

    pre = preregistration()
    pre["protocol_digest"] = digest({
        "preregistration": pre,
        "case_fingerprints": [
            {
                "record_id": row["record_id"],
                "input_ids": row["input_ids"],
                "gold": row["gold"],
                "role_positions": row["role_positions"],
            }
            for row in all_cases
        ],
    })
    write_json(OUT_ROOT / "protocol" / "preregistration.json", pre)

    expected_units = (
        len(MODELS)
        * len(PROMPT_MODES)
        * len(FAMILIES)
        * sum(len(TEMPLATES_BY_SPLIT[s]) for s in SPLITS)
        * len(NAME_POOLS["discovery"])
        * WORLDS_PER_POOL_TEMPLATE
    )
    expected_cases = expected_units * len(STATES)
    discovery_names = {
        name for pool in NAME_POOLS["discovery"] for name in pool
    }
    confirmation_names = {
        name for pool in NAME_POOLS["confirmation"] for name in pool
    }
    audit = {
        "schema_version": "phase1016_protocol_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "case_count": len(all_cases),
        "expected_case_count": expected_cases,
        "unit_count": len(all_units),
        "expected_unit_count": expected_units,
        "cases_per_state": dict(Counter(
            row["state"] for row in all_cases
        )),
        "units_per_model_mode_split": {
            ":".join(key): value
            for key, value in Counter(
                (row["model"], row["prompt_mode"], row["split"])
                for row in all_units
            ).items()
        },
        "family_counts": dict(Counter(
            row["family"] for row in all_units
        )),
        "duplicate_record_ids": (
            len(all_cases)
            - len({row["record_id"] for row in all_cases})
        ),
        "duplicate_unit_ids": (
            len(all_units)
            - len({row["unit_id"] for row in all_units})
        ),
        "discovery_confirmation_name_overlap": len(
            discovery_names & confirmation_names
        ),
        "factorial_single_query_edit_count": sum(
            all(
                edits == [row["query_operator_position"]]
                for edits in row["factorial_edit_positions"].values()
            )
            for row in all_units
        ),
        "identity_exact_count": sum(
            next(
                case for case in all_cases
                if case["record_id"] == row["record_ids"]["s0_l0"]
            )["input_ids"]
            == next(
                case for case in all_cases
                if case["record_id"] == row["record_ids"]["identity"]
            )["input_ids"]
            for row in all_units
        ),
        "token_count_range": [
            min(row["token_count"] for row in all_units),
            max(row["token_count"] for row in all_units),
        ],
        "protocol_digest": pre["protocol_digest"],
    }
    audit["valid"] = all((
        audit["case_count"] == expected_cases,
        audit["unit_count"] == expected_units,
        audit["duplicate_record_ids"] == 0,
        audit["duplicate_unit_ids"] == 0,
        audit["discovery_confirmation_name_overlap"] == 0,
        audit["factorial_single_query_edit_count"] == expected_units,
        audit["identity_exact_count"] == expected_units,
    ))
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(canonical(audit))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
