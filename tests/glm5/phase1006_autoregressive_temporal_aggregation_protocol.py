#!/usr/bin/env python3
"""Freeze the Phase1006 pre-emission temporal aggregation denominator.

The protocol uses a new two-token answer surface.  It distinguishes the state
that predicts a token from the residual state created after that token has
already been appended.  Semantic position labels remain sealed until every
blind source set has been frozen.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1002_multitoken_protocol import NAMES as PHASE1002_NAMES
from phase1003_crossparadigm_protocol import (
    CALIBRATION_NAMES as PHASE1003_CALIBRATION_NAMES,
    CONFIRMATION_NAMES as PHASE1003_CONFIRMATION_NAMES,
    DISCOVERY_NAMES as PHASE1003_DISCOVERY_NAMES,
)
from phase1004_blind_causal_basis_protocol import (
    CONFIRMATION_NAMES as PHASE1004_CONFIRMATION_NAMES,
    DISCOVERY_NAMES as PHASE1004_DISCOVERY_NAMES,
    DOMAINS as PHASE1004_DOMAINS,
)


PHASE = 1006
PROTOCOL_REVISION = 4
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation")
DOMAIN = "paired_code"
SOURCE_DEPTH = 1
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
CODE_PAIRS = {
    "discovery": (
        ("bright", "glass"),
        ("dark", "cotton"),
        ("soft", "wood"),
        ("hard", "steel"),
    ),
    "confirmation": (
        ("warm", "gold"),
        ("cold", "iron"),
        ("heavy", "silk"),
        ("light", "clay"),
    ),
}
REVISION3_NAMES = (
    "Brooke", "Caroline", "Cassandra", "Charlotte", "Chelsea",
    "Christina", "Christine", "Cynthia", "Diane", "Dorothy",
    "Eleanor", "Elizabeth", "Ellen", "Erica", "Erin", "Esther",
    "Evelyn", "Faith", "Florence", "Frances", "Georgia", "Gloria",
    "Gwen", "Harper", "Heather", "Heidi", "Helen", "Isabel", "Janet",
    "Jasmine", "Jean", "Jenna", "Jessica", "Joan", "Joanna", "Joy",
    "Joyce", "Judith", "June", "Katherine", "Kathleen", "Kimberly",
    "Kristen", "Lana", "Lauren", "Leah", "Leslie", "Lori", "Louise",
    "Margaret", "Marilyn", "Marion", "Martha", "Melissa", "Mia",
    "Miranda", "Molly", "Monica", "Natalie", "Nora", "Patricia",
    "Paula", "Pearl", "Rita", "Rose", "Ruby", "Sally", "Savannah",
    "Sharon", "Sheila", "Shelby", "Sierra", "Stacy", "Stella",
    "Stephanie", "Summer", "Sylvia", "Tara", "Teresa", "Tiffany",
)
REVISION3_CODE_WORDS = {
    "smooth", "cedar", "rough", "marble", "amber", "linen", "silver",
    "canvas", "white", "stone", "black", "metal", "purple", "leather",
    "golden", "paper",
}
DISCOVERY_NAMES = (
    "Ada", "Adelaide", "Alexandra", "Ann", "Blair", "Cara", "Carolyn",
    "Cassidy", "Dana", "Danielle", "Deborah", "Denise", "Ella", "Ellie",
    "Elsa", "Emma", "Hope", "Jacqueline", "Jade", "Jill", "Juliet",
    "Lily",
)
CONFIRMATION_NAMES = (
    "Lucy", "Lydia", "Madison", "Mae", "Maggie", "Marie", "Maya",
    "Melanie", "Meredith", "Naomi", "Piper", "Regina", "Riley",
    "Samantha", "Sara", "Scarlett", "Serena", "Simone", "Sonia",
    "Sophia", "Sophie", "Suzanne",
)
WORLD_COUNT_PER_SPLIT = 11
SELECTED_PAIRS_PER_STRATUM = 2
ANSWER_PREFIX = {
    "qwen3": "Answer: ",
    "glm4": "\nAnswer: ",
    "deepseek7b": "Answer: ",
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1006_autoregressive_temporal_aggregation"
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_order(value: str, salt: str) -> str:
    return hashlib.sha256(
        f"phase1006:{salt}:{value}".encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
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


def phrase(parts: tuple[str, str] | list[str]) -> str:
    return f"{parts[0]} {parts[1]}"


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def render_user_prompt(
    template: int,
    entity0: str,
    code0: tuple[str, str] | list[str],
    entity1: str,
    code1: tuple[str, str] | list[str],
    query: str,
) -> str:
    value0 = phrase(code0)
    value1 = phrase(code1)
    instruction = (
        "Answer exactly in this form: Answer: [word1] [word2]. Replace the "
        "bracketed fields with the two lowercase code words. Do not add "
        "punctuation, explanation, or other words."
    )
    if template == 0:
        body = (
            f"Assignment sheet: {entity0} receives {value0}; "
            f"{entity1} receives {value1}.\n"
            f"Give the assigned two-word code for {query}."
        )
    elif template == 1:
        body = (
            f"Two-part labels pair {entity0} with {value0} and pair "
            f"{entity1} with {value1}.\n"
            f"State the complete label paired with {query}."
        )
    elif template == 2:
        body = (
            f"Lookup note one says {entity0}: {value0}. "
            f"Lookup note two says {entity1}: {value1}.\n"
            f"Return the exact two-word lookup value for {query}."
        )
    elif template == 3:
        body = (
            f"A registry associates {value0} with {entity0}; separately, "
            f"it associates {value1} with {entity1}.\n"
            f"Which full two-word value is associated with {query}?"
        )
    else:
        raise KeyError(template)
    return f"{body}\n{instruction}"


def answer_text(
    model_name: str,
    code: tuple[str, str] | list[str],
) -> str:
    return f"{ANSWER_PREFIX[model_name]}{phrase(code)}"


def decision_case(
    case: dict[str, Any],
    *,
    prefix_ids: list[int],
    logical_step: int,
) -> dict[str, Any]:
    if logical_step not in (0, 1, 2):
        raise ValueError(logical_step)
    result = dict(case)
    result["input_ids"] = list(case["input_ids"]) + list(prefix_ids)
    result["input_token_count"] = len(result["input_ids"])
    result["decision_step"] = logical_step
    result["decision_position"] = result["input_token_count"] - 1
    result["prefix_ids"] = list(prefix_ids)
    return result


def selected_directional_rows(
    model_name: str,
    split: str,
) -> list[dict[str, Any]]:
    model_root = OUT_ROOT / "protocol" / model_name
    cases = {
        row["record_id"]: row
        for row in read_jsonl(model_root / "cases.jsonl")
    }
    pairs = read_jsonl(
        model_root / f"{split}_selected_pairs.jsonl"
    )
    rows = []
    for pair_row in pairs:
        arm0 = cases[pair_row["arm0_record_id"]]
        arm1 = cases[pair_row["arm1_record_id"]]
        for direction, donor, target in (
            ("arm0_to_arm1", arm0, arm1),
            ("arm1_to_arm0", arm1, arm0),
        ):
            rows.append({
                "pair_id": pair_row["pair_id"],
                "model": model_name,
                "domain": DOMAIN,
                "split": split,
                "template": int(target["template"]),
                "direction": direction,
                "source": donor,
                "target": target,
            })
    return rows


def select_pairs(
    pairs: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in pairs:
        if row["split"] != split:
            continue
        key = (
            int(row["template"]),
            int(row["display_order"]),
            int(row["value_swap"]),
            int(row["query_role"]),
        )
        strata[key].append(row)
    selected = []
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_order(
                row["pair_id"],
                f"pair:{split}:{key}",
            ),
        )
        if len(ordered) < SELECTED_PAIRS_PER_STRATUM:
            raise RuntimeError(f"underfilled stratum {split}/{key}")
        selected.extend(ordered[:SELECTED_PAIRS_PER_STRATUM])
    expected = 16 * SELECTED_PAIRS_PER_STRATUM
    if len(selected) != expected:
        raise RuntimeError(f"{split}: selected {len(selected)} != {expected}")
    return selected


def prior_names() -> set[str]:
    return (
        set(PHASE1002_NAMES)
        | set(PHASE1003_DISCOVERY_NAMES)
        | set(PHASE1003_CONFIRMATION_NAMES)
        | set(PHASE1003_CALIBRATION_NAMES)
        | set(PHASE1004_DISCOVERY_NAMES)
        | set(PHASE1004_CONFIRMATION_NAMES)
        | set(REVISION3_NAMES)
    )


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    formal_names = set(DISCOVERY_NAMES) | set(CONFIRMATION_NAMES)
    if set(DISCOVERY_NAMES) & set(CONFIRMATION_NAMES):
        raise RuntimeError("discovery/confirmation name overlap")
    overlap = formal_names & prior_names()
    if overlap:
        raise RuntimeError(f"prior phase name overlap: {sorted(overlap)}")

    name_ids = {
        name: one_token_id(tokenizer, " " + name)
        for name in DISCOVERY_NAMES + CONFIRMATION_NAMES
    }
    if len(set(name_ids.values())) != len(name_ids):
        raise RuntimeError(f"{model_name}: name token collision")

    prior_values = {
        item
        for split_values in PHASE1004_DOMAINS.values()
        for values in split_values.values()
        for item in values
    }
    formal_words = {
        word
        for values in CODE_PAIRS.values()
        for code in values
        for word in code
    }
    if formal_words & (prior_values | REVISION3_CODE_WORDS):
        raise RuntimeError(
            "prior value overlap: "
            f"{sorted(formal_words & (prior_values | REVISION3_CODE_WORDS))}"
        )

    prompt_word_ids = {
        word: one_token_id(tokenizer, " " + word)
        for word in sorted(formal_words)
    }
    answer_ids: dict[str, dict[str, list[int]]] = {}
    candidate_ids: dict[str, list[dict[str, int]]] = {}
    semantic_steps: dict[str, list[int]] = {}
    protocol_prefix_ids: dict[str, list[int]] = {}
    for split, codes in CODE_PAIRS.items():
        answer_ids[split] = {}
        for code in codes:
            ids = [
                int(value)
                for value in tokenizer.encode(
                    answer_text(model_name, code),
                    add_special_tokens=False,
                )
            ]
            answer_ids[split][phrase(code)] = ids
        widths = {len(ids) for ids in answer_ids[split].values()}
        if len(widths) != 1:
            raise RuntimeError(
                f"{model_name}/{split}: answer width drift {widths}"
            )
        width = next(iter(widths))
        varying = [
            index
            for index in range(width)
            if len({
                ids[index] for ids in answer_ids[split].values()
            }) > 1
        ]
        if len(varying) != 2 or varying[1] != varying[0] + 1:
            raise RuntimeError(
                f"{model_name}/{split}: semantic steps {varying}"
            )
        if varying[1] != width - 1:
            raise RuntimeError(
                f"{model_name}/{split}: suffix after semantic words"
            )
        prefixes = {
            tuple(ids[:varying[0]])
            for ids in answer_ids[split].values()
        }
        if len(prefixes) != 1:
            raise RuntimeError(
                f"{model_name}/{split}: protocol prefix drift"
            )
        semantic_steps[split] = varying
        protocol_prefix_ids[split] = list(next(iter(prefixes)))
        for logical_step, absolute_step in enumerate(varying):
            if len({
                ids[absolute_step] for ids in answer_ids[split].values()
            }) != len(codes):
                raise RuntimeError(
                    f"{model_name}/{split}: answer collision "
                    f"at {absolute_step}"
                )
        candidate_ids[split] = [
            {
                code[logical_step]: int(
                    answer_ids[split][phrase(code)][absolute_step]
                )
                for code in codes
            }
            for logical_step, absolute_step in enumerate(varying)
        ]

    rng = random.Random(1006_20260724)
    worlds: dict[str, list[tuple[str, str]]] = {}
    for split, names in (
        ("discovery", DISCOVERY_NAMES),
        ("confirmation", CONFIRMATION_NAMES),
    ):
        shuffled = list(names)
        rng.shuffle(shuffled)
        worlds[split] = [
            tuple(shuffled[index:index + 2])
            for index in range(0, len(shuffled), 2)
        ]
        if len(worlds[split]) != WORLD_COUNT_PER_SPLIT:
            raise RuntimeError(f"{split}: world count drift")

    cases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    prompt_lengths: dict[tuple[str, int], set[int]] = defaultdict(set)
    for split, split_worlds in worlds.items():
        codes = CODE_PAIRS[split]
        code_combinations = list(itertools.combinations(codes, 2))
        for world_index, base_entities_tuple in enumerate(split_worlds):
            base_entities = list(base_entities_tuple)
            base_codes = [
                list(value)
                for value in code_combinations[
                    world_index % len(code_combinations)
                ]
            ]
            world_id = f"{split[:1]}w{world_index:02d}"
            for template, display_order, value_swap, query_role in (
                itertools.product(
                    TEMPLATES_BY_SPLIT[split],
                    (0, 1),
                    (0, 1),
                    (0, 1),
                )
            ):
                arms = []
                for entity_swap in (0, 1):
                    slot_entities = (
                        list(base_entities)
                        if not entity_swap
                        else [base_entities[1], base_entities[0]]
                    )
                    slot_codes = (
                        [list(value) for value in base_codes]
                        if not value_swap
                        else [
                            list(base_codes[1]),
                            list(base_codes[0]),
                        ]
                    )
                    query_entity = base_entities[query_role]
                    query_slot = slot_entities.index(query_entity)
                    gold_code = slot_codes[query_slot]
                    foil_code = slot_codes[1 - query_slot]
                    first_slot, second_slot = (
                        (0, 1) if display_order == 0 else (1, 0)
                    )
                    raw_prompt = render_user_prompt(
                        template,
                        slot_entities[first_slot],
                        slot_codes[first_slot],
                        slot_entities[second_slot],
                        slot_codes[second_slot],
                        query_entity,
                    )
                    rendered = render_chat(
                        tokenizer,
                        model_name,
                        raw_prompt,
                    )
                    ids = [
                        int(token_id)
                        for token_id in tokenizer.encode(
                            rendered,
                            add_special_tokens=False,
                        )
                    ]
                    raw_start = rendered.index(raw_prompt)
                    raw_end = raw_start + len(raw_prompt)
                    prefix_ids = [
                        int(token_id)
                        for token_id in tokenizer.encode(
                            rendered[:raw_start],
                            add_special_tokens=False,
                        )
                    ]
                    through_user_ids = [
                        int(token_id)
                        for token_id in tokenizer.encode(
                            rendered[:raw_end],
                            add_special_tokens=False,
                        )
                    ]
                    if (
                        ids[:len(prefix_ids)] != prefix_ids
                        or ids[:len(through_user_ids)] != through_user_ids
                    ):
                        raise RuntimeError(
                            f"{model_name}/{world_id}: user span drift"
                        )
                    user_content_start = len(prefix_ids)
                    user_content_end = len(through_user_ids)
                    if not user_content_start < user_content_end < len(ids):
                        raise RuntimeError(
                            f"{model_name}/{world_id}: invalid user span"
                        )
                    gold_text = phrase(gold_code)
                    gold_answer_ids = answer_ids[split][gold_text]
                    extended = [
                        int(token_id)
                        for token_id in tokenizer.encode(
                            rendered + answer_text(model_name, gold_code),
                            add_special_tokens=False,
                        )
                    ]
                    if extended != ids + gold_answer_ids:
                        raise RuntimeError(
                            f"{model_name}/{world_id}: answer boundary drift"
                        )

                    fact_entity_positions = {}
                    for entity in base_entities:
                        found = positions_of(ids, name_ids[entity])
                        expected = 2 if entity == query_entity else 1
                        if len(found) != expected:
                            raise RuntimeError(
                                f"{model_name}/{world_id}/{entity}: {found}"
                            )
                        fact_entity_positions[entity] = found[0]
                    query_positions = positions_of(
                        ids,
                        name_ids[query_entity],
                    )
                    if len(query_positions) != 2:
                        raise RuntimeError(
                            f"{model_name}/{world_id}: query positions"
                        )
                    code_positions = {}
                    for code in base_codes:
                        locations = []
                        for word in code:
                            found = positions_of(
                                ids,
                                prompt_word_ids[word],
                            )
                            if len(found) != 1:
                                raise RuntimeError(
                                    f"{model_name}/{world_id}/{word}: {found}"
                                )
                            locations.append(found[0])
                        code_positions[phrase(code)] = locations

                    prompt_lengths[(split, template)].add(len(ids))
                    record_id = (
                        f"{model_name}.{world_id}.t{template}."
                        f"d{display_order}.v{value_swap}.q{query_role}."
                        f"e{entity_swap}"
                    )
                    sealed_roles = {
                        "query_name": query_positions[-1],
                        "fact_entity_query": (
                            fact_entity_positions[query_entity]
                        ),
                        "fact_entity_other": (
                            fact_entity_positions[
                                base_entities[1 - query_role]
                            ]
                        ),
                        "gold_word0": code_positions[gold_text][0],
                        "gold_word1": code_positions[gold_text][1],
                        "foil_word0": (
                            code_positions[phrase(foil_code)][0]
                        ),
                        "foil_word1": (
                            code_positions[phrase(foil_code)][1]
                        ),
                    }
                    case = {
                        "schema_version": (
                            "phase1006_temporal_protocol_case.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "domain": DOMAIN,
                        "split": split,
                        "record_id": record_id,
                        "world_id": world_id,
                        "template": template,
                        "display_order": display_order,
                        "value_swap": value_swap,
                        "query_role": query_role,
                        "entity_swap": entity_swap,
                        "raw_prompt": raw_prompt,
                        "rendered_prompt": rendered,
                        "input_ids": ids,
                        "raw_prompt_token_count": len(ids),
                        "user_content_start": user_content_start,
                        "user_content_end": user_content_end,
                        "base_entities": base_entities,
                        "base_codes": [
                            phrase(value) for value in base_codes
                        ],
                        "query_entity": query_entity,
                        "gold": gold_text,
                        "gold_parts": list(gold_code),
                        "foil": phrase(foil_code),
                        "foil_parts": list(foil_code),
                        "answer_text": answer_text(model_name, gold_code),
                        "answer_token_ids": gold_answer_ids,
                        "semantic_steps": semantic_steps[split],
                        "protocol_prefix_ids": protocol_prefix_ids[split],
                        "termination_step": len(gold_answer_ids),
                        "candidate_ids_by_step": candidate_ids[split],
                        "sealed_semantic_role_positions": sealed_roles,
                    }
                    cases.append(case)
                    arms.append(case)
                if arms[0]["gold"] == arms[1]["gold"]:
                    raise RuntimeError(
                        f"{model_name}/{world_id}: arm answer did not flip"
                    )
                pair_id = (
                    f"{model_name}.{world_id}.t{template}."
                    f"d{display_order}.v{value_swap}.q{query_role}"
                )
                pairs.append({
                    "schema_version": (
                        "phase1006_temporal_protocol_pair.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": DOMAIN,
                    "split": split,
                    "pair_id": pair_id,
                    "template": template,
                    "display_order": display_order,
                    "value_swap": value_swap,
                    "query_role": query_role,
                    "arm0_record_id": arms[0]["record_id"],
                    "arm1_record_id": arms[1]["record_id"],
                })

    width_audit = {
        f"{split}.t{template}": sorted(widths)
        for (split, template), widths in prompt_lengths.items()
    }
    if any(len(widths) != 1 for widths in width_audit.values()):
        raise RuntimeError(
            f"{model_name}: prompt width drift {width_audit}"
        )

    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "pairs.jsonl", pairs)
    selected_counts = {}
    for split in SPLITS:
        selected = select_pairs(pairs, split)
        write_jsonl(
            model_root / f"{split}_selected_pairs.jsonl",
            selected,
        )
        selected_counts[split] = len(selected)

    summary = {
        "schema_version": "phase1006_temporal_protocol_model.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "case_count": len(cases),
        "pair_count": len(pairs),
        "selected_pair_counts": selected_counts,
        "selected_direction_counts": {
            split: selected_counts[split] * 2 for split in SPLITS
        },
        "prompt_widths": width_audit,
        "name_count": len(formal_names),
        "prior_name_overlap": [],
        "phase1004_value_overlap": [],
        "answer_ids": answer_ids,
        "candidate_ids": candidate_ids,
        "semantic_steps": semantic_steps,
        "protocol_prefix_ids": protocol_prefix_ids,
        "answer_prefix_text": ANSWER_PREFIX[model_name],
        "time_direction_audit": {
            "step0_predictor": (
                "the final naturally generated protocol-prefix token "
                "predicts semantic answer word 0"
            ),
            "step1_predictor": (
                "semantic answer word 0 predicts semantic answer word 1"
            ),
            "step2_predictor": (
                "semantic answer word 1 predicts effective end-of-turn"
            ),
            "natural_protocol_prefix_precedes_semantic_steps": True,
            "generated_token_cannot_cause_itself": True,
        },
    }
    write_json(model_root / "summary.json", summary)
    return summary


def build_protocol() -> dict[str, Any]:
    summaries = [build_model(model_name) for model_name in MODELS]
    payload = {
        "schema_version": "phase1006_temporal_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Pre-emission autoregressive temporal aggregation causal graph"
        ),
        "models_in_required_execution_order": list(MODELS),
        "domain": DOMAIN,
        "code_pairs": {
            split: [list(value) for value in values]
            for split, values in CODE_PAIRS.items()
        },
        "templates_by_split": {
            split: list(values)
            for split, values in TEMPLATES_BY_SPLIT.items()
        },
        "world_count_per_split": WORLD_COUNT_PER_SPLIT,
        "selected_pairs_per_stratum": SELECTED_PAIRS_PER_STRATUM,
        "model_summaries": summaries,
        "causal_time_order": [
            "rendered prompt -> naturally generated constant Answer prefix",
            "context plus natural Answer prefix -> semantic answer word 0",
            "context plus prefix plus answer word 0 -> answer word 1",
            "context plus prefix plus both answer words -> end-of-turn",
        ],
        "formal_stages": [
            "behavior and exact natural-surface qualification",
            "label-blind depth-1 multi-position source reconstruction",
            "independent confirmation with unseen names, codes, templates",
            "common-prefix prompt-source versus token-feedback factorial",
            "prompt-cache global key/value arm decomposition",
            "all-layer answer-boundary attention/MLP/residual screen",
            "natural source rollout and immediate EOS",
            "cross-model relative functional-topology audit",
            "Qwen3 BF16 audit only after an 8-bit parent result",
        ],
        "source_rule": {
            "event_universe": (
                "every physical position in the original user-content span "
                "at residual depth 1; chat assistant/thinking suffix and "
                "all answer positions are excluded; no semantic labels"
            ),
            "ranking": [
                "descending leave-one-out restored-target sequence rate",
                "descending median leave-one-out mediation over two steps",
                "descending single-position donor sequence rate",
                "ascending physical position",
            ],
            "ranking_screen_n_per_template": 16,
            "ranking_screen_selection": (
                "ascending phase1006 stable hash of recipient record id "
                "within model/split/template"
            ),
            "frozen_evaluation_n_per_template": 32,
            "joint_build": (
                "add ranked positions until two-step donor sequence rate "
                ">= 0.80 and median normalized transfer >= 0.50, then "
                "perform one reverse-delete pass"
            ),
            "labels_revealed_after_freeze": True,
        },
        "revision_audit": {
            "revision_1": (
                "The initial static protocol applied every single-position "
                "and leave-one-out ranking intervention to all 32 directions "
                "per template. Before any model was loaded, this was reduced "
                "to a frozen 16-direction ranking screen followed by a "
                "32-direction evaluation of the selected set. Event types, "
                "thresholds, data, controls, and confirmation remain "
                "unchanged."
            ),
            "revision_1_model_result_observed": False,
            "revision_1_retained_at": (
                "protocol_revision3_behavior_failure/"
                "protocol_revision1_pre_execution.json"
            ),
            "revision_2": (
                "Static code review before model loading found that the "
                "rendered chat prompt includes assistant/thinking control "
                "tokens after the user content. Those positions already form "
                "the pre-emission decision boundary and would confound source "
                "with receiver. Revision 3 restricts blind source discovery "
                "to the tokenizer-verified user-content span. The downstream "
                "assistant boundary remains eligible only as a receiver."
            ),
            "revision_2_model_result_observed": False,
            "revision_2_retained_at": (
                "protocol_revision3_behavior_failure/"
                "protocol_revision2_pre_source_boundary_fix.json"
            ),
            "revision_3": (
                "Behavior-only execution showed that a bare two-word answer "
                "surface was not the models' stable natural interface. Qwen3 "
                "usually emitted two words followed by a chat end marker, "
                "GLM4 inserted a newline and later emitted a turn marker, "
                "and DeepSeek7B often began an explanatory answer. No source "
                "or receiver intervention was executed under revision 3. "
                "Revision 4 therefore uses new names, new code words, new "
                "templates, and a tokenizer-verified model-specific natural "
                "Answer prefix. Effective termination includes the model's "
                "native end-of-turn tokens rather than only a global EOS."
            ),
            "revision_3_behavior_observed": True,
            "revision_3_causal_result_observed": False,
            "revision_3_protocol_retained_at": (
                "protocol_revision3_behavior_failure"
            ),
            "revision_3_behavior_retained_at": (
                "behavior_revision3_failed_surface"
            ),
            "revision_4_is_formal_protocol": True,
        },
        "temporal_factorial": {
            "TT": "target prompt state plus target word-0 prefix",
            "ST": "source-patched prompt state plus target word-0 prefix",
            "TD": "target prompt state plus donor word-0 prefix",
            "SD": "source-patched prompt state plus donor word-0 prefix",
            "interaction": "m_SD - m_ST - m_TD + m_TT",
            "interpretation": (
                "ST isolates direct prompt-state transport; TD isolates "
                "generated-token feedback; SD measures their joint effect"
            ),
        },
        "cache_factorial": {
            "boundary": (
                "prompt cache is built before answer word 0; word 0 is "
                "then processed as the current token to predict word 1"
            ),
            "arms": [
                "target K + target V",
                "source K + target V",
                "target K + source V",
                "source K + source V",
            ],
            "head_or_position_localization_gate": (
                "global cache source effect and a repeated component parent "
                "must pass before any finer decomposition"
            ),
        },
        "receiver_rule": {
            "event_universe": (
                "all layers x attention/MLP/residual at the current "
                "pre-emission decision position"
            ),
            "discovery_screen_n": 16,
            "maximum_frozen_events_per_step": 12,
            "confirmation_n": 64,
            "component_parent_required_for_head_search": True,
        },
        "thresholds": {
            "behavior_each_step_candidate_accuracy": 0.95,
            "behavior_natural_exact_rate": 0.90,
            "behavior_immediate_eos_rate": 0.90,
            "source_joint_donor_sequence_rate": 0.80,
            "source_joint_median_transfer": 0.50,
            "same_answer_target_sequence_rate": 0.95,
            "noop_target_sequence_rate": 0.99,
            "receiver_median_mediation": 0.10,
            "receiver_mean_sufficiency_transfer": 0.10,
            "natural_source_donor_rate": 0.70,
            "natural_source_immediate_eos_rate": 0.90,
        },
        "controls": {
            "new_names": True,
            "new_code_values": True,
            "new_templates": True,
            "new_multitoken_domain": True,
            "revision4_disjoint_from_revision3_names": True,
            "revision4_disjoint_from_revision3_code_values": True,
            "discovery_confirmation_name_disjoint": True,
            "discovery_confirmation_value_disjoint": True,
            "discovery_confirmation_template_disjoint": True,
            "semantic_labels_hidden_during_selection": True,
            "same_answer_cross_world_donor": True,
            "target_noop": True,
            "natural_rollout": True,
            "diagnostic_lens_selects_events": False,
        },
        "forbidden_claims": [
            "the generated token residual caused that same token",
            "0/72 rules out every form of compression",
            "a late residual event is an attention or MLP mechanism",
            "three models plus one precision branch are four models",
            "small-model bias has a numeric percentage without measurement",
            "a measurement equation is the native language law",
        ],
        "claim_boundary": (
            "A positive result may establish a repeated functional "
            "pre-emission aggregation topology in this controlled two-token "
            "task. It cannot establish a neuron-level language law, general "
            "knowledge representation, or an intelligence theory."
        ),
        "preregistration_digest": None,
    }
    digest_payload = dict(payload)
    digest_payload["preregistration_digest"] = None
    payload["preregistration_digest"] = digest(digest_payload)
    write_json(OUT_ROOT / "protocol" / "protocol.json", payload)
    return payload


def main() -> None:
    payload = build_protocol()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
