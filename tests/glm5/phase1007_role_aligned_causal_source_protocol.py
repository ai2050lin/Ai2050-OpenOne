#!/usr/bin/env python3
"""Freeze Phase1007 role-aligned minimal-counterfactual source protocol."""
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
from phase1004_blind_causal_basis_protocol import DOMAINS as PHASE1004_DOMAINS
from phase1006_autoregressive_temporal_aggregation_protocol import (
    ANSWER_PREFIX,
    CODE_PAIRS as PHASE1006_CODE_PAIRS,
    REVISION3_CODE_WORDS,
    prior_names,
)


PHASE = 1007
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation")
CONTRASTS = ("binding_flip", "query_flip")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
CODE_PAIRS = {
    "discovery": (
        ("clear", "quartz"),
        ("dense", "velvet"),
        ("sharp", "bronze"),
        ("mild", "coral"),
    ),
    "confirmation": (
        ("quiet", "pearl"),
        ("rapid", "moss"),
        ("fresh", "copper"),
        ("plain", "ivory"),
    ),
}
DISCOVERY_NAMES = (
    "Alan", "Alec", "Andy", "Ben", "Brad", "Carl", "Chris",
    "Christopher", "Damian", "Eli", "Evan", "Finn", "Fred", "Gary",
    "Gordon", "Grant", "Greg", "Howard", "Hugh", "Ivan", "Jake",
    "James", "Jeff", "Jeremy", "Jerry", "Jesse", "Jim", "Joe",
    "Jordan", "Josh",
)
CONFIRMATION_NAMES = (
    "Joshua", "Justin", "Keith", "Ken", "Kenneth", "Kyle", "Lee",
    "Leo", "Logan", "Louis", "Marcus", "Matt", "Matthew", "Max",
    "Nicholas", "Nick", "Oliver", "Oscar", "Patrick", "Ralph", "Ray",
    "Raymond", "Rick", "Roger", "Ron", "Ross", "Russell", "Seth",
    "Shane", "Stanley",
)
GROUPS_PER_SPLIT = 5
VARIANTS_PER_GROUP = 2
FORMAL_UNITS_PER_STRATUM = 4
HOLDOUT_UNITS_PER_STRATUM = 4
SOURCE_DEPTH = 1
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1007_role_aligned_causal_source"
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
        f"phase1007:{salt}:{value}".encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
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


def phrase(code: tuple[str, str] | list[str]) -> str:
    return f"{code[0]} {code[1]}"


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def answer_text(model_name: str, code: tuple[str, str] | list[str]) -> str:
    return f"{ANSWER_PREFIX[model_name]}{phrase(code)}"


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
        "Reply exactly as Answer: [word1] [word2]. Replace the brackets "
        "with the requested lowercase two-word code. Add nothing else."
    )
    if template == 0:
        body = (
            f"Code record one: {entity0} has {value0}. "
            f"Code record two: {entity1} has {value1}.\n"
            f"Requested person: {query}."
        )
    elif template == 1:
        body = (
            f"The full code beside {entity0} is {value0}; the full code "
            f"beside {entity1} is {value1}.\n"
            f"Return the code beside {query}."
        )
    elif template == 2:
        body = (
            f"Registry entry A assigns {value0} to {entity0}. Registry "
            f"entry B assigns {value1} to {entity1}.\n"
            f"Look up the complete code assigned to {query}."
        )
    elif template == 3:
        body = (
            f"In the paired ledger, {entity0} maps to {value0}, while "
            f"{entity1} maps to {value1}.\n"
            f"Give the two-part ledger value for {query}."
        )
    else:
        raise KeyError(template)
    return f"{body}\n{instruction}"


def decision_case(
    case: dict[str, Any],
    *,
    semantic_prefix: list[int],
    logical_step: int,
) -> dict[str, Any]:
    result = dict(case)
    prefix = (
        [int(value) for value in case["protocol_prefix_ids"]]
        + [int(value) for value in semantic_prefix]
    )
    result["input_ids"] = list(case["input_ids"]) + prefix
    result["input_token_count"] = len(result["input_ids"])
    result["decision_step"] = logical_step
    result["decision_position"] = result["input_token_count"] - 1
    result["prefix_ids"] = prefix
    return result


def semantic_answer_ids(case: dict[str, Any]) -> list[int]:
    return [
        int(case["answer_token_ids"][int(index)])
        for index in case["semantic_steps"]
    ]


def selected_directional_rows(
    model_name: str,
    split: str,
    template: int,
    contrast: str,
    partition: str = "formal",
) -> list[dict[str, Any]]:
    model_root = OUT_ROOT / "protocol" / model_name
    cases = {
        row["record_id"]: row
        for row in read_jsonl(model_root / "cases.jsonl")
    }
    units = read_jsonl(
        model_root / f"{split}_{partition}_units.jsonl"
    )
    rows = []
    for unit in units:
        if (
            int(unit["template"]) != template
            or unit["contrast"] != contrast
        ):
            continue
        worlds = unit["world_cases"]
        for direction, state, opposite in (
            ("base_to_counterfactual", "base", "counterfactual"),
            ("counterfactual_to_base", "counterfactual", "base"),
        ):
            rows.append({
                "unit_id": unit["unit_id"],
                "model": model_name,
                "split": split,
                "template": template,
                "contrast": contrast,
                "partition": partition,
                "direction": direction,
                "target": cases[worlds["target"][state]],
                "within_donor": cases[worlds["target"][opposite]],
                "cross_same": cases[worlds["nuisance"][state]],
                "cross_different": cases[worlds["nuisance"][opposite]],
                "nuisance2_same": cases[worlds["nuisance2"][state]],
            })
    expected = 32
    if len(rows) != expected:
        raise RuntimeError(
            f"{model_name}/{split}/t{template}/{contrast}/"
            f"{partition}: {len(rows)} != {expected}"
        )
    return rows


def prior_code_words() -> set[str]:
    phase1004 = {
        item
        for domain in PHASE1004_DOMAINS.values()
        for values in domain.values()
        for item in values
    }
    phase1006 = {
        word
        for values in PHASE1006_CODE_PAIRS.values()
        for code in values
        for word in code
    }
    return phase1004 | set(REVISION3_CODE_WORDS) | phase1006


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    formal_names = set(DISCOVERY_NAMES) | set(CONFIRMATION_NAMES)
    if set(DISCOVERY_NAMES) & set(CONFIRMATION_NAMES):
        raise RuntimeError("discovery/confirmation name overlap")
    name_overlap = formal_names & prior_names()
    if name_overlap:
        raise RuntimeError(f"prior name overlap: {sorted(name_overlap)}")
    name_ids = {
        name: one_token_id(tokenizer, " " + name)
        for name in DISCOVERY_NAMES + CONFIRMATION_NAMES
    }
    if len(set(name_ids.values())) != len(name_ids):
        raise RuntimeError(f"{model_name}: name token collision")

    formal_words = {
        word
        for codes in CODE_PAIRS.values()
        for code in codes
        for word in code
    }
    word_overlap = formal_words & prior_code_words()
    if word_overlap:
        raise RuntimeError(f"prior code overlap: {sorted(word_overlap)}")
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
        if (
            len(varying) != 2
            or varying[1] != varying[0] + 1
            or varying[1] != width - 1
        ):
            raise RuntimeError(
                f"{model_name}/{split}: semantic steps {varying}"
            )
        prefixes = {
            tuple(ids[:varying[0]])
            for ids in answer_ids[split].values()
        }
        if len(prefixes) != 1:
            raise RuntimeError("protocol prefix drift")
        semantic_steps[split] = varying
        protocol_prefix_ids[split] = list(next(iter(prefixes)))
        candidate_ids[split] = [
            {
                code[logical_step]: int(
                    answer_ids[split][phrase(code)][absolute_step]
                )
                for code in codes
            }
            for logical_step, absolute_step in enumerate(varying)
        ]

    rng = random.Random(1007_20260724)
    groups: dict[str, list[dict[str, list[str]]]] = {}
    for split, names in (
        ("discovery", DISCOVERY_NAMES),
        ("confirmation", CONFIRMATION_NAMES),
    ):
        shuffled = list(names)
        rng.shuffle(shuffled)
        groups[split] = []
        for index in range(GROUPS_PER_SPLIT):
            chunk = shuffled[index * 6:(index + 1) * 6]
            groups[split].append({
                "target": chunk[0:2],
                "nuisance": chunk[2:4],
                "nuisance2": chunk[4:6],
            })

    cases: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    prompt_widths: dict[tuple[str, int], set[int]] = defaultdict(set)
    code_combinations = {
        split: list(itertools.combinations(codes, 2))
        for split, codes in CODE_PAIRS.items()
    }
    for split, split_groups in groups.items():
        for group_index, group in enumerate(split_groups):
            for variant in range(VARIANTS_PER_GROUP):
                codes = code_combinations[split][
                    (group_index * VARIANTS_PER_GROUP + variant)
                    % len(code_combinations[split])
                ]
                base_codes = [list(codes[0]), list(codes[1])]
                for template, contrast, display_order, query_base in (
                    itertools.product(
                        TEMPLATES_BY_SPLIT[split],
                        CONTRASTS,
                        (0, 1),
                        (0, 1),
                    )
                ):
                    unit_id = (
                        f"{model_name}.{split[:1]}g{group_index:02d}."
                        f"v{variant}.t{template}.{contrast}."
                        f"d{display_order}.q{query_base}"
                    )
                    world_cases: dict[str, dict[str, str]] = {}
                    for world_role in ("target", "nuisance", "nuisance2"):
                        entities = list(group[world_role])
                        world_cases[world_role] = {}
                        for state_index, state in enumerate(
                            ("base", "counterfactual")
                        ):
                            if contrast == "binding_flip":
                                assigned_codes = (
                                    [list(value) for value in base_codes]
                                    if state_index == 0
                                    else [
                                        list(base_codes[1]),
                                        list(base_codes[0]),
                                    ]
                                )
                                query_role = query_base
                            else:
                                assigned_codes = [
                                    list(value) for value in base_codes
                                ]
                                query_role = (
                                    query_base
                                    if state_index == 0
                                    else 1 - query_base
                                )
                            query_entity = entities[query_role]
                            gold_code = assigned_codes[query_role]
                            foil_code = assigned_codes[1 - query_role]
                            first, second = (
                                (0, 1) if display_order == 0 else (1, 0)
                            )
                            raw_prompt = render_user_prompt(
                                int(template),
                                entities[first],
                                assigned_codes[first],
                                entities[second],
                                assigned_codes[second],
                                query_entity,
                            )
                            rendered = render_chat(
                                tokenizer, model_name, raw_prompt
                            )
                            ids = [
                                int(value)
                                for value in tokenizer.encode(
                                    rendered,
                                    add_special_tokens=False,
                                )
                            ]
                            raw_start = rendered.index(raw_prompt)
                            raw_end = raw_start + len(raw_prompt)
                            prefix_ids = [
                                int(value)
                                for value in tokenizer.encode(
                                    rendered[:raw_start],
                                    add_special_tokens=False,
                                )
                            ]
                            through_user_ids = [
                                int(value)
                                for value in tokenizer.encode(
                                    rendered[:raw_end],
                                    add_special_tokens=False,
                                )
                            ]
                            if (
                                ids[:len(prefix_ids)] != prefix_ids
                                or ids[:len(through_user_ids)]
                                != through_user_ids
                            ):
                                raise RuntimeError("user span drift")
                            user_start = len(prefix_ids)
                            user_end = len(through_user_ids)
                            if not user_start < user_end < len(ids):
                                raise RuntimeError("invalid user span")

                            gold_text = phrase(gold_code)
                            gold_answer_ids = answer_ids[split][gold_text]
                            extended = [
                                int(value)
                                for value in tokenizer.encode(
                                    rendered
                                    + answer_text(model_name, gold_code),
                                    add_special_tokens=False,
                                )
                            ]
                            if extended != ids + gold_answer_ids:
                                raise RuntimeError("answer boundary drift")

                            fact_positions = {}
                            for entity in entities:
                                found = positions_of(
                                    ids, name_ids[entity]
                                )
                                expected = (
                                    2 if entity == query_entity else 1
                                )
                                if len(found) != expected:
                                    raise RuntimeError(
                                        f"{entity}: positions {found}"
                                    )
                                fact_positions[entity] = found[0]
                            query_positions = positions_of(
                                ids, name_ids[query_entity]
                            )
                            code_positions = {}
                            for code in assigned_codes:
                                locations = []
                                for word in code:
                                    found = positions_of(
                                        ids, prompt_word_ids[word]
                                    )
                                    if len(found) != 1:
                                        raise RuntimeError(
                                            f"{word}: positions {found}"
                                        )
                                    locations.append(found[0])
                                code_positions[phrase(code)] = locations

                            record_id = (
                                f"{unit_id}.{world_role}.{state}"
                            )
                            sealed_roles = {
                                "query_name": query_positions[-1],
                                "fact_entity_query": (
                                    fact_positions[query_entity]
                                ),
                                "fact_entity_other": (
                                    fact_positions[
                                        entities[1 - query_role]
                                    ]
                                ),
                                "gold_word0": (
                                    code_positions[gold_text][0]
                                ),
                                "gold_word1": (
                                    code_positions[gold_text][1]
                                ),
                                "foil_word0": (
                                    code_positions[
                                        phrase(foil_code)
                                    ][0]
                                ),
                                "foil_word1": (
                                    code_positions[
                                        phrase(foil_code)
                                    ][1]
                                ),
                            }
                            case = {
                                "schema_version": (
                                    "phase1007_role_aligned_case.v1"
                                ),
                                "phase": PHASE,
                                "model": model_name,
                                "split": split,
                                "record_id": record_id,
                                "unit_id": unit_id,
                                "world_role": world_role,
                                "state": state,
                                "contrast": contrast,
                                "template": int(template),
                                "display_order": display_order,
                                "query_base": query_base,
                                "query_role": query_role,
                                "raw_prompt": raw_prompt,
                                "rendered_prompt": rendered,
                                "input_ids": ids,
                                "raw_prompt_token_count": len(ids),
                                "user_content_start": user_start,
                                "user_content_end": user_end,
                                "entities": entities,
                                "assigned_codes": [
                                    phrase(value)
                                    for value in assigned_codes
                                ],
                                "query_entity": query_entity,
                                "gold": gold_text,
                                "gold_parts": list(gold_code),
                                "foil": phrase(foil_code),
                                "foil_parts": list(foil_code),
                                "answer_text": answer_text(
                                    model_name, gold_code
                                ),
                                "answer_token_ids": gold_answer_ids,
                                "semantic_steps": semantic_steps[split],
                                "protocol_prefix_ids": (
                                    protocol_prefix_ids[split]
                                ),
                                "candidate_ids_by_step": (
                                    candidate_ids[split]
                                ),
                                "sealed_semantic_role_positions": (
                                    sealed_roles
                                ),
                            }
                            cases.append(case)
                            world_cases[world_role][state] = record_id
                            prompt_widths[(split, int(template))].add(
                                len(ids)
                            )
                    units.append({
                        "schema_version": (
                            "phase1007_role_aligned_unit.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "split": split,
                        "unit_id": unit_id,
                        "group_index": group_index,
                        "variant": variant,
                        "template": int(template),
                        "contrast": contrast,
                        "display_order": display_order,
                        "query_base": query_base,
                        "base_codes": [
                            phrase(value) for value in base_codes
                        ],
                        "world_cases": world_cases,
                    })

    width_audit = {
        f"{split}.t{template}": sorted(widths)
        for (split, template), widths in prompt_widths.items()
    }
    if any(len(widths) != 1 for widths in width_audit.values()):
        raise RuntimeError(
            f"{model_name}: prompt width drift {width_audit}"
        )

    formal_units = []
    holdout_units = []
    strata: dict[tuple[str, int, str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for unit in units:
        key = (
            unit["split"],
            int(unit["template"]),
            unit["contrast"],
            int(unit["display_order"]),
            int(unit["query_base"]),
        )
        strata[key].append(unit)
    for key, rows in sorted(strata.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_order(
                row["unit_id"], f"unit:{key}"
            ),
        )
        required = (
            FORMAL_UNITS_PER_STRATUM
            + HOLDOUT_UNITS_PER_STRATUM
        )
        if len(ordered) < required:
            raise RuntimeError(f"underfilled stratum {key}")
        formal_units.extend(
            ordered[:FORMAL_UNITS_PER_STRATUM]
        )
        holdout_units.extend(
            ordered[
                FORMAL_UNITS_PER_STRATUM:required
            ]
        )
    formal_ids = {unit["unit_id"] for unit in formal_units}
    holdout_ids = {unit["unit_id"] for unit in holdout_units}
    if formal_ids & holdout_ids:
        raise RuntimeError("formal/holdout unit overlap")

    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "units.jsonl", units)
    for split in SPLITS:
        write_jsonl(
            model_root / f"{split}_formal_units.jsonl",
            [row for row in formal_units if row["split"] == split],
        )
        write_jsonl(
            model_root / f"{split}_holdout_units.jsonl",
            [row for row in holdout_units if row["split"] == split],
        )

    summary = {
        "schema_version": "phase1007_role_aligned_model_protocol.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(units),
        "formal_unit_count": len(formal_units),
        "holdout_unit_count": len(holdout_units),
        "formal_holdout_overlap": 0,
        "prompt_widths": width_audit,
        "answer_ids": answer_ids,
        "semantic_steps": semantic_steps,
        "protocol_prefix_ids": protocol_prefix_ids,
        "prior_name_overlap": [],
        "prior_code_overlap": [],
    }
    write_json(model_root / "summary.json", summary)
    return summary


def build_protocol() -> dict[str, Any]:
    summaries = [build_model(model_name) for model_name in MODELS]
    payload = {
        "schema_version": "phase1007_role_aligned_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Role-aligned minimal counterfactual and controlled-delta source"
        ),
        "models_in_required_execution_order": list(MODELS),
        "splits": list(SPLITS),
        "contrasts": list(CONTRASTS),
        "templates_by_split": {
            split: list(values)
            for split, values in TEMPLATES_BY_SPLIT.items()
        },
        "code_pairs": {
            split: [list(value) for value in values]
            for split, values in CODE_PAIRS.items()
        },
        "model_summaries": summaries,
        "counterfactual_worlds": {
            "target": (
                "recipient world with base and minimally changed state"
            ),
            "nuisance": (
                "different names, paired base/counterfactual states"
            ),
            "nuisance2": (
                "second different-name world used for same-answer "
                "nuisance delta"
            ),
        },
        "contrast_definitions": {
            "binding_flip": (
                "same entities, query, template, code vocabulary, and "
                "protocol; only the two code assignments are swapped"
            ),
            "query_flip": (
                "same entities, facts, template, code vocabulary, and "
                "protocol; only the queried entity changes"
            ),
        },
        "intervention_arms": {
            "within_minimal_replace": (
                "replace frozen target positions with the opposite state "
                "from the same target world"
            ),
            "cross_world_whole": (
                "replace with the opposite-answer nuisance-world state"
            ),
            "cross_world_same_answer_whole": (
                "replace with the same-answer nuisance-world state"
            ),
            "causal_delta": (
                "h_target + (h_nuisance_opposite - h_nuisance_same)"
            ),
            "nuisance_delta": (
                "h_target + (h_nuisance2_same - h_nuisance_same)"
            ),
            "target_noop": "rewrite target state into target",
        },
        "source_selection": {
            "semantic_labels_visible": False,
            "depth": SOURCE_DEPTH,
            "event_universe": (
                "every tokenizer-verified user-content position"
            ),
            "screen_n": 16,
            "frozen_evaluation_n": 32,
            "ranking": [
                "descending leave-one-out target restoration",
                "descending leave-one-out median mediation",
                "descending single-position donor sequence rate",
                "ascending physical position",
            ],
            "build": (
                "greedy prefix until within-world donor sequence >= 0.80 "
                "and median normalized transfer >= 0.50, followed by one "
                "reverse-delete pass"
            ),
            "selection_arm": "within_minimal_replace only",
        },
        "behavior_thresholds": {
            "each_semantic_step_accuracy": 0.95,
            "teacher_forced_step1_accuracy": 0.95,
            "natural_exact_rate": 0.90,
            "natural_prefix_rate": 0.90,
            "immediate_end_rate": 0.90,
        },
        "source_thresholds": {
            "within_minimal_donor_sequence_rate": 0.80,
            "within_minimal_median_transfer": 0.50,
            "causal_delta_donor_sequence_rate": 0.80,
            "causal_delta_median_transfer": 0.50,
            "nuisance_delta_target_sequence_rate": 0.95,
            "target_noop_sequence_rate": 0.99,
        },
        "whole_source_gate": (
            "within-minimal basic gate AND cross-world whole basic gate "
            "AND same-answer whole target >= 0.95 AND no-op >= 0.99"
        ),
        "delta_source_gate": (
            "within-minimal basic gate AND causal-delta basic gate AND "
            "nuisance-delta target >= 0.95 AND no-op >= 0.99"
        ),
        "parent_gate": (
            "both contrasts pass delta_source_gate on discovery, "
            "confirmation, holdout, and at least two models"
        ),
        "downstream_authorization": (
            "temporal/KV/receiver decomposition only after parent_gate"
        ),
        "forbidden_claims": [
            "the selected token positions are permanent knowledge storage",
            "a controlled delta is an intrinsic semantic vector",
            "one model/template is cross-model closure",
            "PCA or CCA may select a mechanism before causal parent closure",
            "measurement equations are the native language law",
        ],
        "claim_boundary": (
            "A positive result can show that a matched residual difference "
            "transfers a controlled two-word relation in this task. It "
            "cannot establish a general language tuple, reasoning circuit, "
            "or neuron-level mechanism."
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
