#!/usr/bin/env python3
"""Freeze non-isomorphic structural stress tasks for Phase 1003."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1003_crossparadigm_protocol import (
    ANSWER_PREFIX,
    MODELS,
    OUT_ROOT,
    PHASE,
    canonical,
    digest,
    one_token_id,
    positions_of,
    write_json,
    write_jsonl,
)


STRESS_REVISION = 1
TASKS = (
    "three_entity",
    "multiattribute",
    "negation",
    "relation",
    "pronoun",
)
DISCOVERY_NAMES = (
    "Aaron", "Bella", "Chloe", "Ethan", "Fiona", "Gavin", "Hazel",
    "Isaac", "Julia", "Kevin", "Laura", "Mason", "Nina", "Owen",
    "Paige", "Ryan", "Sofia", "Tyler", "Vera", "Wendy", "Xavier",
    "Zoe", "Amber", "Caleb",
)
CONFIRMATION_NAMES = (
    "Daisy", "Elena", "Felix", "Holly", "Jason", "Kara", "Lucas",
    "Megan", "Noah", "Olivia", "Peter", "Rachel", "Sarah", "Simon",
    "Victor", "Wyatt", "Adam", "Brian", "Cindy", "Derek", "Emily",
    "George", "Hannah", "Jacob",
)
COLORS = ("purple", "orange", "black", "white")
SHAPES = ("circle", "square", "triangle", "diamond")
RELATIONS = ("left", "right", "above", "below")
WORLD_COUNT_PER_SPLIT = 16
STRESS_ROOT = OUT_ROOT / "structural_stress"


def answer_text(model_name: str, value: str) -> str:
    return f"{ANSWER_PREFIX[model_name]}Answer: {value}"


def instruction() -> str:
    return (
        "Answer exactly in this form: Answer: [value] Replace [value] with "
        "the lowercase answer. Do not add punctuation or other words."
    )


def stable_order(value: str, salt: str) -> str:
    return hashlib.sha256(
        f"phase1003-stress:{salt}:{value}".encode("utf-8")
    ).hexdigest()


def name_worlds(names: tuple[str, ...], split: str) -> list[list[str]]:
    rng = random.Random(1003_7723 + (0 if split == "discovery" else 1))
    result = []
    for world in range(WORLD_COUNT_PER_SPLIT):
        shuffled = list(names)
        rng.shuffle(shuffled)
        result.append(shuffled[:4])
    return result


def task_specs(
    task: str,
    template: int,
    names: list[str],
    world: int,
) -> list[dict[str, Any]]:
    e0, e1, e2, e3 = names
    if task == "three_entity":
        values = [
            COLORS[(world + offset) % len(COLORS)]
            for offset in range(3)
        ]
        if len(set(values)) != 3:
            raise RuntimeError("three-entity value collision")
        cases = []
        for query_role in range(3):
            entities = [e0, e1, e2]
            if template == 0:
                body = (
                    f"Ledger: {e0} has color {values[0]}; "
                    f"{e1} has color {values[1]}; "
                    f"{e2} has color {values[2]}.\n"
                    f"What color does {entities[query_role]} have?"
                )
            else:
                body = (
                    f"The recorded colors are {values[0]} for {e0}, "
                    f"{values[1]} for {e1}, and {values[2]} for {e2}.\n"
                    f"Report the color of {entities[query_role]}."
                )
            cases.append({
                "body": body,
                "gold": values[query_role],
                "candidates": list(COLORS),
                "variant": f"q{query_role}",
                "role_specs": [
                    ("entity0", e0, 0),
                    ("value0", values[0], 0),
                    ("entity1", e1, 0),
                    ("value1", values[1], 0),
                    ("entity2", e2, 0),
                    ("value2", values[2], 0),
                    (
                        "query_name",
                        entities[query_role],
                        -1,
                    ),
                ],
            })
        return cases

    if task == "multiattribute":
        colors = [COLORS[world % 4], COLORS[(world + 1) % 4]]
        shapes = [SHAPES[world % 4], SHAPES[(world + 1) % 4]]
        cases = []
        for query_entity, query_attribute in itertools.product(
            (0, 1), ("color", "shape")
        ):
            entities = [e0, e1]
            gold = (
                colors[query_entity]
                if query_attribute == "color"
                else shapes[query_entity]
            )
            if template == 0:
                body = (
                    f"Ledger: {e0} has color {colors[0]} and "
                    f"shape {shapes[0]}; "
                    f"{e1} has color {colors[1]} and shape {shapes[1]}.\n"
                    f"Which {query_attribute} does "
                    f"{entities[query_entity]} have?"
                )
            else:
                body = (
                    f"Record one says {e0} has color {colors[0]} and "
                    f"shape {shapes[0]}. Record two says {e1} has "
                    f"color {colors[1]} and shape {shapes[1]}.\n"
                    f"Report the {query_attribute} of "
                    f"{entities[query_entity]}."
                )
            cases.append({
                "body": body,
                "gold": gold,
                "candidates": list(COLORS + SHAPES),
                "variant": (
                    f"q{query_entity}.{query_attribute}"
                ),
                "role_specs": [
                    ("entity0", e0, 0),
                    ("color0", colors[0], 0),
                    ("shape0", shapes[0], 0),
                    ("entity1", e1, 0),
                    ("color1", colors[1], 0),
                    ("shape1", shapes[1], 0),
                    (
                        "query_name",
                        entities[query_entity],
                        -1,
                    ),
                    ("query_attribute", query_attribute, -1),
                ],
            })
        return cases

    if task == "negation":
        rejected0 = COLORS[world % 4]
        accepted0 = COLORS[(world + 1) % 4]
        rejected1 = COLORS[(world + 2) % 4]
        accepted1 = COLORS[(world + 3) % 4]
        cases = []
        for query_role in (0, 1):
            entities = [e0, e1]
            accepted = [accepted0, accepted1]
            if template == 0:
                body = (
                    f"Ledger: {e0} is not {rejected0}; instead {e0} is "
                    f"{accepted0}. {e1} is not {rejected1}; instead "
                    f"{e1} is {accepted1}.\nConsidering only accepted "
                    f"colors, what color is {entities[query_role]}?"
                )
            else:
                body = (
                    f"Rejected for {e0}: {rejected0}. Accepted for "
                    f"{e0}: {accepted0}. Rejected for {e1}: "
                    f"{rejected1}. Accepted for {e1}: {accepted1}.\n"
                    f"Give the accepted color of {entities[query_role]}."
                )
            cases.append({
                "body": body,
                "gold": accepted[query_role],
                "candidates": list(COLORS),
                "variant": f"q{query_role}",
                "role_specs": [
                    ("entity0_rejected", e0, 0),
                    ("rejected0", rejected0, 0),
                    ("entity0_accepted", e0, 1),
                    ("accepted0", accepted0, 0),
                    ("entity1_rejected", e1, 0),
                    ("rejected1", rejected1, 0),
                    ("entity1_accepted", e1, 1),
                    ("accepted1", accepted1, 0),
                    (
                        "query_name",
                        entities[query_role],
                        -1,
                    ),
                ],
            })
        return cases

    if task == "relation":
        relations = [
            RELATIONS[world % 4],
            RELATIONS[(world + 1) % 4],
        ]
        cases = []
        for query_role in (0, 1):
            pairs = [(e0, e1), (e2, e3)]
            query_subject, query_object = pairs[query_role]
            if template == 0:
                body = (
                    f"Records: {e0} is {relations[0]} of {e1}. "
                    f"{e2} is {relations[1]} of {e3}.\n"
                    f"What relation is recorded from {query_subject} "
                    f"to {query_object}?"
                )
            else:
                body = (
                    f"Relation record one links {e0} to {e1} as "
                    f"{relations[0]}. Relation record two links {e2} "
                    f"to {e3} as {relations[1]}.\nReport the relation "
                    f"from {query_subject} to {query_object}."
                )
            cases.append({
                "body": body,
                "gold": relations[query_role],
                "candidates": list(RELATIONS),
                "variant": f"q{query_role}",
                "role_specs": [
                    ("subject0", e0, 0),
                    ("relation0", relations[0], 0),
                    ("object0", e1, 0),
                    ("subject1", e2, 0),
                    ("relation1", relations[1], 0),
                    ("object1", e3, 0),
                    ("query_subject", query_subject, -1),
                    ("query_object", query_object, -1),
                ],
            })
        return cases

    if task == "pronoun":
        values = [COLORS[world % 4], COLORS[(world + 1) % 4]]
        cases = []
        for referent_role in (0, 1):
            entities = [e0, e1]
            if template == 0:
                body = (
                    f"Ledger: {e0} carries color {values[0]}; "
                    f"{e1} carries "
                    f"color {values[1]}. In the question, the pronoun "
                    f"they refers to {entities[referent_role]}.\n"
                    f"What color do they carry?"
                )
            else:
                body = (
                    f"The color of {e0} is {values[0]}, and the color "
                    f"of {e1} is {values[1]}. Resolve they as "
                    f"{entities[referent_role]}.\nWhich color belongs "
                    f"to them?"
                )
            cases.append({
                "body": body,
                "gold": values[referent_role],
                "candidates": list(COLORS),
                "variant": f"r{referent_role}",
                "role_specs": [
                    ("entity0", e0, 0),
                    ("value0", values[0], 0),
                    ("entity1", e1, 0),
                    ("value1", values[1], 0),
                    (
                        "referent_name",
                        entities[referent_role],
                        -1,
                    ),
                ],
            })
        return cases
    raise ValueError(task)


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    all_words = set(DISCOVERY_NAMES + CONFIRMATION_NAMES)
    all_words.update(COLORS + SHAPES + RELATIONS)
    all_words.update(("color", "shape"))
    token_ids = {
        word: one_token_id(tokenizer, " " + word)
        for word in sorted(all_words)
    }
    cases = []
    task_role_sets = {}
    semantic_steps = {}
    for split, names in (
        ("discovery", DISCOVERY_NAMES),
        ("confirmation", CONFIRMATION_NAMES),
    ):
        worlds = name_worlds(names, split)
        for task in TASKS:
            for world, world_names in enumerate(worlds):
                for template in (0, 1):
                    for spec in task_specs(
                        task, template, world_names, world
                    ):
                        raw_prompt = (
                            spec["body"] + "\n" + instruction()
                        )
                        rendered = render_chat(
                            tokenizer, model_name, raw_prompt
                        )
                        input_ids = [
                            int(value)
                            for value in tokenizer.encode(
                                rendered, add_special_tokens=False
                            )
                        ]
                        answer_ids = {
                            label: [
                                int(value)
                                for value in tokenizer.encode(
                                    answer_text(model_name, label),
                                    add_special_tokens=False,
                                )
                            ]
                            for label in spec["candidates"]
                        }
                        widths = {
                            len(values)
                            for values in answer_ids.values()
                        }
                        if len(widths) != 1:
                            raise RuntimeError(
                                f"{model_name}/{task}: answer width drift"
                            )
                        width = next(iter(widths))
                        varying = [
                            step
                            for step in range(width)
                            if len({
                                values[step]
                                for values in answer_ids.values()
                            }) > 1
                        ]
                        if len(varying) != 1:
                            raise RuntimeError(
                                f"{model_name}/{task}: semantic drift"
                            )
                        semantic_step = varying[0]
                        semantic_steps[task] = semantic_step
                        candidate_ids = {
                            label: values[semantic_step]
                            for label, values in answer_ids.items()
                        }
                        role_positions = {}
                        for role, word, occurrence in spec["role_specs"]:
                            found = positions_of(
                                input_ids, token_ids[word]
                            )
                            if not found:
                                raise RuntimeError(
                                    f"{model_name}/{task}/{role}: "
                                    f"missing {word}"
                                )
                            try:
                                role_positions[role] = found[occurrence]
                            except IndexError as exc:
                                raise RuntimeError(
                                    f"{model_name}/{task}/{role}: "
                                    f"occurrence {occurrence} in {found}"
                                ) from exc
                        role_names = tuple(role_positions)
                        previous = task_role_sets.setdefault(
                            task, role_names
                        )
                        if previous != role_names:
                            raise RuntimeError(
                                f"{task}: role set drift"
                            )
                        record_id = (
                            f"{model_name}.{task}.{split[:1]}w{world:02d}."
                            f"t{template}.{spec['variant']}"
                        )
                        expected = answer_ids[spec["gold"]]
                        extended = [
                            int(value)
                            for value in tokenizer.encode(
                                rendered
                                + answer_text(
                                    model_name, spec["gold"]
                                ),
                                add_special_tokens=False,
                            )
                        ]
                        if extended != input_ids + expected:
                            raise RuntimeError(
                                f"{record_id}: answer boundary drift"
                            )
                        role_positions["answer_boundary"] = (
                            len(input_ids) - 1
                        )
                        cases.append({
                            "schema_version": (
                                "phase1003_structural_stress_case.v1"
                            ),
                            "phase": PHASE,
                            "stress_revision": STRESS_REVISION,
                            "model": model_name,
                            "record_id": record_id,
                            "task": task,
                            "domain": task,
                            "split": split,
                            "world_id": f"{task}.{split[:1]}w{world:02d}",
                            "world_index": world,
                            "template": template,
                            "variant": spec["variant"],
                            "raw_prompt": raw_prompt,
                            "rendered_prompt": rendered,
                            "input_ids": input_ids,
                            "input_token_count": len(input_ids),
                            "gold": spec["gold"],
                            "candidate_labels": spec["candidates"],
                            "candidate_token_ids": candidate_ids,
                            "answer_text": answer_text(
                                model_name, spec["gold"]
                            ),
                            "answer_token_ids": expected,
                            "semantic_step": semantic_step,
                            "anchor_roles": list(role_names),
                            "role_positions": role_positions,
                        })
    root = STRESS_ROOT / "protocol" / model_name
    write_jsonl(root / "cases.jsonl", cases)
    counts = {
        f"{task}:{split}": sum(
            row["task"] == task and row["split"] == split
            for row in cases
        )
        for task in TASKS
        for split in ("discovery", "confirmation")
    }
    audit = {
        "schema_version": (
            "phase1003_structural_stress_protocol_audit.v1"
        ),
        "phase": PHASE,
        "stress_revision": STRESS_REVISION,
        "model": model_name,
        "case_count": len(cases),
        "counts": counts,
        "task_role_sets": {
            task: list(roles)
            for task, roles in task_role_sets.items()
        },
        "semantic_steps": semantic_steps,
        "discovery_confirmation_name_overlap": sorted(
            set(DISCOVERY_NAMES) & set(CONFIRMATION_NAMES)
        ),
        "case_digest": digest(cases),
    }
    write_json(root / "protocol_audit.json", audit)
    return audit


def main() -> None:
    audits = {
        model_name: build_model(model_name) for model_name in MODELS
    }
    main_prereg = json.loads(
        (OUT_ROOT / "preregistered_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    prereg = {
        "schema_version": (
            "phase1003_structural_stress_preregistration.v1"
        ),
        "phase": PHASE,
        "stress_revision": STRESS_REVISION,
        "trigger": (
            "Core attribute-role and cache tests completed; the original "
            "Phase1003 task list also requires non-isomorphic relation, "
            "negation, pronoun, three-entity, and multiattribute stress."
        ),
        "internal_results_used_to_define_stress_prompts": False,
        "tasks": list(TASKS),
        "protocol_audits": audits,
        "source_depths": main_prereg["source_depths"],
        "thresholds": {
            "behavior_candidate_accuracy": 0.90,
            "behavior_exact_answer_rate": 0.90,
            "full_anchor_donor_rate": 0.75,
            "noop_agreement": 0.99,
            "cache_value_donor_rate": 0.70,
            "cross_model_minimum": 2,
        },
        "causal_conditions": (
            "empty target no-op, full task-specific role set, and every "
            "single-role leave-out; no exhaustive formula is imposed on "
            "non-isomorphic role universes"
        ),
        "claim_boundary": (
            "These are controlled structural stress tasks. Pronoun uses an "
            "explicit referent declaration and does not represent unrestricted "
            "coreference; relation is a recorded-pair lookup, not general "
            "spatial reasoning."
        ),
        "preregistration_digest": None,
    }
    prereg["preregistration_digest"] = digest({
        key: value
        for key, value in prereg.items()
        if key != "preregistration_digest"
    })
    write_json(STRESS_ROOT / "preregistered_protocol.json", prereg)
    print(json.dumps(prereg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
