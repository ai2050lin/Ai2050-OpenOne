#!/usr/bin/env python3
"""Frozen Phase326 denominator for distributed carrier-set mapping.

The case bank keeps independent objects separate from prompt templates.  It is
deliberately plain JSON/JSONL so later probes and the visualization client read
the same registered denominator.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase326_distributed_carrier_atlas"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
SCHEMA_VERSION = "5.0.0"
PHASE = "Phase326"
TEMPLATES = ("template_a", "template_b", "template_c")
CONFIRMATION_TEMPLATES = ("template_d", "template_e")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_for(index: int) -> str:
    if index < 4:
        return "discovery"
    if index < 8:
        return "calibration"
    return "heldout"


KNOWLEDGE_FACTS: dict[str, list[tuple[str, str, list[str]]]] = {
    "color_retrieval": [
        ("banana", "yellow", ["blue", "black", "red"]),
        ("grass", "green", ["purple", "white", "orange"]),
        ("coal", "black", ["yellow", "pink", "green"]),
        ("snow", "white", ["black", "red", "brown"]),
        ("clear daytime sky", "blue", ["orange", "black", "green"]),
        ("ripe tomato", "red", ["blue", "white", "purple"]),
        ("carrot", "orange", ["black", "blue", "pink"]),
        ("ripe eggplant", "purple", ["yellow", "white", "green"]),
        ("milk", "white", ["red", "black", "orange"]),
        ("lemon", "yellow", ["purple", "blue", "brown"]),
        ("clear ocean water", "blue", ["red", "pink", "black"]),
        ("emerald", "green", ["orange", "white", "purple"]),
    ],
    "material_retrieval": [
        ("ordinary window pane", "glass", ["wool", "wood", "paper"]),
        ("dining table", "wood", ["glass", "rubber", "steel"]),
        ("car tire", "rubber", ["paper", "glass", "wool"]),
        ("notebook", "paper", ["steel", "ceramic", "rubber"]),
        ("winter sweater", "wool", ["glass", "brick", "copper"]),
        ("drink bottle", "plastic", ["wood", "paper", "brick"]),
        ("kitchen knife", "steel", ["wool", "glass", "rubber"]),
        ("house wall", "brick", ["paper", "plastic", "wool"]),
        ("electrical wire", "copper", ["glass", "wood", "ceramic"]),
        ("coffee mug", "ceramic", ["rubber", "paper", "steel"]),
        ("summer shirt", "cotton", ["glass", "brick", "copper"]),
        ("dining spoon", "silver", ["wood", "plastic", "paper"]),
    ],
    "habitat_retrieval": [
        ("camel", "desert", ["ocean", "forest", "tundra"]),
        ("dolphin", "ocean", ["desert", "cave", "meadow"]),
        ("frog", "pond", ["desert", "nest", "tundra"]),
        ("mole", "underground", ["ocean", "sky", "reef"]),
        ("bee colony", "hive", ["cave", "pond", "desert"]),
        ("lion", "savanna", ["reef", "tundra", "pond"]),
        ("polar bear", "arctic", ["jungle", "reef", "desert"]),
        ("monkey", "forest", ["tundra", "ocean", "desert"]),
        ("salmon", "river", ["desert", "nest", "savanna"]),
        ("bat", "cave", ["reef", "meadow", "pond"]),
        ("coral", "reef", ["forest", "desert", "cave"]),
        ("rabbit", "burrow", ["ocean", "hive", "reef"]),
    ],
    "category_retrieval": [
        ("robin", "bird", ["tool", "metal", "flower"]),
        ("hammer", "tool", ["fish", "plant", "bird"]),
        ("oak", "plant", ["metal", "instrument", "fish"]),
        ("copper", "metal", ["flower", "bird", "tool"]),
        ("violin", "instrument", ["plant", "fish", "metal"]),
        ("salmon", "fish", ["bird", "tool", "flower"]),
        ("rose", "flower", ["metal", "fish", "instrument"]),
        ("carrot", "vegetable", ["bird", "tool", "metal"]),
        ("eagle", "bird", ["plant", "instrument", "vegetable"]),
        ("piano", "instrument", ["fish", "flower", "tool"]),
        ("wrench", "tool", ["bird", "plant", "metal"]),
        ("bamboo", "plant", ["instrument", "fish", "tool"]),
    ],
}


CONFIRMATION_KNOWLEDGE_FACTS: dict[str, list[tuple[str, str, list[str]]]] = {
    "color_retrieval": [
        ("strawberry", "red", ["blue", "white", "green"]),
        ("lime", "green", ["purple", "black", "orange"]),
        ("raven", "black", ["yellow", "white", "pink"]),
        ("daytime cloud", "white", ["black", "brown", "red"]),
        ("sapphire", "blue", ["orange", "green", "pink"]),
        ("pumpkin", "orange", ["purple", "white", "blue"]),
        ("violet flower", "purple", ["yellow", "black", "green"]),
        ("chocolate", "brown", ["blue", "white", "pink"]),
        ("flamingo", "pink", ["green", "black", "orange"]),
        ("sunflower", "yellow", ["purple", "blue", "brown"]),
        ("charcoal", "black", ["white", "green", "red"]),
        ("pearl", "white", ["black", "orange", "purple"]),
        ("turquoise stone", "blue", ["red", "brown", "yellow"]),
        ("lettuce", "green", ["pink", "black", "purple"]),
        ("cherry", "red", ["green", "white", "blue"]),
        ("tangerine", "orange", ["purple", "black", "pink"]),
    ],
    "material_retrieval": [
        ("frying pan", "metal", ["paper", "wool", "glass"]),
        ("raincoat", "plastic", ["wood", "brick", "cotton"]),
        ("printed book", "paper", ["steel", "rubber", "ceramic"]),
        ("walking boot", "leather", ["glass", "paper", "copper"]),
        ("winter blanket", "wool", ["brick", "plastic", "steel"]),
        ("decorative vase", "ceramic", ["rubber", "paper", "wood"]),
        ("preserving jar", "glass", ["wool", "steel", "brick"]),
        ("extension ladder", "aluminum", ["paper", "rubber", "cotton"]),
        ("wedding necklace", "gold", ["wood", "glass", "paper"]),
        ("dining fork", "silver", ["plastic", "brick", "wool"]),
        ("climbing rope", "hemp", ["glass", "steel", "ceramic"]),
        ("waist belt", "leather", ["paper", "copper", "brick"]),
        ("shipping box", "cardboard", ["glass", "rubber", "steel"]),
        ("toy block", "wood", ["paper", "wool", "ceramic"]),
        ("plumbing pipe", "copper", ["glass", "cotton", "brick"]),
        ("party balloon", "rubber", ["wood", "paper", "steel"]),
    ],
    "habitat_retrieval": [
        ("shark", "ocean", ["desert", "forest", "burrow"]),
        ("owl", "forest", ["reef", "tundra", "pond"]),
        ("scorpion", "desert", ["ocean", "meadow", "river"]),
        ("beaver", "river", ["sky", "desert", "reef"]),
        ("crab", "shore", ["forest", "cave", "tundra"]),
        ("seal", "arctic", ["jungle", "desert", "meadow"]),
        ("ant colony", "nest", ["ocean", "pond", "cave"]),
        ("spider", "web", ["river", "tundra", "hive"]),
        ("earthworm", "soil", ["sky", "reef", "ocean"]),
        ("butterfly", "meadow", ["cave", "desert", "tundra"]),
        ("lobster", "reef", ["forest", "burrow", "hive"]),
        ("duck", "pond", ["desert", "cave", "savanna"]),
        ("fox", "den", ["ocean", "reef", "hive"]),
        ("oyster", "seabed", ["sky", "forest", "desert"]),
        ("gorilla", "jungle", ["tundra", "reef", "pond"]),
        ("lizard", "desert", ["ocean", "arctic", "river"]),
    ],
    "category_retrieval": [
        ("sparrow", "bird", ["tool", "metal", "flower"]),
        ("screwdriver", "tool", ["fish", "plant", "bird"]),
        ("maple", "plant", ["instrument", "metal", "fish"]),
        ("iron", "metal", ["flower", "bird", "vegetable"]),
        ("trumpet", "instrument", ["plant", "fish", "tool"]),
        ("trout", "fish", ["bird", "metal", "flower"]),
        ("tulip", "flower", ["tool", "fish", "instrument"]),
        ("potato", "vegetable", ["bird", "metal", "plant"]),
        ("whale", "mammal", ["fish", "tool", "flower"]),
        ("python", "reptile", ["bird", "plant", "instrument"]),
        ("bicycle", "vehicle", ["flower", "fish", "metal"]),
        ("sofa", "furniture", ["bird", "tool", "vegetable"]),
        ("ruby", "mineral", ["plant", "fish", "instrument"]),
        ("jacket", "clothing", ["metal", "bird", "flower"]),
        ("apple", "fruit", ["tool", "fish", "mineral"]),
        ("ant", "insect", ["plant", "metal", "bird"]),
    ],
}


ORDER_ITEMS = [
    ("Ava", "Ben", "Cleo"), ("Dara", "Eli", "Faye"), ("Gus", "Hana", "Ivo"),
    ("Jia", "Kian", "Luz"), ("Mira", "Niko", "Omar"), ("Pia", "Quin", "Ravi"),
    ("Sora", "Tao", "Uma"), ("Vera", "Wes", "Xena"), ("Yara", "Zane", "Arlo"),
    ("Bela", "Ciro", "Dina"), ("Enzo", "Freya", "Gino"), ("Hiro", "Iris", "Juno"),
]
TOKENS = ["dax", "wug", "zorb", "kiv", "mep", "tul", "nib", "fex", "ral", "siv", "bim", "lod"]
OBJECTS = ["gem", "tile", "badge", "cube", "card", "disk", "block", "token", "vase", "box", "flag", "ring"]
PLACES = [
    ("alpha", "beta", "gamma"), ("delta", "echo", "foxtrot"), ("garnet", "hazel", "indigo"),
    ("jade", "khaki", "lilac"), ("maple", "navy", "ochre"), ("pearl", "quartz", "ruby"),
    ("sable", "teal", "umber"), ("violet", "wheat", "xanthic"), ("yarrow", "zinc", "amber"),
    ("birch", "coral", "denim"), ("elm", "flint", "gold"), ("heather", "ivory", "jet"),
]


def knowledge_prompt(subject: str, relation: str, template: str) -> tuple[str, str, str]:
    label = relation.replace("_retrieval", "")
    if template == "template_a":
        prompt = f"What is the usual {label} of the {subject}? Answer with one word:"
        query = f"usual {label}"
    elif template == "template_b":
        prompt = f"Use ordinary world knowledge. Give the {label} associated with the {subject}. One word:"
        query = f"{label} associated"
    elif template == "template_c":
        prompt = f"Complete this knowledge query in one word. The {subject} has what usual {label}? Answer:"
        query = f"usual {label}"
    elif template == "template_d":
        prompt = f"Without answer choices, state the ordinary {label} for the {subject}. Give one word:"
        query = f"ordinary {label}"
    elif template == "template_e":
        prompt = f"Recall general knowledge about the {subject}. What {label} is normally associated with it? One word:"
        query = f"What {label}"
    else:
        raise KeyError(template)
    return prompt, subject, query


def reasoning_item(mechanism: str, index: int) -> tuple[list[str], str, str, list[str]]:
    if mechanism == "transitive_order":
        a, b, c = ORDER_ITEMS[index]
        if index % 2 == 0:
            facts = [f"{a} finished before {b}.", f"{b} finished before {c}."]
            query, target = f"Did {a} finish before {c}?", "yes"
        else:
            facts = [f"{a} finished before {b}.", f"{b} finished before {c}."]
            query, target = f"Did {c} finish before {a}?", "no"
        return facts, query, target, ["no" if target == "yes" else "yes"]
    if mechanism == "implication_chain":
        token = TOKENS[index]
        warm = ["warm", "quiet", "bright", "round"][index % 4]
        active = ["active", "stable", "visible", "ready"][index % 4]
        facts = [f"If a {token} is amber, then it is {warm}.", f"If it is {warm}, then it is {active}."]
        if index % 2 == 0:
            facts.append(f"This {token} is amber.")
            query, target = f"Is this {token} {active}?", "yes"
        else:
            facts.append(f"This {token} is blue.")
            query, target = f"Is this {token} {active}?", "no"
        return facts, query, target, ["no" if target == "yes" else "yes"]
    if mechanism == "conjunction_rule":
        obj = OBJECTS[index]
        shape = ["square", "round", "tall", "small"][index % 4]
        color = ["red", "blue", "green", "white"][index % 4]
        facts = [f"A {obj} is marked only when it is both {color} and {shape}.", f"This {obj} is {color}."]
        if index % 2 == 0:
            facts.append(f"This {obj} is {shape}.")
            target = "yes"
        else:
            facts.append(f"This {obj} is not {shape}.")
            target = "no"
        return facts, f"Is this {obj} marked?", target, ["no" if target == "yes" else "yes"]
    if mechanism == "spatial_composition":
        a, b, c = PLACES[index]
        if index % 2 == 0:
            facts = [f"{a} is east of {b}.", f"{b} is north of {c}."]
            return facts, f"Where is {a} relative to {c}?", "northeast", ["southwest", "northwest", "southeast"]
        facts = [f"{a} is west of {b}.", f"{b} is south of {c}."]
        return facts, f"Where is {a} relative to {c}?", "southwest", ["northeast", "northwest", "southeast"]
    raise KeyError(mechanism)


def reasoning_prompt(facts: list[str], query: str, template: str, candidates: list[str]) -> tuple[str, str, str]:
    source = " ".join(facts)
    choices = ", ".join(candidates)
    if template == "template_a":
        prompt = f"Facts: {source} Question: {query} Answer with one of [{choices}]:"
    elif template == "template_b":
        prompt = f"Reason only from these statements. {source} Now decide: {query} Reply with one of [{choices}]:"
    else:
        prompt = f"Given {source} Query: {query} Select one answer from [{choices}]:"
    return prompt, source, query


def build_cases() -> list[dict[str, Any]]:
    created = now()
    rows: list[dict[str, Any]] = []
    for mechanism, items in KNOWLEDGE_FACTS.items():
        for index, (subject, target, distractors) in enumerate(items):
            for template in TEMPLATES:
                prompt, source, query = knowledge_prompt(subject, mechanism, template)
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": created,
                    "case_id": f"knowledge__{mechanism}__{index:02d}__{template}",
                    "base_case_id": f"knowledge__{mechanism}__{index:02d}",
                    "family_id": "content_knowledge",
                    "mechanism_id": mechanism,
                    "split": split_for(index),
                    "independent_item_index": index,
                    "template_id": template,
                    "prompt": prompt,
                    "source_fragments": [source],
                    "query_fragment": query,
                    "target": target,
                    "distractors": distractors,
                    "candidate_answers": [target, *distractors],
                    "target_absence_required": True,
                    "source_semantics": "object_trigger_without_explicit_answer",
                })
    reasoning_mechanisms = ("transitive_order", "implication_chain", "conjunction_rule", "spatial_composition")
    for mechanism in reasoning_mechanisms:
        for index in range(12):
            facts, query, target, distractors = reasoning_item(mechanism, index)
            for template in TEMPLATES:
                prompt, source, query_fragment = reasoning_prompt(facts, query, template, [target, *distractors])
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": created,
                    "case_id": f"reasoning__{mechanism}__{index:02d}__{template}",
                    "base_case_id": f"reasoning__{mechanism}__{index:02d}",
                    "family_id": "reasoning_constraint",
                    "mechanism_id": mechanism,
                    "split": split_for(index),
                    "independent_item_index": index,
                    "template_id": template,
                    "prompt": prompt,
                    "source_fragments": facts,
                    "source_group_fragment": source,
                    "query_fragment": query_fragment,
                    "target": target,
                    "distractors": distractors,
                    "candidate_answers": [target, *distractors],
                    "target_absence_required": False,
                    "source_semantics": "multi_token_rule_fact_group",
                })
    return rows


def build_confirmation_cases() -> list[dict[str, Any]]:
    created = now()
    rows: list[dict[str, Any]] = []
    for mechanism, items in CONFIRMATION_KNOWLEDGE_FACTS.items():
        for index, (subject, target, distractors) in enumerate(items):
            for template in CONFIRMATION_TEMPLATES:
                prompt, source, query = knowledge_prompt(subject, mechanism, template)
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": created,
                    "case_id": f"confirmation__knowledge__{mechanism}__{index:02d}__{template}",
                    "base_case_id": f"confirmation__knowledge__{mechanism}__{index:02d}",
                    "family_id": "content_knowledge",
                    "mechanism_id": mechanism,
                    "split": "expanded_confirmation",
                    "independent_item_index": index,
                    "template_id": template,
                    "prompt": prompt,
                    "source_fragments": [source],
                    "query_fragment": query,
                    "target": target,
                    "distractors": distractors,
                    "candidate_answers": [target, *distractors],
                    "target_absence_required": True,
                    "source_semantics": "new_object_implicit_retrieval_confirmation",
                })
    return rows


def validate_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case_id")
    for row in rows:
        prompt_lower = row["prompt"].lower()
        if row["target_absence_required"] and row["target"].lower() in prompt_lower:
            errors.append(f"explicit target leak: {row['case_id']}")
        for fragment in row["source_fragments"]:
            if fragment not in row["prompt"]:
                errors.append(f"missing source fragment: {row['case_id']}::{fragment}")
        if row["query_fragment"] not in row["prompt"]:
            errors.append(f"missing query fragment: {row['case_id']}")
    family_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family_id"]] = family_counts.get(row["family_id"], 0) + 1
        key = f"{row['family_id']}/{row['mechanism_id']}"
        mechanism_counts[key] = mechanism_counts.get(key, 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    expected = {"discovery": 96, "calibration": 96, "heldout": 96}
    if len(rows) != 288:
        errors.append(f"expected 288 prompts, got {len(rows)}")
    if split_counts != expected:
        errors.append(f"unexpected split counts: {split_counts}")
    if any(value != 36 for value in mechanism_counts.values()):
        errors.append(f"mechanism denominator is not 36: {mechanism_counts}")
    return {
        "valid": not errors,
        "errors": errors,
        "prompt_case_count": len(rows),
        "independent_base_case_count": len({row["base_case_id"] for row in rows}),
        "family_counts": family_counts,
        "mechanism_counts": mechanism_counts,
        "split_counts": split_counts,
        "knowledge_target_leak_count": sum(
            1 for row in rows if row["target_absence_required"] and row["target"].lower() in row["prompt"].lower()
        ),
    }


def validate_confirmation_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(rows) != 128:
        errors.append(f"expected 128 confirmation prompts, got {len(rows)}")
    if len({row["base_case_id"] for row in rows}) != 64:
        errors.append("expected 64 independent confirmation objects")
    for row in rows:
        if row["target"].lower() in row["prompt"].lower():
            errors.append(f"explicit target leak: {row['case_id']}")
        if row["source_fragments"][0] not in row["prompt"] or row["query_fragment"] not in row["prompt"]:
            errors.append(f"missing registered fragment: {row['case_id']}")
    return {
        "valid": not errors,
        "errors": errors,
        "prompt_case_count": len(rows),
        "independent_base_case_count": len({row["base_case_id"] for row in rows}),
        "mechanism_counts": {
            mechanism: sum(row["mechanism_id"] == mechanism for row in rows)
            for mechanism in CONFIRMATION_KNOWLEDGE_FACTS
        },
        "target_leak_count": sum(row["target"].lower() in row["prompt"].lower() for row in rows),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="distributed_carrier_atlas")
    args = parser.parse_args()
    rows = build_cases()
    validation = validate_cases(rows)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    out = OUT / args.round
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "objective": "Map implicit knowledge retrieval and multi-token reasoning source groups to distributed physical carrier candidates.",
        "frozen_denominator": {
            "families": 2,
            "mechanisms_per_family": 4,
            "independent_items_per_mechanism": 12,
            "templates_per_item": 3,
            "prompt_cases_per_model": 288,
            "registered_models": ["qwen3", "glm4", "deepseek7b"],
            "registered_model_prompt_cases": 864,
        },
        "independent_splits": {"discovery": [0, 3], "calibration": [4, 7], "heldout": [8, 11]},
        "registered_conditions": [
            "baseline", "single_attention_zero", "attention_set_zero", "single_mlp_zero",
            "mlp_set_zero", "joint_set_zero", "matched_random_joint_zero", "wrong_layer_joint_zero",
        ],
        "evidence_limits": [
            "Candidate selection is observational until intervention.",
            "MLP product groups are not single neurons.",
            "Set ablation is a necessity test, not a natural sufficiency or natural-gate test.",
            "L5 requires frozen heldout replication in at least two models with matched controls.",
        ],
    }
    write_jsonl(out / "phase326_registered_cases.jsonl", rows)
    confirmation_rows = build_confirmation_cases()
    confirmation_validation = validate_confirmation_cases(confirmation_rows)
    if not confirmation_validation["valid"]:
        raise SystemExit(json.dumps(confirmation_validation, ensure_ascii=False, indent=2))
    write_jsonl(out / "phase326_expanded_confirmation_cases.jsonl", confirmation_rows)
    write_json(out / "phase326_expanded_confirmation_validation.json", confirmation_validation)
    write_json(out / "phase326_protocol.json", protocol)
    write_json(out / "phase326_case_bank_validation.json", validation)
    write_jsonl(V2 / "phase326_registered_cases.jsonl", rows)
    write_jsonl(V2 / "phase326_expanded_confirmation_cases.jsonl", confirmation_rows)
    write_json(V2 / "phase326_expanded_confirmation_validation.json", confirmation_validation)
    write_json(V2 / "phase326_protocol.json", protocol)
    write_json(V2 / "phase326_case_bank_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
