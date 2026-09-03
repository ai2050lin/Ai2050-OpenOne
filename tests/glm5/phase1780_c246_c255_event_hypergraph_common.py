#!/usr/bin/env python3
"""Shared contract, material, and paths for the C246-C255 campaign."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1768_c234_event_campaign_common as old

core = old.core
graph_base = old.graph_base
previous = old.previous

DIM = 2560
WIDTH = 128
FAMILIES = old.FAMILIES
EFFECTS = old.EFFECTS
ROLES = old.ROLES
SURFACES = ("case_review", "radio_summary")
MODELS = ("qwen3", "glm4", "deepseek7b")

OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for phase, campaign, slug in (
        (1780, "C246", "evidence_correction_master_contract"),
        (1781, "C247", "third_material_semantic_contract"),
        (1782, "C248", "qwen_third_material_full_field"),
        (1783, "C249", "third_material_event_core_prediction"),
        (1784, "C250", "full_token_event_transport_observation"),
        (1785, "C251", "typed_composition_observation"),
        (1786, "C252", "path_consistent_causal_branch"),
        (1787, "C253", "cross_model_abstract_event_replication"),
        (1788, "C254", "event_hypergraph_heatmap"),
        (1789, "C255", "campaign_theory_adjudication"),
    )
}

OLD = {
    "C235": old.OUTS["C235"],
    "C236": old.OUTS["C236"],
    "C237": old.OUTS["C237"],
    "C242": old.OUTS["C242"],
    "C244": RESULT / "phase1778_c244_independent_event_replication",
    "C245": RESULT / "phase1779_c245_confirmed_event_core",
}

UNITS = (
    {"primary": "Lucan", "secondary": "Mireya", "observer": "Neris", "object": "kiwi", "other": "easel", "node": "kivora", "middle": "frulon", "parent": "produce", "wrong": "furniture"},
    {"primary": "Orin", "secondary": "Priya", "observer": "Quilla", "object": "endive", "other": "mallet", "node": "endorin", "middle": "leafor", "parent": "vegetable", "wrong": "tool"},
    {"primary": "Ronan", "secondary": "Sabine", "observer": "Tova", "object": "mulberry", "other": "bucket", "node": "mulvik", "middle": "beralon", "parent": "food", "wrong": "container"},
    {"primary": "Ulric", "secondary": "Viola", "observer": "Wren", "object": "rutabaga", "other": "lyre", "node": "rutarin", "middle": "rootel", "parent": "plant", "wrong": "instrument"},
    {"primary": "Xenia", "secondary": "Yorick", "observer": "Zelda", "object": "kumquat", "other": "anvil", "node": "kumorin", "middle": "citravel", "parent": "organism", "wrong": "metal"},
    {"primary": "Arlen", "secondary": "Bianca", "observer": "Corin", "object": "chard", "other": "sextant", "node": "charvik", "middle": "stemor", "parent": "entity", "wrong": "device"},
    {"primary": "Devon", "secondary": "Elara", "observer": "Florian", "object": "persimmon", "other": "stool", "node": "persalin", "middle": "orchavel", "parent": "edible", "wrong": "building"},
    {"primary": "Gavin", "secondary": "Helena", "observer": "Isolde", "object": "fennel", "other": "gong", "node": "fenorin", "middle": "herbex", "parent": "living thing", "wrong": "sound"},
)


def options(correct: str, wrong: str, order: int) -> tuple[str, int]:
    return old.options(correct, wrong, order)


def wrap(surface: str, fact1: str, fact2: str, question: str) -> str:
    if surface == "case_review":
        return f"A case review records: {fact1} A separate observation records: {fact2} Question: {question}"
    if surface == "radio_summary":
        return f"A radio summary reports that {fact1} It also reports that {fact2} Decide from these reports: {question}"
    raise KeyError(surface)


def semantic_case(family: str, surface: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    obj, other = u["object"], u["other"]
    node, middle, parent, wrong = u["node"], u["middle"], u["parent"], u["wrong"]
    if family == "attitude_event":
        relation = "approves of" if a == 0 else "doubts"
        target = f"{o} {relation} the account that {p} examined the {obj}." if b == 0 else f"{o} {relation} the account that the {obj} was examined by {p}."
        noise = f"{s} moved the {other}."
        question, correct, distractor = f"Who examined the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    elif family == "type_graph":
        relation = "is registered under"
        if a == 0:
            target = f"The {node} is registered under {parent}."
            noise = f"The {middle} is registered under {wrong}."
        else:
            target = f"The {node} is registered under {middle}."
            noise = f"The {middle} is registered under {parent}."
        if b:
            target += f" A cross-reference also registers the {node} directly under {parent}."
        question, correct, distractor = f"Which final class includes the {node}?", parent, wrong
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
    elif family == "contrast":
        if a == 0:
            relation = "nevertheless"
            target = f"{s} looked uncertain; nevertheless, {p} remained composed." if b == 0 else f"{p} remained composed; nevertheless, {s} looked uncertain."
        else:
            relation = "Whereas" if b == 0 else "whereas"
            target = f"Whereas {s} looked uncertain, {p} remained composed." if b == 0 else f"{p} remained composed, whereas {s} looked uncertain."
        noise = f"The {obj} stayed beside the {other}."
        question, correct, distractor = "Who remained composed?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": "composed"}
    elif family == "translation":
        relation = "refers to"
        if a == 0:
            target = f'In this glossary, "{node}" refers to "{parent}".'
            noise = f'"{middle}" refers to "{wrong}".'
        else:
            target = f'In this glossary, "{node}" refers to "{middle}".'
            noise = f'"{middle}" refers to "{parent}".'
        if b:
            target += f' An index also links "{node}" directly to "{parent}".'
        question, correct, distractor = f'What does "{node}" finally refer to?', parent, wrong
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
    elif family == "comparison":
        dimension = "brighter" if a == 0 else "colder"
        inverse = "dimmer" if a == 0 else "warmer"
        relation = dimension if b == 0 else inverse
        target = f"{p} is {dimension} than {s}." if b == 0 else f"{s} is {inverse} than {p}."
        noise = f"The {obj} rests near the {other}."
        question, correct, distractor = f"Who is {dimension}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": s, "query": dimension}
    else:
        raise KeyError(family)
    return {"prompt_core": wrap(surface, target, noise, question), "correct": correct, "wrong": distractor, "roles": roles}


def nested_case(surface: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    value = u["object"] if b else u["parent"]
    relation = "was pleased to learn that" if a else "reported that"
    target = f"{o} {relation} {p} ate the {value}."
    noise = f"{s} moved the {u['other']}."
    question = f"What did {p} eat?"
    roles = {"primary": p, "secondary": s, "relation": relation, "context": value, "query": p}
    return {"prompt_core": wrap(surface, target, noise, question), "correct": value, "wrong": u["parent"] if b else u["object"], "roles": roles}


def material() -> list[dict]:
    rows: list[dict] = []
    for panel, surface, family, unit, a, b, order in itertools.product(("core",), SURFACES, FAMILIES, range(len(UNITS)), (0, 1), (0, 1), (1, -1)):
        case = semantic_case(family, surface, unit, a, b)
        choices, gold = options(case["correct"], case["wrong"], order)
        rows.append({
            "case_id": f"c247-core-{family}-{surface}-u{unit}-{a}{b}-{order:+d}", "panel": panel,
            "family": family, "surface": surface, "unit": unit, "factor_a": a, "factor_b": b,
            "order": order, "gold_position": gold, "correct_answer": case["correct"], "wrong_answer": case["wrong"],
            "prompt_core": case["prompt_core"], "prompt": f"{case['prompt_core']} {choices}. Reply with only A or B.",
            "free_prompt": f"{case['prompt_core']} Answer with only the answer word.", "role_values": case["roles"],
        })
    for surface, unit, a, b, order in itertools.product(SURFACES, range(len(UNITS)), (0, 1), (0, 1), (1, -1)):
        case = nested_case(surface, unit, a, b)
        choices, gold = options(case["correct"], case["wrong"], order)
        rows.append({
            "case_id": f"c247-nested-{surface}-u{unit}-{a}{b}-{order:+d}", "panel": "nested_composition",
            "family": "nested_attitude", "surface": surface, "unit": unit, "factor_a": a, "factor_b": b,
            "order": order, "gold_position": gold, "correct_answer": case["correct"], "wrong_answer": case["wrong"],
            "prompt_core": case["prompt_core"], "prompt": f"{case['prompt_core']} {choices}. Reply with only A or B.",
            "free_prompt": f"{case['prompt_core']} Answer with only the answer word.", "role_values": case["roles"],
        })
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidates):
        raise RuntimeError(candidates)
    system = "Answer only from the supplied text. Do not use outside knowledge."
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = core.chat_ids(tokenizer, system, row["free_prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def factorial_effect(cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    return old.factorial_effect(cells)


def beta_effect(cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    values = factorial_effect(cells)
    values[0] *= 0.5
    values[1] *= 0.5
    values[2] *= 0.25
    return values


def signed_jaccard(pred: np.ndarray, truth: np.ndarray) -> float:
    union = (pred != 0) | (truth != 0)
    return float(np.mean(pred[union] == truth[union])) if union.any() else 1.0
