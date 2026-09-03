#!/usr/bin/env python3
"""Shared frozen objects for the C263-C272 state-conditioned operator campaign."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1780_c246_c255_event_hypergraph_common as prior

core = prior.core
graph_base = prior.graph_base
previous = prior.previous

DIM = 2560
WIDTH = 128
FAMILIES = prior.FAMILIES
EFFECTS = prior.EFFECTS
ROLES = prior.ROLES
SURFACES = ("dossier", "hearing")
MODELS = prior.MODELS

OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for phase, campaign, slug in (
        (1797, "C263", "state_conditioned_operator_contract"),
        (1798, "C264", "qwen_fourth_material_full_field"),
        (1799, "C265", "coordinate_condition_passports"),
        (1800, "C266", "rolling_full_field_prediction"),
        (1801, "C267", "nested_attitude_state_composition"),
        (1802, "C268", "type_graph_state_composition"),
        (1803, "C269", "predicted_edge_local_causal"),
        (1804, "C270", "generated_language_and_side_effects"),
        (1805, "C271", "cross_model_conditional_bisimulation"),
        (1806, "C272", "campaign_adjudication_heatmap"),
    )
}

UNITS = (
    {"primary": "Alden", "secondary": "Brisa", "observer": "Cato", "object": "plantain", "other": "tripod", "node": "plavik", "middle": "frunel", "parent": "produce", "wrong": "mineral"},
    {"primary": "Davin", "secondary": "Esme", "observer": "Faron", "object": "okra", "other": "mandolin", "node": "okrin", "middle": "podrel", "parent": "vegetable", "wrong": "instrument"},
    {"primary": "Gilda", "secondary": "Hadrian", "observer": "Ines", "object": "quince", "other": "caliper", "node": "quinor", "middle": "pomera", "parent": "food", "wrong": "device"},
    {"primary": "Jensen", "secondary": "Kelda", "observer": "Leona", "object": "cassava", "other": "trombone", "node": "casvik", "middle": "tuberon", "parent": "plant", "wrong": "sound"},
    {"primary": "Maris", "secondary": "Nolan", "observer": "Opal", "object": "lychee", "other": "wheelbarrow", "node": "lycorin", "middle": "orchera", "parent": "organism", "wrong": "vehicle"},
    {"primary": "Petra", "secondary": "Ruben", "observer": "Selene", "object": "parsnip", "other": "abacus", "node": "parvik", "middle": "rootera", "parent": "entity", "wrong": "number"},
    {"primary": "Tarin", "secondary": "Ulla", "observer": "Vito", "object": "zucchini", "other": "hourglass", "node": "zucorin", "middle": "gourmet", "parent": "edible", "wrong": "time"},
    {"primary": "Wanda", "secondary": "Xerxes", "observer": "Ysolde", "object": "currant", "other": "harpsichord", "node": "curalin", "middle": "berel", "parent": "living thing", "wrong": "music"},
)


def options(correct: str, wrong: str, order: int) -> tuple[str, int]:
    return prior.options(correct, wrong, order)


def wrap(surface: str, fact1: str, fact2: str, question: str) -> str:
    if surface == "dossier":
        return f"A dossier states: {fact1} A separate entry states: {fact2} Question: {question}"
    if surface == "hearing":
        return f"During a hearing, a witness said that {fact1} The witness also noted that {fact2} Decide: {question}"
    raise KeyError(surface)


def semantic_case(family: str, surface: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    obj, other = u["object"], u["other"]
    node, middle, parent, wrong = u["node"], u["middle"], u["parent"], u["wrong"]
    if family == "attitude_event":
        relation = "endorses" if a == 0 else "challenges"
        target = f"{o} {relation} the report that {p} inspected the {obj}." if b == 0 else f"{o} {relation} the report that the {obj} was inspected by {p}."
        noise = f"{s} adjusted the {other}."
        question, correct, distractor = f"Who inspected the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
        graph = {"nodes": [o, p, obj], "edges": [[o, relation, "report"], [p, "inspected", obj]], "edit_a": "attitude", "edit_b": "voice"}
    elif family == "type_graph":
        relation = "is catalogued within"
        if a == 0:
            target, noise = f"The {node} is catalogued within {parent}.", f"The {middle} is catalogued within {wrong}."
        else:
            target, noise = f"The {node} is catalogued within {middle}.", f"The {middle} is catalogued within {parent}."
        if b:
            target += f" A direct register also places the {node} within {parent}."
        question, correct, distractor = f"Which final class contains the {node}?", parent, wrong
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
        graph = {"nodes": [node, middle, parent, wrong], "edges": [[node, "member", middle if a else parent], [middle, "member", parent if a else wrong]], "direct_shortcut": bool(b)}
    elif family == "contrast":
        relation = "still" if a == 0 else ("While" if b == 0 else "while")
        if a == 0:
            target = f"{s} seemed hesitant; still, {p} stayed decisive." if b == 0 else f"{p} stayed decisive; still, {s} seemed hesitant."
        else:
            target = f"While {s} seemed hesitant, {p} stayed decisive." if b == 0 else f"{p} stayed decisive while {s} seemed hesitant."
        noise = f"The {obj} remained beside the {other}."
        question, correct, distractor = "Who stayed decisive?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": "decisive"}
        graph = {"nodes": [p, s], "edges": [[p, "decisive", True], [s, "hesitant", True]], "connective": relation, "clause_order": b}
    elif family == "translation":
        relation = "stands for"
        if a == 0:
            target, noise = f'In this lexicon, "{node}" stands for "{parent}".', f'"{middle}" stands for "{wrong}".'
        else:
            target, noise = f'In this lexicon, "{node}" stands for "{middle}".', f'"{middle}" stands for "{parent}".'
        if b:
            target += f' A direct note also maps "{node}" to "{parent}".'
        question, correct, distractor = f'What does "{node}" ultimately stand for?', parent, wrong
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
        graph = {"nodes": [node, middle, parent, wrong], "edges": [[node, "maps", middle if a else parent], [middle, "maps", parent if a else wrong]], "direct_shortcut": bool(b)}
    elif family == "comparison":
        dimension, inverse = (("heavier", "lighter") if a == 0 else ("older", "younger"))
        relation = dimension if b == 0 else inverse
        target = f"{p} is {dimension} than {s}." if b == 0 else f"{s} is {inverse} than {p}."
        noise = f"The {obj} lies near the {other}."
        question, correct, distractor = f"Who is {dimension}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": s, "query": dimension}
        graph = {"nodes": [p, s], "edges": [[p, dimension, s]], "dimension": dimension, "surface_direction": b}
    else:
        raise KeyError(family)
    return {"prompt_core": wrap(surface, target, noise, question), "correct": correct, "wrong": distractor, "roles": roles, "semantic_graph": graph}


def nested_case(surface: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    value = u["object"] if b else u["parent"]
    relation = "was pleased to learn that" if a else "reported that"
    target = f"{o} {relation} {p} ate the {value}."
    noise = f"{s} moved the {u['other']}."
    question = f"What did {p} eat?"
    roles = {"primary": p, "secondary": s, "relation": relation, "context": value, "query": p}
    graph = {"nodes": [o, p, value], "edges": [[p, "ate", value], [o, "attitude", "pleased" if a else "reported"]], "patient_specific": bool(b)}
    return {"prompt_core": wrap(surface, target, noise, question), "correct": value, "wrong": u["parent"] if b else u["object"], "roles": roles, "semantic_graph": graph}


def material() -> list[dict]:
    rows: list[dict] = []
    for surface, family, unit, a, b, order in itertools.product(SURFACES, FAMILIES, range(len(UNITS)), (0, 1), (0, 1), (1, -1)):
        case = semantic_case(family, surface, unit, a, b)
        choices, gold = options(case["correct"], case["wrong"], order)
        rows.append({
            "case_id": f"c263-core-{family}-{surface}-u{unit}-{a}{b}-{order:+d}", "panel": "core", "family": family,
            "surface": surface, "unit": unit, "factor_a": a, "factor_b": b, "order": order, "gold_position": gold,
            "correct_answer": case["correct"], "wrong_answer": case["wrong"], "prompt_core": case["prompt_core"],
            "prompt": f"{case['prompt_core']} {choices}. Reply with only A or B.",
            "free_prompt": f"{case['prompt_core']} Answer with only the answer word.",
            "role_values": case["roles"], "semantic_graph": case["semantic_graph"],
        })
    for surface, unit, a, b, order in itertools.product(SURFACES, range(len(UNITS)), (0, 1), (0, 1), (1, -1)):
        case = nested_case(surface, unit, a, b)
        choices, gold = options(case["correct"], case["wrong"], order)
        rows.append({
            "case_id": f"c263-nested-{surface}-u{unit}-{a}{b}-{order:+d}", "panel": "nested_composition", "family": "nested_attitude",
            "surface": surface, "unit": unit, "factor_a": a, "factor_b": b, "order": order, "gold_position": gold,
            "correct_answer": case["correct"], "wrong_answer": case["wrong"], "prompt_core": case["prompt_core"],
            "prompt": f"{case['prompt_core']} {choices}. Reply with only A or B.",
            "free_prompt": f"{case['prompt_core']} Answer with only the answer word.",
            "role_values": case["roles"], "semantic_graph": case["semantic_graph"],
        })
    return rows


def compile_qwen(tokenizer, rows: list[dict]) -> list[dict]:
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


def pair_specs(index: list[dict], family: str, effect: str = "factor_a", panel: str = "core") -> list[tuple[int, int, dict]]:
    rows = [row for row in index if row["panel"] == panel and row["family"] == family and row.get("correct", row.get("behavior_correct", False))]
    key = {(row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in rows}
    pairs = []
    for surface, unit, fixed, order in itertools.product(sorted({r["surface"] for r in rows}), sorted({r["unit"] for r in rows}), (0, 1), (1, -1)):
        left_key = (surface, unit, 0, fixed, order) if effect == "factor_a" else (surface, unit, fixed, 0, order)
        right_key = (surface, unit, 1, fixed, order) if effect == "factor_a" else (surface, unit, fixed, 1, order)
        if left_key in key and right_key in key:
            left, right = key[left_key], key[right_key]
            pairs.append((left["hidden_index"], right["hidden_index"], {"surface": surface, "unit": unit, "fixed_factor": fixed, "order": order}))
    return pairs


def events(delta: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    shape = (len(thresholds),) + (1,) * (delta.ndim - 1)
    t = thresholds.reshape(shape)
    return np.where(delta > t, 1, np.where(delta < -t, -1, 0)).astype(np.int8)


def signed_jaccard(pred: np.ndarray, truth: np.ndarray) -> float:
    union = (pred != 0) | (truth != 0)
    return float(np.mean(pred[union] == truth[union])) if union.any() else 1.0

