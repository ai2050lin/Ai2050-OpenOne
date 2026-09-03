#!/usr/bin/env python3
"""Shared material and full-coordinate utilities for the C223-C233 campaign."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1739_c205_response_ecology_common as previous

DIM = 2560
WIDTH = 128
CHECKPOINTS = ("embedding", "q23", "q24", "q25")
ROLES = previous.ROLES
EFFECTS = ("factor_a", "factor_b", "interaction")
SURFACES = ("records", "briefing", "dialogue", "narrative")
TARGET_FAMILIES = ("attitude_event", "type_graph", "contrast", "translation", "comparison")
CALIBRATION_FAMILIES = ("possession", "negation", "causality")
FAMILIES = TARGET_FAMILIES + CALIBRATION_FAMILIES

OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for phase, campaign, slug in (
        (1757, "C223", "semantic_surface_master_contract"),
        (1758, "C224", "qwen_full_coordinate_observation"),
        (1759, "C225", "coordinate_passport_atlas"),
        (1760, "C226", "surface_transport_tournament"),
        (1761, "C227", "surface_transport_lockbox"),
        (1762, "C228", "five_family_composition_tournament"),
        (1763, "C229", "five_family_composition_lockbox"),
        (1764, "C230", "causal_eligibility_adjudication"),
        (1765, "C231", "cross_model_functional_topology"),
        (1766, "C232", "mathematical_upgrade_adjudication"),
        (1767, "C233", "campaign_synthesis_heatmap"),
    )
}

UNITS = (
    {"primary": "Avery", "secondary": "Bren", "observer": "Luma", "object": "lantern", "other_object": "violin", "node": "loran", "middle": "selvik", "parent": "dravic", "other_parent": "noric"},
    {"primary": "Celia", "secondary": "Daren", "observer": "Mara", "object": "compass", "other_object": "kettle", "node": "merek", "middle": "tovin", "parent": "peloric", "other_parent": "veskan"},
    {"primary": "Elin", "secondary": "Faron", "observer": "Nessa", "object": "harp", "other_object": "bucket", "node": "surnet", "middle": "kalven", "parent": "rimic", "other_parent": "dastor"},
    {"primary": "Gita", "secondary": "Halen", "observer": "Orla", "object": "telescope", "other_object": "basket", "node": "valen", "middle": "pravin", "parent": "cendric", "other_parent": "moltar"},
    {"primary": "Ivara", "secondary": "Jorin", "observer": "Pella", "object": "flute", "other_object": "hammer", "node": "kespar", "middle": "dovik", "parent": "heloran", "other_parent": "trenic"},
    {"primary": "Kara", "secondary": "Lio", "observer": "Rhea", "object": "goblet", "other_object": "anchor", "node": "brelan", "middle": "sovic", "parent": "mantor", "other_parent": "zelric"},
    {"primary": "Mina", "secondary": "Nolan", "observer": "Sela", "object": "camera", "other_object": "drum", "node": "ferrin", "middle": "lasken", "parent": "porvic", "other_parent": "galdor"},
    {"primary": "Oren", "secondary": "Pia", "observer": "Talia", "object": "tablet", "other_object": "helmet", "node": "jovek", "middle": "neldar", "parent": "soramic", "other_parent": "vintor"},
    {"primary": "Quin", "secondary": "Rosa", "observer": "Vera", "object": "tripod", "other_object": "pillow", "node": "wexin", "middle": "corven", "parent": "latric", "other_parent": "bemar"},
)


def partition(unit: int) -> str:
    return ("discovery", "confirmation", "lockbox")[unit // 3]


def options(correct: str, wrong: str, order: int) -> tuple[str, int]:
    if order == 1:
        return f"(A) {correct} (B) {wrong}", 0
    return f"(A) {wrong} (B) {correct}", 1


def wrap(surface: str, fact1: str, fact2: str, question: str) -> str:
    if surface == "records":
        return f"Record one: {fact1} Record two: {fact2} Request: {question}"
    if surface == "briefing":
        return f"A briefing states that {fact1} It also notes that {fact2} Based on the briefing, {question}"
    if surface == "dialogue":
        return f'Mira reports, "{fact1}" Niko adds, "{fact2}" {question}'
    if surface == "narrative":
        return f"In a short account, {fact1} Meanwhile, {fact2} Decide this: {question}"
    raise KeyError(surface)


def phrase(surface: str, choices: tuple[str, str, str, str]) -> str:
    return choices[SURFACES.index(surface)]


def semantic_case(family: str, surface: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    obj, other = u["object"], u["other_object"]
    node, middle, parent, other_parent = u["node"], u["middle"], u["parent"], u["other_parent"]

    if family == "attitude_event":
        positive = phrase(surface, ("likes", "welcomes", "pleased", "approves"))
        negative = phrase(surface, ("dislikes", "opposes", "upset", "objects"))
        relation = positive if a == 0 else negative
        if surface == "dialogue":
            target = f"{o} is {relation} because the {obj} was eaten by {p}." if b else f"{o} is {relation} because {p} ate the {obj}."
        elif surface == "narrative":
            target = f"{o} {relation} to the {obj} being eaten by {p}." if b else f"{o} {relation} of {p} eating the {obj}."
        else:
            target = f"{o} {relation} the fact that the {obj} was eaten by {p}." if b else f"{o} {relation} the fact that {p} ate the {obj}."
        noise = f"{s} ate the {other}."
        question, correct, wrong = f"Who ate the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    elif family == "type_graph":
        relation = phrase(surface, ("kind", "category", "classify", "catalogued"))
        if a == 0:
            target = f"The {node} is a {relation} of {parent}."
            bridge = f"The {middle} is a {relation} of {other_parent}."
        else:
            target = f"The {node} is a {relation} of {middle}."
            bridge = f"The {middle} is a {relation} of {parent}."
        if b:
            target += f" The {node} is also explicitly listed under {parent}."
        question, correct, wrong = f"Which category ultimately contains the {node}?", parent, other_parent
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
        noise = bridge
    elif family == "contrast":
        relation = "but" if a == 0 else "although"
        if a == 0:
            target = f"{p} is ready, but {s} is delayed." if b == 0 else f"{s} is delayed, but {p} is ready."
        else:
            target = f"Although {s} is delayed, {p} is ready." if b == 0 else f"{p} is ready although {s} is delayed."
            relation = "Although" if b == 0 else "although"
        noise = f"The {obj} remains packed."
        question, correct, wrong = "Who is ready?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": "ready"}
    elif family == "translation":
        relation = phrase(surface, ("means", "corresponds", "translates", "signifies"))
        if a == 0:
            target = f'In the code, "{node}" {relation} "{parent}".'
            bridge = f'"{middle}" {relation} "{other_parent}".'
        else:
            target = f'In the code, "{node}" {relation} "{middle}".'
            bridge = f'"{middle}" {relation} "{parent}".'
        if b:
            target += f' A direct note says "{node}" maps to "{parent}".'
        question, correct, wrong = f'What does "{node}" ultimately denote?', parent, other_parent
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
        noise = bridge
    elif family == "comparison":
        dimension = "taller" if a == 0 else "heavier"
        inverse = "shorter" if a == 0 else "lighter"
        relation = dimension if b == 0 else inverse
        target = f"{p} is {dimension} than {s}." if b == 0 else f"{s} is {inverse} than {p}."
        noise = f"The {obj} is beside the {other}."
        question, correct, wrong = f"Who is {dimension}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": s, "query": dimension}
    elif family == "possession":
        relation = phrase(surface, ("owns", "possesses", "belongs", "keeps"))
        if a == 0:
            target = f"{p} {relation} the {obj}."
        else:
            target = f"The {obj} {relation} to {p}." if relation == "belongs" else f"The {obj} is kept by {p}."
            relation = "belongs" if relation == "belongs" else "kept"
        noise = f"{s} owns the {other}."
        if b:
            target, noise = noise, target
        question, correct, wrong = f"Who owns the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    elif family == "negation":
        relation = "approved" if a == 0 else "rejected"
        if a == 0:
            target, noise = f"{p} is approved.", f"{s} is not approved."
        else:
            target, noise = f"{p} is not rejected.", f"{s} is rejected."
        if b:
            target, noise = noise, target
        question, correct, wrong = "Who is approved?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": s, "query": "approved"}
    elif family == "causality":
        relation = phrase(surface, ("caused", "triggered", "produced", "led"))
        if a == 0:
            target = f"{p} {relation} the alarm directly."
            bridge = f"{s} moved the {other}."
        else:
            target = f"{p} activated the {obj}."
            bridge = f"The {obj} {relation} the alarm."
        if b:
            target += f" A separate record says {p} directly caused the alarm."
        question, correct, wrong = "Who ultimately caused the alarm?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj if a else "alarm", "query": "alarm"}
        noise = bridge
    else:
        raise KeyError(family)

    prompt = wrap(surface, target, noise, question)
    return {"prompt_core": prompt, "role_values": roles, "correct": correct, "wrong": wrong}


def material() -> list[dict]:
    rows = []
    for family, surface, unit, a, b, order in itertools.product(FAMILIES, SURFACES, range(len(UNITS)), (0, 1), (0, 1), (1, -1)):
        case = semantic_case(family, surface, unit, a, b)
        choice, gold = options(case["correct"], case["wrong"], order)
        rows.append({
            "case_id": f"c223-{family}-{surface}-u{unit:02d}-{a}{b}-{order:+d}",
            "family": family,
            "family_kind": "target" if family in TARGET_FAMILIES else "calibration",
            "surface": surface,
            "unit": unit,
            "partition": partition(unit),
            "factor_a": a,
            "factor_b": b,
            "order": order,
            "gold_position": gold,
            "prompt": f"{case['prompt_core']} {choice}. Reply with only A or B.",
            "role_values": case["role_values"],
        })
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(ids) != 1 for ids in candidates):
        raise RuntimeError({"candidate_ids": candidates})
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, "Answer only from the supplied text. Reply exactly A or B.", row["prompt"])
        role_positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            role_positions[role] = spans[-1] if role == "query" else spans[0]
        role_positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": role_positions})
    return compiled


def response_cube(states: np.ndarray, key: dict[tuple, int], family: str, surface: str, unit: int) -> np.ndarray:
    cells = {(a, b): np.asarray(states[key[(family, surface, unit, a, b)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}
    a_effect = 0.5 * ((cells[(1, 0)] + cells[(1, 1)]) - (cells[(0, 0)] + cells[(0, 1)]))
    b_effect = 0.5 * ((cells[(0, 1)] + cells[(1, 1)]) - (cells[(0, 0)] + cells[(1, 0)]))
    interaction = cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)]
    return np.stack((a_effect, b_effect, interaction), axis=0)


def hidden_key(index_rows: list[dict]) -> dict[tuple, int]:
    return {(row["family"], row["surface"], int(row["unit"]), int(row["factor_a"]), int(row["factor_b"])): int(row["hidden_index"]) for row in index_rows}


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return previous.nrmse(prediction, truth)


def weighted_sign(prediction: np.ndarray, truth: np.ndarray) -> float:
    return previous.weighted_sign(prediction, truth)
