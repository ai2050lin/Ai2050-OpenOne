#!/usr/bin/env python3
"""Shared material and full-field utilities for the C234-C243 event campaign."""
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
FAMILIES = ("attitude_event", "type_graph", "contrast", "translation", "comparison")
EFFECTS = ("factor_a", "factor_b", "interaction")
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
SURFACE_PARTITION = {
    "archive": "discovery",
    "bulletin": "discovery",
    "consultation": "confirmation",
    "chronicle": "lockbox",
    "dispatch": "fresh",
}
PARTITION_UNITS = {
    "discovery": (0, 1, 2, 3),
    "confirmation": (4, 5, 6),
    "lockbox": (7, 8),
    "fresh": (9, 10, 11),
}
PARTITIONS = tuple(PARTITION_UNITS)
SURFACES = tuple(SURFACE_PARTITION)

OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for phase, campaign, slug in (
        (1768, "C234", "fresh_event_master_contract"),
        (1769, "C235", "qwen_all_layer_full_token_capture"),
        (1770, "C236", "full_coordinate_interval_events"),
        (1771, "C237", "conditional_event_rule_discovery"),
        (1772, "C238", "unseen_surface_event_prediction"),
        (1773, "C239", "five_flagship_event_observation"),
        (1774, "C240", "factor_composition_event_prediction"),
        (1775, "C241", "path_consistent_causal_adjudication"),
        (1776, "C242", "cross_model_abstract_event_graph"),
        (1777, "C243", "campaign_theory_heatmap_closure"),
    )
}

UNITS = (
    {"primary": "Adela", "secondary": "Boris", "observer": "Cora", "object": "apricot", "other": "badger", "node": "navel", "middle": "pome", "parent": "produce", "wrong": "mineral"},
    {"primary": "Dalia", "secondary": "Emil", "observer": "Faye", "object": "radish", "other": "cello", "node": "radin", "middle": "rooten", "parent": "crop", "wrong": "vehicle"},
    {"primary": "Galen", "secondary": "Hana", "observer": "Iris", "object": "mango", "other": "helmet", "node": "mavik", "middle": "tropen", "parent": "food", "wrong": "fabric"},
    {"primary": "Jonas", "secondary": "Kira", "observer": "Lena", "object": "turnip", "other": "lantern", "node": "torin", "middle": "bulven", "parent": "plant", "wrong": "machine"},
    {"primary": "Marek", "secondary": "Nadia", "observer": "Olia", "object": "plum", "other": "violin", "node": "plorin", "middle": "fruven", "parent": "edible", "wrong": "building"},
    {"primary": "Pavel", "secondary": "Rina", "observer": "Sora", "object": "celery", "other": "anchor", "node": "celvik", "middle": "stalken", "parent": "vegetable", "wrong": "instrument"},
    {"primary": "Tomas", "secondary": "Una", "observer": "Vera", "object": "guava", "other": "pillow", "node": "guarin", "middle": "frulan", "parent": "organism", "wrong": "device"},
    {"primary": "Willa", "secondary": "Xeno", "observer": "Yara", "object": "squash", "other": "tablet", "node": "squorin", "middle": "gourven", "parent": "living thing", "wrong": "metal"},
    {"primary": "Amina", "secondary": "Bran", "observer": "Cyra", "object": "papaya", "other": "drum", "node": "paprin", "middle": "frucen", "parent": "object", "wrong": "sound"},
    {"primary": "Dorin", "secondary": "Elsa", "observer": "Freya", "object": "beet", "other": "camera", "node": "bevik", "middle": "rootic", "parent": "material", "wrong": "weather"},
    {"primary": "Greta", "secondary": "Hector", "observer": "Ilia", "object": "pear", "other": "kettle", "node": "perin", "middle": "orchic", "parent": "entity", "wrong": "motion"},
    {"primary": "Jora", "secondary": "Kellan", "observer": "Mira", "object": "yam", "other": "tripod", "node": "yamor", "middle": "tuberic", "parent": "substance", "wrong": "location"},
)


def options(correct: str, wrong: str, order: int) -> tuple[str, int]:
    if order == 1:
        return f"(A) {correct} (B) {wrong}", 0
    return f"(A) {wrong} (B) {correct}", 1


def wrap(surface: str, fact1: str, fact2: str, question: str) -> str:
    if surface == "archive":
        return f"Archive entry alpha: {fact1} Archive entry beta: {fact2} Query: {question}"
    if surface == "bulletin":
        return f"The bulletin records that {fact1} It separately reports that {fact2} On this basis, {question}"
    if surface == "consultation":
        return f'During a consultation, one note says, "{fact1}" Another says, "{fact2}" Please decide: {question}'
    if surface == "chronicle":
        return f"A chronicle first notes that {fact1} Later it notes that {fact2} Determine this: {question}"
    if surface == "dispatch":
        return f"Field dispatch: {fact1} Additional detail: {fact2} Resolve the question: {question}"
    raise KeyError(surface)


def semantic_case(family: str, surface: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    obj, other = u["object"], u["other"]
    node, middle, parent, wrong_parent = u["node"], u["middle"], u["parent"], u["wrong"]

    if family == "attitude_event":
        relation = "welcomes" if a == 0 else "opposes"
        target = f"{o} {relation} the fact that {p} ate the {obj}." if b == 0 else f"{o} {relation} the fact that the {obj} was eaten by {p}."
        noise = f"{s} carried the {other}."
        question, correct, wrong = f"Who ate the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    elif family == "type_graph":
        relation = "belongs"
        if a == 0:
            target = f"The {node} belongs to the category {parent}."
            noise = f"The {middle} belongs to the category {wrong_parent}."
        else:
            target = f"The {node} belongs to the category {middle}."
            noise = f"The {middle} belongs to the category {parent}."
        if b:
            target += f" A direct index also places the {node} under {parent}."
        question, correct, wrong = f"Which final category contains the {node}?", parent, wrong_parent
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
    elif family == "contrast":
        relation = "but" if a == 0 else "although"
        if a == 0:
            target = f"{p} is prepared, but {s} is delayed." if b == 0 else f"{s} is delayed, but {p} is prepared."
        else:
            target = f"Although {s} is delayed, {p} is prepared." if b == 0 else f"{p} is prepared although {s} is delayed."
            relation = "Although" if b == 0 else "although"
        noise = f"The {obj} remains nearby."
        question, correct, wrong = "Who is prepared?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": "prepared"}
    elif family == "translation":
        relation = "denotes"
        if a == 0:
            target = f'In the codebook, "{node}" denotes "{parent}".'
            noise = f'"{middle}" denotes "{wrong_parent}".'
        else:
            target = f'In the codebook, "{node}" denotes "{middle}".'
            noise = f'"{middle}" denotes "{parent}".'
        if b:
            target += f' A direct gloss also maps "{node}" to "{parent}".'
        question, correct, wrong = f'What does "{node}" ultimately denote?', parent, wrong_parent
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
    elif family == "comparison":
        dimension = "taller" if a == 0 else "heavier"
        inverse = "shorter" if a == 0 else "lighter"
        relation = dimension if b == 0 else inverse
        target = f"{p} is {dimension} than {s}." if b == 0 else f"{s} is {inverse} than {p}."
        noise = f"The {obj} is beside the {other}."
        question, correct, wrong = f"Who is {dimension}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": s, "query": dimension}
    else:
        raise KeyError(family)

    prompt_core = wrap(surface, target, noise, question)
    return {"prompt_core": prompt_core, "correct": correct, "wrong": wrong, "roles": roles}


def material() -> list[dict]:
    rows = []
    for surface in SURFACES:
        partition = SURFACE_PARTITION[surface]
        for family, unit, a, b, order in itertools.product(FAMILIES, PARTITION_UNITS[partition], (0, 1), (0, 1), (1, -1)):
            case = semantic_case(family, surface, unit, a, b)
            choices, gold = options(case["correct"], case["wrong"], order)
            rows.append({
                "case_id": f"c234-{family}-{surface}-u{unit:02d}-{a}{b}-{order:+d}",
                "family": family,
                "surface": surface,
                "partition": partition,
                "unit": unit,
                "factor_a": a,
                "factor_b": b,
                "order": order,
                "gold_position": gold,
                "correct_answer": case["correct"],
                "wrong_answer": case["wrong"],
                "prompt_core": case["prompt_core"],
                "prompt": f"{case['prompt_core']} {choices}. Reply with only A or B.",
                "free_prompt": f"{case['prompt_core']} Answer with only the answer word.",
                "role_values": case["roles"],
            })
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(ids) != 1 for ids in candidates):
        raise RuntimeError({"candidate_ids": candidates})
    compiled = []
    system = "Answer only from the supplied text. Do not use outside knowledge."
    for row in rows:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = core.chat_ids(tokenizer, system, row["free_prompt"])
        role_positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            role_positions[role] = spans[-1] if role == "query" else spans[0]
        role_positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids, "candidate_ids": candidates, "role_positions": role_positions})
    return compiled


def hidden_key(index_rows: list[dict]) -> dict[tuple, int]:
    return {
        (row["family"], row["surface"], int(row["unit"]), int(row["factor_a"]), int(row["factor_b"]), int(row["order"])): int(row["hidden_index"])
        for row in index_rows
    }


def factorial_effect(cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    a = 0.5 * ((cells[(1, 0)] + cells[(1, 1)]) - (cells[(0, 0)] + cells[(0, 1)]))
    b = 0.5 * ((cells[(0, 1)] + cells[(1, 1)]) - (cells[(0, 0)] + cells[(1, 0)]))
    ab = cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)]
    return np.stack((a, b, ab), axis=0)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, np.float64).reshape(-1)
    b = np.asarray(right, np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
