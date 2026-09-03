#!/usr/bin/env python3
"""Shared frozen objects for the C277-C289 joint-response campaign."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1797_c263_c272_state_operator_common as previous

core = previous.core
graph_base = previous.graph_base
model_base = previous.previous

DIM = 2560
WIDTH = 128
ROLES = previous.ROLES
FAMILIES = previous.FAMILIES + ("nested_attitude",)
SURFACES = previous.SURFACES
MODELS = previous.MODELS

# The wrapped Transformers API returns embedding + block 1..35 + final norm for
# the old 37-state archives. C278 records the missing block-36 output explicitly.
RAW_CHECKPOINTS = (
    "embedding",
    *tuple(f"block_{i:02d}_output" for i in range(1, 37)),
    "final_norm",
)
CANONICAL_NEW_INDICES = tuple(range(36)) + (37,)
CANONICAL_CHECKPOINTS = (
    "embedding",
    *tuple(f"block_{i:02d}_output" for i in range(1, 36)),
    "final_norm",
)

OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for phase, campaign, slug in (
        (1811, "C277", "joint_response_master_contract"),
        (1812, "C278", "qwen_fifth_material_full_field"),
        (1813, "C279", "joint_state_word_partition"),
        (1814, "C280", "multisource_one_step_prediction"),
        (1815, "C281", "eligible_long_horizon_rollout"),
        (1816, "C282", "natural_attitude_composition"),
        (1817, "C283", "type_graph_composition"),
        (1818, "C284", "order_commutation_comparison"),
        (1819, "C285", "prospective_hyperedge_causal"),
        (1820, "C286", "generation_and_side_effects"),
        (1821, "C287", "cross_model_joint_state_capture"),
        (1822, "C288", "cross_model_automaton_isomorphism"),
        (1823, "C289", "campaign_adjudication_heatmap"),
    )
}

# Fifth, disjoint lexical material. The six task families and two surfaces are
# retained so that old discovery archives can be used without coordinate fitting.
UNITS = (
    {"primary": "Anika", "secondary": "Boris", "observer": "Celine", "object": "apricot", "other": "sextant", "node": "avolin", "middle": "fruxel", "parent": "food", "wrong": "instrument"},
    {"primary": "Dario", "secondary": "Elina", "observer": "Farid", "object": "artichoke", "other": "metronome", "node": "artovin", "middle": "leafora", "parent": "vegetable", "wrong": "device"},
    {"primary": "Greta", "secondary": "Hector", "observer": "Ilona", "object": "kumquat", "other": "theodolite", "node": "kumelin", "middle": "citrava", "parent": "produce", "wrong": "mineral"},
    {"primary": "Jonas", "secondary": "Kira", "observer": "Lucan", "object": "endive", "other": "celesta", "node": "endovin", "middle": "greenset", "parent": "plant", "wrong": "music"},
    {"primary": "Meera", "secondary": "Niko", "observer": "Orla", "object": "persimmon", "other": "astrolabe", "node": "peralin", "middle": "orchavel", "parent": "organism", "wrong": "number"},
    {"primary": "Pavel", "secondary": "Rina", "observer": "Soren", "object": "radicchio", "other": "dulcimer", "node": "radovin", "middle": "rootavel", "parent": "entity", "wrong": "sound"},
    {"primary": "Talia", "secondary": "Ulric", "observer": "Vera", "object": "tamarind", "other": "clinometer", "node": "tamorin", "middle": "podavel", "parent": "edible", "wrong": "time"},
    {"primary": "Willa", "secondary": "Xavian", "observer": "Yara", "object": "watercress", "other": "concertina", "node": "watelin", "middle": "herbora", "parent": "living thing", "wrong": "vehicle"},
)


def material() -> list[dict]:
    """Build a fifth material with the frozen factorial semantics and new words."""
    old_units = previous.UNITS
    try:
        previous.UNITS = UNITS
        rows = previous.material()
    finally:
        previous.UNITS = old_units
    result = []
    for row in rows:
        updated = dict(row)
        updated["case_id"] = row["case_id"].replace("c263-", "c277-")
        updated["material"] = "fifth"
        updated["semantic_graph"] = {**row["semantic_graph"], "material": "fifth"}
        result.append(updated)
    return result


def compile_qwen(tokenizer, rows: list[dict]) -> list[dict]:
    return previous.compile_qwen(tokenizer, rows)


def pair_specs(index: list[dict], family: str, effect: str = "factor_a"):
    panel = "nested_composition" if family == "nested_attitude" else "core"
    return previous.pair_specs(index, family, effect, panel)


def event(delta: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0)).astype(np.int8)


def canonical_new(states: np.ndarray) -> np.ndarray:
    return states[:, CANONICAL_NEW_INDICES]


def thresholds() -> np.ndarray:
    return np.asarray(
        core.load(previous.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"],
        np.float32,
    )


def event_metrics(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    union = (predicted != 0) | (truth != 0)
    return np.asarray([
        ((predicted == truth) & union).sum(),
        union.sum(),
        ((predicted == truth) & (predicted != 0)).sum(),
        (predicted != 0).sum(),
        (truth != 0).sum(),
    ], np.int64)


def metric_dict(total: np.ndarray) -> dict:
    exact, union, active_correct, predicted_active, truth_active = [int(x) for x in total]
    return {
        "signed_jaccard": float(exact / max(union, 1)),
        "signed_precision": float(active_correct / max(predicted_active, 1)),
        "signed_recall": float(active_correct / max(truth_active, 1)),
        "union": union,
    }

