#!/usr/bin/env python3
"""Shared frozen objects for the C293-C309 conditional-response campaign."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1811_c277_c289_joint_response_common as previous

core = previous.core
model_base = previous.model_base
DIM = previous.DIM
WIDTH = previous.WIDTH
ROLES = previous.ROLES
FAMILIES = previous.FAMILIES
SURFACES = previous.SURFACES
MODELS = previous.MODELS
RAW_CHECKPOINTS = previous.RAW_CHECKPOINTS
CANONICAL_NEW_INDICES = previous.CANONICAL_NEW_INDICES
CANONICAL_CHECKPOINTS = previous.CANONICAL_CHECKPOINTS

OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for phase, campaign, slug in (
        (1827, "C293", "conditional_hypergraph_master_contract"),
        (1828, "C294", "sixth_material_semantic_compiler"),
        (1829, "C295", "qwen_sixth_material_full_field"),
        (1830, "C296", "complete_three_state_transition"),
        (1831, "C297", "continuous_amplitude_regimes"),
        (1832, "C298", "cross_coordinate_transfer_map"),
        (1833, "C299", "all_token_aligned_transfer"),
        (1834, "C300", "sixth_material_model_tournament"),
        (1835, "C301", "nonabsorbing_autonomous_rollout"),
        (1836, "C302", "six_family_composition_forecast"),
        (1837, "C303", "type_graph_rename_forecast"),
        (1838, "C304", "nested_attitude_patient_composition"),
        (1839, "C305", "cross_coordinate_causal_qualification"),
        (1840, "C306", "multisource_multitarget_causal"),
        (1841, "C307", "cross_model_anonymous_transition_topology"),
        (1842, "C308", "campaign_adjudication_heatmap"),
        (1843, "C309", "independent_campaign_audit"),
    )
}

# Sixth disjoint vocabulary. The linguistic templates remain unchanged, so this
# is a lexical/graph-renaming lockbox, not a new-syntax lockbox.
UNITS = (
    {"primary": "Adira", "secondary": "Bennet", "observer": "Corin", "object": "jicama", "other": "barometer", "node": "jicorin", "middle": "tuberel", "parent": "produce", "wrong": "instrument"},
    {"primary": "Della", "secondary": "Eamon", "observer": "Flora", "object": "romanesco", "other": "spinet", "node": "romavel", "middle": "brassica", "parent": "vegetable", "wrong": "music"},
    {"primary": "Galen", "secondary": "Hana", "observer": "Ivor", "object": "salsify", "other": "odometer", "node": "salorin", "middle": "rootelin", "parent": "food", "wrong": "device"},
    {"primary": "Jora", "secondary": "Keir", "observer": "Livia", "object": "cherimoya", "other": "cornet", "node": "cheravin", "middle": "orchalin", "parent": "plant", "wrong": "sound"},
    {"primary": "Marek", "secondary": "Nadia", "observer": "Oren", "object": "rutabaga", "other": "seismograph", "node": "rutavel", "middle": "taprorin", "parent": "organism", "wrong": "number"},
    {"primary": "Priya", "secondary": "Quillan", "observer": "Rosa", "object": "celtuce", "other": "zither", "node": "celorin", "middle": "stemavel", "parent": "entity", "wrong": "vehicle"},
    {"primary": "Saira", "secondary": "Tobin", "observer": "Una", "object": "feijoa", "other": "altimeter", "node": "fejavin", "middle": "guavora", "parent": "edible", "wrong": "time"},
    {"primary": "Vela", "secondary": "Wystan", "observer": "Xenia", "object": "kohlrabi", "other": "clavichord", "node": "kohrelin", "middle": "colevara", "parent": "living thing", "wrong": "mineral"},
)


def material() -> list[dict]:
    base = previous.previous
    old_units = base.UNITS
    try:
        base.UNITS = UNITS
        rows = base.material()
    finally:
        base.UNITS = old_units
    result = []
    for row in rows:
        updated = dict(row)
        updated["case_id"] = row["case_id"].replace("c263-", "c293-")
        updated["material"] = "sixth"
        updated["semantic_graph"] = {**row["semantic_graph"], "material": "sixth"}
        result.append(updated)
    return result


def compile_qwen(tokenizer, rows: list[dict]) -> list[dict]:
    return previous.previous.compile_qwen(tokenizer, rows)


def pair_specs(index: list[dict], family: str, effect: str = "factor_a"):
    panel = "nested_composition" if family == "nested_attitude" else "core"
    return previous.previous.pair_specs(index, family, effect, panel)


def event(delta: np.ndarray, threshold: float) -> np.ndarray:
    return previous.event(delta, threshold)


def thresholds() -> np.ndarray:
    return previous.thresholds()


def canonical(states: np.ndarray) -> np.ndarray:
    return states[:, CANONICAL_NEW_INDICES] if states.shape[1] == 38 else states


def transition_kind(current: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """0=persist-zero, 1=birth, 2=persist-active, 3=death, 4=reversal."""
    return np.where(
        current == 0,
        np.where(nxt == 0, 0, 1),
        np.where(nxt == current, 2, np.where(nxt == 0, 3, 4)),
    ).astype(np.int8)


def metric_counts(prediction: np.ndarray, truth: np.ndarray) -> dict:
    union = (prediction != 0) | (truth != 0)
    correct = prediction == truth
    return {
        "signed_jaccard": float(correct[union].mean()) if union.any() else 1.0,
        "exact_accuracy": float(correct.mean()),
        "active_precision": float((correct & (prediction != 0)).sum() / max(int((prediction != 0).sum()), 1)),
        "active_recall": float((correct & (truth != 0)).sum() / max(int((truth != 0).sum()), 1)),
        "union": int(union.sum()),
    }


def transition_recall(prediction: np.ndarray, current: np.ndarray, truth: np.ndarray) -> dict:
    kinds = transition_kind(current, truth)
    names = ("persist_zero", "birth", "persist_active", "death", "reversal")
    values = {}
    recalls = []
    for code, name in enumerate(names):
        mask = kinds == code
        total = int(mask.sum())
        recall = float((prediction[mask] == truth[mask]).mean()) if total else None
        values[name] = {"count": total, "recall": recall}
        if recall is not None:
            recalls.append(recall)
    values["macro_recall"] = float(np.mean(recalls)) if recalls else 0.0
    return values
