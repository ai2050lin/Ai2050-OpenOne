#!/usr/bin/env python3
"""C228: select full-coordinate interaction models for five language families."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C228"]
SOURCE = common.OUTS["C224"]
MODELS = ("additive", "global_interaction", "family_interaction", "surface_interaction", "family_surface_interaction")


def cells(states: np.ndarray, key: dict, family: str, surface: str, unit: int) -> dict:
    return {(a, b): np.asarray(states[key[(family, surface, unit, a, b)]], np.float32) for a in (0, 1) for b in (0, 1)}


def parts(cell: dict) -> tuple[np.ndarray, np.ndarray]:
    additive = (cell[(1, 0)] - cell[(0, 0)]) + (cell[(0, 1)] - cell[(0, 0)])
    truth = cell[(1, 1)] - cell[(0, 0)]
    return additive, truth - additive


def model_interaction(model: str, bank: dict[str, np.ndarray], family_i: int, surface_i: int) -> np.ndarray:
    if model == "additive":
        return np.zeros_like(bank["global"])
    if model == "global_interaction":
        return bank["global"]
    if model == "family_interaction":
        return bank["family"][family_i]
    if model == "surface_interaction":
        return bank["surface"][surface_i]
    if model == "family_surface_interaction":
        return bank["family_surface"][family_i, surface_i]
    raise KeyError(model)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C227"] / "audit/independent_final_audit.json")
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1762, "campaign": "C228", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "five_family_composition_tournament_frozen", "models": list(MODELS),
        "discovery_units": [0, 1, 2], "confirmation_units": [3, 4, 5], "forbidden_units": [6, 7, 8],
        "selection": "lowest median confirmation NRMSE separately for each family; fixed candidate order breaks exact ties",
        "claim_boundary": "The two frozen factors are family-specific controlled edits. A pass is compositional response prediction, not proof of universal linguistic algebra.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C229_open_composition_lockbox_once_with_frozen_family_models",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    states = np.load(SOURCE / "raw/role_states.float16.npy", mmap_mode="r")
    key = common.hidden_key(core.rows(SOURCE / "raw/hidden_index.jsonl"))
    interactions = np.empty((5, 4, 3, 4, 6, common.DIM), np.float32)
    for fi, family in enumerate(common.TARGET_FAMILIES):
        for si, surface in enumerate(common.SURFACES):
            values = []
            for unit in range(3):
                _add, interaction = parts(cells(states, key, family, surface, unit))
                values.append(interaction)
            interactions[fi, si] = np.mean(np.stack(values), axis=0)
    bank = {"global": interactions.mean(axis=(0, 1)), "family": interactions.mean(axis=1), "surface": interactions.mean(axis=0), "family_surface": interactions}
    np.savez_compressed(OUT / "protocol/interaction_templates.npz", **{name: value.astype(np.float16) for name, value in bank.items()})
    rows = []
    for fi, family in enumerate(common.TARGET_FAMILIES):
        for si, surface in enumerate(common.SURFACES):
            for unit in range(3, 6):
                additive, interaction = parts(cells(states, key, family, surface, unit))
                truth = additive + interaction
                for model in MODELS:
                    prediction = additive + model_interaction(model, bank, fi, si)
                    rows.append({"model": model, "family": family, "surface": surface, "unit": unit, "nrmse": common.nrmse(prediction, truth), "weighted_sign": common.weighted_sign(prediction, truth), "interaction_ratio": float(np.linalg.norm(interaction.astype(np.float64)) / max(np.linalg.norm(truth.astype(np.float64)), 1e-30))})
    selected = {}
    for family in common.TARGET_FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        medians = {model: float(np.median([row["nrmse"] for row in family_rows if row["model"] == model])) for model in MODELS}
        selected[family] = min(MODELS, key=lambda model: (medians[model], MODELS.index(model)))
    selected_rows = [row for row in rows if row["model"] == selected[row["family"]]]
    summary = {family: {"selected_model": selected[family], "median_nrmse": float(np.median([row["nrmse"] for row in selected_rows if row["family"] == family])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in selected_rows if row["family"] == family])), "median_interaction_ratio": float(np.median([row["interaction_ratio"] for row in selected_rows if row["family"] == family]))} for family in common.TARGET_FAMILIES}
    freeze = {"selected_by_family": selected, "template_sha256": core.sha(OUT / "protocol/interaction_templates.npz"), "confirmation_summary": summary, "lockbox_rule": "unit 6-8 remain sealed until C229"}
    core.save(OUT / "protocol/composition_model_freeze.json", freeze)
    core.write_rows(OUT / "analysis/confirmation_rows.jsonl", rows)
    report = {"phase": 1762, "campaign": "C228", "status": "composition_models_selected", "selected": selected, "by_family": summary, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/tournament_summary.json", report)
    checks = {"authorization": parent["all_checks_passed"], "rows": len(rows) == 300, "models": set(row["model"] for row in rows) == set(MODELS), "families": set(selected) == set(common.TARGET_FAMILIES), "no_lockbox": max(row["unit"] for row in rows) == 5, "finite": bool(np.isfinite([row[k] for row in rows for k in ("nrmse", "weighted_sign", "interaction_ratio")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1762, "campaign": "C228", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({"checks": checks, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

