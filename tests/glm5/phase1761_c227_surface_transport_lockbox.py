#!/usr/bin/env python3
"""C227: one-time lockbox evaluation of frozen surface transports and nulls."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common
import phase1760_c226_surface_transport_tournament as c226

core = common.core
OUT = common.OUTS["C227"]
SOURCE = common.OUTS["C225"]
FREEZE_OUT = common.OUTS["C226"]
METHODS = ("selected", "identity", "wrong_surface", "wrong_family", "factor_swap", "coordinate_permutation", "same_norm_random", "energy_only")


def load_banks() -> dict:
    values = np.load(FREEZE_OUT / "protocol/fitted_parameters.npz")
    banks = {surface_i: {model: {} for model in c226.MODELS} for surface_i in range(1, 4)}
    for key in values.files:
        surface, model, name = key.split("__")
        banks[int(surface[1:])][model][name] = np.asarray(values[key], np.float32)
    return banks


def random_like(value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(value.shape, dtype=np.float32)
    value_norm = np.sqrt(np.sum(np.square(value, dtype=np.float64), axis=-1, keepdims=True))
    noise_norm = np.sqrt(np.sum(np.square(noise, dtype=np.float64), axis=-1, keepdims=True))
    return noise * (value_norm / np.maximum(noise_norm, 1e-30))


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(FREEZE_OUT / "audit/independent_final_audit.json")
    freeze = core.load(FREEZE_OUT / "protocol/model_freeze.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": 1761, "campaign": "C227", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "transport_lockbox_opened_once", "units": [6, 7, 8], "methods": list(METHODS),
        "selected_by_target_surface": freeze["selected_by_target_surface"],
        "epistemic_rule": "C226 confirmation gate is immutable; lockbox cannot retroactively repair a failed confirmation gate",
        "claim_boundary": "A lockbox pass requires exact signed-field accuracy and all null margins, not family classification alone.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C228_five_family_composition_tournament_continues_regardless_of_transport_outcome",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    cubes = np.load(SOURCE / "analysis/response_cubes.float16.npy", mmap_mode="r")
    banks = load_banks()
    rng = np.random.default_rng(227)
    rows = []
    field_index = []
    storage = np.lib.format.open_memmap(OUT / "analysis/selected_truth_fields.float16.npy", mode="w+", dtype=np.float16, shape=(45, 2, 3, 4, 6, common.DIM))
    case_i = 0
    for family_i, family in enumerate(common.TARGET_FAMILIES):
        global_fi = common.FAMILIES.index(family)
        wrong_fi = common.FAMILIES.index(common.TARGET_FAMILIES[(family_i + 1) % len(common.TARGET_FAMILIES)])
        for surface_i, surface in enumerate(common.SURFACES[1:], start=1):
            selected_model = freeze["selected_by_target_surface"][surface]
            wrong_surface_i = 1 + (surface_i % 3)
            wrong_surface = common.SURFACES[wrong_surface_i]
            wrong_model = freeze["selected_by_target_surface"][wrong_surface]
            for unit in range(6, 9):
                x = np.asarray(cubes[global_fi, 0, unit], np.float32)
                truth = np.asarray(cubes[global_fi, surface_i, unit], np.float32)
                selected_pred = c226.predict(selected_model, banks[surface_i][selected_model], x)
                wrong_x = np.asarray(cubes[wrong_fi, 0, unit], np.float32)
                predictions = {
                    "selected": selected_pred,
                    "identity": x,
                    "wrong_surface": c226.predict(wrong_model, banks[wrong_surface_i][wrong_model], x),
                    "wrong_family": c226.predict(selected_model, banks[surface_i][selected_model], wrong_x),
                    "factor_swap": selected_pred[[1, 0, 2]],
                    "coordinate_permutation": np.roll(selected_pred, 137, axis=-1),
                    "same_norm_random": random_like(selected_pred, rng),
                    "energy_only": np.broadcast_to(np.sqrt(np.mean(np.square(selected_pred, dtype=np.float64), axis=-1, keepdims=True)), selected_pred.shape),
                }
                for method, prediction in predictions.items():
                    rows.append({"method": method, "family": family, "target_surface": surface, "unit": unit, "selected_model": selected_model, "nrmse": common.nrmse(prediction, truth), "weighted_sign": common.weighted_sign(prediction, truth)})
                storage[case_i, 0] = selected_pred.astype(np.float16)
                storage[case_i, 1] = truth.astype(np.float16)
                field_index.append({"field_index": case_i, "family": family, "target_surface": surface, "unit": unit, "selected_model": selected_model})
                case_i += 1
    storage.flush()
    core.write_rows(OUT / "analysis/lockbox_rows.jsonl", rows)
    core.write_rows(OUT / "analysis/field_index.jsonl", field_index)
    summary = {method: {"median_nrmse": float(np.median([row["nrmse"] for row in rows if row["method"] == method])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in rows if row["method"] == method]))} for method in METHODS}
    gate = core.load(common.OUTS["C223"] / "protocol/preregistration.json")["transport_lockbox_gate"]
    selected = summary["selected"]
    margins = {method: summary[method]["median_nrmse"] - selected["median_nrmse"] for method in METHODS if method not in ("selected", "identity")}
    passed = freeze["confirmation_gate_passed"] and selected["median_nrmse"] <= gate["median_nrmse_max"] and selected["median_weighted_sign"] >= gate["median_weighted_sign_min"] and summary["identity"]["median_nrmse"] - selected["median_nrmse"] >= gate["identity_nrmse_improvement_min"] and min(margins.values()) >= gate["all_null_nrmse_margin_min"]
    by_family = {family: {"median_nrmse": float(np.median([row["nrmse"] for row in rows if row["method"] == "selected" and row["family"] == family])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in rows if row["method"] == "selected" and row["family"] == family]))} for family in common.TARGET_FAMILIES}
    report = {"phase": 1761, "campaign": "C227", "status": "transport_lockbox_adjudicated", "confirmation_gate_passed": freeze["confirmation_gate_passed"], "summary": summary, "null_nrmse_margins": margins, "by_family": by_family, "lockbox_gate": gate, "lockbox_gate_passed": passed, "interpretation": "The lockbox benchmarks the frozen transport but cannot repair C226. Null comparisons distinguish signed coordinate prediction from energy or label shortcuts.", "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/lockbox_summary.json", report)
    checks = {"authorization": parent["all_checks_passed"], "rows": len(rows) == 360, "methods": set(row["method"] for row in rows) == set(METHODS), "lockbox_only": {row["unit"] for row in rows} == {6, 7, 8}, "fields": storage.shape == (45, 2, 3, 4, 6, 2560), "finite": bool(np.isfinite([row[k] for row in rows for k in ("nrmse", "weighted_sign")]).all()), "freeze_unchanged": freeze["parameter_sha256"] == core.sha(FREEZE_OUT / "protocol/fitted_parameters.npz")}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1761, "campaign": "C227", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({"checks": checks, "summary": summary, "margins": margins, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
