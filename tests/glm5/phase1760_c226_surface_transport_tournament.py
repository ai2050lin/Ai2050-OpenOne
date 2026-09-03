#!/usr/bin/env python3
"""C226: fit surface transports on calibration discovery and select on target confirmation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C226"]
SOURCE = common.OUTS["C225"]
MODELS = ("identity", "common_offset", "typed_scalar_affine", "typed_coordinate_gain", "typed_coordinate_affine")


def fit_models(x: np.ndarray, y: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    eps = 1e-8
    result: dict[str, dict[str, np.ndarray]] = {"identity": {}}
    result["common_offset"] = {"offset": (y - x).mean(axis=0)}
    axes = tuple(range(x.ndim - 4)) + (-1,)
    # x/y are [sample,effect,checkpoint,role,coordinate]. Scalar fits pool coordinates.
    xm = x.mean(axis=(0, 4), keepdims=False)
    ym = y.mean(axis=(0, 4), keepdims=False)
    xc = x - xm[None, ..., None]
    yc = y - ym[None, ..., None]
    alpha = np.sum(xc * yc, axis=(0, 4)) / np.maximum(np.sum(xc * xc, axis=(0, 4)), eps)
    alpha = np.clip(alpha, -4.0, 4.0)
    beta = ym - alpha * xm
    result["typed_scalar_affine"] = {"alpha": alpha, "beta": beta}
    gain = np.sum(x * y, axis=0) / np.maximum(np.sum(x * x, axis=0), eps)
    result["typed_coordinate_gain"] = {"alpha": np.clip(gain, -4.0, 4.0)}
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xcenter = x - xmean
    ycenter = y - ymean
    coord_alpha = np.sum(xcenter * ycenter, axis=0) / np.maximum(np.sum(xcenter * xcenter, axis=0), eps)
    coord_alpha = np.clip(coord_alpha, -4.0, 4.0)
    result["typed_coordinate_affine"] = {"alpha": coord_alpha, "beta": ymean - coord_alpha * xmean}
    return result


def predict(name: str, params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    if name == "identity":
        return x
    if name == "common_offset":
        return x + params["offset"]
    if name == "typed_scalar_affine":
        return params["alpha"][..., None] * x + params["beta"][..., None]
    if name == "typed_coordinate_gain":
        return params["alpha"] * x
    if name == "typed_coordinate_affine":
        return params["alpha"] * x + params["beta"]
    raise KeyError(name)


def save_params(path: Path, banks: dict[int, dict[str, dict[str, np.ndarray]]]) -> None:
    arrays = {}
    for surface_i, models in banks.items():
        for model, params in models.items():
            for name, value in params.items():
                arrays[f"s{surface_i}__{model}__{name}"] = np.asarray(value, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(SOURCE / "audit/independent_final_audit.json")
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1760, "campaign": "C226", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "surface_transport_tournament_frozen", "candidates": list(MODELS),
        "source_surface": common.SURFACES[0], "target_surfaces": list(common.SURFACES[1:]),
        "fit": {"families": list(common.CALIBRATION_FAMILIES), "units": [0, 1, 2]},
        "selection": {"families": list(common.TARGET_FAMILIES), "units": [3, 4, 5], "criterion": "lowest median NRMSE; frozen candidate order breaks exact ties"},
        "forbidden_units": [6, 7, 8],
        "claim_boundary": "Selection on confirmation is not lockbox evidence. Coordinatewise candidates can overfit nine calibration examples and must survive C227.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C227_open_lockbox_once_with_frozen_selected_transports_and_nulls",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    cubes = np.load(SOURCE / "analysis/response_cubes.float16.npy", mmap_mode="r")
    cal_indices = [common.FAMILIES.index(family) for family in common.CALIBRATION_FAMILIES]
    target_indices = [common.FAMILIES.index(family) for family in common.TARGET_FAMILIES]
    banks = {}
    rows = []
    selected = {}
    for surface_i, surface in enumerate(common.SURFACES[1:], start=1):
        xfit = np.asarray(cubes[cal_indices, 0, :3], np.float32).reshape(-1, 3, 4, 6, common.DIM)
        yfit = np.asarray(cubes[cal_indices, surface_i, :3], np.float32).reshape(-1, 3, 4, 6, common.DIM)
        banks[surface_i] = fit_models(xfit, yfit)
        for model in MODELS:
            for family_i in target_indices:
                family = common.FAMILIES[family_i]
                for unit in range(3, 6):
                    x = np.asarray(cubes[family_i, 0, unit], np.float32)
                    truth = np.asarray(cubes[family_i, surface_i, unit], np.float32)
                    pred = predict(model, banks[surface_i][model], x)
                    rows.append({"model": model, "family": family, "target_surface": surface, "unit": unit, "nrmse": common.nrmse(pred, truth), "weighted_sign": common.weighted_sign(pred, truth)})
        surface_rows = [row for row in rows if row["target_surface"] == surface]
        medians = {model: float(np.median([row["nrmse"] for row in surface_rows if row["model"] == model])) for model in MODELS}
        selected[surface] = min(MODELS, key=lambda model: (medians[model], MODELS.index(model)))
    save_params(OUT / "protocol/fitted_parameters.npz", banks)
    core.write_rows(OUT / "analysis/confirmation_rows.jsonl", rows)
    selected_rows = [row for row in rows if row["model"] == selected[row["target_surface"]]]
    identity_rows = [row for row in rows if row["model"] == "identity"]
    summary_by_model = {model: {"median_nrmse": float(np.median([row["nrmse"] for row in rows if row["model"] == model])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in rows if row["model"] == model]))} for model in MODELS}
    selected_summary = {"median_nrmse": float(np.median([row["nrmse"] for row in selected_rows])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in selected_rows])), "identity_median_nrmse": float(np.median([row["nrmse"] for row in identity_rows]))}
    gate = core.load(common.OUTS["C223"] / "protocol/preregistration.json")["transport_confirmation_gate"]
    passed = selected_summary["median_nrmse"] <= gate["median_nrmse_max"] and selected_summary["median_weighted_sign"] >= gate["median_weighted_sign_min"] and selected_summary["identity_median_nrmse"] - selected_summary["median_nrmse"] >= gate["identity_nrmse_improvement_min"]
    freeze = {"selected_by_target_surface": selected, "selection_rows": len(selected_rows), "confirmation_summary": selected_summary, "confirmation_gate": gate, "confirmation_gate_passed": passed, "parameter_sha256": core.sha(OUT / "protocol/fitted_parameters.npz"), "lockbox_authorization": "open_regardless_of_confirmation_gate_as_preregistered_benchmark_but_confirmatory_claim_requires_gate"}
    core.save(OUT / "protocol/model_freeze.json", freeze)
    report = {"phase": 1760, "campaign": "C226", "status": "transport_candidates_selected", "selected": selected, "summary_by_model": summary_by_model, "selected_summary": selected_summary, "confirmation_gate_passed": passed, "interpretation": "The calibration families fit surface transforms; target confirmation selects them. Lockbox has not been read.", "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/tournament_summary.json", report)
    checks = {"authorization": parent["all_checks_passed"], "rows": len(rows) == 225, "models": set(row["model"] for row in rows) == set(MODELS), "selected_surfaces": set(selected) == set(common.SURFACES[1:]), "no_lockbox": max(row["unit"] for row in rows) == 5, "params": (OUT / "protocol/fitted_parameters.npz").exists(), "finite": bool(np.isfinite([row[k] for row in rows for k in ("nrmse", "weighted_sign")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1760, "campaign": "C226", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({"checks": checks, "selected": selected, "summary": selected_summary, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()

