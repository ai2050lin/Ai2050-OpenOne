#!/usr/bin/env python3
"""C225: build a missing-aware, full-coordinate response passport without lockbox use."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C225"]
SOURCE = common.OUTS["C224"]


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    return {"nrmse": common.nrmse(prediction, truth), "weighted_sign": common.weighted_sign(prediction, truth)}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(SOURCE / "audit/independent_final_audit.json")
    protocol = {
        "phase": 1759, "campaign": "C225", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "coordinate_passport_observation_frozen",
        "fit_units": [0, 1, 2], "confirmation_units": [3, 4, 5], "forbidden_units": [6, 7, 8],
        "outputs": ["full response cubes", "mean signed coordinate passport", "coordinate sign persistence", "within-surface confirmation", "raw cross-surface confirmation"],
        "claim_boundary": "This phase describes response regularity and surface mismatch. It does not select a transport model or inspect lockbox rows.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "top-k-only storage", "lockbox access", "causal claim"],
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C226_fit_frozen_transport_candidates_using_calibration_discovery_then_select_on_target_confirmation",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    core.save(OUT / "protocol/preregistration.json", protocol)
    states = np.load(SOURCE / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(SOURCE / "raw/hidden_index.jsonl")
    key = common.hidden_key(index)
    shape = (len(common.FAMILIES), len(common.SURFACES), len(common.UNITS), len(common.EFFECTS), len(common.CHECKPOINTS), len(common.ROLES), common.DIM)
    cubes = np.lib.format.open_memmap(OUT / "analysis/response_cubes.float16.npy", mode="w+", dtype=np.float16, shape=shape)
    for fi, family in enumerate(common.FAMILIES):
        for si, surface in enumerate(common.SURFACES):
            for unit in range(len(common.UNITS)):
                cubes[fi, si, unit] = common.response_cube(states, key, family, surface, unit).astype(np.float16)
    cubes.flush()
    observed = np.asarray(cubes[:, :, :6], np.float32)
    passport_mean = observed.mean(axis=2)
    signs = np.sign(observed)
    sign_persistence = np.abs(signs.mean(axis=2))
    np.save(OUT / "analysis/passport_mean.float16.npy", passport_mean.astype(np.float16))
    np.save(OUT / "analysis/passport_sign_persistence.float16.npy", sign_persistence.astype(np.float16))
    core.save(OUT / "analysis/axes.json", {"families": list(common.FAMILIES), "surfaces": list(common.SURFACES), "units": list(range(9)), "effects": list(common.EFFECTS), "checkpoints": list(common.CHECKPOINTS), "roles": list(common.ROLES), "coordinates": common.DIM})

    within_rows = []
    cross_rows = []
    for fi, family in enumerate(common.FAMILIES):
        for si, surface in enumerate(common.SURFACES):
            template = np.asarray(cubes[fi, si, :3], np.float32).mean(axis=0)
            for unit in range(3, 6):
                within_rows.append({"family": family, "surface": surface, "unit": unit, **metric(template, np.asarray(cubes[fi, si, unit], np.float32))})
        source_template = np.asarray(cubes[fi, 0, :3], np.float32).mean(axis=0)
        for si, surface in enumerate(common.SURFACES[1:], start=1):
            for unit in range(3, 6):
                cross_rows.append({"family": family, "source_surface": common.SURFACES[0], "target_surface": surface, "unit": unit, **metric(source_template, np.asarray(cubes[fi, si, unit], np.float32))})

    family_summary = {}
    for family in common.FAMILIES:
        same = [row for row in within_rows if row["family"] == family]
        cross = [row for row in cross_rows if row["family"] == family]
        family_summary[family] = {
            "within_surface_confirmation_median_nrmse": float(np.median([row["nrmse"] for row in same])),
            "within_surface_confirmation_median_weighted_sign": float(np.median([row["weighted_sign"] for row in same])),
            "raw_cross_surface_confirmation_median_nrmse": float(np.median([row["nrmse"] for row in cross])),
            "raw_cross_surface_confirmation_median_weighted_sign": float(np.median([row["weighted_sign"] for row in cross])),
        }
    report = {
        "phase": 1759, "campaign": "C225", "status": "full_coordinate_passport_observed",
        "within_surface": {"support": len(within_rows), "median_nrmse": float(np.median([row["nrmse"] for row in within_rows])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in within_rows]))},
        "raw_cross_surface": {"support": len(cross_rows), "median_nrmse": float(np.median([row["nrmse"] for row in cross_rows])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in cross_rows]))},
        "by_family": family_summary,
        "passport_shape": list(passport_mean.shape),
        "interpretation": "The passport preserves every signed physical coordinate. Within-surface lexical stability and cross-surface mismatch are separate observations; neither establishes a semantic operator.",
        "next_authorization": protocol["authorization"],
    }
    core.write_rows(OUT / "analysis/within_surface_rows.jsonl", within_rows)
    core.write_rows(OUT / "analysis/raw_cross_surface_rows.jsonl", cross_rows)
    core.save(OUT / "analysis/coordinate_passport_summary.json", report)
    checks = {
        "authorization": parent["all_checks_passed"], "cube_shape": list(cubes.shape) == list(shape),
        "passport_shape": list(passport_mean.shape) == [8, 4, 3, 4, 6, 2560],
        "within_support": len(within_rows) == 96, "cross_support": len(cross_rows) == 72,
        "no_lockbox_metrics": max(row["unit"] for row in within_rows + cross_rows) == 5,
        "finite": bool(np.isfinite([row[k] for row in within_rows + cross_rows for k in ("nrmse", "weighted_sign")]).all()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1759, "campaign": "C225", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({"checks": checks, "headline": report}, indent=2))


if __name__ == "__main__":
    main()
