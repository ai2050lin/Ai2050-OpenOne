#!/usr/bin/env python3
"""C229: lockbox validation of frozen five-family interaction models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common
import phase1762_c228_five_family_composition_tournament as c228

core = common.core
OUT = common.OUTS["C229"]
SOURCE = common.OUTS["C224"]
FREEZE_OUT = common.OUTS["C228"]
METHODS = ("selected", "additive", "wrong_family", "wrong_surface", "coordinate_permutation")


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(FREEZE_OUT / "audit/independent_final_audit.json")
    freeze = core.load(FREEZE_OUT / "protocol/composition_model_freeze.json")
    values = np.load(FREEZE_OUT / "protocol/interaction_templates.npz")
    bank = {name: np.asarray(values[name], np.float32) for name in values.files}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    protocol = {"phase": 1763, "campaign": "C229", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "composition_lockbox_opened_once", "methods": list(METHODS), "units": [6, 7, 8], "selected_by_family": freeze["selected_by_family"], "claim_boundary": "A family pass predicts its frozen two-factor response in this task panel. It does not prove arbitrary-depth language composition.", "producer_sha256": core.sha(Path(__file__)), "authorization": "C230_causal_eligibility_adjudication_then_continue_cross_model_route"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    states = np.load(SOURCE / "raw/role_states.float16.npy", mmap_mode="r")
    key = common.hidden_key(core.rows(SOURCE / "raw/hidden_index.jsonl"))
    rows = []
    atlas = np.lib.format.open_memmap(OUT / "analysis/prediction_truth_interaction.float16.npy", mode="w+", dtype=np.float16, shape=(60, 3, 3, 4, 6, common.DIM))
    atlas_index = []
    case_i = 0
    for fi, family in enumerate(common.TARGET_FAMILIES):
        selected_model = freeze["selected_by_family"][family]
        wrong_fi = (fi + 1) % len(common.TARGET_FAMILIES)
        for si, surface in enumerate(common.SURFACES):
            wrong_si = (si + 1) % len(common.SURFACES)
            for unit in range(6, 9):
                additive, interaction = c228.parts(c228.cells(states, key, family, surface, unit))
                truth = additive + interaction
                chosen = c228.model_interaction(selected_model, bank, fi, si)
                predictions = {
                    "selected": additive + chosen,
                    "additive": additive,
                    "wrong_family": additive + c228.model_interaction(selected_model, bank, wrong_fi, si),
                    "wrong_surface": additive + c228.model_interaction(selected_model, bank, fi, wrong_si),
                    "coordinate_permutation": additive + np.roll(chosen, 137, axis=-1),
                }
                for method, prediction in predictions.items():
                    rows.append({"method": method, "family": family, "surface": surface, "unit": unit, "selected_model": selected_model, "nrmse": common.nrmse(prediction, truth), "weighted_sign": common.weighted_sign(prediction, truth)})
                atlas[case_i, 0] = predictions["selected"].astype(np.float16)
                atlas[case_i, 1] = truth.astype(np.float16)
                atlas[case_i, 2] = interaction.astype(np.float16)
                atlas_index.append({"field_index": case_i, "family": family, "surface": surface, "unit": unit, "selected_model": selected_model})
                case_i += 1
    atlas.flush()
    core.write_rows(OUT / "analysis/lockbox_rows.jsonl", rows)
    core.write_rows(OUT / "analysis/atlas_index.jsonl", atlas_index)
    gate = core.load(common.OUTS["C223"] / "protocol/preregistration.json")["composition_lockbox_gate"]
    by_family = {}
    for family in common.TARGET_FAMILIES:
        selected_rows = [row for row in rows if row["method"] == "selected" and row["family"] == family]
        summary = {"selected_model": freeze["selected_by_family"][family], "median_nrmse": float(np.median([row["nrmse"] for row in selected_rows])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in selected_rows]))}
        summary["passed"] = summary["median_nrmse"] <= gate["family_median_nrmse_max"] and summary["median_weighted_sign"] >= gate["family_median_weighted_sign_min"]
        by_family[family] = summary
    families_passed = sum(value["passed"] for value in by_family.values())
    campaign_passed = families_passed >= gate["families_min"]
    method_summary = {method: {"median_nrmse": float(np.median([row["nrmse"] for row in rows if row["method"] == method])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in rows if row["method"] == method]))} for method in METHODS}
    report = {"phase": 1763, "campaign": "C229", "status": "composition_lockbox_adjudicated", "by_family": by_family, "families_passed": families_passed, "campaign_gate_passed": campaign_passed, "method_summary": method_summary, "gate": gate, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/lockbox_summary.json", report)
    checks = {"authorization": parent["all_checks_passed"], "rows": len(rows) == 300, "lockbox": {row["unit"] for row in rows} == {6, 7, 8}, "methods": set(row["method"] for row in rows) == set(METHODS), "atlas": atlas.shape == (60, 3, 3, 4, 6, 2560), "template_hash": core.sha(FREEZE_OUT / "protocol/interaction_templates.npz") == freeze["template_sha256"], "finite": bool(np.isfinite([row[k] for row in rows for k in ("nrmse", "weighted_sign")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1763, "campaign": "C229", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({"checks": checks, "by_family": by_family, "method_summary": method_summary, "passed": campaign_passed}, indent=2))


if __name__ == "__main__":
    main()
