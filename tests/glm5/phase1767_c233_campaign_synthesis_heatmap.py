#!/usr/bin/env python3
"""C233: parameter-level heatmap synthesis and C223-C233 major-stage closure."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C233"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c233_surface_transport_composition_atlas.json"
ROLE_IDS = (common.ROLES.index("relation"), common.ROLES.index("boundary"))


def add(rows: list[dict], value: np.ndarray, **metadata) -> None:
    vector = np.asarray(value, np.float32).reshape(-1)
    if vector.shape != (common.DIM,) or not np.isfinite(vector).all():
        raise RuntimeError((metadata, vector.shape))
    rows.append({**metadata, "values": vector.astype(float).tolist()})


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C232"] / "audit/independent_final_audit.json")
    OUT.mkdir(parents=True)
    protocol = {"phase": 1767, "campaign": "C233", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "parameter_heatmap_and_major_stage_closure_frozen", "asset_rows": 840, "coordinates_per_row": 2560, "row_sources": ["C225 passport means", "C227 transport prediction/truth/error", "C229 composition prediction/truth/interaction"], "claim_boundary": "Default coordinates are display aids. The asset contains all 2560 physical activation coordinates and does not turn failed gates into mechanism claims.", "producer_sha256": core.sha(Path(__file__)), "authorization": "close_C223_C233_and_start_only_a_fresh_preregistered_observation_campaign"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    rows = []
    passport = np.load(common.OUTS["C225"] / "analysis/passport_mean.float16.npy", mmap_mode="r")
    for fi, family in enumerate(common.TARGET_FAMILIES):
        for si, surface in enumerate(common.SURFACES):
            for ei, effect in enumerate(common.EFFECTS):
                for qi, checkpoint in enumerate(common.CHECKPOINTS):
                    for role_i in ROLE_IDS:
                        add(rows, passport[fi, si, ei, qi, role_i], source="C225_passport", family=family, surface=surface, field="mean_response", effect=effect, checkpoint=checkpoint, role=common.ROLES[role_i], label=f"passport / {family} / {surface} / {effect} / {checkpoint} / {common.ROLES[role_i]}")

    transport = np.load(common.OUTS["C227"] / "analysis/selected_truth_fields.float16.npy", mmap_mode="r")
    transport_index = core.rows(common.OUTS["C227"] / "analysis/field_index.jsonl")
    for family in common.TARGET_FAMILIES:
        selected = [row["field_index"] for row in transport_index if row["family"] == family]
        fields = {"prediction": np.asarray(transport[selected, 0], np.float32).mean(axis=0), "truth": np.asarray(transport[selected, 1], np.float32).mean(axis=0)}
        fields["error"] = fields["prediction"] - fields["truth"]
        for field, values in fields.items():
            for ei, effect in enumerate(common.EFFECTS):
                for qi in (2, 3):
                    for role_i in ROLE_IDS:
                        add(rows, values[ei, qi, role_i], source="C227_transport_lockbox", family=family, surface="all_target_surfaces", field=field, effect=effect, checkpoint=common.CHECKPOINTS[qi], role=common.ROLES[role_i], label=f"transport {field} / {family} / {effect} / {common.CHECKPOINTS[qi]} / {common.ROLES[role_i]}")

    composition = np.load(common.OUTS["C229"] / "analysis/prediction_truth_interaction.float16.npy", mmap_mode="r")
    composition_index = core.rows(common.OUTS["C229"] / "analysis/atlas_index.jsonl")
    for family in common.TARGET_FAMILIES:
        selected = [row["field_index"] for row in composition_index if row["family"] == family]
        for source_i, field in enumerate(("prediction", "truth", "interaction")):
            values = np.asarray(composition[selected, source_i], np.float32).mean(axis=0)
            for ei, effect in enumerate(common.EFFECTS):
                for qi in (2, 3):
                    for role_i in ROLE_IDS:
                        add(rows, values[ei, qi, role_i], source="C229_composition_lockbox", family=family, surface="all_surfaces", field=field, effect=effect, checkpoint=common.CHECKPOINTS[qi], role=common.ROLES[role_i], label=f"composition {field} / {family} / {effect} / {common.CHECKPOINTS[qi]} / {common.ROLES[role_i]}")

    matrix = np.asarray([row["values"] for row in rows], np.float32)
    c227 = core.load(common.OUTS["C227"] / "analysis/lockbox_summary.json")
    c229 = core.load(common.OUTS["C229"] / "analysis/lockbox_summary.json")
    c231 = core.load(common.OUTS["C231"] / "analysis/cross_model_summary.json")
    c232 = core.load(common.OUTS["C232"] / "analysis/theory_adjudication.json")
    summary = {"transport_confirmation_passed": c227["confirmation_gate_passed"], "transport_lockbox_passed": c227["lockbox_gate_passed"], "transport_selected_nrmse": c227["summary"]["selected"]["median_nrmse"], "composition_families_passed": c229["families_passed"], "composition_campaign_passed": c229["campaign_gate_passed"], "cross_model_passed": c231["cross_model_gate_passed"], "new_mathematics_authorized": c232["new_foundational_mathematics_authorized"]}
    asset = {
        "schema": "c233_surface_transport_composition_atlas.v1", "result_type": "surface_transport_composition_atlas_heatmap",
        "phase": 1767, "campaign": "C233", "model": "Qwen3-4B fields plus dimension-free three-model summary",
        "title": "C223-C233 Surface Transport and Composition Atlas", "dimensions": list(range(common.DIM)),
        "default_coordinates": np.argsort(-np.mean(np.abs(matrix), axis=0))[:64].astype(int).tolist(), "rows": rows, "summary": summary,
        "coordinate_semantics": "Each column is one original Qwen3-4B physical activation coordinate. Passport rows include embedding/q23/q24/q25; lockbox rows show q24/q25 prediction, truth, error or interaction.",
        "claim_boundary": protocol["claim_boundary"],
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    core.save(ASSET, asset)
    report = {"phase": 1767, "campaign": "C233", "status": "major_stage_closed", "asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"), "asset_rows": len(rows), "coordinates": common.DIM, "summary": summary, "strict_conclusion": "The campaign found stable within-surface signed response families, a narrow prospective surface correction, two family-specific compositional regularities, and two pairwise cross-model topology similarities. The strong transport, broad composition, causal, all-model, and new-mathematics gates failed.", "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final_synthesis.json", report)
    checks = {"authorization": parent["all_checks_passed"], "rows": len(rows) == 840, "matrix": matrix.shape == (840, 2560), "all_checkpoints": set(row["checkpoint"] for row in rows) == set(common.CHECKPOINTS), "all_coordinates": len(asset["dimensions"]) == 2560, "asset": ASSET.exists(), "finite": bool(np.isfinite(matrix).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1767, "campaign": "C233", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({"checks": checks, "summary": summary, "asset_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

