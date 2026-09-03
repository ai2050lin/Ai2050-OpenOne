#!/usr/bin/env python3
"""C254: export tri-material and full-token coordinate fields for the client."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C254"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c254_tri_material_event_atlas.json"
CHECKPOINTS = (0, 8, 16, 24, 32, 36)
ROLES = ("relation", "boundary")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = {name: core.load(common.OUTS[name] / "audit/independent_final_audit.json") for name in ("C249", "C250", "C251", "C252", "C253")}
    tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    up = np.load(common.OUTS["C250"] / "analysis/family_effect_checkpoint_coordinate_up_counts.int32.npy", mmap_mode="r")
    down = np.load(common.OUTS["C250"] / "analysis/family_effect_checkpoint_coordinate_down_counts.int32.npy", mmap_mode="r")
    checks = {"parents": all(value["all_checks_passed"] for value in parents.values()), "tri": list(tri.shape) == [5, 3, 37, 6, 2560], "token_counts": list(up.shape) == [5, 2, 37, 2560], "all_coordinates": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {"phase": 1788, "campaign": "C254", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "heatmap_export_frozen", "axes": ["source", "family", "effect", "checkpoint", "role", "all_2560_activation_coordinates"], "producer_sha256": core.sha(Path(__file__)), "authorization": "export_once_and_integrate_client"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    rows = []
    for fi, family in enumerate(common.FAMILIES):
        for ei, effect in enumerate(common.EFFECTS):
            for q in CHECKPOINTS:
                for role in ROLES:
                    ri = common.ROLES.index(role)
                    values = np.asarray(tri[fi, ei, q, ri], np.int8)
                    rows.append({"source": "tri_material_role_core", "family": family, "effect": effect, "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": role, "event_count": int(np.count_nonzero(values)), "label": f"tri/{family}/{effect}/q{q}/{role}", "values": values.astype(int).tolist()})
        for ei, effect in enumerate(("factor_a", "factor_b")):
            for q in CHECKPOINTS:
                total = np.asarray(up[fi, ei, q] + down[fi, ei, q], np.float64)
                values = np.divide(np.asarray(up[fi, ei, q] - down[fi, ei, q], np.float64), np.maximum(total, 1), where=np.ones_like(total, dtype=bool))
                rows.append({"source": "full_token_signed_balance", "family": family, "effect": effect, "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": "all_exactly_aligned_tokens", "event_count": int(total.sum()), "label": f"token/{family}/{effect}/q{q}/all", "values": values.astype(float).tolist()})
    importance = np.count_nonzero(np.asarray(tri)[:, :, CHECKPOINTS], axis=(0, 1, 2, 3))
    default_coordinates = np.argsort(-importance)[:64].astype(int).tolist()
    c249 = core.load(common.OUTS["C249"] / "analysis/summary.json")
    c250 = core.load(common.OUTS["C250"] / "analysis/summary.json")
    c251 = core.load(common.OUTS["C251"] / "analysis/summary.json")
    c252 = core.load(common.OUTS["C252"] / "analysis/summary.json")
    c253 = core.load(common.OUTS["C253"] / "analysis/summary.json")
    payload = {
        "schema": "c254_tri_material_event_atlas.v1", "phase": 1788, "campaign": "C254", "model": "Qwen3-4B",
        "dimensions": list(range(2560)), "default_coordinates": default_coordinates,
        "coordinate_semantics": "Tri rows are -1/0/+1 same-coordinate same-sign events across three material systems. Token rows are signed event balance over every exactly aligned token observation. Coordinate ids are activations, not weights.",
        "claim_boundary": "Role rows average within researcher-defined spans; token rows preserve individual aligned tokens but omit unmatched edited spans. Neither row type is a semantic neuron dictionary or unique causal circuit.",
        "summary": {"behavior_accuracy": core.load(common.OUTS["C248"] / "analysis/behavior_capture.json")["global_accuracy"], "tri_material_events": c249["tri_material_events"], "embedding_events": c249["embedding_events"], "hidden_events": c249["hidden_events"], "token_alignment_coverage": c250["matched_token_coverage_median"], "token_checkpoint_persistence": c250["same_coordinate_same_sign_persistence_median"], "composition": c251["summaries"], "causal_trajectory_passed": c252.get("trajectory_gate_passed", False), "cross_model_gate_passed": c253["cross_model_gate_passed"]},
        "rows": rows,
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    with ASSET.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
    asset_checks = {"rows": len(rows) == 240, "dimensions": all(len(row["values"]) == 2560 for row in rows), "finite": bool(np.isfinite([value for row in rows for value in row["values"]]).all()), "embedding_and_hidden": {row["checkpoint_type"] for row in rows} == {"embedding", "hidden_state"}}
    manifest = {"asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"), "bytes": ASSET.stat().st_size, "sha256": core.sha(ASSET), "checks": asset_checks, "all_checks_passed": all(asset_checks.values())}
    core.save(OUT / "analysis/heatmap_manifest.json", manifest)
    final_checks = {"parents": True, "asset": all(asset_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1788, "campaign": "C254", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": payload["summary"], "asset": manifest, "next_authorization": "C255_theory_and_independent_campaign_audit"}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": final["next_authorization"]})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
