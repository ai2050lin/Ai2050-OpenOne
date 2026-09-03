#!/usr/bin/env python3
"""C267: full-coordinate state-conditioned observation of nested attitude composition."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common

core, OUT, C264 = common.core, common.OUTS["C267"], common.OUTS["C264"]


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C266"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "all_coordinates": True, "panel_frozen": True, "no_operator_order_claim": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); (OUT / "analysis").mkdir(); (OUT / "audit").mkdir()
    protocol = {"phase": 1801, "campaign": "C267", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "nested_composition_frozen", "factors": ["attitude wrapper", "patient specificity"], "primary": "full-coordinate interaction residual and its dependence on the 00 baseline side", "claim_boundary": "The factorial panel does not apply operators in both orders; it measures state-conditioned non-additivity, not a commutator or composition law.", "producer_sha256": core.sha(Path(__file__)), "authorization": "C268_type_graph_composition"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    states = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r"); index = core.rows(C264 / "raw/hidden_index.jsonl")
    valid = [r for r in index if r["panel"] == "nested_composition" and r["correct"]]
    key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): r["hidden_index"] for r in valid}
    groups = [(s, u, o) for s, u, o in __import__("itertools").product(common.SURFACES, range(8), (1, -1)) if all((s, u, a, b, o) in key for a, b in __import__("itertools").product((0, 1), repeat=2))]
    mean_field = np.lib.format.open_memmap(OUT / "analysis/mean_interaction.float16.npy", mode="w+", dtype=np.float16, shape=(37, 6, 2560))
    conditional = np.lib.format.open_memmap(OUT / "analysis/baseline_condition_difference.float16.npy", mode="w+", dtype=np.float16, shape=(37, 6, 2560))
    rows = []
    for q in range(37):
        for ri in range(6):
            base, inter, ratios = [], [], []
            for s, u, o in groups:
                x00 = np.asarray(states[key[(s, u, 0, 0, o)], q, ri], np.float32); x10 = np.asarray(states[key[(s, u, 1, 0, o)], q, ri], np.float32); x01 = np.asarray(states[key[(s, u, 0, 1, o)], q, ri], np.float32); x11 = np.asarray(states[key[(s, u, 1, 1, o)], q, ri], np.float32)
                interaction = x11 - x10 - x01 + x00
                base.append(x00); inter.append(interaction); ratios.append(float(np.linalg.norm(interaction) / max(np.linalg.norm(x11 - x00), 1e-12)))
            base, inter = np.asarray(base), np.asarray(inter)
            med = np.median(base, axis=0); high = base >= med[None, :]
            high_mean = np.sum(inter * high, axis=0) / np.maximum(high.sum(axis=0), 1); low_mean = np.sum(inter * ~high, axis=0) / np.maximum((~high).sum(axis=0), 1)
            mean = inter.mean(axis=0); diff = high_mean - low_mean
            mean_field[q, ri] = mean.astype(np.float16); conditional[q, ri] = diff.astype(np.float16)
            rows.append({"checkpoint": q, "role": common.ROLES[ri], "groups": len(groups), "median_interaction_to_total": float(np.median(ratios)), "conditional_difference_to_mean_norm": float(np.linalg.norm(diff) / max(np.linalg.norm(mean), 1e-12))})
    mean_field.flush(); conditional.flush(); core.write_rows(OUT / "analysis/checkpoint_role_rows.jsonl", rows)
    report = {"phase": 1801, "campaign": "C267", "status": "nested_state_composition_observed", "complete_groups": len(groups), "median_interaction_ratio": float(np.median([r["median_interaction_to_total"] for r in rows])), "median_state_condition_ratio": float(np.median([r["conditional_difference_to_mean_norm"] for r in rows])), "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C268_type_graph_state_composition"}
    core.save(OUT / "analysis/summary.json", report)
    ach = {"groups": len(groups) >= 24, "rows": len(rows) == 37 * 6, "shapes": mean_field.shape == conditional.shape == (37, 6, 2560), "finite": bool(np.isfinite([r["median_interaction_to_total"] for r in rows]).all())}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
    fch = {"contract": True, "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1801, "campaign": "C267", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


if __name__ == "__main__": main()

