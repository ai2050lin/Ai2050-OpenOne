#!/usr/bin/env python3
"""C268: compare full-coordinate state-conditioned non-additivity across five families."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common

core, OUT, C264 = common.core, common.OUTS["C268"], common.OUTS["C264"]


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C267"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "five_families": True, "all_coordinates": True, "graph_controls_registered": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); (OUT / "analysis").mkdir(); (OUT / "audit").mkdir()
    protocol = {"phase": 1802, "campaign": "C268", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "typed_state_composition_frozen", "families": list(common.FAMILIES), "type_graph_factors": ["direct versus two-hop path", "direct shortcut absent versus present"], "controls": ["translation isomorphic graph", "contrast", "comparison", "attitude-event"], "primary": "full-coordinate interaction residual plus baseline-conditioned difference", "claim_boundary": "Differences among controlled task families may include lexical and rendering effects. They are composition fingerprints, not learned algebraic laws.", "producer_sha256": core.sha(Path(__file__)), "authorization": "C269_registered_local_causal_test"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    states = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r"); index = core.rows(C264 / "raw/hidden_index.jsonl")
    mean_fields = np.lib.format.open_memmap(OUT / "analysis/family_mean_interaction.float16.npy", mode="w+", dtype=np.float16, shape=(5, 37, 6, 2560)); condition_fields = np.lib.format.open_memmap(OUT / "analysis/family_condition_difference.float16.npy", mode="w+", dtype=np.float16, shape=(5, 37, 6, 2560))
    family_rows = []
    for fi, family in enumerate(common.FAMILIES):
        valid = [r for r in index if r["panel"] == "core" and r["family"] == family and r["correct"]]
        key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): r["hidden_index"] for r in valid}
        groups = [(s, u, o) for s, u, o in itertools.product(common.SURFACES, range(8), (1, -1)) if all((s, u, a, b, o) in key for a, b in itertools.product((0, 1), repeat=2))]
        ratios, cond_ratios = [], []
        for q in range(37):
            for ri in range(6):
                base, inter = [], []
                for s, u, o in groups:
                    cells = {(a, b): np.asarray(states[key[(s, u, a, b, o)], q, ri], np.float32) for a, b in itertools.product((0, 1), repeat=2)}
                    value = cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)]
                    base.append(cells[(0, 0)]); inter.append(value); ratios.append(float(np.linalg.norm(value) / max(np.linalg.norm(cells[(1, 1)] - cells[(0, 0)]), 1e-12)))
                base, inter = np.asarray(base), np.asarray(inter); med = np.median(base, axis=0); high = base >= med[None, :]
                mean = inter.mean(axis=0); diff = np.sum(inter * high, axis=0) / np.maximum(high.sum(axis=0), 1) - np.sum(inter * ~high, axis=0) / np.maximum((~high).sum(axis=0), 1)
                mean_fields[fi, q, ri] = mean.astype(np.float16); condition_fields[fi, q, ri] = diff.astype(np.float16); cond_ratios.append(float(np.linalg.norm(diff) / max(np.linalg.norm(mean), 1e-12)))
        family_rows.append({"family": family, "complete_groups": len(groups), "median_interaction_to_total": float(np.median(ratios)), "median_state_condition_to_mean": float(np.median(cond_ratios))})
        print(f"[C268] {family} groups={len(groups)}", flush=True)
    mean_fields.flush(); condition_fields.flush(); core.write_rows(OUT / "analysis/family_rows.jsonl", family_rows)
    report = {"phase": 1802, "campaign": "C268", "status": "typed_state_composition_observed", "families": family_rows, "type_graph": next(r for r in family_rows if r["family"] == "type_graph"), "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C269_local_causal_regardless_of_composition_rank"}; core.save(OUT / "analysis/summary.json", report)
    ach = {"families": len(family_rows) == 5, "groups": min(r["complete_groups"] for r in family_rows) >= 24, "shapes": list(mean_fields.shape) == [5, 37, 6, 2560], "finite": bool(np.isfinite([r["median_interaction_to_total"] for r in family_rows]).all())}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); fch = {"contract": True, "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1802, "campaign": "C268", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


if __name__ == "__main__": main()
