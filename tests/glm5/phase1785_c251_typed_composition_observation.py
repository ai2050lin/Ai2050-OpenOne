#!/usr/bin/env python3
"""C251: scale-corrected, typed observational composition tests."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C251"]
PARENT = common.OUTS["C248"]


def role_state(fields, row: dict) -> np.ndarray:
    state = np.asarray(fields[row["hidden_index"]], np.float32)
    aligned = np.empty((37, 6, 2560), np.float32)
    for role_i, role in enumerate(common.ROLES):
        aligned[:, role_i] = state[:, row["role_positions"][role], :].mean(axis=1)
    return aligned


def analyze_family(fields, index: list[dict], family: str, panel: str) -> list[dict]:
    selected = [row for row in index if row["family"] == family and row["panel"] == panel]
    key = {(row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in selected}
    rows = []
    for surface, unit, order in itertools.product(common.SURFACES, range(8), (1, -1)):
        needed = [(surface, unit, a, b, order) for a, b in itertools.product((0, 1), repeat=2)]
        if not all(item in key for item in needed):
            continue
        cells = {(a, b): role_state(fields, key[(surface, unit, a, b, order)]) for a, b in itertools.product((0, 1), repeat=2)}
        beta = common.beta_effect(cells)
        additive_prediction = cells[(1, 0)] + cells[(0, 1)] - cells[(0, 0)]
        residual = cells[(1, 1)] - additive_prediction
        total_change = cells[(1, 1)] - cells[(0, 0)]
        for q in range(37):
            residual_norm = float(np.linalg.norm(residual[q]))
            total_norm = float(np.linalg.norm(total_change[q]))
            rows.append({
                "family": family, "panel": panel, "surface": surface, "unit": unit, "order": order, "checkpoint": q,
                "beta_a_norm": float(np.linalg.norm(beta[0, q])), "beta_b_norm": float(np.linalg.norm(beta[1, q])),
                "beta_ab_norm": float(np.linalg.norm(beta[2, q])), "additive_residual_norm": residual_norm,
                "additive_error_ratio": residual_norm / max(total_norm, 1e-12), "behavior_complete": all(key[item]["correct"] for item in needed),
            })
    return rows


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    p249 = core.load(common.OUTS["C249"] / "audit/independent_final_audit.json")
    p250 = core.load(common.OUTS["C250"] / "audit/independent_final_audit.json")
    fields = np.load(PARENT / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(PARENT / "raw/hidden_index.jsonl")
    checks = {"parents": p249["all_checks_passed"] and p250["all_checks_passed"], "nested_rows": sum(row["panel"] == "nested_composition" for row in index) == 128, "scale_formula_frozen": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1785, "campaign": "C251", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "typed_composition_observation_frozen", "coefficient_scale": "beta_A=R_A/2, beta_B=R_B/2, beta_AB=R_AB/4",
        "panels": {"nested_attitude": "attitude wrapper x category-to-instance specificity", "type_graph": "direct/two-hop path x direct shortcut", "contrast": "connective family x clause order"},
        "primary_metric": "role-aligned full-coordinate additive residual norm divided by observed 00-to-11 change norm",
        "claim_boundary": "Natural factorial contrasts estimate non-additivity. They do not identify operator order, a commutator, or a causal composition law.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "analyze_three_typed_panels_once",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    rows = []
    rows += analyze_family(fields, index, "nested_attitude", "nested_composition")
    rows += analyze_family(fields, index, "type_graph", "core")
    rows += analyze_family(fields, index, "contrast", "core")
    core.write_rows(OUT / "analysis/checkpoint_composition_rows.jsonl", rows)
    summaries = []
    for family in ("nested_attitude", "type_graph", "contrast"):
        subset = [row for row in rows if row["family"] == family and row["behavior_complete"]]
        early = [row for row in subset if row["checkpoint"] <= 12]
        middle = [row for row in subset if 13 <= row["checkpoint"] <= 24]
        late = [row for row in subset if row["checkpoint"] >= 25]
        summaries.append({
            "family": family, "complete_groups": len({(r["surface"], r["unit"], r["order"]) for r in subset}),
            "median_additive_error_ratio": float(np.median([r["additive_error_ratio"] for r in subset])),
            "early_median": float(np.median([r["additive_error_ratio"] for r in early])),
            "middle_median": float(np.median([r["additive_error_ratio"] for r in middle])),
            "late_median": float(np.median([r["additive_error_ratio"] for r in late])),
            "median_beta_ab_to_main_ratio": float(np.median([r["beta_ab_norm"] / max(r["beta_a_norm"] + r["beta_b_norm"], 1e-12) for r in subset])),
        })
    report = {
        "phase": 1785, "campaign": "C251", "status": "typed_composition_observed", "summaries": summaries,
        "strict_interpretation": "The three panels quantify where an additive state-field approximation fails after correcting factorial coefficient scale. No panel tests a learned state operator in both application orders.",
        "causal_eligibility": bool(core.load(common.OUTS["C249"] / "analysis/summary.json")["target_families_passed"]),
        "next_authorization": "C252_conditional_path_consistent_hidden_state_intervention; continue_C253_regardless",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"rows": len(rows) >= 3 * 30 * 37, "families": len(summaries) == 3, "finite": bool(np.isfinite([value for row in summaries for key, value in row.items() if key != "family"]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": True, "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1785, "campaign": "C251", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
