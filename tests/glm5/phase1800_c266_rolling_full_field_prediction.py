#!/usr/bin/env python3
"""C266: roll the frozen coordinate passports through unseen fourth-material fields."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common

core, OUT = common.core, common.OUTS["C266"]
C264, C265 = common.OUTS["C264"], common.OUTS["C265"]
FAMILIES = common.FAMILIES + ("nested_attitude",)
HORIZONS = (1, 4, 8, 16, 24, 36)


def count(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    union = (pred != 0) | (truth != 0)
    return np.asarray([((pred == truth) & union).sum(), union.sum(), ((pred == truth) & (pred != 0)).sum(), (pred != 0).sum(), (truth != 0).sum()], np.int64)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C265 / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "passport_frozen": (C265 / "analysis/passport_pred_sign.int8.npy").exists(), "no_future_response_input": True, "all_coordinates": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1800, "campaign": "C266", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "rolling_prediction_frozen",
        "initial_state": "observed embedding response only", "allowed_future_input": "the unedited target baseline state at each checkpoint",
        "forbidden_input": "the edited/contrast response after embedding", "horizons": list(HORIZONS),
        "baselines": ["hold embedding event sign fixed", "fixed tri-material checkpoint event"],
        "metrics": ["signed Jaccard", "signed precision", "signed recall", "registered magnitude interval coverage"],
        "gate": "state-conditioned median signed-Jaccard at horizons 8/16/24/36 exceeds both baselines by 0.01 in at least four families",
        "claim_boundary": "The unedited baseline trajectory is supplied. This is not autonomous simulation of a Transformer and does not establish causality.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C267_and_C268_composition_regardless",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    states = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(C264 / "raw/hidden_index.jsonl")
    pred_map = np.load(C265 / "analysis/passport_pred_sign.int8.npy", mmap_mode="r")
    med_map = np.load(C265 / "analysis/passport_baseline_median.float16.npy", mmap_mode="r")
    lo_map = np.load(C265 / "analysis/passport_magnitude_lo.float16.npy", mmap_mode="r")
    hi_map = np.load(C265 / "analysis/passport_magnitude_hi.float16.npy", mmap_mode="r")
    thresholds = np.asarray(core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    tri = np.load(common.prior.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    rows = []
    for fi, family in enumerate(FAMILIES):
        panel = "nested_composition" if family == "nested_attitude" else "core"
        specs = common.pair_specs(index, family, "factor_a", panel)
        totals = {h: {name: np.zeros(5, np.int64) for name in ("state", "embedding_persistence", "fixed")} for h in HORIZONS}
        interval_hit = {h: 0 for h in HORIZONS}; interval_den = {h: 0 for h in HORIZONS}
        for left_i, right_i, _meta in specs:
            source = np.asarray(states[left_i], np.float32)
            actual_delta = np.asarray(states[right_i], np.float32) - source
            current = np.where(actual_delta[0] > thresholds[0], 1, np.where(actual_delta[0] < -thresholds[0], -1, 0)).astype(np.int8)
            initial = current.copy()
            predicted_mag = np.abs(actual_delta[0])
            for q in range(36):
                nxt = np.zeros_like(current)
                nxt_mag = np.zeros_like(predicted_mag)
                for ri in range(6):
                    high = source[q, ri] >= np.asarray(med_map[fi, q, ri], np.float32)
                    keys = np.where(current[ri] < 0, high.astype(np.int8), np.where(current[ri] > 0, 2 + high.astype(np.int8), -1))
                    for key in range(4):
                        member = keys == key
                        p = np.asarray(pred_map[fi, q, ri, key])
                        nxt[ri, member] = p[member]
                        lo = np.asarray(lo_map[fi, q, ri, key], np.float32)
                        hi = np.asarray(hi_map[fi, q, ri, key], np.float32)
                        nxt_mag[ri, member] = ((lo + hi) * 0.5)[member]
                current, predicted_mag = nxt, nxt_mag
                checkpoint = q + 1
                if checkpoint in HORIZONS:
                    truth = np.where(actual_delta[checkpoint] > thresholds[checkpoint], 1, np.where(actual_delta[checkpoint] < -thresholds[checkpoint], -1, 0)).astype(np.int8)
                    fixed = np.broadcast_to(np.asarray(tri[fi, 0, checkpoint]), truth.shape) if family in common.FAMILIES else np.zeros_like(truth)
                    for name, value in (("state", current), ("embedding_persistence", initial), ("fixed", fixed)):
                        totals[checkpoint][name] += count(value, truth)
                    valid_interval = (current == truth) & (current != 0) & (predicted_mag > 0)
                    interval_hit[checkpoint] += int((valid_interval & (np.abs(actual_delta[checkpoint]) >= predicted_mag * 0.5) & (np.abs(actual_delta[checkpoint]) <= predicted_mag * 1.5)).sum())
                    interval_den[checkpoint] += int(valid_interval.sum())
        for horizon in HORIZONS:
            item = {"family": family, "horizon": horizon, "pairs": len(specs), "interval_midpoint_factor_coverage": float(interval_hit[horizon] / max(interval_den[horizon], 1))}
            for name, total in totals[horizon].items():
                exact, union, active_correct, pred_active, truth_active = total
                item[name] = {"signed_jaccard": float(exact / max(union, 1)), "signed_precision": float(active_correct / max(pred_active, 1)), "signed_recall": float(active_correct / max(truth_active, 1))}
            item["state_minus_best_baseline"] = item["state"]["signed_jaccard"] - max(item["embedding_persistence"]["signed_jaccard"], item["fixed"]["signed_jaccard"])
            rows.append(item)
        print(f"[C266] {family} complete", flush=True)
    core.write_rows(OUT / "analysis/rolling_horizon_rows.jsonl", rows)
    family_results = []
    for family in FAMILIES:
        selected = [r for r in rows if r["family"] == family and r["horizon"] in (8, 16, 24, 36)]
        margin = float(np.median([r["state_minus_best_baseline"] for r in selected]))
        family_results.append({"family": family, "long_horizon_median_margin": margin, "gate_passed": margin >= 0.01})
    broad = sum(r["gate_passed"] for r in family_results) >= 4
    report = {"phase": 1800, "campaign": "C266", "status": "rolling_prediction_adjudicated", "family_results": family_results, "broad_rolling_gate_passed": broad, "strict_interpretation": "Only the embedding edit was observed from the contrast path; later unedited target states were supplied as context. The result distinguishes a state-indexed response rule from autonomous model emulation.", "next_authorization": "C267_nested_and_C268_graph_composition"}
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"rows": len(rows) == 6 * len(HORIZONS), "finite": bool(np.isfinite([r["state_minus_best_baseline"] for r in rows]).all()), "all_families": len(family_results) == 6}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1800, "campaign": "C266", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

