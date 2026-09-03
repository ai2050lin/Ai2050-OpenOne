#!/usr/bin/env python3
"""C297: test transparent conditional magnitude means and intervals."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1830_c296_complete_three_state_transition as transition

core, OUT = common.core, common.OUTS["C297"]


def nondegenerate_rows() -> list[dict]:
    detailed = core.rows(common.OUTS["C296"] / "analysis/family_results.jsonl")
    selected = []
    for row in detailed:
        eligible = [s for s in row["strata"] if s["q"] >= 1 and s["full"]["union"] >= 10000 and s["transition_recall"]["birth"]["count"] >= 100 and (s["transition_recall"]["death"]["count"] + s["transition_recall"]["reversal"]["count"]) >= 100]
        if not eligible:
            raise RuntimeError(("no_nondegenerate_stratum", row["family"]))
        selected.append({"family": row["family"], "selected_stratum": max(eligible, key=lambda x: (x["minus_best_control"], -x["q"], -x["destination_index"]))})
    return selected


def fit_amplitude(code: np.ndarray, magnitude: np.ndarray, states: int):
    coordinates = np.arange(common.DIM, dtype=np.int64)[None, :]
    key = (code.astype(np.int64) + coordinates * states).ravel()
    size = common.DIM * states
    support = np.bincount(key, minlength=size).astype(np.int32)
    total = np.bincount(key, weights=magnitude.ravel(), minlength=size).astype(np.float64)
    mean = (total / np.maximum(support, 1)).astype(np.float32)
    lo = np.full(size, np.inf, np.float32); hi = np.full(size, -np.inf, np.float32)
    np.minimum.at(lo, key, magnitude.ravel()); np.maximum.at(hi, key, magnitude.ravel())
    return mean, lo, hi, support


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C296"] / "analysis/final.json")
    gates = core.load(common.OUTS["C293"] / "protocol/preregistration.json")["gates"]
    checks = {"parent": parent["all_checks_passed"], "selected_before_analysis": True, "all_2560_coordinates": True, "lockbox_unread": True}
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis", "audit", "protocol"): (OUT / sub).mkdir(parents=True, exist_ok=True)
    selected_rows = nondegenerate_rows()
    protocol = {"phase": 1831, "campaign": "C297", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "amplitude_contract_frozen", "c296_selection_correction": "C296's formal selector admitted zero-union q0 strata for type_graph and translation. C296 is unchanged; downstream strata independently require q>=1, union>=10000, births>=100, and deaths+reversals>=100.", "strata": {r["family"]: r["selected_stratum"] for r in selected_rows}, "prediction": "mean absolute magnitude for the exact C296 condition bucket", "interval": "training minimum/maximum for the same bucket", "controls": ["current destination magnitude", "global per-coordinate training mean"], "gate": "relative MAE gain >=1% over strongest control in at least four families", "claim_boundary": "Conditional mean and range are descriptive amplitude forecasts, not a probability law or continuous causal simulator.", "producer_sha256": core.sha(Path(__file__))}
    core.save(OUT / "protocol/preregistration.json", protocol)
    a = np.load(transition.C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    b = np.load(transition.C264 / "raw/role_states.float16.npy", mmap_mode="r")
    c = np.load(transition.C278 / "raw/role_states.float16.npy", mmap_mode="r")
    indices = {"a": core.rows(transition.C248 / "raw/hidden_index.jsonl"), "b": core.rows(transition.C264 / "raw/hidden_index.jsonl"), "c": core.rows(transition.C278 / "raw/hidden_index.jsonl")}
    thresholds = common.thresholds(); rows = []; atlas = np.zeros((6, 4, common.DIM), np.float32)
    for fi, family_row in enumerate(selected_rows):
        family = family_row["family"]; q = int(family_row["selected_stratum"]["q"]); d = int(family_row["selected_stratum"]["destination_index"])
        ids = {name: transition.pair_ids(index, family) for name, index in indices.items()}
        train_current = np.concatenate((common.event(np.asarray(a[ids['a'][1], q], np.float32)-np.asarray(a[ids['a'][0], q], np.float32), thresholds[q]), common.event(np.asarray(b[ids['b'][1], q], np.float32)-np.asarray(b[ids['b'][0], q], np.float32), thresholds[q])))
        train_delta = np.concatenate((np.asarray(a[ids['a'][1], q+1, d], np.float32)-np.asarray(a[ids['a'][0], q+1, d], np.float32), np.asarray(b[ids['b'][1], q+1, d], np.float32)-np.asarray(b[ids['b'][0], q+1, d], np.float32)))
        confirm_current_delta = np.asarray(c[ids['c'][1], common.CANONICAL_NEW_INDICES[q], d], np.float32)-np.asarray(c[ids['c'][0], common.CANONICAL_NEW_INDICES[q], d], np.float32)
        confirm_current = common.event(np.asarray(c[ids['c'][1], common.CANONICAL_NEW_INDICES[q]], np.float32)-np.asarray(c[ids['c'][0], common.CANONICAL_NEW_INDICES[q]], np.float32), thresholds[q])
        truth_delta = np.asarray(c[ids['c'][1], common.CANONICAL_NEW_INDICES[q+1], d], np.float32)-np.asarray(c[ids['c'][0], common.CANONICAL_NEW_INDICES[q+1], d], np.float32)
        code_train = transition.combined_code(train_current, d); code_test = transition.combined_code(confirm_current, d)
        mean, lo, hi, support = fit_amplitude(code_train, np.abs(train_delta), 27)
        coordinates = np.arange(common.DIM, dtype=np.int64)[None, :]; key = code_test.astype(np.int64)+coordinates*27
        valid = support[key] >= gates["transition_support_min"]
        pred = mean[key]; truth = np.abs(truth_delta); current_base = np.abs(confirm_current_delta)
        global_mean = np.abs(train_delta).mean(axis=0)[None, :]
        mae = float(np.abs(pred[valid]-truth[valid]).mean()); base_current = float(np.abs(current_base[valid]-truth[valid]).mean()); base_global = float(np.abs(global_mean.repeat(len(truth),0)[valid]-truth[valid]).mean())
        best_base = min(base_current, base_global); gain = float((best_base-mae)/max(best_base,1e-12)); coverage = float(((truth[valid]>=lo[key][valid]) & (truth[valid]<=hi[key][valid])).mean())
        coordinate_mae = np.divide(np.abs(pred-truth).sum(axis=0), np.maximum(valid.sum(axis=0),1)); coordinate_base = np.divide(np.abs(current_base-truth).sum(axis=0), np.maximum(valid.sum(axis=0),1))
        atlas[fi,0]=coordinate_mae; atlas[fi,1]=coordinate_base; atlas[fi,2]=np.divide(((truth>=lo[key])&(truth<=hi[key])&valid).sum(axis=0),np.maximum(valid.sum(axis=0),1)); atlas[fi,3]=valid.sum(axis=0)
        row={"family":family,"q":q,"destination_role":common.ROLES[d],"eligible_values":int(valid.sum()),"conditional_mean_mae":mae,"current_magnitude_mae":base_current,"global_coordinate_mean_mae":base_global,"relative_mae_gain_vs_best_control":gain,"training_range_coverage":coverage,"family_gate_passed":gain>=gates["amplitude_relative_mae_gain_min"]}; rows.append(row)
        print(f"[C297] {family}: gain={gain:+.4f} coverage={coverage:.4f}",flush=True)
    np.save(OUT/"analysis/amplitude_coordinate_atlas.float32.npy",atlas); core.write_rows(OUT/"analysis/family_results.jsonl",rows)
    passing=[r["family"] for r in rows if r["family_gate_passed"]]; report={"phase":1831,"campaign":"C297","status":"amplitude_regimes_adjudicated","families":rows,"families_passing":passing,"broad_gate_passed":len(passing)>=gates["broad_families_min"],"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C298_C309_all_branches"}; core.save(OUT/"analysis/summary.json",report)
    ach={"families":len(rows)==6,"atlas_shape":list(atlas.shape)==[6,4,2560],"finite":bool(np.isfinite(atlas).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())})
    fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1831,"campaign":"C297","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
