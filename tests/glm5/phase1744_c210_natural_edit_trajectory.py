#!/usr/bin/env python3
"""C210: predict natural paraphrase-edit trajectories across nine language programs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C210
PHASE, CAMPAIGN = 1744, "C210"
MODELS = ("identity", "global_gain", "role_gain", "coordinate_gain", "role_coordinate_gain")


def paired_rows():
    index = core.rows(common.C198 / "raw/hidden_index.jsonl")
    grouped = {}
    for row in index:
        grouped.setdefault((row["program"], row["unit"]), {})[row["surface"]] = row
    return [(key, value[0], value[1]) for key, value in sorted(grouped.items()) if set(value) == {0, 1}]


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C209 / "audit/independent_final_audit.json")
    pairs = paired_rows()
    checks = {"authorization": parent["all_checks_passed"], "pairs": len(pairs) == 36, "programs": len({key[0] for key, _, _ in pairs}) == 9, "units": {key[1] for key, _, _ in pairs} == {1, 2, 5, 6}}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "natural_edit_trajectory_frozen",
        "source": "C198 baseline role-aligned embedding/q23/q24/q25 states",
        "edit": "surface-1 minus surface-0 for the same program and lexical unit; meaning and answer are intended to stay fixed",
        "pairs": 36,
        "programs": 9,
        "partitions": {"discovery": [1, 2], "confirmation": [5], "fresh": [6]},
        "semantic_eligibility": "both surfaces must be behavior-correct; other pairs remain descriptive execution observations",
        "models": list(MODELS),
        "fit": "zero-intercept gains learned on q24-to-q25 natural edit deltas from discovery pairs only",
        "gates": {"fresh_nrmse_max": 0.75, "fresh_improvement_over_identity_min": 0.05, "fresh_weighted_sign_min": 0.75, "eligible_programs_min": 6},
        "claim_boundary": "paraphrase-edit propagation in controlled English; not an intervention operator, true semantic composition or independently human-rated natural language",
        "forbidden": ["attention", "MLP", "weights", "PCA", "fitting confirmation or fresh"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_C198_natural_edits_then_C211_flagship_ledger",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "models": list(MODELS)}, indent=2))


def fit_models(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
    denom_global = float(np.square(source, dtype=np.float64).sum())
    global_gain = float((source.astype(np.float64) * target.astype(np.float64)).sum() / max(denom_global, 1e-30))
    role_num = (source.astype(np.float64) * target.astype(np.float64)).sum(axis=(0, 2))
    role_den = np.square(source, dtype=np.float64).sum(axis=(0, 2))
    coordinate_num = (source.astype(np.float64) * target.astype(np.float64)).sum(axis=(0, 1))
    coordinate_den = np.square(source, dtype=np.float64).sum(axis=(0, 1))
    rc_num = (source.astype(np.float64) * target.astype(np.float64)).sum(axis=0)
    rc_den = np.square(source, dtype=np.float64).sum(axis=0)
    return {"identity": 1.0, "global_gain": global_gain, "role_gain": (role_num / np.maximum(role_den, 1e-30)).astype(np.float32), "coordinate_gain": (coordinate_num / np.maximum(coordinate_den, 1e-30)).astype(np.float32), "role_coordinate_gain": (rc_num / np.maximum(rc_den, 1e-30)).astype(np.float32)}


def apply_model(source: np.ndarray, name: str, value) -> np.ndarray:
    if name in ("identity", "global_gain"):
        return source * float(value)
    if name == "role_gain":
        return source * value[None, :, None]
    if name == "coordinate_gain":
        return source * value[None, None, :]
    return source * value[None, :, :]


def analyze() -> None:
    baseline = np.load(common.C198 / "raw/natural_baseline_states.float16.npy", mmap_mode="r")
    pairs = paired_rows()
    records = []
    deltas = []
    for pair_i, ((program, unit), left, right) in enumerate(pairs):
        delta = np.asarray(baseline[right["anchor_index"]], np.float32) - np.asarray(baseline[left["anchor_index"]], np.float32)
        deltas.append(delta)
        records.append({"pair_index": pair_i, "program": program, "unit": unit, "partition": left["partition"], "both_behavior_correct": bool(left["behavior_correct"] and right["behavior_correct"]), "case_ids": [left["case_id"], right["case_id"]]})
    deltas = np.stack(deltas)
    eligible = np.asarray([row["both_behavior_correct"] for row in records])
    units = np.asarray([row["unit"] for row in records])
    discovery = eligible & np.isin(units, [1, 2])
    confirmation = eligible & (units == 5)
    fresh = eligible & (units == 6)
    source = deltas[:, 2]
    target = deltas[:, 3]
    models = fit_models(source[discovery], target[discovery])
    (OUT / "analysis/operators").mkdir(parents=True, exist_ok=True)
    table = {}
    for name, value in models.items():
        if isinstance(value, np.ndarray):
            np.save(OUT / f"analysis/operators/{name}.float32.npy", value)
        prediction = apply_model(source, name, value)
        table[name] = {}
        for split, mask in (("discovery", discovery), ("confirmation", confirmation), ("fresh", fresh)):
            table[name][split] = {"support": int(mask.sum()), "nrmse": common.nrmse(prediction[mask], target[mask]), "weighted_sign": common.weighted_sign(prediction[mask], target[mask])}
    best_confirmation = min(MODELS, key=lambda name: table[name]["confirmation"]["nrmse"])
    per_program = {}
    best_prediction = apply_model(source, best_confirmation, models[best_confirmation])
    for program in sorted({row["program"] for row in records}):
        mask = fresh & np.asarray([row["program"] == program for row in records])
        per_program[program] = {"support": int(mask.sum()), "nrmse": common.nrmse(best_prediction[mask], target[mask]) if mask.any() else None, "weighted_sign": common.weighted_sign(best_prediction[mask], target[mask]) if mask.any() else None}
    gates = core.load(OUT / "protocol/preregistration.json")["gates"]
    fresh_best = table[best_confirmation]["fresh"]
    improvement = table["identity"]["fresh"]["nrmse"] - fresh_best["nrmse"]
    eligible_programs = [name for name, row in per_program.items() if row["support"] and row["nrmse"] <= gates["fresh_nrmse_max"] and row["weighted_sign"] >= gates["fresh_weighted_sign_min"]]
    passed = fresh_best["nrmse"] <= gates["fresh_nrmse_max"] and improvement >= gates["fresh_improvement_over_identity_min"] and fresh_best["weighted_sign"] >= gates["fresh_weighted_sign_min"] and len(eligible_programs) >= gates["eligible_programs_min"]
    trajectory = {"embedding_to_q23_sign": common.weighted_sign(deltas[eligible, 0], deltas[eligible, 1]), "q23_to_q24_sign": common.weighted_sign(deltas[eligible, 1], deltas[eligible, 2]), "q24_to_q25_sign": common.weighted_sign(deltas[eligible, 2], deltas[eligible, 3])}
    core.write_rows(OUT / "analysis/pair_index.jsonl", records)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "natural_edit_trajectory_adjudicated", "eligible_pairs": int(eligible.sum()), "descriptive_pairs": int((~eligible).sum()), "trajectory_sign": trajectory, "model_table": table, "selected_on_confirmation": best_confirmation, "fresh_improvement_over_identity": improvement, "per_program_fresh": per_program, "eligible_programs": eligible_programs, "natural_edit_gate_passed": passed, "interpretation": "This is a naturally worded paraphrase edit, not an internal do-intervention. A stable trajectory would support a reusable response pattern but not a semantic operator or causal mechanism.", "next_authorization": "C211_five_flagship_route_eligibility_ledger"}
    core.save(OUT / "analysis/natural_edit_trajectory.json", report)
    checks = {"pairs": len(records) == 36, "models": set(table) == set(MODELS), "fresh_programs": len(per_program) == 9, "finite": bool(np.isfinite([row[key] for model in table.values() for row in model.values() for key in ("nrmse", "weighted_sign")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"eligible_pairs": int(eligible.sum()), "trajectory": trajectory, "selected": best_confirmation, "fresh": fresh_best, "improvement": improvement, "eligible_programs": eligible_programs, "passed": passed, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/natural_edit_trajectory.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()

