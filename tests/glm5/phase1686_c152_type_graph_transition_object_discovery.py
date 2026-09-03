#!/usr/bin/env python3
"""C152: discover a type-graph transition object from existing full-coordinate fields."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1686_c152_type_graph_transition_object_discovery"
C141 = RESULT / "phase1675_c141_multifamily_full_coordinate_atlas"
C143 = RESULT / "phase1677_c143_transition_model_competition"
C151 = RESULT / "phase1685_c151_fresh_transition_window_replication"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1677_c143_transition_model_competition as c143

PHASE, CAMPAIGN = 1686, "C152"
WINDOW = tuple(range(24, 34))
ROLES, DIM = 6, 2560
CANDIDATES = (
    "universal",
    "type_aggregate",
    "conditional_pooled",
    "path_specialized",
    "interference_specialized",
    "exact_structure",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def conditional_trajectories(rows, raw, unit_values, states):
    keys = []
    for unit in unit_values:
        for surface in (1, -1):
            for code in (1, -1):
                for f2 in (1, -1):
                    for f3 in (1, -1):
                        keys.append({"unit": unit, "surface": surface, "code": code, "f2": f2, "f3": f3})
    lookup = {(k["unit"], k["surface"], k["code"], k["f2"], k["f3"]): i for i, k in enumerate(keys)}
    trajectories = np.zeros((len(keys), len(states), ROLES, DIM), np.float32)
    for i, row in enumerate(rows):
        if row["arm"] != "type_graph":
            continue
        unit = int(row["unit_id"].rsplit("-", 1)[1])
        if unit not in unit_values:
            continue
        f = row["factors"]
        key = (unit, row["surface_factor"], row["codebook_factor"], f["f2"], f["f3"])
        decoded = c127.decode(raw[i])
        if decoded.shape[1] == 38:
            decoded = decoded[:, list(states)]
        trajectories[lookup[key]] += float(f["f1"]) * decoded.transpose(1, 0, 2) / 2.0
    return trajectories, keys


def aggregate_type_trajectories(partition):
    data = np.load(C143 / f"analysis/{partition}_primary_trajectories.float32.npy", mmap_mode="r")
    index = core.rows(C143 / f"analysis/{partition}_sample_index.jsonl")
    ids = [i for i, row in enumerate(index) if row["arm"] == "type_graph"]
    return np.asarray(data[ids], np.float32)


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C151 / "audit/independent_closure_audit.json")
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "memo_and_next_stage_assessment",
        "candidates": len(CANDIDATES) == 6,
        "window": WINDOW == tuple(range(24, 34)),
        "c141_raw": (C141 / "raw/qwen3_six_role_field.bf16.npy").is_file(),
        "c151_raw": (C151 / "raw/qwen3_window_role_field.bf16.npy").is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "existing_data_type_graph_object_contract_frozen",
        "object": "f1 HiddenState contrast conditioned on path factor f2 and interference factor f3",
        "candidates": list(CANDIDATES),
        "training": "C141 discovery type-graph units only",
        "historical_panels": ["C141 confirmation", "C151 fresh prospective"],
        "window": list(WINDOW),
        "lambda": 0.01,
        "selection": "maximize the minimum panel median cosine, then minimize maximum panel median relative error",
        "stability": "the selected candidate must independently rank first on both panels and have median cosine >= 0.50 on both",
        "claim_boundary": "retrospective object discovery; no prospective or causal claim",
        "forbidden": ["attention", "MLP", "weights", "PCA", "threshold revision after analysis"],
        "source_hashes": {
            "C141": core.sha(C141 / "raw/qwen3_six_role_field.bf16.npy"),
            "C151": core.sha(C151 / "raw/qwen3_window_role_field.bf16.npy"),
        },
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_C152_existing_fields",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps(protocol, indent=2))


def training_subset(candidate, train_cond, train_keys, train_agg, key):
    if candidate == "universal":
        return None
    if candidate == "type_aggregate":
        return train_agg
    if candidate == "conditional_pooled":
        return train_cond
    if candidate == "path_specialized":
        ids = [i for i, k in enumerate(train_keys) if k["f2"] == key["f2"]]
    elif candidate == "interference_specialized":
        ids = [i for i, k in enumerate(train_keys) if k["f3"] == key["f3"]]
    elif candidate == "exact_structure":
        ids = [i for i, k in enumerate(train_keys) if k["f2"] == key["f2"] and k["f3"] == key["f3"]]
    else:
        raise KeyError(candidate)
    return train_cond[ids]


def predict(candidate, q, x_test, test_keys, train_cond, train_keys, train_agg, train_universal):
    pred = np.zeros_like(x_test)
    groups = {(k["f2"], k["f3"]) for k in test_keys}
    for f2, f3 in groups:
        ids = np.asarray([i for i, k in enumerate(test_keys) if k["f2"] == f2 and k["f3"] == f3])
        key = {"f2": f2, "f3": f3}
        if candidate == "universal":
            source = train_universal
        else:
            source = training_subset(candidate, train_cond, train_keys, train_agg, key)
        x_train = source[:, q].reshape(len(source), -1)
        y_train = (source[:, q + 1] - source[:, q]).reshape(len(source), -1)
        pred[ids] = c143.fit_predict("linear_kernel", x_train, y_train, x_test[ids], 0.01)
    return pred


def evaluate(panel, trajectories, keys, train_cond, train_keys, train_agg, train_universal):
    reports = {}
    state_offset = 0 if trajectories.shape[1] == 38 else 24
    for candidate in CANDIDATES:
        transition_rows = []
        for q in WINDOW:
            j = q - state_offset
            x = trajectories[:, j].reshape(len(trajectories), -1)
            y = (trajectories[:, j + 1] - trajectories[:, j]).reshape(len(trajectories), -1)
            pred = predict(candidate, q, x, keys, train_cond, train_keys, train_agg, train_universal)
            strata = {}
            for f2 in (1, -1):
                for f3 in (1, -1):
                    ids = np.asarray([i for i, k in enumerate(keys) if k["f2"] == f2 and k["f3"] == f3])
                    strata[f"f2={f2},f3={f3}"] = c143.metrics(pred[ids], y[ids])
            transition_rows.append({
                "q": q,
                "target": c143.metrics(pred, y),
                "wrong_role": c143.metrics(np.roll(pred.reshape(len(pred), ROLES, DIM), 1, axis=1).reshape(len(pred), -1), y),
                "wrong_coordinate": c143.metrics(np.roll(pred.reshape(len(pred), ROLES, DIM), 1, axis=2).reshape(len(pred), -1), y),
                "strata": strata,
            })
        reports[candidate] = {
            "panel": panel,
            "median_cosine": float(np.median([r["target"]["cosine"] for r in transition_rows])),
            "median_relative_error": float(np.median([r["target"]["relative_error"] for r in transition_rows])),
            "stratum_median_cosine": {
                s: float(np.median([r["strata"][s]["cosine"] for r in transition_rows]))
                for s in transition_rows[0]["strata"]
            },
            "wrong_role_margin": float(np.median([r["wrong_role"]["relative_error"] - r["target"]["relative_error"] for r in transition_rows])),
            "wrong_coordinate_margin": float(np.median([r["wrong_coordinate"]["relative_error"] - r["target"]["relative_error"] for r in transition_rows])),
            "transition_rows": transition_rows,
        }
    return reports


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows141 = core.rows(C141 / "compiled/qwen3.jsonl")
    raw141 = np.load(C141 / "raw/qwen3_six_role_field.bf16.npy", mmap_mode="r")
    train_cond, train_keys = conditional_trajectories(rows141, raw141, range(4), range(38))
    confirm_cond, confirm_keys = conditional_trajectories(rows141, raw141, range(4, 8), range(38))
    rows151 = core.rows(C151 / "compiled/qwen3.jsonl")
    raw151 = np.load(C151 / "raw/qwen3_window_role_field.bf16.npy", mmap_mode="r")
    fresh_cond, fresh_keys = conditional_trajectories(rows151, raw151, range(4), range(24, 35))
    train_agg = aggregate_type_trajectories("discovery")
    train_universal = np.load(C143 / "analysis/discovery_primary_trajectories.float32.npy", mmap_mode="r")
    panel_reports = {
        "c141_confirmation": evaluate("c141_confirmation", confirm_cond, confirm_keys, train_cond, train_keys, train_agg, train_universal),
        "c151_fresh": evaluate("c151_fresh", fresh_cond, fresh_keys, train_cond, train_keys, train_agg, train_universal),
    }
    def score(candidate):
        reports = [panel_reports[p][candidate] for p in panel_reports]
        return (min(r["median_cosine"] for r in reports), -max(r["median_relative_error"] for r in reports))
    ranking = sorted(CANDIDATES, key=score, reverse=True)
    independent_winners = {
        panel: max(CANDIDATES, key=lambda c: (reports[c]["median_cosine"], -reports[c]["median_relative_error"]))
        for panel, reports in panel_reports.items()
    }
    winner = ranking[0]
    stable = len(set(independent_winners.values())) == 1 and all(panel_reports[p][winner]["median_cosine"] >= 0.50 for p in panel_reports)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "type_graph_transition_object_discovered",
        "panel_reports": panel_reports,
        "ranking": ranking,
        "independent_winners": independent_winners,
        "selected_candidate": winner,
        "stable_candidate": stable,
        "selection_rule": protocol["selection"],
        "claim_boundary": protocol["claim_boundary"],
        "authorization": "freeze_C153_fresh_confirmation" if stable else "continue_type_graph_observation_without_causality",
    }
    core.save(OUT / "analysis/discovery.json", report)
    core.write_rows(OUT / "analysis/train_conditional_index.jsonl", train_keys)
    np.save(OUT / "analysis/train_conditional_trajectories.float32.npy", train_cond)
    checks = {
        "train_shape": list(train_cond.shape) == [64, 38, 6, DIM],
        "confirm_shape": list(confirm_cond.shape) == [64, 38, 6, DIM],
        "fresh_shape": list(fresh_cond.shape) == [64, 11, 6, DIM],
        "candidates": set(ranking) == set(CANDIDATES),
        "panels": set(panel_reports) == {"c141_confirmation", "c151_fresh"},
        "finite": bool(np.isfinite(train_cond).all() and np.isfinite(confirm_cond).all() and np.isfinite(fresh_cond).all()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_candidate_stable": stable, "authorization": report["authorization"]})
    print(json.dumps({
        "ranking": ranking,
        "independent_winners": independent_winners,
        "selected": winner,
        "stable": stable,
        "summary": {p: {c: {k: v for k, v in r.items() if k in ("median_cosine", "median_relative_error", "stratum_median_cosine")} for c, r in rs.items()} for p, rs in panel_reports.items()},
    }, indent=2))


def close():
    report = core.load(OUT / "analysis/discovery.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "boundary": report["claim_boundary"].startswith("retrospective"),
    }
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "type_graph_transition_object_closed",
        "selected_candidate": report["selected_candidate"],
        "stable_candidate": report["stable_candidate"],
        "independent_winners": report["independent_winners"],
        "claim_boundary": report["claim_boundary"],
        "next_authorization": report["authorization"],
    }
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_final_and_memo"})
    print(json.dumps(closure, indent=2))


def main():
    modes = {"contract": contract, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
