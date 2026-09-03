#!/usr/bin/env python3
"""C160: recipient-only prediction of the missing graph counterfactual field."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1694_c160_recipient_only_counterfactual_prediction"
C159 = RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1677_c143_transition_model_competition as c143

PHASE, CAMPAIGN = 1694, "C160"
DIM, ROLES = 2560, 6
LATE = tuple(range(24, 35))
CANDIDATES = ("global_mean", "pooled_diagonal", "pooled_linear", "panel_linear", "relation_linear", "condition_mean")


def now():
    return datetime.now(timezone.utc).isoformat()


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C159 / "audit/independent_final_audit.json")
    pairs = core.rows(C159 / "analysis/late_half_difference_index.jsonl")
    checks = {
        "authorization": parent["all_checks_passed"],
        "behavior_qualified": parent["scientific_behavior_qualified"],
        "pairs": len(pairs) == 768,
        "partitions": all(sum(row["partition"] == part for row in pairs) == 256 for part in ("discovery", "confirmation", "fresh")),
        "candidates": len(CANDIDATES) == 6,
        "no_test_donor": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "recipient_only_prediction_contract_frozen",
        "input": "f1=-1 recipient six-role HiddenState at the same checkpoint plus frozen panel/relation/path/interference/direction labels",
        "target": "paired f1 half-difference field X_q",
        "discovery": "fit all candidates on discovery units",
        "selection": "highest confirmation median sample cosine, then lowest median sample relative error",
        "reveal": "only selected winner and global_mean control are evaluated on fresh units",
        "candidates": list(CANDIDATES),
        "lambda": 0.01,
        "fresh_gates": {"median_cosine_min": 0.20, "median_relative_error_max": 1.10, "each_panel_cosine_min": 0.10, "margin_over_mean_min": 0.05},
        "claim_boundary": "recipient-only effective prediction if passed; not a causal circuit or natural generative mechanism",
        "forbidden": ["attention", "MLP", "weights", "PCA", "test donor input", "post-confirmation candidate change"],
        "source_hashes": {"C159_pairs": core.sha(C159 / "analysis/late_half_difference_index.jsonl"), "C159_field": core.sha(C159 / "analysis/late_half_difference.float16.npy")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_C160_existing_C159_data",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "candidates": CANDIDATES}, indent=2))


def arrays(partition, q_index, pair_rows, raw, targets):
    selected = [row for row in pair_rows if row["partition"] == partition]
    x = np.asarray([c127.decode(raw[row["minus_row"], :, LATE[q_index]]) for row in selected], np.float32).reshape(len(selected), -1)
    y = np.asarray([targets[row["pair_index"], q_index] for row in selected], np.float32).reshape(len(selected), -1)
    return selected, x, y


def grouped_predict(candidate, train_rows, x_train, y_train, test_rows, x_test, lam):
    if candidate == "global_mean":
        return c143.fit_predict("mean", x_train, y_train, x_test)
    if candidate == "pooled_diagonal":
        return c143.fit_predict("diagonal_ridge", x_train, y_train, x_test, lam)
    if candidate == "pooled_linear":
        return c143.fit_predict("linear_kernel", x_train, y_train, x_test, lam)
    if candidate == "panel_linear":
        fields = ("panel",)
        model = "linear_kernel"
    elif candidate == "relation_linear":
        fields = ("panel", "relation_family")
        model = "linear_kernel"
    elif candidate == "condition_mean":
        fields = ("panel", "relation_family", "path", "interference", "direction_form")
        model = "mean"
    else:
        raise KeyError(candidate)
    out = np.zeros_like(x_test, dtype=np.float32)
    for key in sorted({tuple(row[field] for field in fields) for row in test_rows}):
        tr = [i for i, row in enumerate(train_rows) if tuple(row[field] for field in fields) == key]
        te = [i for i, row in enumerate(test_rows) if tuple(row[field] for field in fields) == key]
        if not tr:
            out[te] = y_train.mean(0)
        else:
            out[te] = c143.fit_predict(model, x_train[tr], y_train[tr], x_test[te], lam if model == "linear_kernel" else None)
    return out


def sample_metrics(pred, target):
    dot = np.sum(pred * target, axis=1, dtype=np.float64)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    cos = dot / np.maximum(den, 1e-12)
    rel = np.linalg.norm(pred - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
    return {"median_cosine": float(np.median(cos)), "mean_cosine": float(np.mean(cos)), "median_relative_error": float(np.median(rel)), "mean_relative_error": float(np.mean(rel)), "cosines": cos, "relative_errors": rel}


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    pair_rows = core.rows(C159 / "analysis/late_half_difference_index.jsonl")
    raw = np.load(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    targets = np.load(C159 / "analysis/late_half_difference.float16.npy", mmap_mode="r")
    confirmation = {candidate: [] for candidate in CANDIDATES}
    for qi, q in enumerate(LATE):
        train_rows, x_train, y_train = arrays("discovery", qi, pair_rows, raw, targets)
        test_rows, x_test, y_test = arrays("confirmation", qi, pair_rows, raw, targets)
        for candidate in CANDIDATES:
            pred = grouped_predict(candidate, train_rows, x_train, y_train, test_rows, x_test, protocol["lambda"])
            metric = sample_metrics(pred, y_test)
            confirmation[candidate].append({"q": q, "median_cosine": metric["median_cosine"], "median_relative_error": metric["median_relative_error"]})
    ranking = []
    for candidate, rows in confirmation.items():
        ranking.append({"candidate": candidate, "median_cosine": float(np.median([row["median_cosine"] for row in rows])), "median_relative_error": float(np.median([row["median_relative_error"] for row in rows]))})
    ranking.sort(key=lambda row: (-row["median_cosine"], row["median_relative_error"], row["candidate"]))
    winner = ranking[0]["candidate"]
    core.save(OUT / "protocol/confirmation_selection_lock.json", {"created_at_utc": now(), "ranking": ranking, "selected_candidate": winner, "fresh_unread_at_selection": True, "authorization": "reveal_fresh_selected_only"})

    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    winner_pred = np.lib.format.open_memmap(OUT / "analysis/fresh_selected_predictions.float16.npy", mode="w+", dtype=np.float16, shape=(256, 11, 6, DIM))
    fresh_rows_ref = None
    fresh_results, mean_results = [], []
    for qi, q in enumerate(LATE):
        train_rows, x_train, y_train = arrays("discovery", qi, pair_rows, raw, targets)
        fresh_rows, x_fresh, y_fresh = arrays("fresh", qi, pair_rows, raw, targets)
        fresh_rows_ref = fresh_rows
        pred = grouped_predict(winner, train_rows, x_train, y_train, fresh_rows, x_fresh, protocol["lambda"])
        mean_pred = grouped_predict("global_mean", train_rows, x_train, y_train, fresh_rows, x_fresh, protocol["lambda"])
        winner_pred[:, qi] = pred.reshape(256, 6, DIM).astype(np.float16)
        metric = sample_metrics(pred, y_fresh)
        mean_metric = sample_metrics(mean_pred, y_fresh)
        panel = {}
        for name in ("natural_lexical", "isomorphic_nonce"):
            ids = [i for i, row in enumerate(fresh_rows) if row["panel"] == name]
            panel[name] = sample_metrics(pred[ids], y_fresh[ids])["median_cosine"]
        fresh_results.append({"q": q, "median_cosine": metric["median_cosine"], "median_relative_error": metric["median_relative_error"], "panel_median_cosine": panel})
        mean_results.append({"q": q, "median_cosine": mean_metric["median_cosine"], "median_relative_error": mean_metric["median_relative_error"]})
    winner_pred.flush()
    aggregate = {"median_cosine": float(np.median([row["median_cosine"] for row in fresh_results])), "median_relative_error": float(np.median([row["median_relative_error"] for row in fresh_results])), "panel_median_cosine": {panel: float(np.median([row["panel_median_cosine"][panel] for row in fresh_results])) for panel in ("natural_lexical", "isomorphic_nonce")}}
    mean_aggregate = {"median_cosine": float(np.median([row["median_cosine"] for row in mean_results])), "median_relative_error": float(np.median([row["median_relative_error"] for row in mean_results]))}
    g = protocol["fresh_gates"]
    gates = {"cosine": aggregate["median_cosine"] >= g["median_cosine_min"], "error": aggregate["median_relative_error"] <= g["median_relative_error_max"], "panels": all(value >= g["each_panel_cosine_min"] for value in aggregate["panel_median_cosine"].values()), "mean_margin": aggregate["median_cosine"] - mean_aggregate["median_cosine"] >= g["margin_over_mean_min"]}
    behavior = core.rows(C159 / "raw/qwen3_behavior_index.jsonl")
    correct_lookup = {row["row_index"]: row["correct"] for row in behavior}
    correct_both = [i for i, row in enumerate(fresh_rows_ref) if correct_lookup[row["plus_row"]] and correct_lookup[row["minus_row"]]]
    functional = {"pair_count": len(correct_both)}
    if correct_both:
        qi = len(LATE) - 1
        y = np.asarray([targets[row["pair_index"], qi] for row in fresh_rows_ref], np.float32).reshape(256, -1)
        p = np.asarray(winner_pred[:, qi], np.float32).reshape(256, -1)
        functional.update({key: value for key, value in sample_metrics(p[correct_both], y[correct_both]).items() if key not in ("cosines", "relative_errors")})
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "recipient_only_prediction_adjudicated", "confirmation_ranking": ranking, "selected_candidate": winner, "fresh": {"checkpoint_rows": fresh_results, "aggregate": aggregate, "global_mean_control": mean_aggregate, "gates": gates, "passed": all(gates.values())}, "functional_correct_both_q34": functional, "claim_boundary": protocol["claim_boundary"], "next_authorization": "C161 coordinate transmission observation; C163 may use recipient-only direction only if fresh passed"}
    core.save(OUT / "analysis/prediction.json", report)
    checks = {"selection": winner == core.load(OUT / "protocol/confirmation_selection_lock.json")["selected_candidate"], "fresh_rows": len(fresh_rows_ref) == 256, "prediction_shape": list(winner_pred.shape) == [256, 11, 6, DIM], "finite": bool(np.isfinite(np.asarray(winner_pred)).all()), "ranking": len(ranking) == len(CANDIDATES)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_fresh_passed": all(gates.values()), "authorization": report["next_authorization"]})
    print(json.dumps({"ranking": ranking, "winner": winner, "fresh": report["fresh"], "functional_q34": functional}, indent=2))


def close():
    report = core.load(OUT / "analysis/prediction.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "selected_candidate": report["selected_candidate"], "fresh": report["fresh"], "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_audit_then_C161"})
    print(json.dumps(final, indent=2))


def main():
    modes = {"contract": contract, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
