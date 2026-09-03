#!/usr/bin/env python3
"""C143: held-out residual-increment transition-model competition."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1677_c143_transition_model_competition"
C140 = RESULT / "phase1674_c140_identifiability_and_master_contract"
C141 = RESULT / "phase1675_c141_multifamily_full_coordinate_atlas"
C142 = RESULT / "phase1676_c142_mobius_output_code_separation"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141

PHASE, CAMPAIGN = 1677, "C143"
ARMS, ROLES, DIM = c141.ARMS, c141.ROLES, 2560
MODELS = ("zero", "mean", "scalar", "diagonal_ridge", "role_mixing", "linear_kernel", "quadratic_kernel")
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
CALIBRATION_TRANSITIONS = (0, 7, 15, 23, 31, 35, 36)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den <= 1e-12 else float(np.dot(a.ravel(), b.ravel()) / den)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    norm = max(float(np.linalg.norm(target)), 1e-12)
    mask = np.abs(target) > float(np.median(np.abs(target)))
    sign = float(np.mean(np.sign(pred[mask]) == np.sign(target[mask]))) if np.any(mask) else 0.0
    return {
        "cosine": cosine(pred, target),
        "relative_error": float(np.linalg.norm(pred - target) / norm),
        "mean_absolute_error": float(np.mean(np.abs(pred - target))),
        "sign_accuracy_above_median_target": sign,
    }


def sample_trajectories(partition: str) -> tuple[np.ndarray, list[dict]]:
    rows = core.rows(C141 / "compiled/qwen3.jsonl")
    raw = np.load(C141 / "raw/qwen3_six_role_field.bf16.npy", mmap_mode="r")
    keys = []
    for arm in ARMS:
        units = range(4) if partition == "discovery" else range(4, 8)
        for unit in units:
            for surface in (1, -1):
                for code in (1, -1):
                    keys.append({"arm": arm, "unit": unit, "surface": surface, "code": code})
    lookup = {(row["arm"], row["unit"], row["surface"], row["code"]): i for i, row in enumerate(keys)}
    trajectories = np.zeros((80, 38, 6, DIM), np.float32)
    for i, row in enumerate(rows):
        if row["partition"] != partition:
            continue
        unit = int(row["unit_id"].rsplit("-", 1)[1])
        key = (row["arm"], unit, row["surface_factor"], row["codebook_factor"])
        trajectories[lookup[key]] += float(row["factors"]["f1"]) * c127.decode(raw[i]).transpose(1, 0, 2) / 8.0
    return trajectories, keys


def fit_predict(name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, lam: float | None = None) -> np.ndarray:
    n, d = x_train.shape
    if name == "zero":
        return np.zeros((len(x_test), d), np.float32)
    if name == "mean":
        return np.repeat(y_train.mean(0, keepdims=True), len(x_test), axis=0).astype(np.float32)
    if name == "scalar":
        beta = float(np.sum(x_train * y_train, dtype=np.float64) / max(np.sum(x_train * x_train, dtype=np.float64), 1e-12))
        return (beta * x_test).astype(np.float32)
    if name == "diagonal_ridge":
        den = np.sum(x_train * x_train, axis=0, dtype=np.float64)
        ridge = float(lam) * max(float(np.mean(den)), 1e-12)
        beta = np.sum(x_train * y_train, axis=0, dtype=np.float64) / (den + ridge)
        return (x_test * beta[None]).astype(np.float32)
    if name == "role_mixing":
        xr = x_train.reshape(n, 6, DIM).transpose(0, 2, 1).reshape(-1, 6).astype(np.float64)
        yr = y_train.reshape(n, 6, DIM).transpose(0, 2, 1).reshape(-1, 6).astype(np.float64)
        gram = xr.T @ xr
        ridge = 1e-5 * max(float(np.trace(gram) / 6), 1e-12)
        weight = np.linalg.solve(gram + ridge * np.eye(6), xr.T @ yr)
        xt = x_test.reshape(len(x_test), 6, DIM).transpose(0, 2, 1).reshape(-1, 6)
        return (xt @ weight).reshape(len(x_test), DIM, 6).transpose(0, 2, 1).reshape(len(x_test), d).astype(np.float32)
    x_mean = x_train.mean(0, keepdims=True)
    y_mean = y_train.mean(0, keepdims=True)
    xc = x_train - x_mean
    scale = max(float(np.sqrt(np.mean(np.square(xc), dtype=np.float64))), 1e-8)
    xn = xc / scale
    xt = (x_test - x_mean) / scale
    kernel = (xn @ xn.T) / d
    kernel_test = (xt @ xn.T) / d
    if name == "quadratic_kernel":
        kernel = kernel + np.square(kernel)
        kernel_test = kernel_test + np.square(kernel_test)
    elif name != "linear_kernel":
        raise KeyError(name)
    alpha = np.linalg.solve(kernel + float(lam) * np.eye(n, dtype=np.float32), y_train - y_mean)
    return (y_mean + kernel_test @ alpha).astype(np.float32)


def xy(trajectories: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    x = trajectories[:, q].reshape(len(trajectories), -1)
    y = (trajectories[:, q + 1] - trajectories[:, q]).reshape(len(trajectories), -1)
    return x, y


def discover() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C142 / "audit/independent_closure_audit.json")
    if not parent["all_checks_passed"] or parent["authorization"] != "start_C143":
        raise RuntimeError(parent)
    OUT.mkdir(parents=True)
    (OUT / "analysis").mkdir(); (OUT / "protocol").mkdir(); (OUT / "audit").mkdir()
    trajectories, keys = sample_trajectories("discovery")
    np.save(OUT / "analysis/discovery_primary_trajectories.float32.npy", trajectories)
    core.write_rows(OUT / "analysis/discovery_sample_index.jsonl", keys)
    train = np.asarray([i for i, key in enumerate(keys) if key["unit"] in (0, 1)])
    validation = np.asarray([i for i, key in enumerate(keys) if key["unit"] in (2, 3)])
    selected_lambda = {name: None for name in MODELS}
    lambda_scores = {}
    for name in ("diagonal_ridge", "linear_kernel", "quadratic_kernel"):
        scores = {}
        for lam in LAMBDAS:
            errors = []
            for q in CALIBRATION_TRANSITIONS:
                x, y = xy(trajectories, q)
                pred = fit_predict(name, x[train], y[train], x[validation], lam)
                errors.append(metrics(pred, y[validation])["relative_error"])
            scores[str(lam)] = float(np.mean(errors))
        best = min(LAMBDAS, key=lambda value: scores[str(value)])
        selected_lambda[name] = best
        lambda_scores[name] = scores
    model_rows = {name: [] for name in MODELS}
    for q in range(37):
        x, y = xy(trajectories, q)
        for name in MODELS:
            pred = fit_predict(name, x[train], y[train], x[validation], selected_lambda[name])
            model_rows[name].append({"transition_index": q, **metrics(pred, y[validation])})
    summary = {name: {"median_cosine": float(np.median([row["cosine"] for row in rows])), "median_relative_error": float(np.median([row["relative_error"] for row in rows])), "mean_relative_error": float(np.mean([row["relative_error"] for row in rows]))} for name, rows in model_rows.items()}
    winner = min(MODELS, key=lambda name: summary[name]["median_relative_error"])
    freeze = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "transition_model_frozen",
        "target": "f1 semantic response residual increment delta_q=D_(q+1)-D_q",
        "models": list(MODELS),
        "lambda_candidates": list(LAMBDAS),
        "calibration_transitions": list(CALIBRATION_TRANSITIONS),
        "selected_lambda": selected_lambda,
        "inner_discovery_summary": summary,
        "inner_discovery_rows": model_rows,
        "frozen_winner": winner,
        "confirmation_gate": core.load(C140 / "protocol/preregistration.json")["C143_gate"],
        "confirmation_unread": True,
        "source_paths": {"C141_role": str(C141 / "raw/qwen3_six_role_field.bf16.npy")},
        "source_hashes": {"C141_role": core.sha(C141 / "raw/qwen3_six_role_field.bf16.npy")},
        "claim_boundary": "effective full-role activation prediction, not a unique physical circuit",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "validate_C143_confirmation",
    }
    core.save(OUT / "protocol/frozen_model.json", freeze)
    checks = {"shape": list(trajectories.shape) == [80, 38, 6, DIM], "split": len(train) == len(validation) == 40, "models": len(summary) == 7, "finite": all(np.isfinite(row["median_relative_error"]) for row in summary.values()), "source": freeze["source_hashes"]["C141_role"] == core.load(C141 / "analysis/authoritative_run.json")["capture"]["role_sha256"]}
    core.save(OUT / "audit/internal_discovery_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps({"checks": checks, "selected_lambda": selected_lambda, "summary": summary, "winner": winner}, indent=2))


def validate() -> None:
    freeze = core.load(OUT / "protocol/frozen_model.json")
    discovery = np.load(OUT / "analysis/discovery_primary_trajectories.float32.npy", mmap_mode="r")
    confirmation, keys = sample_trajectories("confirmation")
    np.save(OUT / "analysis/confirmation_primary_trajectories.float32.npy", confirmation)
    core.write_rows(OUT / "analysis/confirmation_sample_index.jsonl", keys)
    model_rows = {name: [] for name in MODELS}
    for q in range(37):
        xd, yd = xy(discovery, q)
        xc, yc = xy(confirmation, q)
        for name in MODELS:
            pred = fit_predict(name, xd, yd, xc, freeze["selected_lambda"][name])
            model_rows[name].append({"transition_index": q, "from_checkpoint": c127.CHECKPOINTS[q], "to_checkpoint": c127.CHECKPOINTS[q + 1], **metrics(pred, yc)})
    summary = {name: {"median_cosine": float(np.median([row["cosine"] for row in rows])), "median_relative_error": float(np.median([row["relative_error"] for row in rows])), "mean_relative_error": float(np.mean([row["relative_error"] for row in rows])), "median_sign_accuracy": float(np.median([row["sign_accuracy_above_median_target"] for row in rows]))} for name, rows in model_rows.items()}
    winner = freeze["frozen_winner"]
    candidate_rows, wrong_role, wrong_coordinate, wrong_checkpoint, rollout_rows = [], [], [], [], []
    previous_pred = None
    current = np.asarray(confirmation[:, 0]).reshape(80, -1).copy()
    for q in range(37):
        xd, yd = xy(discovery, q)
        xc, yc = xy(confirmation, q)
        pred = fit_predict(winner, xd, yd, xc, freeze["selected_lambda"][winner])
        candidate_rows.append(metrics(pred, yc))
        wrong_role.append(metrics(np.roll(pred.reshape(80, 6, DIM), 1, axis=1).reshape(80, -1), yc))
        wrong_coordinate.append(metrics(np.roll(pred.reshape(80, 6, DIM), 1, axis=2).reshape(80, -1), yc))
        if previous_pred is not None:
            wrong_checkpoint.append(metrics(previous_pred, yc))
        previous_pred = pred
        rollout_delta = fit_predict(winner, xd, yd, current, freeze["selected_lambda"][winner])
        current = current + rollout_delta
        rollout_target = np.asarray(confirmation[:, q + 1]).reshape(80, -1)
        rollout_rows.append(metrics(current, rollout_target))
    median = lambda rows, key: float(np.median([row[key] for row in rows]))
    target_error = median(candidate_rows, "relative_error")
    zero_error = summary["zero"]["median_relative_error"]
    role_error = median(wrong_role, "relative_error")
    coordinate_error = median(wrong_coordinate, "relative_error")
    checkpoint_error = median(wrong_checkpoint, "relative_error")
    rollout_error = median(rollout_rows, "relative_error")
    zero_rollout = float(np.median([np.linalg.norm(confirmation[:, 0].reshape(80, -1) - confirmation[:, q + 1].reshape(80, -1)) / max(np.linalg.norm(confirmation[:, q + 1]), 1e-12) for q in range(37)]))
    derived = {
        "relative_error_ratio_vs_zero": target_error / max(zero_error, 1e-12),
        "wrong_role_error_margin": role_error - target_error,
        "wrong_coordinate_error_margin": coordinate_error - target_error,
        "wrong_checkpoint_error_margin": checkpoint_error - target_error,
        "rollout_relative_error_ratio_vs_zero": rollout_error / max(zero_rollout, 1e-12),
        "candidate_median_cosine": median(candidate_rows, "cosine"),
        "candidate_median_relative_error": target_error,
        "rollout_median_relative_error": rollout_error,
    }
    gate = freeze["confirmation_gate"]
    gates = {
        "error_vs_zero": derived["relative_error_ratio_vs_zero"] <= gate["confirmation_relative_error_ratio_vs_zero_max"],
        "cosine": derived["candidate_median_cosine"] >= gate["confirmation_cosine_min"],
        "wrong_role": derived["wrong_role_error_margin"] >= gate["wrong_role_error_margin_min"],
        "wrong_coordinate": derived["wrong_coordinate_error_margin"] >= gate["wrong_coordinate_error_margin_min"],
        "wrong_checkpoint": derived["wrong_checkpoint_error_margin"] >= gate["wrong_role_error_margin_min"],
        "rollout": derived["rollout_relative_error_ratio_vs_zero"] <= gate["rollout_relative_error_ratio_vs_zero_max"],
    }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "confirmation_adjudicated",
        "frozen_winner": winner,
        "model_summary": summary,
        "model_rows": model_rows,
        "candidate_controls": {"target": candidate_rows, "wrong_role": wrong_role, "wrong_coordinate": wrong_coordinate, "wrong_checkpoint": wrong_checkpoint, "rollout": rollout_rows},
        "derived": derived,
        "gates": gates,
        "prediction_gate_passed": all(gates.values()),
        "claim_boundary": "held-out effective prediction of f1 role-state response increments; no unique causal edge",
        "authorization": "close_C143_continue_C144",
    }
    core.save(OUT / "analysis/confirmation.json", report)
    checks = {"shape": list(confirmation.shape) == [80, 38, 6, DIM], "models": len(summary) == 7, "rows": all(len(rows) == 37 for rows in model_rows.values()), "finite": all(np.isfinite(value) for value in derived.values()), "controls": len(wrong_checkpoint) == 36}
    core.save(OUT / "audit/internal_confirmation_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_prediction_gate_passed": all(gates.values()), "authorization": report["authorization"]})
    print(json.dumps({"frozen_winner": winner, "model_summary": summary, "derived": derived, "gates": gates, "prediction_gate_passed": all(gates.values())}, indent=2))
    del confirmation
    gc.collect()


def close() -> None:
    result = core.load(OUT / "analysis/confirmation.json")
    checks = {"discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"], "frozen": result["frozen_winner"] == core.load(OUT / "protocol/frozen_model.json")["frozen_winner"]}
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "status": "transition_competition_closed", "headline": {"winner": result["frozen_winner"], "prediction_gate_passed": result["prediction_gate_passed"], **result["derived"]}, "theory_update": "directly tests whether a bounded effective transition family predicts held-out residual increments and rolled trajectories", "claim_boundary": result["claim_boundary"], "next_authorization": "C144 and C145 continue regardless; C148 causal flag retained only if prediction gate passed"}
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_prediction_gate_passed": result["prediction_gate_passed"], "authorization": "independent_final_then_C144"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"discover": discover, "validate": validate, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("discover|validate|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
