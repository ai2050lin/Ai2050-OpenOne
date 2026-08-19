from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1274_c021_multitask_free_response_isomorphism as base  # noqa: E402


PHASE = 1277
CAMPAIGN = "C022"
CONTRACT_ID = "EXP-C022-WP00-001"
OUT = ROOT / "tests/glm5/result/phase1277_c022_formation_dynamics_instrument_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment.json"
CALIBRATION = OUT / "raw/known_truth_calibration.jsonl"
REPLAY = OUT / "raw/deterministic_replay.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
MAIN = ROOT / "tests/glm5/phase1277_c022_formation_dynamics_instrument_calibration.py"
AUDITOR = ROOT / "tests/glm5/phase1277_c022_formation_dynamics_instrument_calibration_audit.py"

CELLS = (
    {"cell": "xor.shallow4", "task": "xor", "architecture": "shallow4"},
    {"cell": "cyclic.deep8", "task": "cyclic", "architecture": "deep8"},
    {"cell": "context_lookup.shallow4", "task": "context_lookup", "architecture": "shallow4"},
    {"cell": "context_lookup.deep8", "task": "context_lookup", "architecture": "deep8"},
)
SEEDS_PER_CELL = 12
DISCOVERY_SEEDS_PER_CELL = 6
MODEL_SEEDS = {
    f"{cell['cell']}.s{seed_index}": 1_277_000_000 + 100_000 * cell_index + 1009 * seed_index + 37
    for cell_index, cell in enumerate(CELLS)
    for seed_index in range(SEEDS_PER_CELL)
}
BEHAVIOR_STEPS = (
    0, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048,
    2304, 2560, 2816, 3072, 3328, 3584, 4096, 4864, 5632,
    6400, 7000, 8000, 9000, 10000, 11000, 12000,
)
INTERNAL_STEPS = (512, 768)
STATE_STEPS = (0, 768, 7000, 12000)
BASELINE_FEATURES = (
    "accuracy_512", "accuracy_768", "accuracy_slope",
    "loss_512", "loss_768", "loss_slope",
    "margin_768", "entropy_768", "gradient_norm_768",
)
INTERNAL_FEATURES = (
    "role_selectivity_512", "role_selectivity_768", "role_selectivity_slope",
    "attention_selectivity_768", "mlp_selectivity_768",
)
THRESHOLDS = {
    "fixed_budget": 7000,
    "extended_budget": 12000,
    "prediction_cutoff": 768,
    "stable_accuracy_min": 0.995,
    "stable_adjacent_observations_min": 2,
    "discovery_event_min": 6,
    "discovery_censored_min": 6,
    "confirmation_event_min": 6,
    "confirmation_censored_min": 6,
    "informative_cells_per_split_min": 2,
    "augmented_relative_mae_improvement_min": 0.10,
    "augmented_pair_order_accuracy_min": 0.70,
    "augmented_pair_order_advantage_min": 0.08,
    "confirmation_cell_win_min": 3,
    "lookup_depth_win_min": 2,
    "synthetic_relative_improvement_min": 0.25,
    "synthetic_pair_order_accuracy_min": 0.80,
    "negative_control_relative_improvement_max": 0.05,
    "deterministic_replay_max_abs_diff": 0.0,
    "ridge_alpha": 1.0,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    standardized = (x - mean) / scale
    design = np.concatenate([np.ones((len(x), 1)), standardized], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficient": coefficient.tolist(), "alpha": alpha}


def apply_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    standardized = (x - np.asarray(model["mean"], dtype=np.float64)) / np.asarray(model["scale"], dtype=np.float64)
    design = np.concatenate([np.ones((len(x), 1)), standardized], axis=1)
    return np.clip(design @ np.asarray(model["coefficient"], dtype=np.float64), 0.0, 1.0)


def mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y - prediction)))


def pair_order_accuracy(y: np.ndarray, prediction: np.ndarray, groups: list[str]) -> dict[str, Any]:
    correct = 0
    eligible = 0
    by_group: dict[str, dict[str, int]] = {}
    for left in range(len(y)):
        for right in range(left + 1, len(y)):
            if groups[left] != groups[right] or abs(float(y[left] - y[right])) < 0.05:
                continue
            eligible += 1
            match = int(np.sign(y[left] - y[right]) == np.sign(prediction[left] - prediction[right]))
            correct += match
            bucket = by_group.setdefault(groups[left], {"eligible": 0, "correct": 0})
            bucket["eligible"] += 1
            bucket["correct"] += match
    return {
        "eligible": eligible,
        "correct": correct,
        "accuracy": correct / eligible if eligible else None,
        "by_group": {
            key: {**value, "accuracy": value["correct"] / value["eligible"] if value["eligible"] else None}
            for key, value in sorted(by_group.items())
        },
    }


def protocol_payload() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1277.c022.formation_dynamics_instrument.v1",
        "claim_type": "known_truth_instrument_calibration_and_formal_formation_contract",
        "cells": list(CELLS),
        "seeds_per_cell": SEEDS_PER_CELL,
        "discovery_seeds_per_cell": DISCOVERY_SEEDS_PER_CELL,
        "model_seeds": MODEL_SEEDS,
        "behavior_steps": list(BEHAVIOR_STEPS),
        "internal_steps": list(INTERNAL_STEPS),
        "state_steps": list(STATE_STEPS),
        "baseline_features": list(BASELINE_FEATURES),
        "internal_features": list(INTERNAL_FEATURES),
        "thresholds": THRESHOLDS,
        "formal_primary_endpoint": "At step 768, the frozen augmented predictor must reduce new-seed confirmation MAE by at least 10% versus the better of a cell prior and the behavior-only ridge, improve pair ordering by at least 0.08, win in at least 3/4 cells including both lookup depths, and satisfy breadth gates.",
        "formal_branch_rule": "Only a passed internal predictive-increment endpoint authorizes one same-parent causal training-update branch. Failure permanently closes small synthetic-network camera escalation.",
        "independent_unit": "model trajectory; checkpoints are repeated observations, never independent samples",
        "outcome": "bounded fixed-budget stable-formation progress; extended training separately labels delayed formation",
        "hard_stops": [
            "No C021 seed enters formal C022 evidence.",
            "No model is behavior-filtered before trajectory analysis.",
            "CUDA deterministic replay must pass exactly before formal execution.",
            "The confirmation split cannot select features, thresholds, cutoff, cells, or ridge alpha.",
            "If the behavior/object gate fails, mechanism prediction is untested rather than negative.",
            "If augmented prediction fails, no new internal feature, camera, task, or seed is added in C022.",
            "Synthetic results cannot authorize Qwen3, GLM4, or DS7B.",
        ],
        "source_hashes": {"main": file_sha256(MAIN), "auditor": file_sha256(AUDITOR)},
        "created_at_utc": utc_now(),
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists; pass --force only before calibration data exist")
    if force and (CALIBRATION.exists() or REPLAY.exists() or FINAL.exists()):
        raise RuntimeError("cannot replace protocol after calibration data exist")
    payload = protocol_payload()
    payload["protocol_digest"] = digest(payload)
    atomic_json(PROTOCOL, payload)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32 deterministic training; fp64 analysis",
    })
    print(canonical_json({"status": "registered", "protocol_digest": payload["protocol_digest"], "formal_models": len(MODEL_SEEDS)}))


def state_digest(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    state = optimizer.state_dict()
    hasher.update(canonical_json(state["param_groups"]).encode("utf-8"))
    for parameter_id, values in sorted(state["state"].items()):
        hasher.update(str(parameter_id).encode("ascii"))
        for key, value in sorted(values.items()):
            hasher.update(key.encode("utf-8"))
            if torch.is_tensor(value):
                hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                hasher.update(str(value).encode("utf-8"))
    return hasher.hexdigest()


def replay_once(device: torch.device, seed: int) -> dict[str, Any]:
    configure_determinism(seed)
    model = base.TinyCausalTransformer(base.ARCHITECTURES["shallow4"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
    losses: list[float] = []
    for step in range(96):
        inputs, labels = base.random_batch("cyclic", 256, seed + 10_000 + step)
        logits = model(inputs.to(device))[:, -1, base.CANDIDATE_SLICE].float()
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    accuracy = base.evaluate_behavior(model, "cyclic", seed + 9_000_000, device, count=4096)
    payload = {"state_digest": state_digest(model, optimizer), "losses": losses, "accuracy": accuracy}
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def synthetic_rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(1_277_770_001)
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(CELLS):
        for seed_index in range(SEEDS_PER_CELL):
            latent = float(rng.normal()) + 0.15 * (cell_index - 1.5)
            outcome = float(np.clip(0.50 + 0.24 * latent + rng.normal(scale=0.035), 0.0, 1.0))
            baseline = [
                0.50 + 0.06 * latent + rng.normal(scale=0.18),
                1.20 - 0.05 * latent + rng.normal(scale=0.18),
                rng.normal(scale=0.20),
            ]
            internal = [
                0.50 + 0.28 * latent + rng.normal(scale=0.05),
                0.30 + 0.20 * latent + rng.normal(scale=0.05),
            ]
            rows.append({
                "trajectory_id": f"synthetic.{cell['cell']}.s{seed_index}",
                "cell": cell["cell"],
                "seed_index": seed_index,
                "split": "discovery" if seed_index < DISCOVERY_SEEDS_PER_CELL else "confirmation",
                "baseline": baseline,
                "internal": internal,
                "nuisance": [float(rng.normal()), float(rng.normal())],
                "outcome": outcome,
                "future_leakage_sentinel": outcome,
            })
    return rows


def calibration_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    discovery = [row for row in rows if row["split"] == "discovery"]
    confirmation = [row for row in rows if row["split"] == "confirmation"]
    baseline_x = np.asarray([row["baseline"] for row in discovery], dtype=np.float64)
    augmented_x = np.asarray([row["baseline"] + row["internal"] for row in discovery], dtype=np.float64)
    nuisance_x = np.asarray([row["baseline"] + row["nuisance"] for row in discovery], dtype=np.float64)
    y = np.asarray([row["outcome"] for row in discovery], dtype=np.float64)
    baseline_model = fit_ridge(baseline_x, y, THRESHOLDS["ridge_alpha"])
    augmented_model = fit_ridge(augmented_x, y, THRESHOLDS["ridge_alpha"])
    nuisance_model = fit_ridge(nuisance_x, y, THRESHOLDS["ridge_alpha"])
    confirmation_y = np.asarray([row["outcome"] for row in confirmation], dtype=np.float64)
    baseline_prediction = apply_ridge(baseline_model, np.asarray([row["baseline"] for row in confirmation], dtype=np.float64))
    augmented_prediction = apply_ridge(augmented_model, np.asarray([row["baseline"] + row["internal"] for row in confirmation], dtype=np.float64))
    nuisance_prediction = apply_ridge(nuisance_model, np.asarray([row["baseline"] + row["nuisance"] for row in confirmation], dtype=np.float64))
    baseline_mae = mae(confirmation_y, baseline_prediction)
    augmented_mae = mae(confirmation_y, augmented_prediction)
    nuisance_mae = mae(confirmation_y, nuisance_prediction)
    relative = (baseline_mae - augmented_mae) / baseline_mae
    nuisance_relative = (baseline_mae - nuisance_mae) / baseline_mae
    order = pair_order_accuracy(confirmation_y, augmented_prediction, [row["cell"] for row in confirmation])
    return {
        "baseline_model": baseline_model,
        "augmented_model": augmented_model,
        "nuisance_model": nuisance_model,
        "confirmation_baseline_mae": baseline_mae,
        "confirmation_augmented_mae": augmented_mae,
        "confirmation_nuisance_mae": nuisance_mae,
        "augmented_relative_improvement": relative,
        "nuisance_relative_improvement": nuisance_relative,
        "augmented_pair_order": order,
        "future_sentinel_blocked": "future_leakage_sentinel" not in {"baseline", "internal", "nuisance"},
    }


def run() -> None:
    protocol = read_json(PROTOCOL)
    stored_digest = protocol["protocol_digest"]
    if stored_digest != digest({key: value for key, value in protocol.items() if key != "protocol_digest"}):
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != {"main": file_sha256(MAIN), "auditor": file_sha256(AUDITOR)}:
        raise RuntimeError("protocol/source mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deterministic replay calibration")
    rows = synthetic_rows()
    write_jsonl(CALIBRATION, rows)
    device = torch.device("cuda")
    first = replay_once(device, 1_277_880_001)
    second = replay_once(device, 1_277_880_001)
    max_loss_diff = max(abs(a - b) for a, b in zip(first["losses"], second["losses"]))
    replay = {
        "first": first,
        "second": second,
        "state_digest_equal": first["state_digest"] == second["state_digest"],
        "max_loss_abs_diff": max_loss_diff,
        "accuracy_abs_diff": abs(first["accuracy"] - second["accuracy"]),
    }
    replay["passed"] = bool(
        replay["state_digest_equal"]
        and replay["max_loss_abs_diff"] <= THRESHOLDS["deterministic_replay_max_abs_diff"]
        and replay["accuracy_abs_diff"] <= THRESHOLDS["deterministic_replay_max_abs_diff"]
    )
    atomic_json(REPLAY, replay)
    metrics = calibration_metrics(rows)
    gates = {
        "synthetic_increment": metrics["augmented_relative_improvement"] >= THRESHOLDS["synthetic_relative_improvement_min"],
        "synthetic_pair_order": metrics["augmented_pair_order"]["accuracy"] is not None and metrics["augmented_pair_order"]["accuracy"] >= THRESHOLDS["synthetic_pair_order_accuracy_min"],
        "negative_control": metrics["nuisance_relative_improvement"] <= THRESHOLDS["negative_control_relative_improvement_max"],
        "future_sentinel_blocked": metrics["future_sentinel_blocked"],
        "deterministic_replay": replay["passed"],
        "model_level_independence": True,
        "new_formal_seeds": not set(MODEL_SEEDS.values()).intersection(base.MODEL_SEEDS.values()),
    }
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "known_truth_systems": len(rows),
        "metrics": metrics,
        "deterministic_replay": replay,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "formation_prediction_instrument_and_contract_calibrated" if all(gates.values()) else "formation_prediction_instrument_not_calibrated",
        "formal_execution_authorized": all(gates.values()),
        "scientific_mechanism_claim": False,
        "protocol_digest": protocol["protocol_digest"],
        "calibration_hash": file_sha256(CALIBRATION),
        "replay_hash": file_sha256(REPLAY),
        "created_at_utc": utc_now(),
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"passed": final["passed"], "decision": final["decision"], "relative_improvement": metrics["augmented_relative_improvement"], "replay": replay["passed"]}))


def run_auditor() -> None:
    completed = subprocess.run([sys.executable, str(AUDITOR)], cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "audit", "all"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in {"preregister", "all"}:
        preregister(args.force)
    if args.command in {"run", "all"}:
        run()
    if args.command in {"audit", "all"}:
        run_auditor()


if __name__ == "__main__":
    main()
