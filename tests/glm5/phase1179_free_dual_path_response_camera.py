#!/usr/bin/env python3
"""Phase1179 free-training endpoint and prefix response-spectrum cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1179_free_training_library as lib  # noqa: E402


PHASE = 1179
OUT_ROOT = ROOT / "tests/glm5/result/phase1179_free_dual_path_response_camera"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
CAMERA_PATH = OUT_ROOT / "analysis/frozen_camera.json"
SCRIPT_PATH = Path(__file__).resolve()
LIBRARY_PATH = ROOT / "tests/glm5/phase1179_free_training_library.py"
AUDIT_PATH = ROOT / "tests/glm5/phase1179_free_dual_path_response_camera_audit.py"
BLOCKS_PER_TASK = 8
COHORTS = ("endpoint", "formation")
CONFIGS_PER_BLOCK = 4
FEATURE_NAMES = (
    "joint_topology_energy",
    "energy_only",
    "topology_only",
    "gate_only",
    "output_only",
    "behavior_only",
    "progress_only",
)
NULL_FEATURES = tuple(name for name in FEATURE_NAMES if name != "joint_topology_energy")


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    tasks: tuple[lib.TaskSpec, ...]


SPLITS = {
    "discovery": SplitConfig(
        seed=117_900,
        tasks=(
            lib.TaskSpec("d11_poly_a", 11, (2, 3, 5), 1),
            lib.TaskSpec("d11_poly_b", 11, (6, 1, 4, 2), 4),
            lib.TaskSpec("d13_poly_a", 13, (3, 7, 2), 2),
            lib.TaskSpec("d13_poly_b", 13, (8, 2, 5, 1), 6),
        ),
    ),
    "confirmation": SplitConfig(
        seed=217_900,
        tasks=(
            lib.TaskSpec("c17_poly_a", 17, (4, 3, 8), 3),
            lib.TaskSpec("c17_poly_b", 17, (9, 5, 2, 7), 8),
            lib.TaskSpec("c19_poly_a", 19, (5, 11, 3), 5),
            lib.TaskSpec("c19_poly_b", 19, (12, 4, 9, 2), 10),
        ),
    ),
}


THRESHOLDS = {
    "endpoint_natural_accuracy_min": 0.99,
    "endpoint_family_margin_gap_max": 0.35,
    "endpoint_parameter_norm_gap_max": 1.0e-4,
    "family_count_per_task_min": 12,
    "final_gate_extreme_min": 0.95,
    "endpoint_camera_family_accuracy_min": 0.90,
    "endpoint_camera_normalized_error_max": 0.25,
    "endpoint_camera_advantage_min": 0.20,
    "prefix_current_accuracy_gap_max": 0.10,
    "prefix_current_loss_gap_max": 0.10,
    "prefix_current_confidence_gap_max": 0.08,
    "prefix_parameter_norm_gap_max": 1.0e-4,
    "prefix_future_holdout_gap_min": 0.25,
    "prefix_camera_family_accuracy_min": 0.90,
    "prefix_camera_normalized_error_max": 0.30,
    "prefix_camera_holdout_mae_max": 0.12,
    "prefix_camera_advantage_min": 0.20,
    "null_family_accuracy_max": 0.65,
    "positive_sentinel_accuracy_min": 0.99,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(lib.canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_payload(task: lib.TaskSpec) -> dict[str, Any]:
    payload = asdict(task)
    payload["coefficients"] = list(task.coefficients)
    payload["table_digest"] = lib.digest(task.table().tolist())
    return payload


def protocol_payload() -> dict[str, Any]:
    payload = {
        "phase": PHASE,
        "schema_version": "phase1179.protocol.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Free-weight symmetric dual-path calibration: freeze an endpoint response-spectrum camera and "
            "a prefix camera before confirmation. This is implementation-selection calibration, not a "
            "natural-network or language mechanism claim."
        ),
        "scripts": {
            "main_sha256": sha256_file(SCRIPT_PATH),
            "library_sha256": sha256_file(LIBRARY_PATH),
            "audit_sha256": sha256_file(AUDIT_PATH),
        },
        "splits": {
            name: {
                "seed": config.seed,
                "tasks": [task_payload(task) for task in config.tasks],
                "blocks_per_task": BLOCKS_PER_TASK,
                "cohorts": list(COHORTS),
                "configs_per_block": CONFIGS_PER_BLOCK,
                "expected_system_count": len(config.tasks) * BLOCKS_PER_TASK * len(COHORTS) * CONFIGS_PER_BLOCK,
            }
            for name, config in SPLITS.items()
        },
        "training": {
            "steps": lib.TRAIN_STEPS,
            "checkpoint_steps": list(lib.CHECKPOINT_STEPS),
            "prefix_step": lib.PREFIX_STEP,
            "learning_rate": lib.LEARNING_RATE,
            "commitment_weight": lib.COMMITMENT_WEIGHT,
            "gate_bias": lib.GATE_BIAS,
            "optimizer": "Adam",
            "dtype": "float32",
            "device": "CUDA required",
        },
        "diagnostic_interventions": list(lib.INTERVENTIONS),
        "primary_feature": "joint_topology_energy",
        "null_features": list(NULL_FEATURES),
        "thresholds": THRESHOLDS,
        "workflow": [
            "preregister",
            "run discovery",
            "freeze camera from discovery only",
            "run confirmation",
            "analyze with frozen camera",
            "independent audit",
        ],
        "public_schema_excludes": [
            "response_spectrum",
            "response_family",
            "modes_by_slot",
            "config_index",
            "seed",
            "final_response",
        ],
        "stopping_rule": (
            "one-shot; no task, seed, feature, checkpoint, ridge, threshold, or intervention search after "
            "discovery starts; auto continuation remains false regardless of outcome"
        ),
    }
    payload["protocol_digest"] = lib.digest(payload)
    return payload


def preregister(force: bool) -> dict[str, Any]:
    if OUT_ROOT.exists() and force:
        resolved = OUT_ROOT.resolve()
        expected_parent = (ROOT / "tests/glm5/result").resolve()
        if expected_parent not in resolved.parents:
            raise RuntimeError(f"unsafe result path: {resolved}")
        shutil.rmtree(resolved)
    if PROTOCOL_PATH.exists():
        raise RuntimeError(f"protocol already exists: {PROTOCOL_PATH}")
    payload = protocol_payload()
    write_json(PROTOCOL_PATH, payload)
    return payload


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1179 requires CUDA")
    return torch.device("cuda")


def system_seed(split: str, task_index: int, cohort_index: int, block: int, config_index: int) -> int:
    base = SPLITS[split].seed
    return base + task_index * 1_000_003 + cohort_index * 100_003 + block * 1_009 + config_index * 37


def system_id(split: str, task: str, cohort: str, block: int, config_index: int) -> str:
    return lib.digest({
        "salt": "phase1179-opaque-id",
        "split": split,
        "task": task,
        "cohort": cohort,
        "block": block,
        "config": config_index,
    })[:20]


def build_split(split: str, device: torch.device) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("preregister before training")
    if split == "confirmation" and not CAMERA_PATH.exists():
        raise RuntimeError("freeze the discovery camera before confirmation")
    run_root = OUT_ROOT / f"runs/{split}"
    if run_root.exists():
        raise RuntimeError(f"split already exists: {run_root}")
    public_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    config = SPLITS[split]
    for task_index, task in enumerate(config.tasks):
        for cohort_index, cohort in enumerate(COHORTS):
            for block in range(BLOCKS_PER_TASK):
                for config_index in range(CONFIGS_PER_BLOCK):
                    seed = system_seed(split, task_index, cohort_index, block, config_index)
                    sid = system_id(split, task.name, cohort, block, config_index)
                    checkpoints, truth = lib.train_system(task, cohort, seed, config_index, device)
                    for checkpoint in checkpoints:
                        public_rows.append({
                            "system_id": sid,
                            "split": split,
                            "task_name": task.name,
                            "task_digest": lib.digest(task_payload(task)),
                            "cohort": cohort,
                            "block": block,
                            **checkpoint,
                        })
                    truth_rows.append({
                        "system_id": sid,
                        "split": split,
                        "task_name": task.name,
                        "cohort": cohort,
                        "block": block,
                        "config_index": config_index,
                        "seed": seed,
                        **truth,
                    })
    write_jsonl(run_root / "public_trajectory.jsonl", public_rows)
    write_jsonl(run_root / "sealed_truth.jsonl", truth_rows)
    summary = summarize_training_split(split, public_rows, truth_rows)
    write_json(run_root / "training_summary.json", summary)
    return summary


def join_rows(
    public_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    cohort: str,
    step: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    truth = {row["system_id"]: row for row in truth_rows if row["cohort"] == cohort}
    return [
        (row, truth[row["system_id"]])
        for row in public_rows
        if row["cohort"] == cohort and row["step"] == step
    ]


def family_gap(joined: list[tuple[dict[str, Any], dict[str, Any]]], field: str) -> float:
    medians = {}
    for family in lib.MODES:
        values = [float(public[field]) for public, truth in joined if truth["response_family"] == family]
        medians[family] = float(np.median(values))
    return abs(medians[lib.MODES[0]] - medians[lib.MODES[1]])


def summarize_training_split(
    split: str,
    public_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    endpoint = join_rows(public_rows, truth_rows, "endpoint", lib.TRAIN_STEPS)
    prefix = join_rows(public_rows, truth_rows, "formation", lib.PREFIX_STEP)
    formation_final = join_rows(public_rows, truth_rows, "formation", lib.TRAIN_STEPS)
    family_counts = {
        cohort: {
            task.name: {
                family: sum(
                    row["cohort"] == cohort and row["task_name"] == task.name and row["response_family"] == family
                    for row in truth_rows
                )
                for family in lib.MODES
            }
            for task in SPLITS[split].tasks
        }
        for cohort in COHORTS
    }
    endpoint_margin_gap = family_gap(endpoint, "train_margin")
    endpoint_parameter_norm_gap = family_gap(endpoint, "parameter_l2")
    prefix_gaps = {
        "accuracy": family_gap(prefix, "train_accuracy"),
        "loss": family_gap(prefix, "train_loss"),
        "confidence": family_gap(prefix, "train_confidence"),
        "parameter_l2": family_gap(prefix, "parameter_l2"),
    }
    future_gap = family_gap(formation_final, "holdout_accuracy")
    metrics = {
        "system_count": len(truth_rows),
        "public_row_count": len(public_rows),
        "endpoint_natural_accuracy_min": min(public["all_accuracy"] for public, _ in endpoint),
        "endpoint_family_margin_gap": endpoint_margin_gap,
        "endpoint_parameter_norm_gap": endpoint_parameter_norm_gap,
        "final_gate_extreme_min": min(max(truth["final_gate_weights"]) for truth in truth_rows),
        "prefix_current_family_gaps": prefix_gaps,
        "prefix_future_holdout_gap": future_gap,
        "family_counts": family_counts,
    }
    checks = {
        "system_count": metrics["system_count"] == len(SPLITS[split].tasks) * BLOCKS_PER_TASK * len(COHORTS) * CONFIGS_PER_BLOCK,
        "endpoint_natural_behavior": metrics["endpoint_natural_accuracy_min"] >= THRESHOLDS["endpoint_natural_accuracy_min"],
        "endpoint_margin_matched": endpoint_margin_gap <= THRESHOLDS["endpoint_family_margin_gap_max"],
        "endpoint_parameter_norm_matched": endpoint_parameter_norm_gap <= THRESHOLDS["endpoint_parameter_norm_gap_max"],
        "family_breadth": all(
            count >= THRESHOLDS["family_count_per_task_min"]
            for cohort in family_counts.values() for task in cohort.values() for count in task.values()
        ),
        "gate_commitment": metrics["final_gate_extreme_min"] >= THRESHOLDS["final_gate_extreme_min"],
        "prefix_accuracy_matched": prefix_gaps["accuracy"] <= THRESHOLDS["prefix_current_accuracy_gap_max"],
        "prefix_loss_matched": prefix_gaps["loss"] <= THRESHOLDS["prefix_current_loss_gap_max"],
        "prefix_confidence_matched": prefix_gaps["confidence"] <= THRESHOLDS["prefix_current_confidence_gap_max"],
        "prefix_parameter_norm_matched": prefix_gaps["parameter_l2"] <= THRESHOLDS["prefix_parameter_norm_gap_max"],
        "future_behavior_separated": future_gap >= THRESHOLDS["prefix_future_holdout_gap_min"],
        "public_schema_clean": all(
            excluded not in row
            for row in public_rows
            for excluded in read_json(PROTOCOL_PATH)["public_schema_excludes"]
        ),
    }
    summary = {"phase": PHASE, "split": split, "metrics": metrics, "checks": checks, "passed": all(checks.values())}
    summary["summary_digest"] = lib.digest(summary)
    return summary


def camera_arrays(
    public_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    cohort: str,
    step: int,
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joined = join_rows(public_rows, truth_rows, cohort, step)
    x = np.asarray([public["features"][feature_name] for public, _ in joined], dtype=np.float64)
    spectrum = np.asarray([truth["response_spectrum"] for _, truth in joined], dtype=np.float64)
    holdout = np.asarray([truth["final_holdout_accuracy"] for _, truth in joined], dtype=np.float64)
    return x, spectrum, holdout


def freeze_camera() -> dict[str, Any]:
    if CAMERA_PATH.exists():
        raise RuntimeError(f"camera already frozen: {CAMERA_PATH}")
    discovery_root = OUT_ROOT / "runs/discovery"
    public_rows = read_jsonl(discovery_root / "public_trajectory.jsonl")
    truth_rows = read_jsonl(discovery_root / "sealed_truth.jsonl")
    cameras = {"endpoint": {}, "prefix": {}}
    target_spectra = np.asarray(
        [row["response_spectrum"] for row in truth_rows if row["cohort"] == "endpoint"],
        dtype=np.float64,
    )
    table_median = np.median(target_spectra[
        [row["response_family"] == "table" for row in truth_rows if row["cohort"] == "endpoint"]
    ], axis=0)
    relation_median = np.median(target_spectra[
        [row["response_family"] == "relation" for row in truth_rows if row["cohort"] == "endpoint"]
    ], axis=0)
    response_scale = float(np.max(np.abs(table_median - relation_median)))
    endpoint_truth = [row for row in truth_rows if row["cohort"] == "endpoint"]
    sentinel_x = np.asarray(
        [[1.0 if row["response_family"] == "relation" else -1.0] for row in endpoint_truth],
        dtype=np.float64,
    )
    sentinel_y = np.asarray([row["response_spectrum"] for row in endpoint_truth], dtype=np.float64)
    for feature_name in FEATURE_NAMES:
        x, spectrum, _ = camera_arrays(
            public_rows, truth_rows, "endpoint", lib.TRAIN_STEPS, feature_name,
        )
        cameras["endpoint"][feature_name] = lib.fit_ridge(x, spectrum)
        x, spectrum, holdout = camera_arrays(
            public_rows, truth_rows, "formation", lib.PREFIX_STEP, feature_name,
        )
        cameras["prefix"][feature_name] = {
            "spectrum": lib.fit_ridge(x, spectrum),
            "holdout": lib.fit_ridge(x, holdout),
        }
    # Deliberately forbidden truth-derived input: this sealed positive control
    # checks that the fitted camera and family scorer can recover an easy signal.
    cameras["positive_sentinel"] = lib.fit_ridge(sentinel_x, sentinel_y)
    payload = {
        "phase": PHASE,
        "schema_version": "phase1179.frozen_camera.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": read_json(PROTOCOL_PATH)["protocol_digest"],
        "discovery_public_sha256": sha256_file(discovery_root / "public_trajectory.jsonl"),
        "discovery_truth_sha256": sha256_file(discovery_root / "sealed_truth.jsonl"),
        "response_scale": response_scale,
        "family_prototypes": {"table": table_median.tolist(), "relation": relation_median.tolist()},
        "features": list(FEATURE_NAMES),
        "cameras": cameras,
    }
    payload["camera_digest"] = lib.digest(payload)
    write_json(CAMERA_PATH, payload)
    return payload


def evaluate_split(split: str, camera: dict[str, Any], write: bool = True) -> dict[str, Any]:
    run_root = OUT_ROOT / f"runs/{split}"
    public_rows = read_jsonl(run_root / "public_trajectory.jsonl")
    truth_rows = read_jsonl(run_root / "sealed_truth.jsonl")
    response_scale = float(camera["response_scale"])
    endpoint_metrics = {}
    prefix_metrics = {}
    for feature_name in FEATURE_NAMES:
        x, spectrum, _ = camera_arrays(
            public_rows, truth_rows, "endpoint", lib.TRAIN_STEPS, feature_name,
        )
        prediction = lib.apply_ridge(camera["cameras"]["endpoint"][feature_name], x)
        endpoint_metrics[feature_name] = lib.camera_metrics(prediction, spectrum, response_scale)
        x, spectrum, holdout = camera_arrays(
            public_rows, truth_rows, "formation", lib.PREFIX_STEP, feature_name,
        )
        prediction = lib.apply_ridge(camera["cameras"]["prefix"][feature_name]["spectrum"], x)
        values = lib.camera_metrics(prediction, spectrum, response_scale)
        holdout_prediction = lib.apply_ridge(
            camera["cameras"]["prefix"][feature_name]["holdout"], x,
        ).reshape(-1)
        values["holdout_mae"] = float(np.mean(np.abs(holdout_prediction - holdout)))
        prefix_metrics[feature_name] = values
    endpoint_primary = endpoint_metrics["joint_topology_energy"]
    prefix_primary = prefix_metrics["joint_topology_energy"]
    endpoint_best_null = max(endpoint_metrics[name]["family_accuracy"] for name in NULL_FEATURES)
    prefix_best_null = max(prefix_metrics[name]["family_accuracy"] for name in NULL_FEATURES)
    endpoint_truth = [row for row in truth_rows if row["cohort"] == "endpoint"]
    sentinel_x = np.asarray(
        [[1.0 if row["response_family"] == "relation" else -1.0] for row in endpoint_truth],
        dtype=np.float64,
    )
    sentinel_y = np.asarray([row["response_spectrum"] for row in endpoint_truth], dtype=np.float64)
    sentinel_prediction = lib.apply_ridge(camera["cameras"]["positive_sentinel"], sentinel_x)
    positive_sentinel = lib.camera_metrics(sentinel_prediction, sentinel_y, response_scale)
    checks = {
        "training_split_passed": read_json(run_root / "training_summary.json")["passed"],
        "endpoint_family_accuracy": endpoint_primary["family_accuracy"] >= THRESHOLDS["endpoint_camera_family_accuracy_min"],
        "endpoint_spectrum_error": endpoint_primary["normalized_median_linf_error"] <= THRESHOLDS["endpoint_camera_normalized_error_max"],
        "endpoint_nulls_at_chance": endpoint_best_null <= THRESHOLDS["null_family_accuracy_max"],
        "endpoint_advantage": endpoint_primary["family_accuracy"] - endpoint_best_null >= THRESHOLDS["endpoint_camera_advantage_min"],
        "prefix_family_accuracy": prefix_primary["family_accuracy"] >= THRESHOLDS["prefix_camera_family_accuracy_min"],
        "prefix_spectrum_error": prefix_primary["normalized_median_linf_error"] <= THRESHOLDS["prefix_camera_normalized_error_max"],
        "prefix_holdout_prediction": prefix_primary["holdout_mae"] <= THRESHOLDS["prefix_camera_holdout_mae_max"],
        "prefix_nulls_at_chance": prefix_best_null <= THRESHOLDS["null_family_accuracy_max"],
        "prefix_advantage": prefix_primary["family_accuracy"] - prefix_best_null >= THRESHOLDS["prefix_camera_advantage_min"],
        "positive_sentinel": positive_sentinel["family_accuracy"] >= THRESHOLDS["positive_sentinel_accuracy_min"],
    }
    result = {
        "phase": PHASE,
        "split": split,
        "endpoint": endpoint_metrics,
        "prefix": prefix_metrics,
        "endpoint_best_null_family_accuracy": endpoint_best_null,
        "prefix_best_null_family_accuracy": prefix_best_null,
        "positive_sentinel": positive_sentinel,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["score_digest"] = lib.digest(result)
    if write:
        write_json(OUT_ROOT / f"analysis/{split}_score.json", result)
    return result


def analyze() -> dict[str, Any]:
    camera = read_json(CAMERA_PATH)
    scores = {split: evaluate_split(split, camera) for split in SPLITS}
    payload = {
        "phase": PHASE,
        "schema_version": "phase1179.final.v1",
        "protocol_digest": read_json(PROTOCOL_PATH)["protocol_digest"],
        "camera_digest": camera["camera_digest"],
        "split_scores": scores,
        "primary_pass": all(score["passed"] for score in scores.values()),
        "component_status": {
            "free_weight_training": True,
            "endpoint_response_spectrum_camera": scores["confirmation"]["checks"]["endpoint_family_accuracy"],
            "prefix_response_spectrum_prediction": scores["confirmation"]["checks"]["prefix_family_accuracy"],
            "future_holdout_prediction": scores["confirmation"]["checks"]["prefix_holdout_prediction"],
            "natural_network_external_validity": False,
            "transformer_external_validity": False,
        },
        "evidence_scope": (
            "Symmetric finite-task free-weight implementation-selection calibration. The candidate modes and "
            "initial route bias are still architecturally supplied; success does not establish spontaneous "
            "mechanism formation, a Transformer mechanism, or a language mechanism."
        ),
        "auto_continue": False,
        "next_gate": (
            "Independently test whether the camera predicts a freely learned implementation when no semantic "
            "mode assignment or antithetic gate bias is supplied; do not retune Phase1179 on its tasks."
        ),
    }
    payload["final_digest"] = lib.digest(payload)
    write_json(OUT_ROOT / "analysis/final.json", payload)
    return payload


def probe() -> dict[str, Any]:
    device = require_cuda()
    task = lib.TaskSpec("probe_p7_unused", 7, (1, 2, 3), 0)
    rows = []
    for config_index in range(4):
        checkpoints, truth = lib.train_system(task, "formation", 9_000 + config_index, config_index, device)
        rows.append({"config_index": config_index, "prefix": checkpoints[0], "final": checkpoints[-1], "truth": truth})
    payload = {
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
        "passed": (
            {row["truth"]["response_family"] for row in rows} == set(lib.MODES)
            and min(row["final"]["train_accuracy"] for row in rows) >= 0.99
        ),
    }
    write_json(ROOT / "tests/glm5_temp/phase1179_formal_probe.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("probe", "preregister", "run", "freeze-camera", "analyze"),
    )
    parser.add_argument("--split", choices=tuple(SPLITS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "probe":
        result = probe()
    elif args.command == "preregister":
        result = preregister(args.force)
    elif args.command == "run":
        if args.split is None:
            raise SystemExit("--split is required")
        result = build_split(args.split, require_cuda())
    elif args.command == "freeze-camera":
        result = freeze_camera()
    else:
        result = analyze()
    if args.command == "freeze-camera":
        display = {
            "phase": result["phase"],
            "camera_digest": result["camera_digest"],
            "response_scale": result["response_scale"],
            "features": result["features"],
        }
    elif args.command == "run":
        display = {
            "phase": result["phase"],
            "split": result["split"],
            "passed": result["passed"],
            "metrics": result["metrics"],
            "checks": result["checks"],
            "summary_digest": result["summary_digest"],
        }
    elif args.command == "analyze":
        display = {
            "phase": result["phase"],
            "primary_pass": result["primary_pass"],
            "decision": result["decision"],
            "auto_continue": result["auto_continue"],
            "final_digest": result["final_digest"],
        }
    else:
        display = result
    print(lib.canonical(display))


if __name__ == "__main__":
    main()
