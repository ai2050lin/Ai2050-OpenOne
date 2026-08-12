#!/usr/bin/env python3
"""Independent integrity and numerical audit for Phase1180."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1180_free_dual_path_response_camera as main  # noqa: E402
import phase1180_free_training_library as lib  # noqa: E402


AUDIT_OUTPUT = main.OUT_ROOT / "audit/independent_audit.json"


def run_audit(sample_per_split: int = 8) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    protocol = main.read_json(main.PROTOCOL_PATH)
    camera = main.read_json(main.CAMERA_PATH)
    final = main.read_json(main.OUT_ROOT / "analysis/final.json")
    add("protocol_digest", lib.digest({k: v for k, v in protocol.items() if k != "protocol_digest"}) == protocol["protocol_digest"])
    add("main_script_hash", main.sha256_file(main.SCRIPT_PATH) == protocol["scripts"]["main_sha256"])
    add("library_script_hash", main.sha256_file(main.LIBRARY_PATH) == protocol["scripts"]["library_sha256"])
    add("base_library_script_hash", main.sha256_file(main.BASE_LIBRARY_PATH) == protocol["scripts"]["base_library_sha256"])
    add("audit_script_hash", main.sha256_file(main.AUDIT_PATH) == protocol["scripts"]["audit_sha256"])
    preflight = main.read_json(main.PREFLIGHT_PATH)
    add("engineering_preflight_passed", preflight["passed"] is True)
    add("engineering_preflight_link", preflight["preflight_digest"] == protocol["engineering_preflight_digest"])
    add("frozen_thresholds", protocol["thresholds"] == main.THRESHOLDS)
    add("frozen_steps", tuple(protocol["training"]["checkpoint_steps"]) == lib.CHECKPOINT_STEPS)
    add("frozen_prefix_step", protocol["training"]["prefix_step"] == lib.PREFIX_STEP)
    add("frozen_features", protocol["primary_feature"] == "joint_topology_energy" and tuple(protocol["null_features"]) == main.NULL_FEATURES)
    add("camera_digest", lib.digest({k: v for k, v in camera.items() if k != "camera_digest"}) == camera["camera_digest"])
    add("camera_protocol_link", camera["protocol_digest"] == protocol["protocol_digest"])
    discovery_root = main.OUT_ROOT / "runs/discovery"
    add("camera_discovery_public_link", camera["discovery_public_sha256"] == main.sha256_file(discovery_root / "public_trajectory.jsonl"))
    add("camera_discovery_truth_link", camera["discovery_truth_sha256"] == main.sha256_file(discovery_root / "sealed_truth.jsonl"))
    confirmation_public = main.OUT_ROOT / "runs/confirmation/public_trajectory.jsonl"
    add("camera_frozen_before_confirmation", main.CAMERA_PATH.stat().st_mtime_ns < confirmation_public.stat().st_mtime_ns)

    task_names = {split: {task.name for task in config.tasks} for split, config in main.SPLITS.items()}
    moduli = {split: {task.modulus for task in config.tasks} for split, config in main.SPLITS.items()}
    add("split_task_names_disjoint", task_names["discovery"].isdisjoint(task_names["confirmation"]))
    add("split_moduli_disjoint", moduli["discovery"].isdisjoint(moduli["confirmation"]))

    maximum_retrain_spectrum_error = 0.0
    maximum_retrain_feature_error = 0.0
    for split, config in main.SPLITS.items():
        run_root = main.OUT_ROOT / f"runs/{split}"
        public = main.read_jsonl(run_root / "public_trajectory.jsonl")
        truth = main.read_jsonl(run_root / "sealed_truth.jsonl")
        summary = main.read_json(run_root / "training_summary.json")
        score = main.read_json(main.OUT_ROOT / f"analysis/{split}_score.json")
        expected_systems = len(config.tasks) * main.BLOCKS_PER_TASK * len(main.COHORTS) * main.CONFIGS_PER_BLOCK
        expected_public = expected_systems * len(lib.CHECKPOINT_STEPS)
        add(f"{split}_truth_count", len(truth) == expected_systems, len(truth))
        add(f"{split}_public_count", len(public) == expected_public, len(public))
        truth_ids = {row["system_id"] for row in truth}
        public_ids = {row["system_id"] for row in public}
        add(f"{split}_unique_truth_ids", len(truth_ids) == len(truth))
        add(f"{split}_joined_ids", truth_ids == public_ids)
        add(f"{split}_checkpoint_completeness", all(
            {row["step"] for row in public if row["system_id"] == sid} == set(lib.CHECKPOINT_STEPS)
            for sid in truth_ids
        ))
        add(f"{split}_public_schema_clean", all(
            excluded not in row
            for row in public
            for excluded in protocol["public_schema_excludes"]
        ))
        recomputed_summary = main.summarize_training_split(split, public, truth)
        add(f"{split}_training_summary_exact", recomputed_summary == summary)
        recomputed_score = main.evaluate_split(split, camera, write=False)
        add(f"{split}_score_exact", recomputed_score == score)
        add(f"{split}_all_training_checks", all(summary["checks"].values()), summary["checks"])
        add(f"{split}_all_camera_checks", all(score["checks"].values()), score["checks"])

        public_by_key = {(row["system_id"], row["step"]): row for row in public}
        sample_indices = np.linspace(0, len(truth) - 1, sample_per_split, dtype=int)
        task_by_name = {task.name: task for task in config.tasks}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for deterministic trajectory audit")
        for index in sample_indices:
            stored = truth[int(index)]
            checkpoints, new_truth = lib.train_system(
                task_by_name[stored["task_name"]],
                stored["cohort"],
                int(stored["seed"]),
                int(stored["config_index"]),
                torch.device("cuda"),
            )
            maximum_retrain_spectrum_error = max(
                maximum_retrain_spectrum_error,
                float(np.max(np.abs(
                    np.asarray(new_truth["response_spectrum"]) - np.asarray(stored["response_spectrum"])
                ))),
            )
            for checkpoint in checkpoints:
                original = public_by_key[(stored["system_id"], checkpoint["step"])]
                maximum_retrain_feature_error = max(
                    maximum_retrain_feature_error,
                    float(np.max(np.abs(
                        np.asarray(checkpoint["features"]["joint_topology_energy"])
                        - np.asarray(original["features"]["joint_topology_energy"])
                    ))),
                )

    add("sample_retrain_spectrum", maximum_retrain_spectrum_error <= 1.0e-7, maximum_retrain_spectrum_error)
    add("sample_retrain_public_feature", maximum_retrain_feature_error <= 1.0e-7, maximum_retrain_feature_error)
    add("final_digest", lib.digest({k: v for k, v in final.items() if k != "final_digest"}) == final["final_digest"])
    add("final_protocol_link", final["protocol_digest"] == protocol["protocol_digest"])
    add("final_camera_link", final["camera_digest"] == camera["camera_digest"])
    add("final_split_links", all(
        final["split_scores"][split]["score_digest"]
        == main.read_json(main.OUT_ROOT / f"analysis/{split}_score.json")["score_digest"]
        for split in main.SPLITS
    ))
    add("primary_decision_recompute", final["primary_pass"] == all(
        final["split_scores"][split]["passed"] for split in main.SPLITS
    ))
    add("auto_continue_false", final["auto_continue"] is False)
    add("scope_boundary_present", "architecturally supplied" in final["evidence_scope"])

    audit = {
        "phase": main.PHASE,
        "audit": "independent artifact, camera-freeze, score, and sampled CUDA trajectory recomputation",
        "checks": checks,
        "passed_count": sum(row["passed"] for row in checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "check_count": len(checks),
        "maximum_retrain_spectrum_error": maximum_retrain_spectrum_error,
        "maximum_retrain_public_feature_error": maximum_retrain_feature_error,
    }
    audit["passed"] = audit["failed_count"] == 0
    audit["audit_digest"] = lib.digest(audit)
    return audit


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-split", type=int, default=8)
    args = parser.parse_args()
    if AUDIT_OUTPUT.exists():
        raise RuntimeError(f"audit already exists: {AUDIT_OUTPUT}")
    result = run_audit(args.sample_per_split)
    main.write_json(AUDIT_OUTPUT, result)
    print(json.dumps({
        "passed": result["passed"],
        "passed_count": result["passed_count"],
        "check_count": result["check_count"],
        "max_spectrum_error": result["maximum_retrain_spectrum_error"],
        "max_feature_error": result["maximum_retrain_public_feature_error"],
        "audit_digest": result["audit_digest"],
    }, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main_cli()
