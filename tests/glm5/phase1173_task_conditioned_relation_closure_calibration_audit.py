"""Independent result and integrity audit for Phase1173."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase1173_task_conditioned_relation_closure_calibration as phase


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1173_task_conditioned_relation_closure_calibration"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def close(left: Any, right: Any, tolerance: float = 1.0e-10) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isfinite(float(left)) and np.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance)
    return left == right


def load_joined(split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = OUT_ROOT / f"runs/{split}"
    public = phase.read_jsonl(run_root / "public_systems.jsonl")
    truth_rows = phase.read_jsonl(run_root / "sealed_truth.jsonl")
    truth = {row["system_id"]: row["morphology"] for row in truth_rows}
    systems = [dict(row, morphology=truth[row["system_id"]]) for row in public]
    trajectories = phase.read_jsonl(run_root / "formation_trajectory.jsonl")
    return systems, trajectories


def run_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    protocol = phase.read_json(phase.PROTOCOL_PATH)
    protocol_copy = dict(protocol)
    stored_protocol_digest = protocol_copy.pop("protocol_digest")
    add("protocol_digest", digest(protocol_copy) == stored_protocol_digest)
    add("main_script_hash", protocol["script_sha256"] == phase.sha256_file(phase.SCRIPT_PATH))
    add("audit_script_hash", protocol["audit_script_sha256"] == phase.sha256_file(Path(__file__).resolve()))
    add("source_phase1172_hash", protocol["source_phase1172_final_sha256"] == phase.sha256_file(phase.SOURCE_FINAL))
    add("frozen_morphologies", tuple(protocol["morphologies"]) == phase.MORPHOLOGIES)
    add("frozen_thresholds", close(protocol["thresholds"], phase.THRESHOLDS))
    add("no_phase1172_confirmation_input", "phase1172" not in canonical(protocol.get("splits", {})).lower())

    all_system_ids: dict[str, set[str]] = {}
    all_seeds: dict[str, set[int]] = {}
    split_passes: dict[str, bool] = {}
    for split, config in phase.SPLITS.items():
        systems, trajectories = load_joined(split)
        expected_count = phase.REPLICATES * len(phase.MORPHOLOGIES)
        expected_trajectory_count = phase.REPLICATES * len(phase.ALPHAS)
        add(f"{split}_system_count", len(systems) == expected_count, len(systems))
        add(f"{split}_trajectory_count", len(trajectories) == expected_trajectory_count, len(trajectories))
        add(f"{split}_unique_system_ids", len({row["system_id"] for row in systems}) == expected_count)
        add(
            f"{split}_balanced_morphologies",
            all(sum(row["morphology"] == name for row in systems) == phase.REPLICATES for name in phase.MORPHOLOGIES),
        )
        add(
            f"{split}_finite_metrics",
            all(
                np.isfinite(float(value))
                for row in systems
                for value in (
                    row["conditioned_camera"]["reuse_error"],
                    row["conditioned_camera"]["closure_error"],
                    row["conditioned_camera"]["score"],
                    row["unconditioned_camera"]["score"],
                )
            ),
        )
        add(
            f"{split}_label_only_identical",
            len({row["label_only_digest"] for row in systems}) == 1,
        )

        regenerated_systems = []
        for replicate in range(phase.REPLICATES):
            for morphology in phase.MORPHOLOGIES:
                regenerated_systems.append(phase.system_row(config, replicate, morphology))
        stored_by_id = {row["system_id"]: row for row in systems}
        add(
            f"{split}_all_systems_regenerated",
            all(close(stored_by_id[row["system_id"]], row) for row in regenerated_systems),
        )

        regenerated_trajectories = [
            phase.trajectory_row(config, replicate, alpha)
            for replicate in range(phase.REPLICATES)
            for alpha in phase.ALPHAS
        ]
        stored_trajectory = {row["trajectory_id"]: row for row in trajectories}
        add(
            f"{split}_all_trajectories_regenerated",
            all(close(stored_trajectory[row["trajectory_id"]], row) for row in regenerated_trajectories),
        )

        recomputed = phase.summarize_rows(config, systems, trajectories)
        score_name = "discovery_gate.json" if split == "discovery" else "confirmation_score.json"
        scored = phase.read_json(OUT_ROOT / f"analysis/{score_name}")
        add(
            f"{split}_summary_recomputed",
            all(close(recomputed[key], scored[key]) for key in recomputed),
        )
        add(f"{split}_all_frozen_checks_pass", recomputed["passed"], recomputed["checks"])
        split_passes[split] = bool(recomputed["passed"])
        all_system_ids[split] = {row["system_id"] for row in systems}
        all_seeds[split] = {int(row["seed"]) for row in systems}

    add("split_system_ids_disjoint", all_system_ids["discovery"].isdisjoint(all_system_ids["confirmation"]))
    add("split_seeds_disjoint", all_seeds["discovery"].isdisjoint(all_seeds["confirmation"]))
    add(
        "split_dimensions_independent",
        phase.SPLITS["discovery"].modulus != phase.SPLITS["confirmation"].modulus
        and phase.SPLITS["discovery"].contexts != phase.SPLITS["confirmation"].contexts,
    )

    final = phase.read_json(OUT_ROOT / "analysis/final.json")
    final_copy = dict(final)
    stored_final_digest = final_copy.pop("final_digest")
    add("final_digest", digest(final_copy) == stored_final_digest)
    add("final_matches_split_gates", bool(final["relation_closure_camera_calibrated"]) == all(split_passes.values()))
    add("final_scope_is_representation_only", "causal use" in final["evidence_scope"] and "does not" in final["evidence_scope"])
    add("auto_continue_is_false", final["auto_continue"] is False)

    passed = all(check["passed"] for check in checks)
    payload = {
        "phase": phase.PHASE,
        "audit": "independent deterministic regeneration and result audit",
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "failed_count": sum(not check["passed"] for check in checks),
        "passed": passed,
        "checks": checks,
    }
    payload["audit_digest"] = digest(payload)
    phase.write_json(AUDIT_PATH, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    payload = run_audit()
    print(canonical({
        "passed": payload["passed"],
        "passed_count": payload["passed_count"],
        "check_count": payload["check_count"],
        "audit_digest": payload["audit_digest"],
    }))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
