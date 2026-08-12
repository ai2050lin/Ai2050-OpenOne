"""Independent recomputation audit for Phase1183."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1183_gauge_exact_prospective_mechanism_closure as runner  # noqa: E402


AUDIT_PATH = runner.OUT_ROOT / "audit/independent_audit.json"


def close(left: Any, right: Any, atol: float = 1e-10) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[key], right[key], atol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, bool) or isinstance(right, bool):
            return left == right
        return bool(math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=atol))
    return left == right


def camera_max_error(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    errors = []
    for stage in ("endpoint", "prefix"):
        for label in ("null", "joint"):
            for key in ("mean", "scale", "weights"):
                errors.append(float(np.max(np.abs(expected[stage][label][key] - actual[stage][label][key]))))
    return max(errors)


def audit() -> None:
    if AUDIT_PATH.exists():
        raise RuntimeError("audit already exists")
    protocol = runner.read_json(runner.PROTOCOL_PATH)
    stored_protocol_digest = protocol.pop("protocol_digest")
    checks: dict[str, bool] = {}
    checks["protocol_digest"] = runner.digest(protocol) == stored_protocol_digest
    protocol["protocol_digest"] = stored_protocol_digest
    checks["runner_hash"] = runner.file_sha256(runner.SCRIPT) == protocol["scripts"]["runner"]
    checks["audit_hash"] = runner.file_sha256(runner.AUDIT_SCRIPT) == protocol["scripts"]["audit"]
    checks["phase1171_source_hash"] = runner.file_sha256(Path(runner.p1171.__file__)) == protocol["scripts"]["phase1171_source"]
    checks["phase1181_source_hash"] = runner.file_sha256(Path(runner.p1181.__file__)) == protocol["scripts"]["phase1181_source"]
    checks["phase1182_source_hash"] = runner.file_sha256(Path(runner.p1182.__file__)) == protocol["scripts"]["phase1182_source"]

    signatures = {task.name: runner.task_signature(task.name) for task in runner.TASK_SPECS}
    checks["task_table_digests_unique"] = len({value["table_digest"] for value in signatures.values()}) == len(signatures)
    checks["task_quotient_digests_unique"] = len({value["quotient_digest"] for value in signatures.values()}) == len(signatures)
    checks["task_signatures_frozen"] = all(
        task["signature"] == signatures[task["name"]] for task in protocol["tasks"]
    )
    discovery_seeds = {
        runner.model_seed(index, replicate)
        for index, task in enumerate(runner.TASK_SPECS)
        if task.split == "discovery"
        for replicate in range(runner.REPLICATES)
    }
    confirmation_seeds = {
        runner.model_seed(index, replicate)
        for index, task in enumerate(runner.TASK_SPECS)
        if task.split == "confirmation"
        for replicate in range(runner.REPLICATES)
    }
    checks["seed_splits_disjoint"] = discovery_seeds.isdisjoint(confirmation_seeds)

    preflight = runner.read_json(runner.PREFLIGHT_PATH)
    valid = [row for row in preflight["rows"] if row["kind"] != "leak_positive_sentinel"]
    sentinels = [row for row in preflight["rows"] if row["kind"] == "leak_positive_sentinel"]
    recomputed_preflight = {
        "feature_max_error": max(row["feature_error"] for row in valid),
        "fp64_logit_max_error": max(row["fp64_logit_error"] for row in valid),
        "fp32_logit_max_error": max(row["fp32_logit_error"] for row in valid),
        "positive_sentinel_min_error": min(row["feature_error"] for row in sentinels),
    }
    checks["preflight_aggregates"] = all(
        close(preflight[key], value) for key, value in recomputed_preflight.items()
    )
    threshold = protocol["thresholds"]
    recomputed_preflight_pass = bool(
        recomputed_preflight["feature_max_error"] <= threshold["instrument_feature_max_error_max"]
        and recomputed_preflight["fp64_logit_max_error"] <= threshold["instrument_fp64_logit_max_error_max"]
        and recomputed_preflight["fp32_logit_max_error"] <= threshold["instrument_fp32_logit_max_error_max"]
        and recomputed_preflight["positive_sentinel_min_error"] >= threshold["instrument_positive_sentinel_error_min"]
    )
    checks["preflight_decision"] = preflight["preflight_pass"] == recomputed_preflight_pass

    runner.set_seed(11839991)
    device = torch.device("cuda")
    model = runner.p1171.RoleSquareNetwork(
        runner.p1171.RoleSquareConfig(modulus=runner.MODULUS, width=runner.WIDTH)
    ).to(device)
    x = torch.tensor(
        [(a, b) for a in range(runner.MODULUS) for b in range(runner.MODULUS)],
        dtype=torch.long,
    )
    transformed = runner.gauge_model(model, 11839992, device)
    reference = np.asarray(runner.algebraic_internal_features(model, x))
    candidate = np.asarray(runner.algebraic_internal_features(transformed, x))
    checks["fresh_feature_gauge_recompute"] = float(np.max(np.abs(reference - candidate))) <= threshold["instrument_feature_max_error_max"]
    del model, transformed
    torch.cuda.empty_cache()

    split_summaries: dict[str, Any] = {}
    for split in ("discovery", "confirmation"):
        seal_path = runner.OUT_ROOT / "runs" / split / "training_seal.json"
        rows_path = runner.OUT_ROOT / "runs" / split / "systems.jsonl"
        summary_path = runner.OUT_ROOT / "runs" / split / "summary.json"
        if not seal_path.exists():
            checks[f"{split}_not_run_legally"] = split == "confirmation"
            continue
        seal = runner.read_json(seal_path)
        expected_trajectories = len(runner.task_specs(split)) * runner.REPLICATES
        checks[f"{split}_trajectory_count"] = seal["trajectory_count"] == expected_trajectories
        checks[f"{split}_checkpoint_count"] = seal["checkpoint_count"] == expected_trajectories * len(runner.CHECKPOINT_STEPS)
        checks[f"{split}_training_metrics_hash"] = runner.file_sha256(
            runner.OUT_ROOT / "runs" / split / "training_metrics.jsonl"
        ) == seal["training_metrics_sha256"]
        checks[f"{split}_checkpoint_hashes"] = all(
            runner.file_sha256(runner.OUT_ROOT / "runs" / split / "checkpoints" / name) == value
            for name, value in seal["checkpoint_hashes"].items()
        )
        if not rows_path.exists() or not summary_path.exists():
            checks[f"{split}_scan_present"] = False
            continue
        rows = runner.read_jsonl(rows_path)
        summary = runner.read_json(summary_path)
        split_summaries[split] = summary
        checks[f"{split}_row_count"] = len(rows) == expected_trajectories
        checks[f"{split}_rows_digest"] = summary["rows_digest"] == runner.digest(rows)
        material = runner.material_summary(rows, split, threshold)
        checks[f"{split}_material_recompute"] = close(summary["material"], material, atol=1e-9)

        if split == "discovery" and summary.get("discovery_pass", False):
            camera_expected = runner.fit_camera(rows)
            camera_actual = runner.load_camera()
            checks["camera_seal_hash"] = runner.file_sha256(runner.CAMERA_NPZ) == runner.read_json(runner.CAMERA_META)["npz_sha256"]
            checks["camera_refit"] = camera_max_error(camera_expected, camera_actual) <= 1e-10
            test_names = {task.name for task in runner.TASK_SPECS[6:8]}
            scored = [row for row in rows if row["task_name"] in test_names and runner.qualified(row, threshold)]
            endpoint = runner.p1182.score_stage(scored, "endpoint", camera_actual["endpoint"])
            prefix = runner.p1182.score_stage(scored, "prefix", camera_actual["prefix"])
            endpoint["gate_pass"] = runner.p1182.camera_gate("endpoint", endpoint, threshold)
            prefix["gate_pass"] = runner.p1182.camera_gate("prefix", prefix, threshold)
            checks["discovery_endpoint_recompute"] = close(summary["endpoint"], endpoint, atol=1e-9)
            checks["discovery_prefix_recompute"] = close(summary["prefix"], prefix, atol=1e-9)
        if (runner.OUT_ROOT / "runs" / split / "rescue_raw.json").exists():
            raw = runner.read_json(runner.OUT_ROOT / "runs" / split / "rescue_raw.json")
            rescue = runner.p1182.rescue_summary(raw["tasks"], split, threshold)
            checks[f"{split}_rescue_recompute"] = close(summary["rescue"], rescue, atol=1e-9)
        if split == "confirmation" and "endpoint" in summary:
            camera = runner.load_camera()
            scored = [row for row in rows if runner.qualified(row, threshold)]
            endpoint = runner.p1182.score_stage(scored, "endpoint", camera["endpoint"])
            prefix = runner.p1182.score_stage(scored, "prefix", camera["prefix"])
            endpoint["gate_pass"] = runner.p1182.camera_gate("endpoint", endpoint, threshold)
            prefix["gate_pass"] = runner.p1182.camera_gate("prefix", prefix, threshold)
            checks["confirmation_endpoint_recompute"] = close(summary["endpoint"], endpoint, atol=1e-9)
            checks["confirmation_prefix_recompute"] = close(summary["prefix"], prefix, atol=1e-9)

    final = runner.read_json(runner.FINAL_PATH)
    stored_final = final.pop("final_digest")
    checks["final_digest"] = runner.digest(final) == stored_final
    final["final_digest"] = stored_final
    discovery = split_summaries.get("discovery")
    confirmation = split_summaries.get("confirmation")
    expected_primary = bool(
        discovery is not None
        and discovery.get("discovery_pass", False)
        and confirmation is not None
        and confirmation.get("confirmation_pass", False)
    )
    checks["primary_decision"] = final["primary_pass"] == expected_primary
    checks["registry_closed"] = final["registry"] == "closed_after_one_formal_decision"
    checks["auto_continue_scope"] = final["auto_continue"]["authorized"] == expected_primary
    checks["phase1182_confirmation_not_reused"] = protocol["registry_independence"]["phase1182_confirmation_read"] is False

    result = {
        "phase": runner.PHASE,
        "created_at_utc": runner.utc_now(),
        "protocol_digest": stored_protocol_digest,
        "check_count": len(checks),
        "pass_count": sum(checks.values()),
        "checks": checks,
        "audit_pass": all(checks.values()),
        "primary_pass": expected_primary,
    }
    result["audit_digest"] = runner.digest(result)
    runner.write_json(AUDIT_PATH, result)
    print(runner.canonical_json(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    args = parser.parse_args()
    if args.command == "audit":
        audit()


if __name__ == "__main__":
    main()
