from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1192_same_parent_causal_update_fork as p1192  # noqa: E402


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def metric_recompute(rows: list[dict[str, Any]]) -> float:
    maximum = 0.0
    for row in rows:
        for horizon in ("immediate", "horizon"):
            calibration = np.asarray(row[horizon + "_calibration"], dtype=np.float64)
            evaluation = np.asarray(row[horizon + "_evaluation"], dtype=np.float64)
            value = p1192.cosine(calibration, evaluation)
            maximum = max(maximum, abs(value - float(row[horizon + "_true_cosine"])))
    return maximum


def summary_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = (
        "eligible_fraction",
        "loss_gap_max",
        "update_norm_relative_error_max",
        "endpoint_norm_relative_error_max",
        "first_order_relative_error_max",
        "update_cosine_max",
        "orthogonal_fraction_min",
        "immediate_effect_norm_min",
        "horizon_effect_norm_min",
        "immediate_train_accuracy_min",
        "horizon_holdout_accuracy_min",
        "immediate_true_cosine_mean",
        "immediate_null_cosine_mean",
        "immediate_advantage_mean",
        "immediate_positive_fraction",
        "horizon_true_cosine_mean",
        "horizon_null_cosine_mean",
        "horizon_advantage_mean",
        "horizon_positive_fraction",
    )
    return max(abs(float(left[key]) - float(right[key])) for key in keys)


def replay_error(reference: dict[str, Any], replay: dict[str, Any]) -> float:
    scalar_keys = (
        "parent_loss",
        "real_immediate_loss",
        "control_immediate_loss",
        "parent_holdout_accuracy",
        "real_immediate_train_accuracy",
        "control_immediate_train_accuracy",
        "real_horizon_holdout_accuracy",
        "control_horizon_holdout_accuracy",
        "loss_gap",
        "update_norm_relative_error",
        "endpoint_norm_relative_error",
        "first_order_relative_error",
        "update_cosine",
        "orthogonal_fraction",
        "immediate_calibration_norm",
        "immediate_evaluation_norm",
        "immediate_true_cosine",
        "horizon_calibration_norm",
        "horizon_evaluation_norm",
        "horizon_true_cosine",
    )
    errors = [abs(float(reference[key]) - float(replay[key])) for key in scalar_keys]
    for key in ("immediate_calibration", "immediate_evaluation", "horizon_calibration", "horizon_evaluation"):
        errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(reference[key], dtype=np.float64)
                        - np.asarray(replay[key], dtype=np.float64)
                    )
                )
            )
        )
    return max(errors)


def main() -> None:
    checks: list[dict[str, Any]] = []
    protocol = p1192.read_json(p1192.PROTOCOL_PATH)
    seal = p1192.read_json(p1192.TRAINING_SEAL)
    rows = p1192.read_jsonl(p1192.RAW_ROWS)
    summary = p1192.read_json(p1192.SUMMARY_PATH)
    claims = p1192.read_json(p1192.CLAIMS_PATH)

    protocol_expected = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    add_check(checks, "protocol_digest", protocol_expected == protocol["protocol_digest"])
    add_check(checks, "source_hashes", protocol["source_hashes"] == p1192.source_hashes())
    add_check(
        checks,
        "development_rows_hash",
        p1192.file_sha256(p1192.DEVELOPMENT_ROWS) == protocol["upstream"]["development_rows_sha256"],
    )
    add_check(
        checks,
        "development_summary_hash",
        p1192.file_sha256(p1192.DEVELOPMENT_SUMMARY) == protocol["upstream"]["development_summary_sha256"],
    )
    add_check(
        checks,
        "upstream_final_hash",
        p1192.file_sha256(p1192.p1191.FINAL_PATH) == protocol["upstream"]["phase1191_final_sha256"],
    )

    seal_expected = digest({key: value for key, value in seal.items() if key != "seal_digest"})
    add_check(checks, "seal_digest", seal_expected == seal["seal_digest"])
    parent_manifest = {
        path.name: p1192.file_sha256(path) for path in sorted(p1192.PARENT_ROOT.glob("*.pt"))
    }
    raw_manifest = {
        path.name: p1192.file_sha256(path) for path in sorted(p1192.FORMAL_ROW_ROOT.glob("*.json"))
    }
    add_check(checks, "parent_manifest", parent_manifest == seal["parent_manifest"], len(parent_manifest))
    add_check(checks, "raw_row_manifest", raw_manifest == seal["raw_row_manifest"], len(raw_manifest))
    add_check(checks, "analysis_rows_hash", p1192.file_sha256(p1192.RAW_ROWS) == seal["analysis_rows_sha256"])
    add_check(checks, "row_count", len(rows) == 64, len(rows))
    add_check(checks, "unique_trajectories", len({row["trajectory_id"] for row in rows}) == 64)
    add_check(checks, "task_count", len({row["task_name"] for row in rows}) == 8)
    add_check(
        checks,
        "split_counts",
        {split: sum(row["split"] == split for row in rows) for split in ("discovery", "confirmation")}
        == {"discovery": 32, "confirmation": 32},
    )
    vector_lengths = {
        len(row[key])
        for row in rows
        for key in ("immediate_calibration", "immediate_evaluation", "horizon_calibration", "horizon_evaluation")
    }
    add_check(checks, "vector_lengths", vector_lengths == {128}, sorted(vector_lengths))
    add_check(
        checks,
        "finite_vectors",
        all(
            np.isfinite(np.asarray(row[key], dtype=np.float64)).all()
            for row in rows
            for key in ("immediate_calibration", "immediate_evaluation", "horizon_calibration", "horizon_evaluation")
        ),
    )
    metric_error = metric_recompute(rows)
    add_check(checks, "metric_recompute", metric_error <= 1e-12, metric_error)

    lookup = {(row["task_name"], row["replicate"]): row for row in rows}
    null_errors = 0
    for row in rows:
        expected_null = lookup[(row["task_name"], (row["replicate"] + 1) % p1192.FORMAL_REPLICATES)]
        if row["null_trajectory_id"] != expected_null["trajectory_id"]:
            null_errors += 1
        for horizon in ("immediate", "horizon"):
            value = p1192.cosine(
                np.asarray(row[horizon + "_calibration"], dtype=np.float64),
                np.asarray(expected_null[horizon + "_evaluation"], dtype=np.float64),
            )
            if not close(value, row[horizon + "_null_cosine"]):
                null_errors += 1
            if not close(row[horizon + "_true_cosine"] - value, row[horizon + "_advantage"]):
                null_errors += 1
    add_check(checks, "null_mapping_and_metrics", null_errors == 0, null_errors)

    for split in ("discovery", "confirmation"):
        recomputed = p1192.summarize(rows, split, 32, 4)
        error = summary_error(recomputed, summary[split])
        structural = all(
            recomputed[key] == summary[split][key]
            for key in (
                "system_count",
                "task_count",
                "eligible_system_count",
                "positive_task_count",
                "control_gate_pass",
                "positive_gate_pass",
                "negative_boundary_pass",
            )
        )
        add_check(checks, split + ".summary", error <= 1e-12 and structural, error)

    positive = bool(summary["discovery"]["positive_gate_pass"] and summary["confirmation"]["positive_gate_pass"])
    negative = bool(summary["discovery"]["negative_boundary_pass"] and summary["confirmation"]["negative_boundary_pass"])
    expected_decision = "positive" if positive else ("negative_boundary" if negative else "ambiguous")
    add_check(checks, "decision", summary["decision"] == expected_decision, expected_decision)
    recompiled = p1192.compile_claims(summary)
    add_check(checks, "typed_claims_recompile", recompiled == claims)
    add_check(
        checks,
        "typed_claims_accept",
        all(
            item["accepted"]
            for family in claims.values()
            for item in list(family["compiled"].values()) + [family["conjunction"]]
        ),
    )
    claim_key = "negative" if summary["decision"] == "negative_boundary" else "positive"
    add_check(
        checks,
        "typed_decision_gate",
        summary["decision"] == "ambiguous" or bool(claims[claim_key]["gate_pass"]),
    )

    control_recomputed = []
    for row in rows:
        expected = bool(
            row["loss_gap"] <= p1192.CONTROL_THRESHOLDS["loss_gap_max"]
            and row["update_norm_relative_error"] <= p1192.CONTROL_THRESHOLDS["update_norm_relative_error_max"]
            and row["endpoint_norm_relative_error"] <= p1192.CONTROL_THRESHOLDS["endpoint_norm_relative_error_max"]
            and row["first_order_relative_error"] <= p1192.CONTROL_THRESHOLDS["first_order_relative_error_max"]
            and row["update_cosine"] <= p1192.CONTROL_THRESHOLDS["update_cosine_max"]
            and row["orthogonal_fraction"] >= p1192.CONTROL_THRESHOLDS["orthogonal_fraction_min"]
            and row["immediate_calibration_norm"] >= p1192.CONTROL_THRESHOLDS["immediate_effect_norm_min"]
            and row["horizon_calibration_norm"] >= p1192.CONTROL_THRESHOLDS["horizon_effect_norm_min"]
            and min(row["real_immediate_train_accuracy"], row["control_immediate_train_accuracy"])
            >= p1192.CONTROL_THRESHOLDS["immediate_train_accuracy_min"]
            and min(row["real_horizon_holdout_accuracy"], row["control_horizon_holdout_accuracy"])
            >= p1192.CONTROL_THRESHOLDS["horizon_holdout_accuracy_min"]
        )
        control_recomputed.append(expected == bool(row["control_qualified"]))
    add_check(checks, "row_control_recompute", all(control_recomputed))

    replay_details = []
    if not torch.cuda.is_available():
        add_check(checks, "eight_task_cuda_replay", False, "CUDA unavailable")
    else:
        device = torch.device("cuda")
        for task_name in sorted({row["task_name"] for row in rows}):
            reference = lookup[(task_name, 0)]
            capsule = torch.load(p1192.capsule_path(task_name, 0), map_location=device, weights_only=False)
            replay = p1192.run_from_capsule(capsule, device)
            error = replay_error(reference, replay)
            replay_details.append({"task_name": task_name, "max_error": error})
        add_check(
            checks,
            "eight_task_cuda_replay",
            all(item["max_error"] <= 1e-6 for item in replay_details),
            replay_details,
        )

    sentinel_row = max(rows, key=lambda row: abs(float(row["horizon_true_cosine"])))
    calibration = np.asarray(sentinel_row["horizon_calibration"], dtype=np.float64)
    evaluation = np.asarray(sentinel_row["horizon_evaluation"], dtype=np.float64)
    corrupted = p1192.cosine(calibration, -evaluation)
    sentinel_strength = abs(corrupted - float(sentinel_row["horizon_true_cosine"]))
    add_check(checks, "sign_corruption_positive_sentinel", sentinel_strength >= 0.10, sentinel_strength)

    audit = {
        "phase": 1192,
        "audit_kind": "independent_digest_vector_null_type_control_and_cuda_replay",
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "gate_pass": all(check["pass"] for check in checks),
        "audit_digest": None,
    }
    audit["audit_digest"] = digest({key: value for key, value in audit.items() if key != "audit_digest"})
    p1192.write_json(p1192.AUDIT_PATH, audit)
    print(canonical_json({"pass_count": audit["pass_count"], "check_count": audit["check_count"], "gate_pass": audit["gate_pass"]}))


if __name__ == "__main__":
    main()
