"""Independent structural, numerical, null, type, and CUDA replay audit for Phase 1193."""

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

import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def add(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def vector_metric_error(rows: list[dict[str, Any]]) -> float:
    maximum = 0.0
    for row in rows:
        for horizon in ("immediate", "horizon"):
            calibration = np.asarray(row[horizon + "_calibration"], dtype=np.float64)
            evaluation = np.asarray(row[horizon + "_evaluation"], dtype=np.float64)
            maximum = max(
                maximum,
                abs(p1193.cosine(calibration, evaluation) - float(row[horizon + "_true_cosine"])),
                abs(float(np.linalg.norm(calibration)) - float(row[horizon + "_calibration_norm"])),
                abs(float(np.linalg.norm(evaluation)) - float(row[horizon + "_evaluation_norm"])),
            )
    return maximum


def row_control(row: dict[str, Any]) -> bool:
    threshold = p1193.CONTROL_THRESHOLDS
    return bool(
        row["gauge_logit_max_error"] <= threshold["gauge_logit_max_error_max"]
        and row["gauge_response_distance"] <= threshold["gauge_response_distance_max"]
        and row["sentinel_logit_max_change"] >= threshold["sentinel_logit_change_min"]
        and row["sentinel_response_distance"] >= threshold["sentinel_response_distance_min"]
        and row["loss_gap"] <= threshold["loss_gap_max"]
        and row["update_norm_relative_error"] <= threshold["update_norm_relative_error_max"]
        and row["endpoint_norm_relative_error"] <= threshold["endpoint_norm_relative_error_max"]
        and row["first_order_relative_error"] <= threshold["first_order_relative_error_max"]
        and row["update_cosine"] <= threshold["update_cosine_max"]
        and row["orthogonal_fraction"] >= threshold["orthogonal_fraction_min"]
        and row["immediate_calibration_norm"] >= threshold["immediate_effect_norm_min"]
        and row["horizon_calibration_norm"] >= threshold["horizon_effect_norm_min"]
        and row["parent_accuracy"] >= threshold["parent_accuracy_min"]
        and min(row["real_immediate_accuracy"], row["control_immediate_accuracy"])
        >= threshold["immediate_accuracy_min"]
        and min(row["real_horizon_accuracy"], row["control_horizon_accuracy"])
        >= threshold["horizon_accuracy_min"]
    )


def replay_error(reference: dict[str, Any], replay: dict[str, Any]) -> float:
    scalar_keys = (
        "parent_loss",
        "real_immediate_loss",
        "control_immediate_loss",
        "parent_accuracy",
        "real_immediate_accuracy",
        "control_immediate_accuracy",
        "real_horizon_accuracy",
        "control_horizon_accuracy",
        "gauge_logit_max_error",
        "gauge_response_distance",
        "sentinel_logit_max_change",
        "sentinel_response_distance",
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
    protocol = p1193.read_json(p1193.PROTOCOL_PATH)
    seal = p1193.read_json(p1193.TRAINING_SEAL)
    rows = p1193.read_jsonl(p1193.RAW_ROWS)
    summary = p1193.read_json(p1193.SUMMARY_PATH)
    claims = p1193.read_json(p1193.CLAIMS_PATH)

    add(
        checks,
        "protocol_digest",
        digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
        == protocol["protocol_digest"],
    )
    add(checks, "source_hashes", protocol["source_hashes"] == p1193.source_hashes())
    add(
        checks,
        "development_rows_hash",
        p1193.file_sha256(p1193.DEVELOPMENT_ROWS)
        == protocol["upstream"]["development_rows_sha256"],
    )
    add(
        checks,
        "development_summary_hash",
        p1193.file_sha256(p1193.DEVELOPMENT_SUMMARY)
        == protocol["upstream"]["development_summary_sha256"],
    )
    add(
        checks,
        "upstream_final_hash",
        p1193.file_sha256(p1193.p1192.FINAL_PATH)
        == protocol["upstream"]["phase1192_final_sha256"],
    )
    add(
        checks,
        "seal_digest",
        digest({key: value for key, value in seal.items() if key != "seal_digest"})
        == seal["seal_digest"],
    )
    parent_manifest = {
        path.name: p1193.file_sha256(path) for path in sorted(p1193.PARENT_ROOT.glob("*.pt"))
    }
    row_manifest = {
        path.name: p1193.file_sha256(path) for path in sorted(p1193.FORMAL_ROW_ROOT.glob("*.json"))
    }
    add(checks, "parent_manifest", parent_manifest == seal["parent_manifest"], len(parent_manifest))
    add(checks, "row_manifest", row_manifest == seal["raw_row_manifest"], len(row_manifest))
    add(
        checks,
        "analysis_rows_hash",
        p1193.file_sha256(p1193.RAW_ROWS) == seal["analysis_rows_sha256"],
    )
    add(checks, "row_count", len(rows) == 64, len(rows))
    add(checks, "trajectory_uniqueness", len({row["trajectory_id"] for row in rows}) == 64)
    add(checks, "task_count", len({row["task_name"] for row in rows}) == 8)
    add(checks, "architecture_count", {row["architecture"] for row in rows} == set(p1193.ARCHITECTURES))
    add(
        checks,
        "split_counts",
        {split: sum(row["split"] == split for row in rows) for split in ("discovery", "confirmation")}
        == {"discovery": 32, "confirmation": 32},
    )
    expected_dimensions = {
        architecture: config.layers * (config.heads + 1)
        for architecture, config in p1193.ARCHITECTURES.items()
    }
    dimensions_ok = all(
        row["response_dimension"] == expected_dimensions[row["architecture"]]
        and all(
            len(row[key]) == expected_dimensions[row["architecture"]]
            for key in ("immediate_calibration", "immediate_evaluation", "horizon_calibration", "horizon_evaluation")
        )
        for row in rows
    )
    add(checks, "architecture_specific_vector_dimensions", dimensions_ok, expected_dimensions)
    add(
        checks,
        "finite_vectors",
        all(
            np.isfinite(np.asarray(row[key], dtype=np.float64)).all()
            for row in rows
            for key in ("immediate_calibration", "immediate_evaluation", "horizon_calibration", "horizon_evaluation")
        ),
    )
    metric_error = vector_metric_error(rows)
    add(checks, "vector_metric_recompute", metric_error <= 1e-12, metric_error)

    lookup = {
        (row["split"], row["task_index"], row["architecture"], row["replicate"]): row
        for row in rows
    }
    split_indices = {
        split: sorted({row["task_index"] for row in rows if row["split"] == split})
        for split in ("discovery", "confirmation")
    }
    null_errors = 0
    for row in rows:
        replicate_null = lookup[
            (
                row["split"],
                row["task_index"],
                row["architecture"],
                (row["replicate"] + 1) % p1193.FORMAL_REPLICATES,
            )
        ]
        indices = split_indices[row["split"]]
        next_task = indices[(indices.index(row["task_index"]) + 1) % len(indices)]
        task_null = lookup[(row["split"], next_task, row["architecture"], row["replicate"])]
        if row["replicate_null_trajectory_id"] != replicate_null["trajectory_id"]:
            null_errors += 1
        if row["task_null_trajectory_id"] != task_null["trajectory_id"]:
            null_errors += 1
        for horizon in ("immediate", "horizon"):
            calibration = np.asarray(row[horizon + "_calibration"], dtype=np.float64)
            replicate_cosine = p1193.cosine(
                calibration, np.asarray(replicate_null[horizon + "_evaluation"], dtype=np.float64)
            )
            task_cosine = p1193.cosine(
                calibration, np.asarray(task_null[horizon + "_evaluation"], dtype=np.float64)
            )
            conservative = max(replicate_cosine, task_cosine)
            if not close(replicate_cosine, row[horizon + "_replicate_null_cosine"]):
                null_errors += 1
            if not close(task_cosine, row[horizon + "_task_null_cosine"]):
                null_errors += 1
            if not close(conservative, row[horizon + "_null_cosine"]):
                null_errors += 1
            if not close(row[horizon + "_true_cosine"] - conservative, row[horizon + "_advantage"]):
                null_errors += 1
    add(checks, "dual_null_mapping_and_metrics", null_errors == 0, null_errors)

    add(checks, "row_control_recompute", all(row_control(row) == row["control_qualified"] for row in rows))
    for split in ("discovery", "confirmation"):
        recomputed = p1193.summarize(rows, split, 32, 4)
        add(checks, split + ".summary_recompute", recomputed == summary[split])
    positive = summary["discovery"]["positive_gate_pass"] and summary["confirmation"]["positive_gate_pass"]
    negative = summary["discovery"]["negative_boundary_pass"] and summary["confirmation"]["negative_boundary_pass"]
    expected_decision = "positive" if positive else ("negative_boundary" if negative else "ambiguous")
    add(checks, "decision", summary["decision"] == expected_decision, expected_decision)
    add(checks, "typed_claims_recompile", p1193.compile_claims(summary) == claims)
    add(
        checks,
        "typed_claims_accept",
        all(
            claim["accepted"]
            for family in claims.values()
            for claim in list(family["compiled"].values()) + [family["conjunction"]]
        ),
    )

    replay_details: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        add(checks, "four_cell_cuda_replay", False, "CUDA unavailable")
    else:
        device = torch.device("cuda")
        for split in ("discovery", "confirmation"):
            task_name = sorted({row["task_name"] for row in rows if row["split"] == split})[0]
            for architecture in p1193.ARCHITECTURES:
                reference = next(
                    row
                    for row in rows
                    if row["task_name"] == task_name
                    and row["architecture"] == architecture
                    and row["replicate"] == 0
                )
                capsule = torch.load(
                    p1193.capsule_path(task_name, architecture, 0),
                    map_location=device,
                    weights_only=False,
                )
                replay = p1193.run_from_capsule(capsule, device)
                error = replay_error(reference, replay)
                replay_details.append(
                    {"split": split, "task": task_name, "architecture": architecture, "max_error": error}
                )
        add(
            checks,
            "four_cell_cuda_replay",
            all(item["max_error"] <= 1e-6 for item in replay_details),
            replay_details,
        )

    sentinel = max(rows, key=lambda row: float(row["horizon_calibration_norm"]))
    calibration = np.asarray(sentinel["horizon_calibration"], dtype=np.float64)
    evaluation = np.asarray(sentinel["horizon_evaluation"], dtype=np.float64)
    corruption_strength = abs(
        p1193.cosine(calibration, -evaluation) - float(sentinel["horizon_true_cosine"])
    )
    add(checks, "sign_corruption_positive_sentinel", corruption_strength >= 0.5, corruption_strength)

    audit = {
        "phase": 1193,
        "audit_kind": "independent_digest_gauge_vector_dual_null_type_control_and_cuda_replay",
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "gate_pass": all(check["pass"] for check in checks),
        "audit_digest": None,
    }
    audit["audit_digest"] = digest({key: value for key, value in audit.items() if key != "audit_digest"})
    p1193.write_json(p1193.AUDIT_PATH, audit)
    print(
        canonical(
            {
                "check_count": audit["check_count"],
                "pass_count": audit["pass_count"],
                "gate_pass": audit["gate_pass"],
            }
        )
    )
    if not audit["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
