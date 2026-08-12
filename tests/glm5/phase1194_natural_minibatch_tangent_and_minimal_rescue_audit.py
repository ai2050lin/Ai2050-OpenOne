"""Independent digest, metric, null, gate, and CUDA replay audit for Phase 1194."""

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

import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def add(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def prediction_metric_error(rows: list[dict[str, Any]]) -> float:
    maximum = 0.0
    for row in rows:
        target = np.asarray(row["target_evaluation"], dtype=np.float64)
        tangent = np.asarray(row["tangent_prediction"], dtype=np.float64)
        random = np.asarray(row["random_prediction"], dtype=np.float64)
        gradient = np.asarray(row["gradient_prediction"], dtype=np.float64)
        tangent_cosine = p1194.cosine(tangent, target)
        random_cosine = p1194.cosine(random, target)
        gradient_cosine = p1194.cosine(gradient, target)
        null = max(random_cosine, gradient_cosine)
        maximum = max(
            maximum,
            abs(float(np.linalg.norm(target)) - float(row["target_norm"])),
            abs(tangent_cosine - float(row["tangent_cosine"])),
            abs(random_cosine - float(row["random_cosine"])),
            abs(gradient_cosine - float(row["gradient_cosine"])),
            abs(null - float(row["conservative_null_cosine"])),
            abs(tangent_cosine - null - float(row["tangent_advantage"])),
        )
    return maximum


def rescue_metric_error(rows: list[dict[str, Any]]) -> float:
    maximum = 0.0
    null_names = ("wrong_component", "wrong_time", "wrong_task", "random")
    for row in rows:
        if row["stage"] != p1194.RESCUE_STAGE:
            continue
        variants = row["rescue_variants"]
        control_error = float(variants["control"]["response_error"])
        recoveries = {}
        for name, metrics in variants.items():
            recovery = (control_error - float(metrics["response_error"])) / max(control_error, 1e-12)
            recoveries[name] = recovery
            maximum = max(maximum, abs(recovery - float(metrics["response_recovery"])))
        null = max(recoveries[name] for name in null_names)
        advantage = recoveries["correct"] - null
        maximum = max(
            maximum,
            abs(control_error - float(row["rescue_control_error"])),
            abs(recoveries["correct"] - float(row["rescue_correct_recovery"])),
            abs(null - float(row["rescue_null_recovery"])),
            abs(advantage - float(row["rescue_advantage"])),
        )
    return maximum


def replay_error(reference: dict[str, Any], replay: dict[str, Any]) -> float:
    scalar_keys = (
        "event_loss",
        "parent_accuracy",
        "child_accuracy",
        "update_norm",
        "gradient_norm",
        "target_norm",
        "tangent_cosine",
        "random_cosine",
        "gradient_cosine",
        "conservative_null_cosine",
        "tangent_advantage",
    )
    errors = [abs(float(reference[key]) - float(replay[key])) for key in scalar_keys]
    for key in (
        "target_calibration",
        "target_evaluation",
        "tangent_prediction",
        "random_prediction",
        "gradient_prediction",
    ):
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
    protocol = p1194.read_json(p1194.PROTOCOL_PATH)
    seal = p1194.read_json(p1194.TRAINING_SEAL)
    rows = p1194.read_jsonl(p1194.RAW_ROWS)
    summary = p1194.read_json(p1194.SUMMARY_PATH)
    claims = p1194.read_json(p1194.CLAIMS_PATH)

    add(
        checks,
        "protocol_digest",
        digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
        == protocol["protocol_digest"],
    )
    add(checks, "source_hashes", protocol["source_hashes"] == p1194.source_hashes())
    add(
        checks,
        "upstream_final_hash",
        p1194.file_sha256(p1194.p1193.FINAL_PATH)
        == protocol["upstream"]["phase1193_final_sha256"],
    )
    add(
        checks,
        "development_rows_hash",
        p1194.file_sha256(p1194.DEVELOPMENT_ROWS)
        == protocol["upstream"]["development_rows_sha256"],
    )
    add(
        checks,
        "development_summary_hash",
        p1194.file_sha256(p1194.DEVELOPMENT_SUMMARY)
        == protocol["upstream"]["development_summary_sha256"],
    )
    add(
        checks,
        "seal_digest",
        digest({key: value for key, value in seal.items() if key != "seal_digest"})
        == seal["seal_digest"],
    )
    row_manifest = {
        path.name: p1194.file_sha256(path) for path in sorted(p1194.FORMAL_ROW_ROOT.glob("*.json"))
    }
    replay_manifest = {
        path.name: p1194.file_sha256(path) for path in sorted(p1194.REPLAY_ROOT.glob("*.pt"))
    }
    add(checks, "row_manifest", row_manifest == seal["row_manifest"], len(row_manifest))
    add(checks, "replay_manifest", replay_manifest == seal["replay_manifest"], len(replay_manifest))
    add(
        checks,
        "analysis_rows_hash",
        p1194.file_sha256(p1194.RAW_ROWS) == seal["analysis_rows_sha256"],
    )

    add(checks, "row_count", len(rows) == 216, len(rows))
    add(checks, "trajectory_count", len({row["trajectory_id"] for row in rows}) == 72)
    add(checks, "event_uniqueness", len({row["event_id"] for row in rows}) == 216)
    add(checks, "task_count", len({row["task_name"] for row in rows}) == 12)
    add(checks, "architecture_set", {row["architecture"] for row in rows} == set(p1194.ARCHITECTURES))
    add(checks, "stage_set", {row["stage"] for row in rows} == set(p1194.STAGES))
    add(checks, "family_set", {row["family"] for row in rows} == {"affine", "bitmix", "random"})
    add(
        checks,
        "split_counts",
        {split: sum(row["split"] == split for row in rows) for split in ("discovery", "confirmation")}
        == {"discovery": 108, "confirmation": 108},
    )
    add(
        checks,
        "rescue_row_count",
        sum(row["stage"] == p1194.RESCUE_STAGE for row in rows) == 72,
    )
    add(
        checks,
        "development_formal_disjoint",
        not ({task["task_seed"] for task in p1194.DEVELOPMENT_TASKS} & {task["task_seed"] for task in p1194.FORMAL_TASKS}),
    )

    dimensions = {
        architecture: config.layers * (config.heads + 1)
        for architecture, config in p1194.ARCHITECTURES.items()
    }
    vector_keys = (
        "target_calibration",
        "target_evaluation",
        "tangent_prediction",
        "random_prediction",
        "gradient_prediction",
    )
    add(
        checks,
        "architecture_specific_dimensions",
        all(
            row["response_dimension"] == dimensions[row["architecture"]]
            and all(len(row[key]) == dimensions[row["architecture"]] for key in vector_keys)
            for row in rows
        ),
        dimensions,
    )
    add(
        checks,
        "finite_vectors",
        all(np.isfinite(np.asarray(row[key], dtype=np.float64)).all() for row in rows for key in vector_keys),
    )
    prediction_error = prediction_metric_error(rows)
    add(checks, "prediction_metrics_recompute", prediction_error <= 1e-12, prediction_error)
    rescue_error = rescue_metric_error(rows)
    add(checks, "rescue_metrics_recompute", rescue_error <= 1e-12, rescue_error)

    eligibility_errors = 0
    for row in rows:
        expected_prediction = bool(
            np.isfinite(np.asarray(row["target_evaluation"], dtype=np.float64)).all()
            and np.isfinite(np.asarray(row["tangent_prediction"], dtype=np.float64)).all()
            and float(row["target_norm"]) >= p1194.CONTROL_THRESHOLDS["target_norm_min"]
        )
        if expected_prediction != row["prediction_eligible"]:
            eligibility_errors += 1
        if row["stage"] == p1194.RESCUE_STAGE:
            expected_rescue = bool(
                row["rescue_control_error"] >= p1194.CONTROL_THRESHOLDS["rescue_control_error_min"]
                and row["patch_parameter_fraction"]
                <= p1194.CONTROL_THRESHOLDS["patch_parameter_fraction_max"]
                and row["patch_update_fraction"]
                <= p1194.CONTROL_THRESHOLDS["patch_update_fraction_max"]
            )
            if expected_rescue != row["rescue_eligible"]:
                eligibility_errors += 1
    add(checks, "eligibility_recompute", eligibility_errors == 0, eligibility_errors)

    task_map_errors = 0
    for split in ("discovery", "confirmation"):
        task_indices = sorted({row["task_index"] for row in rows if row["split"] == split})
        lookup = {
            (row["task_index"], row["architecture"], row["replicate"]): row
            for row in rows
            if row["split"] == split and row["stage"] == p1194.RESCUE_STAGE
        }
        for row in lookup.values():
            next_task = task_indices[(task_indices.index(row["task_index"]) + 1) % len(task_indices)]
            expected = lookup[(next_task, row["architecture"], row["replicate"])]["trajectory_id"]
            if row["wrong_task_trajectory_id"] != expected:
                task_map_errors += 1
    add(checks, "wrong_task_mapping", task_map_errors == 0, task_map_errors)

    for split in ("discovery", "confirmation"):
        recomputed = p1194.summarize(rows, split)
        add(checks, split + ".summary_recompute", recomputed == summary[split])
    expected_prediction = (
        summary["discovery"]["prediction_gate_pass"]
        and summary["confirmation"]["prediction_gate_pass"]
    )
    expected_rescue = (
        summary["discovery"]["rescue_gate_pass"]
        and summary["confirmation"]["rescue_gate_pass"]
    )
    add(
        checks,
        "independent_decisions",
        summary["prediction_decision"] == ("positive" if expected_prediction else "not_confirmed")
        and summary["rescue_decision"] == ("positive" if expected_rescue else "not_confirmed"),
    )
    add(checks, "typed_claims_recompile", p1194.compile_claims(summary) == claims)
    add(checks, "typed_claims_accepted", all(claim["accepted"] for claim in claims.values()))
    add(
        checks,
        "continuation_rule_not_bypassed",
        protocol["continuation_rule"].startswith("Self-consistent optimizer continuation is authorized only if prediction and rescue pass"),
    )

    replay_details = []
    if not torch.cuda.is_available():
        add(checks, "cuda_replay", False, "CUDA unavailable")
    else:
        device = torch.device("cuda")
        lookup = {row["event_id"]: row for row in rows}
        for path in sorted(p1194.REPLAY_ROOT.glob("*.pt")):
            capsule = torch.load(path, map_location="cpu", weights_only=False)
            event_id = f"{capsule['task']['name']}::{capsule['architecture']}::r{capsule['replicate']}::s{capsule['stage']}"
            replay = p1194.replay_capsule(path, device)
            error = replay_error(lookup[event_id], replay)
            replay_details.append({"event_id": event_id, "max_error": error})
        add(
            checks,
            "cuda_replay",
            len(replay_details) == 12 and all(item["max_error"] <= 1e-6 for item in replay_details),
            replay_details,
        )

    sentinel = max(rows, key=lambda row: abs(float(row["tangent_advantage"])))
    target = np.asarray(sentinel["target_evaluation"], dtype=np.float64)
    prediction = np.asarray(sentinel["tangent_prediction"], dtype=np.float64)
    corruption = abs(p1194.cosine(-prediction, target) - float(sentinel["tangent_cosine"]))
    add(checks, "sign_corruption_positive_sentinel", corruption >= 0.5, corruption)

    audit = {
        "phase": p1194.PHASE,
        "audit_kind": "independent_digest_split_vector_null_rescue_type_and_cuda_replay",
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "gate_pass": all(check["pass"] for check in checks),
        "audit_digest": None,
    }
    audit["audit_digest"] = digest({key: value for key, value in audit.items() if key != "audit_digest"})
    p1194.write_json(p1194.AUDIT_PATH, audit)
    print(canonical({"check_count": audit["check_count"], "pass_count": audit["pass_count"], "gate_pass": audit["gate_pass"]}))
    if not audit["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
