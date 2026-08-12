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

import phase1187_typed_evidence_compiler as p1187  # noqa: E402
import phase1190_natural_sgd_quotient_transition as p1190  # noqa: E402


def add(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def independent_event_metrics(rows: list[dict[str, Any]]) -> tuple[float, int]:
    lookup = {
        (row["task_name"], row["replicate"], row["left_step"], row["right_step"]): row
        for row in rows
    }
    maximum_error = 0.0
    mapping_errors = 0
    for row in rows:
        calibration = np.asarray(row["calibration_delta"], dtype=np.float64)
        evaluation = np.asarray(row["evaluation_delta"], dtype=np.float64)
        observed_norms = (float(np.linalg.norm(calibration)), float(np.linalg.norm(evaluation)))
        maximum_error = max(
            maximum_error,
            abs(observed_norms[0] - row["calibration_norm"]),
            abs(observed_norms[1] - row["evaluation_norm"]),
            abs(p1190.cosine(calibration, evaluation) - row["true_cosine"]),
        )
        replicate_null = lookup[
            (
                row["task_name"],
                (row["replicate"] + 1) % p1190.REPLICATES,
                row["left_step"],
                row["right_step"],
            )
        ]
        null_interval = p1190.TIME_NULL[(row["left_step"], row["right_step"])]
        time_null = lookup[(row["task_name"], row["replicate"], *null_interval)]
        replicate_cosine = p1190.cosine(
            calibration, np.asarray(replicate_null["evaluation_delta"], dtype=np.float64)
        )
        time_cosine = p1190.cosine(
            calibration, np.asarray(time_null["evaluation_delta"], dtype=np.float64)
        )
        maximum_error = max(
            maximum_error,
            abs(replicate_cosine - row["replicate_null_cosine"]),
            abs(time_cosine - row["time_null_cosine"]),
            abs((row["true_cosine"] - replicate_cosine) - row["replicate_advantage"]),
            abs((row["true_cosine"] - time_cosine) - row["time_advantage"]),
        )
        expected_eligible = bool(
            observed_norms[0] >= p1190.THRESHOLDS["event_norm_min"]
            and observed_norms[1] >= p1190.THRESHOLDS["event_norm_min"]
        )
        mapping_errors += int(row["replicate_null_trajectory_id"] != replicate_null["trajectory_id"])
        mapping_errors += int(row["time_null_interval"] != time_null["interval"])
        mapping_errors += int(bool(row["eligible"]) != expected_eligible)
    return maximum_error, mapping_errors


def compare_summary(observed: dict[str, Any], expected: dict[str, Any]) -> float:
    keys = (
        "eligible_event_fraction",
        "true_cosine_mean",
        "replicate_null_cosine_mean",
        "time_null_cosine_mean",
        "replicate_null_advantage_mean",
        "time_null_advantage_mean",
        "replicate_positive_fraction",
        "time_positive_fraction",
        "event_norm_mean",
        "event_norm_min",
    )
    errors = [abs(float(observed[key]) - float(expected[key])) for key in keys]
    for key in ("event_count", "eligible_event_count", "system_count", "task_count", "positive_task_count"):
        errors.append(float(observed[key] != expected[key]))
    errors.append(float(bool(observed["gate_pass"]) != bool(expected["gate_pass"])))
    return max(errors)


def compare_replay(observed: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> float:
    observed_map = {(row["left_step"], row["right_step"]): row for row in observed}
    replay_map = {(row["left_step"], row["right_step"]): row for row in replayed}
    errors = []
    for key, left in observed_map.items():
        right = replay_map[key]
        for field in ("calibration_delta", "evaluation_delta"):
            errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(left[field], dtype=np.float64)
                            - np.asarray(right[field], dtype=np.float64)
                        )
                    )
                )
            )
        errors.extend(
            [
                abs(left["calibration_norm"] - right["calibration_norm"]),
                abs(left["evaluation_norm"] - right["evaluation_norm"]),
                abs(left["true_cosine"] - right["true_cosine"]),
            ]
        )
    return max(errors)


def audit() -> None:
    checks: list[dict[str, Any]] = []
    protocol = p1190.read_json(p1190.PROTOCOL_PATH)
    seal = p1190.read_json(p1190.TRAINING_SEAL)
    summary = p1190.read_json(p1190.SUMMARY_PATH)
    claims = p1190.read_json(p1190.CLAIMS_PATH)
    events = p1190.read_jsonl(p1190.EVENT_ROWS)
    behaviors = p1190.read_jsonl(p1190.BEHAVIOR_ROWS)

    add(
        checks,
        "protocol_digest",
        p1190.digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
        == protocol["protocol_digest"],
    )
    add(checks, "source_hashes", p1190.source_hashes() == protocol["source_hashes"])
    add(
        checks,
        "seal_digest",
        p1190.digest({key: value for key, value in seal.items() if key != "seal_digest"})
        == seal["seal_digest"],
    )
    add(checks, "training_metrics_hash", p1190.file_sha256(p1190.TRAINING_METRICS) == seal["training_metrics_sha256"])
    checkpoint_errors = sum(
        p1190.file_sha256(p1190.CHECKPOINT_ROOT / name) != expected
        for name, expected in seal["checkpoint_hashes"].items()
    )
    add(checks, "checkpoint_hashes", checkpoint_errors == 0, checkpoint_errors)
    add(checks, "checkpoint_count", len(seal["checkpoint_hashes"]) == 64 * len(p1190.STEPS))
    add(checks, "event_hash", p1190.file_sha256(p1190.EVENT_ROWS) == summary["event_rows_sha256"])
    add(checks, "behavior_hash", p1190.file_sha256(p1190.BEHAVIOR_ROWS) == summary["behavior_rows_sha256"])
    add(checks, "event_count", len(events) == 64 * len(p1190.INTERVALS), len(events))
    add(checks, "behavior_count", len(behaviors) == 64, len(behaviors))
    add(checks, "event_uniqueness", len({(row["trajectory_id"], row["interval"]) for row in events}) == len(events))
    add(checks, "trajectory_count", len({row["trajectory_id"] for row in events}) == 64)
    add(checks, "task_count", len({row["task_name"] for row in events}) == 8)
    add(
        checks,
        "split_counts",
        {split: len({row["trajectory_id"] for row in events if row["split"] == split}) for split in ("discovery", "confirmation")}
        == {"discovery": 32, "confirmation": 32},
    )
    add(
        checks,
        "vector_lengths",
        all(len(row[field]) == 128 for row in events for field in ("calibration_delta", "evaluation_delta")),
    )
    add(
        checks,
        "finite_vectors",
        all(np.isfinite(np.asarray(row[field], dtype=np.float64)).all() for row in events for field in ("calibration_delta", "evaluation_delta")),
    )
    metric_error, mapping_errors = independent_event_metrics(events)
    add(checks, "event_metric_recompute", metric_error <= 1e-12, metric_error)
    add(checks, "null_mapping_recompute", mapping_errors == 0, mapping_errors)

    for split in ("discovery", "confirmation"):
        independent = p1190.summarize_events(events, split)
        error = compare_summary(independent, summary["events"][split])
        add(checks, split + ".event_summary", error <= 1e-12, error)
        independent_behavior = p1190.summarize_behavior(behaviors, split)
        add(checks, split + ".behavior_summary", independent_behavior == summary["behavior"][split])

    contract = p1190.read_json(p1187.CONTRACT_PATH)
    recompiled = {
        name: p1187.compile_claim(raw, contract) for name, raw in claims["raw"].items()
    }
    add(checks, "typed_claim_recompile", recompiled == claims["compiled"])
    add(checks, "typed_claim_accept", all(claim["accepted"] for claim in recompiled.values()))
    add(checks, "typed_claim_authorize", all(claim["authorizes"] for claim in recompiled.values()))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for replay")
    endpoint_map = {p1190.trajectory_id(p1190.p1189.load_payload(path)): path for path in p1190.formal_endpoints()}
    replay_details = []
    for task_name in sorted({row["task_name"] for row in events}):
        trajectory = next(
            row["trajectory_id"] for row in events if row["task_name"] == task_name and row["replicate"] == 0
        )
        observed = [row for row in events if row["trajectory_id"] == trajectory]
        replayed = p1190.build_transition_vectors([endpoint_map[trajectory]], "formal", torch.device("cuda"))
        error = compare_replay(observed, replayed)
        replay_details.append({"task_name": task_name, "max_error": error})
        torch.cuda.empty_cache()
    replay_max = max(item["max_error"] for item in replay_details)
    add(checks, "eight_task_cuda_replay", replay_max <= 1e-8, replay_details)

    first = events[0]
    original_evaluation = np.asarray(first["evaluation_delta"], dtype=np.float64)
    broken_evaluation = original_evaluation[::-1].copy()
    sentinel_drop = first["true_cosine"] - p1190.cosine(
        np.asarray(first["calibration_delta"], dtype=np.float64), broken_evaluation
    )
    add(checks, "rank_corruption_positive_sentinel", abs(sentinel_drop) >= 0.05, sentinel_drop)
    add(checks, "formal_gate", bool(summary["formal_gate_pass"]))
    add(checks, "typed_gate", bool(claims["gate_pass"]))

    gate_pass = all(check["pass"] for check in checks)
    result = {
        "phase": p1190.PHASE,
        "audit_kind": "independent_digest_vector_mapping_type_and_cuda_replay",
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "gate_pass": gate_pass,
        "audit_digest": None,
    }
    result["audit_digest"] = p1190.digest({key: value for key, value in result.items() if key != "audit_digest"})
    p1190.write_json(p1190.AUDIT_PATH, result)
    if not gate_pass:
        raise RuntimeError("Phase1190 audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    parser.parse_args()
    audit()


if __name__ == "__main__":
    main()
