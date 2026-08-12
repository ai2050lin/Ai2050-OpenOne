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
import phase1189_quotient_formation_operator_calibration as p1189  # noqa: E402


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def vector_metrics(row: dict[str, Any]) -> dict[str, float]:
    calibration = np.asarray(row["calibration_delta"], dtype=np.float64)
    evaluation = np.asarray(row["evaluation_delta"], dtype=np.float64)
    denominator = max(float(np.linalg.norm(calibration) * np.linalg.norm(evaluation)), 1e-12)
    return {
        "calibration_norm": float(np.linalg.norm(calibration)),
        "evaluation_norm": float(np.linalg.norm(evaluation)),
        "wasserstein2_calibration": float(np.linalg.norm(calibration) / math.sqrt(len(calibration))),
        "wasserstein2_evaluation": float(np.linalg.norm(evaluation) / math.sqrt(len(evaluation))),
        "cosine": float(np.dot(calibration, evaluation) / denominator),
        "relative_error": float(
            np.linalg.norm(calibration - evaluation) / max(float(np.linalg.norm(evaluation)), 1e-12)
        ),
    }


def independent_system_pass(row: dict[str, Any]) -> bool:
    t = p1189.THRESHOLDS
    conditions = [
        row["base_expansion_logit_error"] <= t["logit_equivalence_max"],
        row["positive_expected_logit_error"] <= t["logit_equivalence_max"],
        row["control_expected_logit_error"] <= t["logit_equivalence_max"],
        row["positive_control_logit_error"] <= t["logit_equivalence_max"],
        row["positive_prediction_agreement"] >= t["prediction_agreement_min"],
        row["control_prediction_agreement"] >= t["prediction_agreement_min"],
        abs(row["positive_calibration_loss"] - row["control_calibration_loss"])
        <= t["loss_pair_difference_max"],
        abs(row["positive_evaluation_loss"] - row["control_evaluation_loss"])
        <= t["loss_pair_difference_max"],
        row["update_norm_relative_error"] <= t["update_norm_relative_error_max"],
        row["positive_control_parameter_norm_relative_gap"] <= t["parameter_norm_relative_gap_max"],
        row["positive"]["calibration_norm"] >= t["positive_calibration_transition_norm_min"],
        row["positive"]["evaluation_norm"] >= t["positive_evaluation_transition_norm_min"],
        row["control"]["calibration_norm"] <= t["control_transition_norm_max"],
        row["control"]["evaluation_norm"] <= t["control_transition_norm_max"],
        row["positive"]["calibration_to_evaluation_cosine"] >= t["positive_transfer_cosine_min"],
        row["positive"]["calibration_to_evaluation_relative_error"]
        <= t["positive_transfer_relative_error_max"],
        row["positive"]["gauge_calibration_max_error"] <= t["gauge_transition_error_max"],
        row["positive"]["gauge_evaluation_max_error"] <= t["gauge_transition_error_max"],
        row["control"]["gauge_calibration_max_error"] <= t["gauge_transition_error_max"],
        row["control"]["gauge_evaluation_max_error"] <= t["gauge_transition_error_max"],
        bool(row["positive"]["classified_positive"]),
        not bool(row["control"]["classified_positive"]),
    ]
    return all(conditions)


def independent_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    tasks = sorted({row["task_name"] for row in selected})
    classification_correct = sum(
        int(row["positive"]["classified_positive"]) + int(not row["control"]["classified_positive"])
        for row in selected
    )
    return {
        "system_count": len(selected),
        "task_count": len(tasks),
        "task_names": tasks,
        "system_pass_count": sum(independent_system_pass(row) for row in selected),
        "classification_correct": classification_correct,
        "classification_total": 2 * len(selected),
        "classification_accuracy": classification_correct / max(2 * len(selected), 1),
        "positive_calibration_norm_min": min(row["positive"]["calibration_norm"] for row in selected),
        "positive_evaluation_norm_min": min(row["positive"]["evaluation_norm"] for row in selected),
        "control_transition_norm_max": max(
            max(row["control"]["calibration_norm"], row["control"]["evaluation_norm"])
            for row in selected
        ),
        "positive_transfer_cosine_min": min(
            row["positive"]["calibration_to_evaluation_cosine"] for row in selected
        ),
        "positive_transfer_relative_error_max": max(
            row["positive"]["calibration_to_evaluation_relative_error"] for row in selected
        ),
        "gauge_transition_error_max": max(
            max(
                row[k][field]
                for k in ("positive", "control")
                for field in ("gauge_calibration_max_error", "gauge_evaluation_max_error")
            )
            for row in selected
        ),
    }


def compare_replay(observed: dict[str, Any], replayed: dict[str, Any]) -> float:
    scalar_paths = [
        "base_expansion_logit_error",
        "positive_expected_logit_error",
        "control_expected_logit_error",
        "positive_control_logit_error",
        "positive_prediction_agreement",
        "control_prediction_agreement",
        "positive_update_norm",
        "control_update_norm",
        "update_norm_relative_error",
        "positive_control_parameter_norm_relative_gap",
    ]
    errors = [abs(float(observed[key]) - float(replayed[key])) for key in scalar_paths]
    for branch in ("positive", "control"):
        for key in (
            "calibration_norm",
            "evaluation_norm",
            "calibration_to_evaluation_cosine",
            "calibration_to_evaluation_relative_error",
            "gauge_calibration_max_error",
            "gauge_evaluation_max_error",
        ):
            errors.append(abs(float(observed[branch][key]) - float(replayed[branch][key])))
        for key in ("calibration_delta", "evaluation_delta"):
            errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(observed[branch][key], dtype=np.float64)
                            - np.asarray(replayed[branch][key], dtype=np.float64)
                        )
                    )
                )
            )
    return max(errors)


def run_audit() -> None:
    checks: list[dict[str, Any]] = []
    protocol = p1189.read_json(p1189.PROTOCOL_PATH)
    summary = p1189.read_json(p1189.SUMMARY_PATH)
    claims = p1189.read_json(p1189.CLAIMS_PATH)
    rows = p1189.read_jsonl(p1189.RAW_ROWS)

    protocol_without_digest = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    add_check(checks, "protocol_digest", p1189.digest(protocol_without_digest) == protocol["protocol_digest"])
    add_check(checks, "source_hashes", p1189.source_hashes() == protocol["source_hashes"])
    add_check(
        checks,
        "formal_manifest",
        p1189.checkpoint_manifest(p1189.endpoint_paths(p1189.FORMAL_SOURCE))
        == protocol["formal"]["checkpoint_manifest"],
    )
    add_check(checks, "raw_sha256", p1189.file_sha256(p1189.RAW_ROWS) == summary["raw_rows_sha256"])
    add_check(checks, "row_count", len(rows) == p1189.EXPECTED_SYSTEMS, len(rows))
    add_check(checks, "unique_checkpoints", len({row["checkpoint"] for row in rows}) == len(rows))
    add_check(checks, "task_count", len({row["task_name"] for row in rows}) == p1189.EXPECTED_TASKS)
    add_check(checks, "split_counts", {s: sum(row["split"] == s for row in rows) for s in ("discovery", "confirmation")} == {"discovery": 32, "confirmation": 32})
    add_check(checks, "finite_scalars", all(math.isfinite(float(value)) for row in rows for value in (
        row["base_expansion_logit_error"],
        row["positive_expected_logit_error"],
        row["control_expected_logit_error"],
        row["positive_control_logit_error"],
        row["update_norm_relative_error"],
        row["positive_control_parameter_norm_relative_gap"],
    )))
    add_check(
        checks,
        "vector_lengths",
        all(
            len(row[branch][key]) == 256
            for row in rows
            for branch in ("positive", "control")
            for key in ("calibration_delta", "evaluation_delta")
        ),
    )
    add_check(
        checks,
        "finite_vectors",
        all(
            np.isfinite(np.asarray(row[branch][key], dtype=np.float64)).all()
            for row in rows
            for branch in ("positive", "control")
            for key in ("calibration_delta", "evaluation_delta")
        ),
    )

    metric_errors = []
    classification_errors = 0
    for row in rows:
        for branch in ("positive", "control"):
            observed = row[branch]
            recomputed = vector_metrics(observed)
            metric_errors.extend(
                [
                    abs(recomputed["calibration_norm"] - observed["calibration_norm"]),
                    abs(recomputed["evaluation_norm"] - observed["evaluation_norm"]),
                    abs(recomputed["wasserstein2_calibration"] - observed["wasserstein2_calibration"]),
                    abs(recomputed["wasserstein2_evaluation"] - observed["wasserstein2_evaluation"]),
                    abs(recomputed["cosine"] - observed["calibration_to_evaluation_cosine"]),
                    abs(recomputed["relative_error"] - observed["calibration_to_evaluation_relative_error"]),
                ]
            )
            expected_class = recomputed["calibration_norm"] >= p1189.CLASSIFICATION_THRESHOLD
            classification_errors += int(bool(observed["classified_positive"]) != expected_class)
    add_check(checks, "vector_metric_recompute", max(metric_errors) <= 1e-12, max(metric_errors))
    add_check(checks, "classification_recompute", classification_errors == 0, classification_errors)

    independent = {split: independent_split(rows, split) for split in ("discovery", "confirmation")}
    for split in ("discovery", "confirmation"):
        expected = summary[split]
        observed = independent[split]
        add_check(checks, split + ".system_count", observed["system_count"] == expected["system_count"])
        add_check(checks, split + ".task_count", observed["task_count"] == expected["task_count"])
        add_check(checks, split + ".system_pass_count", observed["system_pass_count"] == expected["system_pass_count"])
        add_check(checks, split + ".classification", close(observed["classification_accuracy"], expected["classification_accuracy"]))
        add_check(checks, split + ".positive_calibration", close(observed["positive_calibration_norm_min"], expected["positive_calibration_norm_min"]))
        add_check(checks, split + ".positive_evaluation", close(observed["positive_evaluation_norm_min"], expected["positive_evaluation_norm_min"]))
        add_check(checks, split + ".control_null", close(observed["control_transition_norm_max"], expected["control_transition_norm_max"]))
        add_check(checks, split + ".transfer_cosine", close(observed["positive_transfer_cosine_min"], expected["positive_transfer_cosine_min"]))
        add_check(checks, split + ".transfer_error", close(observed["positive_transfer_relative_error_max"], expected["positive_transfer_relative_error_max"]))
        add_check(checks, split + ".gauge", close(observed["gauge_transition_error_max"], expected["gauge_transition_error_max"]))

    contract = p1189.read_json(p1187.CONTRACT_PATH)
    recompiled = {
        name: p1187.compile_claim(raw, contract) for name, raw in claims["raw"].items()
    }
    add_check(checks, "typed_claims_recompile", recompiled == claims["compiled"])
    add_check(checks, "typed_claims_accept", all(item["accepted"] for item in recompiled.values()))
    add_check(checks, "typed_claims_authorize", all(item["authorizes"] for item in recompiled.values()))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for replay")
    path_map = {path.name: path for path in p1189.endpoint_paths(p1189.FORMAL_SOURCE)}
    replay_rows = []
    for task_name in sorted({row["task_name"] for row in rows}):
        observed = next(row for row in rows if row["task_name"] == task_name and row["replicate"] == 0)
        replayed = p1189.build_record(path_map[observed["checkpoint"]], "formal", torch.device("cuda"))
        replay_rows.append({"task_name": task_name, "max_error": compare_replay(observed, replayed)})
        torch.cuda.empty_cache()
    replay_max = max(row["max_error"] for row in replay_rows)
    add_check(checks, "eight_task_cuda_replay", replay_max <= 1e-8, replay_rows)

    first = rows[0]
    first_payload = p1189.load_payload(path_map[first["checkpoint"]])
    device = torch.device("cuda")
    original = p1189.load_model(first_payload, device)
    base = p1189.expand_duplicate_pairs(original, device)
    broken = p1189.clone_model(base, device)
    with torch.no_grad():
        broken.output.weight[:, 0].add_(0.25)
    panel = p1189.panel_from_payload(first_payload)
    sentinel_error = float(
        (p1189.fp32_logits(broken, panel.x, device) - p1189.fp32_logits(base, panel.x, device))
        .abs()
        .max()
        .item()
    )
    add_check(checks, "broken_compensation_positive_sentinel", sentinel_error > p1189.THRESHOLDS["logit_equivalence_max"], sentinel_error)
    del original, base, broken
    torch.cuda.empty_cache()

    add_check(checks, "formal_gate", bool(summary["formal_gate_pass"]))
    add_check(checks, "typed_gate", bool(claims["gate_pass"]))
    gate_pass = all(check["pass"] for check in checks)
    result = {
        "phase": p1189.PHASE,
        "audit_kind": "independent_digest_algebra_type_and_cuda_replay",
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "gate_pass": gate_pass,
        "audit_digest": None,
    }
    result["audit_digest"] = p1189.digest({key: value for key, value in result.items() if key != "audit_digest"})
    p1189.write_json(p1189.AUDIT_PATH, result)
    if not gate_pass:
        raise RuntimeError("independent audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    parser.parse_args()
    run_audit()


if __name__ == "__main__":
    main()
