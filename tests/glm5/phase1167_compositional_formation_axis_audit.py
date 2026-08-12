#!/usr/bin/env python3
"""Independent recomputation audit for Phase1167."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PRIMARY_SCRIPT = ROOT / "tests/glm5/phase1167_compositional_formation_axis.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1167_compositional_formation_axis"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1167_compositional_formation_axis as phase  # noqa: E402


p1163 = phase.p1163


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def audit_command() -> None:
    protocol = phase.verify_protocol()
    training_root = OUT_ROOT / "runs/training"
    holdout_root = OUT_ROOT / "runs/holdout"
    seal = p1163.read_json(training_root / "seal.json")
    holdout_summary = p1163.read_json(holdout_root / "summary.json")
    score = p1163.read_json(OUT_ROOT / "analysis/score.json")
    final = p1163.read_json(OUT_ROOT / "analysis/final.json")
    train_rows = p1163.read_jsonl(training_root / "training_metrics.jsonl")
    holdout_rows = p1163.read_jsonl(holdout_root / "holdout_metrics.jsonl")

    checks: dict[str, bool] = {}
    checks["protocol_digest"] = p1163.digest(
        {key: value for key, value in protocol.items() if key != "protocol_digest"}
    ) == protocol["protocol_digest"]
    checks["primary_script_hash"] = (
        p1163.sha256_file(PRIMARY_SCRIPT) == protocol["source_hashes"]["primary_script"]
    )
    checks["audit_script_hash"] = (
        p1163.sha256_file(SCRIPT) == protocol["source_hashes"]["audit_script"]
    )
    checks["training_metrics_hash"] = (
        p1163.sha256_file(training_root / "training_metrics.jsonl")
        == seal["training_metrics_sha256"]
    )
    checks["holdout_metrics_hash"] = (
        p1163.sha256_file(holdout_root / "holdout_metrics.jsonl")
        == holdout_summary["holdout_metrics_sha256"]
    )
    checks["training_seal_digest"] = p1163.digest(
        {key: value for key, value in seal.items() if key != "seal_digest"}
    ) == seal["seal_digest"]
    checks["holdout_summary_digest"] = p1163.digest(
        {key: value for key, value in holdout_summary.items() if key != "summary_digest"}
    ) == holdout_summary["summary_digest"]
    checks["score_digest"] = p1163.digest(
        {key: value for key, value in score.items() if key != "score_digest"}
    ) == score["score_digest"]
    checks["final_digest"] = p1163.digest(
        {key: value for key, value in final.items() if key != "final_digest"}
    ) == final["final_digest"]

    expected_count = len(phase.ARMS) * len(phase.ARCHITECTURES) * phase.REPLICATES
    checks["training_row_count"] = len(train_rows) == expected_count
    checks["holdout_row_count"] = len(holdout_rows) == expected_count
    checks["model_id_alignment"] = [row["model_id"] for row in train_rows] == [
        row["model_id"] for row in holdout_rows
    ]
    checks["all_fixed_budget"] = all(
        row["fixed_steps_completed"] == phase.TRAINING["fixed_steps"] for row in train_rows
    )
    checks["holdout_absent_at_seal"] = seal["holdout_outcomes_absent_at_sealing"]
    checks["no_holdout_gradient"] = all(
        not row["holdout_used_by_gradient"] for row in train_rows
    )
    checks["no_holdout_training_eval"] = all(
        not row["holdout_evaluated_during_training"] for row in train_rows
    )
    checks["all_training_qualified"] = all(
        row["train"]["accuracy"] >= phase.THRESHOLDS["train_accuracy_min"]
        and row["train"]["finite_fraction"] == 1.0
        for row in train_rows
    )
    checks["all_holdout_finite"] = all(
        row["holdout"]["finite_fraction"] == 1.0 for row in holdout_rows
    )
    checks["checkpoint_hashes"] = all(
        p1163.sha256_file(training_root / "checkpoints" / f"{row['model_id']}.pt")
        == row["checkpoint_sha256"]
        == seal["checkpoint_hashes"][row["model_id"]]
        for row in train_rows
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for independent metric recomputation")
    device = torch.device("cuda")
    recomputed = []
    for train_row, holdout_row in zip(train_rows, holdout_rows):
        checkpoint = training_root / "checkpoints" / f"{train_row['model_id']}.pt"
        model, _, lexicon = phase.load_checkpoint(checkpoint, device)
        data = phase.build_examples(lexicon)
        train_metrics = phase.evaluate(
            model, data["train_x"], data["train_y"], lexicon
        )
        holdout_metrics = phase.evaluate(
            model, data["holdout_x"], data["holdout_y"], lexicon
        )
        recomputed.append(
            all(
                close(train_metrics[key], train_row["train"][key])
                for key in ("accuracy", "minimum_probability", "mean_probability", "finite_fraction")
            )
            and all(
                close(holdout_metrics[key], holdout_row["holdout"][key])
                for key in ("accuracy", "minimum_probability", "mean_probability", "finite_fraction")
            )
        )
        del model
        torch.cuda.empty_cache()
    checks["independent_metric_recompute"] = all(recomputed)

    rows = holdout_rows
    recomputed_cells = {
        arm: {
            split: {
                architecture: phase.cell_summary(rows, arm, split, architecture)
                for architecture in phase.ARCHITECTURES
            }
            for split in ("discovery", "confirmation")
        }
        for arm in phase.ARMS
    }
    checks["cell_recompute"] = recomputed_cells == score["cells"]
    baseline_failure = all(
        recomputed_cells["baseline"][split][architecture]["all_matched_failures"]
        for split in ("discovery", "confirmation")
        for architecture in phase.ARCHITECTURES
    )
    primary_generalizer = all(
        recomputed_cells[phase.PRIMARY_GENERALIZER_ARM][split][architecture][
            "all_generalizers"
        ]
        for split in ("discovery", "confirmation")
        for architecture in phase.ARCHITECTURES
    )
    recomputed_authorization = bool(
        baseline_failure
        and primary_generalizer
        and score["results"]["cell_count_ok"]
        and score["results"]["training_matched"]
    )
    checks["authorization_recompute"] = (
        recomputed_authorization
        == score["results"]["behavior_contrast_authorized"]
        == final["behavior_contrast_authorized"]
        == final["hidden_state_scan_authorized"]
        == final["auto_continue"]
    )
    checks["branch_status_consistent"] = final["branch_status"] == (
        "open_only_for_independent_mechanism_signature_preregistration"
        if recomputed_authorization
        else "closed_after_finite_formation_panel"
    )
    checks["natural_mechanism_not_claimed"] = not final["natural_mechanism_recovered"]
    checks["protocol_source_alignment"] = (
        final["protocol_digest"]
        == score["protocol_digest"]
        == holdout_summary["protocol_digest"]
        == seal["protocol_digest"]
        == protocol["protocol_digest"]
    )

    report = {
        "phase": phase.PHASE,
        "created_at_utc": p1163.now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "recomputed_authorization": recomputed_authorization,
        "primary_generalizer_arm": phase.PRIMARY_GENERALIZER_ARM,
    }
    report["audit_digest"] = p1163.digest(report)
    p1163.write_json(OUT_ROOT / "audit/report.json", report)
    print(p1163.canonical(report))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    parser.parse_args()
    audit_command()


if __name__ == "__main__":
    main()
