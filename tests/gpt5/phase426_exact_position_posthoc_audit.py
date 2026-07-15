#!/usr/bin/env python3
"""Decompose Phase426 open-set failures without changing frozen gates."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase426_exact_position_role_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")
SIGNALS = ("formation", "transport", "competition")
SPLITS = ("calibration", "behavior_holdout")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite Phase426 posthoc scalar: {value}")
    return round(float(value), 10)


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def behavior_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        for block_id in sorted({row["block_id"] for row in model_rows}):
            block_rows = [row for row in model_rows if row["block_id"] == block_id]
            for split in SPLITS:
                selected = [row for row in block_rows if row["split"] == split]
                output.append(
                    {
                        "model": model,
                        "block_id": block_id,
                        "candidate": bool(selected[0]["candidate"]),
                        "split": split,
                        "independent_group_count": len(selected),
                        "teacher_sequence_correct_fraction_mean": mean(
                            row["early_role_teacher_sequence_correct_fraction"]
                            for row in selected
                        ),
                        "natural_target_fraction_mean": mean(
                            row["early_role_natural_target_fraction"]
                            for row in selected
                        ),
                        "natural_revision_fraction_mean": mean(
                            row["natural_revision_fraction"] for row in selected
                        ),
                        "natural_boundary_fraction_mean": mean(
                            row["natural_boundary_fraction"] for row in selected
                        ),
                        "natural_stop_fraction_mean": mean(
                            row["natural_stop_fraction"] for row in selected
                        ),
                        "natural_censoring_fraction_mean": mean(
                            row["natural_censoring_fraction"] for row in selected
                        ),
                    }
                )
    return output


def candidate_decomposition(audit: dict[str, Any]) -> dict[str, Any]:
    failed = []
    for signal in SIGNALS:
        if not audit["signals"][signal]["calibration_and_behavior_gate_pass"]:
            failed.append(signal)
    if not audit["identity_map_gate_pass"]:
        failed.append("source_to_write_identity")
    if not audit["behavior"]["calibration"]["teacher_gate_pass"] or not audit[
        "behavior"
    ]["behavior_holdout"]["teacher_gate_pass"]:
        failed.append("teacher_sequence_behavior")
    if not audit["behavior"]["calibration"]["natural_gate_pass"] or not audit[
        "behavior"
    ]["behavior_holdout"]["natural_gate_pass"]:
        failed.append("natural_generation_behavior")
    if not audit["prediction"]["gate_pass"]:
        failed.append("joint_event_prediction")
    if not audit["partial_order_gate_pass"]:
        failed.append("partial_order")

    signals = {}
    for signal in SIGNALS:
        row = audit["signals"][signal]
        signals[signal] = {
            "selected_depth": row["selected_depth"],
            "gate_pass": row["calibration_and_behavior_gate_pass"],
            "calibration_candidate_minus_control": row["split_results"][
                "calibration"
            ]["candidate_minus_control"],
            "behavior_holdout_candidate_minus_control": row["split_results"][
                "behavior_holdout"
            ]["candidate_minus_control"],
            "calibration_role_covariance_median": row["split_results"][
                "calibration"
            ]["role_covariance_median"],
        }
        if signal in {"formation", "transport"}:
            signals[signal].update(
                {
                    "calibration_conditional_covariance_median": row[
                        "split_results"
                    ]["calibration"]["conditional_covariance_median"],
                    "calibration_replica_signal_ratio_median": row[
                        "split_results"
                    ]["calibration"]["replica_signal_ratio_median"],
                }
            )

    prediction = {}
    for target, target_row in audit["prediction"]["targets"].items():
        prediction[target] = {
            "gate_pass": target_row["gate_pass"],
            "splits": {
                split: {
                    "gate_pass": target_row["split_results"][split]["gate_pass"],
                    "physical_r2": target_row["split_results"][split]["physical"][
                        "r2"
                    ],
                    "delta_r2": target_row["split_results"][split]["delta_r2"],
                    "mae_gain": target_row["split_results"][split]["mae_gain"],
                }
                for split in SPLITS
            },
        }

    return {
        "model": audit["model"],
        "block_id": audit["block_id"],
        "matched_control_block_id": audit["matched_control_block_id"],
        "selected_depths": audit["selected_depths"],
        "failed_frozen_gates": failed,
        "signal_audit": signals,
        "source_to_write_identity": audit["source_to_write_identity"],
        "role_dominance_descriptive_only": audit[
            "role_dominance_descriptive_only"
        ],
        "behavior": audit["behavior"],
        "prediction": prediction,
        "partial_order_gate_pass": audit["partial_order_gate_pass"],
        "open_path_gate_pass": audit["open_path_gate_pass"],
    }


def main() -> None:
    gate_freeze = read_json(OUT / "phase426_gate_freeze.json")
    global_summary = read_json(OUT / "phase426_global_summary.json")
    audits = read_jsonl(OUT / "phase426_open_candidate_audits.jsonl")
    registered = read_jsonl(OUT / "phase426_registered_conditions_open.jsonl")
    group_rows = []
    collection_rows = []
    for model in MODELS:
        model_dir = OUT / "models" / model / "open"
        group_rows.extend(read_jsonl(model_dir / "phase426_group_summary_rows.jsonl"))
        collection_rows.append(read_json(model_dir / "phase426_collection_complete.json"))

    decompositions = [candidate_decomposition(audit) for audit in audits]
    failure_counts = Counter(
        failure
        for row in decompositions
        for failure in row["failed_frozen_gates"]
    )
    signal_pass_counts = {
        signal: sum(
            row["signal_audit"][signal]["gate_pass"] for row in decompositions
        )
        for signal in SIGNALS
    }
    target_pass_counts = Counter()
    for row in decompositions:
        for target, target_row in row["prediction"].items():
            if target_row["gate_pass"]:
                target_pass_counts[target] += 1

    exact_position_failures = int(
        global_summary["exact_position_mismatch_count"]
    )
    summary = {
        "schema_version": "phase426_exact_position_posthoc.v1",
        "phase_id": "Phase426-PosthocFailureDecomposition",
        "created_at": now(),
        "posthoc": True,
        "changes_frozen_gate": False,
        "sealed_data_read": False,
        "causal_claim": False,
        "strict_human_double_blind": False,
        "registered_open_condition_count": len(registered),
        "exact_position_contract_failure_count": exact_position_failures,
        "independent_candidate_model_block_count": len(decompositions),
        "failed_gate_counts_over_6_candidate_model_blocks": dict(
            sorted(failure_counts.items())
        ),
        "signal_pass_counts_over_6_candidate_model_blocks": signal_pass_counts,
        "joint_open_path_pass_count": sum(
            row["open_path_gate_pass"] for row in decompositions
        ),
        "single_target_prediction_pass_counts": dict(
            sorted(target_pass_counts.items())
        ),
        "cross_model_open_candidate_count": global_summary[
            "cross_model_open_candidate_count"
        ],
        "sealed_unlock": global_summary["sealed_unlock"],
        "sealed_tested": global_summary["sealed_tested"],
        "causal_tested": global_summary["causal_tested"],
        "strict_mechanism_closure": global_summary["strict_mechanism_closure"],
        "frozen_thresholds": gate_freeze["thresholds"],
        "collection_audit": collection_rows,
        "behavior_by_model_block_split": behavior_summary(group_rows),
        "candidate_decompositions": decompositions,
        "hard_limits": [
            "Exact token positions remove the Phase425 distance mismatch but do not remove all prompt-template and tag-timing effects.",
            "Candidate-control signal gaps are not uniformly positive across blocks and models.",
            "The frozen identity source-to-write alignment is near zero or negative for every candidate; no learned map was fitted posthoc.",
            "No candidate reaches the frozen teacher-sequence or natural-generation qualification thresholds in both open evaluation splits.",
            "Only the DeepSeek7B language-action natural target predictor passes as an isolated target; the teacher target and cross-model gates fail.",
            "The experiment uses synthetic tasks and three small models, so external validity to larger models and natural language remains unknown.",
        ],
        "strict_conclusion": (
            "Phase426 repairs the position-control defect and preserves local candidate-specific "
            "formation or transport signals in some model-blocks. It does not connect source "
            "state to legal attention write, jointly predict teacher-forced and natural events, "
            "or reproduce a complete path across models. The sealed, causal, head, channel, "
            "and neuron stages therefore remain closed."
        ),
    }
    write_json(OUT / "phase426_posthoc_failure_decomposition.json", summary)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key
                not in {
                    "candidate_decompositions",
                    "behavior_by_model_block_split",
                    "collection_audit",
                    "frozen_thresholds",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
