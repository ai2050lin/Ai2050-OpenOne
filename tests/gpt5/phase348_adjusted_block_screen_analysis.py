#!/usr/bin/env python3
"""Apply preregistered Phase348 candidate, spatial, and matched-task gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase348_adjusted_block_screen"
PHASE = "Phase348"
SCHEMA_VERSION = "24.0.0"
ROUND_DEFAULT = "adjusted_natural_candidate_block_screen"
MODELS = ("qwen3", "glm4", "deepseek7b")
TARGETS = ("contiguous_multi_token_answer", "no_morphology_control")
SPLITS = ("discovery", "calibration")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def condition_mean(rows: list[dict[str, Any]], condition: str) -> float:
    values = [row["phrase_margin_loss_vs_baseline"] for row in rows if row["condition"] == condition and row["score_valid"]]
    return mean(values) if values else 0.0


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase348_registered_protocol.json")
    registered = read_jsonl(root / "phase348_registered_cases.jsonl")
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    rows = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase348_screen_rows.jsonl")]
    thresholds = protocol["thresholds"]
    result_rows = []
    for model in MODELS:
        for target in TARGETS:
            result = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "mechanism_id": target,
                "frozen_block_id": "attention_output__late__answer_start",
            }
            all_split_pass = []
            for split in SPLITS:
                target_rows = [row for row in rows if row["model"] == model and row["mechanism_id"] == target and row["split"] == split]
                baseline = [row for row in target_rows if row["condition"] == "baseline"]
                baseline_win = sum(row["target_wins"] for row in baseline) / len(baseline)
                zero_rows = [row for row in target_rows if row["condition"] == "correct_zero" and row["score_valid"]]
                zero_loss = condition_mean(target_rows, "correct_zero")
                half_loss = condition_mean(target_rows, "correct_half")
                positive = sum(row["phrase_margin_loss_vs_baseline"] > 0 for row in zero_rows) / len(zero_rows)
                spatial = max(condition_mean(target_rows, "wrong_depth_zero"), condition_mean(target_rows, "wrong_position_zero"))
                control_tasks = protocol["candidate_control_map"][target]
                control_losses = []
                for control in control_tasks:
                    control_rows = [row for row in rows if row["model"] == model and row["mechanism_id"] == control and row["split"] == split]
                    control_losses.append(condition_mean(control_rows, "correct_zero"))
                matched_control = max(control_losses)
                split_pass = bool(
                    baseline_win >= thresholds["split_baseline_target_win_rate_min"]
                    and zero_loss >= thresholds["split_zero_mean_phrase_margin_loss_min"]
                    and positive >= thresholds["split_zero_positive_rate_min"]
                    and zero_loss - spatial >= thresholds["split_spatial_control_superiority_min"]
                    and zero_loss - matched_control >= thresholds["split_matched_task_superiority_min"]
                    and half_loss >= thresholds["split_half_mean_phrase_margin_loss_min"]
                )
                all_split_pass.append(split_pass)
                result.update({
                    f"{split}_baseline_target_win_rate": round(baseline_win, 7),
                    f"{split}_zero_mean_phrase_margin_loss": round(zero_loss, 7),
                    f"{split}_zero_positive_rate": round(positive, 7),
                    f"{split}_half_mean_phrase_margin_loss": round(half_loss, 7),
                    f"{split}_max_spatial_control_loss": round(spatial, 7),
                    f"{split}_max_matched_task_control_loss": round(matched_control, 7),
                    f"{split}_gate_pass": split_pass,
                })
            result["screen_gate_pass"] = all(all_split_pass)
            result["causal_status"] = "coarse_screen_pass" if result["screen_gate_pass"] else "coarse_screen_failed"
            result["single_unit_causal"] = False
            result_rows.append(result)
    passing = [row for row in result_rows if row["screen_gate_pass"]]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered),
            "screen_registered_case_count": sum(row["split"] in SPLITS for row in registered),
            "sealed_case_count": sum(row["split"] not in SPLITS for row in registered),
            "screen_condition_row_count": len(rows),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "invalid_condition_row_count": sum(not row["score_valid"] for row in rows),
            "actual_model_batch_size": 1,
        },
        "results": {
            "candidate_model_count": len(result_rows), "screen_gate_pass_count": len(passing),
            "heldout_causal_outcome_revealed": False,
            "heldout_entry_gate_open": bool(passing),
            "mcue_entry_gate_open": False,
            "internal_intervention_executed_count": len(rows) - len(rows) // 5,
            "single_unit_causal_count": 0, "behavior_mechanism_closed_count": 0,
        },
        "stop_decision": "run_heldout_for_passing_candidates" if passing else "stop_before_heldout_and_mcue",
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(root / "phase348_candidate_screen_summary.jsonl", result_rows)
    write_json(root / "phase348_global_summary.json", summary)
    report = [
        "# Phase348 Adjusted Natural-Candidate Block Screen", "",
        f"- Screen rows: {len(rows)}", f"- Candidate-model gates: {len(passing)}/{len(result_rows)}",
        f"- Heldout revealed: {summary['results']['heldout_causal_outcome_revealed']}",
        f"- Stop decision: {summary['stop_decision']}", "",
    ]
    for row in result_rows:
        report.append(f"- {row['model']} / {row['mechanism_id']}: discovery={row['discovery_gate_pass']}, calibration={row['calibration_gate_pass']}, full={row['screen_gate_pass']}")
    report.extend(["", "No neuron search, sufficiency test, mediation test, or mechanism closure was executed."])
    (root / "phase348_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
