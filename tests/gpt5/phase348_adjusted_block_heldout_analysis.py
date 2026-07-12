#!/usr/bin/env python3
"""Audit Phase348 DS7B heldout phrase and natural-generation effects."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase348_adjusted_block_screen"
PHASE = "Phase348"
SCHEMA_VERSION = "24.0.0"
ROUND_DEFAULT = "adjusted_natural_candidate_block_screen"
TARGET = "no_morphology_control"
CONTROLS = ("sentence_past_tense", "direct_fact_control")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cmean(rows: list[dict[str, Any]], condition: str, field: str) -> float:
    values = [row[field] for row in rows if row["condition"] == condition]
    return mean(values) if values else 0.0


def crate(rows: list[dict[str, Any]], condition: str, field: str) -> float:
    values = [row for row in rows if row["condition"] == condition]
    return sum(bool(row[field]) for row in values) / len(values) if values else 0.0


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    phrase = read_jsonl(root / "models/deepseek7b/phase348_heldout_phrase_rows.jsonl")
    rollout = read_jsonl(root / "models/deepseek7b/phase348_heldout_rollout_rows.jsonl")
    complete = read_json(root / "models/deepseek7b/phase348_heldout_complete.json")
    split_rows = []
    for split in ("heldout", "private_heldout"):
        target_p = [row for row in phrase if row["mechanism_id"] == TARGET and row["split"] == split]
        target_r = [row for row in rollout if row["mechanism_id"] == TARGET and row["split"] == split]
        control_p = [row for row in phrase if row["mechanism_id"] in CONTROLS and row["split"] == split]
        control_r = [row for row in rollout if row["mechanism_id"] in CONTROLS and row["split"] == split]
        zero_loss = cmean(target_p, "correct_zero", "phrase_margin_loss_vs_baseline")
        spatial_loss = max(cmean(target_p, "wrong_depth_zero", "phrase_margin_loss_vs_baseline"), cmean(target_p, "wrong_position_zero", "phrase_margin_loss_vs_baseline"))
        control_loss = max(
            cmean([row for row in control_p if row["mechanism_id"] == task], "correct_zero", "phrase_margin_loss_vs_baseline")
            for task in CONTROLS
        )
        baseline_behavior = crate(target_r, "baseline", "answer_head_semantic_correct")
        target_behavior_loss = crate(target_r, "correct_zero", "behavior_lost_vs_baseline")
        spatial_behavior_loss = max(crate(target_r, "wrong_depth_zero", "behavior_lost_vs_baseline"), crate(target_r, "wrong_position_zero", "behavior_lost_vs_baseline"))
        control_behavior_loss = max(
            crate([row for row in control_r if row["mechanism_id"] == task], "correct_zero", "behavior_lost_vs_baseline")
            for task in CONTROLS
        )
        phrase_gate = bool(zero_loss >= 0.1 and zero_loss - spatial_loss >= 0.05 and zero_loss - control_loss >= 0.05)
        behavior_gate = bool(
            baseline_behavior >= 0.8 and target_behavior_loss >= 0.5
            and spatial_behavior_loss <= 0.2 and control_behavior_loss <= 0.2
        )
        split_rows.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": "deepseek7b", "mechanism_id": TARGET, "split": split,
            "target_zero_phrase_loss": round(zero_loss, 7),
            "max_spatial_phrase_loss": round(spatial_loss, 7),
            "max_matched_control_phrase_loss": round(control_loss, 7),
            "baseline_behavior_success_rate": round(baseline_behavior, 7),
            "target_zero_behavior_loss_rate": round(target_behavior_loss, 7),
            "max_spatial_behavior_loss_rate": round(spatial_behavior_loss, 7),
            "max_matched_control_behavior_loss_rate": round(control_behavior_loss, 7),
            "phrase_gate_pass": phrase_gate, "behavior_gate_pass": behavior_gate,
            "split_gate_pass": phrase_gate and behavior_gate,
        })
    full = all(row["split_gate_pass"] for row in split_rows)
    summary = read_json(root / "phase348_global_summary.json")
    summary["created_at"] = now()
    summary["results"].update({
        "heldout_causal_outcome_revealed": True,
        "heldout_entry_gate_open": False,
        "heldout_revealed_candidate_model_count": 1,
        "heldout_revealed_case_count": complete["case_count"],
        "heldout_remaining_sealed_case_count": summary["denominator"]["sealed_case_count"] - complete["case_count"],
        "heldout_private_full_gate_pass_count": int(full),
        "mcue_entry_gate_open": False,
        "single_model_task_specific_block_candidate_count": int(full),
        "internal_intervention_executed_count": 4113,
    })
    summary["stop_decision"] = (
        "register_model_specific_task_block_but_stop_before_mcue" if full
        else "heldout_failed_stop_before_mcue"
    )
    summary["claim_boundary"] = {
        "cross_model_mechanism_supported": False,
        "single_neuron_supported": False,
        "sufficiency_or_mediation_tested": False,
        "natural_trace_used_private_features_before_causal_reveal": True,
    }
    write_json(root / "phase348_heldout_summary.json", {"complete": complete, "split_results": split_rows, "full_gate_pass": full})
    write_json(root / "phase348_global_summary.json", summary)
    report = (root / "phase348_report.md").read_text(encoding="utf-8").split("\n## Heldout Reveal", 1)[0].rstrip()
    report = report.replace("- Heldout revealed: False", "- Heldout revealed: True (selective DS7B candidate only)")
    report = report.replace("- Stop decision: run_heldout_for_passing_candidates", "- Initial decision: run heldout for the sole passing candidate")
    report += "\n\n## Heldout Reveal\n\n"
    for row in split_rows:
        report += f"- {row['split']}: phrase={row['phrase_gate_pass']}, behavior={row['behavior_gate_pass']}, full={row['split_gate_pass']}\n"
    report += f"- Full heldout/private gate: {full}\n- Stop decision: {summary['stop_decision']}\n"
    (root / "phase348_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
