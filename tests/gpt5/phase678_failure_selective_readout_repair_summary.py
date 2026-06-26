#!/usr/bin/env python3
"""
Phase 678: Failure-Selective Readout Repair Summary.

Post-processes Phase 677 rows. It simulates a selective gate: keep baseline
success cases untouched, and apply a readout intervention only to baseline
failure cases. No model forward pass is run here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PHASE677_ROOT = Path("results/glm5_phase677_readout_intervention_strength_scan")
OUT_ROOT = Path("results/glm5_phase678_failure_selective_readout_repair_summary")


SELECTED_CONDITIONS = [
    "final_cancel_gap_a1p0",
    "final_cancel_gap_a1p25",
    "final_cancel_gap_a1p5",
    "final_cancel_gap_a2p0",
    "final_remove_comp_a1p5",
    "final_remove_comp_a2p0",
]


def load_rows(model: str) -> list[dict]:
    path = PHASE677_ROOT / f"phase677_{model}_strength_scan_rows.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_model(model: str) -> dict:
    rows = load_rows(model)
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["condition"]] = row

    summaries = {}
    for cond in ["baseline", *SELECTED_CONDITIONS]:
        n = 0
        top1 = 0
        rank_sum = 0.0
        gap_sum = 0.0
        failures = 0
        repaired_failures = 0
        successes = 0
        damaged_successes = 0
        top1_text = {}
        for case_id, conds in by_case.items():
            base = conds["baseline"]
            selected = base
            if cond != "baseline" and not base["expected_top1"]:
                selected = conds[cond]
            n += 1
            top1 += int(selected["expected_top1"])
            rank_sum += selected["expected_rank"]
            gap_sum += selected["gap"]
            failures += int(not base["expected_top1"])
            repaired_failures += int((not base["expected_top1"]) and selected["expected_top1"])
            successes += int(base["expected_top1"])
            damaged_successes += int(base["expected_top1"] and not selected["expected_top1"])
            text = selected["top1_text"].replace("\n", "\\n")
            top1_text[text] = top1_text.get(text, 0) + 1
        summaries[cond] = {
            "condition": cond,
            "n": n,
            "baseline_failures": failures,
            "baseline_successes": successes,
            "selective_expected_top1_rate": top1 / max(1, n),
            "selective_mean_expected_rank": rank_sum / max(1, n),
            "selective_mean_gap": gap_sum / max(1, n),
            "failure_repair_rate": repaired_failures / max(1, failures),
            "success_damage_rate": damaged_successes / max(1, successes),
            "top1_text": dict(sorted(top1_text.items(), key=lambda kv: kv[1], reverse=True)[:8]),
        }
    return {
        "model": model,
        "n_cases": len(by_case),
        "summary": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = [summarize_model(model) for model in args.models]
    result = {
        "phase": 678,
        "title": "Failure-Selective Readout Repair Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase678_selective_repair_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 678 Failure-Selective Readout Repair Summary",
        "",
        f"- generated: `{result['timestamp']}`",
        "",
        "| model | condition | selective_top1 | mean_rank | mean_gap | failure_repair | success_damage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        for cond, row in item["summary"].items():
            lines.append(
                f"| {item['model']} | {cond} | {row['selective_expected_top1_rate']:.3f} | "
                f"{row['selective_mean_expected_rank']:.2f} | {row['selective_mean_gap']:.3f} | "
                f"{row['failure_repair_rate']:.3f} | {row['success_damage_rate']:.3f} |"
            )
    (OUT_ROOT / "phase678_selective_repair_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
