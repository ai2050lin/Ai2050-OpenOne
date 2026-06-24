#!/usr/bin/env python3
"""Summarize Phase193 trajectory-localized causal transition tests."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/gpt5_phase193_trajectory_localized_causal_transition")
MODELS = ("qwen3", "glm4", "deepseek7b")


def fmt(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.4f}"


def load_model(model: str) -> dict:
    path = ROOT / f"phase193_{model}_trajectory_localized_causal_transition_confirm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = list(data["summary"]["by_position"].values())
    positions.sort(
        key=lambda x: (x["repair_gain"], x["transition_specificity"], x["ablation_loss"], -abs(x["wrong_gain"])),
        reverse=True,
    )
    for item in positions:
        item["support_level"] = support_level(item)
    return {
        "model": model,
        "source": str(path),
        "n_cases": data["n_cases"],
        "n_target_cases_seen": data["n_target_cases_seen"],
        "n_rows": data["n_rows"],
        "n_layers": data["n_layers"],
        "total_time_min": data["total_time_min"],
        "positions": positions,
        "best_position": positions[0] if positions else None,
    }


def support_level(item: dict) -> str:
    if item["repair_gain"] > 0.05 and abs(item["wrong_gain"]) < item["repair_gain"] and item["ablation_loss"] > 0.05:
        return "localized_transition_candidate"
    if item["repair_gain"] > 0.05 and abs(item["wrong_gain"]) < item["repair_gain"]:
        return "gain_without_ablation"
    if item["ablation_loss"] > 0.05 and item["repair_gain"] <= 0.02:
        return "ablation_only"
    return "weak_or_failed"


def position_table(items: list[dict]) -> list[str]:
    lines = [
        "| position | repair_gain | wrong_gain | ablation_loss | specificity | repair positive | wrong positive | switch | support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in items:
        lines.append(
            f"| {item['position']} | {fmt(item['repair_gain'])} | {fmt(item['wrong_gain'])} | "
            f"{fmt(item['ablation_loss'])} | {fmt(item['transition_specificity'])} | "
            f"{fmt(item['repair_positive_rate'])} | {fmt(item['wrong_positive_rate'])} | "
            f"{item['repair_switch']}/{item['repair_n']} | {item['support_level']} |"
        )
    return lines


def evidence_update(model: dict) -> str:
    levels = [p["support_level"] for p in model["positions"]]
    if "localized_transition_candidate" in levels:
        return "weak_level5_candidate_for_selected_position"
    if "gain_without_ablation" in levels or "ablation_only" in levels:
        return "partial_transition_evidence_not_closed"
    return "trajectory_correlation_not_local_transition"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"phase": 193, "models": [load_model(model) for model in MODELS]}
    for model in summary["models"]:
        model["evidence_update"] = evidence_update(model)
    (ROOT / "phase193_cross_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Phase193 Cross-Model Summary",
        "",
        "Objective: test whether Phase192 trajectory separation is caused by localized layer transitions.",
        "",
        "Rows are target cases where base was wrong and repair prompt was correct. Each position uses the case-local best transition layer plus radius 1.",
        "",
        "## Model Overview",
        "",
        "| model | cases | target rows | layers | time min | best position | repair_gain | wrong_gain | ablation_loss | specificity | evidence update |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in summary["models"]:
        best = model["best_position"] or {}
        lines.append(
            f"| {model['model']} | {model['n_cases']} | {model['n_target_cases_seen']} | "
            f"{model['n_layers']} | {model['total_time_min']:.2f} | {best.get('position', 'NA')} | "
            f"{fmt(best.get('repair_gain'))} | {fmt(best.get('wrong_gain'))} | "
            f"{fmt(best.get('ablation_loss'))} | {fmt(best.get('transition_specificity'))} | "
            f"{model['evidence_update']} |"
        )

    for model in summary["models"]:
        lines.extend(["", f"## {model['model']} Positions", ""])
        lines.extend(position_table(model["positions"]))

    lines.extend([
        "",
        "## Objective Reading",
        "",
        "- The uploaded Phase192 interpretation is correct: trajectory signal required localized causal transition testing.",
        "- Qwen3 prompt_last is the only position meeting the current weak localized-transition candidate rule: repair_gain positive, wrong_gain smaller/opposite, and repair ablation lowers margin.",
        "- GLM4 query_relation has large repair_gain, but wrong_gain is also large and repair->base ablation does not lower repair margin; this is not closed causal evidence.",
        "- DS7B does not convert its strong trajectory signal into localized transition gain. This is important because DS7B had the largest target set.",
        "- Overall: Phase193 upgrades a narrow Qwen3 prompt_last handle, but does not close the cross-model candidate-ranking mechanism.",
    ])
    (ROOT / "phase193_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    atlas_update = {
        "phase": 193,
        "edge": "candidate-specific ranking repair",
        "old_level": "trajectory-Level4 candidate",
        "new_level": "weak localized transition candidate only for qwen3 prompt_last; no cross-model Level5 closure",
        "success_observation": "qwen3 prompt_last base<-repair transition improves margin while wrong transition does not and repair<-base ablation lowers margin",
        "remaining_failure": "GLM4 lacks ablation support; DS7B trajectory signal does not localize into causal transition gain",
        "failure_types": [
            "localized_transition_not_cross_model",
            "glm4_gain_without_ablation",
            "deepseek7b_trajectory_correlation_not_local_transition",
            "generation_winner_switch_rate_still_low",
        ],
        "next_gap": "qwen3_prompt_last_transition_validation_and_ds7b_nonlocal_dynamics_audit",
    }
    (ROOT / "phase193_atlas_update.json").write_text(
        json.dumps(atlas_update, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    atlas_lines = [
        "# Phase193 Atlas Update",
        "",
        f"- edge: {atlas_update['edge']}",
        f"- old_level: {atlas_update['old_level']}",
        f"- new_level: {atlas_update['new_level']}",
        f"- success_observation: {atlas_update['success_observation']}",
        f"- remaining_failure: {atlas_update['remaining_failure']}",
        f"- next_gap: {atlas_update['next_gap']}",
        "",
        "## Failure Types",
    ]
    for failure in atlas_update["failure_types"]:
        atlas_lines.append(f"- {failure}")
    (ROOT / "phase193_atlas_update.md").write_text("\n".join(atlas_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
