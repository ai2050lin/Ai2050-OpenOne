#!/usr/bin/env python3
"""Summarize Phase192 forward dynamics trajectory results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/gpt5_phase192_forward_dynamics_state_transition")
MODELS = ("qwen3", "glm4", "deepseek7b")


def fmt(x) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.4f}"


def compact_position(name: str, row: dict) -> dict:
    return {
        "position": name,
        "n": row["n"],
        "strict_flip_count": row["strict_flip_count"],
        "repair_over_base_count": row["repair_over_base_count"],
        "strict_flip_rate": row["strict_flip_count"] / max(1, row["n"]),
        "repair_over_base_rate": row["repair_over_base_count"] / max(1, row["n"]),
        "mean_repair_over_base_layer": row["mean_repair_over_base_layer"],
        "mean_strict_flip_layer": row["mean_strict_flip_layer"],
        "mean_best_transition_layer": row["mean_best_transition_layer"],
        "mean_best_repair_minus_base": row["mean_best_repair_minus_base"],
        "mean_best_specificity": row["mean_best_specificity"],
        "mean_best_transition_advantage": row["mean_best_transition_advantage"],
        "mean_final_repair_minus_base": row["mean_final_repair_minus_base"],
        "mean_abs_final_wrong_minus_base": row["mean_abs_final_wrong_minus_base"],
        "final_control_leak": row["final_control_leak"],
    }


def load_model(model: str) -> dict:
    path = ROOT / f"phase192_{model}_forward_dynamics_state_transition_confirm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_pos = data["summary"]["by_position"]
    positions = [compact_position(name, row) for name, row in by_pos.items()]
    positions.sort(
        key=lambda x: (
            x["strict_flip_rate"],
            x["repair_over_base_rate"],
            x["mean_best_repair_minus_base"] or -999.0,
            -(x["final_control_leak"] or 999.0),
        ),
        reverse=True,
    )
    low_leak = [p for p in positions if p["final_control_leak"] is not None and p["final_control_leak"] < 1.0]
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
        "low_leak_positions": low_leak,
    }


def position_table(positions: list[dict]) -> list[str]:
    lines = [
        "| position | n | strict flip | over base | best rb | best transition | leak | mean strict layer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in positions:
        lines.append(
            f"| {p['position']} | {p['n']} | "
            f"{p['strict_flip_count']}/{p['n']} | {p['repair_over_base_count']}/{p['n']} | "
            f"{fmt(p['mean_best_repair_minus_base'])} | {fmt(p['mean_best_transition_advantage'])} | "
            f"{fmt(p['final_control_leak'])} | {fmt(p['mean_strict_flip_layer'])} |"
        )
    return lines


def evidence_update(model: dict) -> str:
    best = model["best_position"]
    if not best:
        return "no_trajectory_evidence"
    if best["strict_flip_rate"] >= 0.8 and model["low_leak_positions"]:
        return "trajectory_level4_upgrade_candidate"
    if best["repair_over_base_rate"] >= 0.8:
        return "trajectory_signal_with_control_pollution"
    return "weak_trajectory_signal"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"phase": 192, "models": [load_model(model) for model in MODELS]}
    for model in summary["models"]:
        model["evidence_update"] = evidence_update(model)
    (ROOT / "phase192_cross_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Phase192 Cross-Model Summary",
        "",
        "Objective: after static patch routes failed, measure base/repair/wrong forward trajectories and locate where candidate margins separate across layers.",
        "",
        "The generator produced 192 valid cases under the confirm settings. Rows below are target cases where base was wrong and repair prompt was correct.",
        "",
        "## Model Overview",
        "",
        "| model | cases | target rows | layers | time min | best position | strict flip | over base | best rb | leak | evidence update |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in summary["models"]:
        best = model["best_position"] or {}
        lines.append(
            f"| {model['model']} | {model['n_cases']} | {model['n_target_cases_seen']} | "
            f"{model['n_layers']} | {model['total_time_min']:.2f} | "
            f"{best.get('position', 'NA')} | "
            f"{best.get('strict_flip_count', 0)}/{best.get('n', 0)} | "
            f"{best.get('repair_over_base_count', 0)}/{best.get('n', 0)} | "
            f"{fmt(best.get('mean_best_repair_minus_base'))} | "
            f"{fmt(best.get('final_control_leak'))} | {model['evidence_update']} |"
        )
    for model in summary["models"]:
        lines.extend(["", f"## {model['model']} Positions", ""])
        lines.extend(position_table(model["positions"]))
        low = ", ".join(p["position"] for p in model["low_leak_positions"]) or "none"
        lines.extend(["", f"Low final-control-leak positions: {low}", ""])

    lines.extend([
        "## Objective Reading",
        "",
        "- Phase191's interpretation is correct: static node/channel patch routes should be downgraded, and the next object is the forward trajectory.",
        "- Phase192 shows strong repair-vs-base separation in the natural forward trajectory: repair_over_base is near-total for most positions in all three models.",
        "- This is not hidden causal repair yet. The wrong trajectory often also diverges from base, so control leak remains large at many positions.",
        "- The cleanest observed low-leak handle is Qwen3 prompt_last and DS7B prompt_last. Query/category positions often show strong trajectory signal but heavy control leak.",
        "- Evidence should be updated from static weak Level4 to trajectory-Level4 candidate, not Level5 repair.",
    ])
    (ROOT / "phase192_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    atlas_update = {
        "phase": 192,
        "edge": "candidate-specific ranking repair",
        "old_level": "weak Level4 candidate after static patch failures",
        "new_level": "trajectory-Level4 candidate; do not upgrade to Level5",
        "success_observation": "natural repair trajectories separate from base across layers",
        "remaining_failure": "wrong/control trajectories often also separate; specificity not closed",
        "failure_types": [
            "wrong_trajectory_control_leak",
            "trajectory_signal_not_causal_patch",
            "candidate_specificity_not_closed",
        ],
        "next_gap": "trajectory-localized causal transition test",
    }
    (ROOT / "phase192_atlas_update.json").write_text(
        json.dumps(atlas_update, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    atlas_lines = [
        "# Phase192 Atlas Update",
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
    (ROOT / "phase192_atlas_update.md").write_text("\n".join(atlas_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
