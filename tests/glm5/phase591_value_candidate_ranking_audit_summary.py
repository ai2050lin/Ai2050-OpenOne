#!/usr/bin/env python3
"""Summarize Phase591 value candidate internal ranking audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase591_value_candidate_ranking_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT = ROOT / "phase591_cross_model_summary.md"


def load(model: str) -> dict:
    path = ROOT / f"phase591_{model}_value_candidate_ranking_audit_confirm.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def mode_row(mode: str, item: dict) -> str:
    return (
        f"| {mode} | {item['target_n']} | {item['switch']}/{item['target_n']} | "
        f"{fmt(item['mean_common_delta'])} | {fmt(item['mean_correct_delta'])} | "
        f"{fmt(item['mean_old_top_wrong_delta'])} | {fmt(item['mean_correct_specific'])} | "
        f"{fmt(item['mean_old_top_wrong_specific'])} | {fmt(item['mean_margin_gain_vs_old_top_wrong'])} |"
    )


def model_section(model: str, data: dict) -> list[str]:
    s = data["summary"]
    lines = [f"## {model}", ""]
    lines.append(
        f"- cases={s['n']}, target_cases={s['target_n']}, layers={data['probe_layers']}, alpha={data['alpha']}"
    )
    lines.append("")
    lines.append("| mode | target | switch | common | correct_delta | old_top_wrong_delta | correct_specific | old_top_wrong_specific | margin_gain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = [
        "repair_prompt",
        "patch_prompt_last_residual_attn",
        "patch_query_relation_residual_attn",
        "wrong_relation_prompt",
        "random_prompt_last_residual_attn",
        "random_query_relation_residual_attn",
    ]
    for mode in order:
        if mode in s["by_mode"]:
            lines.append(mode_row(mode, s["by_mode"][mode]))
    lines.append("")
    lines.append("Top-wrong labels on target cases:")
    lines.append("")
    for k, v in sorted(s["target_top_wrong_label_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Top-wrong values on target cases:")
    lines.append("")
    for k, v in sorted(s["target_top_wrong_value_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(
        f"Mean top-wrong embedding cosine to correct: {s['target_mean_top_wrong_embedding_cosine_to_correct']:.3f}"
    )
    lines.append("")
    return lines


def main() -> None:
    models = {m: load(m) for m in MODELS}
    lines = [
        "# Phase591 Cross-Model Summary",
        "",
        "Value candidate internal ranking audit.",
        "",
    ]
    for model in MODELS:
        lines.extend(model_section(model, models[model]))
    lines.append("## Objective facts")
    lines.append("")
    lines.append("- Prompt-level repair creates large candidate-specific support for the correct value in all three models.")
    lines.append("- Hidden residual+attention patch creates mostly common candidate activation, especially in DS7B.")
    lines.append("- DS7B patch_prompt_last_residual_attn: common +6.110, correct_specific +0.036, margin_gain +0.042, switch 0/21.")
    lines.append("- DS7B repair_prompt: correct_specific +5.385, old_top_wrong_specific -1.994, margin_gain +7.379, switch 21/21.")
    lines.append("- DS7B old top-wrong candidates usually have rule-level overlap: same_relation_other_category 19/21, wrong_relation_any_category 20/21, repeated_value 18/21.")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
