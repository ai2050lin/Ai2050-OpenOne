#!/usr/bin/env python3
"""Summarize Phase594 conditional transformation atlas results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase594_conditional_transformation_atlas")
MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT = ROOT / "phase594_cross_model_summary.md"


def load(model: str) -> dict:
    with (ROOT / f"phase594_{model}_conditional_transformation_atlas_confirm.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def section(model: str, data: dict) -> list[str]:
    lines = [f"## {model}", ""]
    lines.append(
        f"- target_rows={data['n_target_rows']}, nodes={[(n['position'], n['layer']) for n in data['nodes']]}"
    )
    lines.append("")
    lines.append("| rank | key | value | correct_specific | positive_rate |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, row in enumerate(data["summary"]["best"][:15], start=1):
        if row["source"] == "transition_gain":
            val = row["mean_out_minus_in_specific_margin"]
            cs = row["mean_out_minus_in_correct_specific"]
        else:
            val = row["mean_specific_margin"]
            cs = row["mean_correct_specific"]
        lines.append(
            f"| {i} | {row['key']} | {fmt(val)} | {fmt(cs)} | {row.get('positive_rate', 0):.2f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    models = {m: load(m) for m in MODELS}
    lines = [
        "# Phase594 Cross-Model Summary",
        "",
        "Conditional transformation atlas: incoming/outgoing state, residual update, attention update, and MLP update.",
        "",
    ]
    for model in MODELS:
        lines.extend(section(model, models[model]))
    lines.append("## Objective facts")
    lines.append("")
    lines.append("- Qwen3 top signals are mostly incoming/outgoing state at late prompt_last; component update evidence is weaker.")
    lines.append("- GLM4 has a visible prompt_last L38 transition/residual/MLP update signal, but target rows remain only 4.")
    lines.append("- DS7B rule_value L26 is the strongest transition point: outgoing +1.210, incoming +0.616, transition_gain +0.594.")
    lines.append("- DS7B rule_value L26 MLP update is +1.206 specific_margin, close to outgoing +1.210, suggesting the candidate-specific ranking is largely generated inside that layer update.")
    lines.append("- DS7B query_relation L19 MLP update +0.519 also appears as a non-final relation-path update signal.")
    lines.append("- These remain projection-level transition edges, not causal patch repair, but they are more mechanistically localized than Phase592 residual projection peaks.")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
