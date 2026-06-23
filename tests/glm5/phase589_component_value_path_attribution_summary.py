#!/usr/bin/env python3
"""Summarize Phase 589 component-level value path attribution."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase589_component_value_path_attribution")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load(model: str):
    return json.loads((ROOT / f"phase589_{model}_component_value_path_attribution_confirm.json").read_text(encoding="utf-8"))


def best_repair(data):
    items = [v for v in data["summary"]["by_key"].values() if v["mode"] == "repair_delta"]
    return max(items, key=lambda x: (x["target_n"], x["target_switch_rate"], x["mean_margin_gain"], x["mean_correct_gain"]))


def main() -> None:
    lines = [
        "# Phase 589 Component-Level Value Path Attribution Summary",
        "",
        "Confirm setting: 24 value cases per model, prompt_last component-output patch at two late layers.",
        "",
        "| model | target cases | best component | layer | switch | correct gain | top-wrong gain | margin gain |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = load(model)
        item = best_repair(data)
        lines.append(
            f"| {model} | {item['target_n']} | {item['component']} | L{item['layer']} | "
            f"{item['target_switch']}/{item['target_n']} ({pct(item['target_switch_rate'])}) | "
            f"{item['mean_correct_gain']:.3f} | {item['mean_top_wrong_gain']:.3f} | {item['mean_margin_gain']:.3f} |"
        )

    ds = load("deepseek7b")
    lines += ["", "## DS7B Component Details", "", "| component | layer | switch | correct gain | top-wrong gain | margin gain |", "|---|---:|---:|---:|---:|---:|"]
    for key, item in sorted(ds["summary"]["by_key"].items()):
        if item["mode"] != "repair_delta":
            continue
        lines.append(
            f"| {item['component']} | L{item['layer']} | {item['target_switch']}/{item['target_n']} ({pct(item['target_switch_rate'])}) | "
            f"{item['mean_correct_gain']:.3f} | {item['mean_top_wrong_gain']:.3f} | {item['mean_margin_gain']:.3f} |"
        )

    lines += [
        "",
        "## Objective Facts",
        "",
        "- DS7B residual output carries the strongest value candidate co-activation: L26 correct +6.205, top-wrong +6.254, margin -0.049.",
        "- DS7B attention output also co-activates candidates, especially L26: correct +1.835, top-wrong +1.863.",
        "- DS7B MLP output does not improve margin and is weak/negative in this patch setup.",
        "- No component produces winner switch or positive margin control on DS7B.",
        "- Therefore candidate co-activation is visible at component level, but winner selection is still unresolved.",
        "",
    ]
    out = ROOT / "phase589_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
