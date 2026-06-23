#!/usr/bin/env python3
"""Summarize Phase 587 value winner competition audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase587_value_winner_competition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load(model: str):
    path = ROOT / f"phase587_{model}_value_winner_competition_confirm.json"
    return json.loads(path.read_text(encoding="utf-8"))


def best_patch(data):
    items = data["summary"]["by_patch"].items()
    return max(
        items,
        key=lambda kv: (
            kv[1]["target_n"],
            kv[1]["target_switch_rate"],
            kv[1]["mean_margin_gain_target"],
            kv[1]["mean_correct_gain_target"],
        ),
    )


def specific_patch(data, key: str):
    return data["summary"]["by_patch"].get(key)


def main() -> None:
    lines = [
        "# Phase 587 Value Winner Competition Summary",
        "",
        "Confirm setting: 32 value cases per model. Target case means base is wrong and repair prompt is correct.",
        "",
        "| model | target cases | best patch | target switch | correct gain | top-wrong gain | margin gain | correct-up & competitor-up | correct-up but margin<0 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = load(model)
        key, item = best_patch(data)
        lines.append(
            f"| {model} | {data['n_target_cases']} | {key} | "
            f"{item['target_switch']}/{item['target_n']} ({pct(item['target_switch_rate'])}) | "
            f"{item['mean_correct_gain_target']:.3f} | "
            f"{item['mean_top_wrong_gain_target']:.3f} | "
            f"{item['mean_margin_gain_target']:.3f} | "
            f"{item['correct_up_competitor_up']}/{item['target_n']} | "
            f"{item['correct_up_margin_negative']}/{item['target_n']} |"
        )

    ds = load("deepseek7b")
    ds_key = "prompt_last|L21|add_repair"
    ds_item = specific_patch(ds, ds_key)
    if ds_item:
        lines += [
            "",
            "## DS7B Main Diagnostic Patch",
            "",
            f"`{ds_key}`:",
            "",
            f"- target switch: {ds_item['target_switch']}/{ds_item['target_n']} ({pct(ds_item['target_switch_rate'])})",
            f"- mean correct gain: {ds_item['mean_correct_gain_target']:.3f}",
            f"- mean old-top-wrong gain: {ds_item['mean_top_wrong_gain_target']:.3f}",
            f"- mean margin gain: {ds_item['mean_margin_gain_target']:.3f}",
            f"- correct-up and competitor-up: {ds_item['correct_up_competitor_up']}/{ds_item['target_n']}",
            f"- correct-up but final margin negative: {ds_item['correct_up_margin_negative']}/{ds_item['target_n']}",
        ]

    lines += [
        "",
        "## Objective Facts",
        "",
        "- DS7B confirms the Phase586 suspicion: the repair patch raises correct value and top wrong value together.",
        "- DS7B prompt_last L21 add_repair has correct gain +4.782 but old-top-wrong gain +4.718, so margin gain is only +0.063 and target switch remains 0/12.",
        "- This means value-gate failure is not support-only. The missing component is winner-margin control, likely competitor suppression or relation-bound selection.",
        "- Qwen3 has partial switch (2/4) because margin gain is larger relative to its target cases.",
        "- GLM4 target cases remain too few and do not provide a stable value-gate conclusion.",
        "",
    ]
    out = ROOT / "phase587_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
