#!/usr/bin/env python3
"""Summarize Phase 588 value candidate-space decomposition."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase588_value_candidate_space_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load(model: str):
    path = ROOT / f"phase588_{model}_value_candidate_space_decomposition_confirm.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get(data, key: str):
    return data["summary"]["by_key"].get(key)


def main() -> None:
    lines = [
        "# Phase 588 Value Candidate Space Decomposition Summary",
        "",
        "Confirm setting: 32 value cases per model. Components are tested at prompt_last/query_relation and two late layers.",
        "",
        "| model | target cases | diagnostic key | switch | correct gain | top-wrong gain | margin gain |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    diag = {
        "qwen3": "prompt_last|L27|full_delta",
        "glm4": "query_relation|L38|full_delta",
        "deepseek7b": "prompt_last|L21|full_delta",
    }
    for model in MODELS:
        data = load(model)
        item = get(data, diag[model])
        lines.append(
            f"| {model} | {data['n_target_cases']} | {diag[model]} | "
            f"{item['target_switch']}/{item['target_n']} ({pct(item['target_switch_rate'])}) | "
            f"{item['mean_correct_gain']:.3f} | {item['mean_top_wrong_gain']:.3f} | "
            f"{item['mean_margin_gain']:.3f} |"
        )

    ds = load("deepseek7b")
    keys = [
        "prompt_last|L21|full_delta",
        "prompt_last|L21|remove_common",
        "prompt_last|L21|common_only",
        "prompt_last|L21|remove_contrast",
        "prompt_last|L21|contrast_only",
        "prompt_last|L21|suppress_top_wrong",
        "prompt_last|L21|boost_minus_suppress",
    ]
    lines += ["", "## DS7B Component Audit", "", "| component | switch | correct gain | top-wrong gain | margin gain |", "|---|---:|---:|---:|---:|"]
    for key in keys:
        item = get(ds, key)
        if not item:
            continue
        lines.append(
            f"| {key} | {item['target_switch']}/{item['target_n']} ({pct(item['target_switch_rate'])}) | "
            f"{item['mean_correct_gain']:.3f} | {item['mean_top_wrong_gain']:.3f} | {item['mean_margin_gain']:.3f} |"
        )

    lines += [
        "",
        "## Objective Facts",
        "",
        "- DS7B full repair delta again raises correct and top-wrong together: +4.782 vs +4.718, switch 0/12.",
        "- Removing the simple unembedding contrast changes nothing on DS7B, so the harmful shared activation is not captured by W(correct)-W(top_wrong).",
        "- Common-only at DS7B L26 reproduces most of the shared gain: correct +4.318, top-wrong +4.313, margin +0.005.",
        "- Simple suppress_top_wrong lowers correct and wrong together and does not improve winner switch.",
        "- Phase588 therefore does not yet produce a controllable suppression patch. It shows the current unembedding-based decomposition is too crude.",
        "",
    ]
    out = ROOT / "phase588_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
