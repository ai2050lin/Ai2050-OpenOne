#!/usr/bin/env python3
"""Summarize Phase 585 hidden causal patch results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase585_state_gate_hidden_patch")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def best_layer(gate_data):
    items = []
    for layer, stats in gate_data["by_layer"].items():
        target_n = stats["target_n"]
        target_patch = stats["target_patch_accuracy"]
        target_random = stats["target_random_accuracy"]
        patch = stats["patch_accuracy"]
        random = stats["random_accuracy"]
        items.append((target_n, target_patch - target_random, patch - random, layer, stats))
    items.sort(reverse=True)
    return items[0][3], items[0][4]


def main() -> None:
    lines = [
        "# Phase 585 State Gate Hidden Patch Summary",
        "",
        "Confirm setting: value samples=32, polarity negative samples=30, four probe layers per model, alpha=1.0.",
        "",
        "## Best Target-Layer Results",
        "",
        "| model | gate | best layer | base | repair prompt | hidden patch | random control | target patch | target random | target n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        path = ROOT / f"phase585_{model}_state_gate_hidden_patch_confirm.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for gate_key, gate_name in [
            ("value_relation_filter_gate", "value_relation_filter"),
            ("polarity_format_gate", "polarity_format"),
        ]:
            layer, s = best_layer(data[gate_key])
            lines.append(
                f"| {model} | {gate_name} | {layer} | "
                f"{pct(s['base_accuracy'])} | {pct(s['repair_accuracy'])} | "
                f"{pct(s['patch_accuracy'])} | {pct(s['random_accuracy'])} | "
                f"{pct(s['target_patch_accuracy'])} | {pct(s['target_random_accuracy'])} | "
                f"{s['target_n']} |"
            )

    lines += [
        "",
        "## Key Objective Facts",
        "",
        "- Polarity-format hidden patch is strong in middle/late layers for all three models.",
        "- Qwen3 polarity target repair: L27/L34 repaired 3/3 targets while random repaired 0/3.",
        "- GLM4 polarity target repair: L20/L30/L38 repaired 7/7 targets; random repaired 4/7, 1/7, 3/7.",
        "- DS7B polarity target repair: L21 repaired 13/14 and L26 repaired 14/14; random repaired 1/14 and 5/14.",
        "- Value relation-filter prompt repair is strong, but answer_last hidden delta does not transfer on GLM4/DS7B and only weakly transfers on Qwen3.",
        "- Therefore Phase585 supports hidden causal repair for polarity-format gate, but not yet for relation-filter/value gate.",
        "",
    ]
    out = ROOT / "phase585_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
