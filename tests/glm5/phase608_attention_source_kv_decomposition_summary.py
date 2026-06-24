#!/usr/bin/env python3
"""Summarize Phase 608 attention source K/V decomposition."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase608_attention_source_kv_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]
GROUPS = [
    "rule_value",
    "rule_relation",
    "query_relation",
    "query_category",
    "query_object",
    "prompt_last",
    "answer_prefix",
    "random_position",
]
MODES = ["v_delta", "k_delta", "kv_delta", "kv_random"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def line(item: dict) -> str:
    return (
        f"| `{item['key']}` | L{item['layer']} | {item['group']} | {item['mode']} | "
        f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
    )


def main() -> None:
    lines = ["# Phase608 Cross-Model Summary", "", "Attention source-token K/V decomposition.", ""]
    for model in MODELS:
        path = ROOT / f"phase608_{model}_attention_source_kv_decomposition_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, "
            f"target_cases_seen={data['n_target_cases_seen']}, "
            f"layers={data['layers_to_scan']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Patches")
        lines.append("")
        lines.append("| key | layer | group | mode | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:36]:
            lines.append(line(item))
        lines.append("")

        by_patch = data["summary"]["by_patch"]
        lines.append("### Group Mode Grid")
        lines.append("")
        for layer in data["layers_to_scan"]:
            lines.append(f"#### L{layer}")
            lines.append("")
            lines.append("| group | v_delta | k_delta | kv_delta | kv_random |")
            lines.append("|---|---:|---:|---:|---:|")
            for group in GROUPS:
                vals = []
                for mode in MODES:
                    item = by_patch.get(f"L{layer}|{group}|{mode}")
                    if item:
                        vals.append(f"{item['switch']}/{item['n']} ({fmt(item['mean_margin_gain'])})")
                    else:
                        vals.append("")
                lines.append(f"| {group} | " + " | ".join(vals) + " |")
            lines.append("")
    out = ROOT / "phase608_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
