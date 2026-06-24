#!/usr/bin/env python3
"""Summarize Phase 610 cumulative head mixture."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase610_head_cumulative_mixture")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def row(item: dict) -> str:
    return (
        f"| `{item['key']}` | L{item['layer']} | {item['name']} | {item['kind']} | "
        f"{item['heads']} | {item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
    )


def main() -> None:
    lines = ["# Phase610 Cross-Model Summary", "", "Cumulative head-slot mixture audit.", ""]
    for model in MODELS:
        path = ROOT / f"phase610_{model}_head_cumulative_mixture_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"layers={data['layers_to_scan']}, heads={data['heads_by_layer']}, "
            f"top_heads={data.get('top_heads')}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Patches")
        lines.append("")
        lines.append("| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:44]:
            lines.append(row(item))
        lines.append("")

        by_patch = data["summary"]["by_patch"]
        lines.append("### Top Cumulative Curve")
        lines.append("")
        lines.append("| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---|---:|---:|---:|---:|")
        for li in data["layers_to_scan"]:
            for name in ["top1_delta", "top2_delta", "top3_delta", "top4_delta", "top6_delta", "all_delta"]:
                item = by_patch.get(f"L{li}|{name}")
                if item:
                    lines.append(row(item))
        lines.append("")

        lines.append("### Controls")
        lines.append("")
        lines.append("| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---|---:|---:|---:|---:|")
        for li in data["layers_to_scan"]:
            for name in [
                "top1_random_slots",
                "top2_random_slots",
                "top3_random_slots",
                "top4_random_slots",
                "top6_random_slots",
                "weak1_delta",
                "weak2_delta",
                "weak3_delta",
                "weak4_delta",
                "weak6_delta",
                "all_random_slots",
            ]:
                item = by_patch.get(f"L{li}|{name}")
                if item:
                    lines.append(row(item))
        lines.append("")
    out = ROOT / "phase610_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
