#!/usr/bin/env python3
"""Summarize Phase 611 semantic pattern/content split."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase611_semantic_pattern_content_split")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def row(item: dict) -> str:
    return (
        f"| `{item['key']}` | L{item['layer']} | {item['mode']} | {item['heads']} | "
        f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
    )


def main() -> None:
    lines = ["# Phase611 Cross-Model Summary", "", "Semantic source-group pattern/content split.", ""]
    for model in MODELS:
        path = ROOT / f"phase611_{model}_semantic_pattern_content_split_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"layers={data['layers_to_scan']}, top_k={data['top_k']}, "
            f"top_heads={data.get('top_heads')}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Patches")
        lines.append("")
        lines.append("| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:32]:
            lines.append(row(item))
        lines.append("")

        lines.append("### Mode Grid")
        lines.append("")
        lines.append("| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|")
        by_patch = data["summary"]["by_patch"]
        for li in data["layers_to_scan"]:
            for mode in ["actual", "content", "pattern", "pattern_content", "random"]:
                key_prefix = f"L{li}|top{data['top_k']}|{mode}"
                item = by_patch.get(key_prefix)
                if item:
                    lines.append(row(item))
        lines.append("")
    out = ROOT / "phase611_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
