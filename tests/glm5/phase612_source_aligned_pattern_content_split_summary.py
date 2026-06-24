#!/usr/bin/env python3
"""Summarize Phase 612 source-aligned pattern/content split."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase612_source_aligned_pattern_content_split")
MODELS = ["qwen3", "glm4", "deepseek7b"]
MODES = ["actual", "rr_pattern_content", "rb_pattern", "br_content", "bb", "random_actual_norm"]


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 612 Cross Model Summary",
        "",
        "Source-aligned strict pattern/content split. Prompts are filtered to equal token length.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase612_{model}_source_aligned_pattern_content_split_confirm.json"
        if not path.exists():
            lines.append(f"## {model}")
            lines.append("")
            lines.append("missing")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"rows={data['n_rows']}, target_seen={data['n_target_cases_seen']}, "
            f"raw={data['n_raw_cases']}, filtered={data.get('filtered')}, "
            f"layers={data['layers_to_scan']}, top_k={data['top_k']}, "
            f"top_heads={data.get('top_heads')}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        layer = data["layers_to_scan"][0] if data["layers_to_scan"] else None
        for section_name, summary_key in [("all_rows", "summary"), ("target_rows", "target_summary")]:
            lines.append(f"### {section_name}")
            lines.append("")
            lines.append("| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |")
            lines.append("|---|---:|---:|---:|---:|---:|---|")
            by_patch = data.get(summary_key, {}).get("by_patch", {})
            for mode in MODES:
                key = f"L{layer}|top{data['top_k']}|{mode}"
                item = by_patch.get(key)
                if not item:
                    lines.append(f"| `{mode}` | missing | | | | | |")
                    continue
                lines.append(
                    f"| `{mode}` | {item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
                    f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} | "
                    f"{item['positive_margin']}/{item['n']} | `{item['heads']}` |"
                )
            lines.append("")
    out = ROOT / "phase612_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
