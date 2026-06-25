#!/usr/bin/env python3
"""Summarize Phase 636 prefix competitor ladder audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase636_prefix_competitor_ladder_audit")
MODE_ORDER = [
    "base",
    "repair_prompt",
    "source_all6",
    "final_output_repair",
    "final_output_source",
    "readout_delta",
]
CAT_ORDER = [
    "correct_prefix",
    "newline",
    "punctuation",
    "explanation",
    "old_wrong_prefix",
    "value_prefix",
    "word",
    "number",
    "space",
    "symbol",
    "other",
]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_mode(summary, mode):
    for row in summary["by_mode"]:
        if row["mode"] == mode:
            return row
    return None


def get_cat(summary, mode, category):
    for row in summary["by_mode_category"]:
        if row["mode"] == mode and row["category"] == category:
            return row
    return None


def fmt_counts(d):
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def main() -> None:
    lines = []
    lines.append("# Phase 636 Cross-Model Summary\n")
    lines.append("目标：拆解 token0 correct prefix 被哪些 competitor token 类别压制。\n")
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = OUT_ROOT / f"phase636_{model}_prefix_competitor_ladder_audit_confirm.json"
        data = load_model(path)
        if data is None:
            lines.append(f"## {model}\n\nMissing: `{path}`\n")
            continue
        lines.append(f"## {model}\n")
        lines.append(f"- rows: {data['n_rows']} / raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']}")
        lines.append(f"- top_k: {data['top_k']} / readout_scale: {data['readout_scale']}")
        lines.append(f"- source_layer_map: {data['source_layer_map']}\n")
        lines.append("### Mode Ladder\n")
        lines.append("| mode | tok0 | mean_rank | margin_vs_top | top0_category | top0_text |")
        lines.append("|---|---:|---:|---:|---|---|")
        for mode in MODE_ORDER:
            row = get_mode(data["summary"], mode)
            if row is None:
                continue
            lines.append(
                f"| {mode} | {row['tok0_hit']}/{row['n']} | {row['mean_prefix_rank']:.1f} | "
                f"{row['mean_prefix_margin_vs_top']:.3f} | {fmt_counts(row['top0_category'])} | "
                f"{fmt_counts(row['top0_text'])} |"
            )
        lines.append("\n### Category Margins\n")
        lines.append("| mode | category | seen_rate | winner_rate | mean_best_rank | prefix_minus_group_max | max_tokens |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for mode in MODE_ORDER:
            for cat in CAT_ORDER:
                row = get_cat(data["summary"], mode, cat)
                if row is None:
                    continue
                lines.append(
                    f"| {mode} | {cat} | {row['seen_rate']:.2f} | {row['winner_rate']:.2f} | "
                    f"{row['mean_best_rank']:.1f} | {row['mean_prefix_minus_group_max']:.3f} | "
                    f"{fmt_counts(row['max_token_text'])} |"
                )
        lines.append("\n### Examples\n")
        for item in data["examples"][:12]:
            tops = ", ".join(
                f"{x['rank']}:{x['text_clean']}[{x['category']}]" for x in item["top"][:8]
            )
            lines.append(
                f"- sample={item['sample_idx']} mode={item['mode']} prefix_rank={item['prefix_rank']} "
                f"top0={item['top0_text_clean']!r}/{item['top0_category']} ladder={tops}"
            )
        lines.append("")
    out = OUT_ROOT / "phase636_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
