#!/usr/bin/env python3
"""Summarize Phase 660 space/newline readout source backtrace."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase660_space_newline_readout_source_backtrace")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def fmt_components(items) -> str:
    return ", ".join(f"L{x['layer']} {x['component']}" for x in items)


def source_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {fmt_components(r['components'])} | "
        f"{r['n']} | {r['site_post_gap']:.3f}->{r['combo_post_gap']:.3f} | "
        f"{r['gap_reduction']:.3f} | {r['site_post_rank']:.2f}->{r['combo_post_rank']:.2f} | "
        f"{r['rank_improvement']:.2f} | {r['combo_norm_gap_shift']:.3f} | "
        f"{fmt_counts(r['site_post_top1_category'])} | {fmt_counts(r['combo_post_top1_category'])} |"
    )


def writer_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['mode']} | "
        f"{fmt_components(r['components'])} | {r['n']} | "
        f"{r['combo_post_gap']:.3f}->{r['ablated_post_gap']:.3f} | {r['gap_delta_vs_combo']:.3f} | "
        f"{r['combo_rank']:.2f}->{r['ablated_rank']:.2f} | {r['rank_delta_vs_combo']:.2f} | "
        f"{r['ablated_norm_gap_shift']:.3f} | {fmt_counts(r['ablated_top1_category'])} |"
    )


def mode_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['mode']} | "
        f"{fmt_components(r['components'])} | {r['n']} | "
        f"{r['mean_pre_gap']:.3f} | {r['mean_post_gap']:.3f} | {r['mean_norm_gap_shift']:.3f} | "
        f"{r['mean_post_rank']:.2f} | {r['correct_top1_rate']:.3f} | "
        f"{fmt_counts(r['pre_top1_category'])} | {fmt_counts(r['post_top1_category'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 660 Cross-Model Summary\n")
    lines.append(
        "目标：固定 Phase 659 best combo，审计 space/newline final top1 barrier "
        "来自 residual state、final_norm shift、lm_head projection，还是最后残差写入器残留。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase660_{model}_space_newline_readout_source_backtrace_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']} / total_time_min: {data.get('total_time_min', 0):.2f}"
        )
        lines.append(f"- last_layers: `{data['last_layers']}`")
        lines.append(f"- combo_specs: `{data['combo_specs']}`")
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`\n")

        lines.append("### Source Effects\n")
        lines.append(
            "| pair_task | site | combo | components | n | post_gap site->combo | gap_reduction | "
            "rank site->combo | rank_improvement | combo_norm_gap_shift | site_top1 | combo_top1 |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["source_effects"]:
            lines.append(source_line(r))
        lines.append("")

        lines.append("### Last Writer Effects\n")
        lines.append(
            "| pair_task | site | combo | last_writer_mode | components | n | gap combo->ablated | "
            "gap_delta_vs_combo | rank combo->ablated | rank_delta_vs_combo | ablated_norm_shift | ablated_top1 |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for r in data["summary"]["last_writer_effects"][:80]:
            lines.append(writer_line(r))
        lines.append("")

        lines.append("### By Mode\n")
        lines.append(
            "| pair_task | site | combo | mode | components | n | pre_gap | post_gap | "
            "norm_gap_shift | post_rank | correct_top1_rate | pre_top1 | post_top1 |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["by_mode"][:120]:
            lines.append(mode_line(r))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase660_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
