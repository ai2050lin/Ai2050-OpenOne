#!/usr/bin/env python3
"""Summarize Phase 659 final top1 barrier and readout audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase659_final_top1_barrier_readout_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def fmt_components(items) -> str:
    return ", ".join(f"L{x['layer']} {x['component']}" for x in items)


def effect_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {fmt_components(r['components'])} | "
        f"{r['n']} | {r['site_rank']:.2f}->{r['combo_rank']:.2f} | {r['rank_improvement']:.2f} | "
        f"{r['site_gap']:.3f}->{r['combo_gap']:.3f} | {r['gap_reduction']:.3f} | "
        f"{r['site_correct_top1']}->{r['combo_correct_top1']} | {r['delta_correct_top1']} | "
        f"{fmt_counts(r['site_top1_category'])} | {fmt_counts(r['combo_top1_category'])} | {fmt_counts(r['combo_top1_text'])} |"
    )


def mode_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['mode']} | "
        f"{fmt_components(r['components'])} | {r['n']} | {r['mean_rank']:.2f} | "
        f"{r['mean_top1_gap']:.3f} | {r['correct_top1_rate']:.3f} | "
        f"{fmt_counts(r['top1_category'])} | {fmt_counts(r['top1_text'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 659 Cross-Model Summary\n")
    lines.append(
        "目标：固定 Phase 658 best combo，比较 baseline_task / site_restore / combo_ablation "
        "下 correct_prefix 与剩余 top1 competitor 的距离，定位最后 top1 barrier。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase659_{model}_final_top1_barrier_readout_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']} / total_time_min: {data.get('total_time_min', 0):.2f}"
        )
        lines.append(f"- combo_specs: `{data['combo_specs']}`")
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`\n")

        lines.append("### Barrier Effects\n")
        lines.append(
            "| pair_task | site | combo | components | n | rank site->combo | rank_improvement | "
            "gap site->combo | gap_reduction | correct_top1 site->combo | delta | "
            "site_top1_category | combo_top1_category | combo_top1_text |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
        for r in data["summary"]["barrier_effects"]:
            lines.append(effect_line(r))
        lines.append("")

        lines.append("### By Mode\n")
        lines.append(
            "| pair_task | site | combo | mode | components | n | mean_rank | mean_top1_gap | "
            "correct_top1_rate | top1_category | top1_text |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["by_mode"]:
            lines.append(mode_line(r))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase659_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
