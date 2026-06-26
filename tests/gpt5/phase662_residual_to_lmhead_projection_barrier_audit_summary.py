#!/usr/bin/env python3
"""Summarize Phase 662 residual-to-lmhead projection barrier audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase662_residual_to_lmhead_projection_barrier_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def by_mode_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['mode']} | "
        f"{r['n']} | {r['exact_rate']:.3f} | {r['correct_top1_rate']:.3f} | "
        f"{r['mean_rank']:.2f} | {r['mean_gap']:.3f} | {fmt_counts(r['top1_category'])} |"
    )


def failure_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['top1_category']} | {r['n']} | "
        f"{fmt_counts(r['top1_text'])} | {r['mean_post_gap']:.3f} | {r['mean_pre_gap']:.3f} | "
        f"{r['mean_norm_gap_change']:.3f} | {r['mean_needed_unit_delta']:.4f} | "
        f"{r['mean_diff_alignment']:.4f} | {r['mean_correct_cos']:.4f} | "
        f"{r['mean_competitor_cos']:.4f} | {r['mean_competitor_norm_advantage']:.4f} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 662 Cross-Model Summary\n")
    lines.append(
        "目标：固定 Phase 661 partially repaired state，审计剩余失败是否来自 "
        "final_norm output direction 不足或 lm_head/unembedding projection advantage。\n"
    )
    for model in MODELS:
        path = OUT_ROOT / f"phase662_{model}_residual_to_lmhead_projection_barrier_audit_confirm.json"
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
        lines.append(f"- last_writer_map: `{data['last_writer_map']}`")
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`\n")

        lines.append("### By Mode\n")
        lines.append("| pair_task | site | combo | mode | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
        for r in data["summary"]["by_mode"]:
            lines.append(by_mode_line(r))
        lines.append("")

        lines.append("### Plus-Last-Writers Remaining Failure Projection\n")
        lines.append(
            "| pair_task | top1_category | n | top1_text | post_gap | pre_gap | norm_gap_change | "
            "needed_unit_delta | diff_alignment | correct_cos | competitor_cos | competitor_norm_advantage |"
        )
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in data["summary"]["plus_failure_projection"]:
            lines.append(failure_line(r))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase662_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
