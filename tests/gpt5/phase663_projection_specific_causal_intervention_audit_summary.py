#!/usr/bin/env python3
"""Summarize Phase 663 cross-model results."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase663_projection_specific_causal_intervention_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt_num(x, digits=3):
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def load_model(model: str):
    path = OUT_ROOT / f"phase663_{model}_projection_specific_causal_intervention_audit_confirm.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    lines = [
        "# Phase 663 Cross-Model Summary",
        "",
        "目标：对 Phase 662 的 projection barrier 诊断做读出端反事实干预验证，区分 norm advantage、hidden direction alignment 和 continuation failure。",
        "",
    ]
    for model in MODELS:
        data = load_model(model)
        if data is None:
            lines += [f"## {model}", "", "No confirm result found.", ""]
            continue
        lines += [
            f"## {model}",
            "",
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / rows: {data['n_rows']} / total_time_min: {data['total_time_min']:.2f}",
            f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`",
            f"- direction_scales: `{data['direction_scales']}`",
            "",
            "### Plus-Last-Writers Actual State",
            "",
            "| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category | continuation_tag |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for r in data["summary"]["by_site"]:
            top1 = ", ".join(f"{k}:{v}" for k, v in r["top1_category"].items())
            cont = ", ".join(f"{k}:{v}" for k, v in r["continuation_tag"].items())
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['n']} | "
                f"{r['exact_rate']:.3f} | {r['correct_top1_rate']:.3f} | "
                f"{r['mean_rank']:.2f} | {r['mean_gap']:.3f} | {top1} | {cont} |"
            )
        lines += [
            "",
            "### Norm-Neutralized Pair Readout",
            "",
            "| pair_task | top1_category | n | top1_text | actual_gap | neutral_cos_gap | neutral_flip_rate | correct_cos | competitor_cos | norm_adv | needed_delta |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in data["summary"]["failure_norm_neutral"]:
            top1_text = ", ".join(f"{k}:{v}" for k, v in r["top1_text"].items())
            lines.append(
                f"| {r['pair_task']} | {r['top1_category']} | {r['n']} | {top1_text} | "
                f"{r['mean_actual_gap']:.3f} | {r['mean_norm_neutral_cos_gap']:.4f} | "
                f"{r['norm_neutral_flip_rate']:.3f} | {r['mean_correct_cos']:.4f} | "
                f"{r['mean_competitor_cos']:.4f} | {r['mean_competitor_norm_advantage']:.4f} | "
                f"{r['mean_needed_unit_delta']:.4f} |"
            )
        lines += [
            "",
            "### Direction Correction by Scale",
            "",
            "| pair_task | top1_category | scale | n | correct_top1_rate | mean_rank | mean_gap | top1_after |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in data["summary"]["direction_correction_by_scale"]:
            top1_after = ", ".join(f"{k}:{v}" for k, v in r["top1_after"].items())
            lines.append(
                f"| {r['pair_task']} | {r['top1_category']} | {r['scale']:.1f} | {r['n']} | "
                f"{r['correct_top1_rate']:.3f} | {r['mean_rank']:.2f} | {r['mean_gap']:.3f} | {top1_after} |"
            )
        lines += [
            "",
            "### Continuation Failures",
            "",
            "| pair_task | site | combo | n | generation_text |",
            "|---|---|---|---:|---|",
        ]
        for r in data["summary"]["continuation_failures"]:
            texts = ", ".join(f"{k}:{v}" for k, v in r["generation_text"].items())
            lines.append(f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['n']} | {texts} |")
        lines.append("")

    out = OUT_ROOT / "phase663_cross_model_summary.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
