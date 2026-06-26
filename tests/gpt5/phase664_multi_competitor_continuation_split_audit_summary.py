#!/usr/bin/env python3
"""Summarize Phase 664 cross-model results."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase664_multi_competitor_continuation_split_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_model(model: str):
    path = OUT_ROOT / f"phase664_{model}_multi_competitor_continuation_split_audit_confirm.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dict_text(d: dict) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def main() -> None:
    lines = [
        "# Phase 664 Cross-Model Summary",
        "",
        "目标：从 pairwise projection intervention 推进到 multi-competitor readout margin，并审计 correct-prefix-top1 之后的 continuation failure。",
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
            f"- target_categories: `{data['target_categories']}`",
            f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`",
            "",
            "### Actual Multi-Competitor State",
            "",
            "| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_category | max_competitor | continuation_tag |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
        for r in data["summary"]["by_site"]:
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['n']} | "
                f"{r['exact_rate']:.3f} | {r['correct_top1_rate']:.3f} | {r['mean_rank']:.2f} | "
                f"{r['mean_gap']:.3f} | {r['mean_multi_margin']:.3f} | "
                f"{dict_text(r['top1_category'])} | {dict_text(r['max_competitor_category'])} | {dict_text(r['continuation_tag'])} |"
            )
        lines += [
            "",
            "### Multi-Competitor Failures",
            "",
            "| pair_task | max_competitor | n | mean_multi_margin | winner_sets |",
            "|---|---|---:|---:|---|",
        ]
        for r in data["summary"]["multi_competitor_failures"]:
            lines.append(
                f"| {r['pair_task']} | {r['max_competitor_category']} | {r['n']} | "
                f"{r['mean_multi_margin']:.3f} | {dict_text(r['winner_sets'])} |"
            )
        lines += [
            "",
            "### Multi-Correction by Scale",
            "",
            "| pair_task | max_competitor | scale | n | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_after | max_comp_after |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for r in data["summary"]["multi_correction_by_scale"]:
            lines.append(
                f"| {r['pair_task']} | {r['max_competitor_category']} | {r['scale']:.1f} | {r['n']} | "
                f"{r['correct_top1_rate']:.3f} | {r['mean_rank']:.2f} | {r['mean_gap']:.3f} | "
                f"{r['mean_multi_margin']:.3f} | {dict_text(r['top1_after'])} | {dict_text(r['max_comp_after'])} |"
            )
        lines += [
            "",
            "### Continuation Audit",
            "",
            "| pair_task | site | combo | n | token1_match_rate | token2_match_rate | mean_token1_expected_rank | mean_token2_expected_rank | generated_text |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in data["summary"]["continuation_audit"]:
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['n']} | "
                f"{r['token1_match_rate']:.3f} | {r['token2_match_rate']:.3f} | "
                f"{r['mean_token1_expected_rank']:.2f} | {r['mean_token2_expected_rank']:.2f} | "
                f"{dict_text(r['generated_text'])} |"
            )
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase664_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
