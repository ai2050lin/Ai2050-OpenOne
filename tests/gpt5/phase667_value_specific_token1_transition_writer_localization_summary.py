#!/usr/bin/env python3
"""Summarize Phase 667 cross-model writer localization results."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase667_value_specific_token1_transition_writer_localization")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_model(model: str):
    path = OUT_ROOT / f"phase667_{model}_value_specific_token1_transition_writer_localization_confirm.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    lines = [
        "# Phase 667 Cross-Model Summary",
        "",
        "目标：定位 value-specific token1 transition state 的 writer 候选，比较组件输出和 attention head o_proj-input slice 的 correct/mismatch/zero 对照。",
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
            f"- source_phase665: `{data['source_phase665']}`",
            f"- failures_tested: {data['n_failures_tested']} / rows: {data['n_rows']} / total_time_min: {data['total_time_min']:.2f}",
            f"- component_writers: `{data['component_writers']}`",
            f"- head_layers: `{data['head_layers']}` / max_heads: `{data['max_heads']}`",
            "",
            "### Top Writer Specificity",
            "",
            "| kind | pair_task | site | combo | writer | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in data["summary"]["writer_specificity"][:80]:
            lines.append(
                f"| {r['kind']} | {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['writer']} | "
                f"{r['n']} | {r['correct_top1_rate']:.3f} | {r['mismatch_top1_rate']:.3f} | "
                f"{r['correct_minus_mismatch_margin_delta']:.3f} | {r['correct_minus_zero_margin_delta']:.3f} |"
            )
        lines += [
            "",
            "### Intervention Summary",
            "",
            "| kind | pair_task | site | combo | writer | intervention | n | top1_rate | rank_delta | margin_delta |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|",
        ]
        for r in data["summary"]["intervention_summary"][:180]:
            lines.append(
                f"| {r['kind']} | {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['writer']} | "
                f"{r['intervention']} | {r['n']} | {r['expected_top1_rate']:.3f} | "
                f"{r['mean_rank_delta_vs_baseline']:.2f} | {r['mean_margin_delta_vs_baseline']:.3f} |"
            )
        lines.append("")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase667_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
