#!/usr/bin/env python3
"""Summarize Phase 665 cross-model results."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase665_autoregressive_continuation_controller_localization")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_model(model: str):
    path = OUT_ROOT / f"phase665_{model}_autoregressive_continuation_controller_localization_confirm.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dtext(d: dict) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def main() -> None:
    lines = [
        "# Phase 665 Cross-Model Summary",
        "",
        "目标：定位 correct_prefix top1 but exact wrong 后的真实自回归续写控制器，扫描 token1/token2 的 continuation-position source patch。",
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
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / continuation_failures: {data['n_continuation_failures']} / rows: {data['n_rows']} / total_time_min: {data['total_time_min']:.2f}",
            f"- scan_layers: `{data['scan_layers']}`",
            f"- scan_components: `{data['scan_components']}`",
            f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`",
            "",
            "### Selected Continuation Failures",
            "",
            "| pair_task | site | combo | n | generation_text |",
            "|---|---|---|---:|---|",
        ]
        for r in data["summary"]["selected_continuation_failures"]:
            lines.append(f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['n']} | {dtext(r['generation_text'])} |")
        lines += [
            "",
            "### Continuation Baselines",
            "",
            "| pair_task | site | combo | step | n | expected_top1_rate | mean_expected_rank | mean_expected_minus_top1 | top1_text |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in data["summary"]["continuation_baselines"]:
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['step']} | {r['n']} | "
                f"{r['expected_top1_rate']:.3f} | {r['mean_expected_rank']:.2f} | "
                f"{r['mean_expected_minus_top1']:.3f} | {dtext(r['top1_text'])} |"
            )
        lines += [
            "",
            "### Top Component Patch Candidates",
            "",
            "| pair_task | site | combo | step | layer | component | n | flip_rate | mean_rank_improvement | mean_margin_delta | patched_top1 |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
        for r in data["summary"]["component_patch_candidates"][:60]:
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['step']} | {r['layer']} | {r['component']} | "
                f"{r['n']} | {r['flip_rate']:.3f} | {r['mean_rank_improvement']:.2f} | "
                f"{r['mean_margin_delta']:.3f} | {dtext(r['patched_top1'])} |"
            )
        lines.append("")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase665_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
