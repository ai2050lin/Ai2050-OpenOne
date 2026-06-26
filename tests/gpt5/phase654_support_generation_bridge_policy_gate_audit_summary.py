#!/usr/bin/env python3
"""Summarize Phase 654 support-to-generation bridge policy gate audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase654_support_generation_bridge_policy_gate_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def line(row) -> str:
    return (
        f"| {row['pair_task']} | {row['eval_task']} | {row.get('direction') or ''} | "
        f"{row.get('site') or ''} | {','.join(map(str, row.get('layers') or []))} | "
        f"{','.join(row.get('components') or [])} | {row['n']} | "
        f"{row['mean_prefix_rank']:.2f} | {row['mean_prefix_margin_vs_top']:.3f} | "
        f"{row['exact']}/{row['n']} | {row['tok0_hit']}/{row['n']} | "
        f"{row['support_without_generation']}/{row['n']} | {row['mean_final_l2']:.3f} | "
        f"{counts(row['top0_category'])} | {counts(row['gen_first_text'])} |"
    )


def failure_line(row) -> str:
    return (
        f"| {row['pair_task']} | {row['eval_task']} | {row.get('direction') or ''} | "
        f"{row.get('site') or ''} | {row['prefix_rank']} | {row['prefix_margin_vs_top']:.3f} | "
        f"{row['top0_text_clean']} | {row.get('gen_first_text','')} | "
        f"{row['generation_text'].replace(chr(10), '<nl>')[:80]} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 654 Cross-Model Summary\n")
    lines.append(
        "目标：固定 Phase 653 的强峰 restore patch，审计 support_value 到 generate_value 之间的桥接失败。"
        "重点观察 rank 进入前 15 但短生成仍不输出正确值的样本。\n"
    )
    header = (
        "| pair_task | eval_task | direction | site | layers | components | n | mean_rank | "
        "mean_margin_vs_top | exact | tok0 | support_no_gen | final_l2 | top0_category | gen_first_text |"
    )
    sep = "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"
    failure_header = "| pair_task | eval_task | direction | site | prefix_rank | margin_vs_top | top0 | gen_first | generation_text |"
    failure_sep = "|---|---|---|---|---:|---:|---|---|---|"

    for model in MODELS:
        path = OUT_ROOT / f"phase654_{model}_support_generation_bridge_policy_gate_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']} / time: {data.get('total_time_min', 0):.2f} min"
        )
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`")
        lines.append(f"- sites: `{[s['name'] for s in data['site_specs']]}`\n")
        lines.append("### By Mode\n")
        lines.append(header)
        lines.append(sep)
        for row in data["summary"]["by_mode"]:
            lines.append(line(row))
        lines.append("")
        lines.append("### Bridge Failures: rank <= 15 and exact false\n")
        lines.append(failure_header)
        lines.append(failure_sep)
        for row in data["summary"]["bridge_failures"][:40]:
            lines.append(failure_line(row))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase654_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
