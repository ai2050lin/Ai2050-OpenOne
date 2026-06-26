#!/usr/bin/env python3
"""Summarize Phase 661 last-writer combo generation closure."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase661_last_writer_combo_generation_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def fmt_items(items) -> str:
    return ", ".join(f"L{x['layer']} {x['component']}" for x in items)


def effect_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {fmt_items(r['combo'])} | "
        f"{fmt_items(r['last_writers'])} | {r['n']} | "
        f"{r['base_exact']}->{r['ext_exact']} | {r['delta_exact']} | "
        f"{r['base_tok0']}->{r['ext_tok0']} | {r['delta_tok0']} | "
        f"{r['base_rank']:.2f}->{r['ext_rank']:.2f} | {r['rank_improvement']:.2f} | "
        f"{r['base_gap']:.3f}->{r['ext_gap']:.3f} | {r['gap_reduction']:.3f} | "
        f"{fmt_counts(r['base_top1'])} | {fmt_counts(r['ext_top1'])} |"
    )


def mode_line(r) -> str:
    texts = "; ".join(f"{k}:{v}" for k, v in list(r["generation_text"].items())[:4])
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['mode']} | "
        f"{fmt_items(r['combo'])} | {fmt_items(r['last_writers'])} | {r['n']} | "
        f"{r['exact_rate']:.3f} | {r['tok0_rate']:.3f} | {r['mean_rank']:.2f} | "
        f"{r['mean_gap']:.3f} | {fmt_counts(r['top1_category'])} | {texts} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 661 Cross-Model Summary\n")
    lines.append(
        "目标：把 Phase 660 的 strongest last-writer candidates 与 Phase 658 best combo 叠加，"
        "测试剩余 space/newline top1 barrier 是否能进一步进入 generation closure。\n"
    )
    for model in MODELS:
        path = OUT_ROOT / f"phase661_{model}_last_writer_combo_generation_closure_confirm.json"
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

        lines.append("### Closure Effects\n")
        lines.append(
            "| pair_task | site | combo | phase658_combo | extra_last_writers | n | exact base->ext | "
            "delta_exact | tok0 base->ext | delta_tok0 | rank base->ext | rank_improvement | "
            "gap base->ext | gap_reduction | base_top1 | ext_top1 |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["closure_effects"]:
            lines.append(effect_line(r))
        lines.append("")

        lines.append("### By Mode\n")
        lines.append(
            "| pair_task | site | combo | mode | phase658_combo | extra_last_writers | n | exact_rate | "
            "tok0_rate | mean_rank | mean_gap | top1_category | generation_text |"
        )
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["by_mode"]:
            lines.append(mode_line(r))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase661_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
