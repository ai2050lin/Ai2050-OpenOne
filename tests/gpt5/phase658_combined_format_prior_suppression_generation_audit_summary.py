#!/usr/bin/env python3
"""Summarize Phase 658 combined format-prior suppression generation audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase658_combined_format_prior_suppression_generation_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def fmt_components(items) -> str:
    return ", ".join(
        f"L{x['layer']} {x['component']}"
        f"(dE={x.get('phase657_delta_exact', 0)},dT={x.get('phase657_delta_tok0', 0)},dR={x.get('phase657_rank_improvement', 0):.2f})"
        for x in items
    )


def effect_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {fmt_components(r['components'])} | "
        f"{r['n']} | {r['base_exact']}->{r['combo_exact']} | {r['delta_exact']} | "
        f"{r['base_tok0']}->{r['combo_tok0']} | {r['delta_tok0']} | "
        f"{r['base_rank']:.2f}->{r['combo_rank']:.2f} | {r['rank_improvement']:.2f} | "
        f"{fmt_counts(r['base_top0'])} | {fmt_counts(r['combo_top0'])} |"
    )


def mode_line(r) -> str:
    texts = "; ".join(f"{k}:{v}" for k, v in list(r["generation_text"].items())[:4])
    return (
        f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['mode']} | "
        f"{fmt_components(r['components'])} | {r['n']} | "
        f"{r['exact_rate']:.3f} | {r['tok0_rate']:.3f} | {r['mean_rank']:.2f} | "
        f"{fmt_counts(r['top0_category'])} | {texts} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 658 Cross-Model Summary\n")
    lines.append(
        "目标：把 Phase 657 的单点 generation-level 候选按相同 restore site 组合，"
        "检验 final format prior 是否表现为多个 writer 的叠加压力。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase658_{model}_combined_format_prior_suppression_generation_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']} / total_time_min: {data.get('total_time_min', 0):.2f}"
        )
        lines.append(f"- max_cases: {data['max_cases']} / max_new_tokens: {data['max_new_tokens']}")
        lines.append(f"- combo_specs: `{data['combo_specs']}`")
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`\n")

        lines.append("### Combo Effects\n")
        lines.append(
            "| pair_task | site | combo | components | n | exact base->combo | delta_exact | "
            "tok0 base->combo | delta_tok0 | rank base->combo | rank_improvement | base_top0 | combo_top0 |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["combo_effects"]:
            lines.append(effect_line(r))
        lines.append("")

        lines.append("### By Mode\n")
        lines.append(
            "| pair_task | site | combo | mode | components | n | exact_rate | tok0_rate | "
            "mean_rank | top0_category | generation_text |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["by_mode"]:
            lines.append(mode_line(r))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase658_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
