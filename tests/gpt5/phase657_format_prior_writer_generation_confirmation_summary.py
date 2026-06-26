#!/usr/bin/env python3
"""Summarize Phase 657 format-prior writer generation confirmation."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase657_format_prior_writer_generation_confirmation")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def effect_line(r) -> str:
    return (
        f"| {r['pair_task']} | {r['site']} | L{r['candidate_layer']} | {r['candidate_component']} | "
        f"{r['n']} | {r['phase656_dtop']:.3f} | "
        f"{r['base_exact']}->{r['ablation_exact']} | {r['delta_exact']} | "
        f"{r['base_tok0']}->{r['ablation_tok0']} | {r['delta_tok0']} | "
        f"{r['base_rank']:.2f}->{r['ablation_rank']:.2f} | {r['rank_improvement']:.2f} | "
        f"{fmt_counts(r['base_top0'])} | {fmt_counts(r['ablation_top0'])} |"
    )


def mode_line(r) -> str:
    texts = "; ".join(f"{k}:{v}" for k, v in list(r["generation_text"].items())[:4])
    return (
        f"| {r['pair_task']} | {r['site']} | L{r['candidate_layer']} | {r['candidate_component']} | "
        f"{r['mode']} | {r['n']} | {r['phase656_dtop']:.3f} | "
        f"{r['exact_rate']:.3f} | {r['tok0_rate']:.3f} | {r['mean_rank']:.2f} | "
        f"{fmt_counts(r['top0_category'])} | {texts} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 657 Cross-Model Summary\n")
    lines.append(
        "目标：读取 Phase 656 的格式先验写入候选，在固定 intent-gate restore patch 后，"
        "对候选组件做最终位置消融，并用短贪婪生成验证 margin-level 候选是否进入 generation-level。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase657_{model}_format_prior_writer_generation_confirmation_confirm.json"
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
        lines.append(f"- candidate_specs: `{data['candidate_specs']}`")
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`\n")

        lines.append("### Generation Effects\n")
        lines.append(
            "| pair_task | site | layer | component | n | phase656_dtop | exact base->ablate | "
            "delta_exact | tok0 base->ablate | delta_tok0 | rank base->ablate | rank_improvement | "
            "base_top0 | ablation_top0 |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["generation_effects"]:
            lines.append(effect_line(r))
        lines.append("")

        lines.append("### By Mode\n")
        lines.append(
            "| pair_task | site | layer | component | mode | n | phase656_dtop | exact_rate | "
            "tok0_rate | mean_rank | top0_category | generation_text |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
        for r in data["summary"]["by_mode"]:
            lines.append(mode_line(r))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase657_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
