#!/usr/bin/env python3
"""Summarize Phase 653 localized intent-gate generation closure audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase653_localized_intent_gate_generation_closure")
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
        f"{row.get('site') or ''} | {row.get('control') or ''} | "
        f"{','.join(map(str, row.get('layers') or []))} | {','.join(row.get('components') or [])} | "
        f"{row['n']} | {row['baseline_rank'] if row['baseline_rank'] is not None else ''}->{row['mean_prefix_rank']:.2f} | "
        f"{row['rank_improvement'] if row['rank_improvement'] is not None else ''} | "
        f"{row['baseline_exact'] if row['baseline_exact'] is not None else ''}->{row['exact']} | "
        f"{row['baseline_tok0'] if row['baseline_tok0'] is not None else ''}->{row['tok0_hit']} | "
        f"{row['gen_short']}/{row['n']} | {row['gen_yes_no']}/{row['n']} | {row['gen_explanation']}/{row['n']} | "
        f"{counts(row['top0_category'])} | {counts(row['generation_text'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 653 Cross-Model Summary\n")
    lines.append(
        "目标：只取 Phase 652 的强峰站点，加入 restore / random / reverse controls，"
        "并用 rank、tok0、短生成 exact 与文本形态共同判断是否形成局部意图门生成闭环。\n"
    )
    for model in MODELS:
        path = OUT_ROOT / f"phase653_{model}_localized_intent_gate_generation_closure_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']} / time: {data.get('total_time_min', 0):.2f} min"
        )
        lines.append(f"- max_new_tokens: {data['max_new_tokens']} / tasks: `{data['tasks']}`")
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`")
        lines.append(f"- sites: `{[s['name'] for s in data['site_specs']]}`\n")

        header = (
            "| pair_task | eval_task | direction | site | control | layers | components | n | "
            "rank base->patch | rank_improve | exact base->patch | tok0 base->patch | "
            "short | yesno | explanation | top0_category | generation_text |"
        )
        sep = "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"

        lines.append("### Baselines\n")
        lines.append(header)
        lines.append(sep)
        for row in data["summary"]["by_mode"]:
            if row["kind"] == "baseline":
                lines.append(line(row))
        lines.append("")

        lines.append("### Restore Absorption: value_to_task\n")
        lines.append(header)
        lines.append(sep)
        for row in data["summary"]["restore_absorption"][:30]:
            lines.append(line(row))
        lines.append("")

        lines.append("### Restore Suppression: task_to_value\n")
        lines.append(header)
        lines.append(sep)
        for row in data["summary"]["restore_suppression"][:30]:
            lines.append(line(row))
        lines.append("")

        lines.append("### Controls\n")
        lines.append(header)
        lines.append(sep)
        for row in data["summary"]["controls"][:60]:
            lines.append(line(row))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase653_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
