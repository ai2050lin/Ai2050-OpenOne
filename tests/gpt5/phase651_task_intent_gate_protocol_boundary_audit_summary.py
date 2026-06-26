#!/usr/bin/env python3
"""Summarize Phase 651 task-intent gate and protocol boundary audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase651_task_intent_gate_protocol_boundary_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def row_line(row) -> str:
    n = row["n"]
    return (
        f"| {row['pair_task']} | {row['eval_task']} | {row['mode']} | {n} | "
        f"{row.get('position_unit') or ''} | {row.get('direction') or ''} | "
        f"{row.get('interval') or ''} | {row.get('component') or ''} | {row.get('control') or ''} | "
        f"{row['baseline_exact'] if row['baseline_exact'] is not None else ''}->{row['exact']} | "
        f"{row['delta_exact_vs_eval_baseline'] if row['delta_exact_vs_eval_baseline'] is not None else ''} | "
        f"{row['baseline_rank'] if row['baseline_rank'] is not None else ''}->{row['mean_prefix_rank']:.1f} | "
        f"{row['delta_rank_improvement_vs_eval_baseline'] if row['delta_rank_improvement_vs_eval_baseline'] is not None else ''} | "
        f"{row['gen_short']}/{n} | {row['gen_explanation']}/{n} | {row['gen_yes_no']}/{n} | "
        f"{row['gen_full_sentence']}/{n} | {row['newline_top0']}/{n} | "
        f"{fmt_counts(row['top0_category'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 651 Cross-Model Summary\n")
    lines.append(
        "目标：通过显式 Current mode 标签，把短答协议场和任务意图门拆开。"
        "`value_to_task` 的正 delta 表示短值吸附增强；"
        "`task_to_value` 的负 delta 表示任务意图可能压制短答协议。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase651_{model}_task_intent_gate_protocol_boundary_audit_confirm.json"
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
        lines.append(f"- tasks: `{data['tasks']}`")
        lines.append(f"- positions: `{data['position_units']}`")
        lines.append(f"- filtered: `{data['filtered']}` / selection: `{data['selection_stats']}`\n")

        lines.append("### Baselines\n")
        lines.append("| pair_task | eval_task | mode | n | position | direction | interval | component | control | exact base->patch | exact_delta | rank base->patch | rank_improve | short | expl | yes/no | full | newline | top0_category |")
        lines.append("|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["by_mode"]:
            if row["kind"] == "baseline":
                lines.append(row_line(row))
        lines.append("")

        lines.append("### Strongest Value-To-Task Absorption\n")
        lines.append("| pair_task | eval_task | mode | n | position | direction | interval | component | control | exact base->patch | exact_delta | rank base->patch | rank_improve | short | expl | yes/no | full | newline | top0_category |")
        lines.append("|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["strongest_absorption"][:80]:
            lines.append(row_line(row))
        lines.append("")

        lines.append("### Strongest Task-To-Value Suppression\n")
        lines.append("| pair_task | eval_task | mode | n | position | direction | interval | component | control | exact base->patch | exact_delta | rank base->patch | rank_improve | short | expl | yes/no | full | newline | top0_category |")
        lines.append("|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["strongest_suppression"][:80]:
            lines.append(row_line(row))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase651_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
