#!/usr/bin/env python3
"""Summarize Phase 652 intent-gate layer localization audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase652_intent_gate_layer_localization_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def row_line(row) -> str:
    return (
        f"| {row['pair_task']} | {row['eval_task']} | {row.get('position_unit') or ''} | "
        f"{row.get('direction') or ''} | {row.get('layer') if row.get('layer') is not None else ''} | "
        f"{row.get('component') or ''} | {row.get('control') or ''} | {row['n']} | "
        f"{row['baseline_rank'] if row['baseline_rank'] is not None else ''}->{row['mean_prefix_rank']:.2f} | "
        f"{row['rank_improvement'] if row['rank_improvement'] is not None else ''} | "
        f"{row['baseline_tok0'] if row['baseline_tok0'] is not None else ''}->{row['tok0_hit']} | "
        f"{row['newline_top0']}/{row['n']} | {fmt_counts(row['top0_category'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 652 Cross-Model Summary\n")
    lines.append(
        "目标：把 Phase651 的 L14-L22 区间结果收缩到单层、单位置、单组件。"
        "主指标是 correct value prefix rank 的改善或压制。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase652_{model}_intent_gate_layer_localization_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']} / time: {data.get('total_time_min', 0):.2f} min"
        )
        lines.append(f"- layers: `{data['layers']}` / components: `{data['components']}`")
        lines.append(f"- tasks: `{data['tasks']}` / positions: `{data['position_units']}`")
        lines.append(f"- filtered: `{data['filtered']}` / selection: `{data['selection_stats']}`\n")

        lines.append("### Baselines\n")
        lines.append("| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |")
        lines.append("|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["by_mode"]:
            if row["kind"] == "baseline":
                lines.append(row_line(row))
        lines.append("")

        lines.append("### Strongest Absorption: value_to_task\n")
        lines.append("| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |")
        lines.append("|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["strongest_absorption"][:80]:
            lines.append(row_line(row))
        lines.append("")

        lines.append("### Strongest Suppression: task_to_value\n")
        lines.append("| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |")
        lines.append("|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["strongest_suppression"][:80]:
            lines.append(row_line(row))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase652_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
