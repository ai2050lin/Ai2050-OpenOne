#!/usr/bin/env python3
"""Summarize Phase 650 protocol-field template and side-effect audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase650_protocol_field_template_side_effect_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
SPLIT_ORDER = [
    "target_failure",
    "original_correct",
    "relation_changed",
    "explanation_needed",
    "non_value",
]
TEMPLATE_LABELS = ["Answer", "Response", "Value"]
POSITION_UNITS = ["label_aligned", "label_colon", "separator", "relation_tail"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def line(row) -> str:
    n = row["n"]
    return (
        f"| {row['split']} | {row['template_label']} | {row['mode']} | {n} | "
        f"{row.get('position_unit') or ''} | {row.get('direction') or ''} | "
        f"{row.get('component') or ''} | {row.get('control') or ''} | "
        f"{row['exact']}/{n} | {row['wrong_exact']}/{n} | {row['tok0_hit']}/{n} | "
        f"{row['newline_top0']}/{n} | {row['gen_short']}/{n} | "
        f"{row['gen_explanation_signal']}/{n} | {row['mean_prefix_rank']:.1f} | "
        f"{row['mean_prefix_minus_newline']:.3f} | {fmt_counts(row['top0_category'])} |"
    )


def pick_rows(data, split=None, label=None, position=None, direction=None, control="restore", limit=12):
    rows = data["summary"]["by_mode"]
    out = []
    for row in rows:
        if split is not None and row["split"] != split:
            continue
        if label is not None and row["template_label"] != label:
            continue
        if position is not None and row.get("position_unit") != position:
            continue
        if direction is not None and row.get("direction") != direction:
            continue
        if control is not None and row.get("control") != control:
            continue
        out.append(row)
    out.sort(key=lambda r: (-r["exact_rate"], r["newline_top0_rate"], r["mean_prefix_rank"]))
    return out[:limit]


def main() -> None:
    lines = []
    lines.append("# Phase 650 Cross-Model Summary\n")
    lines.append(
        "目标：把 Phase 649 的 label/separator/relation_tail protocol field 放入跨模板与副作用边界，"
        "检查目标修复、模板泛化、以及对非目标语言状态的旧值吸附风险。\n"
    )
    lines.append(
        "说明：relation_changed / explanation_needed / non_value 的 exact 表示仍输出旧正确值，"
        "在这些 split 中应视为短答值吸附风险，不是正向成功率。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase650_{model}_protocol_field_template_side_effect_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']}"
        )
        lines.append(f"- max_per_split: {data['max_per_split']} / templates: `{data['template_labels']}`")
        lines.append(f"- positions: `{data['position_units']}` / interval_specs: `{data['interval_specs']}`")
        lines.append(f"- selection_stats: `{data['selection_stats']}`")
        lines.append(f"- filtered: `{data['filtered']}` / total_time_min: {data.get('total_time_min', 0.0):.2f}\n")

        lines.append("### Baselines\n")
        lines.append("| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |")
        lines.append("|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for split in SPLIT_ORDER:
            for label in TEMPLATE_LABELS:
                rows = [
                    r for r in data["summary"]["by_mode"]
                    if r["split"] == split and r["template_label"] == label and r["kind"] == "baseline"
                ]
                for row in rows:
                    lines.append(line(row))
        lines.append("")

        lines.append("### Target Failure Best Sufficiency\n")
        lines.append("| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |")
        lines.append("|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["best_target_sufficiency"][:40]:
            lines.append(line(row))
        lines.append("")

        lines.append("### Largest Old-Value Side Effects\n")
        lines.append("| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |")
        lines.append("|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in data["summary"]["largest_side_effect_old_value"][:60]:
            lines.append(line(row))
        lines.append("")

        lines.append("### Position Overview\n")
        for pos in POSITION_UNITS:
            lines.append(f"#### {pos}\n")
            rows = pick_rows(data, split="target_failure", position=pos, direction="to_original", limit=12)
            if rows:
                lines.append("Target repair candidates:")
                lines.append("| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |")
                lines.append("|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
                for row in rows:
                    lines.append(line(row))
            rows = []
            for split in ["original_correct", "relation_changed", "explanation_needed", "non_value"]:
                rows.extend(pick_rows(data, split=split, position=pos, limit=4))
            if rows:
                lines.append("\nSide-effect candidates:")
                lines.append("| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |")
                lines.append("|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
                for row in rows:
                    lines.append(line(row))
            lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase650_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
