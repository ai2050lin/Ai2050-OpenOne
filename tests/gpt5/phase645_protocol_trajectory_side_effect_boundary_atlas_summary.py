#!/usr/bin/env python3
"""Summarize Phase 645 protocol trajectory side-effect boundary atlas."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas")
MODELS = ["qwen3", "glm4", "deepseek7b"]
SPLIT_ORDER = [
    "target_failure",
    "original_correct",
    "inline_bad",
    "relation_changed",
    "explanation_needed",
    "non_value",
]
MODE_ORDER = [
    "original",
    "inline",
    "to_original_middle_restore",
    "to_original_middle_random",
    "to_original_middle_reverse",
    "remove_from_inline_middle_restore",
    "remove_from_inline_middle_random",
    "remove_from_inline_middle_reverse",
]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def rows_for_split(data, split):
    rows = [r for r in data["summary"]["by_split_mode"] if r["split"] == split]
    rows.sort(key=lambda r: MODE_ORDER.index(r["mode"]) if r["mode"] in MODE_ORDER else 999)
    return rows


def main() -> None:
    lines = []
    lines.append("# Phase 645 Cross-Model Summary\n")
    lines.append(
        "目标：审计 Phase 643/644 的 L17-L20 middle protocol trajectory 是否只在目标失败样本上有效，"
        "以及它对原本正确样本、关系变化、解释任务和非值任务的副作用边界。\n"
    )
    lines.append(
        "注意：relation_changed / explanation_needed / non_value 的 exact 不是正向成功率，"
        "而是旧 value 吸附或过短回答的风险指标。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase645_{model}_protocol_trajectory_side_effect_boundary_atlas_confirm.json"
        data = load_model(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue

        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']}"
        )
        lines.append(f"- component: `{data['component']}` / layers: `{data['layers']}` / top_k: {data['top_k']}")
        lines.append(f"- max_per_split: {data['max_per_split']} / max_new_tokens: {data['max_new_tokens']}")
        lines.append(f"- selection_stats: `{data['selection_stats']}`")
        lines.append(f"- filtered: `{data['filtered']}`")
        lines.append(f"- total_time_min: {data.get('total_time_min', 0.0):.2f}\n")

        for split in SPLIT_ORDER:
            rows = rows_for_split(data, split)
            if not rows:
                continue
            lines.append(f"### {split}\n")
            lines.append("| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for row in rows:
                n = row["n"]
                lines.append(
                    f"| {row['mode']} | {n} | {row['tok0_hit']}/{n} | {row['exact']}/{n} | "
                    f"{row['wrong_exact']}/{n} | {row['newline_top0']}/{n} | {row['gen_newline']}/{n} | "
                    f"{row['gen_short']}/{n} | {row['mean_prefix_rank']:.1f} | "
                    f"{row['mean_prefix_minus_newline']:.3f} | {fmt_counts(row['top0_category'])} | "
                    f"{fmt_counts(row['generation_text'])} |"
                )
            lines.append("")

        lines.append("### Boundary Notes\n")
        split_rows = data["summary"]["by_split"]
        for split in SPLIT_ORDER:
            rows = {r["mode"]: r for r in split_rows.get(split, [])}
            if not rows:
                continue
            original = rows.get("original")
            inline = rows.get("inline")
            restore = rows.get("to_original_middle_restore")
            remove = rows.get("remove_from_inline_middle_restore")
            if original and restore:
                lines.append(
                    f"- {split}: original exact/old={original['exact']}/{original['n']}, "
                    f"to_original_middle_restore exact/old={restore['exact']}/{restore['n']}, "
                    f"newline {original['newline_top0']}->{restore['newline_top0']}"
                )
            if inline and remove:
                lines.append(
                    f"- {split}: inline exact/old={inline['exact']}/{inline['n']}, "
                    f"remove_from_inline_middle_restore exact/old={remove['exact']}/{remove['n']}, "
                    f"newline {inline['newline_top0']}->{remove['newline_top0']}"
                )
        lines.append("")

    out = OUT_ROOT / "phase645_cross_model_summary.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
