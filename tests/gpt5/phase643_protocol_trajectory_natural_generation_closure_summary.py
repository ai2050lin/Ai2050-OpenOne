#!/usr/bin/env python3
"""Summarize Phase 643 protocol trajectory natural generation closure."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase643_protocol_trajectory_natural_generation_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]
MODE_ORDER = [
    "original",
    "inline",
    "to_original_full_restore",
    "to_original_middle_restore",
    "to_original_full_random",
    "to_original_full_reverse",
    "remove_from_inline_full_restore",
    "remove_from_inline_middle_restore",
    "remove_from_inline_full_random",
    "remove_from_inline_full_reverse",
]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def main() -> None:
    lines = []
    lines.append("# Phase 643 Cross-Model Summary\n")
    lines.append(
        "目标：把 Phase 642 的 L17-L20 protocol trajectory patch 压到 greedy natural generation，"
        "检查 exact generation、newline/explanation tendency 和生成文本分布。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase643_{model}_protocol_trajectory_natural_generation_closure_confirm.json"
        data = load_model(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue

        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / "
            f"cases_written: {data['n_cases_written']} / mode_rows: {data['n_mode_rows']}"
        )
        lines.append(f"- target_only: {data['target_only']} / top_k: {data['top_k']} / max_new_tokens: {data['max_new_tokens']}")
        lines.append(f"- component: `{data['component']}` / interval: `{data['interval']}`")
        lines.append(f"- full_layers: `{data['full_layers']}` / middle_layers: `{data['middle_layers']}`")
        lines.append(f"- filtered: `{data['filtered']}`")
        lines.append(f"- total_time_min: {data.get('total_time_min', 0.0):.2f}\n")

        rows = sorted(
            data["summary"]["by_mode"],
            key=lambda r: MODE_ORDER.index(r["mode"]) if r["mode"] in MODE_ORDER else 999,
        )
        lines.append("| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | generation_text |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in rows:
            n = row["n"]
            lines.append(
                f"| {row['mode']} | {n} | {row['tok0_hit']}/{n} | {row['exact']}/{n} | "
                f"{row['wrong_exact']}/{n} | {row['newline_top0']}/{n} | "
                f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
                f"{fmt_counts(row['top0_category'])} | {fmt_counts(row['generation_text'])} |"
            )
        lines.append("")

    out = OUT_ROOT / "phase643_cross_model_summary.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
