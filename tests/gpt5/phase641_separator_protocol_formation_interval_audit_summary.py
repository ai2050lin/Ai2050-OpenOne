#!/usr/bin/env python3
"""Summarize Phase 641 separator protocol formation interval audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase641_separator_protocol_formation_interval_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONTROLS = ["restore", "random", "reverse"]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def interval_sort_key(row):
    layers = row.get("layers") or []
    return (layers[0] if layers else 999, layers[-1] if layers else 999, row.get("interval") or "")


def add_rows(lines, rows):
    lines.append("| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=interval_sort_key):
        n = row["n"]
        layers = ",".join(str(x) for x in row.get("layers", []))
        lines.append(
            f"| {row.get('interval')} | `{layers}` | {n} | {row['tok0_hit']}/{n} | "
            f"{row['newline_top0']}/{n} | {row['mean_prefix_rank']:.1f} | "
            f"{row['mean_prefix_minus_newline']:.3f} | {fmt_counts(row['top0_category'])} |"
        )


def main() -> None:
    lines = []
    lines.append("# Phase 641 Cross-Model Summary\n")
    lines.append(
        "目标：把 inline separator 的 residual protocol trajectory 按层区间恢复到 original prompt，"
        "审计 protocol state 的形成/携带区间，并用 random/reverse 控制排除普通扰动解释。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase641_{model}_separator_protocol_formation_interval_audit_confirm.json"
        data = load_model(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue

        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / "
            f"cases_written: {data['n_cases_written']} / mode_rows: {data['n_mode_rows']}"
        )
        lines.append(f"- target_only: {data['target_only']} / top_k: {data['top_k']}")
        lines.append(f"- component: `{data.get('component', 'layer_out')}`")
        lines.append(f"- intervals: `{data['intervals']}`")
        lines.append(f"- controls: `{data['controls']}`")
        lines.append(f"- filtered: `{data['filtered']}`")
        lines.append(f"- total_time_min: {data.get('total_time_min', 0.0):.2f}\n")

        lines.append("### Baselines\n")
        lines.append("| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for row in data["summary"]["baselines"]:
            n = row["n"]
            lines.append(
                f"| {row['mode']} | {n} | {row['tok0_hit']}/{n} | {row['newline_top0']}/{n} | "
                f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
                f"{fmt_counts(row['top0_category'])} | {fmt_counts(row['top0_text'])} |"
            )

        for control in CONTROLS:
            rows = data["summary"].get(control, [])
            if not rows:
                continue
            lines.append(f"\n### {control}\n")
            add_rows(lines, rows)
        lines.append("")

    out = OUT_ROOT / "phase641_cross_model_summary.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
