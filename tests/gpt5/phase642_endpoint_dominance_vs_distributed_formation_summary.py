#!/usr/bin/env python3
"""Summarize Phase 642 endpoint dominance vs distributed formation audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase642_endpoint_dominance_vs_distributed_formation")
MODELS = ["qwen3", "glm4", "deepseek7b"]
VARIANT_ORDER = ["full", "first", "last", "without_first", "without_last", "middle"]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def sort_key(row):
    layers = row.get("layers") or []
    return (
        row.get("interval") or "",
        VARIANT_ORDER.index(row["variant"]) if row.get("variant") in VARIANT_ORDER else 99,
        layers[0] if layers else 999,
        layers[-1] if layers else 999,
    )


def add_variant_table(lines, rows):
    lines.append("| interval | variant | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=sort_key):
        n = row["n"]
        layers = ",".join(str(x) for x in row.get("layers", []))
        lines.append(
            f"| {row.get('interval')} | {row.get('variant')} | `{layers}` | {n} | "
            f"{row['tok0_hit']}/{n} | {row['newline_top0']}/{n} | "
            f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
            f"{fmt_counts(row['top0_category'])} |"
        )


def add_control_table(lines, rows):
    lines.append("| direction | interval | variant | control | n | tok0 | newline_top0 | rank | prefix-newline |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (r.get("direction") or "", r.get("interval") or "", r.get("variant") or "", r.get("control") or "")):
        n = row["n"]
        lines.append(
            f"| {row.get('direction')} | {row.get('interval')} | {row.get('variant')} | {row.get('control')} | "
            f"{n} | {row['tok0_hit']}/{n} | {row['newline_top0']}/{n} | "
            f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} |"
        )


def main() -> None:
    lines = []
    lines.append("# Phase 642 Cross-Model Summary\n")
    lines.append(
        "目标：拆分 Phase 641 的强区间，比较 full/endpoint/leave-end/middle，"
        "并同时测试 to_original sufficiency 与 remove_from_inline necessity。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase642_{model}_endpoint_dominance_vs_distributed_formation_confirm.json"
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
        lines.append(f"- component: `{data['component']}`")
        lines.append(f"- intervals: `{data['intervals']}`")
        lines.append(f"- variants: `{data['variants']}`")
        lines.append(f"- controls: `{data['controls']}` / control_variants: `{data['control_variants']}`")
        lines.append(f"- directions: `{data['directions']}`")
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

        lines.append("\n### To Original Restore\n")
        add_variant_table(lines, data["summary"]["restore_to_original"])

        lines.append("\n### Remove From Inline Restore\n")
        add_variant_table(lines, data["summary"]["remove_from_inline"])

        lines.append("\n### Random/Reverse Controls\n")
        add_control_table(lines, data["summary"]["controls"])
        lines.append("")

    out = OUT_ROOT / "phase642_cross_model_summary.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
