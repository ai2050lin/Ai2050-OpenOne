#!/usr/bin/env python3
"""
Phase 629 summary: Format/Prefix Gate Localization
"""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase629_format_prefix_gate_localization")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def load_results() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(OUT_ROOT.glob("phase629_*_format_prefix_gate_localization_confirm.json"))
    ]


def pos_rates(item: dict) -> str:
    rates = item.get("pos_correct_rate", {})
    return ", ".join(f"tok{k}:{fmt(v)}" for k, v in rates.items())


def row_line(mode: str, item: dict) -> str:
    return (
        f"| {mode} | {item['exact_correct']}/{item['n']} | "
        f"{item['exact_wrong']}/{item['n']} | "
        f"{fmt(item['mean_prefix_correct_len'])} | {pos_rates(item)} |"
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = load_results()
    lines = [
        "# Phase 629 Cross-Model Summary",
        "",
        "Prompt-last format/prefix gate localization with semantic cumulative combination.",
        "",
    ]
    if not results:
        lines.append("No confirm results found.")

    baseline_modes = [
        "base",
        "repair_prompt",
        "semantic_cumulative_only",
        "semantic_cumulative_random",
    ]

    for data in results:
        modes = data.get("summary", {}).get("by_mode", {})
        best_exact = data.get("summary", {}).get("best_exact", [])[:20]
        best_tok0 = data.get("summary", {}).get("best_tok0", [])[:20]
        lines.extend([
            f"## {data['model']}",
            "",
            f"- rows: {data['n_rows']} / raw {data['n_raw_cases']}",
            f"- target cases seen: {data['n_target_cases_seen']}",
            f"- format layers: {data['format_layers']}",
            f"- downstream layers: {data['downstream_layers']}",
            f"- components: {data['components']}",
            f"- tokenization: `{data['tokenization']}`",
            "",
            "### Baselines",
            "",
            "| mode | exact | wrong_exact | prefix_len | position_correct |",
            "|---|---:|---:|---:|---|",
        ])
        for mode in baseline_modes:
            if mode in modes:
                lines.append(row_line(mode, modes[mode]))
        lines.extend([
            "",
            "### Best Exact",
            "",
            "| mode | exact | wrong_exact | prefix_len | position_correct |",
            "|---|---:|---:|---:|---|",
        ])
        seen = set()
        for item in best_exact:
            mode = item["mode"]
            if mode in seen:
                continue
            seen.add(mode)
            lines.append(row_line(mode, item))
        lines.extend([
            "",
            "### Best Tok0",
            "",
            "| mode | exact | wrong_exact | prefix_len | position_correct |",
            "|---|---:|---:|---:|---|",
        ])
        seen = set()
        for item in best_tok0:
            mode = item["mode"]
            if mode in seen:
                continue
            seen.add(mode)
            lines.append(row_line(mode, item))

        lines.extend(["", "### Examples", ""])
        example_modes = ["base", "repair_prompt", "semantic_cumulative_only"]
        for item in best_exact[:3]:
            if item["mode"] not in example_modes:
                example_modes.append(item["mode"])
        for row in data.get("rows", [])[:5]:
            case = row["case"]
            lines.append(f"- {case['object']} / {case['relation']} correct={case['correct']} old_wrong={row['old_top_wrong']}")
            for mode in example_modes:
                if mode in row["generations"]:
                    gen = row["generations"][mode]["generation"]
                    lines.append(f"  - {mode}: `{gen['text']}` {gen['tokens']}")
        lines.append("")

    out_path = OUT_ROOT / "phase629_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
