#!/usr/bin/env python3
"""
Phase 627 summary: Natural Generation Token-Position Closure
"""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase627_natural_generation_token_position_closure")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def load_results() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(OUT_ROOT.glob("phase627_*_natural_generation_token_position_closure_confirm.json"))
    ]


def pos_rates(item: dict) -> str:
    rates = item.get("pos_correct_rate", {})
    return ", ".join(f"tok{k}:{fmt(v)}" for k, v in rates.items())


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = load_results()
    lines = [
        "# Phase 627 Cross-Model Summary",
        "",
        "Natural greedy generation token-position closure.",
        "",
    ]
    if not results:
        lines.append("No confirm results found.")

    preferred = [
        "base",
        "repair_prompt",
        "result_only",
        "result_random",
        "cumulative_layer_out",
        "cumulative_layer_out_random",
        "final_output_all",
        "final_output_random_all",
    ]
    for data in results:
        lines.extend(
            [
                f"## {data['model']}",
                "",
                f"- rows: {data['n_rows']} / raw {data['n_raw_cases']}",
                f"- target cases seen: {data['n_target_cases_seen']}",
                f"- result layers: {data['result_layers']}",
                f"- downstream layers: {data['downstream_layers']}",
                f"- tokenization: `{data['tokenization']}`",
                "",
                "| mode | exact | wrong_exact | prefix_len | position_correct |",
                "|---|---:|---:|---:|---|",
            ]
        )
        modes = data.get("summary", {}).get("by_mode", {})
        ordered = [m for m in preferred if m in modes] + [m for m in sorted(modes) if m not in preferred]
        for mode in ordered:
            item = modes[mode]
            lines.append(
                f"| {mode} | {item['exact_correct']}/{item['n']} | "
                f"{item['exact_wrong']}/{item['n']} | "
                f"{fmt(item['mean_prefix_correct_len'])} | {pos_rates(item)} |"
            )
        lines.extend(["", "### Examples", ""])
        for row in data.get("rows", [])[:6]:
            case = row["case"]
            lines.append(f"- {case['object']} / {case['relation']} correct={case['correct']} old_wrong={row['old_top_wrong']}")
            for mode in ["base", "repair_prompt", "result_only", "cumulative_layer_out", "final_output_all"]:
                if mode in row["generations"]:
                    gen = row["generations"][mode]["generation"]
                    lines.append(f"  - {mode}: `{gen['text']}` {gen['tokens']}")
        lines.append("")

    out_path = OUT_ROOT / "phase627_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
