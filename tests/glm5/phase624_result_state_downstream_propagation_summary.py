#!/usr/bin/env python3
"""
Phase 624 summary: Result State Downstream Propagation Atlas
"""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase624_result_state_downstream_propagation_atlas")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def load_results() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(OUT_ROOT.glob("phase624_*_result_state_downstream_propagation_confirm.json"))
    ]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = load_results()
    lines = [
        "# Phase 624 Cross-Model Summary",
        "",
        "Result-state downstream propagation atlas.",
        "",
    ]
    if not results:
        lines.append("No confirm results found.")

    for data in results:
        lines.extend(
            [
                f"## {data['model']}",
                "",
                f"- rows: {data['n_rows']} / raw {data['n_raw_cases']}",
                f"- target cases seen: {data['n_target_cases_seen']}",
                f"- patch layers: {data['patch_layers']}",
                f"- downstream layers: {data['downstream_layers']}",
                "",
                "### Score Modes",
                "",
                "| mode | switch | margin | correct_delta | wrong_delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        score_modes = data.get("summary", {}).get("score_modes", {})
        for mode in sorted(score_modes):
            item = score_modes[mode]
            lines.append(
                f"| {mode} | {item['switch']}/{item['n']} | "
                f"{fmt(item['mean_margin_gain'])} | "
                f"{fmt(item['mean_correct_delta'])} | "
                f"{fmt(item['mean_wrong_delta'])} |"
            )
        lines.extend(
            [
                "",
                "### Top Result-Only Propagation Nodes",
                "",
                "| layer | component | repair_proj | repair_cos | repair_norm | seed_proj | seed_cos |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        prop = [
            x for x in data.get("summary", {}).get("best_propagation", [])
            if x["mode"] == "result_only"
        ][:16]
        for item in prop:
            lines.append(
                f"| L{item['layer']} | {item['component']} | "
                f"{fmt(item['mean_repair_projection'])} | "
                f"{fmt(item['mean_repair_cos'])} | "
                f"{fmt(item['mean_repair_norm_ratio'])} | "
                f"{fmt(item['mean_seed_projection'])} | "
                f"{fmt(item['mean_seed_cos'])} |"
            )
        lines.extend(
            [
                "",
                "### Top All Propagation Nodes",
                "",
                "| mode | layer | component | repair_proj | repair_cos | seed_proj |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for item in data.get("summary", {}).get("best_propagation", [])[:20]:
            lines.append(
                f"| {item['mode']} | L{item['layer']} | {item['component']} | "
                f"{fmt(item['mean_repair_projection'])} | "
                f"{fmt(item['mean_repair_cos'])} | "
                f"{fmt(item['mean_seed_projection'])} |"
            )
        lines.append("")

    out_path = OUT_ROOT / "phase624_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
