#!/usr/bin/env python3
"""Summarize Phase590 cross-model value winner multi-layer patch results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase590_value_winner_multilayer_patch")
MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT = ROOT / "phase590_cross_model_summary.md"


def load_model(model: str) -> dict:
    path = ROOT / f"phase590_{model}_value_winner_multilayer_patch_confirm.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def best_rows(data: dict, mode: str, limit: int = 6) -> list[dict]:
    rows = [
        r
        for r in data["summary"]["by_key"].values()
        if r["mode"] == mode and r["target_n"] > 0
    ]
    rows.sort(
        key=lambda r: (
            r["target_switch"],
            r["target_switch_rate"],
            r["mean_margin_gain"],
            r["mean_correct_gain"],
        ),
        reverse=True,
    )
    return rows[:limit]


def model_section(model: str, data: dict) -> list[str]:
    lines = []
    lines.append(f"## {model}")
    lines.append("")
    lines.append(
        f"- cases={data['n_cases']}, layers={data['probe_layers']}, alpha={data['alpha']}"
    )
    lines.append("")
    lines.append("| mode | position | combo | target | switch | correct_gain | top_wrong_gain | margin_gain |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for mode in ["repair_cumulative", "wrong_relation_cumulative", "random_cumulative"]:
        for row in best_rows(data, mode, limit=3):
            lines.append(
                "| "
                + " | ".join(
                    [
                        mode,
                        row["position"],
                        row["combo"],
                        str(row["target_n"]),
                        f"{row['target_switch']}/{row['target_n']}",
                        fmt(row["mean_correct_gain"]),
                        fmt(row["mean_top_wrong_gain"]),
                        fmt(row["mean_margin_gain"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    return lines


def ds7b_repair_detail(data: dict) -> list[str]:
    lines = ["## DS7B repair detail", ""]
    rows = [
        r
        for r in data["summary"]["by_key"].values()
        if r["mode"] == "repair_cumulative" and r["target_n"] > 0
    ]
    rows.sort(key=lambda r: (r["position"], r["combo"]))
    lines.append("| position | combo | target | switch | correct_gain | top_wrong_gain | margin_gain |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["position"],
                    row["combo"],
                    str(row["target_n"]),
                    f"{row['target_switch']}/{row['target_n']}",
                    fmt(row["mean_correct_gain"]),
                    fmt(row["mean_top_wrong_gain"]),
                    fmt(row["mean_margin_gain"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> None:
    models = {model: load_model(model) for model in MODELS}
    lines = [
        "# Phase590 Cross-Model Summary",
        "",
        "Value winner multi-layer cumulative patch audit.",
        "",
    ]
    for model in MODELS:
        lines.extend(model_section(model, models[model]))
    lines.extend(ds7b_repair_detail(models["deepseek7b"]))

    lines.append("## Objective facts")
    lines.append("")
    lines.append(
        "- Qwen3 has limited target cases and shows 1/2 switch, but random and wrong-relation controls also switch."
    )
    lines.append(
        "- GLM4 has only 1 target case and no switch; residual cumulative patch mostly suppresses both correct and wrong candidates."
    )
    lines.append(
        "- DS7B has 9 target cases; prompt_last repair strongly raises correct and top-wrong together, but target switch remains 0/9."
    )
    lines.append(
        "- DS7B query_relation repair gives small positive margin gains, but still no winner switch."
    )
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
