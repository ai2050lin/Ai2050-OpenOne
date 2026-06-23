#!/usr/bin/env python3
"""Summarize Phase592 relation-specific ranking atlas results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase592_relation_specific_ranking_atlas")
MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT = ROOT / "phase592_cross_model_summary.md"


def load(model: str) -> dict:
    path = ROOT / f"phase592_{model}_relation_specific_ranking_atlas_confirm.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def model_section(model: str, data: dict) -> list[str]:
    lines = [f"## {model}", ""]
    lines.append(
        f"- cases={data['n_cases']}, target_cases={data['target_n']}, n_layers={data['n_layers']}, threshold={data['cross_threshold']}"
    )
    lines.append("")
    lines.append("| rank | position | layer | bucket | spec_margin | correct_specific | old_top_wrong_specific | common | pos_rate |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(data["summary"]["best"][:12], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    row["position"],
                    str(row["layer"]),
                    row["bucket"],
                    fmt(row["mean_specific_margin"]),
                    fmt(row["mean_correct_specific"]),
                    fmt(row["mean_old_top_wrong_specific"]),
                    fmt(row["mean_common"]),
                    f"{row['positive_specific_rate']:.2f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("First crossing by position:")
    lines.append("")
    for pos, item in data["summary"]["first_crossing_by_position"].items():
        if item is None:
            lines.append(f"- {pos}: none")
        else:
            lines.append(
                f"- {pos}: L{item['layer']} {item['bucket']} "
                f"spec_margin={item['mean_specific_margin']:.3f}, "
                f"cspec={item['mean_correct_specific']:.3f}, "
                f"wspec={item['mean_old_top_wrong_specific']:.3f}"
            )
    lines.append("")
    lines.append(
        f"Atlas nodes={len(data['atlas']['nodes'])}, edges={len(data['atlas']['edges'])}"
    )
    lines.append("")
    return lines


def main() -> None:
    models = {m: load(m) for m in MODELS}
    lines = [
        "# Phase592 Cross-Model Summary",
        "",
        "Relation-specific ranking factor atlas. Metrics are projection-level evidence, not causal repair.",
        "",
    ]
    for model in MODELS:
        lines.extend(model_section(model, models[model]))
    lines.append("## Objective facts")
    lines.append("")
    lines.append("- Qwen3 ranking projection peaks at late prompt_last, with query_category also strong.")
    lines.append("- GLM4 ranking projection is weaker and concentrated at late prompt_last.")
    lines.append("- DS7B ranking projection is distributed: rule_value L26, prompt_last L26, rule_relation mid-late layers, and query_relation mid layers.")
    lines.append("- DS7B has several non-prompt_last positions above threshold, supporting an atlas view rather than a single-point mechanism.")
    lines.append("- These are Level 2 decodable projection edges. They locate candidates for causal testing; they do not yet prove repair.")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
