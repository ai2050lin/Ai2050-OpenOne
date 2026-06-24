#!/usr/bin/env python3
"""Summarize Phase593 atlas-guided causal patch validation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase593_atlas_guided_causal_patch")
MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT = ROOT / "phase593_cross_model_summary.md"


def load(model: str) -> dict:
    path = ROOT / f"phase593_{model}_atlas_guided_causal_patch_confirm.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def section(model: str, data: dict) -> list[str]:
    lines = [f"## {model}", ""]
    lines.append(
        f"- target_rows={data['n_target_rows']}, alpha={data['alpha']}, "
        f"nodes={[(n['position'], n['layer']) for n in data['nodes']]}"
    )
    lines.append("")
    lines.append("| rank | node | mode | switch | margin_gain | specific_gain | common | correct_specific | positive_margin |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for i, row in enumerate(data["summary"]["best"][:12], start=1):
        node = f"{row['position']} L{row['layer']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    node,
                    row["mode"],
                    f"{row['switch']}/{row['n']}",
                    fmt(row["mean_margin_gain"]),
                    fmt(row["mean_specific_margin_gain"]),
                    fmt(row["mean_common_delta"]),
                    fmt(row["mean_correct_specific"]),
                    f"{row['positive_margin_rate']:.2f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def mode_best(data: dict, mode: str) -> dict | None:
    rows = [r for r in data["summary"]["by_key"].values() if r["mode"] == mode]
    if not rows:
        return None
    return max(rows, key=lambda r: (r["switch"], r["mean_margin_gain"], r["mean_correct_specific"]))


def main() -> None:
    models = {m: load(m) for m in MODELS}
    lines = [
        "# Phase593 Cross-Model Summary",
        "",
        "Atlas-guided causal patch validation. This tests whether Phase592 Level-2 projection nodes become causal repair nodes.",
        "",
    ]
    for model in MODELS:
        lines.extend(section(model, models[model]))

    lines.append("## Mode Best")
    lines.append("")
    lines.append("| model | mode | node | switch | margin_gain | common | correct_specific |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for model, data in models.items():
        for mode in ["raw", "specific_only", "specific_norm_raw", "common_only", "common_norm_raw", "random_same_norm"]:
            row = mode_best(data, mode)
            if row:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            model,
                            mode,
                            f"{row['position']} L{row['layer']}",
                            f"{row['switch']}/{row['n']}",
                            fmt(row["mean_margin_gain"]),
                            fmt(row["mean_common_delta"]),
                            fmt(row["mean_correct_specific"]),
                        ]
                    )
                    + " |"
                )
    lines.append("")
    lines.append("## Objective facts")
    lines.append("")
    lines.append("- Qwen3 has limited target rows but specific_norm_raw prompt_last patches reach 2/5 switch.")
    lines.append("- GLM4 has no switch and near-zero margin gains.")
    lines.append("- DS7B has 21 target rows; no tested atlas node gives reliable positive margin or winner repair.")
    lines.append("- DS7B observed 1/21 switches are not evidence of repair because their mean margin gains are negative and common/random controls can also switch.")
    lines.append("- Phase592 projection nodes therefore remain candidates; they are not yet upgraded to robust causal repair nodes.")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
