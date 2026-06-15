#!/usr/bin/env python3
"""Summarize Phase 142 support/suppressor and time-path tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase142_support_suppressor_timepath")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase142_{model}_support_suppressor_timepath.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NONE"
    comp = row.get("top_competitor", ["NA", 0.0])
    return (
        f"L{row['layer_id']} {row['mode']} {row['site']} s{row['scale']} "
        f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} "
        f"rec{row['recovery_ratio']:+.2f} clean={row['is_constrained_clean']} "
        f"comp={comp[0]}:{comp[1]:+.2f}"
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 142 Cross-model Support/Suppressor Timepath Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        lines.append(
            f"True last layer: L{result['true_last_layer']}; train/test: "
            f"{result['train_objects_per_category']}/{result['test_objects_per_category']}; "
            f"offsets: {result['layer_offsets']}; sites: {result['restore_sites']}; modes: {result['modes']}"
        )
        lines.append("")
        lines.append("| category@layer | transfer | remove | clean count | best clean | best support | best joint |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, item in result["category_results"].items():
            transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
            rem = f"T{item['remove']['target_delta']:+.2f} R{item['remove']['max_other_delta']:+.2f}"
            lines.append(
                f"| {key} | {transfer} | {rem} | {item['constrained_clean_count']} | "
                f"{fmt(item['best_constrained_clean'])} | {fmt(item['best_support'])} | {fmt(item['best_joint'])} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase142_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
