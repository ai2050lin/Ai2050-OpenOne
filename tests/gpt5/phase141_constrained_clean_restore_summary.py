#!/usr/bin/env python3
"""Summarize Phase 141 constrained clean restore across models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase141_constrained_clean_restore")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase141_{model}_constrained_clean_restore.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NONE"
    comp = ""
    if "top_competitor" in row:
        comp = f" comp={row['top_competitor'][0]}:{row['top_competitor'][1]:+.2f}"
    return (
        f"{row['site']} s{row['scale']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} "
        f"rec{row['recovery_ratio']:+.2f} clean={row['is_constrained_clean']}{comp}"
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 141 Cross-model Constrained Clean Restore Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        lines.append(
            f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; "
            f"rank: {result['rank']}; train/test: "
            f"{result['train_objects_per_category']}/{result['test_objects_per_category']}; "
            f"threshold: {result['release_threshold']}"
        )
        lines.append("")
        lines.append("| category | transfer | remove | clean count | constrained | min release | best target |")
        lines.append("|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
            rem = f"T{item['remove']['target_delta']:+.2f} R{item['remove']['max_other_delta']:+.2f}"
            lines.append(
                f"| {cat} | {transfer} | {rem} | {item['constrained_clean_count']} | "
                f"{fmt(item['best_constrained_clean'])} | {fmt(item['best_min_release'])} | {fmt(item['best_target'])} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase141_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
