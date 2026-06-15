#!/usr/bin/env python3
"""Summarize Phase 138 mechanism transfer closure across models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase138_mechanism_transfer_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase138_{model}_mechanism_transfer_closure.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    text = f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"
    if "swap_category" in row:
        text += f" swap={row['swap_category']} SΔ{row['swap_category_delta']:+.2f}"
    return text


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 138 Cross-model Mechanism Transfer Closure Summary", ""]
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
            f"{result['train_objects_per_category']}/{result['test_objects_per_category']}"
        )
        lines.append("")
        lines.append("| category | transfer | remove | restore | recovery | swap |")
        lines.append("|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            cond = item["conditions"]
            transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
            lines.append(
                f"| {cat} | {transfer} | {fmt(cond['remove'])} | {fmt(cond['restore'])} | "
                f"{item['restore_recovery_ratio']:+.2f} | {fmt(cond['swap'])} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase138_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
