#!/usr/bin/env python3
"""Summarize Phase 139 restore/swap calibration across models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase139_restore_swap_calibration")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase139_{model}_restore_swap_calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    prefix = f"{row['site']} s{row['scale']} " if "site" in row else ""
    rec = f" rec{row['recovery_ratio']:+.2f}" if "recovery_ratio" in row else ""
    extra = ""
    if "swap_category" in row:
        extra = f" swap={row['swap_category']} SΔ{row['swap_category_delta']:+.2f}"
    return f"{prefix}T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f}{rec}{extra}"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 139 Cross-model Restore/Swap Calibration Summary", ""]
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
            f"restore_sites: {result['restore_sites']}"
        )
        lines.append("")
        lines.append("| category | transfer | remove | best restore | best sample swap |")
        lines.append("|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
            lines.append(
                f"| {cat} | {transfer} | {fmt(item['remove'])} | "
                f"{fmt(item['best_restore'])} | {fmt(item['best_sample_swap'])} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase139_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
