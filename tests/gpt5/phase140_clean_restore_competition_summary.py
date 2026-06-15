#!/usr/bin/env python3
"""Summarize Phase 140 clean restore competition results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase140_clean_restore_competition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase140_{model}_clean_restore_competition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    prefix = f"{row['site']} s{row['scale']} " if "site" in row else ""
    rec = f" rec{row['recovery_ratio']:+.2f}" if "recovery_ratio" in row else ""
    clean = f" clean{row['clean_restore_score']:+.2f}" if "clean_restore_score" in row else ""
    comp = ""
    if "top_competitor_after_restore" in row:
        c, v = row["top_competitor_after_restore"]
        comp = f" comp={c}:{v:+.2f}"
    return f"{prefix}T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f}{rec}{clean}{comp}"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 140 Cross-model Clean Restore Competition Summary", ""]
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
            f"lambda: {result['lambda_release']}"
        )
        lines.append("")
        lines.append("| category | transfer | remove | best target | best clean | first tokens |")
        lines.append("|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
            tokens = ", ".join(f"{x['token']}:{x['rate']:.2f}" for x in item["best_clean_restore"]["token_audit"][:3])
            lines.append(
                f"| {cat} | {transfer} | {fmt(item['remove'])} | {fmt(item['best_by_target'])} | "
                f"{fmt(item['best_clean_restore'])} | {tokens} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase140_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
