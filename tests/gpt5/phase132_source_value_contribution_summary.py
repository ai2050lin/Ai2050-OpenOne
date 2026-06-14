#!/usr/bin/env python3
"""Summarize Phase 132 source-specific value contribution results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase132_source_value_contribution")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase132_{model}_source_value_contribution.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    if "source_group" in row:
        label = f"{row['source_group']}:{row['head_mode']}"
    else:
        label = row.get("component", "NA")
    return f"{label} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def find(rows: list[dict[str, Any]], source_group: str, head_mode: str = "all_heads") -> dict[str, Any] | None:
    return next((x for x in rows if x["source_group"] == source_group and x["head_mode"] == head_mode), None)


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda x: x["target_delta"]) if rows else None


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 132 Cross-model Source-specific Value Contribution Summary", ""]
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
            f"heads: {result['num_heads']}; kv_heads: {result['num_kv_heads']}"
        )
        lines.append("")
        lines.append("| category | audit | reference | best | object all | post-object all | all-pre all | self all | all-pre top |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            audit = item["position_audit"]
            audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
            rows = item["conditions"]
            lines.append(
                f"| {cat} | {audit_text} | {fmt(item['reference_condition'])} | {fmt(best(rows))} | "
                f"{fmt(find(rows, 'object_span'))} | "
                f"{fmt(find(rows, 'post_object_pre_answer'))} | "
                f"{fmt(find(rows, 'all_pre_answer'))} | "
                f"{fmt(find(rows, 'self'))} | "
                f"{fmt(find(rows, 'all_pre_answer', 'top_heads'))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase132_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
