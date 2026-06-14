#!/usr/bin/env python3
"""Summarize Phase 129 position-corrected gateway audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase129_position_corrected_gateway_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase129_{model}_position_corrected_gateway_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row(conditions: list[dict[str, Any]], site: str) -> dict[str, Any] | None:
    return next((x for x in conditions if x["site"] == site), None)


def fmt(item: dict[str, Any] | None) -> str:
    if item is None:
        return "NA"
    return f"T{item['target_delta']:+.2f} R{item['max_other_delta']:+.2f} A{item['answer_proj_delta']:+.2f}"


def main() -> None:
    lines = ["# Phase 129 Cross-model Position-corrected Gateway Audit Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        lines.append(f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}")
        lines.append("")
        lines.append("| category | audit | peak input | peak output | last input | last output | final norm input | final norm output |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            audit = item["position_audit"]
            audit_text = (
                f"answer_in_pre={audit['answer_in_pre_count']}, "
                f"old_mismatch={audit['old_answer_pos_mismatch_count']}, "
                f"mean_pre={audit['mean_pre_len']:.1f}"
            )
            conds = item["conditions"]
            lines.append(
                f"| {cat} | {audit_text} | "
                f"{fmt(row(conds, 'peak_block_input'))} | "
                f"{fmt(row(conds, 'peak_block_output'))} | "
                f"{fmt(row(conds, 'last_block_input'))} | "
                f"{fmt(row(conds, 'last_block_output'))} | "
                f"{fmt(row(conds, 'final_norm_input'))} | "
                f"{fmt(row(conds, 'final_norm_output'))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase129_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
