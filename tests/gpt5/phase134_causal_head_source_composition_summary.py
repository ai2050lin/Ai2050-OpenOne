#!/usr/bin/env python3
"""Summarize Phase 134 causal-head source composition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase134_causal_head_source_composition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase134_{model}_causal_head_source_composition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row['source_group']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def src(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((x for x in rows if x["source_group"] == name), None)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 134 Cross-model Causal Head Source Composition Summary", ""]
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
            f"causal heads: {result['causal_heads']}"
        )
        lines.append("")
        lines.append("| category | audit | reference | best | pre-object | object | bridge | structural | tail | all-pre |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            audit = item["position_audit"]
            audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
            ref = item["reference_condition"]
            ref_text = f"{ref['component']} T{ref['target_delta']:+.2f} R{ref['max_other_delta']:+.2f} A{ref['answer_proj_delta']:+.2f}"
            rows = item["conditions"]
            best = min(rows, key=lambda x: x["target_delta"]) if rows else None
            lines.append(
                f"| {cat} | {audit_text} | {ref_text} | {fmt(best)} | "
                f"{fmt(src(rows, 'pre_object'))} | {fmt(src(rows, 'object_span'))} | "
                f"{fmt(src(rows, 'object_to_template_bridge'))} | {fmt(src(rows, 'post_object_structural'))} | "
                f"{fmt(src(rows, 'answer_prompt_tail'))} | {fmt(src(rows, 'all_pre_answer'))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase134_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
