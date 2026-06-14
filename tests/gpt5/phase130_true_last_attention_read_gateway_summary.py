#!/usr/bin/env python3
"""Summarize Phase 130 true-last attention read gateway results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase130_true_last_attention_read_gateway")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase130_{model}_true_last_attention_read_gateway.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    label = row.get("component")
    if label is None and "head_id" in row:
        label = f"H{row['head_id']} pre{row.get('pre_answer_mass', 0.0):.3f}"
    return f"{label} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def component(conditions: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((x for x in conditions if x["component"] == name), None)


def best(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(conditions, key=lambda x: x["target_delta"]) if conditions else None


def main() -> None:
    lines = ["# Phase 130 Cross-model True-last Attention Read Gateway Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        lines.append(f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; heads: {result['num_heads']}")
        lines.append("")
        lines.append("| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer | best head ablation |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            audit = item["position_audit"]
            audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
            comps = item["answer_component_conditions"]
            lines.append(
                f"| {cat} | {audit_text} | {fmt(item['reference_condition'])} | "
                f"{fmt(component(comps, 'last_attention_output_answer'))} | "
                f"{fmt(component(comps, 'last_mlp_input_answer'))} | "
                f"{fmt(component(comps, 'last_mlp_output_answer'))} | "
                f"{fmt(component(comps, 'last_block_output_answer'))} | "
                f"{fmt(component(comps, 'final_norm_output_answer'))} | "
                f"{fmt(best(item['head_ablation_conditions']))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase130_cross_model_summary.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
