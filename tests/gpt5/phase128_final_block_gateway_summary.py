#!/usr/bin/env python3
"""Summarize Phase 128 final block gateway results across models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase128_final_block_gateway")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase128_{model}_final_block_gateway.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row['component']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def best_target_down(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not conditions:
        return None
    return min(conditions, key=lambda x: x["target_delta"])


def component_row(conditions: list[dict[str, Any]], component: str) -> dict[str, Any] | None:
    return next((x for x in conditions if x["component"] == component), None)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {model: load_result(model) for model in MODELS}
    lines = ["# Phase 128 Cross-model Final Block Gateway Summary", ""]
    for model, result in results.items():
        if result is None:
            lines.append(f"## {model}")
            lines.append("")
            lines.append("Missing result.")
            lines.append("")
            continue
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"Peak layer: L{result['peak_layer']}")
        lines.append(f"Available components: {', '.join(result['available_components'])}")
        if result["unavailable_components"]:
            lines.append(f"Unavailable components: {', '.join(result['unavailable_components'])}")
        lines.append("")
        lines.append("| category | audit | best | block input | post-attn norm input | mlp input | mlp output | block output | final norm input | final norm output |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            conds = item["conditions"]
            audit = item["position_audit"]
            audit_text = f"answer_in_pre={audit['answer_in_pre_count']}, mean_pre_len={audit['mean_pre_len']:.1f}"
            lines.append(
                f"| {cat} | {audit_text} | {fmt(best_target_down(conds))} | "
                f"{fmt(component_row(conds, 'block_input'))} | "
                f"{fmt(component_row(conds, 'post_attention_norm_input'))} | "
                f"{fmt(component_row(conds, 'mlp_input'))} | "
                f"{fmt(component_row(conds, 'mlp_output'))} | "
                f"{fmt(component_row(conds, 'block_output'))} | "
                f"{fmt(component_row(conds, 'final_norm_input'))} | "
                f"{fmt(component_row(conds, 'final_norm_output'))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase128_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
