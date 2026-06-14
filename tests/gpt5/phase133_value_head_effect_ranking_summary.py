#!/usr/bin/env python3
"""Summarize Phase 133 value contribution head-effect ranking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase133_value_head_effect_ranking")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase133_{model}_value_head_effect_ranking.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    label = row.get("mode", f"H{row.get('head_id')}")
    return f"{label} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def mode(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((x for x in rows if x["mode"] == name), None)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 133 Cross-model Value Head Effect Ranking Summary", ""]
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
        lines.append("| category | audit | reference | best head | top1 | top2 | top4 | top8 | all heads |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            audit = item["position_audit"]
            audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
            ref = item["reference_condition"]
            ref_text = f"{ref['component']} T{ref['target_delta']:+.2f} R{ref['max_other_delta']:+.2f} A{ref['answer_proj_delta']:+.2f}"
            rows = item["aggregate_conditions"]
            best_head = item["head_ranking"][0] if item["head_ranking"] else None
            lines.append(
                f"| {cat} | {audit_text} | {ref_text} | {fmt(best_head)} | "
                f"{fmt(mode(rows, 'top_causal_1'))} | {fmt(mode(rows, 'top_causal_2'))} | "
                f"{fmt(mode(rows, 'top_causal_4'))} | {fmt(mode(rows, 'top_causal_8'))} | "
                f"{fmt(mode(rows, 'all_heads'))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase133_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
