#!/usr/bin/env python3
"""Summarize Phase 136 long-template head re-ranking across models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase136_long_template_head_reranking")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase136_{model}_long_template_head_reranking.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    label = row.get("mode", f"H{row.get('head_id')}")
    heads = row.get("head_ids")
    if heads is not None and label in {"long_top_4", "long_top_8", "short_template_core"}:
        label = f"{label} {heads}"
    return f"{label} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 136 Cross-model Long-template Head Re-ranking Summary", ""]
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
            f"heads: {result['num_heads']}; kv_heads: {result['num_kv_heads']}; "
            f"short core: {result['short_template_core_heads']}"
        )
        lines.append("")
        lines.append("| category | audit | reference | best head | top4 | top8 | short core | all heads |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            audit = item["position_audit"]
            audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
            ref = item["reference_condition"]
            ref_text = f"{ref['component']} T{ref['target_delta']:+.2f} R{ref['max_other_delta']:+.2f} A{ref['answer_proj_delta']:+.2f}"
            best = item["head_ranking"][0] if item["head_ranking"] else None
            by_mode = {x["mode"]: x for x in item["aggregate_conditions"]}
            lines.append(
                f"| {cat} | {audit_text} | {ref_text} | {fmt(best)} | "
                f"{fmt(by_mode.get('long_top_4'))} | {fmt(by_mode.get('long_top_8'))} | "
                f"{fmt(by_mode.get('short_template_core'))} | {fmt(by_mode.get('all_heads'))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase136_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
