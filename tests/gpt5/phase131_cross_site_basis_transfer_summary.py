#!/usr/bin/env python3
"""Summarize Phase 131 cross-site basis transfer results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/gpt5_phase131_cross_site_basis_transfer")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase131_{model}_cross_site_basis_transfer.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row['component']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def comp(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((x for x in rows if x["component"] == name), None)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 131 Cross-model Cross-site Basis Transfer Summary", ""]
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
        lines.append("| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cat, item in result["category_results"].items():
            audit = item["position_audit"]
            audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
            rows = item["cross_site_conditions"]
            lines.append(
                f"| {cat} | {audit_text} | {fmt(item['reference_condition'])} | "
                f"{fmt(comp(rows, 'last_attention_output_answer'))} | "
                f"{fmt(comp(rows, 'last_mlp_input_answer'))} | "
                f"{fmt(comp(rows, 'last_mlp_output_answer'))} | "
                f"{fmt(comp(rows, 'last_block_output_answer'))} | "
                f"{fmt(comp(rows, 'final_norm_output_answer'))} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase131_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
