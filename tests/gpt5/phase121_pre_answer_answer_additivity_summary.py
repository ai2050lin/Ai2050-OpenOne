#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase121_pre_answer_answer_additivity")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict]) -> dict | None:
    return min(rows, key=lambda r: r["target_delta"]) if rows else None


def fmt(r: dict | None) -> str:
    if r is None:
        return "NA"
    return f"L{r['layer']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"


def classify(pre: dict | None, answer: dict | None, combined: dict | None) -> str:
    if not pre or not answer or not combined:
        return "missing"
    pt = pre["target_delta"]
    at = answer["target_delta"]
    ct = combined["target_delta"]
    if ct <= at - 1.0 and ct <= pt - 1.0:
        return "additive_or_independent"
    if abs(ct - at) <= 0.75:
        return "answer_absorbs_pre"
    if ct > at + 1.0:
        return "interference"
    if at <= -6.0 and pt > -3.0:
        return "answer_dominant"
    return "mixed_small_effect"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase121_{model}_pre_answer_answer_additivity.json").read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            for axis in data["axis_types"]:
                conds = [c for c in item["conditions"] if c["axis_type"] == axis]
                pre = best([c for c in conds if c["patch_mode"] == "pre_only"])
                ans = best([c for c in conds if c["patch_mode"] == "answer_only"])
                combo = best([c for c in conds if c["patch_mode"] == "pre_plus_answer"])
                rows.append({
                    "model": model,
                    "category": cat,
                    "axis": axis,
                    "pre": pre,
                    "answer": ans,
                    "combined": combo,
                    "combined_minus_answer": None if not ans or not combo else combo["target_delta"] - ans["target_delta"],
                    "class": classify(pre, ans, combo),
                })

    first = next(iter(loaded.values()))
    lines = ["# Phase 121 Cross-model Pre-answer and Answer Additivity", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(
        f"- layers: peak-{len(first['patch_layers']) - 1} ... peak; rank: {first['rank']}; "
        f"scale: {first['scale']}; axis types: {', '.join(first['axis_types'])}"
    )
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | axis | best pre-only | best answer-only | best combined | combined-answer | class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        diff = "NA" if r["combined_minus_answer"] is None else f"{r['combined_minus_answer']:+.2f}"
        lines.append(
            f"| {r['model']} | {r['category']} | {r['axis']} | {fmt(r['pre'])} | "
            f"{fmt(r['answer'])} | {fmt(r['combined'])} | {diff} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- combined-answer is target_delta(combined) minus target_delta(answer-only). Negative means combined is stronger than answer-only.")
    lines.append("- answer_absorbs_pre means combined is close to answer-only, so pre-answer adds little under this patch.")
    lines.append("- additive_or_independent means combined is at least 1 logit stronger than answer-only and pre-only.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- Pre-answer and answer axes are selected independently at the same layer.")
    lines.append("- This does not identify the attention/MLP writer of either field.")
    lines.append("- Results are DCF logits, not open generation.")
    out = OUT_DIR / "phase121_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
