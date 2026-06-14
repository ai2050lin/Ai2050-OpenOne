#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase122_pre_answer_to_answer_projection_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best_target(rows: list[dict]) -> dict | None:
    return min(rows, key=lambda r: r["target_delta"]) if rows else None


def best_proj(rows: list[dict]) -> dict | None:
    return min(rows, key=lambda r: r["answer_proj_delta"]) if rows else None


def fmt(r: dict | None) -> str:
    if r is None:
        return "NA"
    return f"L{r['patch_layer']} {r['patch_mode']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f} Aproj{r['answer_proj_delta']:+.2f}"


def classify(pre: dict | None, answer: dict | None, combined: dict | None, pre_proj: dict | None) -> str:
    if not pre or not answer or not combined:
        return "missing"
    combo_extra = combined["target_delta"] - answer["target_delta"]
    pre_proj_delta = pre_proj["answer_proj_delta"] if pre_proj else 0.0
    if combo_extra <= -1.0 and pre_proj_delta <= -0.5:
        return "pre_writes_answer_projection"
    if combo_extra <= -1.0:
        return "pre_adds_without_projection_drop"
    if abs(combo_extra) <= 0.75:
        return "answer_absorbs_pre"
    return "mixed"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase122_{model}_pre_answer_to_answer_projection_closure.json").read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            for axis in data["axis_types"]:
                conds = [c for c in item["conditions"] if c["axis_type"] == axis]
                pre = best_target([c for c in conds if c["patch_mode"] == "pre_remove"])
                ans = best_target([c for c in conds if c["patch_mode"] == "answer_remove"])
                combo = best_target([c for c in conds if c["patch_mode"] == "pre_plus_answer"])
                pre_proj = best_proj([c for c in conds if c["patch_mode"] == "pre_remove"])
                rows.append({
                    "model": model,
                    "category": cat,
                    "axis": axis,
                    "pre": pre,
                    "answer": ans,
                    "combined": combo,
                    "pre_proj": pre_proj,
                    "combo_minus_answer": None if not ans or not combo else combo["target_delta"] - ans["target_delta"],
                    "class": classify(pre, ans, combo, pre_proj),
                })

    first = next(iter(loaded.values()))
    lines = ["# Phase 122 Cross-model Pre-answer to Answer Projection Closure", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- patch layers: {first['patch_layers']}; monitor layer: L{first['monitor_layer']}; rank: {first['rank']}; scale: {first['scale']}")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | axis | best pre | best answer | best combined | combo-answer | strongest pre answer-proj drop | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        diff = "NA" if r["combo_minus_answer"] is None else f"{r['combo_minus_answer']:+.2f}"
        lines.append(
            f"| {r['model']} | {r['category']} | {r['axis']} | {fmt(r['pre'])} | {fmt(r['answer'])} | "
            f"{fmt(r['combined'])} | {diff} | {fmt(r['pre_proj'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- Aproj is the peak answer_last projection delta on the answer-site axis/subspace.")
    lines.append("- pre_writes_answer_projection requires combined to beat answer-only and pre_remove to lower answer projection.")
    lines.append("- pre_adds_without_projection_drop means extra logit effect exists but was not visible as mean answer-axis projection loss.")
    out = OUT_DIR / "phase122_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
