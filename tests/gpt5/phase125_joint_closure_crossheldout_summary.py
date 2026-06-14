#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/gpt5_phase125_joint_closure_crossheldout")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONTROL_SETS = {"object_control", "random_control", "low_pre_value_control"}


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda x: x["target_delta"]) if rows else None


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return (
        f"{row.get('condition','')} {row.get('set_name','')} k{row.get('set_size','')} "
        f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} "
        f"A{row['answer_proj_delta']:+.2f} ratio{row.get('effect_ratio_vs_residual_ref',0.0):+.2f}"
    )


def classify(residual: dict[str, Any] | None, head: dict[str, Any] | None, combo: dict[str, Any] | None,
             control: dict[str, Any] | None, mlp: dict[str, Any] | None) -> str:
    if residual is None:
        return "missing"
    if abs(residual["target_delta"]) < 0.5:
        return "weak_residual_reference"
    control_t = control["target_delta"] if control else 0.0
    combo_ratio = combo.get("effect_ratio_vs_residual_ref", 0.0) if combo else 0.0
    head_ratio = head.get("effect_ratio_vs_residual_ref", 0.0) if head else 0.0
    if combo and combo_ratio >= 0.75 and combo["target_delta"] <= control_t - 0.5:
        return "module_combo_closure_candidate"
    if head and head_ratio >= 0.5 and head["target_delta"] <= control_t - 0.5:
        return "head_set_generalizes"
    if mlp and mlp.get("effect_ratio_vs_residual_ref", 0.0) >= 0.5:
        return "mlp_subspace_generalizes"
    if combo and combo["target_delta"] <= -0.5:
        return "weak_combo_partial"
    return "not_closed"


def main() -> None:
    loaded = {}
    rows = []
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase125_{model}_joint_closure_crossheldout.json").read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            conds = item["evaluation_conditions"]
            residual = best([x for x in conds if x["condition"] == "residual_pre_reference"])
            head = best([x for x in conds if x["condition"] == "head_set_only" and x["set_name"] not in CONTROL_SETS])
            combo = best([x for x in conds if x["condition"] == "head_set_plus_pre_mlp"])
            control = best([x for x in conds if x["set_name"] in CONTROL_SETS])
            mlp = best([x for x in conds if x["condition"] == "pre_mlp_subspace_only"])
            rows.append({
                "model": model,
                "category": cat,
                "residual": residual,
                "head": head,
                "combo": combo,
                "control": control,
                "mlp": mlp,
                "class": classify(residual, head, combo, control, mlp),
            })

    first = next(iter(loaded.values()))
    layers = "; ".join(
        f"{model}: L{data['patch_layers'][0]}-L{data['patch_layers'][-1]} monitor L{data['monitor_layer']}"
        for model, data in loaded.items()
    )
    lines = ["# Phase 125 Cross-model Joint Closure Cross-heldout", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"split train/selection/evaluation objects: {first['train_objects_per_category']}/"
        f"{first['selection_objects_per_category']}/{first['evaluation_objects_per_category']}; "
        f"evaluation prompts/category: {first['evaluation_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- layers: {layers}; rank: {first['rank']}; set sizes: {first['set_sizes']}; candidate pool: {first['candidate_pool']}")
    lines.append("")
    lines.append("| model | category | residual reference | best head only | best head+MLP | best control | pre-MLP only | class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['category']} | {fmt(row['residual'])} | {fmt(row['head'])} | "
            f"{fmt(row['combo'])} | {fmt(row['control'])} | {fmt(row['mlp'])} | {row['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- Head/MLP choices are selected on the selection split and reported on disjoint evaluation objects.")
    lines.append("- ratio is target_delta divided by the selected residual pre-answer reference on evaluation.")
    lines.append("- Controls include object head set, random head set, and value-aligned low-pre-attention set.")
    out = OUT_DIR / "phase125_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
