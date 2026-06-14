#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/gpt5_phase126_residual_gap_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]
COMPONENTS = ["layer_input", "attention_output", "mlp_output", "layer_output", "attention_plus_mlp"]


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda x: x["target_delta"]) if rows else None


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    ratio = ""
    if "effect_ratio_vs_layer_output" in row:
        ratio = f" ratio{row['effect_ratio_vs_layer_output']:+.2f}"
    return f"L{row['patch_layer']} {row['component']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}{ratio}"


def classify(layer_out: dict[str, Any] | None, attn: dict[str, Any] | None, mlp: dict[str, Any] | None,
             combo: dict[str, Any] | None, layer_in: dict[str, Any] | None) -> str:
    if not layer_out or abs(layer_out["target_delta"]) < 0.5:
        return "weak_residual_output"
    combo_ratio = combo.get("effect_ratio_vs_layer_output", 0.0) if combo else 0.0
    if combo and combo_ratio >= 0.7:
        return "module_outputs_explain_gap"
    if combo and combo_ratio >= 0.35:
        return "partial_module_gap"
    if layer_in and layer_in["target_delta"] <= layer_out["target_delta"] * 0.7:
        return "upstream_carry_candidate"
    if attn and mlp and min(attn["target_delta"], mlp["target_delta"]) > -0.5:
        return "residual_carry_or_norm_candidate"
    return "unresolved_gap"


def main() -> None:
    loaded = {}
    rows = []
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase126_{model}_residual_gap_decomposition.json").read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            conds = item["conditions"]
            picks = {c: best([x for x in conds if x["component"] == c]) for c in COMPONENTS}
            rows.append({
                "model": model,
                "category": cat,
                **picks,
                "class": classify(picks["layer_output"], picks["attention_output"], picks["mlp_output"], picks["attention_plus_mlp"], picks["layer_input"]),
            })

    first = next(iter(loaded.values()))
    layers = "; ".join(
        f"{model}: L{data['patch_layers'][0]}-L{data['patch_layers'][-1]} monitor L{data['monitor_layer']}"
        for model, data in loaded.items()
    )
    lines = ["# Phase 126 Cross-model Residual Gap Decomposition", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- layers: {layers}; rank: {first['rank']}; components: {', '.join(first['components'])}")
    lines.append("")
    lines.append("| model | category | layer input | attention output | MLP output | layer output | attention+MLP | class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {fmt(r['layer_input'])} | {fmt(r['attention_output'])} | "
            f"{fmt(r['mlp_output'])} | {fmt(r['layer_output'])} | {fmt(r['attention_plus_mlp'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- layer_output is the residual stream after the full block and acts as the residual reference.")
    lines.append("- attention+MLP ratio is measured against the layer_output condition at its own best row.")
    lines.append("- A is answer projection delta at the peak answer site.")
    out = OUT_DIR / "phase126_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
