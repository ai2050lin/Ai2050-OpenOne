#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/gpt5_phase127_upstream_residual_carry_backtrace")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda x: x["target_delta"]) if rows else None


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"L{row['patch_layer']} T{row['target_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def classify(input_metrics: dict[str, Any], output_metrics: dict[str, Any], final_in: dict[str, Any] | None,
             final_out: dict[str, Any] | None) -> str:
    if output_metrics["best_target_delta"] is None or output_metrics["best_target_delta"] > -0.5:
        return "weak_residual_path"
    if input_metrics["first_active_layer"] is not None and input_metrics["first_active_layer"] <= output_metrics["first_active_layer"]:
        if final_in and final_out and final_out["target_delta"] <= final_in["target_delta"] - 0.75:
            return "carry_plus_final_reemergence"
        return "upstream_residual_carry"
    return "late_output_emergence"


def main() -> None:
    loaded = {}
    rows = []
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase127_{model}_upstream_residual_carry_backtrace.json").read_text(encoding="utf-8"))
        loaded[model] = data
        final_layer = data["patch_layers"][-1]
        for cat, item in data["category_results"].items():
            conds = item["conditions"]
            inp = [r for r in conds if r["component"] == "layer_input"]
            out = [r for r in conds if r["component"] == "layer_output"]
            final_in = next((r for r in inp if r["patch_layer"] == final_layer), None)
            final_out = next((r for r in out if r["patch_layer"] == final_layer), None)
            mi = item["curve_metrics"]["layer_input"]
            mo = item["curve_metrics"]["layer_output"]
            rows.append({
                "model": model,
                "category": cat,
                "input_onset": mi["first_active_layer"],
                "output_onset": mo["first_active_layer"],
                "best_input": best(inp),
                "best_output": best(out),
                "final_input": final_in,
                "final_output": final_out,
                "class": classify(mi, mo, final_in, final_out),
            })
    first = next(iter(loaded.values()))
    layers = "; ".join(
        f"{model}: L{data['patch_layers'][0]}-L{data['patch_layers'][-1]} monitor L{data['monitor_layer']}"
        for model, data in loaded.items()
    )
    lines = ["# Phase 127 Cross-model Upstream Residual Carry Backtrace", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- layers: {layers}; rank: {first['rank']}; onset threshold: -0.5")
    lines.append("")
    lines.append("| model | category | input onset | output onset | best input | best output | final input | final output | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | L{r['input_onset']} | L{r['output_onset']} | "
            f"{fmt(r['best_input'])} | {fmt(r['best_output'])} | {fmt(r['final_input'])} | "
            f"{fmt(r['final_output'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- onset is the first scanned layer with target_delta <= -0.5.")
    lines.append("- final output re-emergence means the final scanned layer output is much stronger than its input.")
    out = OUT_DIR / "phase127_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
