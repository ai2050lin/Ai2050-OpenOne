#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase118_causal_axis_transport_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict]) -> dict | None:
    return min(rows, key=lambda r: r["target_delta"]) if rows else None


def fmt(r: dict | None) -> str:
    if r is None:
        return "NA"
    return (
        f"L{r['patch_layer']} {r['patch_site']} "
        f"T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f} "
        f"Aproj{r['answer_axis_proj_delta']:+.2f}"
    )


def classify(object_best: dict | None, answer_best: dict | None, both_best: dict | None) -> str:
    obj = object_best["target_delta"] if object_best else 0.0
    ans = answer_best["target_delta"] if answer_best else 0.0
    both = both_best["target_delta"] if both_best else 0.0
    if ans <= -6.0 and obj > -1.0:
        return "answer_site_assembled"
    if obj <= -3.0 and ans <= -3.0:
        return "source_to_answer_supported"
    if both <= min(obj, ans) - 1.0:
        return "distributed_site_effect"
    if ans <= -2.0:
        return "answer_site_dominant"
    if obj <= -1.0:
        return "weak_source_signal"
    return "weak_or_no_closure"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase118_{model}_causal_axis_transport_closure.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            for axis in data["axis_types"]:
                conds = [c for c in item["conditions"] if c["axis_type"] == axis]
                object_best = best([c for c in conds if c["patch_site"] == "object_last"])
                answer_best = best([c for c in conds if c["patch_site"] == "answer_last"])
                both_best = best([c for c in conds if c["patch_site"] == "both"])
                rows.append({
                    "model": model,
                    "category": cat,
                    "axis": axis,
                    "varimax_selection": item["varimax_best_selection"],
                    "object_best": object_best,
                    "answer_best": answer_best,
                    "both_best": both_best,
                    "class": classify(object_best, answer_best, both_best),
                })

    first = next(iter(loaded.values()))
    lines = ["# Phase 118 Cross-model Causal Axis Transport Closure", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(
        f"- monitor layer: model boundary layer; patch layers: monitor-layer-{len(first['patch_layers']) - 1} ... monitor-layer; "
        f"rank: {first['rank']}; scale: {first['scale']}"
    )
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | axis | selected varimax single | best object_last | best answer_last | best both | class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        sel = r["varimax_selection"]
        selection = f"b{sel['basis_index']} T{sel['selection_target_delta']:+.2f} R{sel['selection_max_other_delta']:+.2f}"
        lines.append(
            f"| {r['model']} | {r['category']} | {r['axis']} | {selection} | "
            f"{fmt(r['object_best'])} | {fmt(r['answer_best'])} | {fmt(r['both_best'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- object_last tests whether the answer-site axis already has upstream causal leverage at the object token.")
    lines.append("- answer_last is the direct answer-site removal baseline.")
    lines.append("- both tests whether source and answer removals add or interfere.")
    lines.append("- Aproj is the mean answer-layer projection delta on the selected varimax axis.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- Axes are built at the monitor layer, then reused across nearby layers; layer-wise bases are not refit.")
    lines.append("- Projection closure is measured on DCF logits, not open generation.")
    lines.append("- A weak object_last effect does not prove absence of upstream encoding; it may use a different coordinate before the answer layer.")
    out = OUT_DIR / "phase118_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
