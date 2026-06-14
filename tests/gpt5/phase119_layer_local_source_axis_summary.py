#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase119_layer_local_source_axis")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict]) -> dict | None:
    return min(rows, key=lambda r: r["target_delta"]) if rows else None


def fmt(r: dict | None) -> str:
    if r is None:
        return "NA"
    return f"L{r['layer']} {r['site']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"


def classify(source_best: dict | None, answer_best: dict | None) -> str:
    src = source_best["target_delta"] if source_best else 0.0
    ans = answer_best["target_delta"] if answer_best else 0.0
    if src <= -3.0 and ans <= -3.0:
        return "local_source_and_answer_axes"
    if src <= -3.0:
        return "local_source_axis_found"
    if ans <= -6.0 and src > -1.5:
        return "answer_late_assembly"
    if ans <= -2.0:
        return "answer_site_dominant"
    if src <= -1.0:
        return "weak_local_source_axis"
    return "weak_or_no_local_source"


def main() -> None:
    loaded = {}
    rows = []
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase119_{model}_layer_local_source_axis.json").read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            for axis in data["axis_types"]:
                conds = [c for c in item["conditions"] if c["axis_type"] == axis]
                source_best = best([c for c in conds if c["site"] != "answer_last"])
                answer_best = best([c for c in conds if c["site"] == "answer_last"])
                object_last = best([c for c in conds if c["site"] == "object_last"])
                object_span = best([c for c in conds if c["site"] == "object_span_mean"])
                post_object = best([c for c in conds if c["site"] == "post_object_mean"])
                rows.append({
                    "model": model,
                    "category": cat,
                    "axis": axis,
                    "object_last": object_last,
                    "object_span": object_span,
                    "post_object": post_object,
                    "source_best": source_best,
                    "answer_best": answer_best,
                    "class": classify(source_best, answer_best),
                })

    first = next(iter(loaded.values()))
    lines = ["# Phase 119 Cross-model Layer-local Source Axis Discovery", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(
        f"- layers: peak-{len(first['patch_layers']) - 1} ... peak; sites: {', '.join(first['sites'])}; "
        f"rank: {first['rank']}; scale: {first['scale']}"
    )
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | axis | object_last | object_span | post_object | best source | answer_last | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {r['axis']} | {fmt(r['object_last'])} | "
            f"{fmt(r['object_span'])} | {fmt(r['post_object'])} | {fmt(r['source_best'])} | "
            f"{fmt(r['answer_best'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- Each local axis is fit at its own layer and site, then patched at that same layer and site.")
    lines.append("- Source sites are object_last, object_span_mean, and post_object_mean.")
    lines.append("- answer_last remains the readout-site baseline.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- Local source axes are selected by single-site removal, not by an explicit source-to-answer transform fit.")
    lines.append("- object_span_mean and post_object_mean patch all tokens in the group with one local mean-derived axis.")
    lines.append("- Results are DCF logits, not open generation.")
    out = OUT_DIR / "phase119_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
