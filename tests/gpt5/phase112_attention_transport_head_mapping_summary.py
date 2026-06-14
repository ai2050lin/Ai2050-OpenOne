#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase112_attention_transport_head_mapping")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(c: dict | None) -> str:
    if c is None:
        return "NA"
    return (
        f"L{c['patch_layer']} H{c['head_id']} obj{c['object_mass']:.3f} "
        f"T{c.get('target_delta', 0.0):+.2f} A{c.get('answer_transport_proj_delta', 0.0):+.2f}"
    )


def classify(best_t: dict | None, best_p: dict | None, top: dict | None) -> str:
    if best_t is None:
        return "no_data"
    if best_t["target_delta"] <= -1.0 and best_t["object_mass"] >= 0.05:
        return "candidate_transport_head"
    if best_t["target_delta"] <= -0.5:
        return "weak_candidate"
    if best_p is not None and best_p["answer_transport_proj_delta"] <= -0.5 and best_t["target_delta"] > -0.25:
        return "projection_only"
    if top is not None and top["object_mass"] < 0.02:
        return "low_source_attention"
    return "weak"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase112_{model}_attention_transport_head_mapping.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            selected = item["selected_heads"]
            conds = item["conditions"]
            top = selected[0] if selected else None
            best_t = min(conds, key=lambda c: c["target_delta"]) if conds else None
            best_p = min(conds, key=lambda c: c["answer_transport_proj_delta"]) if conds else None
            rows.append({
                "model": model,
                "category": cat,
                "top": top,
                "best_t": best_t,
                "best_p": best_p,
                "class": classify(best_t, best_p, top),
            })

    first = next(iter(loaded.values()))
    lines = ["# Phase 112 Cross-model Attention Transport Head Mapping", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts per category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- layers: peak-{len(first['patch_layers']) - 1} ... peak; selected heads per category: {first['top_k_heads']}")
    lines.append("- source score: answer_last attention mass to object_span + object_last")
    lines.append("- intervention: zero selected head slice at answer_last before o_proj")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | top source head | strongest target-down head | strongest projection-down head | class |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {fmt(r['top'])} | {fmt(r['best_t'])} | "
            f"{fmt(r['best_p'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Objective Reading Rules")
    lines.append("- candidate_transport_head means a high object-source attention head also causes target logits to drop when ablated.")
    lines.append("- projection_only means the monitored T_c projection moves but logits do not.")
    lines.append("- low_source_attention means answer_last barely attends to object sources in the scanned layers.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- Head ablation zeroes one head slice at answer_last before o_proj; it does not separate Q/K/V causes.")
    lines.append("- Attention mass is only a candidate selector, not a causal metric.")
    lines.append("- This phase does not yet perform value transplant or generation audit.")
    out = OUT_DIR / "phase112_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
