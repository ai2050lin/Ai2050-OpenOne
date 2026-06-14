#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase114_answer_site_causal_subspace")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(conds: list[dict], kind: str) -> dict | None:
    xs = [c for c in conds if c["kind"] == kind]
    return min(xs, key=lambda c: c["target_delta"]) if xs else None


def fmt(c: dict | None) -> str:
    if c is None:
        return "NA"
    return f"r{c['rank']} s{c['scale']} T{c['target_delta']:+.2f} R{c['max_other_delta']:+.2f}"


def classify(tc: dict | None, sub: dict | None, rnd: dict | None) -> str:
    if tc is None or sub is None:
        return "no_data"
    if abs(tc["target_delta"]) < 0.5 and abs(sub["target_delta"]) < 0.5:
        return "weak"
    if rnd and rnd["target_delta"] <= sub["target_delta"] + 0.25:
        return "control_sensitive"
    if sub["target_delta"] <= tc["target_delta"] - 1.0:
        return "subspace_stronger"
    if sub["target_delta"] <= tc["target_delta"] - 0.25:
        return "subspace_slightly_stronger"
    if tc["target_delta"] <= sub["target_delta"] - 0.5:
        return "single_direction_stronger"
    return "similar"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase114_{model}_answer_site_causal_subspace.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            conds = item["conditions"]
            tc = best(conds, "transport_direction")
            sub = best(conds, "answer_contrast_subspace")
            rnd = best(conds, "random_subspace")
            rows.append({
                "model": model,
                "category": cat,
                "tc": tc,
                "subspace": sub,
                "random": rnd,
                "class": classify(tc, sub, rnd),
            })

    first = next(iter(loaded.values()))
    lines = ["# Phase 114 Cross-model Answer-site Causal Subspace", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts per category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- ranks: {first['ranks']}; scales: {first['scales']}; layer: model-specific causal peak")
    lines.append("- subspace: SVD of answer-site target-vs-other category contrast rows")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | best T_c | best answer contrast subspace | best random subspace | class |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {fmt(r['tc'])} | "
            f"{fmt(r['subspace'])} | {fmt(r['random'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Objective Reading Rules")
    lines.append("- subspace_stronger means the answer contrast subspace reduces target logits at least 1.0 more than T_c.")
    lines.append("- control_sensitive means random same-rank subspace is too close or stronger.")
    lines.append("- R is max positive non-target release delta.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- The contrast subspace is category geometry, not yet an automatically discovered causal subspace.")
    lines.append("- Random controls are same rank but not matched to norm spectrum.")
    lines.append("- This phase still uses DCF logits, not open generation.")
    out = OUT_DIR / "phase114_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
