#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase110_orthogonal_subspace_split")
MODELS = ["qwen3", "glm4", "deepseek7b"]
KINDS = ["orthogonal_full", "neighbor_aligned", "transport_aligned", "residual", "random_same_norm"]


def best_condition(item: dict, kind: str) -> dict:
    xs = [c for c in item["conditions"] if c["kind"] == kind]
    return min(xs, key=lambda c: c["target_delta"])


def best_release(item: dict, kind: str) -> dict:
    xs = [c for c in item["conditions"] if c["kind"] == kind]
    return max(xs, key=lambda c: c["max_other_delta"])


def fmt_cond(c: dict) -> str:
    return f"{c['position']} s{c['scale']} Δ{c['target_delta']:+.2f}"


def dominant_component(best: dict[str, dict]) -> str:
    core = ["neighbor_aligned", "transport_aligned", "residual"]
    return min(core, key=lambda k: best[k]["target_delta"])


def classify(item: dict, best: dict[str, dict]) -> str:
    dom = dominant_component(best)
    orth = best["orthogonal_full"]["target_delta"]
    dom_delta = best[dom]["target_delta"]
    random_delta = best["random_same_norm"]["target_delta"]
    if abs(orth) < 0.25 and abs(dom_delta) < 0.5:
        return "weak"
    if dom == "transport_aligned" and dom_delta <= -1.0:
        return "transport_support"
    if dom == "residual" and dom_delta <= -1.0:
        return "residual_support"
    if dom == "neighbor_aligned" and dom_delta <= -1.0:
        return "neighbor_competition"
    if orth > 0.25 and dom_delta <= -1.0:
        return "component_cancellation"
    if random_delta <= dom_delta + 0.25:
        return "control_sensitive"
    return "mixed"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase110_{model}_orthogonal_subspace_split.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            best = {kind: best_condition(item, kind) for kind in KINDS}
            release = best_release(item, "residual")
            rows.append({
                "model": model,
                "layer": data["layer"],
                "category": cat,
                "fractions": item["norm_fractions"],
                "best": best,
                "release": release,
                "class": classify(item, best),
            })

    lines = ["# Phase 110 Cross-model Orthogonal Subspace Split", ""]
    lines.append("## Test Scope")
    first = next(iter(loaded.values()))
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts per category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append("- components: orthogonal_full, neighbor_aligned, transport_aligned, residual, random_same_norm")
    lines.append("- positions: answer_last, both; scales: 1.0, 1.5")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append(
        "| model | category | frac N/T/R | best neighbor | best transport | best residual | best orth | best random | residual release | class |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        fr = r["fractions"]
        b = r["best"]
        rel = r["release"]
        top = rel["top_releases"][0] if rel["top_releases"] else {"category": "none", "delta": 0.0}
        lines.append(
            f"| {r['model']} | {r['category']} | {fr['neighbor']:.2f}/{fr['transport']:.2f}/{fr['residual']:.2f} | "
            f"{fmt_cond(b['neighbor_aligned'])} | {fmt_cond(b['transport_aligned'])} | {fmt_cond(b['residual'])} | "
            f"{fmt_cond(b['orthogonal_full'])} | {fmt_cond(b['random_same_norm'])} | "
            f"{top['category']} {top['delta']:+.2f} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Objective Facts")
    lines.append("- Qwen3 number/time show real target-down effects in neighbor and transport components; number is strongest in transport.")
    lines.append("- Qwen3 container/clothing/plant expose component cancellation: transport removal can reduce target strongly even when full orthogonal removal is weak or target-up.")
    lines.append("- DS7B container/clothing/furniture/plant are transport-dominant: removing the object-to-answer transport-aligned component gives large target-down effects.")
    lines.append("- DS7B number differs from the above pattern: the residual component gives the strongest subcomponent target-down, while full orthogonal remains strongest overall.")
    lines.append("- GLM4 effects remain weak, so it is still unsuitable for strong mechanism conclusions in this probe family.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- The transport direction is a mean object_last to answer_last vector, not a direct causal path proof.")
    lines.append("- Neighbor basis is hand-defined by category adjacency and can miss hidden competitors.")
    lines.append("- Single-layer intervention can create cancellation artifacts; multi-layer cumulative tests are still needed.")
    lines.append("- Some component removals are stronger than full orthogonal removal, indicating non-additive interaction inside the residual stream.")
    lines.append("")
    out = OUT_DIR / "phase110_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
