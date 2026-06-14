#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase116_subspace_basis_component_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict], key: str = "target_delta") -> dict | None:
    return min(rows, key=lambda r: r[key]) if rows else None


def split(rr: dict, name: str) -> dict | None:
    return next((r for r in rr["split_sets"] if r["set_name"] == name), None)


def fmt(r: dict | None) -> str:
    if r is None:
        return "NA"
    idx = r.get("basis_index", r.get("set_size", ""))
    label = r.get("component_label", r.get("set_name", ""))
    return f"{label}{idx} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"


def classify(best_single: dict | None, best_cum: dict | None, support: dict | None, release: dict | None) -> str:
    if best_cum is None:
        return "no_data"
    single_t = best_single["target_delta"] if best_single else 0.0
    cum_t = best_cum["target_delta"]
    size = int(best_cum.get("set_size", 0))
    if cum_t <= -8.0 and size >= 8:
        return "distributed_support"
    if single_t <= -4.0:
        return "compact_support"
    if support and support["target_delta"] <= -3.0 and support["max_other_delta"] <= 0.5:
        return "support_set_clean"
    if release and release["max_other_delta"] >= 1.0:
        return "release_components"
    if cum_t <= -2.0:
        return "moderate_support"
    return "weak_or_mixed"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase116_{model}_subspace_basis_component_audit.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            for rank, rr in item["rank_results"].items():
                best_single = best(rr["basis_components"])
                best_random = best(rr["random_components"])
                best_cum = best(rr["cumulative"])
                support = split(rr, "support")
                release = split(rr, "release")
                mixed = split(rr, "mixed")
                rows.append({
                    "model": model,
                    "category": cat,
                    "rank": rank,
                    "best_single": best_single,
                    "best_random": best_random,
                    "best_cum": best_cum,
                    "support": support,
                    "release": release,
                    "mixed": mixed,
                    "class": classify(best_single, best_cum, support, release),
                })

    first = next(iter(loaded.values()))
    lines = ["# Phase 116 Cross-model Subspace Basis Component Audit", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- ranks: {first['ranks']}; scale: {first['scale']}; cumulative sizes: {first['set_sizes']}")
    lines.append("- component labels: support, release, mixed, weak")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | rank | best single | best random single | best cumulative | support set | release set | mixed set | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {r['rank']} | {fmt(r['best_single'])} | "
            f"{fmt(r['best_random'])} | {fmt(r['best_cum'])} | {fmt(r['support'])} | "
            f"{fmt(r['release'])} | {fmt(r['mixed'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Objective Reading Rules")
    lines.append("- distributed_support means cumulative top basis components are much stronger than any single component.")
    lines.append("- compact_support means one basis component alone has a strong target-down effect.")
    lines.append("- release_components means basis-level release is directly visible.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- SVD basis ordering is geometric, not guaranteed to be causal ordering.")
    lines.append("- Component labels are heuristic and should be treated as audit tags, not final theory.")
    lines.append("- This phase still uses DCF logits, not open generation.")
    out = OUT_DIR / "phase116_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
