#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase115_causal_subspace_robustness")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict], kind: str) -> dict | None:
    xs = [r for r in rows if r["kind"] == kind]
    return min(xs, key=lambda r: r["target_delta"]) if xs else None


def mean_best_lto(rows: list[dict], kind: str) -> dict | None:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row["kind"] == kind:
            grouped.setdefault(int(row["heldout_template_id"]), []).append(row)
    if not grouped:
        return None
    bests = [min(v, key=lambda r: r["target_delta"]) for v in grouped.values()]
    return {
        "rank": "fold_best",
        "scale": "fold_best",
        "target_delta": sum(r["target_delta"] for r in bests) / len(bests),
        "max_other_delta": sum(r["max_other_delta"] for r in bests) / len(bests),
        "folds": bests,
    }


def fmt(r: dict | None) -> str:
    if r is None:
        return "NA"
    return f"r{r['rank']} s{r['scale']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"


def classify(full: dict | None, rand: dict | None, lto: dict | None, lto_rand: dict | None) -> str:
    if full is None or lto is None:
        return "no_data"
    if rand and rand["target_delta"] <= full["target_delta"] + 0.5:
        return "control_sensitive"
    if lto_rand and lto_rand["target_delta"] <= lto["target_delta"] + 0.5:
        return "lto_control_sensitive"
    if full["target_delta"] <= -5.0 and lto["target_delta"] <= -3.0:
        return "robust_strong"
    if full["target_delta"] <= -2.0 and lto["target_delta"] <= -1.0:
        return "robust_moderate"
    if full["target_delta"] <= -2.0 and lto["target_delta"] > -1.0:
        return "template_sensitive"
    if abs(full["target_delta"]) < 0.5:
        return "weak"
    return "mixed"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase115_{model}_causal_subspace_robustness.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            conds = item["conditions"]
            lto = item["leave_template_out"]
            full = best(conds, "answer_contrast_subspace")
            rand = best(conds, "matched_spectrum_random")
            rel_ex = best(conds, "release_excluded_subspace")
            lto_mean = mean_best_lto(lto, "lto_answer_contrast_subspace")
            lto_rand = mean_best_lto(lto, "lto_matched_spectrum_random")
            rows.append({
                "model": model,
                "category": cat,
                "full": full,
                "random": rand,
                "release_excluded": rel_ex,
                "lto": lto_mean,
                "lto_random": lto_rand,
                "class": classify(full, rand, lto_mean, lto_rand),
            })

    first = next(iter(loaded.values()))
    lines = ["# Phase 115 Cross-model Causal Subspace Robustness", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; full prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- ranks: {first['ranks']}; scales: {first['scales']}; layer: model-specific causal peak")
    lines.append("- robustness: full-template basis, leave-template-out basis, matched-spectrum random, release-excluded subspace")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | full subspace | matched random | release-excluded | LTO mean | LTO random mean | class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {fmt(r['full'])} | {fmt(r['random'])} | "
            f"{fmt(r['release_excluded'])} | {fmt(r['lto'])} | {fmt(r['lto_random'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Objective Reading Rules")
    lines.append("- LTO mean averages the best heldout-template result across the four heldout templates.")
    lines.append("- robust_strong requires full-template target_delta <= -5 and LTO mean <= -3, with controls weaker.")
    lines.append("- release-excluded removes the strongest release category from contrast construction.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- Matched-spectrum random is implemented through synthetic contrast matrices, but the intervention still uses orthonormal bases.")
    lines.append("- Release decomposition only excludes the strongest observed release category; it is not a full support/release factorization.")
    lines.append("- This phase still uses DCF logits, not open generation.")
    out = OUT_DIR / "phase115_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
