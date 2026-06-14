#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase117_basis_rotation_causal_axis")
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


def counts(rows: list[dict]) -> str:
    names = ["support", "release", "mixed", "weak"]
    return ",".join(f"{n}:{sum(1 for r in rows if r.get('component_label') == n)}" for n in names)


def nearest_cumulative(rr: dict, size: int) -> dict | None:
    rows = [r for r in rr["cumulative"] if int(r.get("set_size", -1)) == size]
    return rows[0] if rows else best(rr["cumulative"])


def classify(cat_rows: dict[str, dict]) -> str:
    svd = cat_rows.get("svd")
    varimax = cat_rows.get("varimax")
    greedy = cat_rows.get("causal_greedy")
    if not svd or not varimax:
        return "missing"
    svd_full = nearest_cumulative(svd, 16)
    var_full = nearest_cumulative(varimax, 16)
    greedy_top4 = nearest_cumulative(greedy, 4) if greedy else None
    if svd_full and var_full and svd_full["target_delta"] <= -6.0 and var_full["target_delta"] <= -6.0:
        if greedy_top4 and greedy_top4["target_delta"] <= 0.7 * svd_full["target_delta"]:
            return "stable_subspace_causal_axes"
        return "stable_subspace_distributed"
    if svd_full and var_full and abs(svd_full["target_delta"] - var_full["target_delta"]) <= 1.0:
        return "rotation_similar_weak_or_moderate"
    if svd_full and var_full and svd_full["target_delta"] <= -2.0 and var_full["target_delta"] > -1.0:
        return "basis_sensitive"
    return "weak_or_mixed"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase117_{model}_basis_rotation_causal_axis.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            basis_rows = {}
            for name, rr in item["basis_variants"].items():
                one = {
                    "model": model,
                    "category": cat,
                    "basis": name,
                    "best_single": best(rr["components"]),
                    "best_cumulative": best(rr["cumulative"]),
                    "top4": nearest_cumulative(rr, 4),
                    "top8": nearest_cumulative(rr, 8),
                    "top16": nearest_cumulative(rr, 16),
                    "support": split(rr, "support"),
                    "release": split(rr, "release"),
                    "mixed": split(rr, "mixed"),
                    "counts": counts(rr["components"]),
                }
                rows.append(one)
                basis_rows[name] = rr
            for r in rows:
                if r["model"] == model and r["category"] == cat:
                    r["class"] = classify(basis_rows)

    first = next(iter(loaded.values()))
    lines = ["# Phase 117 Cross-model Basis Rotation and Causal Axis Stabilization", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(
        f"- rank: {first['rank']}; scale: {first['scale']}; set sizes: {first['set_sizes']}; "
        f"random rotations/category: {first['random_rotations']}; causal candidates/category: {first['causal_candidates']}"
    )
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | basis | counts | best single | top4 | top8 | top16 | support set | release set | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {r['basis']} | {r['counts']} | {fmt(r['best_single'])} | "
            f"{fmt(r['top4'])} | {fmt(r['top8'])} | {fmt(r['top16'])} | {fmt(r['support'])} | "
            f"{fmt(r['release'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- Top16 compares full-rank removal inside the same subspace; it should stay similar under orthogonal rotation.")
    lines.append("- Best single, support/release counts, and top4/top8 show whether Phase116 component labels are basis-sensitive.")
    lines.append("- causal_greedy samples directions inside the same subspace, then greedily orthogonalizes strongest target-down candidates.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- This is a rotation audit of answer-site DCF logits, not open generation.")
    lines.append("- causal_greedy is a finite random search, not a proof of optimal causal axes.")
    lines.append("- If full-rank effects are stable but component labels move, keep the subspace-level claim and downgrade single-basis interpretation.")
    out = OUT_DIR / "phase117_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
