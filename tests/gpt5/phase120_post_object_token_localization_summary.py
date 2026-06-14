#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase120_post_object_token_localization")
MODELS = ["qwen3", "glm4", "deepseek7b"]
PRE_SITES = {"after_object_first", "after_object_middle", "pre_answer_last", "post_object_excluding_answer"}


def best(rows: list[dict]) -> dict | None:
    return min(rows, key=lambda r: r["target_delta"]) if rows else None


def fmt(r: dict | None) -> str:
    if r is None:
        return "NA"
    return f"L{r['layer']} {r['site']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"


def classify(pre_best: dict | None, answer: dict | None, incl: dict | None) -> str:
    pre = pre_best["target_delta"] if pre_best else 0.0
    ans = answer["target_delta"] if answer else 0.0
    inc = incl["target_delta"] if incl else 0.0
    if pre <= -6.0 and ans <= -6.0:
        return "pre_answer_interface_and_answer"
    if pre <= -6.0 and ans > -3.0:
        return "pre_answer_interface_dominant"
    if ans <= -6.0 and pre > -2.0:
        return "answer_leakage_dominant"
    if inc <= -6.0 and pre > -2.0 and ans > -2.0:
        return "mean_only_effect"
    if pre <= -2.0:
        return "moderate_pre_answer_interface"
    return "weak_or_answer_only"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase120_{model}_post_object_token_localization.json").read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            for axis in data["axis_types"]:
                conds = [c for c in item["conditions"] if c["axis_type"] == axis]
                row = {
                    "model": model,
                    "category": cat,
                    "axis": axis,
                    "object_last": best([c for c in conds if c["site"] == "object_last"]),
                    "after_first": best([c for c in conds if c["site"] == "after_object_first"]),
                    "after_middle": best([c for c in conds if c["site"] == "after_object_middle"]),
                    "pre_answer": best([c for c in conds if c["site"] == "pre_answer_last"]),
                    "excluding": best([c for c in conds if c["site"] == "post_object_excluding_answer"]),
                    "answer": best([c for c in conds if c["site"] == "answer_last"]),
                    "including": best([c for c in conds if c["site"] == "post_object_including_answer"]),
                }
                row["pre_best"] = best([x for x in [row["after_first"], row["after_middle"], row["pre_answer"], row["excluding"]] if x])
                row["class"] = classify(row["pre_best"], row["answer"], row["including"])
                rows.append(row)

    first = next(iter(loaded.values()))
    lines = ["# Phase 120 Cross-model Post-object Token Localization", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(
        f"- layers: peak-{len(first['patch_layers']) - 1} ... peak; sites: {', '.join(first['sites'])}; "
        f"rank: {first['rank']}; scale: {first['scale']}; axis types: {', '.join(first['axis_types'])}"
    )
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | axis | object_last | after_first | after_middle | pre_answer | excluding_answer | answer_last | including_answer | best pre-answer | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {r['axis']} | {fmt(r['object_last'])} | {fmt(r['after_first'])} | "
            f"{fmt(r['after_middle'])} | {fmt(r['pre_answer'])} | {fmt(r['excluding'])} | {fmt(r['answer'])} | "
            f"{fmt(r['including'])} | {fmt(r['pre_best'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- excluding_answer is the full post-object span before answer_last.")
    lines.append("- including_answer reproduces Phase119 post_object_mean style with answer_last included.")
    lines.append("- If excluding_answer remains strong, the Phase119 source effect is not just answer_last leakage.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- Token groups are based on template token positions, not linguistic parse labels.")
    lines.append("- Single-token and mean-group local axes are fit independently and may use different coordinates.")
    lines.append("- Results are DCF logits, not open generation.")
    out = OUT_DIR / "phase120_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
