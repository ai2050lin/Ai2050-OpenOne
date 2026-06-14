#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/gpt5_phase124_writer_set_value_alignment")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda x: x["target_delta"]) if rows else None


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    name = row.get("set_name", row.get("site", ""))
    k = "" if "set_size" not in row else f" k{row['set_size']}"
    layer = "" if "patch_layer" not in row else f" L{row['patch_layer']}"
    return f"{name}{k}{layer} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def classify(attn: dict[str, Any] | None, value: dict[str, Any] | None, target: dict[str, Any] | None,
             obj: dict[str, Any] | None, rand: dict[str, Any] | None, mlp: dict[str, Any] | None) -> str:
    candidates = [x for x in [attn, value, target] if x is not None]
    best_head = min(candidates, key=lambda x: x["target_delta"]) if candidates else None
    control_floor = min([x["target_delta"] for x in [obj, rand] if x is not None], default=0.0)
    if best_head and best_head["target_delta"] <= -1.0 and best_head["target_delta"] <= control_floor - 0.5:
        return "head_set_candidate"
    if mlp and mlp["target_delta"] <= -1.0:
        return "pre_mlp_subspace_candidate"
    if best_head and best_head["target_delta"] <= -1.0:
        return "head_set_control_like"
    if mlp and mlp["target_delta"] <= -0.5:
        return "weak_pre_mlp_subspace"
    return "weak_or_control_like"


def main() -> None:
    loaded = {}
    rows = []
    for model in MODELS:
        data = json.loads((OUT_DIR / f"phase124_{model}_writer_set_value_alignment.json").read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            sets = item["set_conditions"]
            mlps = item["mlp_subspace_conditions"]
            attn = best([x for x in sets if x["set_name"] == "attention_mass"])
            value = best([x for x in sets if x["set_name"] in ("value_aligned", "abs_value_aligned")])
            target = best([x for x in sets if x["set_name"] == "target_discovered"])
            proj = best([x for x in sets if x["set_name"] == "projection_discovered"])
            obj = best([x for x in sets if x["set_name"] == "object_control"])
            rand = best([x for x in sets if x["set_name"] == "random_control"])
            mlp = best(mlps)
            rows.append({
                "model": model,
                "category": cat,
                "attn": attn,
                "value": value,
                "target": target,
                "proj": proj,
                "obj": obj,
                "rand": rand,
                "mlp": mlp,
                "class": classify(attn, value, target, obj, rand, mlp),
            })

    first = next(iter(loaded.values()))
    layers = "; ".join(
        f"{model}: L{data['patch_layers'][0]}-L{data['patch_layers'][-1]} monitor L{data['monitor_layer']}"
        for model, data in loaded.items()
    )
    lines = ["# Phase 124 Cross-model Writer Set Value Alignment", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- layers: {layers}; rank: {first['rank']}; set sizes: {first['set_sizes']}; candidate pool: {first['candidate_pool']}")
    lines.append("")
    lines.append("| model | category | attention set | value set | target-discovered set | projection set | object control | random control | pre-MLP subspace | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['category']} | {fmt(row['attn'])} | {fmt(row['value'])} | "
            f"{fmt(row['target'])} | {fmt(row['proj'])} | {fmt(row['obj'])} | {fmt(row['rand'])} | "
            f"{fmt(row['mlp'])} | {row['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- attention set is ranked by answer-token attention mass to pre-answer tokens.")
    lines.append("- value set is ranked by attention head output alignment with the answer monitor axis.")
    lines.append("- target-discovered set is ranked by measured single-head target_delta within the candidate pool.")
    lines.append("- A is answer projection delta at the peak answer site.")
    out = OUT_DIR / "phase124_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
