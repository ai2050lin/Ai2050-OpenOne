#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/gpt5_phase123_attention_mlp_writer_localization")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda x: x["target_delta"]) if rows else None


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    head = "" if "head_id" not in row else f" H{row['head_id']}"
    site = row.get("site", row.get("selection_group", ""))
    return f"L{row.get('patch_layer','NA')}{head} {site} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} Aproj{row['answer_proj_delta']:+.2f}"


def classify(pre_head: dict[str, Any] | None, obj_head: dict[str, Any] | None, rand: dict[str, Any] | None,
             pre_mlp: dict[str, Any] | None, ans_mlp: dict[str, Any] | None) -> str:
    pre_t = pre_head["target_delta"] if pre_head else 0.0
    obj_t = obj_head["target_delta"] if obj_head else 0.0
    rand_t = rand["target_delta"] if rand else 0.0
    pre_a = pre_head["answer_proj_delta"] if pre_head else 0.0
    pre_mlp_t = pre_mlp["target_delta"] if pre_mlp else 0.0
    pre_mlp_a = pre_mlp["answer_proj_delta"] if pre_mlp else 0.0
    ans_mlp_t = ans_mlp["target_delta"] if ans_mlp else 0.0
    if pre_t <= -1.0 and pre_a <= -0.5 and pre_t <= min(obj_t, rand_t) - 0.5:
        return "attention_pre_writer_candidate"
    if pre_mlp_t <= -1.0 and pre_mlp_a <= -0.5:
        return "pre_mlp_writer_candidate"
    if ans_mlp_t <= -1.0 and ans_mlp_t < pre_mlp_t:
        return "answer_mlp_readout_candidate"
    if min(pre_t, pre_mlp_t, ans_mlp_t) <= -1.0:
        return "module_effect_without_clean_projection"
    return "weak_or_control_like"


def main() -> None:
    loaded = {}
    rows = []
    for model in MODELS:
        path = OUT_DIR / f"phase123_{model}_attention_mlp_writer_localization.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            heads = item["head_conditions"]
            mlps = item["mlp_conditions"]
            pre_head = best([x for x in heads if x["selection_group"] == "pre_answer_top"])
            obj_head = best([x for x in heads if x["selection_group"] == "object_top"])
            self_head = best([x for x in heads if x["selection_group"] == "self_top"])
            rand = best([x for x in heads if x["selection_group"] == "random"])
            pre_mlp = best([x for x in mlps if x["site"] == "pre_answer"])
            ans_mlp = best([x for x in mlps if x["site"] == "answer_last"])
            rows.append({
                "model": model,
                "category": cat,
                "pre_head": pre_head,
                "obj_head": obj_head,
                "self_head": self_head,
                "rand": rand,
                "pre_mlp": pre_mlp,
                "ans_mlp": ans_mlp,
                "class": classify(pre_head, obj_head, rand, pre_mlp, ans_mlp),
            })

    first = next(iter(loaded.values()))
    layer_desc = "; ".join(
        f"{model}: L{data['patch_layers'][0]}-L{data['patch_layers'][-1]} monitor L{data['monitor_layer']}"
        for model, data in loaded.items()
    )
    lines = ["# Phase 123 Cross-model Attention MLP Writer Localization", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts/category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- layers: {layer_desc}; rank: {first['rank']}; top-k heads/group: {first['top_k_heads']}")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | best pre-head | best object-head | best self-head | best random-head | best pre-MLP | best answer-MLP | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['category']} | {fmt(row['pre_head'])} | {fmt(row['obj_head'])} | "
            f"{fmt(row['self_head'])} | {fmt(row['rand'])} | {fmt(row['pre_mlp'])} | "
            f"{fmt(row['ans_mlp'])} | {row['class']} |"
        )
    lines.append("")
    lines.append("## Reading Rules")
    lines.append("- pre-head means answer-token attention heads selected by post_object/pre-answer attention mass.")
    lines.append("- object-head and random-head are controls.")
    lines.append("- Aproj is the peak answer_last projection delta on the selected answer-site monitor axis.")
    lines.append("- writer_candidate requires target drop and answer projection drop, while beating object/random controls.")
    out = OUT_DIR / "phase123_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
