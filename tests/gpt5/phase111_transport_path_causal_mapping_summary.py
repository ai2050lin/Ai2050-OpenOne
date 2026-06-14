#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase111_transport_path_causal_mapping")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(conds: list[dict], site: str | None, mode: str, key: str = "target_delta") -> dict | None:
    xs = [c for c in conds if c["patch_mode"] == mode and (site is None or c["patch_site"] == site)]
    if not xs:
        return None
    return min(xs, key=lambda c: c[key])


def fmt(c: dict | None) -> str:
    if c is None:
        return "NA"
    return (
        f"L{c['patch_layer']} {c['patch_site']} s{c['scale']} "
        f"T{c['target_delta']:+.2f} A{c['answer_transport_proj_delta']:+.2f}"
    )


def classify(obj: dict | None, ans: dict | None, proj: dict | None, rand: dict | None) -> str:
    obj_t = obj["target_delta"] if obj else 0.0
    ans_t = ans["target_delta"] if ans else 0.0
    proj_a = proj["answer_transport_proj_delta"] if proj else 0.0
    rand_t = rand["target_delta"] if rand else 0.0
    if obj_t <= -1.0 and proj_a <= -0.25 and obj_t < rand_t - 0.5:
        return "object_path_supported"
    if ans_t <= -1.0 and obj_t > -0.5:
        return "answer_site_only"
    if obj_t <= -1.0 and proj_a > -0.1:
        return "logit_without_projection_sync"
    if rand_t <= obj_t + 0.25 and rand_t <= -0.5:
        return "control_sensitive"
    if abs(obj_t) < 0.5 and abs(ans_t) < 0.5:
        return "weak"
    return "mixed"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase111_{model}_transport_path_causal_mapping.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            conds = item["conditions"]
            obj = best(conds, "object_last", "remove_target", "target_delta")
            ans = best(conds, "answer_last", "remove_target", "target_delta")
            proj = best(
                [c for c in conds if c["patch_site"] == "object_last" and c["patch_mode"] == "remove_target"],
                None,
                "remove_target",
                "answer_transport_proj_delta",
            )
            wrong = best(conds, None, "wrong_inject_abs", "target_delta")
            rand = best(conds, None, "random_remove", "target_delta")
            rows.append({
                "model": model,
                "category": cat,
                "wrong_category": item["wrong_category"],
                "object_remove": obj,
                "answer_remove": ans,
                "object_proj": proj,
                "wrong_inject": wrong,
                "random": rand,
                "class": classify(obj, ans, proj, rand),
            })

    first = next(iter(loaded.values()))
    lines = ["# Phase 111 Cross-model Transport Path Causal Mapping", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts per category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- monitor layer: model-specific peak; patch layers: peak-{len(first['patch_layers']) - 1} ... peak")
    lines.append(f"- sites: {', '.join(first['patch_sites'])}; modes: {', '.join(first['patch_modes'])}; scales: {first['scales']}")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | wrong cat | object remove | answer remove | object answer-proj down | wrong inject | random | class |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {r['wrong_category']} | "
            f"{fmt(r['object_remove'])} | {fmt(r['answer_remove'])} | {fmt(r['object_proj'])} | "
            f"{fmt(r['wrong_inject'])} | {fmt(r['random'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Objective Reading Rules")
    lines.append("- object_path_supported means object_last removal reduced target logits, reduced answer_last transport projection, and beat random control.")
    lines.append("- answer_site_only means answer_last removal was strong while object_last removal was weak.")
    lines.append("- logit_without_projection_sync means logits moved but monitored answer transport projection did not move together.")
    lines.append("- control_sensitive means random control was too close or stronger, so the condition is not reliable.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- This phase monitors one peak-layer transport projection; hidden path changes outside that projection can be missed.")
    lines.append("- wrong-category injection uses fixed neighbor choices, not an automatically learned release graph.")
    lines.append("- Generation audit is still not included in this script.")
    out = OUT_DIR / "phase111_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
