#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("results/gpt5_phase113_head_set_mlp_relay_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def best(conds: list[dict], fn) -> dict | None:
    xs = [c for c in conds if fn(c)]
    return min(xs, key=lambda c: c["target_delta"]) if xs else None


def fmt(c: dict | None) -> str:
    if c is None:
        return "NA"
    return (
        f"{c.get('set_name','tc')} {c.get('relay','')} k{c.get('set_size','')} "
        f"T{c['target_delta']:+.2f} R{c.get('effect_ratio_vs_tc_remove',1.0):+.2f} "
        f"A{c.get('answer_transport_proj_delta',0.0):+.2f}"
    )


def classify(tc: dict, head: dict | None, relay: dict | None, mlp: dict | None, rand: dict | None) -> str:
    tc_delta = tc["target_delta"]
    if tc_delta >= -0.5:
        return "weak_reference"
    ratios = [x.get("effect_ratio_vs_tc_remove", 0.0) for x in [head, relay, mlp] if x is not None]
    positive_ratios = [r for r in ratios if r > 0]
    best_ratio = max(positive_ratios, default=0.0)
    rand_ratio = rand.get("effect_ratio_vs_tc_remove", 0.0) if rand else 0.0
    if best_ratio >= 0.5 and best_ratio > rand_ratio + 0.2:
        return "partial_closure"
    if best_ratio >= 0.2:
        return "poor_closure"
    if rand and rand["target_delta"] <= (head["target_delta"] if head else 0.0) - 0.1:
        return "control_sensitive"
    return "not_closed"


def main() -> None:
    rows = []
    loaded = {}
    for model in MODELS:
        path = OUT_DIR / f"phase113_{model}_head_set_mlp_relay_closure.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[model] = data
        for cat, item in data["category_results"].items():
            conds = item["conditions"]
            tc = item["tc_remove_reference"]
            head = best(conds, lambda c: c["relay"] == "heads_only" and c["set_name"] != "random")
            relay = best(conds, lambda c: c["relay"] == "heads_plus_mlp" and c["set_name"] != "random")
            mlp = best(conds, lambda c: c["relay"] == "mlp_only")
            rand = best(conds, lambda c: c["set_name"] == "random")
            rows.append({
                "model": model,
                "category": cat,
                "tc": tc,
                "head": head,
                "relay": relay,
                "mlp": mlp,
                "random": rand,
                "class": classify(tc, head, relay, mlp, rand),
            })

    first = next(iter(loaded.values()))
    lines = ["# Phase 113 Cross-model Head Set and MLP Relay Closure", ""]
    lines.append("## Test Scope")
    lines.append(
        f"- models: {', '.join(MODELS)}; categories: {', '.join(first['test_categories'])}; "
        f"train/test objects per category: {first['train_objects_per_category']}/{first['test_objects_per_category']}; "
        f"templates: {len(first['templates'])}; prompts per category: {first['test_objects_per_category'] * len(first['templates'])}"
    )
    lines.append(f"- layers: peak-{len(first['patch_layers']) - 1} ... peak; candidate heads: {first['candidate_heads']}; set sizes: {first['set_sizes']}")
    lines.append("- conditions: source/projection/target/mixed/random head sets; heads only, MLP only, heads+MLP")
    lines.append("")
    lines.append("## Cross-model Table")
    lines.append("| model | category | T_c reference | best heads only | best heads+MLP | best MLP only | best random | class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['category']} | {fmt(r['tc'])} | {fmt(r['head'])} | "
            f"{fmt(r['relay'])} | {fmt(r['mlp'])} | {fmt(r['random'])} | {r['class']} |"
        )
    lines.append("")
    lines.append("## Objective Reading Rules")
    lines.append("- R is target_delta divided by the answer-site T_c removal target_delta; useful closure should be positive and close to 1.")
    lines.append("- not_closed means head sets and MLP relay did not approach the T_c reference.")
    lines.append("- control_sensitive means random head sets were comparable or stronger than selected sets.")
    lines.append("")
    lines.append("## Hard Limits")
    lines.append("- MLP ablation zeroes MLP output at answer_last across scanned layers; it is a coarse intervention.")
    lines.append("- Candidate projection heads are chosen from the expanded source-head pool, not from all heads.")
    lines.append("- Generation audit and Q/K/V split are still not included.")
    out = OUT_DIR / "phase113_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
