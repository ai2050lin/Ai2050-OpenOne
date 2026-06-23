#!/usr/bin/env python3
"""Summarize Phase 584 gate-repair confirm results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase584_gate_repair")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    rows = []
    lines = [
        "# Phase 584 Gate Repair Cross-Model Summary",
        "",
        "Confirm setting: n_tables=15, two-hop samples=80 per model, polarity samples=60 per model.",
        "",
        "| model | direct | gold-cat | rel-emphasis | rel-filter | polarity base | polarity rule+fmt | neg base | neg rule+fmt | wrong-cat->wrong-val | wrong-cat->correct-val |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        path = ROOT / f"phase584_{model}_gate_repair_confirm.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        part_a = data["partA_choice_gate_repair"]
        part_b = data["partB_value_retrieval_repair"]
        part_c = data["partC_polarity_gate_repair"]
        part_d = data["partD_bypass_mechanism"]
        row = {
            "model": model,
            "direct": part_a["direct_accuracy"],
            "gold_cat": part_a["gold_accuracy"],
            "rel_emphasis": part_b["repair1_accuracy"],
            "rel_filter": part_b["repair2_accuracy"],
            "pol_base": part_c["base_accuracy"],
            "pol_rule_fmt": part_c["rule_format_accuracy"],
            "neg_base": part_c["neg_base_accuracy"],
            "neg_rule_fmt": part_c["neg_rule_format_accuracy"],
            "wrong_val": part_d["bypass_matches_wrong_rate"],
            "correct_val": part_d["bypass_matches_correct_rate"],
        }
        rows.append(row)
        lines.append(
            "| {model} | {direct} | {gold_cat} | {rel_emphasis} | {rel_filter} | "
            "{pol_base} | {pol_rule_fmt} | {neg_base} | {neg_rule_fmt} | "
            "{wrong_val} | {correct_val} |".format(
                model=model,
                direct=pct(row["direct"]),
                gold_cat=pct(row["gold_cat"]),
                rel_emphasis=pct(row["rel_emphasis"]),
                rel_filter=pct(row["rel_filter"]),
                pol_base=pct(row["pol_base"]),
                pol_rule_fmt=pct(row["pol_rule_fmt"]),
                neg_base=pct(row["neg_base"]),
                neg_rule_fmt=pct(row["neg_rule_fmt"]),
                wrong_val=pct(row["wrong_val"]),
                correct_val=pct(row["correct_val"]),
            )
        )

    lines += [
        "",
        "Key objective facts:",
        "",
        "- Relation-filter repair is the strongest value-retrieval repair: qwen3 100.0%, glm4 92.5%, deepseek7b 97.5%.",
        "- Rule+format polarity repair nearly closes the negative-answer gate: qwen3 100.0%, glm4 100.0%, deepseek7b 96.7% on negatives.",
        "- System instruction alone is weak for polarity repair: glm4 negative 10.0%, deepseek7b negative 0.0%.",
        "- Wrong-category forcing often changes value choice, but not cleanly enough to prove a single mandatory O-C-V path.",
        "- No-CRV accuracy remains near 26-28%, close to chance among four value tokens, so values are not simply memorized without relation rules.",
        "",
    ]

    out = ROOT / "phase584_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
