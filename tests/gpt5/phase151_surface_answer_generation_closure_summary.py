#!/usr/bin/env python3
"""Summarize Phase 151 surface-answer set closure."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase151_surface_answer_generation_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase151_{model}_surface_answer_generation_closure.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def rank(item: dict[str, Any], variant: str, set_name: str, metric: str) -> float:
    return float(item["variants"][variant]["surface_ranks"][set_name][metric])


def greedy_rate(item: dict[str, Any], variant: str, classes: set[str]) -> float:
    rates = item["variants"][variant].get("greedy_class_rates", {})
    return float(sum(v for k, v in rates.items() if k in classes))


def top_greedy_class(item: dict[str, Any]) -> str:
    rates = item["variants"]["final_norm_output_lm"].get("greedy_class_rates", {})
    if not rates:
        return ""
    return max(rates.items(), key=lambda x: x[1])[0]


def emit(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### By {title}")
    lines.append("")
    lines.append("| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for group, items in sorted(grouped.items()):
        cls = Counter(top_greedy_class(x) for x in items).most_common(1)
        lines.append(
            f"| {group} | {len(items)} | "
            f"{mean([rank(x, 'clean', 'expanded', 'argmax') for x in items]):.2f} | "
            f"{mean([rank(x, 'support_only', 'expanded', 'argmax') for x in items]):.2f} | "
            f"{mean([rank(x, 'final_norm_output_lm', 'expanded', 'argmax') for x in items]):.2f} | "
            f"{mean([rank(x, 'final_norm_output_lm', 'expanded', 'rank') for x in items]):.1f} | "
            f"{mean([rank(x, 'final_norm_output_lm', 'canonical', 'rank') for x in items]):.1f} | "
            f"{mean([rank(x, 'final_norm_output_lm', 'synonyms', 'rank') for x in items]):.1f} | "
            f"{mean([greedy_rate(x, 'final_norm_output_lm', {'canonical','synonym','object_near','option_like'}) for x in items]):.2f} | "
            f"{cls[0][0] if cls else ''} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 151 Cross-model Surface Answer Generation Closure Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        for group_name, idx in [("category", 3), ("format", 2), ("family", 1)]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key, item in result["results"].items():
                grouped[key.split(":")[idx]].append(item)
            emit(lines, group_name, grouped)
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | greedy_class | examples |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, item in sorted(result["results"].items()):
            final = item["variants"]["final_norm_output_lm"]
            examples = " ".join(x.replace("\n", "\\n") for x in final["greedy_text_examples"][:3])
            lines.append(
                f"| {key} | {final['surface_ranks']['expanded']['argmax']:.2f} | "
                f"{final['surface_ranks']['expanded']['rank']:.1f} | "
                f"{final['surface_ranks']['canonical']['rank']:.1f} | "
                f"{final['surface_ranks']['synonyms']['rank']:.1f} | "
                f"{top_greedy_class(item)} | {examples} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase151_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
