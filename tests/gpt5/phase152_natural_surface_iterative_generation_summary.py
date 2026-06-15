#!/usr/bin/env python3
"""Summarize Phase 152 natural surface iterative generation results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase152_natural_surface_iterative_generation")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase152_{model}_natural_surface_iterative_generation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def gen(item: dict[str, Any], name: str) -> dict[str, Any]:
    return item["generation"][name]


def top_cls(row: dict[str, Any]) -> str:
    rates = row.get("final_class_rates", {})
    return max(rates.items(), key=lambda x: x[1])[0] if rates else ""


def emit(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### By {title}")
    lines.append("")
    lines.append("| group | n | clean_hit | remove_hit | remove_restore_hit | best_add_hit | best_fmt_later | best_add_scale | clean_class | best_class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for group, items in sorted(grouped.items()):
        scales = [str(gen(x, "best_additive").get("scale")) for x in items]
        scale = Counter(scales).most_common(1)[0][0] if scales else ""
        clean_class = Counter(top_cls(gen(x, "clean")) for x in items).most_common(1)
        best_class = Counter(top_cls(gen(x, "best_additive")) for x in items).most_common(1)
        lines.append(
            f"| {group} | {len(items)} | "
            f"{mean([gen(x, 'clean')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'remove')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'remove_restore')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'best_additive')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'best_additive')['format_first_answer_later_rate'] for x in items]):.2f} | "
            f"{scale} | {clean_class[0][0] if clean_class else ''} | {best_class[0][0] if best_class else ''} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 152 Cross-model Natural Surface Iterative Generation Summary", ""]
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
        lines.append("| case | clean | remove_restore | best_add | best_variant | fmt_later | clean_class | best_class | examples |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key, item in sorted(result["results"].items()):
            clean = gen(item, "clean")
            rr = gen(item, "remove_restore")
            ba = gen(item, "best_additive")
            examples = " ".join(x.replace("\n", "\\n") for x in ba.get("examples", [])[:3])
            lines.append(
                f"| {key} | {clean['hit_rate']:.2f} | {rr['hit_rate']:.2f} | {ba['hit_rate']:.2f} | "
                f"{ba.get('variant')}:{ba.get('scale')} | {ba['format_first_answer_later_rate']:.2f} | "
                f"{top_cls(clean)} | {top_cls(ba)} | {examples} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase152_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
