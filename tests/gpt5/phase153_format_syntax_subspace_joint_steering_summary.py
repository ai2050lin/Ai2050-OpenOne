#!/usr/bin/env python3
"""Summarize Phase 153 format-syntax subspace and joint steering results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase153_format_syntax_subspace_joint_steering")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase153_{model}_format_syntax_subspace_joint_steering.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def gen(item: dict[str, Any], name: str) -> dict[str, Any]:
    return item["generation"][name]


def answer_rank(row: dict[str, Any]) -> float:
    return float(row.get("first_audit", {}).get("ranks", {}).get("expanded_answer", {}).get("rank", 0.0))


def format_rank(row: dict[str, Any]) -> float:
    return float(row.get("first_audit", {}).get("ranks", {}).get("all_format", {}).get("rank", 0.0))


def top_group(row: dict[str, Any]) -> str:
    groups = row.get("first_audit", {}).get("argmax_group_rates", {})
    return max(groups.items(), key=lambda x: x[1])[0] if groups else ""


def top_cls(row: dict[str, Any]) -> str:
    rates = row.get("final_class_rates", {})
    return max(rates.items(), key=lambda x: x[1])[0] if rates else ""


def emit(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### By {title}")
    lines.append("")
    lines.append("| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group, items in sorted(grouped.items()):
        best = [gen(x, "best_joint") for x in items]
        top_groups = Counter(top_group(x) for x in best).most_common(1)
        best_classes = Counter(top_cls(x) for x in best).most_common(1)
        lines.append(
            f"| {group} | {len(items)} | "
            f"{mean([x['semantic_format_overlap']['max_abs_cos'] for x in items]):.3f} | "
            f"{mean([gen(x, 'clean')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'semantic_additive')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'format_internal')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'format_lm')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'best_joint')['hit_rate'] for x in items]):.2f} | "
            f"{mean([gen(x, 'best_joint')['hit_rate'] - gen(x, 'semantic_additive')['hit_rate'] for x in items]):+.2f} | "
            f"{mean([answer_rank(gen(x, 'best_joint')) for x in items]):.1f} | "
            f"{mean([format_rank(gen(x, 'best_joint')) for x in items]):.1f} | "
            f"{top_groups[0][0] if top_groups else ''} | {best_classes[0][0] if best_classes else ''} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 153 Cross-model Format-Syntax Subspace Joint Steering Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        items_by_key = result["results"]
        lines.append(
            f"cases={len(items_by_key)}, formats={','.join(result.get('formats', []))}, "
            f"semantic_scale={result.get('semantic_scale')}, format_scales={result.get('format_scales')}"
        )
        lines.append("")
        for group_name, idx in [("category", 3), ("format", 2), ("family", 1), ("split", 0)]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key, item in items_by_key.items():
                grouped[key.split(":")[idx]].append(item)
            emit(lines, group_name, grouped)
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | overlap | clean | sem | fmt_int | fmt_lm | best_joint | gain | joint | fmt_group | answer_rank | examples |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, item in sorted(items_by_key.items()):
            bj = gen(item, "best_joint")
            examples = " ".join(x.replace("\n", "\\n") for x in bj.get("examples", [])[:3])
            lines.append(
                f"| {key} | {item['semantic_format_overlap']['max_abs_cos']:.3f} | "
                f"{gen(item, 'clean')['hit_rate']:.2f} | {gen(item, 'semantic_additive')['hit_rate']:.2f} | "
                f"{gen(item, 'format_internal')['hit_rate']:.2f} | {gen(item, 'format_lm')['hit_rate']:.2f} | "
                f"{bj['hit_rate']:.2f} | {bj['hit_rate'] - gen(item, 'semantic_additive')['hit_rate']:+.2f} | "
                f"{bj.get('variant')}:{bj.get('format_scale')} | {top_group(bj)} | {answer_rank(bj):.1f} | {examples} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase153_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
