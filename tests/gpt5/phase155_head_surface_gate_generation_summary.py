#!/usr/bin/env python3
"""Summarize Phase 155 head-level surface gate generation results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase155_head_surface_gate_generation")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase155_{model}_head_surface_gate_generation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def gen(case: dict[str, Any], name: str) -> dict[str, Any]:
    return case["generations"][name]


def emit(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### By {title}")
    lines.append("")
    lines.append("| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group, cases in sorted(grouped.items()):
        top_answer_heads = Counter(f"H{c['top_answer']['head_id']}" for c in cases).most_common(1)
        top_format_heads = Counter(f"H{c['top_format']['head_id']}" for c in cases).most_common(1)
        top_joint_heads = Counter(f"H{c['top_joint']['head_id']}" for c in cases).most_common(1)
        lines.append(
            f"| {group} | {len(cases)} | "
            f"{mean([gen(c, 'clean')['hit_rate'] for c in cases]):.2f} | "
            f"{mean([gen(c, 'top_answer')['hit_rate'] for c in cases]):.2f} | "
            f"{mean([gen(c, 'top_format')['hit_rate'] for c in cases]):.2f} | "
            f"{mean([gen(c, 'top_joint')['hit_rate'] for c in cases]):.2f} | "
            f"{mean([gen(c, 'random')['hit_rate'] for c in cases]):.2f} | "
            f"{mean([gen(c, 'top_answer')['hit_rate'] - gen(c, 'clean')['hit_rate'] for c in cases]):+.2f} | "
            f"{mean([gen(c, 'top_format')['hit_rate'] - gen(c, 'clean')['hit_rate'] for c in cases]):+.2f} | "
            f"{mean([gen(c, 'top_joint')['hit_rate'] - gen(c, 'clean')['hit_rate'] for c in cases]):+.2f} | "
            f"{top_answer_heads[0][0] if top_answer_heads else ''} | "
            f"{top_format_heads[0][0] if top_format_heads else ''} | "
            f"{top_joint_heads[0][0] if top_joint_heads else ''} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 155 Cross-model Head Surface Gate Generation Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        cases_by_key = result["results"]
        cases = list(cases_by_key.values())
        lines.append(f"cases={len(cases)}, layer=L{result['layer_id']}, heads={result['num_heads']}")
        lines.append("")
        for title, idx in [("category", 3), ("format", 2), ("family", 1), ("split", 0)]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key, case in cases_by_key.items():
                grouped[key.split(":")[idx]].append(case)
            emit(lines, title, grouped)
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | clean | ans | fmt | joint | random | top_answer | top_format | top_joint |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key, case in sorted(cases_by_key.items()):
            lines.append(
                f"| {key} | {gen(case, 'clean')['hit_rate']:.2f} | "
                f"{gen(case, 'top_answer')['hit_rate']:.2f} | "
                f"{gen(case, 'top_format')['hit_rate']:.2f} | "
                f"{gen(case, 'top_joint')['hit_rate']:.2f} | "
                f"{gen(case, 'random')['hit_rate']:.2f} | "
                f"H{case['top_answer']['head_id']} dA{case['top_answer']['answer_rank_delta']:+.1f} | "
                f"H{case['top_format']['head_id']} dF{case['top_format']['format_rank_delta']:+.1f} | "
                f"H{case['top_joint']['head_id']} dA{case['top_joint']['answer_rank_delta']:+.1f}/dF{case['top_joint']['format_rank_delta']:+.1f} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase155_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
