#!/usr/bin/env python3
"""Summarize Phase 150 open-vocab competitor gate results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase150_open_vocab_competitor_gate")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase150_{model}_open_vocab_competitor_gate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def subset(item: dict[str, Any], variant: str, subset_name: str, metric: str) -> float:
    return float(
        item["variants"].get(variant, {})
        .get("logit_audit", {})
        .get("subset_metrics", {})
        .get(subset_name, {})
        .get(metric, 0.0)
    )


def top_class(item: dict[str, Any], variant: str) -> str:
    rates = item["variants"].get(variant, {}).get("logit_audit", {}).get("argmax_class_rates", {})
    if not rates:
        return ""
    return max(rates.items(), key=lambda x: x[1])[0]


def emit(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### By {title}")
    lines.append("")
    lines.append("| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for group, items in sorted(grouped.items()):
        classes = Counter(top_class(x, "final_norm_output_lm") for x in items)
        cls = classes.most_common(1)[0][0] if classes else ""
        lines.append(
            f"| {group} | {len(items)} | "
            f"{mean([subset(x, 'final_norm_output_lm', 'candidate4', 'argmax') for x in items]):.2f} | "
            f"{mean([subset(x, 'final_norm_output_lm', 'semantic_all_categories', 'argmax') for x in items]):.2f} | "
            f"{mean([subset(x, 'final_norm_output_lm', 'alphabetic', 'rank') for x in items]):.1f} | "
            f"{mean([subset(x, 'final_norm_output_lm', 'non_format', 'rank') for x in items]):.1f} | "
            f"{mean([subset(x, 'final_norm_output_lm', 'full', 'rank') for x in items]):.1f} | "
            f"{mean([subset(x, 'final_norm_output_lm', 'full', 'argmax') for x in items]):.2f} | "
            f"{mean([subset(x, 'support_only', 'full', 'rank') for x in items]):.1f} | {cls} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 150 Cross-model Open-Vocab Competitor Gate Summary", ""]
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
        lines.append("| case | cand4 | semantic | alpha_rank | nonfmt_rank | full_rank | arg_class | top_tokens |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, item in sorted(result["results"].items()):
            audit = item["variants"]["final_norm_output_lm"]["logit_audit"]
            sm = audit["subset_metrics"]
            top = " ".join(x["token"].replace("\n", "\\n") for x in audit.get("top_tokens", [])[:4])
            lines.append(
                f"| {key} | {sm['candidate4']['argmax']:.2f} | {sm['semantic_all_categories']['argmax']:.2f} | "
                f"{sm['alphabetic']['rank']:.1f} | {sm['non_format']['rank']:.1f} | {sm['full']['rank']:.1f} | "
                f"{top_class(item, 'final_norm_output_lm')} | {top} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase150_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
