#!/usr/bin/env python3
"""Summarize Phase 149 final-norm and candidate-set token gate tests."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase149_final_norm_candidate_gate")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase149_{model}_final_norm_candidate_gate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def metric(item: dict[str, Any], key: str, row_name: str = "best_candidate") -> float:
    return float(item[row_name].get("token", {}).get(key, 0.0))


def lens_metric(item: dict[str, Any], site: str, key: str) -> float:
    return float(item["best_candidate"].get("logit_lens", {}).get(site, {}).get(key, 0.0))


def emit_group(lines: list[str], name: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### By {name}")
    lines.append("")
    lines.append("| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for group, items in sorted(grouped.items()):
        lines.append(
            f"| {group} | {len(items)} | "
            f"{mean([1.0 if x['phase147_held_clean'] else 0.0 for x in items]):.2f} | "
            f"{mean([metric(x, 'candidate_argmax_rate') for x in items]):.2f} | "
            f"{mean([metric(x, 'candidate_rank_mean') for x in items]):.2f} | "
            f"{mean([metric(x, 'candidate_margin_mean') for x in items]):+.2f} | "
            f"{mean([metric(x, 'full_vocab_rank_mean') for x in items]):.1f} | "
            f"{mean([metric(x, 'full_vocab_argmax_rate') for x in items]):.2f} | "
            f"{mean([lens_metric(x, 'final_norm_input', 'candidate_argmax_rate') for x in items]):.2f} | "
            f"{mean([lens_metric(x, 'final_norm_output', 'candidate_argmax_rate') for x in items]):.2f} | "
            f"{mean([metric(x, 'full_vocab_rank_mean', 'best_full_vocab') for x in items]):.1f} |"
        )
    lines.append("")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 149 Cross-model Final-Norm Candidate Gate Summary", ""]
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
            emit_group(lines, group_name, grouped)
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | variant | cand_arg | cand_rank | full_rank | full_arg | lens_in/out | top |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, item in sorted(result["results"].items()):
            b = item["best_candidate"]
            tok = b.get("token", {})
            lens = b.get("logit_lens", {})
            top = tok.get("top_tokens_examples", [[]])
            first_top = ""
            if top and top[0]:
                first_top = " ".join(x["text"].replace("\n", "\\n") for x in top[0][:3])
            lines.append(
                f"| {key} | {b['variant']}:{b['lm_scale']}:{b['suppress_scale']} | "
                f"{tok.get('candidate_argmax_rate', 0):.2f} | {tok.get('candidate_rank_mean', 0):.2f} | "
                f"{tok.get('full_vocab_rank_mean', 0):.1f} | {tok.get('full_vocab_argmax_rate', 0):.2f} | "
                f"{lens.get('final_norm_input', {}).get('candidate_argmax_rate', 0):.2f}/"
                f"{lens.get('final_norm_output', {}).get('candidate_argmax_rate', 0):.2f} | {first_top} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase149_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
