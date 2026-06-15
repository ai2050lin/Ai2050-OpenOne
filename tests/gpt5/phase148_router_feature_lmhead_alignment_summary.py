#!/usr/bin/env python3
"""Summarize Phase 148 router feature and LM-head alignment tests."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase148_router_feature_lmhead_alignment")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase148_{model}_router_feature_lmhead_alignment.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 148 Cross-model Router Feature LM-Head Alignment Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        for group_name, idx in [("category", 3), ("format", 2)]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key, item in result["results"].items():
                grouped[key.split(":")[idx]].append(item)
            lines.append(f"### By {group_name}")
            lines.append("")
            lines.append("| group | n | prev_clean | best_clean | pre_ov | ans_ov | held_R2 | cos | rank0 | rank_best | arg_best |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for group, items in sorted(grouped.items()):
                prev_clean = mean([1.0 if x["phase147_held_clean"] else 0.0 for x in items])
                best_clean = mean([1.0 if x["best_steering"]["is_constrained_clean"] else 0.0 for x in items])
                pre = mean([x["pre_basis_overlap"] for x in items])
                ans = mean([x["ans_basis_overlap"] for x in items])
                r2 = mean([x["heldout_transfer_r2"] for x in items])
                cos = mean([x["support_lm_cosine"] for x in items])
                rank0 = mean([next(r for r in x["steering_rows"] if r["lm_scale"] == 0.0)["token"]["target_token_rank_mean"] for x in items])
                rankb = mean([x["best_steering"]["token"]["target_token_rank_mean"] for x in items])
                argb = mean([x["best_steering"]["token"]["target_token_argmax_rate"] for x in items])
                lines.append(f"| {group} | {len(items)} | {prev_clean:.2f} | {best_clean:.2f} | {pre:.2f} | {ans:.2f} | {r2:+.2f} | {cos:+.2f} | {rank0:.1f} | {rankb:.1f} | {argb:.2f} |")
            lines.append("")
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | prev | pre/ans | R2 | cos | best_lm | rank0 | rank_best | arg | clean |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, item in sorted(result["results"].items()):
            b = item["best_steering"]
            row0 = next(r for r in item["steering_rows"] if r["lm_scale"] == 0.0)
            lines.append(
                f"| {key} | {item['phase147_held_clean']} | {item['pre_basis_overlap']:.2f}/{item['ans_basis_overlap']:.2f} | "
                f"{item['heldout_transfer_r2']:+.2f} | {item['support_lm_cosine']:+.2f} | {b['lm_scale']} | "
                f"{row0['token']['target_token_rank_mean']:.1f} | {b['token']['target_token_rank_mean']:.1f} | "
                f"{b['token']['target_token_argmax_rate']:.2f} | {b['is_constrained_clean']} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase148_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
