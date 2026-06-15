#!/usr/bin/env python3
"""Summarize Phase 146 template-conditioned router sweep."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase146_template_router_token_gap")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase146_{model}_template_router_token_gap.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 146 Cross-model Template Router Token Gap Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in result["results"].values():
            grouped[item["best"]["category"]].append(item["best"])
        lines.append(
            f"families={result['families']}; splits={result['splits']}; "
            f"layers={result['layer_offsets']}; sites={result['sites']}; scales={result['scales']}"
        )
        lines.append("")
        lines.append("| category | cases | clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax | common_best_paths |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cat, rows in sorted(grouped.items()):
            clean_rate = mean([1.0 if r["is_constrained_clean"] else 0.0 for r in rows])
            rec = mean([float(r["recovery_ratio"]) for r in rows])
            rel = mean([float(r["max_other_delta"]) for r in rows])
            rank = mean([float(r.get("token", {}).get("target_token_rank_mean", 0.0)) for r in rows])
            arg = mean([float(r.get("token", {}).get("target_token_argmax_rate", 0.0)) for r in rows])
            paths = defaultdict(int)
            for r in rows:
                paths[f"L{r['layer_id']} {r['site']} s{r['scale']}"] += 1
            common = "; ".join(f"{k}({v})" for k, v in sorted(paths.items(), key=lambda x: -x[1])[:4])
            lines.append(f"| {cat} | {len(rows)} | {clean_rate:.2f} | {rec:+.2f} | {rel:+.2f} | {rank:.1f} | {arg:.2f} | {common} |")
        lines.append("")
        lines.append("| case | best path | T | R | rec | clean | token_rank | token_argmax |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, item in sorted(result["results"].items()):
            b = item["best"]
            tok = b.get("token", {})
            lines.append(
                f"| {key} | L{b['layer_id']} {b['site']} s{b['scale']} | "
                f"{b['target_delta']:+.2f} | {b['max_other_delta']:+.2f} | {b['recovery_ratio']:+.2f} | "
                f"{b['is_constrained_clean']} | {tok.get('target_token_rank_mean', 0):.1f} | {tok.get('target_token_argmax_rate', 0):.2f} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase146_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
