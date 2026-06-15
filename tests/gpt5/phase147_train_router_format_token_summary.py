#!/usr/bin/env python3
"""Summarize Phase 147 train-router and format-token tests."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase147_train_router_format_token")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase147_{model}_train_router_format_token.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 147 Cross-model Train Router Format Token Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        lines.append(
            f"families={result['families']}; splits={result['splits']}; formats={result['formats']}; "
            f"train/test={result['train_objects']}/{result['test_objects']}"
        )
        lines.append("")
        for group_name, key_fn in [
            ("category", lambda key, item: key.split(":")[-1]),
            ("format", lambda key, item: key.split(":")[2]),
        ]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key, item in result["results"].items():
                grouped[key_fn(key, item)].append(item["heldout"])
            lines.append(f"### By {group_name}")
            lines.append("")
            lines.append("| group | n | held_clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax |")
            lines.append("|---|---|---|---|---|---|---|")
            for group, rows in sorted(grouped.items()):
                clean = mean([1.0 if r["is_constrained_clean"] else 0.0 for r in rows])
                rec = mean([float(r["recovery_ratio"]) for r in rows])
                rel = mean([float(r["max_other_delta"]) for r in rows])
                rank = mean([float(r.get("token", {}).get("target_token_rank_mean", 0.0)) for r in rows])
                arg = mean([float(r.get("token", {}).get("target_token_argmax_rate", 0.0)) for r in rows])
                lines.append(f"| {group} | {len(rows)} | {clean:.2f} | {rec:+.2f} | {rel:+.2f} | {rank:.1f} | {arg:.2f} |")
            lines.append("")
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | train path | train clean | held T | held R | held rec | held clean | token_rank | token_argmax |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key, item in sorted(result["results"].items()):
            tr = item["train_best"]
            h = item["heldout"]
            tok = h.get("token", {})
            lines.append(
                f"| {key} | L{tr['layer_id']} {tr['site']} s{tr['scale']} | {tr['is_constrained_clean']} | "
                f"{h['target_delta']:+.2f} | {h['max_other_delta']:+.2f} | {h['recovery_ratio']:+.2f} | "
                f"{h['is_constrained_clean']} | {tok.get('target_token_rank_mean', 0):.1f} | {tok.get('target_token_argmax_rate', 0):.2f} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase147_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
