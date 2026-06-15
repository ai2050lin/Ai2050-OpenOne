#!/usr/bin/env python3
"""Summarize Phase 145 mechanism stability matrix."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase145_mechanism_stability_generation")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase145_{model}_mechanism_stability_generation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 145 Cross-model Mechanism Stability Summary", ""]
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in result["path_results"].values():
            grouped[(row["category"], row["path"]["name"], row["path"]["kind"])].append(row)
        lines.append(
            f"families={result['families']}; splits={result['splits']}; "
            f"train/test={result['train_objects']}/{result['test_objects']}"
        )
        lines.append("")
        lines.append("| category | path | kind | n | clean_rate | mean_rec | mean_release | category_argmax |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for (cat, path, kind), rows in sorted(grouped.items()):
            clean_rate = mean([1.0 if r["is_constrained_clean"] else 0.0 for r in rows])
            rec = mean([float(r["recovery_ratio"]) for r in rows])
            rel = mean([float(r["max_other_delta"]) for r in rows])
            carg = mean([float(r["first_token_summary"]["category_argmax_rate"]) for r in rows])
            lines.append(f"| {cat} | {path} | {kind} | {len(rows)} | {clean_rate:.2f} | {rec:+.2f} | {rel:+.2f} | {carg:.2f} |")
        lines.append("")
    path = OUT_ROOT / "phase145_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
