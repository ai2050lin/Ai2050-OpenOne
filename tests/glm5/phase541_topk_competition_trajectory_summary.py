#!/usr/bin/env python3
"""Summary for Phase541 top-k competition trajectory audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase541_topk_competition_trajectory")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["residual_perp", "residual_parallel", "residual_full"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase541_{model}_topk_competition_trajectory.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{float(x):+.3f}"


def best_row(d: dict[str, Any], source: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {win: d["audit"][win]["sources"][source][condition] for win in d["audit"]}
    win = max(rows, key=lambda w: rows[w]["target_logit_delta"] - rows[w]["competitor_logit_delta"])
    return win, rows[win]


def cell(row: dict[str, Any]) -> str:
    by_type = row["baseline_topk_delta_by_type"]
    rank_by_type = row["baseline_topk_rank_delta_by_type"]
    return (
        f"targetLogit {fmt(row['target_logit_delta'])}, "
        f"compLogit {fmt(row['competitor_logit_delta'])}, "
        f"targetRank {row['target_rank_delta']:+.1f}, "
        f"compRank {row['competitor_rank_delta']:+.1f}, "
        f"churn {float(row['mean_topk_churn']):.2f}, "
        f"topOtherΔ {fmt(by_type['other'])}, "
        f"topOtherRank {rank_by_type['other']:+.1f}"
    )


def classify(row: dict[str, Any]) -> str:
    target = float(row["target_logit_delta"])
    comp = float(row["competitor_logit_delta"])
    churn = float(row["mean_topk_churn"])
    other_delta = float(row["baseline_topk_delta_by_type"]["other"])
    if target > 0.5 and comp > -0.15 and churn < 0.35:
        return "target_push_low_churn"
    if comp < -0.5 and churn < 0.45:
        return "specific_competitor_suppression"
    if comp < -0.5 and churn >= 0.45:
        return "suppression_with_topk_reshaping"
    if other_delta < -0.5:
        return "broad_topk_suppression"
    return "mixed_or_weak"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase541 Top-K Competition Trajectory Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"top_k={d['top_k']}, core={d['core_sources']}, windows={d['windows']}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, alphas={d['alphas']}, "
            f"attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("| source | condition | win | alpha | trajectory | class |")
        lines.append("|---|---|---|---:|---|---|")
        for source in d["core_sources"]:
            for condition in CONDITIONS:
                win, row = best_row(d, source, condition)
                cls = classify(row)
                if condition == "residual_parallel":
                    compact.append({
                        "model": model,
                        "source": source,
                        "win": win,
                        "target": row["target_logit_delta"],
                        "competitor": row["competitor_logit_delta"],
                        "target_rank": row["target_rank_delta"],
                        "competitor_rank": row["competitor_rank_delta"],
                        "churn": row["mean_topk_churn"],
                        "other_delta": row["baseline_topk_delta_by_type"]["other"],
                        "class": cls,
                    })
                lines.append(
                    f"| {source} | {condition} | {win} | {float(row['best_alpha']):.1f} | "
                    f"{cell(row)} | {cls} |"
                )
        lines.append("")

    if compact:
        lines.append("## Residual Parallel Compact")
        lines.append("")
        lines.append("| model | source | win | target logit | competitor logit | target rank Δ | competitor rank Δ | churn | top other Δ | class |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for r in compact:
            lines.append(
                f"| {r['model']} | {r['source']} | {r['win']} | {fmt(r['target'])} | "
                f"{fmt(r['competitor'])} | {float(r['target_rank']):+.1f} | "
                f"{float(r['competitor_rank']):+.1f} | {float(r['churn']):.2f} | "
                f"{fmt(r['other_delta'])} | {r['class']} |"
            )
        lines.append("")

    out = root / "phase541_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
